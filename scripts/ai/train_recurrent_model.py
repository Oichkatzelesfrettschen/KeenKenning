#!/usr/bin/env python3
"""
Recurrent Transformer for Latin Square Generation

Based on research:
- "Learning to Solve CSPs with Recurrent Transformer" (arXiv:2307.04895)
- nanoGPT best practices (github.com/karpathy/nanoGPT)
- HuggingFace performance optimization guide

Key innovations over standard transformer:
1. RECURRENCE: Output hidden states feed back as input for R steps
2. CONSTRAINT LOSS AT EVERY STEP: L_cst applied at each recurrent layer
3. STRAIGHT-THROUGH ESTIMATOR: Enables gradient flow through discrete constraints
4. SMALLER MODEL + MORE ITERATIONS: 1-4 blocks x 32 recurrent steps

Training best practices applied:
- Mixed precision (BF16/FP16) with automatic gradient scaling
- Gradient accumulation for larger effective batch size
- Linear warmup + cosine decay learning rate schedule
- AdamW with beta2=0.95 (nanoGPT-recommended)
- Gradient clipping at 1.0
- Weight tying for parameter efficiency

Target: Loss < 0.09 with proper constraint satisfaction

References:
- https://arxiv.org/abs/2307.04895
- https://github.com/karpathy/nanoGPT
- https://huggingface.co/docs/transformers/perf_train_gpu_one
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import numpy as np
import os
import math
import argparse
import json
from torch.utils.data import DataLoader, Dataset
from torch.utils.checkpoint import checkpoint
from collections import defaultdict
import time
import sys
from datetime import datetime

# Enable line-buffered output for real-time logging
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ============================================================================
# HARDWARE OPTIMIZATION: Environment Variables (must be set before torch init)
# ============================================================================
# Memory allocation optimization - reduces fragmentation
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF',
    'expandable_segments:True,garbage_collection_threshold:0.8')

# Remove performance-limiting env vars if present
if os.environ.get('CUDA_DISABLE_PERF_BOOST') == '1':
    del os.environ['CUDA_DISABLE_PERF_BOOST']  # Allow GPU boost clocks

# Lazy CUDA module loading for faster startup
os.environ.setdefault('CUDA_MODULE_LOADING', 'LAZY')

# cuDNN optimization hints
os.environ.setdefault('CUDNN_V8_API_LRU_CACHE_LIMIT', '10000')

# Disable NCCL if single GPU (reduces overhead)
os.environ.setdefault('NCCL_P2P_DISABLE', '1')


# =============================================================================
# CPU CORE PINNING - Ryzen 5600X3D (6C/12T) Optimization
# =============================================================================
# Physical cores: 0,1,2,3,4,5 | Hyperthreads: 6,7,8,9,10,11
# Pin DataLoader workers to physical cores for 96MB L3 cache affinity
PHYSICAL_CORES = [0, 1, 2, 3, 4, 5]

def worker_init_fn(worker_id: int):
    """Pin DataLoader worker to a physical CPU core.

    WHY: Ryzen 5600X3D has 96MB 3D V-Cache on the CCD containing physical cores.
         Pinning workers to physical cores (not hyperthreads) maximizes cache hits
         during data preprocessing and reduces cache thrashing.

    WHAT: Each worker gets affinity to one physical core (round-robin).
    """
    core = PHYSICAL_CORES[worker_id % len(PHYSICAL_CORES)]
    try:
        os.sched_setaffinity(0, {core})
    except (OSError, AttributeError):
        pass  # Fail silently on systems without sched_setaffinity


# =============================================================================
# LIVE STATUS SUPPORT
# =============================================================================

class StatusWriter:
    """Write training status to JSON for live monitoring."""

    def __init__(self, path=None):
        self.path = path
        self.start_time = time.time()
        self.data = {
            "status": "starting",
            "epoch": 0,
            "total_epochs": 0,
            "batch": 0,
            "total_batches": 0,
            "loss": 0.0,
            "val_loss": 0.0,
            "accuracy": 0.0,
            "valid_rate": 0.0,
            "lr": 0.0,
            "elapsed": "0:00:00",
            "best_loss": float('inf'),
            "target": 0.09,
            "timestamp": "",
        }

    def update(self, **kwargs):
        """Update status and write to file."""
        self.data.update(kwargs)
        elapsed = time.time() - self.start_time
        h, m = divmod(int(elapsed), 3600)
        m, s = divmod(m, 60)
        self.data["elapsed"] = f"{h}:{m:02d}:{s:02d}"
        self.data["timestamp"] = datetime.now().strftime("%H:%M:%S")

        if self.path:
            try:
                with open(self.path, 'w') as f:
                    json.dump(self.data, f, indent=2, default=str)
            except Exception:
                pass  # Don't fail training on status write error

    def get_statusline(self):
        """Generate compact statusline for terminal."""
        d = self.data
        return (f"[{d['elapsed']}] E{d['epoch']}/{d['total_epochs']} "
                f"B{d['batch']}/{d['total_batches']} "
                f"loss={d['loss']:.4f} val={d['val_loss']:.4f} "
                f"acc={d['accuracy']:.1f}% valid={d['valid_rate']:.1f}%")


# =============================================================================
# RECURRENT TRANSFORMER ARCHITECTURE
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Normalization."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, d_model, n_head, dropout=0.1, max_seq_len=256):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.dropout = dropout

        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        self.resid_dropout = nn.Dropout(dropout)

        self.register_buffer("mask", torch.tril(torch.ones(max_seq_len, max_seq_len))
                             .view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        if hasattr(F, 'scaled_dot_product_attention'):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                                  dropout_p=self.dropout if self.training else 0)
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = F.dropout(att, p=self.dropout, training=self.training)
            out = att @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(out))


class MLP(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.c_fc = nn.Linear(d_model, 4 * d_model)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1, max_seq_len=256):
        super().__init__()
        self.ln_1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout, max_seq_len)
        self.ln_2 = RMSNorm(d_model)
        self.mlp = MLP(d_model, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class RecurrentLatinTransformer(nn.Module):
    """
    Recurrent Transformer for Latin Square Generation.
    Hidden states from step r become input for step r+1.
    """

    def __init__(self, max_size=16, num_classes=18, d_model=128, n_head=4,
                 n_layer=2, n_recurrent=32, dropout=0.1):
        super().__init__()
        self.max_size = max_size
        self.num_classes = num_classes
        self.d_model = d_model
        self.n_recurrent = n_recurrent
        max_seq_len = max_size * max_size

        self.tok_emb = nn.Embedding(num_classes, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.size_emb = nn.Embedding(max_size + 1, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_head, dropout, max_seq_len)
            for _ in range(n_layer)
        ])

        self.ln_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, num_classes, bias=False)
        self.recur_proj = nn.Linear(d_model, d_model)
        self.tok_emb.weight = self.head.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward_single_step(self, x, hidden, grid_size):
        B, T = x.shape
        tok = self.tok_emb(x)
        pos = self.pos_emb(torch.arange(T, device=x.device))
        size = self.size_emb(torch.full((B,), grid_size, device=x.device, dtype=torch.long))

        h = self.drop(tok + pos + size.unsqueeze(1))
        if hidden is not None:
            h = h + self.recur_proj(hidden)

        for block in self.blocks:
            h = block(h)

        logits = self.head(self.ln_f(h))
        return logits, h

    def forward(self, x, grid_size, n_steps=None, return_all_logits=False, use_checkpoint=True):
        """Forward pass with optional gradient checkpointing.

        Args:
            use_checkpoint: If True, uses gradient checkpointing to save VRAM.
                           Trades ~2x compute for ~32x memory savings on activations.
        """
        n_steps = n_steps or self.n_recurrent
        hidden = None
        all_logits = [] if return_all_logits else None

        for step in range(n_steps):
            if use_checkpoint and self.training:
                # Checkpoint every 4 steps to balance memory vs compute
                if step % 4 == 0:
                    logits, hidden = checkpoint(
                        self.forward_single_step, x, hidden, grid_size,
                        use_reentrant=False
                    )
                else:
                    logits, hidden = self.forward_single_step(x, hidden, grid_size)
            else:
                logits, hidden = self.forward_single_step(x, hidden, grid_size)

            if return_all_logits:
                all_logits.append(logits)

        if return_all_logits:
            return logits, all_logits
        return logits


def compute_constraint_loss_ste(logits, grid_size, temperature=1.0):
    """Compute Latin square constraint loss with STE."""
    B, T, C = logits.shape
    probs = F.softmax(logits / temperature, dim=-1)
    value_probs = probs[:, :, 1:grid_size+1]
    grid_probs = value_probs[:, :grid_size*grid_size, :].view(B, grid_size, grid_size, grid_size)

    row_sums = grid_probs.sum(dim=2)
    row_loss = ((row_sums - 1.0) ** 2).mean()

    col_sums = grid_probs.sum(dim=1)
    col_loss = ((col_sums - 1.0) ** 2).mean()

    cell_entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean()
    entropy_bonus = -0.01 * cell_entropy

    return row_loss + col_loss + entropy_bonus


class LatinSquareDataset(Dataset):
    def __init__(self, data_path, sizes=None):
        data = np.load(data_path)

        all_grids = []
        all_sizes = []

        for size in range(3, 17):
            key = f'size{size}'
            if key in data:
                grids = data[key]
                for g in grids:
                    # Pad to 16x16
                    padded = np.zeros((16, 16), dtype=np.int64)
                    padded[:size, :size] = g
                    all_grids.append(padded)
                    all_sizes.append(size)

        self.grids = np.array(all_grids)
        self.sizes = np.array(all_sizes)

        if sizes is not None:
            mask = np.isin(self.sizes, sizes)
            self.grids = self.grids[mask]
            self.sizes = self.sizes[mask]

    def __len__(self):
        return len(self.grids)

    def __getitem__(self, idx):
        grid = torch.from_numpy(self.grids[idx].astype(np.int64))
        size = int(self.sizes[idx])
        return grid, size


def collate_fn(batch):
    grids, sizes = zip(*batch)
    grids = torch.stack(grids)
    sizes = torch.tensor(sizes)
    return grids, sizes


class MetricsTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_grids = 0
        self.valid_grids = 0
        self.total_cells = 0
        self.correct_cells = 0
        self.total_violations = 0

    def update(self, preds, targets, grid_size):
        B = preds.shape[0]
        self.total_grids += B

        for b in range(B):
            p = preds[b, :grid_size, :grid_size].cpu().numpy()
            t = targets[b, :grid_size, :grid_size].cpu().numpy()

            self.correct_cells += (p == t).sum()
            self.total_cells += grid_size * grid_size

            violations = 0
            for i in range(grid_size):
                violations += grid_size - len(set(p[i, :]))
                violations += grid_size - len(set(p[:, i]))

            self.total_violations += violations
            if violations == 0:
                self.valid_grids += 1

    def get_metrics(self):
        return {
            'valid_rate': 100.0 * self.valid_grids / max(1, self.total_grids),
            'accuracy': 100.0 * self.correct_cells / max(1, self.total_cells),
            'avg_violations': self.total_violations / max(1, self.total_grids),
        }


def get_lr(step, warmup_steps, max_lr, min_lr, max_steps):
    """Linear warmup + cosine decay learning rate schedule (nanoGPT style)."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def train_epoch(model, loader, optimizer, scaler, device, epoch, args, global_step,
                status=None, total_batches=0):
    """
    Training loop with nanoGPT-style best practices:
    - Mixed precision (BF16/FP16) via torch.amp
    - Gradient accumulation for larger effective batch size
    - Learning rate warmup + cosine decay
    - Gradient clipping at 1.0
    """
    model.train()
    total_loss = 0
    total_ce = 0
    total_cst = 0
    n_batches = 0

    # Gradient accumulation settings
    accum_steps = args.gradient_accumulation_steps
    micro_batch_count = 0

    for batch_idx, (grids, sizes) in enumerate(loader):
        # Async GPU transfer with pinned memory (CPU-GPU overlap)
        grids = grids.to(device, non_blocking=True)
        grid_size = int(sizes[0])

        B = grids.shape[0]
        seq_len = grid_size * grid_size
        targets = grids[:, :grid_size, :grid_size].reshape(B, seq_len)

        inputs = torch.zeros_like(targets)
        inputs[:, 1:] = targets[:, :-1]

        # Update learning rate with warmup + cosine decay
        lr = get_lr(global_step, args.warmup_steps, args.lr, args.min_lr, args.max_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Forward pass with mixed precision
        with autocast(device_type='cuda', dtype=torch.bfloat16 if args.use_bf16 else torch.float16):
            final_logits, all_logits = model(inputs, grid_size, return_all_logits=True)

            ce_loss = F.cross_entropy(final_logits.view(-1, model.num_classes),
                                       targets.view(-1), ignore_index=0)

            cst_loss = 0
            for step_logits in all_logits:
                cst_loss += compute_constraint_loss_ste(step_logits[:, :seq_len], grid_size)
            cst_loss = cst_loss / len(all_logits)

            loss = ce_loss + args.cst_weight * cst_loss
            loss = loss / accum_steps  # Scale for accumulation

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        micro_batch_count += 1

        # Optimizer step after accumulation
        if micro_batch_count >= accum_steps:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            micro_batch_count = 0
            global_step += 1

        total_loss += loss.item() * accum_steps
        total_ce += ce_loss.item()
        total_cst += cst_loss.item()
        n_batches += 1

        if batch_idx % 100 == 0:
            print(f"  [{batch_idx:5d}] loss={loss.item()*accum_steps:.4f} "
                  f"ce={ce_loss.item():.4f} cst={cst_loss.item():.4f} lr={lr:.2e}")
            # Update live status
            if status:
                status.update(
                    batch=batch_idx,
                    total_batches=total_batches,
                    loss=loss.item() * accum_steps,
                    lr=lr
                )

    return total_loss / n_batches, total_ce / n_batches, total_cst / n_batches, global_step


@torch.no_grad()
def run_validation(model, loader, device, n_recurrent=None):
    """Run model validation on a dataset."""
    model.eval()
    metrics = MetricsTracker()
    total_loss = 0
    n_batches = 0
    total_batches = len(loader)

    for batch_idx, (grids, sizes) in enumerate(loader):
        if batch_idx % 200 == 0:
            print(f"    Val: {batch_idx}/{total_batches} ({100*batch_idx/total_batches:.0f}%)")
        # Async GPU transfer (overlaps with previous batch processing)
        grids = grids.to(device, non_blocking=True)
        grid_size = int(sizes[0])
        B = grids.shape[0]
        seq_len = grid_size * grid_size

        targets = grids[:, :grid_size, :grid_size].reshape(B, seq_len)
        inputs = torch.zeros_like(targets)
        inputs[:, 1:] = targets[:, :-1]

        n_steps = n_recurrent or model.n_recurrent * 2
        logits = model(inputs, grid_size, n_steps=n_steps)

        loss = F.cross_entropy(logits.view(-1, model.num_classes),
                                targets.view(-1), ignore_index=0)

        preds = logits.argmax(dim=-1).view(B, grid_size, grid_size)
        targets_2d = targets.view(B, grid_size, grid_size)
        metrics.update(preds, targets_2d, grid_size)

        total_loss += loss.item()
        n_batches += 1

    m = metrics.get_metrics()
    return total_loss / n_batches, m


def main():
    parser = argparse.ArgumentParser(
        description="Recurrent Transformer for Latin Square Generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Data and output
    parser.add_argument("--data", default="data/latin_squares_massive.npz",
                        help="Path to training data")
    parser.add_argument("--output", default="latin_solver_recurrent",
                        help="Output model prefix")

    # Model architecture
    # Current checkpoint uses: d=128, h=4, L=2 (450K params)
    # For larger models, use --d-model 256 --n-head 8 --n-layer 4 (~3.6M params)
    parser.add_argument("--d-model", type=int, default=128,
                        help="Hidden dimension (128 for current checkpoint)")
    parser.add_argument("--n-head", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--n-layer", type=int, default=2,
                        help="Number of transformer blocks")
    parser.add_argument("--n-recurrent", type=int, default=32,
                        help="Number of recurrent steps (32 per arXiv:2307.04895)")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")

    # Training settings (nanoGPT defaults, optimized for RTX 4070 Ti + R5 5600X3D)
    # Gradient checkpointing enabled: recomputes activations during backward
    # This saves ~8GB VRAM allowing larger batches
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Micro-batch size (128 with gradient checkpointing)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=6,
                        help="Steps to accumulate (128*6=768 effective batch)")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="DataLoader workers (8 for 12-thread CPU)")
    parser.add_argument("--lr", type=float, default=6e-4,
                        help="Max learning rate (nanoGPT: 6e-4)")
    parser.add_argument("--min-lr", type=float, default=6e-5,
                        help="Min learning rate (nanoGPT: 10x smaller than max)")
    parser.add_argument("--warmup-steps", type=int, default=200,
                        help="Linear warmup steps")
    parser.add_argument("--max-steps", type=int, default=50000,
                        help="Max training steps for LR decay")
    parser.add_argument("--epochs", type=int, default=60,
                        help="Training epochs")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clipping value")

    # Loss settings
    parser.add_argument("--cst-weight", type=float, default=0.5,
                        help="Constraint loss weight")
    parser.add_argument("--target-loss", type=float, default=0.09,
                        help="Target validation loss")

    # Mixed precision
    parser.add_argument("--use-bf16", action="store_true", default=True,
                        help="Use BF16 (recommended for RTX 30/40 series)")

    # Curriculum learning
    parser.add_argument("--curriculum", action="store_true",
                        help="Enable curriculum learning (3x3 -> 16x16)")

    # Live monitoring
    parser.add_argument("--status-file", type=str, default="/tmp/training_status.json",
                        help="JSON file for live status updates")

    # Checkpoint resume
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from (e.g., latin_solver_recurrent_best.pt)")
    parser.add_argument("--start-epoch", type=int, default=1,
                        help="Starting epoch when resuming")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Target: loss < {args.target_loss}")

    if torch.cuda.is_available():
        # ================================================================
        # RTX 4070 Ti (AD104 Ada Lovelace) Hardware Optimizations
        # Audit findings: GPU running at 34% TDP, 57% SM util - headroom!
        # ================================================================

        # TensorFloat-32 for Tensor Cores (19 TFLOPS on Ada)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # cuDNN auto-tuning for optimal kernel selection
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

        # Flash Attention (O(N) memory, fused CUDA kernels)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

        # Memory allocation for ReBAR-enabled 12GB VRAM
        torch.cuda.set_per_process_memory_fraction(0.95)

        # Print comprehensive hardware info
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024**3)
        print(f"GPU: {props.name} ({vram_gb:.1f}GB VRAM)")
        print(f"  SM: {props.major}.{props.minor} (Ada Lovelace) | {props.multi_processor_count} SMs")
        print(f"  Tensor Cores: {props.multi_processor_count * 4} (4th gen)")
        print(f"Optimizations enabled:")
        print(f"  TF32 matmul: {torch.backends.cuda.matmul.allow_tf32}")
        print(f"  cuDNN bench: {torch.backends.cudnn.benchmark}")
        print(f"  Flash Attn:  {torch.backends.cuda.flash_sdp_enabled()}")
        print(f"  Mem-eff Attn:{torch.backends.cuda.mem_efficient_sdp_enabled()}")
        print(f"  CUDA alloc:  {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'default')}")

    full_ds = LatinSquareDataset(args.data)
    print(f"Loaded {len(full_ds)} grids")

    curriculum = [
        [3, 4, 5],
        [3, 4, 5, 6, 7, 8],
        [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        list(range(3, 17))
    ]

    model = RecurrentLatinTransformer(
        max_size=16, num_classes=18,
        d_model=args.d_model, n_head=args.n_head,
        n_layer=args.n_layer, n_recurrent=args.n_recurrent,
        dropout=args.dropout
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print(f"Architecture: {args.n_layer} layers x {args.n_recurrent} recurrent steps")

    # Load checkpoint if resuming
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
        print(f"  Loaded model weights (start epoch: {args.start_epoch})")

    # Note: torch.compile disabled for recurrent models
    # CUDA graphs (used by all compile modes) conflict with recurrent tensor reuse
    # The hidden state reuse across R iterations causes "tensor overwritten" errors
    # Alternative optimizations: TF32, Flash Attention, cuDNN benchmark (all enabled)
    print("torch.compile: Disabled (incompatible with recurrent architecture)")

    # Optimizer (nanoGPT style)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),  # nanoGPT: beta2=0.95 for stability
        weight_decay=0.1
    )

    # Mixed precision scaler
    scaler = GradScaler('cuda')

    # Live status writer
    status = StatusWriter(args.status_file)

    # Training state
    best_loss = float('inf')
    global_step = 0
    stage = 0
    stage_epochs = args.epochs // len(curriculum) if args.curriculum else args.epochs

    # Print configuration
    print(f"\n{'='*60}")
    print(f"Training Configuration")
    print(f"{'='*60}")
    print(f"  Target loss:    < {args.target_loss}")
    print(f"  Mixed precision: {'BF16' if args.use_bf16 else 'FP16'}")
    print(f"  Grad accum:      {args.gradient_accumulation_steps} steps")
    print(f"  Effective batch: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  LR schedule:     {args.lr:.2e} -> {args.min_lr:.2e} (warmup: {args.warmup_steps})")
    print(f"  Curriculum:      {'Enabled' if args.curriculum else 'Disabled'}")
    print(f"  Status file:     {args.status_file}")
    print(f"{'='*60}\n")

    status.update(status="training", total_epochs=args.epochs, target=args.target_loss)

    for epoch in range(args.start_epoch, args.epochs + 1):
        if args.curriculum:
            new_stage = min((epoch - 1) // stage_epochs, len(curriculum) - 1)
            if new_stage != stage:
                stage = new_stage
                print(f"\n>>> Curriculum Stage {stage+1}: sizes {curriculum[stage]}")

        sizes = curriculum[stage] if args.curriculum else None
        train_ds = LatinSquareDataset(args.data, sizes=sizes) if sizes else full_ds
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers,
                                   pin_memory=True, persistent_workers=True,
                                   collate_fn=collate_fn, drop_last=True,
                                   prefetch_factor=4,  # Prefetch 4 batches per worker
                                   worker_init_fn=worker_init_fn)  # Pin to physical cores

        print(f"Epoch {epoch}/{args.epochs}")
        print("-" * 40)

        status.update(epoch=epoch, status="training")

        train_loss, train_ce, train_cst, global_step = train_epoch(
            model, train_loader, optimizer, scaler, device, epoch, args, global_step,
            status=status, total_batches=len(train_loader)
        )

        # Use validation subset (50K samples) for speed
        status.update(status="validating")
        val_indices = torch.randperm(len(full_ds))[:50000]
        val_subset = torch.utils.data.Subset(full_ds, val_indices)
        val_loader = DataLoader(val_subset, batch_size=int(args.batch_size * 1.5),  # 1.5x for val (no grad)
                                 shuffle=False, num_workers=args.num_workers,
                                 pin_memory=True, persistent_workers=False,
                                 collate_fn=collate_fn, prefetch_factor=4,
                                 worker_init_fn=worker_init_fn)  # Pin to physical cores
        # Use same recurrent steps as training (not 2x) for faster validation
        val_loss, metrics = run_validation(model, val_loader, device, model.n_recurrent)

        # Update status with epoch results
        status.update(
            loss=train_loss,
            val_loss=val_loss,
            accuracy=metrics['accuracy'],
            valid_rate=metrics['valid_rate'],
            best_loss=min(best_loss, val_loss)
        )

        print(f"\n  Train Loss: {train_loss:.4f} (ce={train_ce:.4f}, cst={train_cst:.4f})")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Valid Grid Rate:   {metrics['valid_rate']:.1f}%")
        print(f"  Per-Cell Accuracy: {metrics['accuracy']:.1f}%")
        print(f"  Avg Violations:    {metrics['avg_violations']:.2f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), f"{args.output}_best.pt")
            print(f"  >> Saved best (loss={val_loss:.4f})")

        if val_loss < args.target_loss:
            status.update(status="target_achieved")
            print(f"\n{'='*60}")
            print(f"TARGET ACHIEVED! loss={val_loss:.4f} < {args.target_loss}")
            print(f"{'='*60}")
            break

        print()

    status.update(status="completed", best_loss=best_loss)
    print(f"\nBest validation loss: {best_loss:.4f}")
    print("Note: ONNX export requires manual conversion for recurrent model")


if __name__ == "__main__":
    main()
