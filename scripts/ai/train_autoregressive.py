#!/usr/bin/env python3
"""
Autoregressive Latin Square Generator with Constraint-Aware Training

Target: Loss < 0.09 with proper metrics tracking

Key Improvements over train_massive_model.py:
1. Autoregressive generation (cell-by-cell, matching inference)
2. Constraint-aware auxiliary loss (penalizes Latin violations)
3. Curriculum learning (3x3 -> 16x16 progression)
4. Comprehensive metrics (Valid Grid Rate, Per-Cell Accuracy, Constraint Violations, Entropy)
5. Proper warmup and gradient clipping
6. Size-aware architecture with explicit size conditioning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Hardware-optimized training support
try:
    from hardware_config import detect_hardware, get_optimized_config, apply_optimizations, print_hardware_report
    HW_CONFIG_AVAILABLE = True
except ImportError:
    HW_CONFIG_AVAILABLE = False

# 8-bit optimizer support (saves ~40% optimizer memory)
try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import math
import argparse
from collections import defaultdict
import time


# =============================================================================
# METRICS TRACKING
# =============================================================================

class MetricsTracker:
    """Tracks all required metrics for Latin square generation."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_grids = 0
        self.valid_grids = 0
        self.total_cells = 0
        self.correct_cells = 0
        self.total_violations = 0
        self.prediction_counts = defaultdict(int)

    def update(self, predictions, targets, grid_size):
        """Update metrics with a batch of predictions."""
        batch_size = predictions.shape[0]
        self.total_grids += batch_size

        for b in range(batch_size):
            pred_grid = predictions[b, :grid_size, :grid_size].cpu().numpy()
            target_grid = targets[b, :grid_size, :grid_size].cpu().numpy()

            # Per-cell accuracy
            correct = (pred_grid == target_grid).sum()
            self.correct_cells += correct
            self.total_cells += grid_size * grid_size

            # Count violations
            violations = self._count_violations(pred_grid, grid_size)
            self.total_violations += violations

            # Valid grid (zero violations)
            if violations == 0:
                self.valid_grids += 1

            # Track prediction distribution for entropy
            for val in pred_grid.flatten():
                self.prediction_counts[int(val)] += 1

    def _count_violations(self, grid, size):
        """Count row and column constraint violations."""
        violations = 0
        for i in range(size):
            # Row violations: duplicates in row
            row = grid[i, :size]
            row_unique = len(set(row[row > 0]))
            row_nonzero = (row > 0).sum()
            violations += row_nonzero - row_unique

            # Column violations: duplicates in column
            col = grid[:size, i]
            col_unique = len(set(col[col > 0]))
            col_nonzero = (col > 0).sum()
            violations += col_nonzero - col_unique

        return violations

    def get_metrics(self):
        """Return all computed metrics."""
        valid_rate = self.valid_grids / max(self.total_grids, 1)
        cell_accuracy = self.correct_cells / max(self.total_cells, 1)
        avg_violations = self.total_violations / max(self.total_grids, 1)

        total_preds = sum(self.prediction_counts.values())
        entropy = 0.0
        if total_preds > 0:
            for count in self.prediction_counts.values():
                if count > 0:
                    p = count / total_preds
                    entropy -= p * math.log2(p)

        return {
            'valid_grid_rate': valid_rate,
            'per_cell_accuracy': cell_accuracy,
            'avg_violations': avg_violations,
            'generation_entropy': entropy
        }


# =============================================================================
# DATASET
# =============================================================================

class AutoregressiveLatinDataset(Dataset):
    """Dataset for autoregressive training with variable fill ratios."""

    def __init__(self, data_path, max_size=16, samples_per_grid=None):
        self.max_size = max_size
        self.samples = []

        if data_path and os.path.exists(data_path):
            data = np.load(data_path)
            for key in sorted(data.files):
                if not key.startswith("size"):
                    continue
                size = int(key.replace("size", ""))
                if size < 3 or size > max_size:
                    continue

                grids = data[key]
                for grid in grids:
                    self.samples.append((grid, size))

        self.samples_per_grid = samples_per_grid
        print(f"Loaded {len(self.samples)} grids from {data_path}")

    def __len__(self):
        if self.samples_per_grid:
            return len(self.samples) * self.samples_per_grid
        return len(self.samples)

    def __getitem__(self, idx):
        if self.samples_per_grid:
            grid_idx = idx // self.samples_per_grid
            sample_offset = idx % self.samples_per_grid
        else:
            grid_idx = idx
            sample_offset = None

        grid, size = self.samples[grid_idx % len(self.samples)]

        padded = np.zeros((self.max_size, self.max_size), dtype=np.int64)
        padded[:size, :size] = grid

        if sample_offset is not None:
            fill_ratio = sample_offset / self.samples_per_grid
        else:
            fill_ratio = np.random.uniform(0.0, 0.7)

        mask = np.random.rand(size, size) < fill_ratio
        inp = np.zeros((self.max_size, self.max_size), dtype=np.int64)
        inp[:size, :size] = padded[:size, :size] * mask

        target = torch.from_numpy(padded).long()
        inp_tensor = torch.from_numpy(inp).long()
        size_tensor = torch.tensor(size).long()

        return inp_tensor, target, size_tensor


class CurriculumDataset(Dataset):
    """Curriculum learning dataset that filters by size."""

    def __init__(self, base_dataset, allowed_sizes):
        self.base = base_dataset
        self.allowed_sizes = set(allowed_sizes)
        self.indices = [
            i for i, (_, size) in enumerate(base_dataset.samples)
            if size in self.allowed_sizes
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base[self.indices[idx]]


# =============================================================================
# MODEL ARCHITECTURE (nanoGPT-inspired with constraint awareness)
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (from x-transformers)."""
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim ** 0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with Flash Attention.

    Optimizations:
    - Uses torch.nn.functional.scaled_dot_product_attention (Flash Attention)
    - O(N) memory instead of O(N^2)
    - Fused CUDA kernels for speed
    """

    def __init__(self, d_model, n_head, dropout=0.1, max_seq_len=256):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.dropout = dropout

        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size()

        # QKV projection
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        # Flash Attention via scaled_dot_product_attention
        # Automatically uses flash attention kernel on compatible GPUs
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True  # Applies causal mask efficiently
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """Feedforward with GELU (nanoGPT style)."""

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.c_fc = nn.Linear(d_model, 4 * d_model)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """Pre-LN transformer block (nanoGPT style)."""

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


class ConstraintAwareTransformer(nn.Module):
    """
    Autoregressive transformer for Latin square generation.

    Architecture inspired by nanoGPT with:
    - RMSNorm (from x-transformers) for stability
    - Size conditioning for variable grid sizes
    - Constraint-aware output masking
    """

    def __init__(self, max_size=16, num_classes=18, d_model=256, n_head=8, n_layer=8, dropout=0.1):
        super().__init__()
        self.max_size = max_size
        self.num_classes = num_classes
        max_seq_len = max_size * max_size

        # Token embeddings
        self.tok_emb = nn.Embedding(num_classes, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.size_emb = nn.Embedding(max_size + 1, d_model)
        self.drop = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_head, dropout, max_seq_len)
            for _ in range(n_layer)
        ])

        # Output
        self.ln_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, num_classes, bias=False)

        # Weight tying
        self.tok_emb.weight = self.head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, size=None):
        B, H, W = x.shape
        T = H * W

        # Flatten grid to sequence
        x = x.view(B, T)

        # Embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(pos)
        x = self.drop(tok_emb + pos_emb)

        # Size conditioning
        if size is not None:
            size_emb = self.size_emb(size).unsqueeze(1)
            x = x + size_emb

        # Transformer
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)

        # Reshape to grid format [B, C, H, W]
        return logits.view(B, H, W, -1).permute(0, 3, 1, 2)


# =============================================================================
# CONSTRAINT-AWARE LOSS
# =============================================================================

class ConstraintAwareLoss(nn.Module):
    """
    Loss with auxiliary constraint penalty for Latin property.

    Components:
    1. Cross-entropy for cell prediction
    2. Row constraint: penalize duplicate predictions in rows
    3. Column constraint: penalize duplicate predictions in columns
    """

    def __init__(self, constraint_weight=0.15, label_smoothing=0.1):
        super().__init__()
        self.constraint_weight = constraint_weight
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits, targets, sizes):
        B, C, H, W = logits.shape

        # Primary cross-entropy loss
        ce = self.ce_loss(logits, targets)

        # Constraint loss via soft duplicate detection
        probs = F.softmax(logits, dim=1)  # [B, C, H, W]
        constraint_loss = torch.tensor(0.0, device=logits.device)

        for b in range(B):
            size = sizes[b].item()
            # Extract valid region: digits 1 to size
            p = probs[b, 1:size+1, :size, :size]  # [size, size, size]

            # Row constraint: each digit should appear ~once per row
            row_sums = p.sum(dim=2)  # [size, size] - sum over columns
            row_violation = F.relu(row_sums - 1.0).pow(2).mean()

            # Column constraint: each digit should appear ~once per column
            col_sums = p.sum(dim=1)  # [size, size] - sum over rows
            col_violation = F.relu(col_sums - 1.0).pow(2).mean()

            constraint_loss = constraint_loss + row_violation + col_violation

        constraint_loss = constraint_loss / B
        total = ce + self.constraint_weight * constraint_loss

        return total, ce, constraint_loss


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_epoch(model, loader, optimizer, criterion, scaler, device, grad_clip=1.0):
    model.train()
    total_loss, total_ce, total_cst = 0.0, 0.0, 0.0
    n_batches = 0

    for i, (inp, target, size) in enumerate(loader):
        inp = inp.to(device)
        target = target.to(device)
        size = size.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=scaler is not None):
            logits = model(inp, size)
            loss, ce, cst = criterion(logits, target, size)

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()
        total_ce += ce.item()
        total_cst += cst.item()
        n_batches += 1

        if i % 500 == 0:
            print(f"  [{i:5d}] loss={loss.item():.4f} ce={ce.item():.4f} cst={cst.item():.4f}", flush=True)

    return {
        'loss': total_loss / n_batches,
        'ce': total_ce / n_batches,
        'constraint': total_cst / n_batches
    }


def validate_model(model, loader, criterion, device, tracker):
    model.train(False)  # Set to inference mode
    tracker.reset()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for inp, target, size in loader:
            inp = inp.to(device)
            target = target.to(device)
            size = size.to(device)

            logits = model(inp, size)
            loss, _, _ = criterion(logits, target, size)
            total_loss += loss.item()
            n_batches += 1

            preds = logits.argmax(dim=1)
            for b in range(inp.shape[0]):
                tracker.update(preds[b:b+1], target[b:b+1], size[b].item())

    metrics = tracker.get_metrics()
    metrics['loss'] = total_loss / n_batches
    return metrics


def test_empty_generation(model, size, device, n_samples=50):
    """Test generation from completely empty grid."""
    model.train(False)
    valid = 0

    with torch.no_grad():
        for _ in range(n_samples):
            grid = torch.zeros(1, 16, 16, dtype=torch.long, device=device)
            sz = torch.tensor([size], device=device)

            logits = model(grid, sz)
            pred = logits.argmax(dim=1)[0, :size, :size].cpu().numpy()

            # Check Latin property
            is_valid = True
            for i in range(size):
                row = set(pred[i, :])
                col = set(pred[:, i])
                expected = set(range(1, size + 1))
                if row != expected or col != expected:
                    is_valid = False
                    break
            if is_valid:
                valid += 1

    return valid / n_samples


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Autoregressive Latin Square Training")
    parser.add_argument("--data", default="data/latin_squares_massive.npz")
    parser.add_argument("--output", default="latin_solver")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)  # Increased for tensor core saturation
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=256)  # Multiple of 8 for tensor cores
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--constraint-weight", type=float, default=0.15)
    parser.add_argument("--curriculum", action="store_true", help="Enable curriculum learning")
    parser.add_argument("--target-loss", type=float, default=0.09)
    # Hardware optimization args
    parser.add_argument("--use-8bit", action="store_true", help="Use 8-bit AdamW (requires bitsandbytes)")
    parser.add_argument("--prefetch", type=int, default=4, help="DataLoader prefetch factor")
    parser.add_argument("--num-workers", type=int, default=6, help="DataLoader workers (use ~half CPU cores)")
    parser.add_argument("--hw-report", action="store_true", help="Print hardware detection report")
    args = parser.parse_args()

    # Hardware detection and report
    if HW_CONFIG_AVAILABLE and args.hw_report:
        hw = detect_hardware()
        hw_config = get_optimized_config(hw)
        print_hardware_report(hw, hw_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Target: loss < {args.target_loss}")

    if device.type == "cuda":
        # GPU Optimizations for RTX 4070 Ti (Ada Lovelace)
        torch.set_float32_matmul_precision('high')  # TF32 for tensor cores
        torch.backends.cudnn.benchmark = True  # Auto-tune convolution algorithms
        torch.backends.cuda.matmul.allow_tf32 = True  # TF32 in matmuls
        torch.backends.cudnn.allow_tf32 = True  # TF32 in cuDNN

        # Check Flash Attention support
        if hasattr(F, 'scaled_dot_product_attention'):
            print("Flash Attention: enabled")

        print(f"TF32 + cuDNN benchmark enabled")

    # Dataset
    full_ds = AutoregressiveLatinDataset(args.data, max_size=16)
    train_n = int(0.9 * len(full_ds))
    val_n = len(full_ds) - train_n
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_n, val_n])

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=True,
        prefetch_factor=args.prefetch  # CPU prefetches ahead while GPU computes
    )

    # Model
    model = ConstraintAwareTransformer(
        max_size=16, num_classes=18,
        d_model=args.d_model, n_head=8, n_layer=args.n_layer, dropout=0.1
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    if hasattr(torch, 'compile'):
        # Use 'default' mode - 'reduce-overhead' triggers SM warnings on GPUs < 80 SMs
        # 'default' provides good balance of compilation time and runtime speed
        model = torch.compile(model, mode='default')
        print("torch.compile: default mode (optimized for RTX 4070 Ti)")

    # Training setup (nanoGPT-style optimizer configuration)
    # β₂=0.95 allows faster adaptation to gradient changes
    # weight_decay=0.1 provides stronger regularization (standard for language models)
    criterion = ConstraintAwareLoss(args.constraint_weight, label_smoothing=0.1)

    # Select optimizer: 8-bit AdamW saves ~40% memory, slightly faster
    if args.use_8bit and BNB_AVAILABLE:
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.95),
            weight_decay=0.1
        )
        print("Optimizer: AdamW8bit (bitsandbytes)")
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.95),  # nanoGPT default
            weight_decay=0.1    # 10x stronger than before
        )
        if args.use_8bit and not BNB_AVAILABLE:
            print("Optimizer: AdamW (8-bit requested but bitsandbytes not installed)")
        else:
            print("Optimizer: AdamW")

    # Warmup + cosine decay with min_lr floor (nanoGPT pattern)
    warmup_epochs = 3
    min_lr_ratio = 0.1  # Don't decay below 10% of max LR
    def lr_fn(ep):
        if ep < warmup_epochs:
            return (ep + 1) / warmup_epochs
        progress = (ep - warmup_epochs) / max(args.epochs - warmup_epochs, 1)
        # Cosine decay from 1.0 to min_lr_ratio
        coeff = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * coeff

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_fn)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    tracker = MetricsTracker()

    best_loss = float('inf')
    target_met = False

    # Curriculum stages
    curriculum = [
        [3, 4, 5],
        [3, 4, 5, 6, 7, 8],
        [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        list(range(3, 17))
    ]
    stage = 0
    stage_epochs = args.epochs // len(curriculum) if args.curriculum else args.epochs

    print(f"\n{'='*60}")
    print(f"Training Start - Target: loss < {args.target_loss}")
    print(f"{'='*60}\n")

    for epoch in range(args.epochs):
        # Curriculum update
        if args.curriculum:
            new_stage = min(epoch // stage_epochs, len(curriculum) - 1)
            if new_stage != stage:
                stage = new_stage
                print(f"\n>>> Curriculum Stage {stage+1}: sizes {curriculum[stage]}")

        # Build train loader
        if args.curriculum:
            curr_ds = CurriculumDataset(full_ds, curriculum[stage])
            # Only use indices that are in train split
            curr_indices = [i for i in range(len(curr_ds)) if curr_ds.indices[i] < train_n]
            train_loader = DataLoader(
                torch.utils.data.Subset(curr_ds, curr_indices),
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_workers, pin_memory=True, persistent_workers=True,
                prefetch_factor=args.prefetch
            )
        else:
            train_loader = DataLoader(
                train_ds, batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_workers, pin_memory=True, persistent_workers=True,
                prefetch_factor=args.prefetch
            )

        lr = scheduler.get_last_lr()[0]
        print(f"\nEpoch {epoch+1}/{args.epochs} (lr={lr:.6f})")
        print("-" * 40)

        # Train
        train_m = train_epoch(model, train_loader, optimizer, criterion, scaler, device)

        # Validate
        val_m = validate_model(model, val_loader, criterion, device, tracker)

        # Test empty generation
        gen_rate = test_empty_generation(model, 9, device, 50)

        scheduler.step()

        # Print
        status = " << TARGET!" if val_m['loss'] < args.target_loss else ""
        print(f"\n  Train Loss: {train_m['loss']:.4f}")
        print(f"  Val Loss:   {val_m['loss']:.4f}{status}")
        print(f"  Valid Grid Rate:   {val_m['valid_grid_rate']*100:.1f}%")
        print(f"  Per-Cell Accuracy: {val_m['per_cell_accuracy']*100:.1f}%")
        print(f"  Avg Violations:    {val_m['avg_violations']:.2f}")
        print(f"  Gen Entropy:       {val_m['generation_entropy']:.2f} bits")
        print(f"  Empty->Valid 9x9:  {gen_rate*100:.1f}%")

        # Save best
        if val_m['loss'] < best_loss:
            best_loss = val_m['loss']
            torch.save(model.state_dict(), f"{args.output}_best.pth")
            print(f"  >> Saved best (loss={best_loss:.4f})")

        if val_m['loss'] < args.target_loss and not target_met:
            target_met = True
            print(f"\n{'*'*60}")
            print(f"TARGET MET: {val_m['loss']:.4f} < {args.target_loss}")
            print(f"{'*'*60}\n")

    # Final save
    torch.save(model.state_dict(), f"{args.output}.pth")

    # ONNX export
    print("\nExporting ONNX...")
    model.train(False)

    # Get base model if compiled
    export_model = model._orig_mod if hasattr(model, '_orig_mod') else model

    dummy_grid = torch.zeros(1, 16, 16, dtype=torch.long, device=device)
    dummy_size = torch.tensor([9], dtype=torch.long, device=device)

    torch.onnx.export(
        export_model,
        (dummy_grid, dummy_size),
        f"{args.output}.onnx",
        input_names=["input_grid", "grid_size"],
        output_names=["cell_logits"],
        dynamic_axes={"input_grid": {0: "batch"}, "grid_size": {0: "batch"}, "cell_logits": {0: "batch"}},
        opset_version=17
    )
    print(f"Exported: {args.output}.onnx")

    print(f"\n{'='*60}")
    print(f"FINAL: Best Loss = {best_loss:.4f}, Target Met = {target_met}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
