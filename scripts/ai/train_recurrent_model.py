#!/usr/bin/env python3
"""
train_recurrent_model.py: Enhanced Recurrent Transformer for Latin Squares

Based on research findings from arXiv:2307.04895 (Recurrent Transformers for CSPs)
which achieves 93.5% vs 64.8% baseline on Sudoku puzzles.

Key improvements over train_massive_model.py:
- Extended grid support: 3x3 to 16x16 (hex digits for 10-16)
- Learnable 2D positional encoding (separate row/column embeddings)
- Recurrent Transformer blocks with weight sharing
- Constraint loss for Latin square violations

SPDX-License-Identifier: MIT
SPDX-FileCopyrightText: Copyright (C) 2024-2025 Orthogon Contributors
"""

import argparse
import math
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split


class ExtendedLatinSquareDataset(Dataset):
    """
    Dataset for Latin squares from 3x3 to 16x16.

    Extended sizes (10-16) use hex-style representation internally
    but are stored as integers 0-16 (0 = empty, 1-16 = digits).
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        max_size: int = 16,
        mask_ratio: float = 0.7,
    ):
        self.max_size = max_size
        self.mask_ratio = mask_ratio
        self.samples = []

        if data_path and os.path.exists(data_path):
            data = np.load(data_path)
            for key in data.files:
                if not key.startswith("size"):
                    continue
                size = int(key.replace("size", ""))
                if size < 3 or size > max_size:
                    continue
                grids = data[key]
                for g in grids:
                    self.samples.append((g, size))
            print(f"Loaded {len(self.samples)} samples from {data_path}")
        else:
            print(f"No data file found at {data_path}, using random generation")

    def __len__(self) -> int:
        return len(self.samples) if self.samples else 10000

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.samples:
            # Fallback: random valid Latin square (for testing)
            size = np.random.randint(3, min(10, self.max_size + 1))
            grid = self._generate_random_latin_square(size)
        else:
            grid, size = self.samples[idx % len(self.samples)]

        # Pad to max_size
        padded = np.zeros((self.max_size, self.max_size), dtype=np.int64)
        padded[:size, :size] = grid

        target = torch.from_numpy(padded).long()

        # Create masked input (70% masked by default)
        mask = torch.rand(padded.shape) < self.mask_ratio
        inp = target.clone()
        inp[mask] = 0
        # Clear padding area
        inp[size:, :] = 0
        inp[:, size:] = 0

        return inp, target, torch.tensor(size).long()

    def _generate_random_latin_square(self, n: int) -> np.ndarray:
        """Generate a random Latin square using row permutation."""
        base = np.arange(1, n + 1)
        grid = np.zeros((n, n), dtype=np.int64)
        for i in range(n):
            grid[i] = np.roll(base, i)
        # Shuffle rows and columns
        np.random.shuffle(grid)
        grid = grid[:, np.random.permutation(n)]
        return grid


class LearnablePositionalEncoding2D(nn.Module):
    """
    2D positional encoding with separate learnable row and column embeddings.

    This allows the model to understand grid structure better than 1D flattened
    positional encoding. Each position (i,j) gets: row_embed[i] + col_embed[j]
    """

    def __init__(self, max_size: int, d_model: int):
        super().__init__()
        self.row_embed = nn.Embedding(max_size, d_model)
        self.col_embed = nn.Embedding(max_size, d_model)

        # Initialize with small values
        nn.init.normal_(self.row_embed.weight, std=0.02)
        nn.init.normal_(self.col_embed.weight, std=0.02)

    def forward(self, batch_size: int, height: int, width: int) -> torch.Tensor:
        """
        Returns positional encoding of shape [B, H*W, D].
        """
        device = self.row_embed.weight.device

        rows = torch.arange(height, device=device)
        cols = torch.arange(width, device=device)

        # Create 2D grid of row and column indices
        row_pos = rows.unsqueeze(1).expand(height, width)  # [H, W]
        col_pos = cols.unsqueeze(0).expand(height, width)  # [H, W]

        # Get embeddings and combine
        row_enc = self.row_embed(row_pos)  # [H, W, D]
        col_enc = self.col_embed(col_pos)  # [H, W, D]

        pos_enc = row_enc + col_enc  # [H, W, D]
        pos_enc = pos_enc.view(height * width, -1)  # [H*W, D]

        return pos_enc.unsqueeze(0).expand(batch_size, -1, -1)  # [B, H*W, D]


class RecurrentTransformerBlock(nn.Module):
    """
    Single Transformer block designed for weight-sharing across iterations.

    Based on arXiv:2307.04895: each iteration refines the solution using
    the same weights, similar to how iterative message-passing works in
    belief propagation.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Pre-LayerNorm architecture (more stable for deep/recurrent networks)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention
        normed = self.norm1(x)
        attn_out, _ = self.self_attn(normed, normed, normed)
        x = x + attn_out

        # Pre-norm FFN
        normed = self.norm2(x)
        x = x + self.ffn(normed)

        return x


class RecurrentLatinTransformer(nn.Module):
    """
    Recurrent Transformer for Latin Square solving (3x3 to 16x16).

    Architecture:
    1. Token embedding + 2D positional encoding
    2. Initial projection
    3. Recurrent Transformer blocks (weight-shared across iterations)
    4. Output projection to class logits

    The recurrent nature allows iterative refinement of the solution,
    similar to how humans solve puzzles by repeatedly scanning for constraints.
    """

    def __init__(
        self,
        max_size: int = 16,
        num_classes: int = 17,  # 0=empty + 1-16 for hex grids
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        num_iterations: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_size = max_size
        self.num_classes = num_classes
        self.num_iterations = num_iterations

        # Token embedding
        self.embedding = nn.Embedding(num_classes, d_model)

        # 2D positional encoding
        self.pos_encoding = LearnablePositionalEncoding2D(max_size, d_model)

        # Initial projection
        self.input_proj = nn.Linear(d_model, d_model)

        # Recurrent Transformer layers (shared weights)
        self.layers = nn.ModuleList([
            RecurrentTransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # Final normalization
        self.final_norm = nn.LayerNorm(d_model)

        # Output head
        self.output_head = nn.Linear(d_model, num_classes)

    def forward(
        self, x: torch.Tensor, return_all_iterations: bool = False
    ) -> torch.Tensor:
        """
        Forward pass with iterative refinement.

        Args:
            x: Input grid [B, H, W] with 0=empty, 1-N=digits
            return_all_iterations: If True, return outputs from all iterations

        Returns:
            Logits [B, C, H, W] or list of logits for each iteration
        """
        B, H, W = x.shape

        # Flatten grid to sequence
        x_flat = x.view(B, -1)  # [B, H*W]

        # Token embedding
        h = self.embedding(x_flat)  # [B, H*W, D]

        # Add 2D positional encoding
        pos_enc = self.pos_encoding(B, H, W)
        h = h + pos_enc

        # Initial projection
        h = self.input_proj(h)

        # Collect outputs from all iterations if requested
        all_outputs = []

        # Iterative refinement
        for iteration in range(self.num_iterations):
            # Apply all layers
            for layer in self.layers:
                h = layer(h)

            if return_all_iterations:
                normed = self.final_norm(h)
                logits = self.output_head(normed)
                logits = logits.view(B, H, W, -1).permute(0, 3, 1, 2)
                all_outputs.append(logits)

        if return_all_iterations:
            return all_outputs

        # Final output
        h = self.final_norm(h)
        logits = self.output_head(h)  # [B, H*W, C]

        # Reshape to grid format
        logits = logits.view(B, H, W, -1).permute(0, 3, 1, 2)  # [B, C, H, W]

        return logits


def constraint_loss(
    logits: torch.Tensor, size_tensor: torch.Tensor, weight: float = 0.1
) -> torch.Tensor:
    """
    Additional loss term for Latin square constraint violations.

    Penalizes:
    - Duplicate predictions in rows
    - Duplicate predictions in columns

    This guides the model to learn Latin square structure explicitly,
    not just from data patterns.

    Args:
        logits: [B, C, H, W] class logits
        size_tensor: [B] actual grid sizes (for padding handling)
        weight: Loss weight

    Returns:
        Scalar constraint loss
    """
    B, C, H, W = logits.shape
    device = logits.device

    # Convert to probabilities
    probs = F.softmax(logits, dim=1)  # [B, C, H, W]

    total_loss = torch.tensor(0.0, device=device)

    for b in range(B):
        size = size_tensor[b].item()

        # Extract valid region probabilities
        valid_probs = probs[b, 1:size+1, :size, :size]  # [size, size, size]

        # Row constraint: sum of probabilities per class per row should be ~1
        row_sums = valid_probs.sum(dim=2)  # [size, size] - sum across columns
        row_violation = (row_sums - 1.0).pow(2).mean()

        # Column constraint: sum of probabilities per class per column should be ~1
        col_sums = valid_probs.sum(dim=1)  # [size, size] - sum across rows
        col_violation = (col_sums - 1.0).pow(2).mean()

        total_loss = total_loss + row_violation + col_violation

    return weight * total_loss / B


def get_gpu_info() -> dict:
    """Detect GPU capabilities for optimal compile settings."""
    if not torch.cuda.is_available():
        return {"name": "CPU", "sm_count": 0, "compute_cap": (0, 0)}
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "sm_count": props.multi_processor_count,
        "compute_cap": (props.major, props.minor),
    }


def train(args):
    """Main training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    gpu_info = get_gpu_info()
    print(f"GPU: {gpu_info['name']} ({gpu_info['sm_count']} SMs)")
    print(f"Training Recurrent Model (3x3 to {args.max_size}x{args.max_size})")
    print(f"Device: {device}, AMP: {use_amp}")
    print(f"Iterations: {args.iterations}, Layers: {args.layers}")

    # Enable TF32 for tensor cores (Ampere+)
    if device.type == "cuda" and gpu_info["compute_cap"][0] >= 8:
        torch.set_float32_matmul_precision("high")
        print("TF32 enabled for tensor cores.")

    # Load dataset
    full_dataset = ExtendedLatinSquareDataset(
        args.data_path, max_size=args.max_size, mask_ratio=args.mask_ratio
    )

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Create model
    num_classes = args.max_size + 1  # 0=empty + 1-N
    model = RecurrentLatinTransformer(
        max_size=args.max_size,
        num_classes=num_classes,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.layers,
        num_iterations=args.iterations,
        dim_feedforward=args.d_model * 4,
        dropout=args.dropout,
    ).to(device)

    # Count parameters
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # Optional torch.compile
    if args.compile and hasattr(torch, "compile") and device.type == "cuda":
        compile_mode = "max-autotune" if gpu_info["sm_count"] >= 80 else "reduce-overhead"
        print(f"torch.compile mode: {compile_mode}")
        model = torch.compile(model, mode=compile_mode)

    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_constraint = 0.0

        for i, (inp, target, size) in enumerate(train_loader):
            inp, target, size = inp.to(device), target.to(device), size.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp):
                output = model(inp)
                ce_loss = criterion(output, target)
                c_loss = constraint_loss(output, size, weight=args.constraint_weight)
                loss = ce_loss + c_loss

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            train_loss += ce_loss.item()
            train_constraint += c_loss.item()

            if i % 100 == 0:
                print(
                    f"Epoch {epoch} [{i}/{len(train_loader)}] "
                    f"CE: {ce_loss.item():.4f} Constraint: {c_loss.item():.4f}"
                )

        # Validation phase
        model.train(False)
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inp, target, size in val_loader:
                inp, target, size = inp.to(device), target.to(device), size.to(device)
                output = model(inp)
                val_loss += criterion(output, target).item()

                # Calculate accuracy
                pred = output.argmax(dim=1)
                for b in range(inp.shape[0]):
                    s = size[b].item()
                    correct += (pred[b, :s, :s] == target[b, :s, :s]).sum().item()
                    total += s * s

        val_loss /= max(len(val_loader), 1)
        accuracy = correct / max(total, 1) * 100

        print(f"Epoch {epoch} Val Loss: {val_loss:.4f} Accuracy: {accuracy:.2f}%")
        print(f"  Train CE: {train_loss/len(train_loader):.4f} "
              f"Constraint: {train_constraint/len(train_loader):.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.output.replace(".onnx", "_best.pth"))

        scheduler.step()

    # Save final model
    pth_path = args.output.replace(".onnx", ".pth")
    torch.save(model.state_dict(), pth_path)
    print(f"Model saved: {pth_path}")

    # Export to ONNX
    export_onnx(model, args.max_size, args.output, device)


def export_onnx(model: nn.Module, max_size: int, output_path: str, device: torch.device):
    """Export model to ONNX format with dynamic axes."""
    print(f"Exporting to ONNX: {output_path}")

    model = model.module if hasattr(model, "module") else model
    model = model._orig_mod if hasattr(model, "_orig_mod") else model
    model.eval()

    dummy_input = torch.zeros(1, max_size, max_size, dtype=torch.long, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input_grid"],
        output_names=["cell_logits"],
        dynamic_axes={
            "input_grid": {0: "batch_size"},
            "cell_logits": {0: "batch_size"},
        },
        opset_version=17,
    )
    print(f"ONNX export complete: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train Recurrent Transformer for Latin Squares"
    )
    parser.add_argument(
        "--data-path", type=str, default="data/latin_squares_extended.npz",
        help="Path to training data"
    )
    parser.add_argument("--max-size", type=int, default=16, help="Maximum grid size")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--d-model", type=int, default=256, help="Model dimension")
    parser.add_argument("--nhead", type=int, default=8, help="Attention heads")
    parser.add_argument("--layers", type=int, default=4, help="Transformer layers")
    parser.add_argument("--iterations", type=int, default=8, help="Recurrent iterations")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--mask-ratio", type=float, default=0.7, help="Input mask ratio")
    parser.add_argument("--constraint-weight", type=float, default=0.1, help="Constraint loss weight")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument(
        "--output", type=str, default="keen_solver_16x16.onnx",
        help="Output ONNX file"
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
