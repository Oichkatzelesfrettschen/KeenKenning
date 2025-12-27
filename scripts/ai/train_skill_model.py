#!/usr/bin/env python3
"""
train_skill_model.py: Train a lightweight skill model for adaptive difficulty

This creates a 3-layer MLP (~50KB) that predicts player skill from performance
metrics. The model is exported to ONNX for on-device inference via ONNX Runtime.

Architecture:
  Input: 8 features (normalized)
  Hidden1: Dense(32, ReLU)
  Hidden2: Dense(16, ReLU)
  Output: Dense(1, Sigmoid) -> skill score 0-1

SPDX-License-Identifier: MIT
SPDX-FileCopyrightText: Copyright (C) 2024-2025 Orthogon Contributors
"""

import argparse
from pathlib import Path

import numpy as np

# Lazy imports for optional dependencies
torch = None
onnx = None


def lazy_import_torch():
    """Import PyTorch lazily to allow running without it for data generation."""
    global torch
    if torch is None:
        import torch as torch_module
        torch = torch_module


# ============================================================================
# Feature Engineering
# ============================================================================

# Input feature indices and names
FEATURE_NAMES = [
    "solve_time_ratio",    # actual_time / target_time (0.5-3.0 typical)
    "error_rate",          # errors / cells (0.0-0.5 typical)
    "hint_rate",           # hints / puzzles (0.0-1.0)
    "undo_rate",           # undos / cells (0.0-0.5 typical)
    "streak",              # current win streak (0-20 clamped)
    "puzzle_size",         # grid size (3-16, normalized)
    "difficulty",          # difficulty level (0-4, normalized)
    "session_puzzles",     # puzzles completed this session (0-50 clamped)
]

NUM_FEATURES = len(FEATURE_NAMES)


def normalize_features(raw_features: np.ndarray) -> np.ndarray:
    """
    Normalize raw features to 0-1 range for model input.

    Args:
        raw_features: Array of shape (batch, 8) with raw feature values

    Returns:
        Normalized features in 0-1 range
    """
    # Define normalization ranges (min, max) for each feature
    ranges = np.array([
        [0.0, 3.0],     # solve_time_ratio
        [0.0, 0.5],     # error_rate
        [0.0, 1.0],     # hint_rate
        [0.0, 0.5],     # undo_rate
        [0.0, 20.0],    # streak
        [3.0, 16.0],    # puzzle_size
        [0.0, 4.0],     # difficulty
        [0.0, 50.0],    # session_puzzles
    ])

    mins = ranges[:, 0]
    maxs = ranges[:, 1]

    normalized = (raw_features - mins) / (maxs - mins + 1e-8)
    return np.clip(normalized, 0.0, 1.0)


# ============================================================================
# Synthetic Data Generation
# ============================================================================

def generate_synthetic_data(n_samples: int = 10000, seed: int = 42) -> tuple:
    """
    Generate synthetic training data for skill prediction.

    Creates realistic player performance distributions based on
    heuristic skill levels.

    Returns:
        (features, labels) tuple
    """
    np.random.seed(seed)

    features = []
    labels = []

    for _ in range(n_samples):
        # Generate a "true" skill level
        true_skill = np.random.beta(2, 2)  # Centered around 0.5

        # Generate features correlated with skill
        # Skilled players: faster, fewer errors, fewer hints, longer streaks

        # Time ratio: skilled = 0.5-1.0, unskilled = 1.5-3.0
        time_ratio = np.random.normal(
            loc=2.0 - true_skill * 1.5,
            scale=0.3
        )
        time_ratio = np.clip(time_ratio, 0.3, 3.0)

        # Error rate: skilled = 0.0-0.1, unskilled = 0.2-0.4
        error_rate = np.random.normal(
            loc=0.3 * (1 - true_skill),
            scale=0.05
        )
        error_rate = np.clip(error_rate, 0.0, 0.5)

        # Hint rate: skilled rarely use hints
        hint_rate = np.random.exponential(0.1 + 0.3 * (1 - true_skill))
        hint_rate = np.clip(hint_rate, 0.0, 1.0)

        # Undo rate: skilled make fewer undos
        undo_rate = np.random.normal(
            loc=0.2 * (1 - true_skill),
            scale=0.05
        )
        undo_rate = np.clip(undo_rate, 0.0, 0.5)

        # Streak: skilled have longer streaks
        streak = np.random.poisson(lam=3 + 10 * true_skill)
        streak = min(streak, 20)

        # Puzzle size: advanced players attempt larger grids
        preferred_size = 4 + int(true_skill * 8)  # 4-12 based on skill
        puzzle_size = np.random.choice(
            range(3, 17),
            p=_size_distribution(preferred_size)
        )

        # Difficulty: skilled attempt harder puzzles
        preferred_diff = int(true_skill * 4)
        difficulty = np.random.choice(
            [0, 1, 2, 3, 4],
            p=_diff_distribution(preferred_diff)
        )

        # Session puzzles: engaged players play longer
        session_puzzles = np.random.poisson(lam=10 + 20 * true_skill)
        session_puzzles = min(session_puzzles, 50)

        raw_feature = np.array([
            time_ratio,
            error_rate,
            hint_rate,
            undo_rate,
            streak,
            puzzle_size,
            difficulty,
            session_puzzles,
        ], dtype=np.float32)

        features.append(raw_feature)
        labels.append(true_skill)

    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)

    # Normalize features
    features = normalize_features(features)

    return features, labels


def _size_distribution(preferred: int) -> list:
    """Create a probability distribution centered on preferred size."""
    sizes = list(range(3, 17))
    probs = []
    for s in sizes:
        dist = abs(s - preferred)
        prob = np.exp(-0.5 * dist)  # Gaussian-like decay
        probs.append(prob)
    total = sum(probs)
    return [p / total for p in probs]


def _diff_distribution(preferred: int) -> list:
    """Create a probability distribution centered on preferred difficulty."""
    probs = []
    for d in range(5):
        dist = abs(d - preferred)
        prob = np.exp(-0.8 * dist)
        probs.append(prob)
    total = sum(probs)
    return [p / total for p in probs]


# ============================================================================
# Model Definition (PyTorch)
# ============================================================================

# Model class defined dynamically after torch import
_SkillModelClass = None


def _get_skill_model_class():
    """Get or create the SkillModel class (requires torch)."""
    global _SkillModelClass
    if _SkillModelClass is not None:
        return _SkillModelClass

    lazy_import_torch()

    class SkillModel(torch.nn.Module):
        """
        3-layer MLP skill model.

        Architecture optimized for:
        - Small size (~50KB ONNX)
        - Fast inference (<1ms on mobile)
        - Stable predictions (BatchNorm + Dropout)
        """

        def __init__(self):
            super().__init__()

            self.layers = torch.nn.Sequential(
                # Hidden 1
                torch.nn.Linear(NUM_FEATURES, 32),
                torch.nn.BatchNorm1d(32),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                # Hidden 2
                torch.nn.Linear(32, 16),
                torch.nn.BatchNorm1d(16),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                # Output
                torch.nn.Linear(16, 1),
                torch.nn.Sigmoid(),
            )

        def forward(self, x):
            return self.layers(x).squeeze(-1)

    _SkillModelClass = SkillModel
    return _SkillModelClass


def create_skill_model():
    """Create the skill model instance."""
    SkillModel = _get_skill_model_class()
    return SkillModel()


# ============================================================================
# Training (PyTorch)
# ============================================================================

def train_model(
    model: "SkillModel",
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 64
) -> dict:
    """Train the model with early stopping."""
    lazy_import_torch()

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    # Dataset and loader
    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = torch.nn.MSELoss()

    # Training loop with early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    history = {"train_loss": [], "val_loss": [], "val_mae": []}

    for epoch in range(epochs):
        # Train phase
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X_batch)
        train_loss /= len(X_train_t)

        # Validation phase (inference mode)
        model.train(False)
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()
            val_mae = torch.abs(val_pred - y_val_t).mean().item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, "
              f"val_loss={val_loss:.4f}, val_mae={val_mae:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return history


# ============================================================================
# ONNX Export
# ============================================================================

def export_onnx(model: "SkillModel", output_path: str) -> int:
    """
    Export model to ONNX format.

    Returns:
        Size of the exported model in bytes
    """
    lazy_import_torch()

    # Set to inference mode
    model.train(False)

    # Create dummy input for tracing
    dummy_input = torch.randn(1, NUM_FEATURES)

    # Export to ONNX
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["features"],
        output_names=["skill_score"],
        dynamic_axes={
            "features": {0: "batch_size"},
            "skill_score": {0: "batch_size"},
        },
        opset_version=13,
    )

    # Get file size
    return Path(output_path).stat().st_size


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train skill prediction model for adaptive difficulty"
    )
    parser.add_argument(
        "--samples", type=int, default=20000,
        help="Number of synthetic training samples"
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Maximum training epochs"
    )
    parser.add_argument(
        "--output", type=str, default="models/skill_model.onnx",
        help="Output ONNX model path"
    )
    parser.add_argument(
        "--data-only", action="store_true",
        help="Only generate training data (no PyTorch required)"
    )
    args = parser.parse_args()

    print("=== Skill Model Training ===")
    print(f"Features: {FEATURE_NAMES}")
    print()

    # Generate data
    print(f"Generating {args.samples} synthetic samples...")
    X, y = generate_synthetic_data(args.samples)

    # Split train/val
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    print(f"Feature shape: {X_train.shape}")
    print(f"Label range: [{y.min():.3f}, {y.max():.3f}]")
    print()

    if args.data_only:
        # Save data for external training
        Path("data").mkdir(exist_ok=True)
        np.savez(
            "data/skill_training_data.npz",
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            feature_names=FEATURE_NAMES
        )
        print("Data saved to data/skill_training_data.npz")
        return

    # Import PyTorch and train
    lazy_import_torch()

    print("Creating model...")
    model = create_skill_model()
    print(model)
    print()

    print("Training...")
    history = train_model(model, X_train, y_train, X_val, y_val, epochs=args.epochs)

    final_loss = history["val_loss"][-1]
    final_mae = history["val_mae"][-1]
    print(f"\nFinal validation loss: {final_loss:.4f}")
    print(f"Final validation MAE: {final_mae:.4f}")
    print()

    # Export to ONNX
    print(f"Exporting to {args.output}...")
    size = export_onnx(model, args.output)
    print(f"Model size: {size / 1024:.1f} KB")

    # Also save PyTorch checkpoint for debugging
    pt_path = args.output.replace(".onnx", ".pt")
    torch.save(model.state_dict(), pt_path)
    print(f"PyTorch checkpoint saved to {pt_path}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
