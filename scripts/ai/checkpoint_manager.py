"""
Checkpoint Manager for KeenKenning ML Training Pipeline

Provides complete checkpoint saving/loading with full state serialization
for crash-resilient training resumption.

SPDX-License-Identifier: MIT
SPDX-FileCopyrightText: Copyright (C) 2025 KeenKenning Contributors
"""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import numpy as np
import torch

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Dataclasses (P0.1.3 - P0.1.5)
# =============================================================================


@dataclass(frozen=True)
class ModelConfig:
    """Immutable model architecture configuration.

    Frozen to prevent accidental modification during training.
    All parameters required to reconstruct the model architecture.
    """

    vocab_size: int = 35  # PAD + EMPTY + 33 values (VAL_-16 to VAL_16)
    d_model: int = 256
    n_head: int = 8
    n_layer: int = 8
    d_ff: int = 1024  # Feed-forward dimension
    max_size: int = 16
    num_modes: int = 11  # Number of game modes
    dropout: float = 0.1
    use_checkpoint: bool = False  # Gradient checkpointing (memory/speed tradeoff)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        """Reconstruct from dictionary."""
        return cls(**d)


@dataclass(frozen=True)
class OptimizerConfig:
    """Immutable optimizer configuration."""

    name: str = "AdamW"
    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.95)  # nanoGPT-style beta2
    eps: float = 1e-8
    use_8bit: bool = False  # bitsandbytes 8-bit AdamW
    dtype: str = "fp16"  # fp32, fp16, or bf16 (for AMP autocast)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        d["betas"] = list(d["betas"])  # Tuple -> List for JSON
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OptimizerConfig":
        """Reconstruct from dictionary."""
        d = d.copy()
        d["betas"] = tuple(d["betas"])  # List -> Tuple
        return cls(**d)


@dataclass(frozen=True)
class SchedulerConfig:
    """Immutable learning rate scheduler configuration."""

    name: str = "CosineAnnealingWarmRestarts"
    warmup_epochs: int = 5
    min_lr: float = 1e-5
    t_0: int = 10  # Restart period for CosineAnnealingWarmRestarts
    t_mult: int = 2  # Period multiplier after each restart

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SchedulerConfig":
        """Reconstruct from dictionary."""
        return cls(**d)


# =============================================================================
# Training State Dataclass (P0.1.2)
# =============================================================================


@dataclass
class TrainingState:
    """Mutable training runtime state.

    Tracks all transient state that changes during training.
    This is NOT frozen because it's updated every step/epoch.
    """

    epoch: int = 0
    global_step: int = 0
    best_loss: float = float("inf")
    best_epoch: int = 0
    curriculum_stage: int = 0  # Legacy: size stage only (for backward compat)
    loss_history: List[float] = field(default_factory=list)
    valid_rate_history: List[float] = field(default_factory=list)

    # Multi-dimensional curriculum state (P5: size, mode, fill stages)
    curriculum_scheduler_state: Optional[Dict[str, Any]] = None

    # Training timestamps
    started_at: Optional[str] = None
    last_saved_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingState":
        """Reconstruct from dictionary."""
        # Handle legacy checkpoints without curriculum_scheduler_state
        if 'curriculum_scheduler_state' not in d:
            d['curriculum_scheduler_state'] = None
        return cls(**d)

    def update_on_epoch_end(self, epoch: int, loss: float, valid_rate: float) -> None:
        """Update state at end of epoch."""
        self.epoch = epoch
        self.loss_history.append(loss)
        self.valid_rate_history.append(valid_rate)
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_epoch = epoch
        self.last_saved_at = datetime.now(timezone.utc).isoformat()


# =============================================================================
# RNG State Management (P0.2.3 - P0.2.4)
# =============================================================================


class RNGStates(TypedDict):
    """Type definition for RNG states dictionary."""

    python: tuple
    numpy: Dict[str, Any]
    torch_cpu: torch.Tensor
    torch_cuda: Optional[List[torch.Tensor]]


def save_rng_states() -> RNGStates:
    """Capture all RNG states for reproducibility.

    Returns:
        Dictionary containing Python, NumPy, and PyTorch RNG states.
    """
    states: RNGStates = {
        "python": random.getstate(),
        "numpy": {
            "state": np.random.get_state()[1].tolist(),
            "pos": int(np.random.get_state()[2]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
    }

    if torch.cuda.is_available():
        states["torch_cuda"] = [
            torch.cuda.get_rng_state(device=i)
            for i in range(torch.cuda.device_count())
        ]

    return states


def restore_rng_states(states: RNGStates) -> None:
    """Restore all RNG states from saved snapshot.

    Args:
        states: Dictionary containing RNG states from save_rng_states().
    """
    random.setstate(states["python"])

    # Reconstruct NumPy state tuple
    np_state = (
        "MT19937",
        np.array(states["numpy"]["state"], dtype=np.uint32),
        states["numpy"]["pos"],
        0,
        0.0,
    )
    np.random.set_state(np_state)

    torch.set_rng_state(states["torch_cpu"])

    if torch.cuda.is_available() and states["torch_cuda"] is not None:
        for i, state in enumerate(states["torch_cuda"]):
            if i < torch.cuda.device_count():
                torch.cuda.set_rng_state(state, device=i)


# =============================================================================
# Checkpoint V2 Schema (P0.1.1)
# =============================================================================


@dataclass
class CheckpointV2:
    """Complete checkpoint schema for crash-resilient training.

    Contains all state required to resume training identically:
    - Model weights
    - Optimizer state
    - LR scheduler state
    - Training progress
    - RNG states for reproducibility
    - Configuration for validation
    """

    # Version for future compatibility
    version: str = "2.0.0"

    # Configurations (frozen, for validation on load)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)

    # Training state (mutable progress)
    training_state: TrainingState = field(default_factory=TrainingState)

    # PyTorch state dicts (populated during save)
    model_state_dict: Optional[Dict[str, Any]] = None
    optimizer_state_dict: Optional[Dict[str, Any]] = None
    scheduler_state_dict: Optional[Dict[str, Any]] = None

    # GradScaler state (for AMP, None if using BF16)
    scaler_state_dict: Optional[Dict[str, Any]] = None

    # RNG states for exact reproducibility
    rng_states: Optional[RNGStates] = None

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    pytorch_version: str = field(default_factory=lambda: torch.__version__)
    cuda_version: Optional[str] = field(
        default_factory=lambda: torch.version.cuda if torch.cuda.is_available() else None
    )

    def to_save_dict(self) -> Dict[str, Any]:
        """Convert to dictionary suitable for torch.save().

        Handles nested dataclasses and special types.
        """
        return {
            "version": self.version,
            "model_config": self.model_config.to_dict(),
            "optimizer_config": self.optimizer_config.to_dict(),
            "scheduler_config": self.scheduler_config.to_dict(),
            "training_state": self.training_state.to_dict(),
            "model_state_dict": self.model_state_dict,
            "optimizer_state_dict": self.optimizer_state_dict,
            "scheduler_state_dict": self.scheduler_state_dict,
            "scaler_state_dict": self.scaler_state_dict,
            "rng_states": self.rng_states,
            "created_at": self.created_at,
            "pytorch_version": self.pytorch_version,
            "cuda_version": self.cuda_version,
        }

    @classmethod
    def from_load_dict(cls, d: Dict[str, Any]) -> "CheckpointV2":
        """Reconstruct from dictionary loaded via torch.load()."""
        return cls(
            version=d.get("version", "1.0.0"),
            model_config=ModelConfig.from_dict(d.get("model_config", {})),
            optimizer_config=OptimizerConfig.from_dict(d.get("optimizer_config", {})),
            scheduler_config=SchedulerConfig.from_dict(d.get("scheduler_config", {})),
            training_state=TrainingState.from_dict(d.get("training_state", {})),
            model_state_dict=d.get("model_state_dict"),
            optimizer_state_dict=d.get("optimizer_state_dict"),
            scheduler_state_dict=d.get("scheduler_state_dict"),
            scaler_state_dict=d.get("scaler_state_dict"),
            rng_states=d.get("rng_states"),
            created_at=d.get("created_at", ""),
            pytorch_version=d.get("pytorch_version", ""),
            cuda_version=d.get("cuda_version"),
        )


# =============================================================================
# Save/Load Functions (P0.2.1 - P0.2.2)
# =============================================================================


def save_checkpoint(
    checkpoint: CheckpointV2,
    path: Path,
    atomic: bool = True,
) -> Path:
    """Save checkpoint to disk with optional atomic write.

    Args:
        checkpoint: CheckpointV2 instance with all state populated.
        path: Destination path for checkpoint file.
        atomic: If True, write to temp file then rename (crash-safe).

    Returns:
        Path where checkpoint was saved.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = checkpoint.to_save_dict()

    if atomic:
        # Write to temp file, then atomic rename
        temp_path = path.with_suffix(".tmp")
        torch.save(save_dict, temp_path)
        shutil.move(str(temp_path), str(path))
    else:
        torch.save(save_dict, path)

    logger.info(f"Saved checkpoint to {path} (epoch {checkpoint.training_state.epoch})")
    return path


def load_checkpoint(
    path: Path,
    map_location: Optional[str] = None,
) -> CheckpointV2:
    """Load checkpoint from disk.

    Args:
        path: Path to checkpoint file.
        map_location: Device mapping (e.g., 'cpu', 'cuda:0').

    Returns:
        CheckpointV2 instance with all state restored.

    Raises:
        FileNotFoundError: If checkpoint doesn't exist.
        RuntimeError: If checkpoint is corrupted or incompatible.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        save_dict = torch.load(path, map_location=map_location, weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint {path}: {e}")

    checkpoint = CheckpointV2.from_load_dict(save_dict)

    logger.info(
        f"Loaded checkpoint from {path} "
        f"(epoch {checkpoint.training_state.epoch}, "
        f"version {checkpoint.version})"
    )
    return checkpoint


def load_checkpoint_legacy(
    path: Path,
    map_location: Optional[str] = None,
) -> Dict[str, Any]:
    """Load legacy v1 checkpoint (model weights only).

    For backward compatibility with existing checkpoints.

    Args:
        path: Path to legacy checkpoint file.
        map_location: Device mapping.

    Returns:
        Dictionary with 'model_state_dict' key, or raw state dict.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    save_dict = torch.load(path, map_location=map_location, weights_only=False)

    # Check if it's already a state dict (old format)
    if "model_state_dict" not in save_dict and "version" not in save_dict:
        # Raw state dict from old training script
        return {"model_state_dict": save_dict}

    return save_dict


# =============================================================================
# Rolling Checkpoint Manager (P0.3.1 - P0.3.4)
# =============================================================================


class RollingCheckpointManager:
    """Manages checkpoint rotation with configurable retention.

    Keeps last N checkpoints and separately tracks the best checkpoint.
    Automatically cleans up old checkpoints on save.
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        prefix: str = "checkpoint",
        keep_last_n: int = 3,
        keep_best: bool = True,
    ):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints.
            prefix: Filename prefix for checkpoints.
            keep_last_n: Number of recent checkpoints to keep.
            keep_best: Whether to separately track best checkpoint.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.prefix = prefix
        self.keep_last_n = keep_last_n
        self.keep_best = keep_best

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Track checkpoint history
        self._rolling_checkpoints: List[Path] = []
        self._best_checkpoint: Optional[Path] = None
        self._best_loss: float = float("inf")

        # Discover existing checkpoints
        self._discover_existing()

    def _discover_existing(self) -> None:
        """Discover existing checkpoints in directory."""
        pattern = f"{self.prefix}_epoch_*.pt"
        existing = sorted(
            self.checkpoint_dir.glob(pattern),
            key=lambda p: p.stat().st_mtime,
        )
        self._rolling_checkpoints = existing[-self.keep_last_n :]

        best_path = self.checkpoint_dir / f"{self.prefix}_best.pt"
        if best_path.exists():
            self._best_checkpoint = best_path

    def save(
        self,
        checkpoint: CheckpointV2,
        is_best: bool = False,
    ) -> Path:
        """Save checkpoint with automatic rotation.

        Args:
            checkpoint: CheckpointV2 to save.
            is_best: Whether this is the best checkpoint so far.

        Returns:
            Path to saved checkpoint.
        """
        epoch = checkpoint.training_state.epoch

        # Save epoch checkpoint
        epoch_path = self.checkpoint_dir / f"{self.prefix}_epoch_{epoch:04d}.pt"
        save_checkpoint(checkpoint, epoch_path)
        self._rolling_checkpoints.append(epoch_path)

        # Cleanup old checkpoints
        while len(self._rolling_checkpoints) > self.keep_last_n:
            old_path = self._rolling_checkpoints.pop(0)
            if old_path.exists() and old_path != self._best_checkpoint:
                old_path.unlink()
                logger.debug(f"Removed old checkpoint: {old_path}")

        # Save best checkpoint separately
        if self.keep_best and is_best:
            best_path = self.checkpoint_dir / f"{self.prefix}_best.pt"
            save_checkpoint(checkpoint, best_path)
            self._best_checkpoint = best_path
            self._best_loss = checkpoint.training_state.best_loss
            logger.info(f"New best checkpoint (loss={self._best_loss:.6f})")

        return epoch_path

    def load_latest(self, map_location: Optional[str] = None) -> Optional[CheckpointV2]:
        """Load most recent checkpoint.

        Returns:
            CheckpointV2 or None if no checkpoints exist.
        """
        if not self._rolling_checkpoints:
            return None
        return load_checkpoint(self._rolling_checkpoints[-1], map_location)

    def load_best(self, map_location: Optional[str] = None) -> Optional[CheckpointV2]:
        """Load best checkpoint.

        Returns:
            CheckpointV2 or None if no best checkpoint exists.
        """
        if self._best_checkpoint is None or not self._best_checkpoint.exists():
            return None
        return load_checkpoint(self._best_checkpoint, map_location)

    def get_latest_path(self) -> Optional[Path]:
        """Get path to latest checkpoint."""
        return self._rolling_checkpoints[-1] if self._rolling_checkpoints else None

    def get_best_path(self) -> Optional[Path]:
        """Get path to best checkpoint."""
        return self._best_checkpoint


# =============================================================================
# Convenience Functions for Training Integration
# =============================================================================


def create_checkpoint_from_training(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,  # LR scheduler
    training_state: TrainingState,
    model_config: ModelConfig,
    optimizer_config: OptimizerConfig,
    scheduler_config: SchedulerConfig,
    scaler: Optional[Any] = None,  # GradScaler for FP16
    include_rng: bool = True,
) -> CheckpointV2:
    """Create a complete checkpoint from current training state.

    Convenience function to gather all state into CheckpointV2.

    Args:
        model: The model being trained.
        optimizer: The optimizer.
        scheduler: Learning rate scheduler.
        training_state: Current training progress.
        model_config: Model architecture config.
        optimizer_config: Optimizer config.
        scheduler_config: Scheduler config.
        scaler: Optional GradScaler for FP16 training.
        include_rng: Whether to capture RNG states.

    Returns:
        Complete CheckpointV2 ready for saving.
    """
    checkpoint = CheckpointV2(
        model_config=model_config,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
        training_state=training_state,
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        scheduler_state_dict=scheduler.state_dict() if scheduler else None,
        scaler_state_dict=scaler.state_dict() if scaler else None,
        rng_states=save_rng_states() if include_rng else None,
    )
    return checkpoint


def restore_training_from_checkpoint(
    checkpoint: CheckpointV2,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Optional[Any] = None,
    restore_rng: bool = True,
) -> TrainingState:
    """Restore training state from checkpoint.

    Loads all state dicts into provided objects.

    Args:
        checkpoint: Loaded CheckpointV2.
        model: Model to restore weights into.
        optimizer: Optimizer to restore state into.
        scheduler: Scheduler to restore state into.
        scaler: Optional GradScaler to restore.
        restore_rng: Whether to restore RNG states.

    Returns:
        TrainingState from checkpoint.
    """
    # Load model weights
    if checkpoint.model_state_dict:
        model.load_state_dict(checkpoint.model_state_dict)
        logger.info("Restored model weights")

    # Load optimizer state
    if checkpoint.optimizer_state_dict:
        optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        logger.info("Restored optimizer state")

    # Load scheduler state
    if scheduler and checkpoint.scheduler_state_dict:
        scheduler.load_state_dict(checkpoint.scheduler_state_dict)
        logger.info("Restored scheduler state")

    # Load scaler state (FP16)
    if scaler and checkpoint.scaler_state_dict:
        scaler.load_state_dict(checkpoint.scaler_state_dict)
        logger.info("Restored GradScaler state")

    # Restore RNG states
    if restore_rng and checkpoint.rng_states:
        restore_rng_states(checkpoint.rng_states)
        logger.info("Restored RNG states")

    return checkpoint.training_state


if __name__ == "__main__":
    # Quick sanity check
    logging.basicConfig(level=logging.INFO)

    print("=== CheckpointV2 Sanity Check ===")

    # Create configs
    mc = ModelConfig(d_model=128, n_layer=4)
    oc = OptimizerConfig(lr=1e-4)
    sc = SchedulerConfig(warmup_epochs=3)

    print(f"ModelConfig: {mc}")
    print(f"OptimizerConfig: {oc}")
    print(f"SchedulerConfig: {sc}")

    # Create training state
    ts = TrainingState()
    ts.update_on_epoch_end(0, 2.5, 0.75)
    ts.update_on_epoch_end(1, 1.8, 0.82)
    print(f"TrainingState: {ts}")

    # Create checkpoint
    ckpt = CheckpointV2(
        model_config=mc,
        optimizer_config=oc,
        scheduler_config=sc,
        training_state=ts,
    )

    # Test serialization round-trip
    save_dict = ckpt.to_save_dict()
    restored = CheckpointV2.from_load_dict(save_dict)

    assert restored.model_config.d_model == 128
    assert restored.training_state.epoch == 1
    assert len(restored.training_state.loss_history) == 2

    print("\nRound-trip serialization: PASSED")
    print(f"Checkpoint version: {restored.version}")
    print(f"PyTorch version: {restored.pytorch_version}")
