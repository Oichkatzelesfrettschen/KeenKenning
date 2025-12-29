"""
Unit and Integration Tests for Checkpoint Manager

Tests P0.4.1-P0.4.3:
- Save/load round-trip
- RNG state restoration
- Crash-resume simulation

Run with: pytest test_checkpoint_manager.py -v

SPDX-License-Identifier: MIT
SPDX-FileCopyrightText: Copyright (C) 2025 KeenKenning Contributors
"""

import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from checkpoint_manager import (
    CheckpointV2,
    ModelConfig,
    OptimizerConfig,
    RollingCheckpointManager,
    SchedulerConfig,
    TrainingState,
    create_checkpoint_from_training,
    load_checkpoint,
    restore_rng_states,
    restore_training_from_checkpoint,
    save_checkpoint,
    save_rng_states,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test checkpoints."""
    path = Path(tempfile.mkdtemp())
    yield path
    shutil.rmtree(path)


@pytest.fixture
def sample_model():
    """Create a simple test model."""

    class SimpleModel(nn.Module):
        def __init__(self, d_model=64, n_layer=2):
            super().__init__()
            self.embed = nn.Embedding(35, d_model)
            self.layers = nn.ModuleList(
                [nn.Linear(d_model, d_model) for _ in range(n_layer)]
            )
            self.out = nn.Linear(d_model, 35)

        def forward(self, x):
            h = self.embed(x)
            for layer in self.layers:
                h = torch.relu(layer(h))
            return self.out(h)

    return SimpleModel()


@pytest.fixture
def sample_configs():
    """Create sample configuration objects."""
    return {
        "model": ModelConfig(vocab_size=35, d_model=64, n_layer=2),
        "optimizer": OptimizerConfig(lr=1e-3),
        "scheduler": SchedulerConfig(warmup_epochs=2),
    }


# =============================================================================
# P0.4.1: Save/Load Round-Trip Tests
# =============================================================================


class TestSaveLoadRoundTrip:
    """Test checkpoint save/load preserves all state."""

    def test_config_roundtrip(self, temp_dir):
        """Test configuration dataclasses serialize correctly."""
        mc = ModelConfig(vocab_size=35, d_model=128, n_layer=4, dropout=0.2)
        oc = OptimizerConfig(lr=1e-4, weight_decay=0.05, betas=(0.9, 0.99))
        sc = SchedulerConfig(warmup_epochs=10, min_lr=1e-6)

        checkpoint = CheckpointV2(
            model_config=mc,
            optimizer_config=oc,
            scheduler_config=sc,
        )

        path = temp_dir / "config_test.pt"
        save_checkpoint(checkpoint, path)
        loaded = load_checkpoint(path)

        assert loaded.model_config.vocab_size == 35
        assert loaded.model_config.d_model == 128
        assert loaded.model_config.dropout == 0.2
        assert loaded.optimizer_config.lr == 1e-4
        assert loaded.optimizer_config.betas == (0.9, 0.99)
        assert loaded.scheduler_config.warmup_epochs == 10

    def test_training_state_roundtrip(self, temp_dir):
        """Test training state serializes correctly."""
        ts = TrainingState(
            epoch=42,
            global_step=10000,
            best_loss=0.123,
            best_epoch=35,
            curriculum_stage=2,
            loss_history=[2.0, 1.5, 1.0, 0.5, 0.123],
            valid_rate_history=[0.5, 0.7, 0.85, 0.92, 0.95],
        )

        checkpoint = CheckpointV2(training_state=ts)
        path = temp_dir / "state_test.pt"
        save_checkpoint(checkpoint, path)
        loaded = load_checkpoint(path)

        assert loaded.training_state.epoch == 42
        assert loaded.training_state.global_step == 10000
        assert loaded.training_state.best_loss == pytest.approx(0.123)
        assert loaded.training_state.curriculum_stage == 2
        assert len(loaded.training_state.loss_history) == 5
        assert loaded.training_state.loss_history[-1] == pytest.approx(0.123)

    def test_model_state_dict_roundtrip(self, temp_dir, sample_model):
        """Test model weights serialize correctly."""
        # Set specific weights for verification
        with torch.no_grad():
            sample_model.embed.weight[0, 0] = 42.0

        checkpoint = CheckpointV2(model_state_dict=sample_model.state_dict())
        path = temp_dir / "model_test.pt"
        save_checkpoint(checkpoint, path)
        loaded = load_checkpoint(path)

        assert loaded.model_state_dict is not None
        assert loaded.model_state_dict["embed.weight"][0, 0] == pytest.approx(42.0)

    def test_optimizer_state_dict_roundtrip(self, temp_dir, sample_model):
        """Test optimizer state serializes correctly."""
        optimizer = torch.optim.AdamW(sample_model.parameters(), lr=1e-3)

        # Run a step to populate optimizer state
        x = torch.randint(0, 35, (2, 4))
        loss = sample_model(x).sum()
        loss.backward()
        optimizer.step()

        checkpoint = CheckpointV2(
            model_state_dict=sample_model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
        )
        path = temp_dir / "optim_test.pt"
        save_checkpoint(checkpoint, path)
        loaded = load_checkpoint(path)

        assert loaded.optimizer_state_dict is not None
        assert "state" in loaded.optimizer_state_dict
        assert len(loaded.optimizer_state_dict["state"]) > 0

    def test_full_checkpoint_roundtrip(self, temp_dir, sample_model, sample_configs):
        """Test complete checkpoint with all components."""
        optimizer = torch.optim.AdamW(sample_model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        # Train a step
        x = torch.randint(0, 35, (2, 4))
        loss = sample_model(x).sum()
        loss.backward()
        optimizer.step()
        scheduler.step()

        training_state = TrainingState(epoch=5, global_step=1000)
        training_state.update_on_epoch_end(5, 0.5, 0.9)

        checkpoint = create_checkpoint_from_training(
            model=sample_model,
            optimizer=optimizer,
            scheduler=scheduler,
            training_state=training_state,
            model_config=sample_configs["model"],
            optimizer_config=sample_configs["optimizer"],
            scheduler_config=sample_configs["scheduler"],
        )

        path = temp_dir / "full_test.pt"
        save_checkpoint(checkpoint, path)
        loaded = load_checkpoint(path)

        assert loaded.version == "2.0.0"
        assert loaded.model_state_dict is not None
        assert loaded.optimizer_state_dict is not None
        assert loaded.scheduler_state_dict is not None
        assert loaded.training_state.epoch == 5
        assert loaded.rng_states is not None

    def test_atomic_save(self, temp_dir):
        """Test atomic save doesn't leave temp files on success."""
        checkpoint = CheckpointV2()
        path = temp_dir / "atomic_test.pt"
        temp_path = path.with_suffix(".tmp")

        save_checkpoint(checkpoint, path, atomic=True)

        assert path.exists()
        assert not temp_path.exists()


# =============================================================================
# P0.4.2: RNG State Restoration Tests
# =============================================================================


class TestRNGStateRestoration:
    """Test RNG state capture and restoration for reproducibility."""

    def test_python_random_restoration(self):
        """Test Python random module state restoration."""
        # Capture state
        states = save_rng_states()

        # Generate some random numbers
        original = [random.random() for _ in range(10)]

        # Restore state
        restore_rng_states(states)

        # Generate again - should match
        restored = [random.random() for _ in range(10)]

        assert original == restored

    def test_numpy_random_restoration(self):
        """Test NumPy random state restoration."""
        states = save_rng_states()

        original = np.random.randn(100)

        restore_rng_states(states)

        restored = np.random.randn(100)

        np.testing.assert_array_equal(original, restored)

    def test_torch_cpu_restoration(self):
        """Test PyTorch CPU RNG state restoration."""
        states = save_rng_states()

        original = torch.randn(100)

        restore_rng_states(states)

        restored = torch.randn(100)

        torch.testing.assert_close(original, restored)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_torch_cuda_restoration(self):
        """Test PyTorch CUDA RNG state restoration."""
        states = save_rng_states()

        original = torch.randn(100, device="cuda")

        restore_rng_states(states)

        restored = torch.randn(100, device="cuda")

        torch.testing.assert_close(original, restored)

    def test_rng_roundtrip_through_checkpoint(self, temp_dir):
        """Test RNG states survive checkpoint save/load cycle."""
        # Create checkpoint with RNG states
        states = save_rng_states()
        checkpoint = CheckpointV2(rng_states=states)

        path = temp_dir / "rng_test.pt"
        save_checkpoint(checkpoint, path)
        loaded = load_checkpoint(path)

        # Generate numbers with original state
        original_py = [random.random() for _ in range(5)]
        original_np = np.random.randn(5).tolist()
        original_torch = torch.randn(5).tolist()

        # Restore from loaded checkpoint
        restore_rng_states(loaded.rng_states)

        # Generate again - should match
        restored_py = [random.random() for _ in range(5)]
        restored_np = np.random.randn(5).tolist()
        restored_torch = torch.randn(5).tolist()

        assert original_py == restored_py
        assert original_np == restored_np
        assert original_torch == restored_torch


# =============================================================================
# P0.4.3: Crash-Resume Integration Tests
# =============================================================================


class TestCrashResumeSimulation:
    """Simulate training crash and resume scenarios."""

    def test_resume_continues_training_correctly(self, temp_dir, sample_configs):
        """Test that resumed training produces same results as uninterrupted."""

        class DeterministicModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)

            def forward(self, x):
                return self.linear(x)

        # Set seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        # === First run: Train for 3 epochs, save at epoch 2 ===
        model1 = DeterministicModel()
        optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.01)
        scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=1)

        training_state1 = TrainingState()

        losses1 = []
        for epoch in range(3):
            x = torch.randn(4, 10)
            loss = model1(x).sum()
            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()
            scheduler1.step()
            losses1.append(loss.item())

            if epoch == 1:  # Save after epoch 2
                training_state1.update_on_epoch_end(epoch, loss.item(), 0.8)
                checkpoint = create_checkpoint_from_training(
                    model=model1,
                    optimizer=optimizer1,
                    scheduler=scheduler1,
                    training_state=training_state1,
                    model_config=sample_configs["model"],
                    optimizer_config=sample_configs["optimizer"],
                    scheduler_config=sample_configs["scheduler"],
                )
                save_checkpoint(checkpoint, temp_dir / "checkpoint.pt")

        # === Second run: Resume from epoch 2, train epoch 3 ===
        model2 = DeterministicModel()
        optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)
        scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=1)

        loaded = load_checkpoint(temp_dir / "checkpoint.pt")
        training_state2 = restore_training_from_checkpoint(
            loaded, model2, optimizer2, scheduler2
        )

        # Continue training from epoch 2
        start_epoch = training_state2.epoch + 1
        losses2 = []
        for epoch in range(start_epoch, 3):
            x = torch.randn(4, 10)
            loss = model2(x).sum()
            optimizer2.zero_grad()
            loss.backward()
            optimizer2.step()
            scheduler2.step()
            losses2.append(loss.item())

        # The third epoch loss should be exactly the same
        # (because RNG state was restored)
        assert losses1[2] == pytest.approx(losses2[0], rel=1e-5)

    def test_rolling_manager_keeps_n_checkpoints(self, temp_dir):
        """Test RollingCheckpointManager keeps only last N checkpoints."""
        manager = RollingCheckpointManager(
            checkpoint_dir=temp_dir,
            prefix="test",
            keep_last_n=3,
        )

        # Save 5 checkpoints
        for epoch in range(5):
            ts = TrainingState(epoch=epoch)
            checkpoint = CheckpointV2(training_state=ts)
            manager.save(checkpoint)

        # Should only have 3 checkpoints
        checkpoints = list(temp_dir.glob("test_epoch_*.pt"))
        assert len(checkpoints) == 3

        # Should be epochs 2, 3, 4
        epochs = sorted([int(p.stem.split("_")[-1]) for p in checkpoints])
        assert epochs == [2, 3, 4]

    def test_rolling_manager_preserves_best(self, temp_dir):
        """Test best checkpoint is preserved separately from rolling."""
        manager = RollingCheckpointManager(
            checkpoint_dir=temp_dir,
            prefix="test",
            keep_last_n=2,
            keep_best=True,
        )

        # Save 5 checkpoints, best is epoch 1
        for epoch in range(5):
            ts = TrainingState(epoch=epoch, best_loss=float(epoch + 1))
            if epoch == 1:
                ts.best_loss = 0.1  # Best loss
            checkpoint = CheckpointV2(training_state=ts)
            manager.save(checkpoint, is_best=(epoch == 1))

        # Should have 2 rolling + 1 best
        rolling = list(temp_dir.glob("test_epoch_*.pt"))
        best = temp_dir / "test_best.pt"

        assert len(rolling) == 2
        assert best.exists()

        # Best should be epoch 1
        loaded_best = manager.load_best()
        assert loaded_best.training_state.epoch == 1

    def test_resume_from_latest(self, temp_dir):
        """Test resuming from latest checkpoint."""
        manager = RollingCheckpointManager(checkpoint_dir=temp_dir, keep_last_n=3)

        # Save some checkpoints
        for epoch in range(5):
            ts = TrainingState(epoch=epoch, global_step=epoch * 100)
            checkpoint = CheckpointV2(training_state=ts)
            manager.save(checkpoint)

        # Load latest
        latest = manager.load_latest()
        assert latest.training_state.epoch == 4
        assert latest.training_state.global_step == 400


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Test error handling and edge cases."""

    def test_load_nonexistent_checkpoint(self, temp_dir):
        """Test loading a checkpoint that doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_checkpoint(temp_dir / "nonexistent.pt")

    def test_empty_rolling_manager(self, temp_dir):
        """Test RollingCheckpointManager with no existing checkpoints."""
        manager = RollingCheckpointManager(checkpoint_dir=temp_dir)

        assert manager.load_latest() is None
        assert manager.load_best() is None
        assert manager.get_latest_path() is None

    def test_training_state_update(self):
        """Test TrainingState.update_on_epoch_end() logic."""
        ts = TrainingState()

        # First epoch with loss 2.0
        ts.update_on_epoch_end(0, 2.0, 0.5)
        assert ts.epoch == 0
        assert ts.best_loss == 2.0
        assert ts.best_epoch == 0

        # Second epoch with worse loss
        ts.update_on_epoch_end(1, 2.5, 0.45)
        assert ts.epoch == 1
        assert ts.best_loss == 2.0  # Unchanged
        assert ts.best_epoch == 0  # Unchanged

        # Third epoch with better loss
        ts.update_on_epoch_end(2, 1.5, 0.6)
        assert ts.epoch == 2
        assert ts.best_loss == 1.5  # Updated
        assert ts.best_epoch == 2  # Updated


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
