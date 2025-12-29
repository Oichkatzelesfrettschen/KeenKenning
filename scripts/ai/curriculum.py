"""
Multi-Dimensional Curriculum Scheduler for Latin Square Training

Handles progressive introduction of:
- Grid sizes (3-5 → 3-8 → 3-12 → 3-16)
- Game modes (STANDARD → +ZERO → +NEGATIVE)
- Fill ratios (70-50% → 50-30% → 30-10% → 10-0%)

The curriculum scheduler coordinates all dimensions and provides
epoch-based or metric-based stage transitions.

SPDX-License-Identifier: MIT
SPDX-FileCopyrightText: Copyright (C) 2025 KeenKenning Contributors
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

from token_vocabulary import GameMode


class SizeStage(IntEnum):
    """Grid size stages."""
    SMALL = 0      # 3-5
    MEDIUM = 1     # 3-8
    LARGE = 2      # 3-12
    FULL = 3       # 3-16


class ModeStage(IntEnum):
    """Game mode stages."""
    STANDARD_ONLY = 0    # STANDARD
    WITH_ZERO = 1        # STANDARD + ZERO_INCLUSIVE
    ALL_MODES = 2        # All including NEGATIVE


class FillStage(IntEnum):
    """Fill ratio stages (percentage of cells revealed)."""
    HIGH = 0     # 50-70% revealed (easy, many hints)
    MEDIUM = 1   # 30-50% revealed
    LOW = 2      # 10-30% revealed
    MINIMAL = 3  # 0-10% revealed (expert, nearly empty)


# Stage definitions
SIZE_STAGES: Dict[SizeStage, List[int]] = {
    SizeStage.SMALL: [3, 4, 5],
    SizeStage.MEDIUM: [3, 4, 5, 6, 7, 8],
    SizeStage.LARGE: [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    SizeStage.FULL: list(range(3, 17)),
}

MODE_STAGES: Dict[ModeStage, List[GameMode]] = {
    ModeStage.STANDARD_ONLY: [GameMode.STANDARD],
    ModeStage.WITH_ZERO: [GameMode.STANDARD, GameMode.ZERO_INCLUSIVE],
    ModeStage.ALL_MODES: [GameMode.STANDARD, GameMode.ZERO_INCLUSIVE, GameMode.NEGATIVE],
}

FILL_STAGES: Dict[FillStage, Tuple[float, float]] = {
    FillStage.HIGH: (0.50, 0.70),      # 50-70% revealed
    FillStage.MEDIUM: (0.30, 0.50),    # 30-50% revealed
    FillStage.LOW: (0.10, 0.30),       # 10-30% revealed
    FillStage.MINIMAL: (0.00, 0.10),   # 0-10% revealed
}


@dataclass
class CurriculumState:
    """Current state of the curriculum."""
    size_stage: SizeStage = SizeStage.SMALL
    mode_stage: ModeStage = ModeStage.STANDARD_ONLY
    fill_stage: FillStage = FillStage.HIGH
    epoch: int = 0

    def to_dict(self) -> dict:
        """Serialize for checkpoint."""
        return {
            'size_stage': self.size_stage.value,
            'mode_stage': self.mode_stage.value,
            'fill_stage': self.fill_stage.value,
            'epoch': self.epoch,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'CurriculumState':
        """Deserialize from checkpoint."""
        return cls(
            size_stage=SizeStage(d['size_stage']),
            mode_stage=ModeStage(d['mode_stage']),
            fill_stage=FillStage(d['fill_stage']),
            epoch=d['epoch'],
        )


@dataclass
class CurriculumScheduler:
    """
    Multi-dimensional curriculum scheduler.

    Advances stages based on epoch thresholds or metric criteria.
    """
    total_epochs: int
    enable_size: bool = True
    enable_mode: bool = True
    enable_fill: bool = True

    # Epoch fractions for each stage (must sum to 1.0)
    size_schedule: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)
    mode_schedule: Tuple[float, float, float] = (0.50, 0.30, 0.20)
    fill_schedule: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)

    # Current state
    state: CurriculumState = field(default_factory=CurriculumState)

    # History for logging
    history: List[Tuple[int, CurriculumState]] = field(default_factory=list)

    def get_size_stage(self, epoch: int) -> SizeStage:
        """Get size stage for given epoch."""
        if not self.enable_size:
            return SizeStage.FULL

        cumulative = 0.0
        for i, frac in enumerate(self.size_schedule):
            cumulative += frac
            if epoch < self.total_epochs * cumulative:
                return SizeStage(i)
        return SizeStage.FULL

    def get_mode_stage(self, epoch: int) -> ModeStage:
        """Get mode stage for given epoch."""
        if not self.enable_mode:
            return ModeStage.ALL_MODES

        cumulative = 0.0
        for i, frac in enumerate(self.mode_schedule):
            cumulative += frac
            if epoch < self.total_epochs * cumulative:
                return ModeStage(i)
        return ModeStage.ALL_MODES

    def get_fill_stage(self, epoch: int) -> FillStage:
        """Get fill stage for given epoch."""
        if not self.enable_fill:
            return FillStage.MINIMAL

        cumulative = 0.0
        for i, frac in enumerate(self.fill_schedule):
            cumulative += frac
            if epoch < self.total_epochs * cumulative:
                return FillStage(i)
        return FillStage.MINIMAL

    def step(self, epoch: int) -> CurriculumState:
        """Update curriculum state for new epoch."""
        new_state = CurriculumState(
            size_stage=self.get_size_stage(epoch),
            mode_stage=self.get_mode_stage(epoch),
            fill_stage=self.get_fill_stage(epoch),
            epoch=epoch,
        )

        # Log transitions
        if new_state != self.state:
            self.history.append((epoch, new_state))

        self.state = new_state
        return new_state

    def get_allowed_sizes(self) -> List[int]:
        """Get currently allowed grid sizes."""
        return SIZE_STAGES[self.state.size_stage]

    def get_allowed_modes(self) -> List[GameMode]:
        """Get currently allowed game modes."""
        return MODE_STAGES[self.state.mode_stage]

    def get_fill_range(self) -> Tuple[float, float]:
        """Get current fill ratio range (min, max)."""
        return FILL_STAGES[self.state.fill_stage]

    def get_mode_weights(self) -> Dict[GameMode, float]:
        """Get sampling weights for modes.

        STANDARD is weighted higher because it's most common in real usage.
        """
        modes = self.get_allowed_modes()
        if len(modes) == 1:
            return {modes[0]: 1.0}

        # Give STANDARD 50% weight, distribute rest equally
        weights = {}
        other_weight = 0.5 / (len(modes) - 1)
        for mode in modes:
            if mode == GameMode.STANDARD:
                weights[mode] = 0.5
            else:
                weights[mode] = other_weight
        return weights

    def describe(self) -> str:
        """Human-readable description of current curriculum state."""
        sizes = self.get_allowed_sizes()
        modes = [m.name for m in self.get_allowed_modes()]
        fill_min, fill_max = self.get_fill_range()

        return (
            f"Curriculum[epoch={self.state.epoch}]: "
            f"sizes={sizes[0]}-{sizes[-1]}, "
            f"modes={'+'.join(modes)}, "
            f"fill={fill_min*100:.0f}-{fill_max*100:.0f}%"
        )

    def to_dict(self) -> dict:
        """Serialize for checkpoint."""
        return {
            'total_epochs': self.total_epochs,
            'enable_size': self.enable_size,
            'enable_mode': self.enable_mode,
            'enable_fill': self.enable_fill,
            'size_schedule': list(self.size_schedule),
            'mode_schedule': list(self.mode_schedule),
            'fill_schedule': list(self.fill_schedule),
            'state': self.state.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'CurriculumScheduler':
        """Deserialize from checkpoint."""
        scheduler = cls(
            total_epochs=d['total_epochs'],
            enable_size=d.get('enable_size', True),
            enable_mode=d.get('enable_mode', True),
            enable_fill=d.get('enable_fill', True),
            size_schedule=tuple(d.get('size_schedule', (0.25, 0.25, 0.25, 0.25))),
            mode_schedule=tuple(d.get('mode_schedule', (0.50, 0.30, 0.20))),
            fill_schedule=tuple(d.get('fill_schedule', (0.25, 0.25, 0.25, 0.25))),
        )
        if 'state' in d:
            scheduler.state = CurriculumState.from_dict(d['state'])
        return scheduler


if __name__ == "__main__":
    # Quick sanity check
    print("=== Curriculum Scheduler Sanity Check ===\n")

    scheduler = CurriculumScheduler(
        total_epochs=60,
        enable_size=True,
        enable_mode=True,
        enable_fill=True,
    )

    print("Epoch progression:")
    for epoch in [0, 10, 20, 30, 40, 50, 59]:
        scheduler.step(epoch)
        print(f"  {scheduler.describe()}")

    print("\n" + "=" * 50)
    print("Serialization test:")
    d = scheduler.to_dict()
    restored = CurriculumScheduler.from_dict(d)
    print(f"  Restored: {restored.describe()}")
