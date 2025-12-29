"""
Mode-Aware Value Transformation for Latin Square Training

Transforms Latin square values based on game mode:
- STANDARD: 1 to N (no transformation needed)
- ZERO_INCLUSIVE: 0 to N-1 (subtract 1)
- NEGATIVE: symmetric around 0 (complex mapping)

The key insight is that a valid Latin square remains valid under
any bijective value transformation. This allows training on multiple
modes without regenerating data.

SPDX-License-Identifier: MIT
SPDX-FileCopyrightText: Copyright (C) 2025 KeenKenning Contributors
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from token_vocabulary import GameMode


def transform_grid_values(
    grid: np.ndarray,
    size: int,
    source_mode: GameMode,
    target_mode: GameMode
) -> np.ndarray:
    """Transform grid values from one mode to another.

    Args:
        grid: Input grid with values in source_mode format
        size: Grid size (N)
        source_mode: Current value format (usually STANDARD with 1-N)
        target_mode: Desired value format

    Returns:
        New grid with transformed values
    """
    if source_mode == target_mode:
        return grid.copy()

    # Get source and target digit ranges
    source_digits = get_digits_for_mode(source_mode, size)
    target_digits = get_digits_for_mode(target_mode, size)

    # Create mapping from source to target
    mapping = dict(zip(source_digits, target_digits))

    # Apply transformation (use int64 to support negative values)
    result = np.zeros((size, size), dtype=np.int64)
    for r in range(size):
        for c in range(size):
            result[r, c] = mapping[int(grid[r, c])]

    return result


def get_digits_for_mode(mode: GameMode, size: int) -> List[int]:
    """Get the ordered list of valid digits for a mode and size.

    Returns digits in ascending order for consistent mapping.
    """
    if mode == GameMode.ZERO_INCLUSIVE:
        # 0 to N-1
        return list(range(size))

    elif mode == GameMode.NEGATIVE:
        # Symmetric around 0
        half = size // 2
        if size % 2 == 1:
            # Odd: include 0
            return list(range(-half, half + 1))
        else:
            # Even: exclude 0
            return list(range(-half, 0)) + list(range(1, half + 1))

    else:
        # STANDARD: 1 to N
        return list(range(1, size + 1))


def transform_standard_to_mode(
    grid: np.ndarray,
    size: int,
    target_mode: GameMode
) -> np.ndarray:
    """Convenience function to transform from STANDARD mode.

    Most training data uses STANDARD (1-N), so this is the common case.
    """
    return transform_grid_values(grid, size, GameMode.STANDARD, target_mode)


def random_mode_for_size(size: int, rng: np.random.Generator = None) -> GameMode:
    """Select a random mode appropriate for the given size.

    Some modes have constraints:
    - NEGATIVE for even sizes excludes 0
    - All modes work for all sizes

    Args:
        size: Grid size
        rng: Random number generator (optional)

    Returns:
        Randomly selected GameMode
    """
    if rng is None:
        rng = np.random.default_rng()

    # For training diversity, use weighted selection
    # STANDARD is most common in real usage
    modes = [GameMode.STANDARD, GameMode.ZERO_INCLUSIVE, GameMode.NEGATIVE]
    weights = [0.5, 0.25, 0.25]  # 50% STANDARD, 25% each ZERO/NEGATIVE

    return rng.choice(modes, p=weights)


if __name__ == "__main__":
    # Quick sanity check
    print("=== Mode Transform Sanity Check ===\n")

    # Test 6x6 grid - only test first row for clarity
    size = 6
    # Create a 1-row test grid with values 1-6
    test_row = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int64)

    print(f"Source digits (STANDARD size=6): {get_digits_for_mode(GameMode.STANDARD, size)}")
    print(f"Target digits (ZERO size=6):     {get_digits_for_mode(GameMode.ZERO_INCLUSIVE, size)}")
    print(f"Target digits (NEGATIVE size=6): {get_digits_for_mode(GameMode.NEGATIVE, size)}")
    print()

    # Demonstrate the mapping
    src = get_digits_for_mode(GameMode.STANDARD, size)
    tgt_neg = get_digits_for_mode(GameMode.NEGATIVE, size)
    print(f"STANDARD -> NEGATIVE mapping for size {size}:")
    for s, t in zip(src, tgt_neg):
        print(f"  {s} -> {t}")

    print()

    # Test 5x5 (odd - NEGATIVE includes 0)
    size = 5
    print(f"Source digits (STANDARD size=5): {get_digits_for_mode(GameMode.STANDARD, size)}")
    print(f"Target digits (NEGATIVE size=5): {get_digits_for_mode(GameMode.NEGATIVE, size)}")

    src = get_digits_for_mode(GameMode.STANDARD, size)
    tgt_neg = get_digits_for_mode(GameMode.NEGATIVE, size)
    print(f"STANDARD -> NEGATIVE mapping for size {size}:")
    for s, t in zip(src, tgt_neg):
        print(f"  {s} -> {t}")
