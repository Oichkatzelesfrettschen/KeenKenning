"""
Isotopism Augmentation for Latin Square Training

Isotopism is the mathematical term for the symmetry group of Latin squares.
An isotopism consists of three independent permutations:
- Row permutation (σ): Reorder rows
- Column permutation (τ): Reorder columns
- Symbol permutation (ρ): Relabel digits

Any composition of these preserves the Latin square property.

For an n×n grid, the isotopism group has size n! × n! × n!.
This provides massive data augmentation without generating new puzzles.

SPDX-License-Identifier: MIT
SPDX-FileCopyrightText: Copyright (C) 2025 KeenKenning Contributors
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def apply_row_permutation(grid: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Apply a row permutation to a Latin square.

    Args:
        grid: Input grid of shape (N, N)
        perm: Permutation of row indices (length N)

    Returns:
        Grid with rows reordered according to perm
    """
    return grid[perm, :]


def apply_column_permutation(grid: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Apply a column permutation to a Latin square.

    Args:
        grid: Input grid of shape (N, N)
        perm: Permutation of column indices (length N)

    Returns:
        Grid with columns reordered according to perm
    """
    return grid[:, perm]


def apply_symbol_permutation(grid: np.ndarray, mapping: dict) -> np.ndarray:
    """Apply a symbol permutation (relabeling) to a Latin square.

    Args:
        grid: Input grid of shape (N, N)
        mapping: Dictionary mapping old symbols to new symbols

    Returns:
        Grid with symbols relabeled according to mapping
    """
    result = np.zeros_like(grid)
    for old_val, new_val in mapping.items():
        result[grid == old_val] = new_val
    return result


def random_isotopism(
    grid: np.ndarray,
    size: int,
    rng: np.random.Generator = None,
    apply_row: bool = True,
    apply_col: bool = True,
    apply_symbol: bool = True,
) -> np.ndarray:
    """Apply a random isotopism to a Latin square.

    Args:
        grid: Input grid (may be padded, only first size×size is used)
        size: Actual grid size (N)
        rng: Random number generator
        apply_row: Whether to apply row permutation
        apply_col: Whether to apply column permutation
        apply_symbol: Whether to apply symbol permutation

    Returns:
        Transformed grid with same shape as input
    """
    if rng is None:
        rng = np.random.default_rng()

    # Extract the active part of the grid
    active = grid[:size, :size].copy()

    # Get the unique values in the grid (symbols)
    symbols = np.unique(active)

    if apply_row:
        row_perm = rng.permutation(size)
        active = apply_row_permutation(active, row_perm)

    if apply_col:
        col_perm = rng.permutation(size)
        active = apply_column_permutation(active, col_perm)

    if apply_symbol:
        # Create a random bijection of symbols
        shuffled = rng.permutation(symbols)
        symbol_map = dict(zip(symbols, shuffled))
        active = apply_symbol_permutation(active, symbol_map)

    # Put the transformed grid back (preserving padding if any)
    result = grid.copy()
    result[:size, :size] = active
    return result


def random_isotopism_with_mask(
    grid: np.ndarray,
    mask: np.ndarray,
    size: int,
    rng: np.random.Generator = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the same random isotopism to both grid and mask.

    This is useful when the mask indicates which cells are revealed.
    The same row/column permutations must be applied to keep them aligned.

    Args:
        grid: Input grid
        mask: Boolean mask of same shape
        size: Actual grid size
        rng: Random number generator

    Returns:
        (transformed_grid, transformed_mask)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Generate permutations once
    row_perm = rng.permutation(size)
    col_perm = rng.permutation(size)

    # Get symbols and create mapping
    symbols = np.unique(grid[:size, :size])
    shuffled = rng.permutation(symbols)
    symbol_map = dict(zip(symbols, shuffled))

    # Apply to grid
    active_grid = grid[:size, :size].copy()
    active_grid = apply_row_permutation(active_grid, row_perm)
    active_grid = apply_column_permutation(active_grid, col_perm)
    active_grid = apply_symbol_permutation(active_grid, symbol_map)

    # Apply row/col to mask (symbol permutation doesn't apply to mask)
    active_mask = mask[:size, :size].copy()
    active_mask = apply_row_permutation(active_mask, row_perm)
    active_mask = apply_column_permutation(active_mask, col_perm)

    # Reconstruct full arrays
    result_grid = grid.copy()
    result_grid[:size, :size] = active_grid

    result_mask = mask.copy()
    result_mask[:size, :size] = active_mask

    return result_grid, result_mask


if __name__ == "__main__":
    # Quick sanity check
    print("=== Isotopism Sanity Check ===\n")

    # Create a simple 4x4 Latin square
    grid = np.array([
        [1, 2, 3, 4],
        [2, 1, 4, 3],
        [3, 4, 1, 2],
        [4, 3, 2, 1],
    ], dtype=np.int64)

    print("Original grid:")
    print(grid)
    print()

    rng = np.random.default_rng(42)

    # Apply random isotopism
    transformed = random_isotopism(grid, 4, rng)
    print("After random isotopism:")
    print(transformed)
    print()

    # Verify it's still a Latin square
    def is_latin_square(g, n):
        for i in range(n):
            if len(set(g[i, :])) != n:
                return False
            if len(set(g[:, i])) != n:
                return False
        return True

    print(f"Original is Latin square: {is_latin_square(grid, 4)}")
    print(f"Transformed is Latin square: {is_latin_square(transformed, 4)}")
