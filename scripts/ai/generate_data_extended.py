#!/usr/bin/env python3
"""
generate_data_extended.py: Data generation for Latin squares 3x3 to 16x16

For sizes 3-9: Uses optimized C binary (latin_gen) if available
For sizes 10-16: Uses Python-based generation (Jacobson-Matthews algorithm)

SPDX-License-Identifier: MIT
SPDX-FileCopyrightText: Copyright (C) 2024-2025 Orthogon Contributors
"""

import argparse
import multiprocessing
import os
import subprocess
import sys
import time
from typing import List, Optional

import numpy as np

# Try to import numba for JIT acceleration
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("Note: numba not installed, using pure NumPy (slower for large grids)")


def generate_latin_square_numpy(n: int) -> np.ndarray:
    """
    Generate a random Latin square using row/column shuffling.

    This is a simple but valid approach for any size.
    """
    # Start with a cyclic Latin square
    square = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(n):
            square[i, j] = (i + j) % n + 1

    # Shuffle rows
    np.random.shuffle(square)

    # Shuffle columns
    col_perm = np.random.permutation(n)
    square = square[:, col_perm]

    # Relabel symbols randomly
    symbols = np.random.permutation(n) + 1
    relabeled = np.zeros_like(square)
    for i in range(n):
        relabeled[square == i + 1] = symbols[i]

    return relabeled


if HAS_NUMBA:
    @njit(cache=True)
    def _jacobson_matthews_step(square: np.ndarray, n: int) -> np.ndarray:
        """
        Single step of Jacobson-Matthews random walk.

        This generates uniformly random Latin squares by performing
        local moves in the space of Latin squares.
        """
        # Find a random cell
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        current_val = square[i, j]

        # Find a different value that could go in this cell
        new_val = np.random.randint(1, n + 1)
        if new_val == current_val:
            return square

        # Find where new_val appears in row i
        for jj in range(n):
            if square[i, jj] == new_val:
                j2 = jj
                break

        # Find where new_val appears in column j
        for ii in range(n):
            if square[ii, j] == new_val:
                i2 = ii
                break

        # Perform the swap
        square[i, j] = new_val
        square[i, j2] = current_val
        square[i2, j] = current_val
        square[i2, j2] = new_val

        return square


def generate_latin_square_jacobson(n: int, mixing_steps: int = 100) -> np.ndarray:
    """
    Generate a random Latin square using Jacobson-Matthews algorithm.

    This produces uniformly random Latin squares by random walk.
    """
    if HAS_NUMBA:
        # Start with cyclic square
        square = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            for j in range(n):
                square[i, j] = (i + j) % n + 1

        # Apply mixing steps
        for _ in range(mixing_steps * n * n):
            square = _jacobson_matthews_step(square, n)

        return square
    else:
        # Fall back to simple shuffle method
        return generate_latin_square_numpy(n)


def generate_batch_python(n: int, count: int, worker_id: int = 0) -> np.ndarray:
    """Generate a batch of Latin squares using Python."""
    print(f"[Worker {worker_id}] Generating {count} squares of size {n}x{n} (Python)")

    squares = []
    start_time = time.time()

    # Use simpler method for small grids, Jacobson-Matthews for large
    mixing_steps = max(50, n * 10)

    for i in range(count):
        if n <= 9:
            square = generate_latin_square_numpy(n)
        else:
            square = generate_latin_square_jacobson(n, mixing_steps)
        squares.append(square)

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"[Worker {worker_id}] Progress: {i+1}/{count} ({rate:.1f} grids/sec)")

    duration = time.time() - start_time
    print(f"[Worker {worker_id}] Done: {count} grids in {duration:.2f}s")

    return np.array(squares, dtype=np.uint8)


def worker_task_binary(args):
    """Worker task using the latin_gen binary (fast, for sizes 3-9)."""
    executable, size, count, worker_id = args
    cmd = [executable, "--raw", "--soak", str(size)]

    print(f"[Worker {worker_id}] Starting binary generation: {count} grids of {size}x{size}")

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, text=True, bufsize=4 * 1024 * 1024
    )

    grids = []
    current_grid = []
    count_generated = 0
    start_time = time.time()

    try:
        for line in process.stdout:
            line = line.strip()
            if not line:
                if len(current_grid) == size:
                    grids.append(current_grid)
                    current_grid = []
                    count_generated += 1

                    if count_generated % 5000 == 0:
                        rate = count_generated / (time.time() - start_time)
                        print(f"[Worker {worker_id}] Progress: {count_generated}/{count} ({rate:.1f}/s)")

                    if count_generated >= count:
                        break
                continue

            parts = [int(x) for x in line.split()]
            if len(parts) == size:
                current_grid.append(parts)

    finally:
        process.kill()

    duration = time.time() - start_time
    print(f"[Worker {worker_id}] Finished: {count_generated} grids in {duration:.2f}s")
    return np.array(grids, dtype=np.uint8)


def worker_task_python(args):
    """Worker task using Python generation (for sizes 10-16)."""
    size, count, worker_id = args
    return generate_batch_python(size, count, worker_id)


def generate_parallel_binary(size: int, total_count: int, executable: str) -> np.ndarray:
    """Generate grids using the C binary in parallel."""
    num_cores = multiprocessing.cpu_count()
    chunk_size = total_count // num_cores

    tasks = []
    for i in range(num_cores):
        c = chunk_size if i < num_cores - 1 else total_count - (chunk_size * (num_cores - 1))
        tasks.append((executable, size, c, i))

    print(f"--- Binary Generation: {total_count} grids, {size}x{size}, {num_cores} cores ---")

    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.map(worker_task_binary, tasks)

    return np.concatenate(results)


def generate_parallel_python(size: int, total_count: int) -> np.ndarray:
    """Generate grids using Python in parallel."""
    num_cores = min(multiprocessing.cpu_count(), 8)  # Limit for memory
    chunk_size = total_count // num_cores

    tasks = []
    for i in range(num_cores):
        c = chunk_size if i < num_cores - 1 else total_count - (chunk_size * (num_cores - 1))
        tasks.append((size, c, i))

    print(f"--- Python Generation: {total_count} grids, {size}x{size}, {num_cores} cores ---")

    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.map(worker_task_python, tasks)

    return np.concatenate(results)


def find_executable() -> Optional[str]:
    """Find the latin_gen executable."""
    candidates = ["./latin_gen_opt", "./latin_gen", "latin_gen"]
    for exe in candidates:
        if os.path.exists(exe):
            return exe
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate Latin square training data (3x3 to 16x16)"
    )
    parser.add_argument(
        "--count", type=int, default=5000,
        help="Base count per size (reduced for larger sizes)"
    )
    parser.add_argument(
        "--min-size", type=int, default=3, help="Minimum grid size"
    )
    parser.add_argument(
        "--max-size", type=int, default=16, help="Maximum grid size"
    )
    parser.add_argument(
        "--output", type=str, default="data/latin_squares_extended.npz",
        help="Output file path"
    )
    parser.add_argument(
        "--force-python", action="store_true",
        help="Force Python generation even for small grids"
    )
    args = parser.parse_args()

    executable = find_executable()
    if executable and not args.force_python:
        print(f"Using binary: {executable}")
    else:
        print("Using Python generation (binary not found or --force-python)")
        executable = None

    os.makedirs(os.path.dirname(args.output) or "data", exist_ok=True)
    all_data = {}

    # Size-scaled counts (larger grids = fewer samples due to time/memory)
    size_counts = {
        3: args.count,
        4: args.count,
        5: args.count,
        6: args.count,
        7: args.count,
        8: args.count,
        9: args.count,
        10: args.count // 2,  # 2500 default
        11: args.count // 2,
        12: args.count // 4,  # 1250 default
        13: args.count // 4,
        14: args.count // 5,  # 1000 default
        15: args.count // 5,
        16: args.count // 10,  # 500 default
    }

    for size in range(args.min_size, args.max_size + 1):
        count = size_counts.get(size, args.count // 10)

        print(f"\n=== Size {size}x{size}: {count} grids ===")

        if size <= 9 and executable:
            # Use fast C binary for small grids
            data = generate_parallel_binary(size, count, executable)
        else:
            # Use Python for large grids or if no binary
            data = generate_parallel_python(size, count)

        all_data[f"size{size}"] = data
        print(f"Generated {len(data)} grids for size {size}")

    # Save compressed
    np.savez_compressed(args.output, **all_data)
    file_size = os.path.getsize(args.output)
    print(f"\n=== SUCCESS ===")
    print(f"Output: {args.output}")
    print(f"Size: {file_size / 1024 / 1024:.2f} MB")

    # Summary
    print("\nSize breakdown:")
    for key in sorted(all_data.keys()):
        print(f"  {key}: {len(all_data[key])} grids")


if __name__ == "__main__":
    main()
