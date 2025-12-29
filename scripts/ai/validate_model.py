#!/usr/bin/env python3
"""
Validate ONNX model for Latin square generation.

Tests:
1. Model loads correctly
2. Input/output shapes match expected format
3. Generated grids satisfy Latin square constraints
4. Performance across grid sizes 3-16
"""

import argparse
import numpy as np
import time

def load_onnx_model(model_path):
    """Load ONNX model and return session."""
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(model_path)
        return session
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None


def validate_inputs_outputs(session):
    """Validate model has expected inputs and outputs."""
    inputs = session.get_inputs()
    outputs = session.get_outputs()

    print("\n=== Model I/O Validation ===")
    print(f"Inputs: {len(inputs)}")
    for inp in inputs:
        print(f"  - {inp.name}: {inp.shape} ({inp.type})")

    print(f"Outputs: {len(outputs)}")
    for out in outputs:
        print(f"  - {out.name}: {out.shape} ({out.type})")

    # Check expected format
    expected_inputs = {"input_grid", "grid_size"}
    actual_inputs = {inp.name for inp in inputs}

    if expected_inputs == actual_inputs:
        print("\n[PASS] Input names match expected format")
        return True
    else:
        print(f"\n[WARN] Expected inputs: {expected_inputs}")
        print(f"       Actual inputs: {actual_inputs}")
        return False


def is_valid_latin_square(grid, size):
    """Check if grid satisfies Latin square constraints."""
    for i in range(size):
        # Check row
        row = grid[i, :size]
        if len(set(row)) != size or min(row) < 1 or max(row) > size:
            return False
        # Check column
        col = grid[:size, i]
        if len(set(col)) != size or min(col) < 1 or max(col) > size:
            return False
    return True


def count_violations(grid, size):
    """Count row and column constraint violations."""
    violations = 0
    for i in range(size):
        row = grid[i, :size]
        col = grid[:size, i]
        # Row duplicates
        violations += len(row) - len(set(row))
        # Column duplicates
        violations += len(col) - len(set(col))
    return violations


def generate_grid(session, size):
    """Generate a Latin square of given size."""
    # Prepare inputs
    input_grid = np.zeros((1, 16, 16), dtype=np.int64)
    grid_size = np.array([size], dtype=np.int64)

    inputs = {
        "input_grid": input_grid,
        "grid_size": grid_size
    }

    # Run inference
    outputs = session.run(None, inputs)
    logits = outputs[0]  # [1, 17, 16, 16]

    # Argmax to get predictions (skip class 0 = empty)
    grid = np.zeros((size, size), dtype=np.int32)
    for y in range(size):
        for x in range(size):
            # Get logits for classes 1-size
            cell_logits = logits[0, 1:size+1, y, x]
            grid[y, x] = np.argmax(cell_logits) + 1

    return grid


def validate_generation(session, sizes=[3, 4, 5, 6, 9, 12, 16], samples=10):
    """Test generation across different sizes."""
    print("\n=== Generation Validation ===")
    print(f"Testing {samples} samples per size\n")

    results = {}

    for size in sizes:
        valid_count = 0
        total_violations = 0
        times = []

        for _ in range(samples):
            start = time.time()
            try:
                grid = generate_grid(session, size)
                elapsed = time.time() - start
                times.append(elapsed)

                if is_valid_latin_square(grid, size):
                    valid_count += 1
                else:
                    total_violations += count_violations(grid, size)
            except Exception as e:
                print(f"  Error generating {size}x{size}: {e}")

        valid_rate = valid_count / samples * 100
        avg_time = np.mean(times) * 1000 if times else 0
        avg_violations = total_violations / max(samples - valid_count, 1)

        status = "PASS" if valid_rate >= 80 else ("WARN" if valid_rate >= 50 else "FAIL")
        print(f"  {size:2d}x{size:2d}: [{status}] {valid_rate:5.1f}% valid, {avg_time:6.1f}ms/grid, ~{avg_violations:.1f} violations")

        results[size] = {
            "valid_rate": valid_rate,
            "avg_time_ms": avg_time,
            "avg_violations": avg_violations
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Validate Latin square ONNX model")
    parser.add_argument("--model", default="latin_solver.onnx", help="ONNX model path")
    parser.add_argument("--samples", type=int, default=20, help="Samples per size")
    parser.add_argument("--sizes", type=str, default="3,4,5,6,9,12,16", help="Comma-separated sizes")
    args = parser.parse_args()

    print(f"Validating model: {args.model}")

    # Load model
    session = load_onnx_model(args.model)
    if session is None:
        print("\nFailed to load model. Exiting.")
        return 1

    # Validate I/O
    io_valid = validate_inputs_outputs(session)

    # Validate generation
    sizes = [int(s) for s in args.sizes.split(",")]
    results = validate_generation(session, sizes, args.samples)

    # Summary
    print("\n=== Summary ===")
    total_valid = sum(r["valid_rate"] for r in results.values()) / len(results)
    print(f"Overall valid rate: {total_valid:.1f}%")

    if total_valid >= 80:
        print("Status: PASS - Model ready for deployment")
        return 0
    elif total_valid >= 50:
        print("Status: WARN - Model needs improvement")
        return 1
    else:
        print("Status: FAIL - Model not suitable for deployment")
        return 2


if __name__ == "__main__":
    exit(main())
