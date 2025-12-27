import torch
import time
import argparse
import os


def generate_latin_squares_cuda(batch_size=1000, size=9, device="cuda", max_steps=5000):
    """
    Generates Latin Squares using a tensor-based Min-Conflicts / Simulated Annealing approach.
    Faster than CPU backtracking for large batches of small-to-medium grids.
    """
    print(
        f"--- CUDA Generation: {batch_size} grids of size {size}x{size} on {device} ---"
    )

    # Initialize random grids: (B, N, N) with values 0..N-1
    # We ensure rows are valid permutations initially to reduce search space
    # Shape: [B, N, N]
    grids = torch.zeros(batch_size, size, size, dtype=torch.long, device=device)
    for i in range(size):
        # Fill each row with a random permutation of 0..N-1
        # randperm is not batchable easily in old torch, so we do a trick or loop
        # For speed in generation script, a loop over N rows is negligible vs B batches
        raw = (
            torch.arange(size, device=device).unsqueeze(0).repeat(batch_size, 1)
        )  # [B, N]
        # Shuffle each row independently? Fast approximation:
        # Add random noise and argsort
        noise = torch.rand(batch_size, size, device=device)
        perm = torch.argsort(noise, dim=1)
        grids[:, i, :] = perm

    # Now rows are valid. We only need to fix Column conflicts.
    # Optimizer: Swap elements within rows to minimize column duplicates.

    # Conflict calculation
    # We want to minimize: Sum of duplicates in each column

    start_time = time.time()

    # We will run a simple randomized descent (Stochastic Local Search)
    # Pick a random row, pick two random cols, swap if it reduces column conflicts (or with probability)

    # Pre-calculate column counts: [B, N, N] -> count of value v in col c
    # This is expensive to update fully.
    # Faster: Compute total conflicts.

    completed_grids = []

    # Mask of active (unsolved) grids
    active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

    for step in range(max_steps):
        if not active_mask.any():
            break

        # 1. Compute Conflicts for active grids
        # Get active subset
        curr_grids = grids[active_mask]
        B_curr = curr_grids.shape[0]

        # Check validity: Columns must be permutations.
        # fast check: sort columns and compare to 0..N-1
        sorted_cols, _ = torch.sort(curr_grids, dim=1)
        expected = (
            torch.arange(size, device=device)
            .view(1, size, 1)
            .expand(B_curr, size, size)
        )

        # Conflict matrix: [B, N, N] boolean, True if match
        conflicts = sorted_cols != expected
        num_conflicts = conflicts.sum(dim=(1, 2))  # [B]

        # Check for solutions
        solved_indices = (num_conflicts == 0).nonzero(as_tuple=True)[0]

        if len(solved_indices) > 0:
            # Save solved grids
            # We need to map back to original indices
            original_indices = active_mask.nonzero(as_tuple=True)[0][solved_indices]

            for idx in range(len(solved_indices)):
                local_idx = solved_indices[idx]
                global_idx = original_indices[idx]

                # Convert 0-based tensor to 1-based list
                grid_cpu = (curr_grids[local_idx] + 1).cpu().numpy().tolist()
                completed_grids.append(grid_cpu)

            # Update active mask
            active_mask[original_indices] = False
            if not active_mask.any():
                break

            # Refresh current view
            curr_grids = grids[active_mask]
            B_curr = curr_grids.shape[0]

        # 2. Perturbation (Batched Swaps)
        # Pick random row r: [B]
        r = torch.randint(0, size, (B_curr,), device=device)
        # Pick two random cols c1, c2
        c1 = torch.randint(0, size, (B_curr,), device=device)
        c2 = torch.randint(0, size, (B_curr,), device=device)

        # Perform swap on ALL active grids tentatively
        # To do this efficiently without cloning everything:
        # We assume simulated annealing logic: just swap.
        # If we want greedy, we check conflicts.
        # For simplicity/speed in this demo: Just Swap Randomly (Random Walk)
        # A pure random walk is slow. We need Min-Conflicts.
        # But implementing vectorized Min-Conflicts is tricky.
        # Let's try "Swap and Revert if worse" (Greedy Descent)

        # Current cost
        # We already computed sorted checks, but that doesn't give local conflict count easily.
        # Cost = Number of unique elements per column summed? No.
        # Cost = (N - number of unique elements in col) summed over cols.

        def get_cost(g):
            # g: [B, N, N]
            # Cost: sum of duplicates in columns.
            # Efficient: One-hot encode values -> Sum over rows -> Subtract 1 -> Relu -> Sum
            # Values 0..N-1
            one_hot = torch.nn.functional.one_hot(
                g, num_classes=size
            )  # [B, N, N, N] (Batch, Row, Col, Val)
            col_counts = one_hot.sum(dim=1)  # [B, N, N] (Batch, Col, Val) counts
            # If a col has value V count=2, that's 1 conflict. count=1 -> 0 conflicts.
            conflicts = (col_counts - 1).clamp(min=0).sum(dim=(1, 2))
            return conflicts

        current_cost = get_cost(curr_grids)

        # Make swap
        # We need advanced indexing for batched swap
        batch_idx = torch.arange(B_curr, device=device)

        val1 = curr_grids[batch_idx, r, c1]
        val2 = curr_grids[batch_idx, r, c2]

        # Swap
        curr_grids[batch_idx, r, c1] = val2
        curr_grids[batch_idx, r, c2] = val1

        new_cost = get_cost(curr_grids)

        # Accept if new_cost <= current_cost (Greedy + Plateau)
        # Reject if new_cost > current_cost
        reject = new_cost > current_cost

        if reject.any():
            # Revert swaps for rejected
            # (Swap back)
            r_rej = r[reject]
            c1_rej = c1[reject]
            c2_rej = c2[reject]
            b_rej = batch_idx[reject]

            # We can't use curr_grids directly if we act on it in place.
            # We must be careful.
            # Actually, easiest is:
            # Save old values before swap?
            pass
            # Re-swap
            val1_back = curr_grids[b_rej, r_rej, c1_rej]  # This is val2
            val2_back = curr_grids[b_rej, r_rej, c2_rej]  # This is val1

            curr_grids[b_rej, r_rej, c1_rej] = val2_back
            curr_grids[b_rej, r_rej, c2_rej] = val1_back

        # Write back to main storage
        grids[active_mask] = curr_grids

    duration = time.time() - start_time
    print(
        f"Finished. Generated {len(completed_grids)}/{batch_size} unique grids in {duration:.2f}s"
    )
    return completed_grids


def generate_massive_dataset(output_path="data/latin_squares_massive.npz",
                             min_size=3, max_size=16,
                             base_count=5000, device="cuda"):
    """
    Generate Latin squares for all sizes from min_size to max_size.
    Scales down count for larger grids (they take longer to generate).
    """
    import numpy as np
    import os

    os.makedirs("data", exist_ok=True)
    all_data = {}

    for size in range(min_size, max_size + 1):
        # Scale count based on size
        if size <= 9:
            count = base_count
        elif size <= 12:
            count = base_count // 2  # 50% for 10-12
        else:
            count = base_count // 4  # 25% for 13-16

        # Increase batch size and steps for larger grids
        if size <= 6:
            max_steps = 2000
        elif size <= 9:
            max_steps = 5000
        elif size <= 12:
            max_steps = 10000
        else:
            max_steps = 20000  # More steps for 13-16

        grids = generate_latin_squares_cuda(
            batch_size=count, size=size, device=device, max_steps=max_steps
        )

        if grids:
            all_data[f"size{size}"] = np.array(grids, dtype=np.uint8)
            print(f"  size{size}: {len(grids)} grids collected")
        else:
            print(f"  size{size}: WARNING - no grids generated!")

    np.savez_compressed(output_path, **all_data)
    print(f"\nSaved dataset to {output_path}")
    print(f"Total size: {os.path.getsize(output_path)/1024/1024:.2f} MB")


def extend_existing_dataset(existing_path="data/latin_squares_massive.npz",
                            output_path="data/latin_squares_extended.npz",
                            min_size=10, max_size=16,
                            base_count=5000, device="cuda"):
    """
    Generate additional sizes and merge with existing dataset.
    """
    import numpy as np
    import os

    # Load existing data
    if os.path.exists(existing_path):
        existing = dict(np.load(existing_path))
        print(f"Loaded existing data with {len(existing)} size groups")
    else:
        existing = {}

    # Generate new sizes
    for size in range(min_size, max_size + 1):
        key = f"size{size}"
        if key in existing:
            print(f"  {key}: already exists with {len(existing[key])} grids, skipping")
            continue

        # Scale count
        if size <= 9:
            count = base_count
        elif size <= 12:
            count = base_count // 2
        else:
            count = base_count // 4

        # More steps for larger sizes
        max_steps = 5000 if size <= 9 else (10000 if size <= 12 else 20000)

        grids = generate_latin_squares_cuda(
            batch_size=count, size=size, device=device, max_steps=max_steps
        )

        if grids:
            existing[key] = np.array(grids, dtype=np.uint8)
            print(f"  {key}: {len(grids)} grids generated")
        else:
            print(f"  {key}: WARNING - no grids generated!")

    np.savez_compressed(output_path, **existing)
    print(f"\nSaved extended dataset to {output_path}")
    print(f"Total size: {os.path.getsize(output_path)/1024/1024:.2f} MB")


if __name__ == "__main__":
    import sys
    import numpy as np

    parser = argparse.ArgumentParser(description="CUDA-accelerated Latin square generator")
    parser.add_argument("--batch", type=int, default=1000, help="Batch size per generation")
    parser.add_argument("--size", type=int, default=9, help="Grid size (3-16)")
    parser.add_argument("--extend", action="store_true",
                        help="Extend existing dataset with sizes 10-16")
    parser.add_argument("--full", action="store_true",
                        help="Generate complete dataset for sizes 3-16")
    parser.add_argument("--count", type=int, default=5000,
                        help="Base count per size (scaled down for larger sizes)")
    args = parser.parse_args()

    if args.size > 16:
        print("Error: Size limited to 16 for this model architecture.")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: CUDA not found, using CPU tensors (slower).")

    if args.extend:
        # Extend existing dataset with sizes 10-16
        extend_existing_dataset(
            existing_path="data/latin_squares_massive.npz",
            output_path="data/latin_squares_massive.npz",  # Overwrite
            min_size=10, max_size=16,
            base_count=args.count, device=device
        )
    elif args.full:
        # Generate complete dataset from scratch
        generate_massive_dataset(
            output_path="data/latin_squares_massive.npz",
            min_size=3, max_size=16,
            base_count=args.count, device=device
        )
    else:
        # Single size generation
        grids = generate_latin_squares_cuda(args.batch, args.size, device)
        if grids:
            arr = np.array(grids, dtype=np.uint8)
            output = f"data/latin_squares_{args.size}x{args.size}.npz"
            np.savez_compressed(output, **{f"size{args.size}": arr})
            print(f"Saved {len(grids)} grids to {output}")
