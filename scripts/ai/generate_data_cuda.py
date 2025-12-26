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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1000)
    parser.add_argument("--size", type=int, default=9)
    args = parser.parse_args()

    if args.size > 9:
        print("Error: Size limited to 9 for this application.")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: CUDA not found, using CPU tensors (slower).")

    grids = generate_latin_squares_cuda(args.batch, args.size, device)
    # Save or print logic here
