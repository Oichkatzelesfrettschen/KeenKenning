import subprocess
import numpy as np
import os
import time
import multiprocessing
import argparse
import sys


def worker_task(args):
    executable, size, count, worker_id = args
    # Use --raw to suppress status strings and speed up parsing
    cmd = [executable, "--raw", "--soak", str(size)]

    print(f"[Worker {worker_id}] Starting: target={count} grids of size {size}")

    # Increase buffer size to 4MB for high-throughput pipes
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, text=True, bufsize=4 * 1024 * 1024
    )

    grids = []
    current_grid = []

    count_generated = 0
    start_time = time.time()

    try:
        # Optimized parsing: read line-by-line but with minimal logic
        for line in process.stdout:
            line = line.strip()
            if not line:
                if len(current_grid) == size:
                    grids.append(current_grid)
                    current_grid = []
                    count_generated += 1

                    if count_generated % 5000 == 0:
                        now = time.time()
                        rate = count_generated / (now - start_time)
                        print(
                            f"[Worker {worker_id}] Progress: {count_generated}/{count} ({rate:.1f} grids/sec)"
                        )

                    if count_generated >= count:
                        break
                continue

            # Simple space split is faster than regex
            parts = [int(x) for x in line.split()]
            if len(parts) == size:
                current_grid.append(parts)

    except Exception as e:
        print(f"[Worker {worker_id}] Error: {e}")
    finally:
        process.kill()

    duration = time.time() - start_time
    print(f"[Worker {worker_id}] Finished: {count_generated} grids in {duration:.2f}s")
    return np.array(grids, dtype=np.uint8)


def generate_parallel(size, total_count, executable="./latin_gen"):
    num_cores = multiprocessing.cpu_count()
    chunk_size = total_count // num_cores

    tasks = []
    for i in range(num_cores):
        # Last worker gets the remainder
        c = (
            chunk_size
            if i < num_cores - 1
            else total_count - (chunk_size * (num_cores - 1))
        )
        tasks.append((executable, size, c, i))

    print(
        f"--- Parallel Generation: {total_count} grids, {size}x{size}, {num_cores} cores ---"
    )

    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.map(worker_task, tasks)

    return np.concatenate(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=5000, help="Grids per size")
    args = parser.parse_args()

    if not os.path.exists("./latin_gen_opt"):
        # Fallback to standard opt if simd_opt not found
        executable = "./latin_gen"
    else:
        executable = "./latin_gen_opt"

    print(f"--- Massive Data Generation (3x3 to 16x16) ---")
    print(f"Using executable: {executable}")

    os.makedirs("data", exist_ok=True)
    all_data = {}

    for size in range(3, 17):
        # Scale down count for larger grids (they take exponentially longer to generate)
        if size <= 9:
            count = args.count
        elif size <= 12:
            count = args.count // 2  # 50% for 10-12
        else:
            count = args.count // 4  # 25% for 13-16

        data = generate_parallel(size, count, executable)
        all_data[f"size{size}"] = data

    output_path = "data/latin_squares_massive.npz"
    np.savez_compressed(output_path, **all_data)
    print(f"\nSUCCESS: Massive dataset saved to {output_path}")
    print(f"Total size: {os.path.getsize(output_path)/1024/1024:.2f} MB")


if __name__ == "__main__":
    main()
