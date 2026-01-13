#!/usr/bin/env python3
"""
Benchmark for MPS batched inference (Phase 1.5).

Compares:
1. Direct 8-thread execution (current, fails/crashes at 4+ threads)
2. Batched 8-thread execution (new, works via limited workers; default: 1)

This demonstrates that batching achieves correct results at 8 threads
where direct execution fails due to Apple Metal bugs.

Run:
    python3 tests/benchmark_batching.py
"""

import argparse
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn

# Enable MPS graph path for thread safety
os.environ["MPS_FORCE_GRAPH_PATH"] = "1"


class SimpleMLP(nn.Module):
    """Simple MLP for benchmarking."""

    def __init__(self, input_dim=256, hidden_dim=512, output_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class TransformerMLPBlock(nn.Module):
    """More complex model with LayerNorm."""

    def __init__(self, d_model=256, d_ff=1024):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm1(x)
        x = self.fc2(self.gelu(self.fc1(x)))
        return self.norm2(x)


def benchmark_sequential(model, inputs, num_iterations):
    """Benchmark sequential (single-threaded) execution."""
    torch.mps.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        for _ in range(num_iterations):
            for inp in inputs:
                _ = model(inp)
            torch.mps.synchronize()

    t_total = time.perf_counter() - t0
    return t_total


def benchmark_direct_parallel(model, inputs, num_threads, num_iterations):
    """
    Benchmark direct parallel execution (8 threads each doing inference).

    WARNING: This may crash or produce incorrect results at 4+ threads
    due to Apple Metal framework bugs.
    """
    results = []
    errors = []

    def worker(tid, iteration):
        try:
            with torch.no_grad():
                inp = inputs[tid % len(inputs)]
                result = model(inp)
            torch.mps.synchronize()
            return result
        except Exception as e:
            errors.append((tid, iteration, e))
            return None

    torch.mps.synchronize()
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i in range(num_iterations):
            for tid in range(num_threads):
                futures.append(executor.submit(worker, tid, i))
        for f in futures:
            f.result()

    t_total = time.perf_counter() - t0

    if errors:
        print(f"  WARNING: {len(errors)} errors during direct parallel execution")
        for tid, i, e in errors[:5]:  # Show first 5
            print(f"    Thread {tid}, iter {i}: {e}")

    return t_total, len(errors)


def benchmark_batched_parallel(
    model, inputs, num_threads, num_iterations, num_workers: int
):
    """
    Benchmark batched parallel execution via MPSBatchQueue.

    N user threads submit requests to a queue processed by M workers.
    """
    if not hasattr(torch.mps, "BatchQueue"):
        print("  SKIP: torch.mps.BatchQueue not available")
        return None, 0

    queue = torch.mps.BatchQueue(batch_size=num_threads, num_workers=num_workers)
    queue.start()

    errors = []

    def submit_batch(inputs):
        futures = []
        for inp in inputs:
            futures.append(queue.submit(inp, lambda x: model(x)))
        return futures

    torch.mps.synchronize()
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for _ in range(num_iterations):
            # Each thread submits one input
            all_futures = list(executor.map(
                lambda inp: queue.submit(inp, lambda x: model(x)),
                (inputs * (num_threads // len(inputs) + 1))[:num_threads]
            ))
            # Wait for all results
            for f in all_futures:
                try:
                    f.result(timeout=30.0)
                except Exception as e:
                    errors.append(e)

    t_total = time.perf_counter() - t0
    queue.stop()

    return t_total, len(errors)


def run_benchmark(model_name, model, batch_size=4, input_dim=256,
                  num_threads=8, iterations=50, num_workers: int = 1):
    """Run full benchmark suite for a model."""
    print(f"\n{'=' * 60}")
    print(f"Benchmark: {model_name}")
    print(
        f"Threads: {num_threads}, Iterations: {iterations}, "
        f"Batch: {batch_size}, Workers: {num_workers}"
    )
    print("=" * 60)

    # Create inputs
    inputs = [
        torch.randn(batch_size, input_dim, device="mps")
        for _ in range(num_threads)
    ]
    torch.mps.synchronize()

    # 1. Sequential baseline
    print("\n--- Sequential (1 thread) ---")
    t_seq = benchmark_sequential(model, inputs, iterations)
    ops_seq = num_threads * iterations
    print(f"Time: {t_seq:.3f}s, Throughput: {ops_seq / t_seq:.1f} ops/s")

    # 2. Direct parallel (may crash at 4+)
    print(f"\n--- Direct Parallel ({num_threads} threads) ---")
    try:
        t_direct, errors_direct = benchmark_direct_parallel(
            model, inputs, num_threads, iterations
        )
        ops_direct = num_threads * iterations
        speedup_direct = t_seq / t_direct if t_direct > 0 else 0
        efficiency_direct = speedup_direct / num_threads * 100
        print(f"Time: {t_direct:.3f}s, Throughput: {ops_direct / t_direct:.1f} ops/s")
        print(f"Speedup: {speedup_direct:.2f}x, Efficiency: {efficiency_direct:.1f}%")
        if errors_direct > 0:
            print(f"Errors: {errors_direct} (Apple Metal bug)")
    except Exception as e:
        print(f"CRASHED: {type(e).__name__}: {e}")
        t_direct, errors_direct = None, -1

    # 3. Batched parallel (M workers)
    print(f"\n--- Batched Parallel ({num_threads} threads -> {num_workers} workers) ---")
    t_batched, errors_batched = benchmark_batched_parallel(
        model, inputs, num_threads, iterations, num_workers=num_workers
    )
    if t_batched is not None:
        ops_batched = num_threads * iterations
        speedup_batched = t_seq / t_batched if t_batched > 0 else 0
        efficiency_batched = speedup_batched / num_workers * 100  # vs worker threads
        print(f"Time: {t_batched:.3f}s, Throughput: {ops_batched / t_batched:.1f} ops/s")
        print(f"Speedup: {speedup_batched:.2f}x vs sequential")
        print(f"Efficiency vs {num_workers} workers: {efficiency_batched:.1f}%")
        if errors_batched > 0:
            print(f"Errors: {errors_batched}")

    # Summary
    print("\n--- Summary ---")
    if t_batched is not None and errors_direct == 0:
        print("Direct parallel works (no Apple bug hit)")
        print(f"Batching overhead: {(t_batched / t_direct - 1) * 100:.1f}%")
    elif t_batched is not None:
        print("Direct parallel FAILED (Apple Metal bug)")
        print(f"Batched parallel SUCCEEDED with speedup {speedup_batched:.2f}x")
    else:
        print("Batched parallel not available (need Phase 1.3 bindings)")


def run_fallback_benchmark():
    """Run benchmark without BatchQueue bindings."""
    print("\n=== Fallback Benchmark (No BatchQueue) ===")

    model = SimpleMLP().to("mps").eval()
    batch_size = 4
    input_dim = 256
    iterations = 50

    inputs = [torch.randn(batch_size, input_dim, device="mps") for _ in range(2)]
    torch.mps.synchronize()

    # Sequential
    t_seq = benchmark_sequential(model, inputs, iterations)
    print(f"Sequential: {t_seq:.3f}s ({2 * iterations / t_seq:.1f} ops/s)")

    # 2 threads (safe)
    t_2t, errors_2t = benchmark_direct_parallel(model, inputs, 2, iterations)
    speedup_2t = t_seq / t_2t if t_2t > 0 else 0
    print(f"2 threads:  {t_2t:.3f}s ({2 * iterations / t_2t:.1f} ops/s)")
    print(f"            Speedup: {speedup_2t:.2f}x, Efficiency: {speedup_2t / 2 * 100:.1f}%")

    print("\nNote: 8-thread batching requires Phase 1.3 Python bindings")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark MPS batched inference"
    )
    parser.add_argument(
        "--model", choices=["mlp", "transformer"], default="mlp",
        help="Model to benchmark"
    )
    parser.add_argument(
        "--threads", type=int, default=8,
        help="Number of user threads"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of BatchQueue worker threads (default: 1)",
    )
    parser.add_argument(
        "--iterations", type=int, default=50,
        help="Iterations per benchmark"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size per inference"
    )
    args = parser.parse_args()

    if args.workers < 1 or args.workers > 3:
        print("ERROR: --workers must be 1-3", file=sys.stderr)
        sys.exit(2)

    if not torch.backends.mps.is_available():
        print("ERROR: MPS not available")
        sys.exit(1)

    # Check if BatchQueue is available
    has_batch_queue = hasattr(torch.mps, "BatchQueue")
    if not has_batch_queue:
        print("Note: torch.mps.BatchQueue not available")
        print("      Running fallback benchmark (2 threads only)")
        run_fallback_benchmark()
        return

    # Create model
    if args.model == "mlp":
        model = SimpleMLP().to("mps").eval()
        model_name = "SimpleMLP"
        input_dim = 256
    else:
        model = TransformerMLPBlock().to("mps").eval()
        model_name = "TransformerMLPBlock"
        input_dim = 256

    # Run benchmark
    run_benchmark(
        model_name, model,
        batch_size=args.batch_size,
        input_dim=input_dim,
        num_threads=args.threads,
        iterations=args.iterations,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
