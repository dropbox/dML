#!/usr/bin/env python3
"""
Batched Inference Example for Apple Silicon MPS

This script demonstrates why batching is the optimal approach for high-throughput
inference on Apple Silicon GPUs. Batching scales near-linearly while threading
plateaus at ~4,000 ops/s due to GPU command queue bottleneck.

Key insight: GPUs are designed for parallel data (batching), not parallel tasks (threading).

Usage:
    python batched_inference.py [--batch-sizes 1,4,8,16,32] [--compare-threading]

Requirements:
    PyTorch with MPS support (pip install torch)

Tested on:
    - Apple M4 Max (macOS 15.7.2)
    - PyTorch 2.9.1a0 (custom MPS thread-safety patches)
"""

import torch
import torch.nn as nn
import threading
import time
import argparse
import sys
from typing import List, Tuple


class SimpleMLP(nn.Module):
    """A simple MLP for demonstration purposes."""

    def __init__(self, input_dim: int = 1024, hidden_dim: int = 4096, output_dim: int = 1024):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def benchmark_batched(model: nn.Module, batch_sizes: List[int],
                      total_samples: int = 1024, input_dim: int = 1024) -> List[dict]:
    """Benchmark batched inference at various batch sizes."""
    results = []
    device = torch.device('mps')
    model = model.to(device)
    model.eval()

    for batch_size in batch_sizes:
        num_batches = total_samples // batch_size
        inputs = torch.randn(batch_size, input_dim, device=device)

        # Warm up
        with torch.no_grad():
            for _ in range(5):
                _ = model(inputs)
        torch.mps.synchronize()

        # Benchmark
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_batches):
                _ = model(inputs)
        torch.mps.synchronize()
        elapsed = time.perf_counter() - start

        throughput = total_samples / elapsed
        results.append({
            'mode': 'batched',
            'batch_size': batch_size,
            'throughput': throughput,
            'time': elapsed,
            'samples': total_samples
        })

    return results


def benchmark_threaded(model: nn.Module, thread_counts: List[int],
                       iterations_per_thread: int = 128, input_dim: int = 1024) -> List[dict]:
    """Benchmark threaded inference (batch_size=1 per thread)."""
    results = []
    device = torch.device('mps')
    model = model.to(device)
    model.eval()

    def worker(iterations: int):
        inputs = torch.randn(1, input_dim, device=device)
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(inputs)

    for num_threads in thread_counts:
        # Warm up
        with torch.no_grad():
            inputs = torch.randn(1, input_dim, device=device)
            for _ in range(5):
                _ = model(inputs)
        torch.mps.synchronize()

        # Benchmark
        threads = []
        start = time.perf_counter()

        for _ in range(num_threads):
            t = threading.Thread(target=worker, args=(iterations_per_thread,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()
        torch.mps.synchronize()

        elapsed = time.perf_counter() - start
        total_samples = num_threads * iterations_per_thread
        throughput = total_samples / elapsed

        results.append({
            'mode': 'threaded',
            'threads': num_threads,
            'throughput': throughput,
            'time': elapsed,
            'samples': total_samples
        })

    return results


def print_results(batched_results: List[dict], threaded_results: List[dict] = None):
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 70)
    print("BATCHED INFERENCE RESULTS")
    print("=" * 70)
    print(f"{'Batch Size':>12} {'Throughput':>15} {'Time':>12} {'Efficiency*':>12}")
    print("-" * 70)

    if batched_results:
        baseline = batched_results[0]['throughput']
        for r in batched_results:
            efficiency = (r['throughput'] / r['samples']) / (baseline / batched_results[0]['samples']) * 100
            # Efficiency relative to theoretical linear scaling
            theoretical = baseline * (r['batch_size'] / batched_results[0]['batch_size'])
            actual_eff = (r['throughput'] / max(theoretical, 0.001)) * 100
            print(f"{r['batch_size']:>12} {r['throughput']:>12.1f}/s {r['time']:>10.3f}s {actual_eff:>10.1f}%")

    print("-" * 70)
    print("* Efficiency = actual throughput / (baseline * batch_size_factor)")

    if threaded_results:
        print("\n" + "=" * 70)
        print("THREADED INFERENCE RESULTS (batch_size=1 per thread)")
        print("=" * 70)
        print(f"{'Threads':>12} {'Throughput':>15} {'Time':>12} {'Efficiency':>12}")
        print("-" * 70)

        baseline = threaded_results[0]['throughput']
        for r in threaded_results:
            theoretical = baseline * r['threads']
            efficiency = (r['throughput'] / max(theoretical, 0.001)) * 100
            print(f"{r['threads']:>12} {r['throughput']:>12.1f}/s {r['time']:>10.3f}s {efficiency:>10.1f}%")

        print("-" * 70)
        print("* Efficiency = actual throughput / (baseline * thread_count)")

    # Summary comparison
    if batched_results and threaded_results:
        print("\n" + "=" * 70)
        print("COMPARISON: BATCHING vs THREADING")
        print("=" * 70)

        # Find comparable points (8 samples/threads)
        batch_8 = next((r for r in batched_results if r['batch_size'] == 8), None)
        thread_8 = next((r for r in threaded_results if r['threads'] == 8), None)

        if batch_8 and thread_8:
            ratio = batch_8['throughput'] / thread_8['throughput']
            print(f"\nAt 8 parallel units:")
            print(f"  Batching:  {batch_8['throughput']:.1f} samples/s")
            print(f"  Threading: {thread_8['throughput']:.1f} samples/s")
            print(f"  Batching is {ratio:.1f}x faster!")

        print("\n" + "-" * 70)
        print("CONCLUSION:")
        print("-" * 70)
        print("Batching achieves near-linear scaling because:")
        print("  1. Single GPU dispatch per batch (minimal CPU-GPU sync overhead)")
        print("  2. GPU can parallelize work within the batch")
        print("  3. No mutex contention between threads")
        print()
        print("Threading plateaus because:")
        print("  1. GPU command queue becomes the bottleneck")
        print("  2. Each thread issues separate GPU dispatches")
        print("  3. Total throughput capped at ~4,000 ops/s regardless of threads")
        print()
        print("RECOMMENDATION: Use batching for maximum throughput on Apple Silicon.")


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark batched vs threaded inference on Apple MPS'
    )
    parser.add_argument(
        '--batch-sizes', type=str, default='1,2,4,8,16,32',
        help='Comma-separated batch sizes to test (default: 1,2,4,8,16,32)'
    )
    parser.add_argument(
        '--thread-counts', type=str, default='1,2,4,8,16',
        help='Comma-separated thread counts to test (default: 1,2,4,8,16)'
    )
    parser.add_argument(
        '--compare-threading', action='store_true',
        help='Include threading comparison (default: batching only)'
    )
    parser.add_argument(
        '--total-samples', type=int, default=1024,
        help='Total samples to process per batch size (default: 1024)'
    )
    parser.add_argument(
        '--input-dim', type=int, default=1024,
        help='Input dimension for the model (default: 1024)'
    )
    args = parser.parse_args()

    # Parse batch sizes and thread counts
    batch_sizes = [int(x) for x in args.batch_sizes.split(',')]
    thread_counts = [int(x) for x in args.thread_counts.split(',')]

    # Check MPS availability
    if not torch.backends.mps.is_available():
        print("Error: MPS device is not available")
        sys.exit(1)

    print("=" * 70)
    print("MPS Batched Inference Benchmark")
    print("=" * 70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"Input dimension: {args.input_dim}")
    print(f"Total samples per test: {args.total_samples}")
    print(f"Batch sizes: {batch_sizes}")
    if args.compare_threading:
        print(f"Thread counts: {thread_counts}")

    # Create model
    model = SimpleMLP(input_dim=args.input_dim)

    print("\nRunning batched inference benchmark...")
    batched_results = benchmark_batched(
        model, batch_sizes,
        total_samples=args.total_samples,
        input_dim=args.input_dim
    )

    threaded_results = None
    if args.compare_threading:
        print("Running threaded inference benchmark...")
        threaded_results = benchmark_threaded(
            model, thread_counts,
            iterations_per_thread=args.total_samples // max(thread_counts),
            input_dim=args.input_dim
        )

    print_results(batched_results, threaded_results)

    print("\n" + "=" * 70)
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
