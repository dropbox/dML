#!/usr/bin/env python3
"""
MPS Synchronization Comparison: Device Sync vs Event Sync

This script demonstrates that torch.mps.synchronize() has device-wide
synchronization semantics that destroy parallel efficiency in multi-threaded
workloads. Use torch.mps.Event for per-stream synchronization instead.

Usage:
    python3 tests/mps_sync_comparison.py

Expected output shows event sync achieves ~2-3x better throughput at 8 threads
for non-GPU-saturated workloads.

Created by Andrew Yates
Date: 2025-12-20
For PyTorch upstream issue reproduction
"""

import torch
import time
from concurrent.futures import ThreadPoolExecutor
import sys


def benchmark_device_sync(num_threads: int, iters_per_thread: int = 500) -> float:
    """Benchmark using torch.mps.synchronize() - device-wide sync.

    This waits for ALL streams across all threads, serializing execution.
    """
    def worker(thread_id: int) -> int:
        a = torch.randn(512, 512, dtype=torch.float16, device='mps')
        b = torch.randn(512, 512, dtype=torch.float16, device='mps')
        for _ in range(iters_per_thread):
            c = torch.mm(a, b)
            torch.mps.synchronize()  # Syncs ALL streams - destroys parallelism!
        return iters_per_thread

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        results = list(pool.map(worker, range(num_threads)))
    elapsed = time.perf_counter() - start
    total_ops = sum(results)
    return total_ops / elapsed


def benchmark_event_sync(num_threads: int, iters_per_thread: int = 500) -> float:
    """Benchmark using torch.mps.Event - per-stream sync.

    This waits only for the current stream, enabling true parallelism.
    """
    def worker(thread_id: int) -> int:
        a = torch.randn(512, 512, dtype=torch.float16, device='mps')
        b = torch.randn(512, 512, dtype=torch.float16, device='mps')
        # Create thread-local event for per-stream synchronization
        event = torch.mps.Event(enable_timing=False)
        for _ in range(iters_per_thread):
            c = torch.mm(a, b)
            event.record()
            event.synchronize()  # Syncs only THIS stream - enables parallelism!
        return iters_per_thread

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        results = list(pool.map(worker, range(num_threads)))
    elapsed = time.perf_counter() - start
    total_ops = sum(results)
    return total_ops / elapsed


def main():
    # Check MPS availability
    if not torch.backends.mps.is_available():
        print("ERROR: MPS not available on this system")
        sys.exit(1)

    print("=" * 70)
    print("MPS Synchronization Comparison")
    print("=" * 70)
    print()
    print("This demonstrates that torch.mps.synchronize() has device-wide semantics")
    print("that destroy parallel efficiency. Use torch.mps.Event for multi-threading.")
    print()
    print("Hardware:", end=" ")
    # Get device info
    try:
        import subprocess
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True, text=True
        )
        for line in result.stdout.split('\n'):
            if 'Chipset Model' in line:
                print(line.strip().replace('Chipset Model: ', ''))
                break
        else:
            print("Apple Silicon")
    except:
        print("Apple Silicon")
    print()

    print("-" * 70)
    print(f"{'Threads':<10} {'Device Sync':<20} {'Event Sync':<20} {'Ratio':<10}")
    print(f"{'':10} {'(ops/s)':20} {'(ops/s)':20} {'':10}")
    print("-" * 70)

    device_baseline = None
    event_baseline = None
    device_results: dict[int, float] = {}
    event_results: dict[int, float] = {}

    for n_threads in [1, 2, 4, 8]:
        # Warmup
        if n_threads == 1:
            _ = benchmark_device_sync(1, iters_per_thread=50)
            _ = benchmark_event_sync(1, iters_per_thread=50)

        # Run benchmarks
        device_ops = benchmark_device_sync(n_threads)
        event_ops = benchmark_event_sync(n_threads)

        device_results[n_threads] = device_ops
        event_results[n_threads] = event_ops

        if n_threads == 1:
            device_baseline = device_ops
            event_baseline = event_ops

        ratio = event_ops / device_ops

        print(f"{n_threads:<10} {device_ops:<20.0f} {event_ops:<20.0f} {ratio:.2f}x")

    print("-" * 70)
    print()

    # Summary
    assert device_baseline is not None
    assert event_baseline is not None

    max_threads = max(device_results.keys())
    device_ops_max = device_results[max_threads]
    event_ops_max = event_results[max_threads]

    device_scaling = device_ops_max / device_baseline
    event_scaling = event_ops_max / event_baseline
    device_efficiency = device_scaling / max_threads
    event_efficiency = event_scaling / max_threads
    event_over_device = event_ops_max / device_ops_max

    print("KEY FINDINGS:")
    print()
    print("1. Device sync (torch.mps.synchronize()):")
    print(f"   - Scaling at {max_threads} threads: {device_scaling:.2f}x")
    print(f"   - Efficiency at {max_threads} threads: {device_efficiency:.1%}")
    print("   - WRONG for multi-threaded workloads")
    print()
    print("2. Event sync (torch.mps.Event):")
    print(f"   - Scaling at {max_threads} threads: {event_scaling:.2f}x")
    print(f"   - Efficiency at {max_threads} threads: {event_efficiency:.1%}")
    print(f"   - Speedup vs device sync at {max_threads} threads: {event_over_device:.2f}x")
    print("   - CORRECT for multi-threaded workloads")
    print()
    print("RECOMMENDATION:")
    print("  For multi-threaded MPS code, use torch.mps.Event for synchronization:")
    print()
    print("    event = torch.mps.Event(enable_timing=False)")
    print("    output = model(input)")
    print("    event.record()")
    print("    event.synchronize()  # Per-stream sync")
    print()


if __name__ == "__main__":
    main()
