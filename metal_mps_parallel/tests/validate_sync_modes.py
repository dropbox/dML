#!/usr/bin/env python3
"""
Quick Validation: Event Sync vs Device Sync Performance

This test validates the key finding from N=1399:
- Device sync (torch.mps.synchronize): NO scaling at 8 threads
- Event sync (torch.mps.Event): ~2.5-2.8x throughput vs device sync

Run: python3 tests/validate_sync_modes.py
Expected: Event sync achieves >2x throughput vs device sync (for small matrices)

Note: Uses 512x512 FP16 matmul - the workload where the difference is most visible.
For GPU-saturated workloads (2048x2048+), neither sync mode helps significantly.
"""

import torch
import time
import threading
from concurrent.futures import ThreadPoolExecutor

def main():
    print("=" * 60)
    print("SYNC MODE VALIDATION TEST")
    print("=" * 60)

    if not torch.backends.mps.is_available():
        print("ERROR: MPS not available")
        return 1

    device = torch.device("mps")
    torch.manual_seed(42)

    # Use 512x512 FP16 matmul - the workload where event sync shines
    num_threads = 8
    size = 512
    dtype = torch.float16
    iterations = 200

    # Create matrices for each thread (avoid contention on same tensors)
    matrices_a = [torch.randn(size, size, device=device, dtype=dtype) for _ in range(num_threads)]
    matrices_b = [torch.randn(size, size, device=device, dtype=dtype) for _ in range(num_threads)]

    # Warmup
    print("\nWarming up GPU...")
    for i in range(num_threads):
        for _ in range(10):
            _ = matrices_a[i] @ matrices_b[i]
    torch.mps.synchronize()

    # Test 1: Device sync
    print(f"\n[1/3] Testing DEVICE sync ({num_threads} threads, {iterations} iters each)...")

    barrier = threading.Barrier(num_threads)

    def device_sync_worker(tid):
        a, b = matrices_a[tid], matrices_b[tid]
        barrier.wait()
        for _ in range(iterations):
            _ = a @ b
            torch.mps.synchronize()  # Device-wide sync - waits for ALL streams
        return iterations

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        results = list(pool.map(device_sync_worker, range(num_threads)))
    device_elapsed = time.perf_counter() - start
    device_ops = sum(results) / device_elapsed

    print(f"   Device sync: {device_ops:.0f} ops/s")

    # Test 2: Event sync
    print(f"\n[2/3] Testing EVENT sync ({num_threads} threads, {iterations} iters each)...")

    barrier = threading.Barrier(num_threads)

    def event_sync_worker(tid):
        a, b = matrices_a[tid], matrices_b[tid]
        event = torch.mps.Event(enable_timing=False)
        barrier.wait()
        for _ in range(iterations):
            _ = a @ b
            event.record()
            event.synchronize()  # Per-stream sync - waits only for this thread's ops
        return iterations

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        results = list(pool.map(event_sync_worker, range(num_threads)))
    event_elapsed = time.perf_counter() - start
    event_ops = sum(results) / event_elapsed

    print(f"   Event sync: {event_ops:.0f} ops/s")

    # Test 3: Single-thread baseline (with device sync)
    print(f"\n[3/3] Testing SINGLE-THREAD baseline ({iterations} iters)...")

    a, b = matrices_a[0], matrices_b[0]

    start = time.perf_counter()
    for _ in range(iterations):
        _ = a @ b
        torch.mps.synchronize()
    baseline_elapsed = time.perf_counter() - start
    baseline_ops = iterations / baseline_elapsed

    print(f"   Baseline: {baseline_ops:.0f} ops/s")

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    device_scaling = device_ops / baseline_ops
    event_scaling = event_ops / baseline_ops
    improvement = event_ops / device_ops

    print(f"\nBaseline (1 thread):    {baseline_ops:.0f} ops/s")
    print(f"Device sync (8 thread): {device_ops:.0f} ops/s ({device_scaling:.2f}x scaling)")
    print(f"Event sync (8 thread):  {event_ops:.0f} ops/s ({event_scaling:.2f}x scaling)")
    print(f"\nEvent/Device ratio:     {improvement:.2f}x")

    print("\n" + "=" * 60)
    if improvement >= 2.0:
        print(f"VALIDATION PASSED: Event sync is {improvement:.1f}x better than device sync")
        print("This confirms the finding from N=1399")
        print("=" * 60)
        return 0
    elif improvement >= 1.5:
        print(f"VALIDATION PARTIAL: Event sync is {improvement:.1f}x better")
        print("Lower than expected (2.5x), but still demonstrates the improvement")
        print("=" * 60)
        return 0
    else:
        print(f"VALIDATION INCONCLUSIVE: Only {improvement:.1f}x improvement")
        print("Expected >2x. May indicate GPU saturation or other factors.")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    exit(main())
