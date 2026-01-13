#!/usr/bin/env python3
"""
Test raw Metal compute vs MPS to see where serialization occurs.

This test bypasses PyTorch/MPS and uses Metal directly via PyObjC
to understand if the serialization is in MPS or the Metal driver.
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor

try:
    import Metal
    import MetalPerformanceShaders as MPS
    HAS_PYOBJC = True
except ImportError:
    HAS_PYOBJC = False
    print("PyObjC not available. Install with: pip install pyobjc-framework-Metal")
    print("Skipping raw Metal tests.")


def benchmark_raw_metal_sequential(device, num_iters: int = 1000) -> float:
    """Benchmark raw Metal compute - sequential."""
    queue = device.newCommandQueue()

    # Simple compute pass - just testing command buffer overhead
    start = time.perf_counter()

    for _ in range(num_iters):
        buffer = queue.commandBuffer()
        buffer.commit()
        buffer.waitUntilCompleted()

    elapsed = time.perf_counter() - start
    return num_iters / elapsed


def benchmark_raw_metal_parallel(device, num_threads: int = 8, iters_per_thread: int = 500) -> float:
    """Benchmark raw Metal compute - parallel threads."""

    def worker(thread_id: int) -> int:
        # Each thread gets its own command queue
        queue = device.newCommandQueue()

        for _ in range(iters_per_thread):
            buffer = queue.commandBuffer()
            buffer.commit()
            buffer.waitUntilCompleted()

        return iters_per_thread

    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        results = list(pool.map(worker, range(num_threads)))

    elapsed = time.perf_counter() - start
    return sum(results) / elapsed


def benchmark_raw_metal_shared_queue(device, num_threads: int = 8, iters_per_thread: int = 500) -> float:
    """Benchmark raw Metal with shared command queue - tests queue contention."""
    queue = device.newCommandQueue()

    def worker(thread_id: int) -> int:
        for _ in range(iters_per_thread):
            buffer = queue.commandBuffer()
            buffer.commit()
            buffer.waitUntilCompleted()

        return iters_per_thread

    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        results = list(pool.map(worker, range(num_threads)))

    elapsed = time.perf_counter() - start
    return sum(results) / elapsed


def main():
    if not HAS_PYOBJC:
        return

    device = Metal.MTLCreateSystemDefaultDevice()
    if not device:
        print("ERROR: No Metal device available")
        return

    print(f"Device: {device.name()}")
    print("=" * 60)

    # Sequential baseline
    seq_ops = benchmark_raw_metal_sequential(device, 2000)
    print(f"Sequential (1 thread):     {seq_ops:.0f} ops/s")

    # Parallel with separate queues
    for n_threads in [2, 4, 8]:
        parallel_ops = benchmark_raw_metal_parallel(device, n_threads, 500)
        scaling = parallel_ops / seq_ops
        efficiency = scaling / n_threads
        print(f"Parallel {n_threads} threads (sep queues): {parallel_ops:.0f} ops/s "
              f"({scaling:.2f}x, {efficiency:.1%} eff)")

    # Parallel with shared queue
    print()
    for n_threads in [2, 4, 8]:
        shared_ops = benchmark_raw_metal_shared_queue(device, n_threads, 500)
        scaling = shared_ops / seq_ops
        efficiency = scaling / n_threads
        print(f"Parallel {n_threads} threads (shared queue): {shared_ops:.0f} ops/s "
              f"({scaling:.2f}x, {efficiency:.1%} eff)")


if __name__ == "__main__":
    main()
