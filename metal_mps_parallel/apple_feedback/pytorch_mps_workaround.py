#!/usr/bin/env python3
"""
PyTorch MPS with mutex workaround for Metal thread safety.

This script demonstrates that PyTorch MPS CAN work with multiple threads
when internal mutex serialization is applied. However, efficiency is
limited to ~29% at 8 threads due to the serialization overhead.

Requirements:
    PyTorch with MPS support

Expected Result:
    Completes without crash, but with reduced parallel efficiency.

Tested on:
    - Apple M4 Max (macOS 15.7.2)
    - PyTorch 2.9.1a0 (custom MPS thread-safety patches)
"""

import torch
import threading
import time

def matrix_worker(thread_id: int, results: dict, iterations: int = 50):
    """Perform matrix multiplication in a loop on MPS device."""
    device = torch.device('mps')
    try:
        for i in range(iterations):
            a = torch.randn(1024, 1024, device=device)
            b = torch.randn(1024, 1024, device=device)
            c = torch.mm(a, b)
        torch.mps.synchronize()
        results[thread_id] = 'PASS'
    except Exception as e:
        results[thread_id] = f'FAIL: {e}'

def run_parallel_test(num_threads: int = 4):
    """Run matrix operations in parallel threads."""
    print(f"Starting {num_threads} threads...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS Available: {torch.backends.mps.is_available()}")
    print()

    results = {}
    threads = []
    start_time = time.time()

    for i in range(num_threads):
        t = threading.Thread(target=matrix_worker, args=(i, results))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    elapsed = time.time() - start_time

    print(f"\nResults after {elapsed:.2f}s:")
    for tid, result in sorted(results.items()):
        print(f"  Thread {tid}: {result}")

    passed = sum(1 for r in results.values() if r == 'PASS')
    print(f"\n{passed}/{num_threads} threads completed successfully")

def run_efficiency_test():
    """Measure efficiency at different thread counts."""
    print("\n" + "=" * 60)
    print("Efficiency Test (50 iterations per thread)")
    print("=" * 60)

    device = torch.device('mps')

    # Single-threaded baseline
    start = time.time()
    for _ in range(50):
        a = torch.randn(1024, 1024, device=device)
        b = torch.randn(1024, 1024, device=device)
        c = torch.mm(a, b)
    torch.mps.synchronize()
    single_time = time.time() - start
    print(f"\nSingle-threaded baseline: {single_time:.3f}s")

    for num_threads in [2, 4, 8]:
        results = {}
        threads = []
        start = time.time()

        for i in range(num_threads):
            t = threading.Thread(target=matrix_worker, args=(i, results, 50))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        multi_time = time.time() - start

        # Ideal time would be single_time (all threads run in parallel)
        # Actual efficiency = ideal_time / actual_time
        ideal_time = single_time  # Perfect parallelism
        efficiency = (ideal_time / multi_time) * 100

        print(f"{num_threads} threads: {multi_time:.3f}s, efficiency: {efficiency:.1f}%")

if __name__ == "__main__":
    print("=" * 60)
    print("PyTorch MPS Multi-threaded Test (with mutex workaround)")
    print("=" * 60)
    print()

    run_parallel_test(num_threads=4)
    run_efficiency_test()
