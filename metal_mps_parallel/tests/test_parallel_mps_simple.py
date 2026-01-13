#!/usr/bin/env python3
"""
Simple parallel MPS test - minimal operations to isolate the stream pool behavior.
"""
import threading
import time
import os

# Disable any profiling that might interfere
os.environ['PYTORCH_NO_MPS_PROFILER'] = '1'

import torch

def test_basic():
    """Test basic MPS availability."""
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")

    assert torch.backends.mps.is_available(), "MPS not available"

    x = torch.randn(10, 10, device='mps')
    y = x + x
    torch.mps.synchronize()
    print("Basic MPS test: PASS")


def worker(thread_id: int, results: list, errors: list, lock, iterations=10):
    """Worker function for parallel test."""
    try:
        for i in range(iterations):
            # Simple operations
            x = torch.randn(64, 64, device='mps')
            y = torch.randn(64, 64, device='mps')
            z = torch.mm(x, y)
            torch.mps.synchronize()

            with lock:
                results.append((thread_id, i))
    except Exception as e:
        with lock:
            errors.append((thread_id, str(e)))


def test_parallel(num_threads=2, iterations=5):
    """Test parallel MPS operations."""
    assert torch.backends.mps.is_available(), "MPS not available"

    errors = []
    results = []
    lock = threading.Lock()

    threads = []
    start = time.time()

    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(i, results, errors, lock, iterations))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    elapsed = time.time() - start

    print(f"Threads: {num_threads}, Iterations: {iterations}")
    print(f"Completed: {len(results)}, Errors: {len(errors)}")
    print(f"Time: {elapsed:.2f}s")

    if errors:
        for tid, err in errors:
            print(f"  Thread {tid} error: {err}")
        assert False, f"Got {len(errors)} errors"

    expected = num_threads * iterations
    assert len(results) == expected, f"Expected {expected}, got {len(results)}"
    print(f"Parallel test ({num_threads} threads): PASS")


if __name__ == '__main__':
    print("=== Simple MPS Parallel Test ===\n")

    try:
        test_basic()
    except AssertionError as e:
        print(f"FAILED: {e}")
        exit(1)

    print()

    # Start with 2 threads
    try:
        test_parallel(2, 5)
    except AssertionError as e:
        print(f"FAILED: {e}")
        exit(1)

    print()

    # Try 4 threads
    try:
        test_parallel(4, 5)
    except AssertionError as e:
        print(f"FAILED: {e}")
        exit(1)

    print("\n=== All tests passed ===")
