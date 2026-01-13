#!/usr/bin/env python3
"""
Extended stress test for MPS parallel inference.
Tests the limits of the stream pool implementation.
"""
import threading
import time
import sys


def test_extended_stress(num_threads=8, iterations=100):
    """
    Extended stress test - 8 threads x 100 iterations = 800 operations.
    """
    import torch

    errors = []
    results = []
    lock = threading.Lock()

    def worker(thread_id: int):
        try:
            for i in range(iterations):
                # Pattern from working test
                x = torch.randn(256, 256, device='mps')
                y = torch.randn(256, 256, device='mps')
                z = torch.mm(x, y)
                z = torch.relu(z)
                torch.mps.synchronize()

                with lock:
                    results.append((thread_id, i))
        except Exception as e:
            with lock:
                errors.append((thread_id, str(e)))

    threads = []
    start_time = time.time()

    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    elapsed = time.time() - start_time
    total_ops = num_threads * iterations

    print(f"\n=== Extended Stress Test ===")
    print(f"Threads: {num_threads}")
    print(f"Iterations/thread: {iterations}")
    print(f"Total operations: {total_ops}")
    print(f"Successful: {len(results)}")
    print(f"Errors: {len(errors)}")
    print(f"Elapsed: {elapsed:.3f}s")
    print(f"Throughput: {len(results)/elapsed:.1f} ops/sec")

    if errors:
        print("\nFirst 5 errors:")
        for thread_id, error in errors[:5]:
            print(f"  Thread {thread_id}: {error}")
        assert False, f"Got {len(errors)} errors"

    assert len(results) == total_ops, f"Expected {total_ops}, got {len(results)}"
    print("PASS")


def test_max_threads(max_threads=16, iterations=50):
    """
    Test with maximum threads to find limits.
    """
    import torch

    errors = []
    results = []
    lock = threading.Lock()

    def worker(thread_id: int):
        try:
            for i in range(iterations):
                x = torch.randn(256, 256, device='mps')
                y = torch.randn(256, 256, device='mps')
                z = torch.mm(x, y)
                z = torch.relu(z)
                torch.mps.synchronize()

                with lock:
                    results.append((thread_id, i))
        except Exception as e:
            with lock:
                errors.append((thread_id, str(e)))

    threads = []
    start_time = time.time()

    for i in range(max_threads):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    elapsed = time.time() - start_time
    total_ops = max_threads * iterations

    print(f"\n=== Max Threads Test ({max_threads} threads) ===")
    print(f"Threads: {max_threads}")
    print(f"Iterations/thread: {iterations}")
    print(f"Total operations: {total_ops}")
    print(f"Successful: {len(results)}")
    print(f"Errors: {len(errors)}")
    print(f"Elapsed: {elapsed:.3f}s")
    print(f"Throughput: {len(results)/elapsed:.1f} ops/sec")

    if errors:
        print("\nFirst 5 errors:")
        for thread_id, error in errors[:5]:
            print(f"  Thread {thread_id}: {error}")
        assert False, f"Got {len(errors)} errors"

    assert len(results) == total_ops, f"Expected {total_ops}, got {len(results)}"
    print("PASS")


def test_large_tensors():
    """
    Test with larger tensor sizes.
    """
    import torch

    errors = []
    results = []
    lock = threading.Lock()
    num_threads = 4
    iterations = 20

    def worker(thread_id: int):
        try:
            for i in range(iterations):
                # Larger tensors - 1024x1024
                x = torch.randn(1024, 1024, device='mps')
                y = torch.randn(1024, 1024, device='mps')
                z = torch.mm(x, y)
                z = torch.relu(z)
                torch.mps.synchronize()

                with lock:
                    results.append((thread_id, i))
        except Exception as e:
            with lock:
                errors.append((thread_id, str(e)))

    threads = []
    start_time = time.time()

    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    elapsed = time.time() - start_time
    total_ops = num_threads * iterations

    print(f"\n=== Large Tensor Test (1024x1024) ===")
    print(f"Threads: {num_threads}")
    print(f"Iterations/thread: {iterations}")
    print(f"Total operations: {total_ops}")
    print(f"Successful: {len(results)}")
    print(f"Errors: {len(errors)}")
    print(f"Elapsed: {elapsed:.3f}s")
    print(f"Throughput: {len(results)/elapsed:.1f} ops/sec")

    if errors:
        print("\nFirst 5 errors:")
        for thread_id, error in errors[:5]:
            print(f"  Thread {thread_id}: {error}")
        assert False, f"Got {len(errors)} errors"

    assert len(results) == total_ops, f"Expected {total_ops}, got {len(results)}"
    print("PASS")


def warmup():
    """Warmup MPS with single-threaded operations."""
    import torch

    print("\n--- Warmup: Single-threaded MPS ---")
    x = torch.randn(100, 100, device='mps')
    y = torch.randn(100, 100, device='mps')
    z = torch.mm(x, y)
    torch.mps.synchronize()
    print("Warmup complete")


if __name__ == '__main__':
    print("=" * 60)
    print("MPS Stream Pool Extended Stress Tests")
    print("=" * 60)

    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")

    all_passed = True

    # Warmup first
    warmup()

    # Test 1: Extended stress (8 threads x 100 iterations)
    try:
        test_extended_stress(8, 100)
    except AssertionError as e:
        print(f"FAILED: {e}")
        all_passed = False

    # Test 2: Max threads (16 threads)
    try:
        test_max_threads(16, 50)
    except AssertionError as e:
        print(f"FAILED: {e}")
        all_passed = False

    # Test 3: Large tensors
    try:
        test_large_tensors()
    except AssertionError as e:
        print(f"FAILED: {e}")
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL EXTENDED STRESS TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)
    print("=" * 60)
