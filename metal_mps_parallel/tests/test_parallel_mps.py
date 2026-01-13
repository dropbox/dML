#!/usr/bin/env python3
"""
Test script for MPS Stream Pool parallel inference.
Tests that multiple threads can run MPS operations concurrently without crashes.
"""
import threading
import time
import sys

def test_basic_mps():
    """Test basic MPS tensor operations."""
    import torch

    assert torch.backends.mps.is_available(), "MPS not available"

    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")

    # Simple tensor operation
    x = torch.randn(100, 100, device='mps')
    y = torch.randn(100, 100, device='mps')
    z = torch.mm(x, y)
    torch.mps.synchronize()

    print(f"Basic MPS test passed. Result shape: {z.shape}")


def test_parallel_inference(num_threads=4, iterations=10):
    """
    Test concurrent MPS operations from multiple threads.
    This is the scenario that crashes with the singleton MPSStream.
    """
    import torch

    assert torch.backends.mps.is_available(), "MPS not available"

    errors = []
    results = []
    lock = threading.Lock()

    def worker(thread_id: int):
        try:
            for i in range(iterations):
                # Each iteration: create tensors and do computation
                x = torch.randn(256, 256, device='mps')
                y = torch.randn(256, 256, device='mps')

                # Matrix multiplication - exercises MPS compute
                z = torch.mm(x, y)

                # Activation function
                z = torch.relu(z)

                # Synchronize to ensure computation completes
                torch.mps.synchronize()

                with lock:
                    results.append((thread_id, i, z.shape))
        except Exception as e:
            with lock:
                errors.append((thread_id, str(e)))

    threads = []
    start_time = time.time()

    # Launch threads
    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    # Wait for all threads
    for t in threads:
        t.join()

    elapsed = time.time() - start_time

    # Report results
    print(f"\n=== Parallel MPS Test Results ===")
    print(f"Threads: {num_threads}")
    print(f"Iterations per thread: {iterations}")
    print(f"Total operations: {num_threads * iterations}")
    print(f"Successful operations: {len(results)}")
    print(f"Errors: {len(errors)}")
    print(f"Elapsed time: {elapsed:.2f}s")
    print(f"Throughput: {len(results)/elapsed:.1f} ops/sec")

    if errors:
        print("\n=== Errors ===")
        for thread_id, error in errors:
            print(f"  Thread {thread_id}: {error}")
        assert False, f"Got {len(errors)} errors"

    expected = num_threads * iterations
    assert len(results) == expected, f"Expected {expected} results, got {len(results)}"
    print(f"\nPASS: All {expected} operations completed successfully!")


def test_stress(num_threads=8, iterations=100):
    """Stress test with more threads and iterations."""
    test_parallel_inference(num_threads, iterations)


if __name__ == '__main__':
    print("=" * 60)
    print("MPS Stream Pool Parallel Inference Tests")
    print("=" * 60)

    # Test 1: Basic MPS
    print("\n--- Test 1: Basic MPS ---")
    try:
        test_basic_mps()
    except AssertionError as e:
        print(f"FAILED: {e}")
        sys.exit(1)

    # Test 2: 4 threads, 10 iterations each
    print("\n--- Test 2: Parallel (4 threads x 10 iterations) ---")
    try:
        test_parallel_inference(4, 10)
    except AssertionError as e:
        print(f"FAILED: {e}")
        sys.exit(1)

    # Test 3: 8 threads, 50 iterations (stress test)
    print("\n--- Test 3: Stress Test (8 threads x 50 iterations) ---")
    try:
        test_stress(8, 50)
    except AssertionError as e:
        print(f"FAILED: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
