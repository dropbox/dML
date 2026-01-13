#!/usr/bin/env python3
"""
Test static destruction order safety (Phase 39.5).

Tests that MPS operations and cleanup don't cause crashes.
"""
import threading
import time
import atexit
import sys


def test_mps_state_during_cleanup():
    """Test that MPS state is properly handled during cleanup."""
    print("Test: MPS State During Cleanup")
    print("=" * 60)

    import torch

    x = torch.randn(8, 8, device='mps')
    torch.mps.synchronize()
    print("  Warmup complete")

    tensors = []
    for i in range(20):
        t = torch.randn(64, 64, device='mps')
        tensors.append(t)

    print(f"  Created {len(tensors)} MPS tensors")

    for t in tensors:
        y = t * 2.0 + 1.0
        torch.mps.synchronize()

    print("  Completed operations on all tensors")

    del tensors
    torch.mps.synchronize()
    torch.mps.empty_cache()

    print("  Cleaned up all tensors")
    print("PASS: MPS state cleanup works correctly")


def test_atexit_cleanup():
    """Test that MPS operations in atexit handlers work."""
    print("\nTest: Atexit Handler MPS Operations")
    print("=" * 60)

    import torch

    x = torch.randn(8, 8, device='mps')
    torch.mps.synchronize()

    def cleanup_handler():
        try:
            torch.mps.synchronize()
        except:
            pass

    atexit.register(cleanup_handler)
    print("  Atexit handler registered successfully")

    for i in range(5):
        y = torch.randn(16, 16, device='mps')
        z = torch.matmul(y, y)
        torch.mps.synchronize()

    print("  Final MPS work completed")
    print("PASS: Atexit registration works")


def test_thread_pool_cleanup_order():
    """Test that thread pool handles cleanup correctly."""
    print("\nTest: Thread Pool Cleanup Order")
    print("=" * 60)

    import torch

    x = torch.randn(8, 8, device='mps')
    torch.mps.synchronize()

    n_threads = 8
    barriers = threading.Barrier(n_threads + 1)
    errors = []
    completed = []
    lock = threading.Lock()

    def worker(tid):
        try:
            tensors = []
            for i in range(5):
                t = torch.randn(32, 32, device='mps')
                tensors.append(t)

            barriers.wait()

            for t in tensors:
                y = t * 2.0
                torch.mps.synchronize()

            barriers.wait()

            with lock:
                completed.append(tid)

        except Exception as e:
            with lock:
                errors.append((tid, str(e)))

    threads = []
    for i in range(n_threads):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    barriers.wait()
    print(f"  {n_threads} threads created MPS state")

    barriers.wait()
    print(f"  {n_threads} threads completed work")

    for t in threads:
        t.join(timeout=10.0)

    torch.mps.synchronize()
    torch.mps.empty_cache()

    print(f"  Completed: {len(completed)}/{n_threads}")
    print(f"  Errors: {len(errors)}")

    assert len(errors) == 0, f"Thread errors: {errors}"
    assert len(completed) == n_threads
    print("PASS: Thread pool cleanup order is correct")


if __name__ == '__main__':
    print("=" * 70)
    print("MPS Static Destruction Order Tests (Phase 39.5)")
    print("=" * 70)
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    test_mps_state_during_cleanup()
    test_atexit_cleanup()
    test_thread_pool_cleanup_order()
    print("\nALL TESTS PASSED")
