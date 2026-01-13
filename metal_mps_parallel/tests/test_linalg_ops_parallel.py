#!/usr/bin/env python3
"""
Test script for MPS parallel execution of linear algebra operations.

These operations have mutex protection added in Phase 23 to prevent crashes
when executed concurrently on multiple threads:
- torch.bmm (batch matrix multiply) - Phase 23.1
- torch.linalg.solve_triangular - Phase 23.5
- Strided tensor views (via MPSNDArrayIdentity) - Phase 23.13
"""
import threading
import time
import sys

def test_bmm_parallel(num_threads=4, iterations=20):
    """
    Test torch.bmm (batch matrix multiplication) with multiple threads.

    This operation uses MPSNDArrayMatrixMultiplication which has internal
    shared state. Phase 23.1 added s_bmm_tiled_mutex to protect it.
    """
    import torch

    errors = []
    results = []
    lock = threading.Lock()

    def worker(thread_id: int):
        try:
            for i in range(iterations):
                # Batch matrix multiplication
                a = torch.randn(32, 64, 128, device='mps')  # [batch, M, K]
                b = torch.randn(32, 128, 64, device='mps')  # [batch, K, N]
                c = torch.bmm(a, b)  # [batch, M, N]
                torch.mps.synchronize()

                # Verify shape
                assert c.shape == (32, 64, 64), f"Expected (32, 64, 64), got {c.shape}"

                with lock:
                    results.append((thread_id, i, c.shape))
        except Exception as e:
            with lock:
                errors.append((thread_id, str(e), type(e).__name__))

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

    print(f"\n=== torch.bmm Parallel Test ===")
    print(f"Threads: {num_threads}")
    print(f"Iterations/thread: {iterations}")
    print(f"Total operations: {total_ops}")
    print(f"Successful: {len(results)}")
    print(f"Errors: {len(errors)}")
    print(f"Elapsed: {elapsed:.3f}s")

    if errors:
        print("\nErrors:")
        for thread_id, error, error_type in errors[:5]:
            print(f"  Thread {thread_id} ({error_type}): {error[:100]}")
        assert False, f"Got {len(errors)} errors"

    assert len(results) == total_ops, "Some operations did not complete"
    print("PASS")


def test_strided_views_parallel(num_threads=4, iterations=20):
    """
    Test strided tensor operations with multiple threads.

    This exercises MPSNDArrayIdentity which has mutex protection
    added in Phase 23.13.
    """
    import torch

    errors = []
    results = []
    lock = threading.Lock()

    def worker(thread_id: int):
        try:
            for i in range(iterations):
                # Create tensor with non-contiguous strides
                x = torch.randn(64, 128, device='mps')
                y = x.t()  # Transpose creates strided view
                z = torch.mm(y, x)  # Force computation with strided input
                torch.mps.synchronize()

                # Verify shape
                assert z.shape == (128, 128), f"Expected (128, 128), got {z.shape}"

                with lock:
                    results.append((thread_id, i, z.shape))
        except Exception as e:
            with lock:
                errors.append((thread_id, str(e), type(e).__name__))

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

    print(f"\n=== Strided Views Parallel Test ===")
    print(f"Threads: {num_threads}")
    print(f"Iterations/thread: {iterations}")
    print(f"Total operations: {total_ops}")
    print(f"Successful: {len(results)}")
    print(f"Errors: {len(errors)}")
    print(f"Elapsed: {elapsed:.3f}s")

    if errors:
        print("\nErrors:")
        for thread_id, error, error_type in errors[:5]:
            print(f"  Thread {thread_id} ({error_type}): {error[:100]}")
        assert False, f"Got {len(errors)} errors"

    assert len(results) == total_ops, "Some operations did not complete"
    print("PASS")


def test_matmul_parallel(num_threads=4, iterations=20):
    """
    Test torch.matmul with multiple threads.

    This operation uses the graph path (thread-local MPSGraphCache)
    and should be inherently thread-safe.
    """
    import torch

    errors = []
    results = []
    lock = threading.Lock()

    def worker(thread_id: int):
        try:
            for i in range(iterations):
                a = torch.randn(64, 128, device='mps')
                b = torch.randn(128, 64, device='mps')
                c = torch.matmul(a, b)
                torch.mps.synchronize()

                assert c.shape == (64, 64), f"Expected (64, 64), got {c.shape}"

                with lock:
                    results.append((thread_id, i, c.shape))
        except Exception as e:
            with lock:
                errors.append((thread_id, str(e), type(e).__name__))

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

    print(f"\n=== torch.matmul Parallel Test ===")
    print(f"Threads: {num_threads}")
    print(f"Iterations/thread: {iterations}")
    print(f"Total operations: {total_ops}")
    print(f"Successful: {len(results)}")
    print(f"Errors: {len(errors)}")
    print(f"Elapsed: {elapsed:.3f}s")

    if errors:
        print("\nErrors:")
        for thread_id, error, error_type in errors[:5]:
            print(f"  Thread {thread_id} ({error_type}): {error[:100]}")
        assert False, f"Got {len(errors)} errors"

    assert len(results) == total_ops, "Some operations did not complete"
    print("PASS")


def warmup():
    """Warmup MPS with single-threaded operations."""
    import torch

    print("\n--- Warmup: Single-threaded MPS ---")
    x = torch.randn(64, 64, device='mps')
    y = torch.bmm(x.unsqueeze(0), x.unsqueeze(0))
    z = x.t().contiguous()
    torch.mps.synchronize()
    print("Warmup complete")


def main():
    import torch

    if not torch.backends.mps.is_available():
        print("MPS not available, skipping tests")
        return 1

    warmup()

    all_passed = True

    # Test with 2-4 threads (safe range)
    # Note: Higher thread counts may trigger Apple MPS framework limitations
    for num_threads in [2, 4]:
        print(f"\n{'='*50}")
        print(f"Testing with {num_threads} threads")
        print(f"{'='*50}")

        try:
            test_bmm_parallel(num_threads=num_threads, iterations=20)
        except AssertionError as e:
            print(f"FAILED: {e}")
            all_passed = False

        try:
            test_strided_views_parallel(num_threads=num_threads, iterations=20)
        except AssertionError as e:
            print(f"FAILED: {e}")
            all_passed = False

        try:
            test_matmul_parallel(num_threads=num_threads, iterations=20)
        except AssertionError as e:
            print(f"FAILED: {e}")
            all_passed = False

    print("\n" + "="*50)
    if all_passed:
        print("ALL TESTS PASSED")
        return 0
    else:
        print("SOME TESTS FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
