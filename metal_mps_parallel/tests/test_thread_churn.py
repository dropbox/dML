#!/usr/bin/env python3
"""
Test MPS stream pool thread churn behavior.

Test scenarios:
1. Sequential thread churn: 50 threads spawned one-at-a-time
2. Batch thread churn: 4 batches of 20 threads (80 total, but max 20 concurrent)
"""
import threading
import time
import sys


def test_sequential_thread_churn():
    """
    Spawn threads sequentially (one at a time, each exits before next starts).
    Should succeed even when total thread count exceeds pool size.
    """
    import torch

    print("Test: Sequential Thread Churn (50 threads, one at a time)")
    print("=" * 60)

    # Warmup on main thread
    x = torch.randn(10, 10, device='mps')
    torch.mps.synchronize()
    print("Main thread warmed up (stream 0)")

    errors = []
    successes = []

    def single_mps_op(tid):
        """Single thread that does MPS work and exits."""
        try:
            x = torch.randn(16, 16, device='mps')
            y = torch.matmul(x, x)
            torch.mps.synchronize()
            return (tid, None)
        except Exception as e:
            return (tid, str(e))

    # Spawn 50 threads sequentially (more than the 31 pooled worker streams)
    n_threads = 50
    for i in range(n_threads):
        t = threading.Thread(target=lambda tid=i: single_mps_op(tid))
        t.start()
        t.join()  # Wait for thread to complete before starting next

        # Check result by spawning another test thread
        result = [None]
        def check_op():
            try:
                x = torch.randn(8, 8, device='mps')
                torch.mps.synchronize()
                result[0] = True
            except Exception as e:
                result[0] = str(e)

        check = threading.Thread(target=check_op)
        check.start()
        check.join()

        if result[0] is True:
            successes.append(i)
        else:
            errors.append((i, result[0]))
            break  # Stop on first failure

        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{n_threads} sequential threads")

    print(f"\nResults:")
    print(f"  Successful sequential threads: {len(successes)}/{n_threads}")
    print(f"  Errors: {len(errors)}")

    if len(errors) == 0:
        print("\nPASS: Sequential thread churn succeeded")
    else:
        assert False, f"Thread {errors[0][0]} failed: {errors[0][1]}"


def test_batch_thread_churn():
    """
    Spawn threads in batches. Each batch uses workers, they all exit,
    then next batch starts. Exercises thread churn with moderate concurrency.

    4 batches x 20 workers = 80 total threads, but max 20 concurrent.
    """
    import torch

    print("\nTest: Batch Thread Churn (4 batches x 20 workers = 80 total)")
    print("=" * 60)

    # Warmup on main thread
    x = torch.randn(10, 10, device='mps')
    torch.mps.synchronize()

    batch_results = []

    def worker_batch(batch_id, n_workers):
        """Run a batch of workers, wait for all to complete."""
        results = []
        lock = threading.Lock()
        barrier = threading.Barrier(n_workers + 1)

        def worker(wid):
            try:
                barrier.wait()
                x = torch.randn(32, 32, device='mps')
                y = torch.matmul(x, x)
                torch.mps.synchronize()
                with lock:
                    results.append((wid, None))
            except Exception as e:
                with lock:
                    results.append((wid, str(e)))

        threads = []
        for i in range(n_workers):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        barrier.wait()  # Coordinator signals start

        for t in threads:
            t.join()

        return results

    n_batches = 4
    workers_per_batch = 20  # Well under 31 concurrent limit

    for batch in range(n_batches):
        results = worker_batch(batch, workers_per_batch)
        errors = [r for r in results if r[1] is not None]
        successes = [r for r in results if r[1] is None]

        print(f"  Batch {batch + 1}: {len(successes)}/{workers_per_batch} succeeded")

        if errors:
            print(f"    First error: {errors[0]}")
            batch_results.append((batch, False, errors[0]))
        else:
            batch_results.append((batch, True, None))

    all_passed = all(r[1] for r in batch_results)

    print(f"\nResults:")
    print(f"  Batches passed: {sum(1 for r in batch_results if r[1])}/{n_batches}")
    print(f"  Total threads used: {n_batches * workers_per_batch}")

    if all_passed:
        print("\nPASS: Batch thread churn works - slots properly recycled between batches")
    else:
        failed = [r for r in batch_results if not r[1]]
        assert False, f"Batch {failed[0][0]} failed: {failed[0][2]}"


if __name__ == '__main__':
    import torch
    print("=" * 70)
    print("MPS Stream Pool Thread Churn Test")
    print("=" * 70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print("")

    all_passed = True

    # Test 1: Sequential thread churn
    print("-" * 70)
    try:
        test_sequential_thread_churn()
    except AssertionError as e:
        print(f"FAILED: {e}")
        all_passed = False

    # Test 2: Batch thread churn
    print("-" * 70)
    try:
        test_batch_thread_churn()
    except AssertionError as e:
        print(f"FAILED: {e}")
        all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL THREAD CHURN TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)
    print("=" * 70)
