#!/usr/bin/env python3
"""
Test MPS stream pool boundary conditions (Phase 39.4).

Tests:
1. Round-robin wraparound simulation (200 sequential threads)
2. Stream reuse under thread churn (10 rounds x 8 workers)
3. synchronizeAllStreams() during active use
"""
import threading
import time
import sys

import torch
import torch.mps


def test_round_robin_wraparound():
    """Test round-robin wraparound with 200 sequential thread spawns."""
    print("Test: Round-Robin Wraparound Simulation")
    print("=" * 60)

    x = torch.randn(8, 8, device='mps')
    torch.mps.synchronize()

    n_threads = 200
    errors = []
    success_count = 0

    def single_op(tid):
        try:
            x = torch.randn(8, 8, device='mps')
            y = x + x
            torch.mps.synchronize()
            return True, None
        except Exception as e:
            return False, str(e)

    for i in range(n_threads):
        result = [None]
        def run_and_capture(tid=i):
            result[0] = single_op(tid)

        t = threading.Thread(target=run_and_capture)
        t.start()
        t.join(timeout=5.0)

        if result[0] is None:
            errors.append((i, "Timeout"))
        elif result[0][0]:
            success_count += 1
        else:
            errors.append((i, result[0][1]))

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{n_threads} ({success_count} success)")

    print(f"\nResults: {success_count}/{n_threads} successful")
    assert len(errors) == 0, f"Failed with {len(errors)} errors"
    print("PASS: Round-robin wraparound handled correctly")


def test_stream_reuse_under_churn():
    """Test stream slot reuse across 10 rounds of 8 workers each."""
    print("\nTest: Stream Reuse Under Thread Churn")
    print("=" * 60)

    x = torch.randn(8, 8, device='mps')
    torch.mps.synchronize()

    n_rounds = 10
    threads_per_round = 8
    round_errors = []

    def worker(round_id, worker_id):
        try:
            x = torch.randn(32, 32, device='mps')
            y = torch.matmul(x, x)
            torch.mps.synchronize()
            return (round_id, worker_id, None)
        except Exception as e:
            return (round_id, worker_id, str(e))

    for r in range(n_rounds):
        results = []
        lock = threading.Lock()
        threads = []

        for w in range(threads_per_round):
            def run(rid=r, wid=w):
                result = worker(rid, wid)
                with lock:
                    results.append(result)

            t = threading.Thread(target=run)
            threads.append(t)

        for t in threads:
            t.start()

        for t in threads:
            t.join(timeout=10.0)

        errors = [res for res in results if res[2] is not None]
        if errors:
            round_errors.extend(errors)

        print(f"  Round {r + 1}/{n_rounds}: {threads_per_round - len(errors)}/{threads_per_round} OK")

    print(f"\nResults: {n_rounds * threads_per_round} threads, {len(round_errors)} errors")
    assert len(round_errors) == 0, f"Failed with {len(round_errors)} errors"
    print("PASS: Stream slots correctly recycled")


def test_synchronize_all_during_active_use():
    """Test synchronizeAllStreams() while 8 workers are active."""
    print("\nTest: synchronizeAllStreams() During Active Use")
    print("=" * 60)

    x = torch.randn(8, 8, device='mps')
    torch.mps.synchronize()

    n_workers = 8
    iterations_per_worker = 50
    sync_count = 0
    worker_errors = []

    def long_running_worker(tid):
        try:
            count = 0
            for i in range(iterations_per_worker):
                x = torch.randn(64, 64, device='mps')
                y = torch.matmul(x, x)
                torch.mps.synchronize()
                count += 1
            return (tid, None, count)
        except Exception as e:
            return (tid, str(e), 0)

    results = []
    lock = threading.Lock()
    threads = []

    for tid in range(n_workers):
        def run(t=tid):
            result = long_running_worker(t)
            with lock:
                results.append(result)

        t = threading.Thread(target=run)
        threads.append(t)
        t.start()

    time.sleep(0.1)

    try:
        for i in range(10):
            torch.mps.synchronize()
            sync_count += 1
            time.sleep(0.05)
    except Exception as e:
        print(f"  Main thread sync error: {e}")
        worker_errors.append(("main", str(e), 0))

    print(f"  Main thread completed {sync_count} synchronize() calls")

    for t in threads:
        t.join(timeout=30.0)

    for result in results:
        if result[1] is not None:
            worker_errors.append(result)
        else:
            print(f"  Worker {result[0]}: {result[2]} iterations")

    print(f"\nResults: {n_workers} workers, {sync_count} main syncs, {len(worker_errors)} errors")
    assert len(worker_errors) == 0, f"Failed: {worker_errors}"
    print("PASS: synchronizeAllStreams() works during active use")


if __name__ == '__main__':
    print("=" * 70)
    print("MPS Stream Pool Boundary Tests (Phase 39.4)")
    print("=" * 70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    test_round_robin_wraparound()
    test_stream_reuse_under_churn()
    test_synchronize_all_during_active_use()
    print("\nALL TESTS PASSED")
