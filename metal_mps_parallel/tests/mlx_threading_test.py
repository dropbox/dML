#!/usr/bin/env python3
"""
MLX Threading Test - Phase 6.1 Analysis

Worker: N=1474
Purpose: Understand how MLX handles multi-threaded GPU operations
         and compare with our MPS implementation.

Questions to answer:
1. How does MLX handle multi-threading?
2. What synchronization does MLX use?
3. Does MLX crash under concurrent load?
4. What throughput does MLX achieve?

Usage:
  python3 tests/mlx_threading_test.py
"""

import threading
import time
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("MLX not available")


def mlx_worker(thread_id: int, num_ops: int, results_queue: queue.Queue):
    """Perform MLX operations in a thread."""
    try:
        success_count = 0
        error_count = 0
        start = time.perf_counter()

        for i in range(num_ops):
            try:
                # Create arrays on GPU
                a = mx.random.normal((256, 256))
                b = mx.random.normal((256, 256))

                # Matrix multiply
                c = mx.matmul(a, b)

                # Force evaluation
                mx.eval(c)

                success_count += 1
            except Exception as e:
                error_count += 1
                if error_count <= 3:
                    print(f"Thread {thread_id} error on op {i}: {e}")

        elapsed = time.perf_counter() - start
        results_queue.put({
            'thread_id': thread_id,
            'success': success_count,
            'errors': error_count,
            'elapsed': elapsed,
            'ops_per_sec': success_count / elapsed if elapsed > 0 else 0
        })
    except Exception as e:
        results_queue.put({
            'thread_id': thread_id,
            'success': 0,
            'errors': num_ops,
            'elapsed': 0,
            'error': str(e),
            'traceback': traceback.format_exc()
        })


def test_mlx_threading(num_threads: int, ops_per_thread: int = 50):
    """Test MLX with multiple threads."""
    print(f"\n=== MLX Threading Test: {num_threads} threads, {ops_per_thread} ops/thread ===")

    results_queue = queue.Queue()
    threads = []

    start = time.perf_counter()

    for i in range(num_threads):
        t = threading.Thread(target=mlx_worker, args=(i, ops_per_thread, results_queue))
        threads.append(t)
        t.start()

    # Wait for all threads
    for t in threads:
        t.join(timeout=60)

    elapsed = time.perf_counter() - start

    # Collect results
    results = []
    while not results_queue.empty():
        results.append(results_queue.get())

    total_success = sum(r.get('success', 0) for r in results)
    total_errors = sum(r.get('errors', 0) for r in results)
    total_ops = num_threads * ops_per_thread

    print(f"Results:")
    print(f"  Total ops: {total_ops}")
    print(f"  Successful: {total_success}")
    print(f"  Errors: {total_errors}")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Throughput: {total_success/elapsed:.1f} ops/s")
    print(f"  Success rate: {100*total_success/total_ops:.1f}%")

    # Report any errors
    for r in results:
        if 'error' in r:
            print(f"\n  Thread {r['thread_id']} fatal error: {r['error']}")

    return {
        'threads': num_threads,
        'ops_per_thread': ops_per_thread,
        'total_success': total_success,
        'total_errors': total_errors,
        'elapsed': elapsed,
        'throughput': total_success / elapsed if elapsed > 0 else 0,
        'success_rate': total_success / total_ops if total_ops > 0 else 0
    }


def main():
    if not MLX_AVAILABLE:
        print("MLX not available, skipping test")
        return

    print("=" * 60)
    print("MLX Threading Analysis - Phase 6.1")
    print("=" * 60)

    # Device info
    device = mx.default_device()
    print(f"\nDevice: {device}")

    # Single-threaded baseline
    print("\n--- Single-threaded baseline ---")
    baseline = test_mlx_threading(1, 100)

    # Scaling tests
    print("\n--- Threading scaling tests ---")
    results = [baseline]

    for threads in [2, 4, 8]:
        try:
            r = test_mlx_threading(threads, 50)
            results.append(r)
        except Exception as e:
            print(f"\n{threads} threads CRASHED: {e}")
            traceback.print_exc()
            results.append({
                'threads': threads,
                'crashed': True,
                'error': str(e)
            })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY - MLX Threading Behavior")
    print("=" * 60)
    print(f"{'Threads':>8} {'Success':>10} {'Errors':>8} {'Throughput':>12} {'Notes'}")
    print("-" * 60)

    for r in results:
        if r.get('crashed'):
            print(f"{r['threads']:>8} {'CRASHED':>10} {'-':>8} {'-':>12} {r.get('error', '')[:30]}")
        else:
            notes = ""
            if r['total_errors'] > 0:
                notes = f"{r['total_errors']} errors"
            print(f"{r['threads']:>8} {r['total_success']:>10} {r['total_errors']:>8} {r['throughput']:>10.1f}/s {notes}")

    print("\n" + "=" * 60)
    print("Analysis for Phase 6.1 report")
    print("=" * 60)


if __name__ == "__main__":
    main()
