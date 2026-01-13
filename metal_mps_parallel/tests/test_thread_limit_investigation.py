#!/usr/bin/env python3
"""
Investigation test for 3+ thread limitation with MPS stream pool.

Worker N=36 - December 13, 2025

According to N=35, the behavior is:
- 2 threads: PASS consistently
- 3+ threads: SIGSEGV intermittently

This test systematically investigates:
1. Raw tensor ops vs nn.Module
2. Number of threads (2, 3, 4, 5, 6, 7, 8)
3. Staggered vs simultaneous thread start
4. Different model complexities
"""

import threading
import time
import sys
import signal
import os


def signal_handler(sig, frame):
    print(f"\nReceived signal {sig}, exiting...")
    sys.exit(1)


signal.signal(signal.SIGTERM, signal_handler)


def warmup_mps():
    """Warmup MPS on main thread."""
    import torch
    x = torch.randn(100, 100, device='mps')
    y = torch.randn(100, 100, device='mps')
    z = torch.mm(x, y)
    torch.mps.synchronize()
    return True


def run_raw_tensor_ops(num_threads: int, iterations: int, barrier: threading.Barrier = None):
    """Test with raw tensor operations (no nn.Module)."""
    import torch

    errors = []
    success_count = [0]  # Use list for mutable counter
    lock = threading.Lock()

    def worker(tid):
        try:
            for i in range(iterations):
                x = torch.randn(64, 64, device='mps')
                y = torch.randn(64, 64, device='mps')
                z = torch.mm(x, y)
                torch.mps.synchronize()
                with lock:
                    success_count[0] += 1
        except Exception as e:
            with lock:
                errors.append((tid, str(e), type(e).__name__))

    threads = []
    if barrier:
        # Use barrier to start all threads simultaneously
        def wrapped_worker(tid):
            barrier.wait()
            worker(tid)

        for i in range(num_threads):
            t = threading.Thread(target=wrapped_worker, args=(i,))
            threads.append(t)
    else:
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)

    start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.time() - start

    expected = num_threads * iterations
    passed = len(errors) == 0 and success_count[0] == expected
    return passed, success_count[0], expected, elapsed, errors


def run_nn_linear_simple(num_threads: int, iterations: int, barrier: threading.Barrier = None):
    """Test with simple nn.Linear (uses MPSGraph)."""
    import torch
    import torch.nn as nn

    errors = []
    success_count = [0]
    lock = threading.Lock()

    def worker(tid):
        try:
            # Each thread creates its own model
            model = nn.Linear(64, 64).to('mps')
            model.eval()

            for i in range(iterations):
                with torch.no_grad():
                    x = torch.randn(4, 64, device='mps')
                    y = model(x)
                    torch.mps.synchronize()
                    with lock:
                        success_count[0] += 1
        except Exception as e:
            with lock:
                errors.append((tid, str(e), type(e).__name__))

    threads = []
    if barrier:
        def wrapped_worker(tid):
            barrier.wait()
            worker(tid)

        for i in range(num_threads):
            t = threading.Thread(target=wrapped_worker, args=(i,))
            threads.append(t)
    else:
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)

    start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.time() - start

    expected = num_threads * iterations
    passed = len(errors) == 0 and success_count[0] == expected
    return passed, success_count[0], expected, elapsed, errors


def run_nn_mlp(num_threads: int, iterations: int, barrier: threading.Barrier = None):
    """Test with 3-layer MLP (more complex MPSGraph)."""
    import torch
    import torch.nn as nn

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
            )

        def forward(self, x):
            return self.layers(x)

    errors = []
    success_count = [0]
    lock = threading.Lock()

    def worker(tid):
        try:
            model = MLP().to('mps')
            model.eval()

            for i in range(iterations):
                with torch.no_grad():
                    x = torch.randn(4, 64, device='mps')
                    y = model(x)
                    torch.mps.synchronize()
                    with lock:
                        success_count[0] += 1
        except Exception as e:
            with lock:
                errors.append((tid, str(e), type(e).__name__))

    threads = []
    if barrier:
        def wrapped_worker(tid):
            barrier.wait()
            worker(tid)

        for i in range(num_threads):
            t = threading.Thread(target=wrapped_worker, args=(i,))
            threads.append(t)
    else:
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)

    start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.time() - start

    expected = num_threads * iterations
    passed = len(errors) == 0 and success_count[0] == expected
    return passed, success_count[0], expected, elapsed, errors


def run_test_suite(test_func, name: str, thread_counts: list, iterations: int, use_barrier: bool):
    """Run a test function with various thread counts."""
    print(f"\n{'='*70}")
    print(f"Testing: {name}")
    print(f"Barrier (simultaneous start): {use_barrier}")
    print(f"Iterations per thread: {iterations}")
    print(f"{'='*70}")

    results = []
    for num_threads in thread_counts:
        barrier = threading.Barrier(num_threads) if use_barrier else None

        print(f"\n  {num_threads} threads: ", end="", flush=True)
        try:
            passed, success, expected, elapsed, errors = test_func(num_threads, iterations, barrier)
            status = "PASS" if passed else "FAIL"
            throughput = success / elapsed if elapsed > 0 else 0
            print(f"{status} ({success}/{expected} ops, {throughput:.0f} ops/s)")
            if errors:
                for tid, err, typ in errors[:3]:
                    print(f"    Thread {tid} ({typ}): {err[:80]}")
            results.append((num_threads, passed, success, expected))
        except Exception as e:
            print(f"CRASH - {type(e).__name__}: {str(e)[:60]}")
            results.append((num_threads, False, 0, num_threads * iterations))

    return results


def test_raw_tensor_ops():
    """Pytest-compatible test for raw tensor ops."""
    warmup_mps()
    passed, success, expected, elapsed, errors = run_raw_tensor_ops(4, 10)
    assert passed, f"Raw tensor ops test failed: {success}/{expected}, errors: {errors}"


def test_nn_linear_simple():
    """Pytest-compatible test for nn.Linear."""
    warmup_mps()
    passed, success, expected, elapsed, errors = run_nn_linear_simple(4, 10)
    assert passed, f"nn.Linear test failed: {success}/{expected}, errors: {errors}"


def test_nn_mlp():
    """Pytest-compatible test for MLP."""
    warmup_mps()
    passed, success, expected, elapsed, errors = run_nn_mlp(4, 10)
    assert passed, f"MLP test failed: {success}/{expected}, errors: {errors}"


if __name__ == '__main__':
    import torch

    print("=" * 70)
    print("MPS Stream Pool - Thread Limit Investigation")
    print("=" * 70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"PID: {os.getpid()}")

    # Warmup
    print("\nWarming up MPS on main thread...")
    warmup_mps()
    print("Warmup complete")

    thread_counts = [2, 3, 4, 5, 6, 7, 8]
    iterations = 10

    # Test 1: Raw tensor ops (no barrier)
    run_test_suite(
        run_raw_tensor_ops,
        "Raw Tensor Ops (matmul) - staggered start",
        thread_counts,
        iterations,
        use_barrier=False
    )

    # Test 2: Raw tensor ops (with barrier - simultaneous)
    run_test_suite(
        run_raw_tensor_ops,
        "Raw Tensor Ops (matmul) - simultaneous start",
        thread_counts,
        iterations,
        use_barrier=True
    )

    # Test 3: nn.Linear (no barrier)
    run_test_suite(
        run_nn_linear_simple,
        "nn.Linear - staggered start",
        thread_counts,
        iterations,
        use_barrier=False
    )

    # Test 4: nn.Linear (with barrier)
    run_test_suite(
        run_nn_linear_simple,
        "nn.Linear - simultaneous start",
        thread_counts,
        iterations,
        use_barrier=True
    )

    # Test 5: MLP (no barrier)
    run_test_suite(
        run_nn_mlp,
        "MLP (3-layer) - staggered start",
        thread_counts,
        iterations,
        use_barrier=False
    )

    # Test 6: MLP (with barrier)
    run_test_suite(
        run_nn_mlp,
        "MLP (3-layer) - simultaneous start",
        thread_counts,
        iterations,
        use_barrier=True
    )

    print("\n" + "=" * 70)
    print("Investigation Complete")
    print("=" * 70)
