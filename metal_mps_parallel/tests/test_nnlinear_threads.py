#!/usr/bin/env python3
"""
Focused test: nn.Linear thread limit investigation.
Run multiple trials at each thread count to determine reliability.
"""

import threading
import time
import sys
import os


def warmup_mps():
    import torch
    x = torch.randn(100, 100, device='mps')
    y = torch.randn(100, 100, device='mps')
    z = torch.mm(x, y)
    torch.mps.synchronize()


def run_nn_linear(num_threads: int, iterations: int):
    """Test nn.Linear with given thread count. Returns (passed, ops_completed)."""
    import torch
    import torch.nn as nn

    errors = []
    success_count = [0]
    lock = threading.Lock()

    def worker(tid):
        try:
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
                errors.append((tid, str(e)))

    threads = []
    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    expected = num_threads * iterations
    passed = len(errors) == 0 and success_count[0] == expected
    return passed, success_count[0], expected, errors


def test_nn_linear():
    """Pytest-compatible test for nn.Linear with parallel threads."""
    warmup_mps()
    passed, success, expected, errors = run_nn_linear(4, 10)
    assert passed, f"nn.Linear test failed: {success}/{expected}, errors: {errors}"


if __name__ == '__main__':
    import torch

    print("=" * 60)
    print("nn.Linear Thread Limit Test")
    print("=" * 60)
    print(f"PyTorch: {torch.__version__}")
    print(f"PID: {os.getpid()}")

    warmup_mps()
    print("Warmup complete\n")

    # Test specific thread count from command line, or run all
    if len(sys.argv) > 1:
        num_threads = int(sys.argv[1])
        iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 10

        print(f"Testing {num_threads} threads, {iterations} iterations...")
        passed, success, expected, errors = run_nn_linear(num_threads, iterations)
        status = "PASS" if passed else "FAIL"
        print(f"Result: {status} ({success}/{expected})")
        if errors:
            for tid, err in errors:
                print(f"  Thread {tid}: {err}")
        sys.exit(0 if passed else 1)

    else:
        # Run multiple trials for each thread count
        trials = 3
        iterations = 10

        for num_threads in [2, 3, 4, 5]:
            print(f"\n{num_threads} threads: ", end="", flush=True)
            results = []
            for trial in range(trials):
                passed, success, expected, errors = run_nn_linear(num_threads, iterations)
                results.append(passed)
                print("." if passed else "X", end="", flush=True)

            pass_rate = sum(results) / len(results) * 100
            print(f" {pass_rate:.0f}% ({sum(results)}/{len(results)})")
