#!/usr/bin/env python3
"""
Test: Compare multi-process vs multi-thread parallelism.

If multi-process works but multi-thread fails, it confirms the issue
is with Apple's MPS framework's per-process threading limitations.
"""

import subprocess
import sys
import time
import os


def run_multiprocess(num_processes: int, iterations: int):
    """Run multiple processes in parallel, each doing inference."""
    script = '''
import torch
import torch.nn as nn

# Single-thread inference test
model = nn.Linear(64, 64).to('mps')
model.eval()

for i in range({iterations}):
    with torch.no_grad():
        x = torch.randn(4, 64, device='mps')
        y = model(x)
        torch.mps.synchronize()

print("OK")
'''.format(iterations=iterations)

    processes = []
    start = time.time()

    for i in range(num_processes):
        p = subprocess.Popen(
            [sys.executable, '-c', script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ}
        )
        processes.append(p)

    # Wait for all processes
    results = []
    for p in processes:
        stdout, stderr = p.communicate()
        results.append((p.returncode, stdout.decode(), stderr.decode()))

    elapsed = time.time() - start

    passed = all(rc == 0 for rc, _, _ in results)
    successful = sum(1 for rc, _, _ in results if rc == 0)

    return passed, successful, num_processes, elapsed


def run_multithread(num_threads: int, iterations: int):
    """Run multiple threads in parallel, each doing inference."""
    import threading
    import torch
    import torch.nn as nn

    errors = []
    success_count = [0]
    lock = threading.Lock()

    def worker():
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
                errors.append(str(e))

    threads = []
    start = time.time()

    for i in range(num_threads):
        t = threading.Thread(target=worker)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    elapsed = time.time() - start

    expected = num_threads * iterations
    passed = len(errors) == 0 and success_count[0] == expected
    return passed, success_count[0], expected, elapsed


def test_multiprocess():
    """Pytest-compatible test for multi-process parallelism."""
    passed, successful, total, elapsed = run_multiprocess(4, 10)
    assert passed, f"Multi-process test failed: {successful}/{total}"


def test_multithread():
    """Pytest-compatible test for multi-thread parallelism."""
    import torch
    # Warmup
    x = torch.randn(10, 10, device='mps')
    torch.mps.synchronize()

    passed, successful, expected, elapsed = run_multithread(4, 10)
    assert passed, f"Multi-thread test failed: {successful}/{expected}"


if __name__ == '__main__':
    import torch
    print("=" * 60)
    print("Multi-Process vs Multi-Thread Comparison")
    print("=" * 60)
    print(f"PyTorch: {torch.__version__}")
    print()

    # Warmup
    x = torch.randn(10, 10, device='mps')
    y = torch.mm(x, x)
    torch.mps.synchronize()
    print("Warmup complete\n")

    iterations = 10

    # Test multi-process
    print("Multi-Process Tests (each process = 1 thread):")
    for num_procs in [2, 4, 6, 8]:
        passed, successful, total, elapsed = run_multiprocess(num_procs, iterations)
        status = "PASS" if passed else "FAIL"
        print(f"  {num_procs} processes: {status} ({successful}/{total} procs OK, {elapsed:.2f}s)")

    print()

    # Test multi-thread
    print("Multi-Thread Tests (all threads in same process):")
    for num_threads in [2, 3, 4]:
        try:
            passed, successful, expected, elapsed = run_multithread(num_threads, iterations)
            status = "PASS" if passed else "FAIL"
            print(f"  {num_threads} threads: {status} ({successful}/{expected} ops, {elapsed:.2f}s)")
        except Exception as e:
            print(f"  {num_threads} threads: CRASH - {type(e).__name__}: {e}")
            break
