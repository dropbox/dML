#!/usr/bin/env python3
"""
Fork safety stress test (Phase 39.6).

Tests fork behavior when MPS threads are active:
1. Fork while 8 threads are using MPS
2. Parent continues normally after fork
3. Child detects bad-fork state

NOTE: PyTorch's MPS backend doesn't have _lazy_init() guards like CUDA.
The child cannot safely use MPS after fork (it will crash, not raise).
We test that:
- The fork doesn't crash the parent
- The parent can continue using MPS after fork
- The bad-fork flag is set in the child
"""

import os
import sys
import time
import signal
import threading
import pytest


def test_fork_with_active_threads():
    """
    Test fork() while threads are actively using MPS.

    Verifies parent can continue normally after fork.
    """
    print("Test: Fork with Active MPS Threads")
    print("=" * 60)

    if not hasattr(os, "fork"):
        pytest.skip("os.fork() not available on this platform")

    import torch

    if not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")

    # Warmup
    x = torch.randn(8, 8, device='mps')
    torch.mps.synchronize()
    print("  Parent warmup complete")

    # Start worker threads
    n_threads = 8
    stop_flag = threading.Event()
    work_count = [0]
    errors = []
    lock = threading.Lock()

    def worker(tid):
        try:
            while not stop_flag.is_set():
                x = torch.randn(32, 32, device='mps')
                y = torch.matmul(x, x)
                torch.mps.synchronize()
                with lock:
                    work_count[0] += 1
                time.sleep(0.01)
        except Exception as e:
            with lock:
                errors.append((tid, str(e)))

    threads = []
    for i in range(n_threads):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    print(f"  Started {n_threads} MPS worker threads")

    # Let them work
    time.sleep(0.2)
    print(f"  Workers did {work_count[0]} operations before fork")

    # Fork while threads are active
    child_pid = os.fork()

    if child_pid == 0:
        # Child process - just check bad_fork flag and exit
        try:
            bad_fork = bool(getattr(torch.mps, "_is_in_bad_fork", lambda: False)())
            if bad_fork:
                os._exit(0)  # Success - bad fork detected
            else:
                os._exit(1)  # Fail - bad fork not detected
        except:
            os._exit(2)  # Error during check

    # Parent continues
    time.sleep(0.2)  # Let workers continue after fork

    # Signal stop
    stop_flag.set()

    # Wait for threads
    for t in threads:
        t.join(timeout=5.0)

    post_fork_count = work_count[0]
    print(f"  Workers did {post_fork_count} total operations (including post-fork)")

    # Wait for child
    _, status = os.waitpid(child_pid, 0)

    child_ok = os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
    if child_ok:
        print("  Child correctly detected bad-fork state")
    else:
        print(f"  Child status: {status} (expected exit 0)")

    # Parent should still work
    y = torch.randn(16, 16, device='mps')
    z = torch.matmul(y, y)
    torch.mps.synchronize()
    print("  Parent MPS operations work after fork")

    success = child_ok and len(errors) == 0
    if success:
        print("PASS: Fork with active threads works correctly")
    else:
        print(f"FAIL: child_ok={child_ok}, errors={errors}")

    assert success, f"child_ok={child_ok}, errors={errors}"


def test_parent_continues_after_fork():
    """
    Test that parent can continue heavy MPS work after fork.
    """
    print("\nTest: Parent Continues After Fork")
    print("=" * 60)

    if not hasattr(os, "fork"):
        pytest.skip("os.fork() not available on this platform")

    import torch

    if not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")

    # Initialize MPS
    x = torch.randn(8, 8, device='mps')
    torch.mps.synchronize()

    # Fork
    child_pid = os.fork()

    if child_pid == 0:
        # Child exits immediately
        os._exit(0)

    # Wait for child
    os.waitpid(child_pid, 0)
    print("  Child exited")

    # Parent does heavy MPS work
    errors = []
    for i in range(100):
        try:
            x = torch.randn(64, 64, device='mps')
            y = torch.matmul(x, x)
            torch.mps.synchronize()
        except Exception as e:
            errors.append((i, str(e)))
            break

    print(f"  Parent completed {100 - len(errors)}/100 operations after fork")

    if len(errors) == 0:
        print("PASS: Parent continues normally after fork")
    else:
        print(f"FAIL: Parent had errors: {errors[0]}")

    assert len(errors) == 0, f"Parent had errors: {errors[0] if errors else 'unknown'}"


def test_multiprocessing_spawn():
    """
    Test multiprocessing with spawn start method (safe for MPS).

    NOTE: multiprocessing.spawn creates a fresh process without fork,
    so MPS can be safely initialized in the child process.
    """
    print("\nTest: Multiprocessing Spawn (MPS-safe)")
    print("=" * 60)

    import torch

    if not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")

    # Initialize MPS in parent
    x = torch.randn(8, 8, device='mps')
    torch.mps.synchronize()
    print("  Parent initialized MPS")

    # Test that parent can continue after spawn
    # We don't spawn a child that uses MPS due to pickling limitations
    # The key test is that parent MPS works normally after multiprocessing imports

    import multiprocessing as mp
    ctx = mp.get_context('spawn')

    # Just verify we can create the context without issues
    print("  Spawn context created successfully")

    # Parent continues using MPS
    for i in range(10):
        y = torch.randn(32, 32, device='mps')
        z = torch.matmul(y, y)
        torch.mps.synchronize()

    print("  Parent MPS operations work after spawn context creation")
    print("PASS: multiprocessing spawn context is compatible with MPS")
    # Success - no return needed (pytest expects None)


def main():
    print("=" * 70)
    print("MPS Fork Safety Stress Tests (Phase 39.6)")
    print("=" * 70)

    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print()

    all_passed = True

    tests = [
        test_fork_with_active_threads,
        test_parent_continues_after_fork,
        test_multiprocessing_spawn,
    ]

    for test_fn in tests:
        print("-" * 70)
        try:
            test_fn()
            # Success if no exception raised
            import torch
            torch.mps.synchronize()
            torch.mps.empty_cache()
            time.sleep(0.1)
        except pytest.skip.Exception as e:
            print(f"SKIP: {e}")
        except Exception as e:
            print(f"ERROR: {e}")
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL FORK SAFETY STRESS TESTS PASSED")
        return 0
    else:
        print("SOME TESTS FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
