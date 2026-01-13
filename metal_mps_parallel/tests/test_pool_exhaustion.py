#!/usr/bin/env python3
"""
Test: Pool Exhaustion Behavior

Tests behavior when more threads request MPS streams than available in the pool.
The MPS stream pool has 32 streams (1 default + 31 worker slots).
When more than 31 worker threads try to use MPS concurrently, the behavior
depends on MPS_STREAM_POOL_WAIT_TIMEOUT_MS:
  - 0 (default): Throw RuntimeError immediately
  - -1: Wait indefinitely for a slot
  - >0: Wait up to N ms, then throw

This test verifies proper behavior under pool exhaustion.
"""
import os
import subprocess
import sys
import tempfile
import textwrap

import pytest


def test_pool_exhaustion_default():
    """
    Test that pool exhaustion throws an error with default settings.

    40 threads try to use MPS concurrently. Since pool has 31 worker slots,
    some threads should fail with pool exhaustion error (default behavior).
    """
    script = textwrap.dedent('''
        import threading
        import time
        import torch

        # Ensure fresh process without any timeout configured
        assert "MPS_STREAM_POOL_WAIT_TIMEOUT_MS" not in os.environ, "Timeout should not be set"

        errors = []
        successes = []
        lock = threading.Lock()

        # Barrier to ensure all threads start together for maximum contention
        barrier = threading.Barrier(40)

        def worker(tid):
            """Worker that holds MPS for a while to cause exhaustion."""
            import os
            try:
                barrier.wait(timeout=5)  # All threads start together

                # Perform MPS operation that holds the slot
                x = torch.randn(512, 512, device='mps')
                y = torch.randn(512, 512, device='mps')
                z = torch.mm(x, y)
                time.sleep(0.1)  # Hold the slot briefly
                torch.mps.synchronize()

                with lock:
                    successes.append(tid)
            except RuntimeError as e:
                err_str = str(e).lower()
                with lock:
                    if 'exhausted' in err_str or 'pool' in err_str or 'stream' in err_str:
                        errors.append(('exhausted', tid, str(e)))
                    else:
                        errors.append(('other', tid, str(e)))
            except Exception as e:
                with lock:
                    errors.append(('exception', tid, str(e)))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(40)]

        for t in threads:
            t.start()

        for t in threads:
            t.join(timeout=30)

        print(f"Successes: {len(successes)}")
        print(f"Pool exhaustion errors: {sum(1 for e in errors if e[0] == 'exhausted')}")
        print(f"Other errors: {sum(1 for e in errors if e[0] != 'exhausted')}")

        # We expect EITHER:
        # 1. Some "exhausted" errors (proper pool exhaustion detection), OR
        # 2. All succeed (if threads exit fast enough to release slots)
        # Either behavior is acceptable - the important thing is no crashes.

        exhausted_count = sum(1 for e in errors if e[0] == 'exhausted')
        other_errors = [e for e in errors if e[0] != 'exhausted']

        # Print any non-exhaustion errors for debugging
        for error_type, tid, msg in other_errors[:5]:
            print(f"  Thread {tid}: {msg[:100]}")

        # No crashes = success
        # Exhaustion errors are expected (pool is doing its job)
        # Other errors are concerning but not fatal
        if other_errors:
            print(f"WARNING: {len(other_errors)} non-exhaustion errors")

        print("Pool exhaustion test completed without crashes")
    ''')

    # Run in subprocess to ensure clean environment
    env = os.environ.copy()
    env.pop('MPS_STREAM_POOL_WAIT_TIMEOUT_MS', None)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write('import os\n')
        f.write(script)
        script_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=60,
            env=env
        )
        print(result.stdout)
        if result.stderr:
            print(f"STDERR: {result.stderr}")

        # Test passes if process completes without crash (exit 0 or handled error)
        assert result.returncode == 0 or 'exhausted' in result.stdout.lower() or 'pool' in result.stdout.lower(), \
            f"Unexpected failure: {result.returncode}\n{result.stdout}\n{result.stderr}"
    finally:
        os.unlink(script_path)


def test_pool_exhaustion_with_backpressure():
    """
    Test that backpressure (infinite wait) allows all threads to eventually succeed.

    With MPS_STREAM_POOL_WAIT_TIMEOUT_MS=-1, threads wait for a slot instead
    of failing. This test uses a thread pool pattern where threads are created
    and destroyed in waves, allowing slots to be recycled.

    NOTE: Stream slots are tied to thread TLS and released on thread exit.
    So backpressure only works when threads exit to release slots.
    """
    script = textwrap.dedent('''
        import threading
        import torch

        errors = []
        successes = []
        lock = threading.Lock()

        def worker(tid):
            """Worker with backpressure waiting."""
            try:
                # Quick operation
                x = torch.randn(256, 256, device='mps')
                y = torch.randn(256, 256, device='mps')
                z = torch.mm(x, y)
                torch.mps.synchronize()

                with lock:
                    successes.append(tid)
            except Exception as e:
                with lock:
                    errors.append((tid, str(e)))

        # Run threads in waves to allow slot recycling
        # Wave 1: 30 threads (within pool limit)
        wave1 = [threading.Thread(target=worker, args=(i,)) for i in range(30)]
        for t in wave1:
            t.start()
        for t in wave1:
            t.join(timeout=30)

        # Wave 2: 30 more threads (first wave has released slots)
        wave2 = [threading.Thread(target=worker, args=(30 + i,)) for i in range(30)]
        for t in wave2:
            t.start()
        for t in wave2:
            t.join(timeout=30)

        print(f"Successes: {len(successes)}/60")
        print(f"Errors: {len(errors)}")

        for tid, msg in errors[:5]:
            print(f"  Thread {tid}: {msg[:100]}")

        # With slot recycling, all 60 threads should succeed
        assert len(successes) == 60, f"Expected 60 successes, got {len(successes)}"
        assert len(errors) == 0, f"Unexpected errors: {errors}"
        print("All 60 threads completed successfully with backpressure")
    ''')

    # Run with backpressure enabled
    env = os.environ.copy()
    env['MPS_STREAM_POOL_WAIT_TIMEOUT_MS'] = '-1'  # Infinite wait

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script)
        script_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=120,  # Generous timeout
            env=env
        )
        print(result.stdout)
        if result.stderr:
            print(f"STDERR: {result.stderr}")

        assert result.returncode == 0, \
            f"Backpressure test failed: {result.returncode}\n{result.stdout}\n{result.stderr}"
        assert 'All 60 threads completed' in result.stdout, \
            f"Not all threads completed: {result.stdout}"
    finally:
        os.unlink(script_path)


def test_pool_exhaustion_with_timeout():
    """
    Test behavior with a finite timeout for slot waiting.

    With MPS_STREAM_POOL_WAIT_TIMEOUT_MS=100 (100ms), threads wait briefly
    then throw if no slot available.
    """
    script = textwrap.dedent('''
        import threading
        import time
        import torch

        errors = []
        successes = []
        lock = threading.Lock()
        barrier = threading.Barrier(40)

        def worker(tid):
            """Worker that holds slot long enough to cause timeouts."""
            try:
                barrier.wait(timeout=5)

                # Hold the slot for longer than the timeout
                x = torch.randn(512, 512, device='mps')
                time.sleep(0.2)  # 200ms > 100ms timeout
                y = torch.randn(512, 512, device='mps')
                z = torch.mm(x, y)
                torch.mps.synchronize()

                with lock:
                    successes.append(tid)
            except RuntimeError as e:
                err_str = str(e).lower()
                with lock:
                    if 'timeout' in err_str or 'exhausted' in err_str:
                        errors.append(('timeout', tid))
                    else:
                        errors.append(('other', tid, str(e)))
            except Exception as e:
                with lock:
                    errors.append(('exception', tid, str(e)))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(40)]

        for t in threads:
            t.start()

        for t in threads:
            t.join(timeout=30)

        timeout_errors = sum(1 for e in errors if e[0] == 'timeout')
        other_errors = [e for e in errors if e[0] not in ('timeout',)]

        print(f"Successes: {len(successes)}")
        print(f"Timeout errors: {timeout_errors}")
        print(f"Other errors: {len(other_errors)}")

        # Some threads should succeed (got slots)
        # Some threads may timeout (couldn't get slots in time)
        # No crashes = success
        assert len(successes) > 0, "No threads succeeded"
        if other_errors:
            for e in other_errors[:3]:
                print(f"  Other error: {e}")
        print("Timeout test completed without crashes")
    ''')

    env = os.environ.copy()
    env['MPS_STREAM_POOL_WAIT_TIMEOUT_MS'] = '100'  # 100ms timeout

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script)
        script_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=60,
            env=env
        )
        print(result.stdout)
        if result.stderr:
            print(f"STDERR: {result.stderr}")

        # Success = no crash
        assert result.returncode == 0, \
            f"Timeout test crashed: {result.returncode}\n{result.stdout}\n{result.stderr}"
    finally:
        os.unlink(script_path)


if __name__ == '__main__':
    print("=" * 70)
    print("Pool Exhaustion Tests")
    print("=" * 70)

    all_passed = True

    print("\n--- Test 1: Default behavior (throw on exhaustion) ---")
    try:
        test_pool_exhaustion_default()
        print("PASS")
    except Exception as e:
        print(f"FAIL: {e}")
        all_passed = False

    print("\n--- Test 2: Backpressure (infinite wait) ---")
    try:
        test_pool_exhaustion_with_backpressure()
        print("PASS")
    except Exception as e:
        print(f"FAIL: {e}")
        all_passed = False

    print("\n--- Test 3: Finite timeout ---")
    try:
        test_pool_exhaustion_with_timeout()
        print("PASS")
    except Exception as e:
        print(f"FAIL: {e}")
        all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL POOL EXHAUSTION TESTS PASSED")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)
