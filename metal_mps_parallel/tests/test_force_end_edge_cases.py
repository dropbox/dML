#!/usr/bin/env python3
"""
Gap 8: Force-End Edge Case Tests

Tests edge cases in the AGX fix's force-end mechanism:
1. Rapid create-end-create cycles
2. Concurrent operations during commit
3. Multiple commits on same command buffer pattern
4. Stress test with interleaved operations

These tests verify the AGX fix handles encoder lifecycle edge cases correctly.
"""

import torch
import torch.nn as nn
import threading
import time
import sys
import os

# Verify we're using the AGX fix
dylib = os.environ.get('DYLD_INSERT_LIBRARIES', '')
if 'libagx_fix' not in dylib:
    print("WARNING: Running without AGX fix dylib")
    print("         Edge case tests may not exercise fix code paths")

DEVICE = 'mps'


def test_rapid_encoder_cycles():
    """
    Test 1: Rapid create-end-create cycles

    Creates and destroys many encoders rapidly to stress the
    force-end tracking data structures.
    """
    print("\n=== Test 1: Rapid Encoder Cycles ===")

    model = nn.Linear(64, 64).to(DEVICE).eval()

    errors = []
    ops = 0

    # Rapid create-compute-sync cycles
    for i in range(200):
        try:
            x = torch.randn(4, 64, device=DEVICE)
            with torch.no_grad():
                y = model(x)
            torch.mps.synchronize()
            ops += 1
        except Exception as e:
            errors.append(str(e))

    print(f"Operations: {ops}/200")
    print(f"Errors: {len(errors)}")

    if errors:
        print(f"First error: {errors[0]}")
        return False

    print("PASS")
    return True


def test_concurrent_ops_during_sync():
    """
    Test 2: Concurrent operations during synchronize

    Simulates the race between commit/sync and new encoder creation.
    One thread does sync while others continue creating work.
    """
    print("\n=== Test 2: Concurrent Ops During Sync ===")

    model = nn.Linear(32, 32).to(DEVICE).eval()

    stop_flag = threading.Event()
    errors = []
    ops_counter = [0]  # List for mutability
    lock = threading.Lock()

    def worker():
        while not stop_flag.is_set():
            try:
                x = torch.randn(2, 32, device=DEVICE)
                with torch.no_grad():
                    y = model(x)
                with lock:
                    ops_counter[0] += 1
            except Exception as e:
                with lock:
                    errors.append(str(e))

    def syncer():
        sync_count = 0
        while not stop_flag.is_set():
            torch.mps.synchronize()
            sync_count += 1
            time.sleep(0.001)  # Small delay
        return sync_count

    # Start workers
    workers = [threading.Thread(target=worker) for _ in range(4)]
    sync_thread = threading.Thread(target=syncer)

    for w in workers:
        w.start()
    sync_thread.start()

    # Run for 2 seconds
    time.sleep(2.0)
    stop_flag.set()

    for w in workers:
        w.join()
    sync_thread.join()

    print(f"Operations completed: {ops_counter[0]}")
    print(f"Errors: {len(errors)}")

    if errors:
        print(f"First error: {errors[0]}")
        return False

    print("PASS")
    return True


def test_multiple_models_interleaved():
    """
    Test 3: Multiple models with interleaved operations

    Tests that force-end correctly handles multiple command buffers
    with different encoders being created and ended.
    """
    print("\n=== Test 3: Multiple Models Interleaved ===")

    models = [nn.Linear(32, 32).to(DEVICE).eval() for _ in range(4)]

    errors = []
    completed = [0, 0, 0, 0]
    lock = threading.Lock()

    def worker(tid):
        for i in range(50):
            try:
                # Each thread uses different model
                x = torch.randn(2, 32, device=DEVICE)
                with torch.no_grad():
                    y = models[tid](x)
                # Interleaved sync - some threads sync, some don't
                if i % 3 == tid % 3:
                    torch.mps.synchronize()
                with lock:
                    completed[tid] += 1
            except Exception as e:
                with lock:
                    errors.append(f"T{tid}: {e}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Final sync
    torch.mps.synchronize()

    total = sum(completed)
    print(f"Operations: {total}/200")
    print(f"Per-thread: {completed}")
    print(f"Errors: {len(errors)}")

    if errors:
        print(f"First error: {errors[0]}")
        return False

    print("PASS")
    return True


def test_graph_vs_eager_interleaved():
    """
    Test 4: Graph compilation vs eager interleaved

    Tests that force-end handles the different encoder patterns
    from graph-compiled vs eager execution.
    """
    print("\n=== Test 4: Graph vs Eager Interleaved ===")

    model = nn.Linear(64, 64).to(DEVICE).eval()

    errors = []
    graph_ops = 0
    eager_ops = 0

    # Alternating graph and eager execution
    for i in range(100):
        try:
            x = torch.randn(4, 64, device=DEVICE)

            if i % 2 == 0:
                # Graph-style: no_grad + sync
                with torch.no_grad():
                    y = model(x)
                torch.mps.synchronize()
                graph_ops += 1
            else:
                # Eager-style: just compute, no immediate sync
                with torch.no_grad():
                    y = model(x)
                eager_ops += 1

        except Exception as e:
            errors.append(str(e))

    # Final sync
    torch.mps.synchronize()

    print(f"Graph-style ops: {graph_ops}/50")
    print(f"Eager-style ops: {eager_ops}/50")
    print(f"Errors: {len(errors)}")

    if errors:
        print(f"First error: {errors[0]}")
        return False

    print("PASS")
    return True


def test_empty_command_buffer_commit():
    """
    Test 5: Empty/minimal command buffer patterns

    Tests edge case where commit is called with minimal or
    no actual encoder work.
    """
    print("\n=== Test 5: Empty Command Buffer Patterns ===")

    errors = []

    try:
        # Just sync with no prior work
        torch.mps.synchronize()

        # Create tensor, immediate sync
        for i in range(50):
            x = torch.randn(2, 2, device=DEVICE)
            torch.mps.synchronize()

        # Multiple syncs in a row
        for i in range(10):
            torch.mps.synchronize()

    except Exception as e:
        errors.append(str(e))

    print(f"Errors: {len(errors)}")

    if errors:
        print(f"First error: {errors[0]}")
        return False

    print("PASS")
    return True


def test_stress_interleaved():
    """
    Test 6: High-stress interleaved operations

    Maximum stress with many threads, varied operations,
    and frequent syncs.
    """
    print("\n=== Test 6: Stress Interleaved Operations ===")

    model = nn.Linear(16, 16).to(DEVICE).eval()

    errors = []
    ops = [0]
    lock = threading.Lock()

    def worker(tid):
        for i in range(100):
            try:
                x = torch.randn(1, 16, device=DEVICE)
                with torch.no_grad():
                    y = model(x)
                if i % 5 == 0:
                    torch.mps.synchronize()
                with lock:
                    ops[0] += 1
            except Exception as e:
                with lock:
                    errors.append(f"T{tid}:{i}: {e}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]

    start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    torch.mps.synchronize()
    elapsed = time.time() - start

    print(f"Operations: {ops[0]}/800")
    print(f"Throughput: {ops[0]/elapsed:.1f} ops/s")
    print(f"Errors: {len(errors)}")

    if errors:
        print(f"First error: {errors[0]}")
        return False

    if ops[0] < 800:
        print(f"FAIL: Only {ops[0]}/800 completed")
        return False

    print("PASS")
    return True


def main():
    print("=" * 60)
    print("Gap 8: Force-End Edge Case Tests")
    print("=" * 60)

    # Check MPS availability
    if not torch.backends.mps.is_available():
        print("ERROR: MPS not available")
        return 1

    results = {}

    results['rapid_cycles'] = test_rapid_encoder_cycles()
    results['concurrent_sync'] = test_concurrent_ops_during_sync()
    results['interleaved_models'] = test_multiple_models_interleaved()
    results['graph_eager'] = test_graph_vs_eager_interleaved()
    results['empty_cb'] = test_empty_command_buffer_commit()
    results['stress'] = test_stress_interleaved()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    print("=" * 60)
    if all_pass:
        print("ALL FORCE-END EDGE CASE TESTS PASSED")
        return 0
    else:
        print("SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
