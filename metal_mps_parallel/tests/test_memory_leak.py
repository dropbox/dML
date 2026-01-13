#!/usr/bin/env python3
"""
Test for memory leak in g_encoder_states.

Gap 2 from VERIFICATION_GAPS_ROADMAP.md claims g_encoder_states entries
are never removed. This test verifies that cleanup IS working correctly
by monitoring the active encoder count during a stress test.
"""

import os
import sys
import ctypes
import threading
import time

# Check AGX fix dylib is loaded
dylib_path = os.environ.get('DYLD_INSERT_LIBRARIES', '')
if 'libagx_fix_v2_9' not in dylib_path:
    print("ERROR: Must run with AGX fix v2.9 dylib loaded")
    print("Run with: ./scripts/run_test_with_crash_check.sh python3 tests/test_memory_leak.py")
    sys.exit(1)

import torch

# Load the dylib to query stats
dylib = ctypes.CDLL(dylib_path.split(':')[0])
dylib.agx_fix_v2_9_get_active_count.restype = ctypes.c_size_t
dylib.agx_fix_v2_9_get_encoders_created.restype = ctypes.c_uint64
dylib.agx_fix_v2_9_get_encoders_released.restype = ctypes.c_uint64

def get_active_count():
    return dylib.agx_fix_v2_9_get_active_count()

def get_stats():
    return {
        'active': dylib.agx_fix_v2_9_get_active_count(),
        'created': dylib.agx_fix_v2_9_get_encoders_created(),
        'released': dylib.agx_fix_v2_9_get_encoders_released(),
    }

def mps_op():
    """Simple MPS operation that creates and ends an encoder."""
    x = torch.randn(64, 64, device='mps')
    y = x @ x
    torch.mps.synchronize()
    return y

def test_memory_leak():
    print("=" * 60)
    print("Memory Leak Test (Gap 2 Verification)")
    print("=" * 60)

    # Warmup
    print("\n--- Warmup ---")
    for _ in range(10):
        mps_op()
    torch.mps.synchronize()
    time.sleep(0.1)

    initial_stats = get_stats()
    print(f"Initial state: active={initial_stats['active']}, created={initial_stats['created']}, released={initial_stats['released']}")

    # Run many iterations
    iterations = 1000
    print(f"\n--- Running {iterations} iterations ---")

    samples = []
    for i in range(iterations):
        mps_op()
        if (i + 1) % 100 == 0:
            torch.mps.synchronize()
            time.sleep(0.01)  # Allow cleanup
            stats = get_stats()
            samples.append(stats['active'])
            print(f"  After {i+1} ops: active={stats['active']}, created={stats['created']}, released={stats['released']}")

    # Final sync and cleanup
    torch.mps.synchronize()
    time.sleep(0.5)

    final_stats = get_stats()
    print(f"\n--- Final State ---")
    print(f"Active encoders: {final_stats['active']}")
    print(f"Total created: {final_stats['created']}")
    print(f"Total released: {final_stats['released']}")
    print(f"Leak (created - released): {final_stats['created'] - final_stats['released']}")

    # Check for leak
    # Active count should be small (a few encoders may be in-flight)
    # and should NOT grow linearly with iterations
    max_active = max(samples)
    min_active = min(samples)

    print(f"\n--- Analysis ---")
    print(f"Active count range: {min_active} to {max_active}")

    # If there's a leak, active count would grow to ~1000 (one per iteration)
    # Without a leak, it should stay bounded (< 100 typically)
    if max_active < 100:
        print(f"PASS: Active count stayed bounded (max={max_active})")
        print("Gap 2 verification: No memory leak detected")
        return True
    else:
        print(f"FAIL: Active count grew unbounded (max={max_active})")
        print("Gap 2 verification: Memory leak detected")
        return False

def test_multithreaded_leak():
    """Test for leaks under multithreaded stress."""
    print("\n" + "=" * 60)
    print("Multithreaded Memory Leak Test")
    print("=" * 60)

    torch.mps.synchronize()
    time.sleep(0.1)

    initial_stats = get_stats()
    print(f"Initial: active={initial_stats['active']}")

    iterations_per_thread = 100
    num_threads = 8
    barrier = threading.Barrier(num_threads)

    def worker():
        barrier.wait()
        for _ in range(iterations_per_thread):
            mps_op()

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    torch.mps.synchronize()
    time.sleep(0.5)

    final_stats = get_stats()
    total_ops = num_threads * iterations_per_thread
    print(f"After {total_ops} ops ({num_threads} threads x {iterations_per_thread})")
    print(f"Final: active={final_stats['active']}, created={final_stats['created']}, released={final_stats['released']}")
    print(f"Leak: {final_stats['created'] - final_stats['released']}")

    if final_stats['active'] < 100:
        print("PASS: No leak under multithreaded stress")
        return True
    else:
        print("FAIL: Leak detected under multithreaded stress")
        return False

if __name__ == '__main__':
    if not torch.backends.mps.is_available():
        print("MPS not available")
        sys.exit(1)

    result1 = test_memory_leak()
    result2 = test_multithreaded_leak()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Single-threaded: {'PASS' if result1 else 'FAIL'}")
    print(f"Multi-threaded:  {'PASS' if result2 else 'FAIL'}")

    if result1 and result2:
        print("\nGap 2 CLOSED: Memory cleanup is working correctly")
        sys.exit(0)
    else:
        print("\nGap 2 OPEN: Memory leak needs fixing")
        sys.exit(1)
