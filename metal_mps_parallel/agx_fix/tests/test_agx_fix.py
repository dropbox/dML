#!/usr/bin/env python3
"""
Test script to verify the AGX driver fix works.

This script runs the same multi-threaded workload that triggers the crash
WITHOUT our PyTorch mutex workaround, relying ONLY on libagx_fix.dylib.

Usage:
    # Without fix (will crash ~55% of the time):
    MPS_DISABLE_ENCODING_MUTEX=1 python3 test_agx_fix.py

    # With fix (should not crash):
    DYLD_INSERT_LIBRARIES=../build/libagx_fix.dylib python3 test_agx_fix.py
"""

import os
import sys
import time
import ctypes
from concurrent.futures import ThreadPoolExecutor, as_completed

# Check if running with the fix
def check_agx_fix_loaded():
    """Check if libagx_fix.dylib is loaded."""
    try:
        lib = ctypes.CDLL("libagx_fix.dylib")
        enabled = lib.agx_fix_is_enabled()
        return enabled
    except OSError:
        return False

def main():
    print("=" * 60)
    print("AGX Driver Fix Test")
    print("=" * 60)

    # Check environment
    fix_loaded = check_agx_fix_loaded()
    mutex_disabled = os.environ.get("MPS_DISABLE_ENCODING_MUTEX") == "1"

    print(f"libagx_fix.dylib loaded: {fix_loaded}")
    print(f"PyTorch mutex disabled:  {mutex_disabled}")
    print()

    if not fix_loaded and not mutex_disabled:
        print("WARNING: Running with PyTorch's built-in mutex.")
        print("         This test won't verify libagx_fix.dylib.")
        print("         Set MPS_DISABLE_ENCODING_MUTEX=1 to test properly.")
        print()

    # Import PyTorch
    import torch

    if not torch.backends.mps.is_available():
        print("ERROR: MPS not available")
        sys.exit(1)

    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print()

    # Create a simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(512, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 512),
    ).to("mps")
    model.eval()

    # Test parameters
    num_threads = 8
    ops_per_thread = 50
    total_ops = num_threads * ops_per_thread

    print(f"Running stress test:")
    print(f"  Threads: {num_threads}")
    print(f"  Ops per thread: {ops_per_thread}")
    print(f"  Total ops: {total_ops}")
    print()

    # Worker function
    def worker(thread_id):
        results = []
        for i in range(ops_per_thread):
            x = torch.randn(32, 512, device="mps")
            with torch.no_grad():
                y = model(x)
            results.append(y.shape)
        return thread_id, len(results)

    # Run stress test
    start_time = time.perf_counter()
    completed = 0
    crashed = False

    try:
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            futures = [pool.submit(worker, i) for i in range(num_threads)]

            for future in as_completed(futures):
                thread_id, count = future.result()
                completed += count
                print(f"  Thread {thread_id}: {count} ops completed")

    except Exception as e:
        crashed = True
        print(f"\nCRASH: {type(e).__name__}: {e}")

    elapsed = time.perf_counter() - start_time

    # Results
    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Completed: {completed}/{total_ops} ops")
    print(f"Time: {elapsed:.2f}s")
    print(f"Throughput: {completed/elapsed:.0f} ops/s")
    print(f"Crashed: {crashed}")
    print()

    # Get statistics from libagx_fix if loaded
    if fix_loaded:
        try:
            lib = ctypes.CDLL("libagx_fix.dylib")
            acquisitions = lib.agx_fix_get_acquisitions()
            contentions = lib.agx_fix_get_contentions()
            print(f"AGX Fix Statistics:")
            print(f"  Mutex acquisitions: {acquisitions}")
            print(f"  Mutex contentions:  {contentions}")
            if acquisitions > 0:
                print(f"  Contention rate:    {100*contentions/acquisitions:.1f}%")
        except Exception as e:
            print(f"Could not get AGX fix stats: {e}")
        print()

    # Verdict
    if crashed:
        print("FAIL: Test crashed (AGX driver race condition triggered)")
        sys.exit(1)
    elif completed == total_ops:
        print("PASS: All operations completed successfully")
        if fix_loaded:
            print("      libagx_fix.dylib prevented the crash!")
        sys.exit(0)
    else:
        print(f"PARTIAL: Only {completed}/{total_ops} ops completed")
        sys.exit(1)

if __name__ == "__main__":
    main()
