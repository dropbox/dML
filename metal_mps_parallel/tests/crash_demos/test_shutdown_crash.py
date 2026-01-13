#!/usr/bin/env python3
"""
Minimal test to reproduce the shutdown crash (N=1424 investigation).

This test creates threads that do MPS work and then exits immediately.
With MPS_DISABLE_ENCODING_MUTEX=1, this should crash ~55% of the time.
"""
import threading
import sys
import os

def main():
    print("Starting test...")

    import torch

    # Warmup
    x = torch.randn(8, 8, device='mps')
    torch.mps.synchronize()
    print("Warmup done")

    n_threads = 8
    barrier = threading.Barrier(n_threads + 1)
    completed = []
    lock = threading.Lock()

    def worker(tid):
        # Do MPS work
        for _ in range(10):
            t = torch.randn(32, 32, device='mps')
            y = t * 2.0
        torch.mps.synchronize()
        with lock:
            completed.append(tid)
        barrier.wait()

    threads = []
    for i in range(n_threads):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    barrier.wait()  # Wait for all threads to finish work
    print(f"  {n_threads} threads created MPS state")

    for t in threads:
        t.join(timeout=5.0)

    print(f"  {n_threads} threads completed work")
    print(f"  All threads joined. Completed: {len(completed)}/{n_threads}")

    # Exit immediately without cleanup - this triggers the crash
    print("  Exiting without cleanup...")
    # Don't call synchronize or empty_cache - let Python interpreter cleanup
    # race with MPS static destruction

if __name__ == '__main__':
    main()
