#!/usr/bin/env python3
"""
Minimal reproduction of Metal command encoder race condition using MLX.

This script demonstrates that MLX crashes when multiple Python threads
perform matrix operations concurrently, due to a race condition in
Apple's AGX command buffer encoder.

Requirements:
    pip install mlx

Expected Result:
    Crash with assertion:
    'A command encoder is already encoding to this command buffer'

Tested on:
    - Apple M4 Max (macOS 15.7.2)
    - MLX 0.30.0
"""

import mlx.core as mx
import threading
import time

def matrix_worker(thread_id: int, iterations: int = 20):
    """Perform matrix multiplication in a loop."""
    for i in range(iterations):
        a = mx.random.normal((1024, 1024))
        b = mx.random.normal((1024, 1024))
        c = a @ b
        mx.eval(c)  # Force GPU execution
    print(f"Thread {thread_id} completed {iterations} iterations")

def run_parallel_test(num_threads: int = 4):
    """Run matrix operations in parallel threads."""
    print(f"Starting {num_threads} threads...")
    print(f"MLX version: {mx.__version__}")
    print()

    threads = []
    start_time = time.time()

    for i in range(num_threads):
        t = threading.Thread(target=matrix_worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    elapsed = time.time() - start_time
    print(f"\nAll threads completed in {elapsed:.2f}s")
    print("SUCCESS - No crash occurred")

if __name__ == "__main__":
    print("=" * 60)
    print("MLX Multi-threaded Metal Crash Reproduction")
    print("=" * 60)
    print()
    print("This script will likely crash with:")
    print("  'A command encoder is already encoding to this command buffer'")
    print()

    try:
        run_parallel_test(num_threads=4)
    except Exception as e:
        print(f"Exception: {e}")
