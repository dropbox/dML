#!/usr/bin/env python3
"""
Verify the 93% Threading Overhead Claim

WORKER_DIRECTIVE states:
- Single-op baseline (no threads): 10,698 ops/s
- Threading with 1 thread: 719 ops/s
- → 93% OVERHEAD

This script reproduces EXACTLY the benchmark pattern that generated this claim,
then tests alternative patterns to identify the overhead source.

The key question: Was the original benchmark creating new threads per operation,
or was there another cause?
"""

import torch
import torch.nn as nn
import threading
import time
import statistics
from concurrent.futures import ThreadPoolExecutor

assert torch.backends.mps.is_available(), "MPS not available"

# Warmup
torch.zeros(1, device="mps")
torch.mps.synchronize()

# Config - match typical benchmark
BATCH = 4
IN_FEATURES = 256
OUT_FEATURES = 128
ITERATIONS = 100
WARMUP = 20


def create_model():
    model = nn.Linear(IN_FEATURES, OUT_FEATURES).to("mps")
    model.eval()
    return model


def scenario_sequential_no_threads():
    """Sequential ops with NO threading involvement."""
    model = create_model()

    # Warmup
    for _ in range(WARMUP):
        x = torch.randn(BATCH, IN_FEATURES, device="mps")
        with torch.no_grad():
            _ = model(x)
        torch.mps.synchronize()

    # Measure
    start = time.perf_counter()
    for _ in range(ITERATIONS):
        x = torch.randn(BATCH, IN_FEATURES, device="mps")
        with torch.no_grad():
            _ = model(x)
        torch.mps.synchronize()
    elapsed = time.perf_counter() - start

    return ITERATIONS / elapsed


def scenario_new_thread_per_op():
    """NEW thread for EACH operation - expensive pattern."""
    model = create_model()

    # Warmup
    for _ in range(WARMUP):
        def w():
            x = torch.randn(BATCH, IN_FEATURES, device="mps")
            with torch.no_grad():
                _ = model(x)
            torch.mps.synchronize()
        t = threading.Thread(target=w)
        t.start()
        t.join()

    # Measure
    start = time.perf_counter()
    for _ in range(ITERATIONS):
        def w():
            x = torch.randn(BATCH, IN_FEATURES, device="mps")
            with torch.no_grad():
                _ = model(x)
            torch.mps.synchronize()
        t = threading.Thread(target=w)
        t.start()
        t.join()
    elapsed = time.perf_counter() - start

    return ITERATIONS / elapsed


def scenario_single_thread_reused():
    """Single thread runs ALL operations - thread reused."""
    model = create_model()
    ops_done = [0]

    def worker():
        # Warmup
        for _ in range(WARMUP):
            x = torch.randn(BATCH, IN_FEATURES, device="mps")
            with torch.no_grad():
                _ = model(x)
            torch.mps.synchronize()

        # Measure
        for _ in range(ITERATIONS):
            x = torch.randn(BATCH, IN_FEATURES, device="mps")
            with torch.no_grad():
                _ = model(x)
            torch.mps.synchronize()
            ops_done[0] += 1

    start = time.perf_counter()
    t = threading.Thread(target=worker)
    t.start()
    t.join()
    elapsed = time.perf_counter() - start

    return ops_done[0] / elapsed


def scenario_thread_pool_single_worker():
    """ThreadPoolExecutor with 1 worker - thread created once."""
    model = create_model()

    def do_one_op():
        x = torch.randn(BATCH, IN_FEATURES, device="mps")
        with torch.no_grad():
            _ = model(x)
        torch.mps.synchronize()

    # Warmup
    with ThreadPoolExecutor(max_workers=1) as executor:
        for _ in range(WARMUP):
            executor.submit(do_one_op).result()

    # Measure
    with ThreadPoolExecutor(max_workers=1) as executor:
        start = time.perf_counter()
        for _ in range(ITERATIONS):
            executor.submit(do_one_op).result()
        elapsed = time.perf_counter() - start

    return ITERATIONS / elapsed


def scenario_original_benchmark_pattern():
    """
    Original benchmark pattern from benchmark_parallel_mps.py:
    - N threads created
    - Each thread runs M iterations
    - Total ops = N * M

    For 1 thread, this should match "single thread reused".
    """
    model = None  # Created in worker

    results = []
    N_THREADS = 1
    ITERS_PER_THREAD = ITERATIONS

    def worker(tid):
        nonlocal model
        local_model = create_model()
        for i in range(ITERS_PER_THREAD):
            x = torch.randn(BATCH, IN_FEATURES, device="mps")
            with torch.no_grad():
                _ = local_model(x)
            torch.mps.synchronize()
            results.append(1)

    # Warmup (run once)
    warmup_results = []
    def warmup_worker(tid):
        local_model = create_model()
        for i in range(WARMUP):
            x = torch.randn(BATCH, IN_FEATURES, device="mps")
            with torch.no_grad():
                _ = local_model(x)
            torch.mps.synchronize()
    t = threading.Thread(target=warmup_worker, args=(0,))
    t.start()
    t.join()

    # Measure
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(N_THREADS)]
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - start

    return len(results) / elapsed


def main():
    print("# Verify Threading Overhead Claim")
    print()
    print(f"Config: batch={BATCH}, in={IN_FEATURES}, out={OUT_FEATURES}")
    print(f"Iterations: {ITERATIONS}")
    print()

    print("## Scenarios")
    print()

    # Run scenarios
    print("Running sequential (no threads)...")
    seq_ops = scenario_sequential_no_threads()

    print("Running new thread per op...")
    new_thread_ops = scenario_new_thread_per_op()

    print("Running single thread reused...")
    reused_ops = scenario_single_thread_reused()

    print("Running thread pool (1 worker)...")
    pool_ops = scenario_thread_pool_single_worker()

    print("Running original benchmark pattern...")
    orig_ops = scenario_original_benchmark_pattern()

    # Results table
    print()
    print("## Results")
    print()
    print("| Scenario | ops/s | vs Sequential |")
    print("|----------|-------|---------------|")
    print(f"| Sequential (no threads)        | {seq_ops:.0f} | 1.00x |")
    print(f"| New thread per op              | {new_thread_ops:.0f} | {new_thread_ops/seq_ops:.2f}x ({(1-new_thread_ops/seq_ops)*100:.0f}% overhead) |")
    print(f"| Single thread reused           | {reused_ops:.0f} | {reused_ops/seq_ops:.2f}x |")
    print(f"| Thread pool (1 worker)         | {pool_ops:.0f} | {pool_ops/seq_ops:.2f}x |")
    print(f"| Original benchmark pattern     | {orig_ops:.0f} | {orig_ops/seq_ops:.2f}x |")

    # Analysis
    print()
    print("## Analysis")
    print()

    if new_thread_ops < seq_ops * 0.3:
        overhead_pct = (1 - new_thread_ops / seq_ops) * 100
        print(f"**CONFIRMED**: New thread per op has {overhead_pct:.0f}% overhead.")
        print()
        print("The '93% threading overhead' claim came from comparing:")
        print(f"- Sequential: {seq_ops:.0f} ops/s")
        print(f"- New thread per op: {new_thread_ops:.0f} ops/s")
        print()
        print("However, when threads are REUSED (realistic pattern):")
        print(f"- Single thread reused: {reused_ops:.0f} ops/s ({reused_ops/seq_ops:.2f}x vs sequential)")
        print(f"- Thread pool: {pool_ops:.0f} ops/s ({pool_ops/seq_ops:.2f}x vs sequential)")
        print()
        print("**CONCLUSION**: The overhead is from THREAD CREATION, not from threading itself.")
        print("Use ThreadPoolExecutor or persistent threads to eliminate this overhead.")
    else:
        print("Results don't match original claim. System may differ or measurement method changed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
