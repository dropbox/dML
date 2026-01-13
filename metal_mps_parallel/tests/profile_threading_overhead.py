#!/usr/bin/env python3
"""
Profile Threading Overhead Analysis

Task A from WORKER_DIRECTIVE.md: Investigate the 93% threading overhead.

The WORKER_DIRECTIVE states:
- Single-op baseline (no threads): 10,698 ops/s
- Threading with 1 thread: 719 ops/s
- → 93% OVERHEAD from Python/PyTorch threading!

This script profiles each component to identify where the overhead comes from:
1. Thread creation/join overhead
2. Model creation overhead
3. Input tensor creation overhead
4. torch.no_grad() context overhead
5. torch.mps.synchronize() overhead
6. Lock acquisition overhead (for result tracking)

Usage:
    python tests/profile_threading_overhead.py
"""

import torch
import torch.nn as nn
import threading
import time
import statistics

# Ensure MPS is available
assert torch.backends.mps.is_available(), "MPS not available"

# Warmup
torch.zeros(1, device="mps")
torch.mps.synchronize()

# Configuration - match WORKER_DIRECTIVE benchmark
IN_FEATURES = 256
OUT_FEATURES = 128
BATCH = 4
ITERATIONS = 200
WARMUP = 50


def measure_baseline_no_threads():
    """Baseline: Direct sequential operations, no threading."""
    model = nn.Linear(IN_FEATURES, OUT_FEATURES).to("mps")
    model.eval()

    # Warmup
    for _ in range(WARMUP):
        x = torch.randn(BATCH, IN_FEATURES, device="mps")
        with torch.no_grad():
            _ = model(x)
        torch.mps.synchronize()

    # Measure
    times = []
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        x = torch.randn(BATCH, IN_FEATURES, device="mps")
        with torch.no_grad():
            _ = model(x)
        torch.mps.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    return times


def measure_single_thread():
    """Single thread: Same operations but inside a thread."""
    model = nn.Linear(IN_FEATURES, OUT_FEATURES).to("mps")
    model.eval()
    times = []

    def worker():
        # Warmup
        for _ in range(WARMUP):
            x = torch.randn(BATCH, IN_FEATURES, device="mps")
            with torch.no_grad():
                _ = model(x)
            torch.mps.synchronize()

        # Measure
        for _ in range(ITERATIONS):
            start = time.perf_counter()
            x = torch.randn(BATCH, IN_FEATURES, device="mps")
            with torch.no_grad():
                _ = model(x)
            torch.mps.synchronize()
            end = time.perf_counter()
            times.append(end - start)

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    return times


def measure_thread_create_join_only():
    """Measure pure thread creation/join overhead."""
    times = []

    def empty_worker():
        pass

    # Warmup
    for _ in range(WARMUP):
        t = threading.Thread(target=empty_worker)
        t.start()
        t.join()

    for _ in range(ITERATIONS):
        start = time.perf_counter()
        t = threading.Thread(target=empty_worker)
        t.start()
        t.join()
        end = time.perf_counter()
        times.append(end - start)

    return times


def measure_thread_with_tensor_creation():
    """Thread with just tensor creation (no model, no sync)."""
    times = []

    def worker():
        x = torch.randn(BATCH, IN_FEATURES, device="mps")

    # Warmup
    for _ in range(WARMUP):
        t = threading.Thread(target=worker)
        t.start()
        t.join()

    for _ in range(ITERATIONS):
        start = time.perf_counter()
        t = threading.Thread(target=worker)
        t.start()
        t.join()
        end = time.perf_counter()
        times.append(end - start)

    return times


def measure_thread_with_sync():
    """Thread with just synchronize (no tensor, no model)."""
    times = []

    def worker():
        torch.mps.synchronize()

    # Warmup
    for _ in range(WARMUP):
        t = threading.Thread(target=worker)
        t.start()
        t.join()

    for _ in range(ITERATIONS):
        start = time.perf_counter()
        t = threading.Thread(target=worker)
        t.start()
        t.join()
        end = time.perf_counter()
        times.append(end - start)

    return times


def measure_reused_thread_pool():
    """Reuse threads via concurrent.futures to avoid creation overhead."""
    from concurrent.futures import ThreadPoolExecutor
    import queue

    model = nn.Linear(IN_FEATURES, OUT_FEATURES).to("mps")
    model.eval()

    def worker(dummy):
        start = time.perf_counter()
        x = torch.randn(BATCH, IN_FEATURES, device="mps")
        with torch.no_grad():
            _ = model(x)
        torch.mps.synchronize()
        end = time.perf_counter()
        return end - start

    # Create pool once, reuse threads
    with ThreadPoolExecutor(max_workers=1) as executor:
        # Warmup
        for _ in range(WARMUP):
            list(executor.map(worker, [None]))

        # Measure
        times = []
        for _ in range(ITERATIONS):
            results = list(executor.map(worker, [None]))
            times.extend(results)

    return times


def measure_event_sync_vs_device_sync():
    """Compare event sync vs device sync in threads."""
    model = nn.Linear(IN_FEATURES, OUT_FEATURES).to("mps")
    model.eval()

    # Event sync
    event_times = []

    def worker_event():
        event = torch.mps.Event(enable_timing=False)
        for _ in range(WARMUP):
            x = torch.randn(BATCH, IN_FEATURES, device="mps")
            with torch.no_grad():
                _ = model(x)
            event.record()
            event.synchronize()

        for _ in range(ITERATIONS):
            start = time.perf_counter()
            x = torch.randn(BATCH, IN_FEATURES, device="mps")
            with torch.no_grad():
                _ = model(x)
            event.record()
            event.synchronize()
            end = time.perf_counter()
            event_times.append(end - start)

    t = threading.Thread(target=worker_event)
    t.start()
    t.join()

    # Device sync
    device_times = []

    def worker_device():
        for _ in range(WARMUP):
            x = torch.randn(BATCH, IN_FEATURES, device="mps")
            with torch.no_grad():
                _ = model(x)
            torch.mps.synchronize()

        for _ in range(ITERATIONS):
            start = time.perf_counter()
            x = torch.randn(BATCH, IN_FEATURES, device="mps")
            with torch.no_grad():
                _ = model(x)
            torch.mps.synchronize()
            end = time.perf_counter()
            device_times.append(end - start)

    t = threading.Thread(target=worker_device)
    t.start()
    t.join()

    return event_times, device_times


def measure_preallocated_tensors():
    """Reuse pre-allocated tensors instead of creating new ones."""
    model = nn.Linear(IN_FEATURES, OUT_FEATURES).to("mps")
    model.eval()

    # Pre-allocate input tensor
    x = torch.randn(BATCH, IN_FEATURES, device="mps")

    times = []

    def worker():
        nonlocal x
        for _ in range(WARMUP):
            x.normal_()  # In-place randomize
            with torch.no_grad():
                _ = model(x)
            torch.mps.synchronize()

        for _ in range(ITERATIONS):
            start = time.perf_counter()
            x.normal_()  # In-place randomize instead of new tensor
            with torch.no_grad():
                _ = model(x)
            torch.mps.synchronize()
            end = time.perf_counter()
            times.append(end - start)

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    return times


def analyze_times(times, label):
    """Analyze timing results."""
    if not times:
        return {"label": label, "ops_per_sec": 0}

    mean_ms = statistics.mean(times) * 1000
    std_ms = statistics.stdev(times) * 1000 if len(times) > 1 else 0
    ops_per_sec = 1.0 / statistics.mean(times) if statistics.mean(times) > 0 else 0

    return {
        "label": label,
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "ops_per_sec": ops_per_sec,
        "count": len(times)
    }


def main():
    print("# Threading Overhead Profiling")
    print(f"Config: batch={BATCH}, in={IN_FEATURES}, out={OUT_FEATURES}")
    print(f"Iterations: {ITERATIONS} (warmup: {WARMUP})")
    print()

    results = []

    # 1. Baseline (no threads)
    print("Measuring baseline (no threads)...")
    times = measure_baseline_no_threads()
    results.append(analyze_times(times, "Baseline (no threads)"))

    # 2. Single thread (full operation)
    print("Measuring single thread (full op)...")
    times = measure_single_thread()
    results.append(analyze_times(times, "Single thread (full op)"))

    # 3. Thread create/join only
    print("Measuring thread create/join only...")
    times = measure_thread_create_join_only()
    results.append(analyze_times(times, "Thread create/join only"))

    # 4. Thread + tensor creation
    print("Measuring thread + tensor creation...")
    times = measure_thread_with_tensor_creation()
    results.append(analyze_times(times, "Thread + tensor creation"))

    # 5. Thread + sync
    print("Measuring thread + sync only...")
    times = measure_thread_with_sync()
    results.append(analyze_times(times, "Thread + sync only"))

    # 6. Thread pool (reused threads)
    print("Measuring thread pool (reused threads)...")
    times = measure_reused_thread_pool()
    results.append(analyze_times(times, "Thread pool (reused)"))

    # 7. Pre-allocated tensors
    print("Measuring pre-allocated tensors...")
    times = measure_preallocated_tensors()
    results.append(analyze_times(times, "Pre-allocated tensors"))

    # 8. Event sync vs device sync
    print("Measuring event sync vs device sync...")
    event_times, device_times = measure_event_sync_vs_device_sync()
    results.append(analyze_times(event_times, "Thread + event sync"))
    results.append(analyze_times(device_times, "Thread + device sync"))

    # Output results
    print()
    print("## Results")
    print()
    print("| Method | Mean (ms) | Std (ms) | ops/s | vs Baseline |")
    print("|--------|-----------|----------|-------|-------------|")

    baseline_ops = results[0]["ops_per_sec"]
    for r in results:
        ratio = r["ops_per_sec"] / baseline_ops if baseline_ops > 0 else 0
        overhead = (1 - ratio) * 100 if ratio > 0 else 100
        print(f"| {r['label']:<30} | {r.get('mean_ms', 0):.3f} | {r.get('std_ms', 0):.3f} | {r['ops_per_sec']:.0f} | {ratio:.2f}x ({overhead:+.0f}% overhead) |")

    # Analysis
    print()
    print("## Analysis")
    print()

    baseline_ops = results[0]["ops_per_sec"]
    single_thread_ops = results[1]["ops_per_sec"]
    thread_only_ops = results[2]["ops_per_sec"]
    thread_tensor_ops = results[3]["ops_per_sec"]
    thread_sync_ops = results[4]["ops_per_sec"]
    thread_pool_ops = results[5]["ops_per_sec"]
    prealloc_ops = results[6]["ops_per_sec"]

    print(f"Baseline (no threads): {baseline_ops:.0f} ops/s")
    print(f"Single thread:         {single_thread_ops:.0f} ops/s ({single_thread_ops/baseline_ops*100:.1f}% of baseline)")
    print()

    print("### Overhead Breakdown")
    print()
    print(f"1. Pure thread create/join: {1000/thread_only_ops:.3f} ms per op ({thread_only_ops:.0f} ops/s capacity)")
    print(f"2. Thread + tensor creation: {1000/thread_tensor_ops:.3f} ms per op")
    print(f"3. Thread + sync only: {1000/thread_sync_ops:.3f} ms per op")
    print()

    print("### Optimization Impact")
    print()
    print(f"Thread pool (avoid create/join): {thread_pool_ops:.0f} ops/s ({thread_pool_ops/single_thread_ops:.2f}x vs single thread)")
    print(f"Pre-allocated tensors:           {prealloc_ops:.0f} ops/s ({prealloc_ops/single_thread_ops:.2f}x vs single thread)")
    print()

    # Conclusions
    print("## Conclusions")
    print()

    # Calculate contributions
    thread_overhead_ms = 1000/thread_only_ops if thread_only_ops > 0 else 0
    total_thread_op_ms = 1000/single_thread_ops if single_thread_ops > 0 else 0
    baseline_op_ms = 1000/baseline_ops if baseline_ops > 0 else 0
    overhead_ms = total_thread_op_ms - baseline_op_ms

    print(f"Total overhead per threaded op: {overhead_ms:.3f} ms")
    print(f"- Thread create/join: ~{thread_overhead_ms:.3f} ms ({thread_overhead_ms/total_thread_op_ms*100:.1f}% of total)")
    print()

    if thread_pool_ops > single_thread_ops * 1.2:
        print("**Key Finding**: Thread pools significantly reduce overhead.")
        print("Recommendation: Use ThreadPoolExecutor instead of creating threads per-op.")
    elif prealloc_ops > single_thread_ops * 1.2:
        print("**Key Finding**: Pre-allocated tensors reduce overhead.")
        print("Recommendation: Reuse input tensors with in-place operations.")
    else:
        print("**Key Finding**: Overhead is inherent to threading + MPS interaction.")
        print("The Python GIL and PyTorch/Metal interop add unavoidable latency.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
