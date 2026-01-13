#!/usr/bin/env python3
"""
Gap 7: Bottleneck Profiling Test

Profile to identify the actual bottleneck causing non-monotonic throughput
when scaling threads. Measures timing breakdown for:
- Tensor allocation time
- Compute time (mm, relu)
- Synchronization time
- Python lock contention

The goal is to understand why efficiency is ~14% at 8 threads instead of
higher, and why throughput can be non-monotonic (e.g., 4 threads sometimes
slower than 2 threads).

Author: Claude (AI Worker N=3678)
Date: 2025-12-25
"""

import threading
import time
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class TimingBreakdown:
    """Timing breakdown for a single operation."""
    alloc_ns: int  # Tensor allocation time
    compute_ns: int  # Compute operations time
    sync_ns: int  # Synchronization time
    lock_ns: int  # Lock acquisition time (Python)
    total_ns: int  # Total operation time


def profile_threading_bottleneck(num_threads: int, iterations: int = 50):
    """
    Profile where time is spent during threaded MPS operations.

    Returns detailed timing breakdown per thread.
    """
    import torch

    # Per-thread timing storage
    thread_timings: Dict[int, List[TimingBreakdown]] = defaultdict(list)
    errors = []
    result_lock = threading.Lock()

    def worker(thread_id: int):
        local_timings = []

        try:
            for i in range(iterations):
                # 1. Measure tensor allocation
                t0 = time.perf_counter_ns()
                x = torch.randn(256, 256, device='mps')
                y = torch.randn(256, 256, device='mps')
                t1 = time.perf_counter_ns()
                alloc_ns = t1 - t0

                # 2. Measure compute operations
                t2 = time.perf_counter_ns()
                z = torch.mm(x, y)
                z = torch.relu(z)
                t3 = time.perf_counter_ns()
                compute_ns = t3 - t2

                # 3. Measure synchronization
                t4 = time.perf_counter_ns()
                # Use .cpu() sync which is safer than torch.mps.synchronize()
                _ = z.sum().cpu()
                t5 = time.perf_counter_ns()
                sync_ns = t5 - t4

                # 4. Measure lock acquisition (for result storage)
                t6 = time.perf_counter_ns()
                with result_lock:
                    pass  # Just measure lock acquisition
                t7 = time.perf_counter_ns()
                lock_ns = t7 - t6

                total_ns = t7 - t0

                local_timings.append(TimingBreakdown(
                    alloc_ns=alloc_ns,
                    compute_ns=compute_ns,
                    sync_ns=sync_ns,
                    lock_ns=lock_ns,
                    total_ns=total_ns
                ))

        except Exception as e:
            with result_lock:
                errors.append((thread_id, str(e)))

        # Store results
        with result_lock:
            thread_timings[thread_id] = local_timings

    # Run threads
    threads = []
    start_time = time.perf_counter()

    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    elapsed = time.perf_counter() - start_time

    return thread_timings, errors, elapsed


def analyze_timings(thread_timings: Dict[int, List[TimingBreakdown]], num_threads: int, elapsed: float):
    """Analyze timing data and produce statistics."""

    # Aggregate all timings
    all_alloc = []
    all_compute = []
    all_sync = []
    all_lock = []
    all_total = []

    for thread_id, timings in thread_timings.items():
        for t in timings:
            all_alloc.append(t.alloc_ns / 1_000_000)  # Convert to ms
            all_compute.append(t.compute_ns / 1_000_000)
            all_sync.append(t.sync_ns / 1_000_000)
            all_lock.append(t.lock_ns / 1_000_000)
            all_total.append(t.total_ns / 1_000_000)

    total_ops = len(all_total)

    if total_ops == 0:
        return None

    stats = {
        'num_threads': num_threads,
        'total_ops': total_ops,
        'elapsed_s': elapsed,
        'throughput': total_ops / elapsed,
        'alloc_mean_ms': statistics.mean(all_alloc),
        'alloc_p50_ms': statistics.median(all_alloc),
        'alloc_p99_ms': sorted(all_alloc)[int(len(all_alloc) * 0.99)] if len(all_alloc) >= 100 else max(all_alloc),
        'compute_mean_ms': statistics.mean(all_compute),
        'compute_p50_ms': statistics.median(all_compute),
        'compute_p99_ms': sorted(all_compute)[int(len(all_compute) * 0.99)] if len(all_compute) >= 100 else max(all_compute),
        'sync_mean_ms': statistics.mean(all_sync),
        'sync_p50_ms': statistics.median(all_sync),
        'sync_p99_ms': sorted(all_sync)[int(len(all_sync) * 0.99)] if len(all_sync) >= 100 else max(all_sync),
        'lock_mean_ms': statistics.mean(all_lock),
        'lock_p50_ms': statistics.median(all_lock),
        'lock_p99_ms': sorted(all_lock)[int(len(all_lock) * 0.99)] if len(all_lock) >= 100 else max(all_lock),
        'total_mean_ms': statistics.mean(all_total),
        'total_p50_ms': statistics.median(all_total),
        'total_p99_ms': sorted(all_total)[int(len(all_total) * 0.99)] if len(all_total) >= 100 else max(all_total),
    }

    # Calculate percentage breakdown
    total_time_ms = stats['total_mean_ms']
    if total_time_ms > 0:
        stats['alloc_pct'] = (stats['alloc_mean_ms'] / total_time_ms) * 100
        stats['compute_pct'] = (stats['compute_mean_ms'] / total_time_ms) * 100
        stats['sync_pct'] = (stats['sync_mean_ms'] / total_time_ms) * 100
        stats['lock_pct'] = (stats['lock_mean_ms'] / total_time_ms) * 100

    return stats


def print_report(all_stats: List[Dict]):
    """Print a formatted profiling report."""

    print("\n" + "=" * 80)
    print("GAP 7: THREADING BOTTLENECK PROFILING REPORT")
    print("=" * 80)

    # Summary table
    print("\n### THROUGHPUT VS THREADS ###")
    print(f"{'Threads':<10} {'Ops/s':<12} {'Mean (ms)':<12} {'P99 (ms)':<12} {'Efficiency':<12}")
    print("-" * 58)

    baseline_throughput = None
    for stats in all_stats:
        if baseline_throughput is None:
            baseline_throughput = stats['throughput']
            efficiency = 100.0
        else:
            # Linear scaling efficiency: (actual / (baseline * threads)) * 100
            expected = baseline_throughput * stats['num_threads']
            efficiency = (stats['throughput'] / expected) * 100

        print(f"{stats['num_threads']:<10} {stats['throughput']:<12.1f} {stats['total_mean_ms']:<12.2f} {stats['total_p99_ms']:<12.2f} {efficiency:<12.1f}%")

    # Timing breakdown table
    print("\n### TIMING BREAKDOWN (Mean, ms) ###")
    print(f"{'Threads':<10} {'Alloc':<12} {'Compute':<12} {'Sync':<12} {'Lock':<12} {'Total':<12}")
    print("-" * 70)

    for stats in all_stats:
        print(f"{stats['num_threads']:<10} {stats['alloc_mean_ms']:<12.3f} {stats['compute_mean_ms']:<12.3f} {stats['sync_mean_ms']:<12.3f} {stats['lock_mean_ms']:<12.3f} {stats['total_mean_ms']:<12.3f}")

    # Percentage breakdown
    print("\n### TIME DISTRIBUTION (% of total) ###")
    print(f"{'Threads':<10} {'Alloc %':<12} {'Compute %':<12} {'Sync %':<12} {'Lock %':<12}")
    print("-" * 58)

    for stats in all_stats:
        print(f"{stats['num_threads']:<10} {stats.get('alloc_pct', 0):<12.1f} {stats.get('compute_pct', 0):<12.1f} {stats.get('sync_pct', 0):<12.1f} {stats.get('lock_pct', 0):<12.1f}")

    # P99 latency comparison
    print("\n### P99 LATENCY (ms) ###")
    print(f"{'Threads':<10} {'Alloc P99':<12} {'Compute P99':<12} {'Sync P99':<12} {'Lock P99':<12}")
    print("-" * 58)

    for stats in all_stats:
        print(f"{stats['num_threads']:<10} {stats['alloc_p99_ms']:<12.3f} {stats['compute_p99_ms']:<12.3f} {stats['sync_p99_ms']:<12.3f} {stats['lock_p99_ms']:<12.3f}")

    # Analysis
    print("\n### ANALYSIS ###")

    if len(all_stats) >= 2:
        # Compare 1-thread vs multi-thread
        single = all_stats[0]
        multi = all_stats[-1]  # Highest thread count

        print(f"\nSingle-threaded baseline (1 thread):")
        print(f"  Throughput: {single['throughput']:.1f} ops/s")
        print(f"  Mean latency: {single['total_mean_ms']:.2f} ms")

        print(f"\nHighest thread count ({multi['num_threads']} threads):")
        print(f"  Throughput: {multi['throughput']:.1f} ops/s")
        print(f"  Mean latency: {multi['total_mean_ms']:.2f} ms")
        print(f"  Speedup: {multi['throughput'] / single['throughput']:.2f}x")

        # Find the bottleneck
        sync_increase = multi['sync_mean_ms'] / single['sync_mean_ms'] if single['sync_mean_ms'] > 0 else 0
        alloc_increase = multi['alloc_mean_ms'] / single['alloc_mean_ms'] if single['alloc_mean_ms'] > 0 else 0
        compute_increase = multi['compute_mean_ms'] / single['compute_mean_ms'] if single['compute_mean_ms'] > 0 else 0

        print(f"\nLatency increase ratios ({multi['num_threads']}t vs 1t):")
        print(f"  Allocation: {alloc_increase:.2f}x")
        print(f"  Compute: {compute_increase:.2f}x")
        print(f"  Synchronization: {sync_increase:.2f}x")

        # Identify bottleneck
        bottleneck = "Unknown"
        max_increase = max(sync_increase, alloc_increase, compute_increase)
        if max_increase == sync_increase:
            bottleneck = "SYNCHRONIZATION"
        elif max_increase == alloc_increase:
            bottleneck = "ALLOCATION"
        else:
            bottleneck = "COMPUTE"

        print(f"\n*** PRIMARY BOTTLENECK: {bottleneck} ***")

        # Check for non-monotonic behavior
        throughputs = [s['throughput'] for s in all_stats]
        non_monotonic = False
        for i in range(1, len(throughputs)):
            if throughputs[i] < throughputs[i-1]:
                non_monotonic = True
                break

        if non_monotonic:
            print(f"\n*** NON-MONOTONIC BEHAVIOR DETECTED ***")
            print("Throughput decreased when adding more threads.")
            print("This indicates contention or resource exhaustion.")
        else:
            print(f"\nThroughput is monotonically increasing (no regression).")


def test_bottleneck_profiling():
    """Main test function - profiles bottleneck across thread counts."""

    print("Gap 7: Bottleneck Profiling Test")
    print("================================")
    print("Profiling MPS threading performance...")

    thread_counts = [1, 2, 4, 8]
    iterations = 100  # Operations per thread
    all_stats = []

    for num_threads in thread_counts:
        print(f"\nProfiling with {num_threads} thread(s)...")

        timings, errors, elapsed = profile_threading_bottleneck(num_threads, iterations)

        if errors:
            print(f"  ERRORS: {len(errors)}")
            for tid, err in errors[:3]:
                print(f"    Thread {tid}: {err}")
            continue

        stats = analyze_timings(timings, num_threads, elapsed)
        if stats:
            all_stats.append(stats)
            print(f"  Throughput: {stats['throughput']:.1f} ops/s")
            print(f"  Mean latency: {stats['total_mean_ms']:.2f} ms")

    # Print full report
    print_report(all_stats)

    # Assertions for test pass/fail
    assert len(all_stats) == len(thread_counts), f"Expected {len(thread_counts)} results, got {len(all_stats)}"

    # Verify we have data
    for stats in all_stats:
        assert stats['total_ops'] > 0, "No operations completed"

    print("\n" + "=" * 80)
    print("PASS: Bottleneck profiling completed successfully")
    print("=" * 80)

    return all_stats


if __name__ == '__main__':
    test_bottleneck_profiling()
