#!/usr/bin/env python3
"""
Production Metrics Test - Measures TBD metrics from EFFICIENCY_ROADMAP.md

Metrics measured:
1. Memory growth (MB/hour) - Target: < 100 MB/hour
2. P99 latency (ms) - Target: < 50ms

Run with:
  ./scripts/run_test_with_crash_check.sh python3 tests/test_production_metrics.py

Or for extended run (more accurate memory growth estimate):
  ./scripts/run_test_with_crash_check.sh python3 tests/test_production_metrics.py --extended
"""

import argparse
import os
import sys
import threading
import time
import statistics
import tracemalloc

import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    """Simple MLP model for benchmarking."""

    def __init__(self, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(256, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 64),
        )

    def forward(self, x):
        return self.net(x)


def measure_latency(model, batch_size=32, num_samples=1000, warmup=100):
    """Measure inference latency distribution."""
    print(f"\n--- Latency Measurement (batch={batch_size}, samples={num_samples}) ---")

    # Warmup
    for _ in range(warmup):
        x = torch.randn(batch_size, 256, device='mps')
        with torch.no_grad():
            y = model(x)
        _ = y.sum().cpu()

    # Collect latency samples
    latencies = []
    for i in range(num_samples):
        x = torch.randn(batch_size, 256, device='mps')

        start = time.perf_counter()
        with torch.no_grad():
            y = model(x)
        _ = y.sum().cpu()  # Safe sync
        end = time.perf_counter()

        latencies.append((end - start) * 1000)  # Convert to ms

        if (i + 1) % 200 == 0:
            print(f"  Progress: {i+1}/{num_samples} samples")

    # Calculate statistics
    latencies.sort()
    avg_latency = statistics.mean(latencies)
    median_latency = statistics.median(latencies)
    p50 = latencies[int(len(latencies) * 0.50)]
    p90 = latencies[int(len(latencies) * 0.90)]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    min_latency = min(latencies)
    max_latency = max(latencies)

    print(f"\nLatency Results:")
    print(f"  Min:    {min_latency:.3f} ms")
    print(f"  Avg:    {avg_latency:.3f} ms")
    print(f"  Median: {median_latency:.3f} ms")
    print(f"  P50:    {p50:.3f} ms")
    print(f"  P90:    {p90:.3f} ms")
    print(f"  P95:    {p95:.3f} ms")
    print(f"  P99:    {p99:.3f} ms")
    print(f"  Max:    {max_latency:.3f} ms")

    return {
        'avg': avg_latency,
        'median': median_latency,
        'p50': p50,
        'p90': p90,
        'p95': p95,
        'p99': p99,
        'min': min_latency,
        'max': max_latency,
        'samples': num_samples,
        'batch_size': batch_size,
    }


def measure_memory_growth(model, duration_seconds=60, batch_size=32, num_threads=2):
    """Measure memory growth over time with concurrent inference."""
    print(f"\n--- Memory Growth Measurement ({duration_seconds}s, {num_threads} threads) ---")

    # Start memory tracking
    tracemalloc.start()
    torch.mps.empty_cache()

    initial_mem = tracemalloc.get_traced_memory()[0] / (1024 * 1024)  # MB
    mps_initial = torch.mps.current_allocated_memory() / (1024 * 1024)  # MB

    print(f"Initial Python memory: {initial_mem:.2f} MB")
    print(f"Initial MPS memory: {mps_initial:.2f} MB")

    # Track memory over time
    memory_samples = []
    mps_samples = []
    operations = [0]  # Use list for mutable closure
    running = True
    semaphore = threading.Semaphore(2)

    def worker():
        while running:
            with semaphore:
                x = torch.randn(batch_size, 256, device='mps')
                with torch.no_grad():
                    y = model(x)
                _ = y.sum().cpu()
                operations[0] += 1

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(num_threads)]
    for t in threads:
        t.start()

    start_time = time.time()
    sample_interval = 5  # Sample every 5 seconds

    while time.time() - start_time < duration_seconds:
        time.sleep(sample_interval)
        elapsed = time.time() - start_time
        current_mem = tracemalloc.get_traced_memory()[0] / (1024 * 1024)
        mps_mem = torch.mps.current_allocated_memory() / (1024 * 1024)
        memory_samples.append((elapsed, current_mem))
        mps_samples.append((elapsed, mps_mem))
        print(f"  {elapsed:.0f}s: Python={current_mem:.2f}MB, MPS={mps_mem:.2f}MB, ops={operations[0]}")

    running = False
    for t in threads:
        t.join(timeout=2.0)

    # Final measurements
    final_mem = tracemalloc.get_traced_memory()[0] / (1024 * 1024)
    mps_final = torch.mps.current_allocated_memory() / (1024 * 1024)
    tracemalloc.stop()

    total_time = time.time() - start_time
    python_growth = final_mem - initial_mem
    mps_growth = mps_final - mps_initial

    # Extrapolate to MB/hour
    python_growth_per_hour = (python_growth / total_time) * 3600
    mps_growth_per_hour = (mps_growth / total_time) * 3600

    print(f"\nMemory Growth Results:")
    print(f"  Duration: {total_time:.1f}s ({total_time/60:.2f} min)")
    print(f"  Operations: {operations[0]}")
    print(f"  Python memory: {initial_mem:.2f} -> {final_mem:.2f} MB (delta: {python_growth:+.2f} MB)")
    print(f"  MPS memory: {mps_initial:.2f} -> {mps_final:.2f} MB (delta: {mps_growth:+.2f} MB)")
    print(f"  Python growth rate: {python_growth_per_hour:+.2f} MB/hour")
    print(f"  MPS growth rate: {mps_growth_per_hour:+.2f} MB/hour")

    return {
        'duration_seconds': total_time,
        'operations': operations[0],
        'python_initial_mb': initial_mem,
        'python_final_mb': final_mem,
        'python_growth_mb': python_growth,
        'python_growth_mb_per_hour': python_growth_per_hour,
        'mps_initial_mb': mps_initial,
        'mps_final_mb': mps_final,
        'mps_growth_mb': mps_growth,
        'mps_growth_mb_per_hour': mps_growth_per_hour,
        'samples': memory_samples,
        'mps_samples': mps_samples,
    }


def main():
    parser = argparse.ArgumentParser(description='Production Metrics Test')
    parser.add_argument('--extended', action='store_true',
                       help='Run extended test (5 min for memory, more latency samples)')
    parser.add_argument('--latency-samples', type=int, default=1000,
                       help='Number of latency samples (default: 1000)')
    parser.add_argument('--memory-duration', type=int, default=60,
                       help='Memory test duration in seconds (default: 60)')
    args = parser.parse_args()

    if args.extended:
        args.latency_samples = 5000
        args.memory_duration = 300  # 5 minutes

    if not torch.backends.mps.is_available():
        print("ERROR: MPS not available")
        sys.exit(1)

    print("=" * 70)
    print("Production Metrics Test")
    print("=" * 70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"Mode: {'Extended' if args.extended else 'Standard'}")

    # Create model
    model = SimpleModel().to('mps').eval()

    # 1. Measure latency
    latency_results = measure_latency(
        model,
        batch_size=32,
        num_samples=args.latency_samples
    )

    # 2. Measure memory growth
    memory_results = measure_memory_growth(
        model,
        duration_seconds=args.memory_duration,
        batch_size=32,
        num_threads=2
    )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Latency assessment
    p99 = latency_results['p99']
    p99_target = 50.0  # ms
    p99_pass = p99 < p99_target
    print(f"\nP99 Latency: {p99:.3f} ms (target: <{p99_target}ms) -> {'PASS' if p99_pass else 'FAIL'}")

    # Memory growth assessment
    # Use MPS memory growth as primary metric (more relevant for GPU workloads)
    mem_growth = memory_results['mps_growth_mb_per_hour']
    mem_target = 100.0  # MB/hour
    mem_pass = mem_growth < mem_target
    print(f"Memory Growth: {mem_growth:+.2f} MB/hour (target: <{mem_target}MB/hour) -> {'PASS' if mem_pass else 'FAIL'}")

    print("\n--- Results for EFFICIENCY_ROADMAP.md ---")
    print(f"| Metric | Baseline | Target | Current |")
    print(f"|--------|----------|--------|---------|")
    print(f"| Memory growth (MB/hour) | Unknown | <100 | {mem_growth:.0f} {'✅' if mem_pass else '❌'} |")
    print(f"| P99 latency (ms) | Unknown | <50 | {p99:.1f} {'✅' if p99_pass else '❌'} |")

    # Overall result
    overall_pass = p99_pass and mem_pass
    print(f"\nOverall: {'PASS' if overall_pass else 'FAIL'}")

    sys.exit(0 if overall_pass else 1)


if __name__ == '__main__':
    main()
