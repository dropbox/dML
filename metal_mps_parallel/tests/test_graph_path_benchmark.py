#!/usr/bin/env python3
"""
Benchmark comparing MPS_FORCE_GRAPH_PATH=1 vs default mode.

Tests:
1. Single-threaded performance (simple and complex models)
2. Multi-threaded performance with Semaphore(2) throttling
3. Thread safety verification

Run:
    # Compare both modes
    python3 tests/test_graph_path_benchmark.py

    # With crash checking
    ./scripts/run_test_with_crash_check.sh python3 tests/test_graph_path_benchmark.py
"""

import torch
import torch.nn as nn
import threading
import time
import os
import sys
import subprocess
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class BenchmarkResult:
    name: str
    mode: str
    samples_per_sec: float
    iterations: int
    batch_size: int
    elapsed: float
    errors: int


class SimpleModel(nn.Module):
    """Simple 2-layer MLP"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class ConvModel(nn.Module):
    """More complex conv model"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(32, 64, 3, padding=1)
        self.conv2 = nn.Conv1d(64, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(16)
        self.fc = nn.Linear(64 * 16, 128)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def benchmark_single_thread(model, input_shape, iterations=500, warmup=50):
    """Benchmark single-threaded inference"""
    x = torch.randn(*input_shape, device='mps')

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            y = model(x)
            _ = y.sum().cpu()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        with torch.no_grad():
            y = model(x)
            _ = y.sum().cpu()
    elapsed = time.perf_counter() - start

    batch_size = input_shape[0]
    return (iterations * batch_size) / elapsed, elapsed


def benchmark_multi_thread(model, input_shape, num_threads=4, iterations_per_thread=100):
    """Benchmark multi-threaded inference with Semaphore(2) throttling"""
    throttle = threading.Semaphore(2)
    errors = []
    total_ops = [0]
    lock = threading.Lock()

    def worker(thread_id):
        ops = 0
        for _ in range(iterations_per_thread):
            with throttle:
                try:
                    x = torch.randn(*input_shape, device='mps')
                    with torch.no_grad():
                        y = model(x)
                        _ = y.sum().cpu()
                    ops += 1
                except Exception as e:
                    errors.append((thread_id, str(e)))
        with lock:
            total_ops[0] += ops

    threads = []
    start = time.perf_counter()
    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
    elapsed = time.perf_counter() - start

    batch_size = input_shape[0]
    samples_per_sec = (total_ops[0] * batch_size) / elapsed if total_ops[0] > 0 else 0
    return samples_per_sec, elapsed, len(errors)


def run_benchmark_suite(mode_name="default"):
    """Run full benchmark suite"""
    results = []

    print(f"\n{'='*60}")
    print(f"BENCHMARK: {mode_name.upper()} mode")
    print(f"MPS_FORCE_GRAPH_PATH={os.environ.get('MPS_FORCE_GRAPH_PATH', 'not set')}")
    print(f"{'='*60}\n")

    # Test 1: Simple model (single-threaded)
    print("1. SimpleModel (single-thread, batch=32)...")
    model = SimpleModel().to('mps').eval()
    rates = []
    for _ in range(3):
        rate, _ = benchmark_single_thread(model, (32, 256))
        rates.append(rate)
    avg_rate = sum(rates) / len(rates)
    print(f"   {avg_rate:.0f} samples/s (avg of 3 trials)")
    results.append(BenchmarkResult("SimpleModel-1T", mode_name, avg_rate, 500, 32, 0, 0))

    # Test 2: Conv model (single-threaded)
    print("2. ConvModel (single-thread, batch=16)...")
    model = ConvModel().to('mps').eval()
    rates = []
    for _ in range(3):
        rate, _ = benchmark_single_thread(model, (16, 32, 256))
        rates.append(rate)
    avg_rate = sum(rates) / len(rates)
    print(f"   {avg_rate:.0f} samples/s (avg of 3 trials)")
    results.append(BenchmarkResult("ConvModel-1T", mode_name, avg_rate, 500, 16, 0, 0))

    # Test 3: Conv model (multi-threaded)
    print("3. ConvModel (4 threads, Semaphore(2))...")
    model = ConvModel().to('mps').eval()
    rates = []
    total_errors = 0
    for _ in range(3):
        rate, elapsed, errors = benchmark_multi_thread(model, (16, 32, 256))
        rates.append(rate)
        total_errors += errors
    avg_rate = sum(rates) / len(rates)
    print(f"   {avg_rate:.0f} samples/s (avg of 3 trials), errors={total_errors}")
    results.append(BenchmarkResult("ConvModel-4T", mode_name, avg_rate, 400, 16, 0, total_errors))

    return results


def main():
    graph_mode = os.environ.get("MPS_FORCE_GRAPH_PATH", "0")
    mode_name = "graph_path" if graph_mode == "1" else "default"

    print("="*60)
    print("MPS_FORCE_GRAPH_PATH Benchmark")
    print("="*60)
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {torch.backends.mps.is_available()}")
    print(f"Mode: {mode_name}")

    results = run_benchmark_suite(mode_name)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Test':<20} {'Mode':<12} {'Samples/s':>12} {'Errors':>8}")
    print("-"*60)
    for r in results:
        print(f"{r.name:<20} {r.mode:<12} {r.samples_per_sec:>12.0f} {r.errors:>8}")

    # Determine pass/fail
    all_passed = all(r.errors == 0 for r in results)
    if all_passed:
        print("\n" + "="*60)
        print("PASS: All benchmarks completed without errors")
        print("="*60)
        return 0
    else:
        print("\n" + "="*60)
        print("FAIL: Some benchmarks had errors")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
