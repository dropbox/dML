#!/usr/bin/env python3
"""
Comprehensive MPS Parallel Testing Suite

This suite provides HONEST measurements of what works and what doesn't.
No inflated claims - just facts.

Test Categories:
1. CORRECTNESS - Do outputs match CPU/sequential?
2. THREAD_SAFETY - Can we run N threads without crashes?
3. THROUGHPUT - What is the actual ops/s at each thread count?
4. STRESS - Does it survive long runs?

Usage:
    python tests/comprehensive_test_suite.py
    python tests/comprehensive_test_suite.py --category correctness
    python tests/comprehensive_test_suite.py --threads 1,2,4,8 --iterations 100
"""

import argparse
import json
import os
import statistics
import sys
import time
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn as nn

# Ensure MPS available
if not torch.backends.mps.is_available():
    print("ERROR: MPS not available", file=sys.stderr)
    sys.exit(1)

DEVICE = torch.device("mps")


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

class SmallMLP(nn.Module):
    """Small model - overhead dominated."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(256, 256)

    def forward(self, x):
        return self.fc(x)


class MediumMLP(nn.Module):
    """Medium model - mixed compute/overhead."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
        )

    def forward(self, x):
        return self.layers(x)


class LargeTransformer(nn.Module):
    """Large model - compute dominated (same as claimed 3.64x tests)."""
    def __init__(self, d_model=512, nhead=8, layers=6):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward=d_model*4,
                batch_first=True, dropout=0
            )
            for _ in range(layers)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class SimpleMatmul(nn.Module):
    """Raw matmul - pure GPU compute."""
    def __init__(self, size=512):
        super().__init__()
        self.size = size
        self.register_buffer('weight', torch.randn(size, size))

    def forward(self, x):
        return torch.matmul(x, self.weight)


MODELS = {
    'small_mlp': (SmallMLP, lambda: torch.randn(32, 256, device=DEVICE)),
    'medium_mlp': (MediumMLP, lambda: torch.randn(32, 512, device=DEVICE)),
    'large_transformer': (LargeTransformer, lambda: torch.randn(4, 128, 512, device=DEVICE)),
    'matmul': (SimpleMatmul, lambda: torch.randn(512, 512, device=DEVICE)),
}


@dataclass
class TestResult:
    """Single test result."""
    name: str
    category: str
    status: str  # PASS, FAIL, ERROR
    message: str
    metrics: Dict[str, Any]
    duration_ms: float


@dataclass
class SuiteReport:
    """Complete test suite report."""
    timestamp: str
    system: Dict[str, Any]
    total_tests: int
    passed: int
    failed: int
    errors: int
    results: List[Dict[str, Any]]


# =============================================================================
# CORRECTNESS TESTS
# =============================================================================

def test_correctness_single_thread(model_name: str) -> TestResult:
    """Test that MPS output matches CPU for single thread."""
    start = time.perf_counter()

    try:
        model_cls, input_fn = MODELS[model_name]

        # Create models
        model_cpu = model_cls().cpu().eval()
        model_mps = model_cls().to(DEVICE).eval()

        # Copy weights
        model_mps.load_state_dict(model_cpu.state_dict())

        # Create input
        x_mps = input_fn()
        x_cpu = x_mps.cpu()

        # Forward pass
        with torch.no_grad():
            y_cpu = model_cpu(x_cpu)
            y_mps = model_mps(x_mps)
            torch.mps.synchronize()

        # Compare
        y_mps_cpu = y_mps.cpu()
        max_diff = (y_cpu - y_mps_cpu).abs().max().item()
        mean_diff = (y_cpu - y_mps_cpu).abs().mean().item()

        # Tolerance depends on dtype
        tolerance = 1e-3 if y_cpu.dtype == torch.float32 else 1e-2
        passed = max_diff < tolerance

        return TestResult(
            name=f"correctness_single_{model_name}",
            category="correctness",
            status="PASS" if passed else "FAIL",
            message=f"max_diff={max_diff:.6f}, tol={tolerance}",
            metrics={"max_diff": max_diff, "mean_diff": mean_diff, "tolerance": tolerance},
            duration_ms=(time.perf_counter() - start) * 1000
        )

    except Exception as e:
        return TestResult(
            name=f"correctness_single_{model_name}",
            category="correctness",
            status="ERROR",
            message=str(e),
            metrics={},
            duration_ms=(time.perf_counter() - start) * 1000
        )


def test_correctness_parallel(model_name: str, num_threads: int) -> TestResult:
    """Test correctness under parallel load."""
    start = time.perf_counter()

    try:
        model_cls, input_fn = MODELS[model_name]

        # Reference on CPU
        model_cpu = model_cls().cpu().eval()

        # Per-thread MPS models
        models_mps = [model_cls().to(DEVICE).eval() for _ in range(num_threads)]
        for m in models_mps:
            m.load_state_dict(model_cpu.state_dict())

        results = []
        errors = []

        def worker(tid):
            try:
                x_mps = input_fn()
                x_cpu = x_mps.cpu()

                with torch.no_grad():
                    y_cpu = model_cpu(x_cpu)
                    y_mps = models_mps[tid](x_mps)
                    torch.mps.synchronize()

                y_mps_cpu = y_mps.cpu()
                max_diff = (y_cpu - y_mps_cpu).abs().max().item()
                return tid, max_diff, None
            except Exception as e:
                return tid, None, str(e)

        # Run in parallel
        with ThreadPoolExecutor(max_workers=num_threads) as ex:
            futures = [ex.submit(worker, i) for i in range(num_threads)]
            for f in as_completed(futures):
                tid, diff, err = f.result()
                if err:
                    errors.append(err)
                else:
                    results.append(diff)

        if errors:
            return TestResult(
                name=f"correctness_parallel_{model_name}_{num_threads}t",
                category="correctness",
                status="ERROR",
                message=f"{len(errors)} errors: {errors[0]}",
                metrics={"errors": errors},
                duration_ms=(time.perf_counter() - start) * 1000
            )

        max_diff = max(results)
        tolerance = 1e-3
        passed = max_diff < tolerance

        return TestResult(
            name=f"correctness_parallel_{model_name}_{num_threads}t",
            category="correctness",
            status="PASS" if passed else "FAIL",
            message=f"max_diff={max_diff:.6f} across {num_threads} threads",
            metrics={"max_diff": max_diff, "all_diffs": results},
            duration_ms=(time.perf_counter() - start) * 1000
        )

    except Exception as e:
        return TestResult(
            name=f"correctness_parallel_{model_name}_{num_threads}t",
            category="correctness",
            status="ERROR",
            message=str(e),
            metrics={},
            duration_ms=(time.perf_counter() - start) * 1000
        )


# =============================================================================
# THREAD SAFETY TESTS
# =============================================================================

def test_thread_safety(model_name: str, num_threads: int, iterations: int) -> TestResult:
    """Test that N threads can run without crashes."""
    start = time.perf_counter()

    try:
        model_cls, input_fn = MODELS[model_name]
        models = [model_cls().to(DEVICE).eval() for _ in range(num_threads)]

        completed = [0] * num_threads
        errors = []

        def worker(tid):
            try:
                for i in range(iterations):
                    x = input_fn()
                    with torch.no_grad():
                        _ = models[tid](x)
                    torch.mps.synchronize()
                    completed[tid] += 1
            except Exception as e:
                errors.append((tid, str(e)))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        total_completed = sum(completed)
        expected = num_threads * iterations

        if errors:
            return TestResult(
                name=f"thread_safety_{model_name}_{num_threads}t",
                category="thread_safety",
                status="FAIL",
                message=f"{len(errors)} errors: {errors[0][1]}",
                metrics={"completed": total_completed, "expected": expected, "errors": errors},
                duration_ms=(time.perf_counter() - start) * 1000
            )

        passed = total_completed == expected
        return TestResult(
            name=f"thread_safety_{model_name}_{num_threads}t",
            category="thread_safety",
            status="PASS" if passed else "FAIL",
            message=f"{total_completed}/{expected} completed",
            metrics={"completed": total_completed, "expected": expected},
            duration_ms=(time.perf_counter() - start) * 1000
        )

    except Exception as e:
        return TestResult(
            name=f"thread_safety_{model_name}_{num_threads}t",
            category="thread_safety",
            status="ERROR",
            message=f"Crash: {str(e)}",
            metrics={"traceback": traceback.format_exc()},
            duration_ms=(time.perf_counter() - start) * 1000
        )


# =============================================================================
# THROUGHPUT TESTS
# =============================================================================

def test_throughput(model_name: str, num_threads: int, iterations: int) -> TestResult:
    """Measure actual throughput at given thread count."""
    start = time.perf_counter()

    try:
        model_cls, input_fn = MODELS[model_name]
        models = [model_cls().to(DEVICE).eval() for _ in range(num_threads)]

        # Warmup
        for m in models:
            x = input_fn()
            with torch.no_grad():
                _ = m(x)
        torch.mps.synchronize()

        completed = [0] * num_threads
        thread_times = [[] for _ in range(num_threads)]

        def worker(tid):
            for i in range(iterations):
                x = input_fn()
                op_start = time.perf_counter()
                with torch.no_grad():
                    _ = models[tid](x)
                torch.mps.synchronize()
                op_end = time.perf_counter()
                thread_times[tid].append(op_end - op_start)
                completed[tid] += 1

        # Time the parallel execution
        torch.mps.synchronize()
        run_start = time.perf_counter()

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        run_end = time.perf_counter()

        total_completed = sum(completed)
        wall_time = run_end - run_start
        throughput = total_completed / wall_time

        # Per-op timing stats
        all_times = [t for times in thread_times for t in times]
        mean_op_time = statistics.mean(all_times) * 1000 if all_times else 0

        return TestResult(
            name=f"throughput_{model_name}_{num_threads}t",
            category="throughput",
            status="PASS",
            message=f"{throughput:.1f} ops/s ({total_completed} ops in {wall_time:.2f}s)",
            metrics={
                "throughput_ops_s": throughput,
                "total_ops": total_completed,
                "wall_time_s": wall_time,
                "mean_op_time_ms": mean_op_time,
            },
            duration_ms=(time.perf_counter() - start) * 1000
        )

    except Exception as e:
        return TestResult(
            name=f"throughput_{model_name}_{num_threads}t",
            category="throughput",
            status="ERROR",
            message=str(e),
            metrics={},
            duration_ms=(time.perf_counter() - start) * 1000
        )


def test_throughput_scaling(model_name: str, iterations: int) -> TestResult:
    """Test throughput scaling from 1 to 8 threads."""
    start = time.perf_counter()

    try:
        results = {}

        for n in [1, 2, 4, 8]:
            r = test_throughput(model_name, n, iterations)
            if r.status == "PASS":
                results[n] = r.metrics["throughput_ops_s"]
            else:
                results[n] = 0

        baseline = results.get(1, 1)
        scaling = {n: results[n] / baseline if baseline > 0 else 0 for n in results}

        # Determine if scaling is meaningful
        max_speedup = max(scaling.values()) if scaling else 0
        best_threads = max(scaling, key=scaling.get) if scaling else 1

        status = "PASS" if max_speedup >= 1.0 else "WARN"

        return TestResult(
            name=f"throughput_scaling_{model_name}",
            category="throughput",
            status=status,
            message=f"Best: {max_speedup:.2f}x at {best_threads} threads",
            metrics={
                "throughput_by_threads": results,
                "scaling_by_threads": scaling,
                "max_speedup": max_speedup,
                "best_threads": best_threads,
            },
            duration_ms=(time.perf_counter() - start) * 1000
        )

    except Exception as e:
        return TestResult(
            name=f"throughput_scaling_{model_name}",
            category="throughput",
            status="ERROR",
            message=str(e),
            metrics={},
            duration_ms=(time.perf_counter() - start) * 1000
        )


# =============================================================================
# STRESS TESTS
# =============================================================================

def test_stress(model_name: str, num_threads: int, duration_s: float) -> TestResult:
    """Run continuous stress test for specified duration."""
    start = time.perf_counter()

    try:
        model_cls, input_fn = MODELS[model_name]
        models = [model_cls().to(DEVICE).eval() for _ in range(num_threads)]

        stop_flag = threading.Event()
        completed = [0] * num_threads
        errors = []

        def worker(tid):
            while not stop_flag.is_set():
                try:
                    x = input_fn()
                    with torch.no_grad():
                        _ = models[tid](x)
                    torch.mps.synchronize()
                    completed[tid] += 1
                except Exception as e:
                    errors.append((tid, str(e)))
                    break

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()

        time.sleep(duration_s)
        stop_flag.set()

        for t in threads:
            t.join()

        total_completed = sum(completed)
        actual_duration = time.perf_counter() - start
        throughput = total_completed / actual_duration

        if errors:
            return TestResult(
                name=f"stress_{model_name}_{num_threads}t_{duration_s}s",
                category="stress",
                status="FAIL",
                message=f"Crashed after {total_completed} ops: {errors[0][1]}",
                metrics={"completed": total_completed, "errors": errors},
                duration_ms=(time.perf_counter() - start) * 1000
            )

        return TestResult(
            name=f"stress_{model_name}_{num_threads}t_{duration_s}s",
            category="stress",
            status="PASS",
            message=f"{total_completed} ops in {actual_duration:.1f}s ({throughput:.1f} ops/s)",
            metrics={"completed": total_completed, "throughput": throughput},
            duration_ms=(time.perf_counter() - start) * 1000
        )

    except Exception as e:
        return TestResult(
            name=f"stress_{model_name}_{num_threads}t_{duration_s}s",
            category="stress",
            status="ERROR",
            message=str(e),
            metrics={},
            duration_ms=(time.perf_counter() - start) * 1000
        )


# =============================================================================
# SUITE RUNNER
# =============================================================================

def get_system_info() -> Dict[str, Any]:
    """Collect system information."""
    import subprocess

    try:
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except:
        chip = "Unknown"

    return {
        "chip": chip,
        "cpu_cores": os.cpu_count(),
        "pytorch": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
    }


def run_suite(categories: List[str], models: List[str],
              thread_counts: List[int], iterations: int) -> SuiteReport:
    """Run the complete test suite."""
    results: List[TestResult] = []

    print("=" * 70)
    print("COMPREHENSIVE MPS PARALLEL TEST SUITE")
    print("=" * 70)
    print()

    # CORRECTNESS TESTS
    if "correctness" in categories or "all" in categories:
        print("### CORRECTNESS TESTS ###")
        for model_name in models:
            print(f"  {model_name}:")

            # Single thread
            r = test_correctness_single_thread(model_name)
            print(f"    Single thread: {r.status} - {r.message}")
            results.append(r)

            # Parallel
            for n in thread_counts:
                r = test_correctness_parallel(model_name, n)
                print(f"    {n} threads:    {r.status} - {r.message}")
                results.append(r)
        print()

    # THREAD SAFETY TESTS
    if "thread_safety" in categories or "all" in categories:
        print("### THREAD SAFETY TESTS ###")
        for model_name in models:
            print(f"  {model_name}:")
            for n in thread_counts:
                r = test_thread_safety(model_name, n, iterations)
                print(f"    {n} threads: {r.status} - {r.message}")
                results.append(r)
        print()

    # THROUGHPUT TESTS
    if "throughput" in categories or "all" in categories:
        print("### THROUGHPUT TESTS ###")
        for model_name in models:
            print(f"  {model_name}:")

            # Individual thread counts
            baseline = None
            for n in thread_counts:
                r = test_throughput(model_name, n, iterations)
                throughput = r.metrics.get("throughput_ops_s", 0)
                if n == 1:
                    baseline = throughput
                    print(f"    {n} threads: {throughput:.1f} ops/s (baseline)")
                else:
                    speedup = throughput / baseline if baseline else 0
                    print(f"    {n} threads: {throughput:.1f} ops/s ({speedup:.2f}x)")
                results.append(r)

            # Scaling summary
            r = test_throughput_scaling(model_name, iterations)
            print(f"    Scaling: {r.message}")
            results.append(r)
        print()

    # STRESS TESTS
    if "stress" in categories or "all" in categories:
        print("### STRESS TESTS ###")
        for model_name in models:
            print(f"  {model_name}:")
            for n in thread_counts:
                r = test_stress(model_name, n, 5.0)  # 5 second stress test
                print(f"    {n} threads: {r.status} - {r.message}")
                results.append(r)
        print()

    # Summary
    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")
    errors = sum(1 for r in results if r.status == "ERROR")

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total:  {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Errors: {errors}")
    print()

    if failed > 0 or errors > 0:
        print("FAILURES/ERRORS:")
        for r in results:
            if r.status in ("FAIL", "ERROR"):
                print(f"  - {r.name}: {r.status} - {r.message}")

    return SuiteReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        system=get_system_info(),
        total_tests=len(results),
        passed=passed,
        failed=failed,
        errors=errors,
        results=[asdict(r) for r in results]
    )


def main():
    parser = argparse.ArgumentParser(description="Comprehensive MPS Parallel Test Suite")
    parser.add_argument("--categories", default="all",
                        help="Test categories (comma-separated): correctness,thread_safety,throughput,stress,all")
    parser.add_argument("--models", default="small_mlp,matmul,large_transformer",
                        help="Models to test (comma-separated)")
    parser.add_argument("--threads", default="1,2,4,8",
                        help="Thread counts (comma-separated)")
    parser.add_argument("--iterations", type=int, default=20,
                        help="Iterations per test (default: 20)")
    parser.add_argument("--output", help="JSON output file")

    args = parser.parse_args()

    categories = [c.strip() for c in args.categories.split(",")]
    models = [m.strip() for m in args.models.split(",")]
    thread_counts = [int(t) for t in args.threads.split(",")]

    report = run_suite(categories, models, thread_counts, args.iterations)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(asdict(report), f, indent=2)
        print(f"\nResults written to: {args.output}")

    return 0 if report.failed == 0 and report.errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
