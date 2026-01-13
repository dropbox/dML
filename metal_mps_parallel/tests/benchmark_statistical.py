#!/usr/bin/env python3
"""
R1: Reproducible Statistical Benchmark Suite

Addresses Reviewer Objection #1: No reproducible benchmark suite

Features:
- N=30 trials minimum per configuration
- Statistical metrics: mean, std, min, max, p50, p95, p99
- Hardware specs programmatically collected
- Cold start vs warm cache comparison
- JSON output for CI regression tracking

Usage:
    python tests/benchmark_statistical.py
    python tests/benchmark_statistical.py --trials 50 --output results.json
    python tests/benchmark_statistical.py --threads 1,2,4,8 --model linear
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

# Early validation
try:
    import torch
    if not torch.backends.mps.is_available():
        print("ERROR: MPS backend not available", file=sys.stderr)
        sys.exit(2)
except ImportError:
    print("ERROR: PyTorch not found", file=sys.stderr)
    sys.exit(2)


@dataclass
class SystemInfo:
    """Hardware and software specifications."""
    chip: str
    chip_cores: int
    gpu_cores: Optional[int]
    ram_gb: float
    macos_version: str
    macos_build: str
    pytorch_version: str
    metal_support: bool
    mps_available: bool
    timestamp_utc: str
    hostname: str


@dataclass
class TrialStats:
    """Statistical summary of benchmark trials."""
    n_trials: int
    mean_ns: float
    std_ns: float
    min_ns: float
    max_ns: float
    p50_ns: float
    p95_ns: float
    p99_ns: float
    coefficient_of_variation: float  # std/mean as percentage


@dataclass
class BenchmarkResult:
    """Complete benchmark result with stats and context."""
    model_type: str
    model_config: Dict[str, Any]
    thread_count: int
    iterations_per_thread: int
    warmup_iterations: int
    cold_start_stats: Optional[TrialStats]
    warm_cache_stats: TrialStats
    total_ops: int
    ops_per_second: float
    scaling_efficiency: Optional[float]  # vs single thread baseline


def get_system_info() -> SystemInfo:
    """Collect hardware specifications programmatically."""
    # Get chip info from sysctl
    try:
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        chip = "Unknown"

    # Try to get Apple Silicon chip name more accurately
    try:
        hw_model = subprocess.check_output(
            ["sysctl", "-n", "hw.model"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        # Also try IORegistry for detailed chip info
        ioreg_output = subprocess.check_output(
            ["system_profiler", "SPHardwareDataType", "-json"],
            stderr=subprocess.DEVNULL
        ).decode()
        hw_data = json.loads(ioreg_output)
        hw_info = hw_data.get("SPHardwareDataType", [{}])[0]
        chip_type = hw_info.get("chip_type", chip)
        if chip_type:
            chip = chip_type
    except Exception:
        pass

    # Get GPU core count from system profiler
    gpu_cores = None
    try:
        sp_output = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            stderr=subprocess.DEVNULL
        ).decode()
        sp_data = json.loads(sp_output)
        displays = sp_data.get("SPDisplaysDataType", [])
        for display in displays:
            cores = display.get("sppci_cores")
            if cores:
                gpu_cores = int(cores.replace(" cores", "").strip())
                break
    except Exception:
        pass

    # Get CPU core count
    cpu_cores = os.cpu_count() or 0

    # Get RAM
    try:
        page_size = os.sysconf('SC_PAGE_SIZE')
        phys_pages = os.sysconf('SC_PHYS_PAGES')
        ram_gb = (page_size * phys_pages) / (1024 ** 3)
    except Exception:
        ram_gb = 0.0

    # Get macOS version
    mac_ver = platform.mac_ver()
    macos_version = mac_ver[0]

    # Get build number
    try:
        macos_build = subprocess.check_output(
            ["sw_vers", "-buildVersion"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        macos_build = "Unknown"

    # Get hostname
    try:
        hostname = subprocess.check_output(
            ["hostname", "-s"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        hostname = "Unknown"

    return SystemInfo(
        chip=chip,
        chip_cores=cpu_cores,
        gpu_cores=gpu_cores,
        ram_gb=round(ram_gb, 1),
        macos_version=macos_version,
        macos_build=macos_build,
        pytorch_version=torch.__version__,
        metal_support=True,  # We already verified MPS is available
        mps_available=torch.backends.mps.is_available(),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        hostname=hostname,
    )


def compute_stats(times_ns: List[float]) -> TrialStats:
    """Compute statistical summary of timing data."""
    n = len(times_ns)
    if n == 0:
        return TrialStats(0, 0, 0, 0, 0, 0, 0, 0, 0)

    mean = statistics.mean(times_ns)
    std = statistics.stdev(times_ns) if n > 1 else 0.0

    sorted_times = sorted(times_ns)
    p50_idx = n // 2
    p95_idx = int(n * 0.95)
    p99_idx = int(n * 0.99)

    cv = (std / mean * 100) if mean > 0 else 0.0

    return TrialStats(
        n_trials=n,
        mean_ns=mean,
        std_ns=std,
        min_ns=min(times_ns),
        max_ns=max(times_ns),
        p50_ns=sorted_times[p50_idx],
        p95_ns=sorted_times[min(p95_idx, n-1)],
        p99_ns=sorted_times[min(p99_idx, n-1)],
        coefficient_of_variation=cv,
    )


def run_benchmark_trials(
    model_type: str,
    thread_count: int,
    n_trials: int,
    iterations_per_thread: int,
    warmup_iterations: int,
    model_config: Dict[str, Any],
    include_cold_start: bool = False,
) -> BenchmarkResult:
    """
    Run benchmark with multiple trials for statistical rigor.

    Each trial:
    1. Creates fresh model and input
    2. Runs warmup iterations (not timed)
    3. Runs timed iterations
    4. Records total time

    Returns full statistical analysis.
    """

    code = f'''
import torch
import torch.nn as nn
import threading
import time
import json
import sys

# Configuration
MODEL_TYPE = "{model_type}"
THREAD_COUNT = {thread_count}
N_TRIALS = {n_trials}
ITERATIONS = {iterations_per_thread}
WARMUP = {warmup_iterations}
INCLUDE_COLD = {include_cold_start}
CONFIG = {json.dumps(model_config)}

# Initialize MPS
torch.zeros(1, device="mps")
torch.mps.synchronize()

def make_model():
    if MODEL_TYPE == "linear":
        return nn.Linear(CONFIG["in_features"], CONFIG["out_features"]).to("mps")
    elif MODEL_TYPE == "mlp":
        return nn.Sequential(
            nn.Linear(CONFIG["in_features"], CONFIG["hidden"]),
            nn.ReLU(),
            nn.Linear(CONFIG["hidden"], CONFIG["out_features"]),
        ).to("mps")
    elif MODEL_TYPE == "matmul":
        # Just return None - we'll use raw matmul
        return None
    else:
        raise ValueError(f"Unknown model type: {{MODEL_TYPE}}")

def make_input():
    if MODEL_TYPE == "matmul":
        size = CONFIG.get("matrix_size", 512)
        return torch.randn(size, size, device="mps")
    else:
        return torch.randn(CONFIG["batch_size"], CONFIG["in_features"], device="mps")

def run_single_trial():
    """Run one complete trial with all threads."""
    results = []
    lock = threading.Lock()

    def worker(tid):
        try:
            model = make_model()
            if model:
                model.eval()

            for i in range(ITERATIONS):
                x = make_input()
                with torch.no_grad():
                    if model:
                        _ = model(x)
                    else:
                        _ = torch.matmul(x, x)
                    torch.mps.synchronize()
                with lock:
                    results.append(1)
        except Exception as e:
            with lock:
                results.append(0)  # Record failure

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(THREAD_COUNT)]

    start = time.perf_counter_ns()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed_ns = time.perf_counter_ns() - start

    success_count = sum(results)
    return elapsed_ns, success_count

# Warmup phase (not timed, not counted in trials)
for _ in range(WARMUP):
    run_single_trial()

# Cold start measurement (fresh model, first invocation)
cold_times = []
if INCLUDE_COLD:
    # Clear any cached data
    torch.mps.empty_cache()

    # Run a few cold trials
    for _ in range(min(5, N_TRIALS)):
        elapsed, _ = run_single_trial()
        cold_times.append(elapsed)

# Warm cache trials
warm_times = []
total_successes = 0
for _ in range(N_TRIALS):
    elapsed, successes = run_single_trial()
    warm_times.append(elapsed)
    total_successes += successes

# Output results
expected_ops = THREAD_COUNT * ITERATIONS * N_TRIALS
print(json.dumps({{
    "warm_times_ns": warm_times,
    "cold_times_ns": cold_times if cold_times else None,
    "total_successes": total_successes,
    "expected_ops": expected_ops,
}}))
'''

    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        if result.returncode != 0:
            raise RuntimeError(f"Subprocess failed: {result.stderr}")

        data = json.loads(result.stdout.strip())
        warm_times = data["warm_times_ns"]
        cold_times = data.get("cold_times_ns")
        total_ops = data["total_successes"]

        warm_stats = compute_stats(warm_times)
        cold_stats = compute_stats(cold_times) if cold_times else None

        # Calculate ops/second from mean time
        total_ops_per_trial = thread_count * iterations_per_thread
        mean_time_seconds = warm_stats.mean_ns / 1e9
        ops_per_second = total_ops_per_trial / mean_time_seconds if mean_time_seconds > 0 else 0

        return BenchmarkResult(
            model_type=model_type,
            model_config=model_config,
            thread_count=thread_count,
            iterations_per_thread=iterations_per_thread,
            warmup_iterations=warmup_iterations,
            cold_start_stats=cold_stats,
            warm_cache_stats=warm_stats,
            total_ops=total_ops,
            ops_per_second=ops_per_second,
            scaling_efficiency=None,  # Computed later
        )

    except subprocess.TimeoutExpired:
        raise RuntimeError("Benchmark timed out")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse benchmark output: {e}")


def format_time(ns: float) -> str:
    """Format nanoseconds to human-readable string."""
    if ns >= 1e9:
        return f"{ns/1e9:.3f}s"
    elif ns >= 1e6:
        return f"{ns/1e6:.3f}ms"
    elif ns >= 1e3:
        return f"{ns/1e3:.3f}μs"
    else:
        return f"{ns:.0f}ns"


def print_markdown_report(
    system_info: SystemInfo,
    results: List[BenchmarkResult],
    baseline_ops: float,
) -> None:
    """Print results as markdown table."""

    print("\n# Statistical Benchmark Report")
    print(f"\n**Generated**: {system_info.timestamp_utc}")
    print(f"\n## System Specifications")
    print(f"| Property | Value |")
    print(f"|----------|-------|")
    print(f"| Chip | {system_info.chip} |")
    print(f"| CPU Cores | {system_info.chip_cores} |")
    if system_info.gpu_cores:
        print(f"| GPU Cores | {system_info.gpu_cores} |")
    print(f"| RAM | {system_info.ram_gb:.1f} GB |")
    print(f"| macOS | {system_info.macos_version} ({system_info.macos_build}) |")
    print(f"| PyTorch | {system_info.pytorch_version} |")
    print(f"| Hostname | {system_info.hostname} |")

    print(f"\n## Benchmark Configuration")
    if results:
        r = results[0]
        print(f"- **Model**: {r.model_type}")
        print(f"- **Config**: {r.model_config}")
        print(f"- **Trials**: {r.warm_cache_stats.n_trials}")
        print(f"- **Iterations/thread**: {r.iterations_per_thread}")
        print(f"- **Warmup**: {r.warmup_iterations} iterations")

    print(f"\n## Results (Warm Cache)")
    print("| Threads | Mean | Std | Min | Max | P50 | P95 | P99 | CV% | ops/s |")
    print("|---------|------|-----|-----|-----|-----|-----|-----|-----|-------|")

    for r in results:
        s = r.warm_cache_stats
        print(f"| {r.thread_count} | {format_time(s.mean_ns)} | {format_time(s.std_ns)} | "
              f"{format_time(s.min_ns)} | {format_time(s.max_ns)} | {format_time(s.p50_ns)} | "
              f"{format_time(s.p95_ns)} | {format_time(s.p99_ns)} | {s.coefficient_of_variation:.1f}% | "
              f"{r.ops_per_second:.0f} |")

    # Scaling efficiency table
    print(f"\n## Scaling Efficiency")
    print("| Threads | ops/s | Speedup | Efficiency |")
    print("|---------|-------|---------|------------|")

    for r in results:
        speedup = r.ops_per_second / baseline_ops if baseline_ops > 0 else 0
        efficiency = (speedup / r.thread_count * 100) if r.thread_count > 0 else 0
        print(f"| {r.thread_count} | {r.ops_per_second:.0f} | {speedup:.2f}x | {efficiency:.1f}% |")


def main():
    parser = argparse.ArgumentParser(
        description="Reproducible Statistical Benchmark Suite (R1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python benchmark_statistical.py
    python benchmark_statistical.py --trials 50 --output results.json
    python benchmark_statistical.py --threads 1,2,4,8 --model matmul
        """
    )
    parser.add_argument("--trials", type=int, default=30,
                        help="Number of trials per configuration (default: 30)")
    parser.add_argument("--iterations", type=int, default=50,
                        help="Iterations per thread per trial (default: 50)")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Warmup iterations before timing (default: 5)")
    parser.add_argument("--threads", default="1,2,4,8",
                        help="Comma-separated thread counts (default: 1,2,4,8)")
    parser.add_argument("--model", choices=["linear", "mlp", "matmul"], default="matmul",
                        help="Model type to benchmark (default: matmul)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for linear/mlp (default: 32)")
    parser.add_argument("--in-features", type=int, default=512,
                        help="Input features for linear/mlp (default: 512)")
    parser.add_argument("--out-features", type=int, default=512,
                        help="Output features for linear/mlp (default: 512)")
    parser.add_argument("--hidden", type=int, default=1024,
                        help="Hidden size for mlp (default: 1024)")
    parser.add_argument("--matrix-size", type=int, default=512,
                        help="Matrix size for matmul (default: 512)")
    parser.add_argument("--cold-start", action="store_true",
                        help="Include cold start measurements")
    parser.add_argument("--output", type=str,
                        help="JSON output file path")
    parser.add_argument("--quiet", action="store_true",
                        help="Only output JSON, no markdown")

    args = parser.parse_args()

    # Parse thread counts
    thread_counts = sorted(set(int(x) for x in args.threads.split(",") if x.strip()))
    if not thread_counts or any(t <= 0 for t in thread_counts):
        print(f"ERROR: Invalid thread counts: {args.threads}", file=sys.stderr)
        sys.exit(1)

    # Build model config
    model_config = {
        "batch_size": args.batch_size,
        "in_features": args.in_features,
        "out_features": args.out_features,
        "hidden": args.hidden,
        "matrix_size": args.matrix_size,
    }

    # Collect system info
    system_info = get_system_info()

    if not args.quiet:
        print(f"Running {args.model} benchmark with {args.trials} trials...")
        print(f"Thread counts: {thread_counts}")
        print(f"System: {system_info.chip}, {system_info.ram_gb}GB RAM, macOS {system_info.macos_version}")
        print()

    # Run benchmarks
    results = []
    baseline_ops = 0.0

    for thread_count in thread_counts:
        if not args.quiet:
            print(f"  Benchmarking {thread_count} threads...", end=" ", flush=True)

        try:
            result = run_benchmark_trials(
                model_type=args.model,
                thread_count=thread_count,
                n_trials=args.trials,
                iterations_per_thread=args.iterations,
                warmup_iterations=args.warmup,
                model_config=model_config,
                include_cold_start=args.cold_start,
            )

            # Track baseline for efficiency calculation
            if thread_count == 1:
                baseline_ops = result.ops_per_second

            # Calculate scaling efficiency
            if baseline_ops > 0:
                speedup = result.ops_per_second / baseline_ops
                result.scaling_efficiency = speedup / thread_count * 100

            results.append(result)

            if not args.quiet:
                s = result.warm_cache_stats
                print(f"mean={format_time(s.mean_ns)}, std={format_time(s.std_ns)}, "
                      f"ops/s={result.ops_per_second:.0f}")

        except Exception as e:
            if not args.quiet:
                print(f"FAILED: {e}")
            # Create a failed result
            results.append(BenchmarkResult(
                model_type=args.model,
                model_config=model_config,
                thread_count=thread_count,
                iterations_per_thread=args.iterations,
                warmup_iterations=args.warmup,
                cold_start_stats=None,
                warm_cache_stats=TrialStats(0, 0, 0, 0, 0, 0, 0, 0, 0),
                total_ops=0,
                ops_per_second=0,
                scaling_efficiency=None,
            ))

    # Print markdown report
    if not args.quiet:
        print_markdown_report(system_info, results, baseline_ops)

    # Prepare JSON output
    output_data = {
        "benchmark_version": "1.0",
        "system_info": asdict(system_info),
        "configuration": {
            "model_type": args.model,
            "model_config": model_config,
            "n_trials": args.trials,
            "iterations_per_thread": args.iterations,
            "warmup_iterations": args.warmup,
            "thread_counts": thread_counts,
            "cold_start_included": args.cold_start,
        },
        "results": [
            {
                "thread_count": r.thread_count,
                "warm_cache_stats": asdict(r.warm_cache_stats),
                "cold_start_stats": asdict(r.cold_start_stats) if r.cold_start_stats else None,
                "total_ops": r.total_ops,
                "ops_per_second": r.ops_per_second,
                "scaling_efficiency_percent": r.scaling_efficiency,
            }
            for r in results
        ],
        "summary": {
            "baseline_ops_per_second": baseline_ops,
            "max_thread_count": max(thread_counts),
            "max_scaling_efficiency": max((r.scaling_efficiency or 0) for r in results),
        },
    }

    # Write JSON output
    if args.output:
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        if not args.quiet:
            print(f"\nJSON output written to: {args.output}")
    elif args.quiet:
        print(json.dumps(output_data, indent=2))

    # Determine exit code
    # Success if all thread counts completed trials
    all_passed = all(r.warm_cache_stats.n_trials == args.trials for r in results)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
