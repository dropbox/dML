#!/usr/bin/env python3
"""
1-hour soak test for MPS parallel inference stability.

This test runs continuous parallel inference to detect:
- Memory leaks (CPU and GPU)
- Stability issues over time
- GPU memory fragmentation
- Handle exhaustion
- Thermal throttling effects

Run with: python tests/test_soak_1hour.py [--duration SECONDS] [--threads N]

Default: 3600 seconds (1 hour), 4 threads
Quick validation: python tests/test_soak_1hour.py --duration 60
"""

import argparse
import gc
import json
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import psutil
import torch
import torch.nn as nn


@dataclass
class SoakMetrics:
    """Metrics collected during soak test."""
    timestamp: str = ""
    elapsed_seconds: float = 0.0
    iteration: int = 0
    cpu_memory_mb: float = 0.0
    cpu_memory_peak_mb: float = 0.0
    gpu_memory_allocated_mb: float = 0.0
    gpu_memory_reserved_mb: float = 0.0
    successful_ops: int = 0
    failed_ops: int = 0
    threads_alive: int = 0
    ops_per_second: float = 0.0
    errors: List[str] = field(default_factory=list)


@dataclass
class SoakResults:
    """Final results from soak test."""
    start_time: str
    end_time: str
    duration_seconds: float
    total_iterations: int
    total_successful_ops: int
    total_failed_ops: int
    initial_cpu_memory_mb: float
    final_cpu_memory_mb: float
    peak_cpu_memory_mb: float
    memory_growth_mb: float
    initial_gpu_memory_mb: float
    final_gpu_memory_mb: float
    peak_gpu_memory_mb: float
    gpu_memory_growth_mb: float
    average_ops_per_second: float
    hardware_info: Dict
    passed: bool
    failure_reason: Optional[str] = None
    metrics_history: List[Dict] = field(default_factory=list)


class SimpleMPSModel(nn.Module):
    """Simple model for soak testing - exercises common MPS operations."""

    def __init__(self, size: int = 256):
        super().__init__()
        self.linear1 = nn.Linear(size, size * 2)
        self.ln1 = nn.LayerNorm(size * 2)
        self.linear2 = nn.Linear(size * 2, size)
        self.ln2 = nn.LayerNorm(size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.ln1(x)
        x = self.linear2(x)
        x = self.ln2(x)
        return x


# Global semaphore for throttling MPS operations (Semaphore(2) is the safe maximum)
# Allows 2 concurrent MPS operations while preventing AGX driver race condition
_mps_throttle = threading.Semaphore(2)


class SoakTestWorker:
    """Worker thread that performs continuous inference.

    Note: MPS operations are throttled via Semaphore(2) to prevent AGX driver
    race condition while allowing limited parallelism. This is the recommended
    pattern for production use with parallel inference requests.
    """

    def __init__(self, worker_id, model, batch_size=32, dim=256):
        self.worker_id = worker_id
        self.model = model
        self.batch_size = batch_size
        self.dim = dim
        self.successful_ops = 0
        self.failed_ops = 0
        self.running = False
        self.thread = None
        self.errors = []
        self.lock = threading.Lock()
        self._mps_throttle = _mps_throttle  # Explicitly bind the module semaphore to instance

    def start(self):
        """Start the worker thread."""
        self.running = True
        self.thread = threading.Thread(target=self._work_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the worker thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=10)

    def _work_loop(self):
        """Main work loop - continuous inference with throttled MPS access."""
        mps_throttle = self._mps_throttle  # Local reference for faster access
        model = self.model
        batch_size = self.batch_size
        dim = self.dim

        while self.running:
            with mps_throttle:
                x = torch.randn(batch_size, dim, device='mps')
                with torch.no_grad():
                    output = model(x)
                torch.mps.synchronize()
            with self.lock:
                self.successful_ops += 1

    def get_stats(self) -> tuple:
        """Get current stats thread-safely."""
        with self.lock:
            return self.successful_ops, self.failed_ops, list(self.errors[-10:])  # Last 10 errors


def get_hardware_info() -> Dict:
    """Collect hardware information."""
    import subprocess

    info = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
    }

    # Get chip info
    try:
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        info["chip"] = chip
    except Exception:
        info["chip"] = "unknown"

    # Get macOS version
    try:
        import platform
        info["macos_version"] = platform.mac_ver()[0]
    except Exception:
        info["macos_version"] = "unknown"

    # Get memory info
    try:
        mem = psutil.virtual_memory()
        info["total_ram_gb"] = round(mem.total / (1024**3), 1)
    except Exception:
        info["total_ram_gb"] = "unknown"

    # Get GPU core count
    try:
        output = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType"],
            stderr=subprocess.DEVNULL
        ).decode()
        for line in output.split('\n'):
            if 'Total Number of Cores' in line:
                cores = line.split(':')[1].strip()
                info["gpu_cores"] = cores
                break
    except Exception:
        info["gpu_cores"] = "unknown"

    return info


def get_memory_stats() -> tuple:
    """Get current CPU and GPU memory usage."""
    process = psutil.Process()
    cpu_mb = process.memory_info().rss / (1024 * 1024)

    gpu_allocated_mb = 0.0
    gpu_reserved_mb = 0.0

    if torch.backends.mps.is_available():
        try:
            # MPS memory stats.
            #
            # IMPORTANT: These calls can touch MPS/Metal driver state. During the soak test
            # we must include them in the global Semaphore(2) throttle to avoid exceeding
            # the known-safe limit of 2 concurrent MPS operations (AGX race at 3+).
            with _mps_throttle:
                gpu_allocated_mb = torch.mps.current_allocated_memory() / (1024 * 1024)
                gpu_reserved_mb = torch.mps.driver_allocated_memory() / (1024 * 1024)
        except Exception:
            pass

    return cpu_mb, gpu_allocated_mb, gpu_reserved_mb


def run_soak_test(
    duration_seconds: int = 3600,
    num_threads: int = 4,
    checkpoint_interval: int = 60,
    memory_growth_limit_mb: float = 1000.0,
    verbose: bool = True
) -> SoakResults:
    """
    Run the soak test.

    Args:
        duration_seconds: How long to run (default 3600 = 1 hour)
        num_threads: Number of parallel inference threads
        checkpoint_interval: Seconds between metric checkpoints
        memory_growth_limit_mb: Fail if memory grows more than this
        verbose: Print progress to stdout

    Returns:
        SoakResults with all metrics and pass/fail status
    """
    # Collect hardware info - use simplified version to avoid subprocess issues
    hardware_info = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
    }

    if not torch.backends.mps.is_available():
        return SoakResults(
            start_time=datetime.now(timezone.utc).isoformat(),
            end_time=datetime.now(timezone.utc).isoformat(),
            duration_seconds=0,
            total_iterations=0,
            total_successful_ops=0,
            total_failed_ops=0,
            initial_cpu_memory_mb=0,
            final_cpu_memory_mb=0,
            peak_cpu_memory_mb=0,
            memory_growth_mb=0,
            initial_gpu_memory_mb=0,
            final_gpu_memory_mb=0,
            peak_gpu_memory_mb=0,
            gpu_memory_growth_mb=0,
            average_ops_per_second=0,
            hardware_info=hardware_info,
            passed=False,
            failure_reason="MPS not available"
        )

    # Create model on MPS
    model = SimpleMPSModel(size=256).to('mps')
    model.eval()

    # Warmup the model to ensure MPS is fully initialized
    with torch.no_grad():
        warmup_input = torch.randn(32, 256, device='mps')
        _ = model(warmup_input)
        torch.mps.synchronize()

    # Initial memory snapshot
    gc.collect()
    torch.mps.empty_cache()
    initial_cpu_mb, initial_gpu_allocated, initial_gpu_reserved = get_memory_stats()

    # Track metrics
    metrics_history: List[Dict] = []
    peak_cpu_mb = initial_cpu_mb
    peak_gpu_mb = initial_gpu_allocated

    # Create workers
    workers = [SoakTestWorker(i, model) for i in range(num_threads)]

    # Start workers
    for worker in workers:
        worker.start()

    start_time = time.time()
    start_timestamp = datetime.now(timezone.utc).isoformat()
    iteration = 0
    last_checkpoint = start_time
    last_total_ops = 0
    failure_reason = None

    if verbose:
        print(f"Starting {duration_seconds}s soak test with {num_threads} threads...")
        print(f"Hardware: {hardware_info.get('chip', 'unknown')}")
        print(f"Initial CPU memory: {initial_cpu_mb:.1f} MB")
        print(f"Initial GPU memory: {initial_gpu_allocated:.1f} MB allocated")
        print("-" * 60)

    try:
        while time.time() - start_time < duration_seconds:
            current_time = time.time()

            # Checkpoint every interval
            if current_time - last_checkpoint >= checkpoint_interval:
                iteration += 1
                elapsed = current_time - start_time

                # Collect metrics
                cpu_mb, gpu_allocated, gpu_reserved = get_memory_stats()
                peak_cpu_mb = max(peak_cpu_mb, cpu_mb)
                peak_gpu_mb = max(peak_gpu_mb, gpu_allocated)

                # Aggregate worker stats
                total_success = 0
                total_fail = 0
                all_errors = []
                alive_count = 0

                for worker in workers:
                    success, fail, errors = worker.get_stats()
                    total_success += success
                    total_fail += fail
                    all_errors.extend(errors)
                    if worker.thread and worker.thread.is_alive():
                        alive_count += 1

                ops_per_second = (total_success - last_total_ops) / checkpoint_interval
                last_total_ops = total_success

                # Create checkpoint metrics
                metrics = SoakMetrics(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds=elapsed,
                    iteration=iteration,
                    cpu_memory_mb=cpu_mb,
                    cpu_memory_peak_mb=peak_cpu_mb,
                    gpu_memory_allocated_mb=gpu_allocated,
                    gpu_memory_reserved_mb=gpu_reserved,
                    successful_ops=total_success,
                    failed_ops=total_fail,
                    threads_alive=alive_count,
                    ops_per_second=ops_per_second,
                    errors=all_errors[-5:]  # Last 5 errors
                )
                metrics_history.append(vars(metrics))

                if verbose:
                    memory_growth = cpu_mb - initial_cpu_mb
                    gpu_growth = gpu_allocated - initial_gpu_allocated
                    print(f"[{elapsed:.0f}s] Iter {iteration}: "
                          f"Ops={total_success:,} ({ops_per_second:.0f}/s) | "
                          f"CPU: {cpu_mb:.1f}MB (+{memory_growth:.1f}) | "
                          f"GPU: {gpu_allocated:.1f}MB (+{gpu_growth:.1f}) | "
                          f"Threads: {alive_count}/{num_threads} | "
                          f"Fails: {total_fail}")

                # Check for memory leak
                memory_growth = cpu_mb - initial_cpu_mb
                if memory_growth > memory_growth_limit_mb:
                    failure_reason = f"Memory leak detected: {memory_growth:.1f}MB growth exceeds {memory_growth_limit_mb}MB limit"
                    break

                # Check for thread death
                if alive_count < num_threads:
                    failure_reason = f"Thread death: only {alive_count}/{num_threads} threads alive"
                    break

                # Check for high failure rate
                if total_fail > 0 and total_fail / max(1, total_success + total_fail) > 0.01:
                    failure_reason = f"High failure rate: {total_fail} failures out of {total_success + total_fail} ops"
                    break

                last_checkpoint = current_time

            # Brief sleep to not spin too hard
            time.sleep(0.1)

    except KeyboardInterrupt:
        if verbose:
            print("\nInterrupted by user")
        failure_reason = "Interrupted by user"
    except Exception as e:
        failure_reason = f"Exception: {type(e).__name__}: {str(e)}"
    finally:
        # Stop workers
        for worker in workers:
            worker.stop()

    # Final metrics
    end_time = time.time()
    end_timestamp = datetime.now(timezone.utc).isoformat()
    final_cpu_mb, final_gpu_allocated, final_gpu_reserved = get_memory_stats()

    # Aggregate final stats
    total_success = sum(w.successful_ops for w in workers)
    total_fail = sum(w.failed_ops for w in workers)

    duration = end_time - start_time
    avg_ops_per_sec = total_success / duration if duration > 0 else 0

    results = SoakResults(
        start_time=start_timestamp,
        end_time=end_timestamp,
        duration_seconds=duration,
        total_iterations=iteration,
        total_successful_ops=total_success,
        total_failed_ops=total_fail,
        initial_cpu_memory_mb=initial_cpu_mb,
        final_cpu_memory_mb=final_cpu_mb,
        peak_cpu_memory_mb=peak_cpu_mb,
        memory_growth_mb=final_cpu_mb - initial_cpu_mb,
        initial_gpu_memory_mb=initial_gpu_allocated,
        final_gpu_memory_mb=final_gpu_allocated,
        peak_gpu_memory_mb=peak_gpu_mb,
        gpu_memory_growth_mb=final_gpu_allocated - initial_gpu_allocated,
        average_ops_per_second=avg_ops_per_sec,
        hardware_info=hardware_info,
        passed=failure_reason is None,
        failure_reason=failure_reason,
        metrics_history=metrics_history
    )

    if verbose:
        print("-" * 60)
        print(f"Soak test {'PASSED' if results.passed else 'FAILED'}")
        print(f"Duration: {duration:.1f}s")
        print(f"Total operations: {total_success:,} successful, {total_fail:,} failed")
        print(f"Average throughput: {avg_ops_per_sec:.1f} ops/sec")
        print(f"CPU memory growth: {results.memory_growth_mb:.1f} MB")
        print(f"GPU memory growth: {results.gpu_memory_growth_mb:.1f} MB")
        if failure_reason:
            print(f"Failure reason: {failure_reason}")

    return results


def save_results(results: SoakResults, output_path: Path):
    """Save results to JSON file."""
    # Convert dataclass to dict
    data = {
        "start_time": results.start_time,
        "end_time": results.end_time,
        "duration_seconds": results.duration_seconds,
        "total_iterations": results.total_iterations,
        "total_successful_ops": results.total_successful_ops,
        "total_failed_ops": results.total_failed_ops,
        "memory": {
            "cpu": {
                "initial_mb": results.initial_cpu_memory_mb,
                "final_mb": results.final_cpu_memory_mb,
                "peak_mb": results.peak_cpu_memory_mb,
                "growth_mb": results.memory_growth_mb,
            },
            "gpu": {
                "initial_mb": results.initial_gpu_memory_mb,
                "final_mb": results.final_gpu_memory_mb,
                "peak_mb": results.peak_gpu_memory_mb,
                "growth_mb": results.gpu_memory_growth_mb,
            }
        },
        "performance": {
            "average_ops_per_second": results.average_ops_per_second,
        },
        "hardware_info": results.hardware_info,
        "passed": results.passed,
        "failure_reason": results.failure_reason,
        "metrics_history": results.metrics_history,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="MPS Parallel Inference Soak Test")
    parser.add_argument("--duration", type=int, default=3600,
                        help="Test duration in seconds (default: 3600 = 1 hour)")
    parser.add_argument("--threads", type=int, default=4,
                        help="Number of parallel inference threads (default: 4)")
    parser.add_argument("--checkpoint", type=int, default=60,
                        help="Seconds between metric checkpoints (default: 60)")
    parser.add_argument("--memory-limit", type=float, default=1000.0,
                        help="Memory growth limit in MB before failing (default: 1000)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path (default: reports/main/soak_test_<timestamp>.json)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output")

    args = parser.parse_args()

    # Run the test
    results = run_soak_test(
        duration_seconds=args.duration,
        num_threads=args.threads,
        checkpoint_interval=args.checkpoint,
        memory_growth_limit_mb=args.memory_limit,
        verbose=not args.quiet
    )

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"reports/main/soak_test_{timestamp}.json")

    # Save results
    save_results(results, output_path)
    print(f"\nResults saved to: {output_path}")

    # Exit with appropriate code
    sys.exit(0 if results.passed else 1)


if __name__ == "__main__":
    main()
