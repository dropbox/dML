# Copyright 2024-2025 Andrew Yates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Performance Benchmark

Benchmarks PyTorch and MLX model performance including:
- Latency (single inference time)
- Throughput (inferences per second)
- Memory usage
- Parallel inference scaling
"""

import statistics
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class LatencyStats:
    """Latency statistics from benchmark."""

    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    samples: int


@dataclass
class ThroughputStats:
    """Throughput statistics from benchmark."""

    samples_per_second: float
    total_samples: int
    total_time_seconds: float
    batch_size: int


@dataclass
class MemoryStats:
    """Memory usage statistics."""

    peak_mb: float
    allocated_mb: float
    reserved_mb: float


@dataclass
class ParallelStats:
    """Statistics for parallel inference."""

    num_threads: int
    total_samples: int
    total_time_seconds: float
    throughput: float  # samples/second
    per_thread_throughput: float
    scaling_efficiency: float  # vs single thread


@dataclass
class BenchmarkResult:
    """Complete benchmark result."""

    model_name: str
    framework: str  # 'pytorch' or 'mlx'
    latency: LatencyStats
    throughput: ThroughputStats
    memory: MemoryStats | None = None
    parallel: ParallelStats | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Benchmark:
    """
    Benchmarks model performance for PyTorch and MLX.

    Measures:
    - Single inference latency
    - Batch inference throughput
    - Memory usage
    - Parallel inference scaling
    """

    def __init__(
        self,
        warmup_iterations: int = 10,
        benchmark_iterations: int = 100,
        batch_size: int = 1,
    ):
        """
        Initialize benchmark.

        Args:
            warmup_iterations: Number of warmup runs before measuring
            benchmark_iterations: Number of benchmark iterations
            batch_size: Batch size for throughput testing
        """
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        self.batch_size = batch_size

    def benchmark_pytorch(
        self,
        model: Any,
        inputs: dict[str, Any],
        model_name: str = "pytorch_model",
    ) -> BenchmarkResult:
        """
        Benchmark PyTorch model.

        Args:
            model: PyTorch model (should be in eval mode)
            inputs: Input dictionary
            model_name: Name for the model

        Returns:
            BenchmarkResult with performance metrics
        """
        import torch

        model.eval()

        # Convert inputs to tensors
        pt_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, np.ndarray):
                pt_inputs[k] = torch.from_numpy(v)
            else:
                pt_inputs[k] = v

        # Move to available device
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = model.to(device)
        pt_inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in pt_inputs.items()
        }

        def run_inference() -> Any:
            with torch.no_grad():
                if len(pt_inputs) == 1:
                    output = model(list(pt_inputs.values())[0])
                else:
                    output = model(**pt_inputs)
                if device == "mps":
                    torch.mps.synchronize()
            return output

        # Warmup
        for _ in range(self.warmup_iterations):
            run_inference()

        # Benchmark latency
        latency_times = []
        for _ in range(self.benchmark_iterations):
            start = time.perf_counter()
            run_inference()
            end = time.perf_counter()
            latency_times.append((end - start) * 1000)  # ms

        latency = self._compute_latency_stats(latency_times)

        # Benchmark throughput
        throughput = self._benchmark_throughput(run_inference, self.batch_size)

        # Get memory stats if available
        memory = None
        if device == "mps" and hasattr(torch.mps, "driver_allocated_memory"):
            try:
                allocated = torch.mps.driver_allocated_memory() / (1024 * 1024)
                memory = MemoryStats(
                    peak_mb=allocated, allocated_mb=allocated, reserved_mb=allocated,
                )
            except Exception:
                pass

        return BenchmarkResult(
            model_name=model_name,
            framework="pytorch",
            latency=latency,
            throughput=throughput,
            memory=memory,
            metadata={"device": device},
        )

    def benchmark_mlx(
        self,
        model: Any,
        inputs: dict[str, Any],
        model_name: str = "mlx_model",
    ) -> BenchmarkResult:
        """
        Benchmark MLX model.

        Args:
            model: MLX model
            inputs: Input dictionary
            model_name: Name for the model

        Returns:
            BenchmarkResult with performance metrics
        """
        import mlx.core as mx

        # Convert inputs to MLX arrays
        mlx_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, np.ndarray):
                mlx_inputs[k] = mx.array(v)
            else:
                mlx_inputs[k] = v

        def run_inference() -> Any:
            if len(mlx_inputs) == 1:
                output = model(list(mlx_inputs.values())[0])
            else:
                output = model(**mlx_inputs)
            mx.eval(output)  # Force evaluation of lazy computation
            return output

        # Warmup
        for _ in range(self.warmup_iterations):
            run_inference()

        # Benchmark latency
        latency_times = []
        for _ in range(self.benchmark_iterations):
            start = time.perf_counter()
            run_inference()
            end = time.perf_counter()
            latency_times.append((end - start) * 1000)  # ms

        latency = self._compute_latency_stats(latency_times)

        # Benchmark throughput
        throughput = self._benchmark_throughput(run_inference, self.batch_size)

        # MLX memory stats (if available)
        memory = None
        if hasattr(mx.metal, "get_peak_memory"):
            try:
                peak = mx.metal.get_peak_memory() / (1024 * 1024)
                memory = MemoryStats(peak_mb=peak, allocated_mb=peak, reserved_mb=peak)
            except Exception:
                pass

        return BenchmarkResult(
            model_name=model_name,
            framework="mlx",
            latency=latency,
            throughput=throughput,
            memory=memory,
            metadata={"device": "metal"},
        )

    def benchmark_parallel_mlx(
        self,
        model: Any,
        inputs: dict[str, Any],
        num_threads: list[int] | None = None,
        iterations_per_thread: int = 50,
    ) -> list[ParallelStats]:
        """
        Benchmark MLX parallel inference scaling.

        Args:
            model: MLX model
            inputs: Input dictionary
            num_threads: List of thread counts to test
            iterations_per_thread: Iterations each thread runs

        Returns:
            List of ParallelStats for each thread count
        """
        import mlx.core as mx

        if num_threads is None:
            num_threads = [1, 2, 4, 8]

        # Convert inputs
        mlx_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, np.ndarray):
                mlx_inputs[k] = mx.array(v)
            else:
                mlx_inputs[k] = v

        # First get single-thread baseline
        baseline_throughput = None
        results = []

        for n_threads in num_threads:
            stats = self._run_parallel_benchmark(
                model, mlx_inputs, n_threads, iterations_per_thread,
            )

            if n_threads == 1:
                baseline_throughput = stats.throughput

            # Calculate scaling efficiency
            if baseline_throughput:
                expected = baseline_throughput * n_threads
                stats.scaling_efficiency = stats.throughput / expected
            else:
                stats.scaling_efficiency = 1.0

            results.append(stats)

        return results

    def _run_parallel_benchmark(
        self,
        model: Any,
        inputs: dict[str, Any],
        num_threads: int,
        iterations: int,
    ) -> ParallelStats:
        """Run parallel inference benchmark."""
        import mlx.core as mx

        results = []
        lock = threading.Lock()

        def worker() -> None:
            thread_times: list[float] = []
            for _ in range(iterations):
                start = time.perf_counter()
                if len(inputs) == 1:
                    output = model(list(inputs.values())[0])
                else:
                    output = model(**inputs)
                mx.eval(output)
                end = time.perf_counter()
                thread_times.append(end - start)

            with lock:
                results.append(thread_times)

        # Warmup
        for _ in range(5):
            if len(inputs) == 1:
                output = model(list(inputs.values())[0])
            else:
                output = model(**inputs)
            mx.eval(output)

        # Run parallel benchmark
        threads = [threading.Thread(target=worker) for _ in range(num_threads)]

        start_time = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        total_time = time.perf_counter() - start_time

        total_samples = num_threads * iterations
        throughput = total_samples / total_time

        return ParallelStats(
            num_threads=num_threads,
            total_samples=total_samples,
            total_time_seconds=total_time,
            throughput=throughput,
            per_thread_throughput=throughput / num_threads,
            scaling_efficiency=0.0,  # Filled in by caller
        )

    def _compute_latency_stats(self, times_ms: list[float]) -> LatencyStats:
        """Compute latency statistics from timing samples."""
        sorted_times = sorted(times_ms)
        n = len(sorted_times)

        return LatencyStats(
            mean_ms=statistics.mean(times_ms),
            std_ms=statistics.stdev(times_ms) if n > 1 else 0.0,
            min_ms=sorted_times[0],
            max_ms=sorted_times[-1],
            p50_ms=sorted_times[n // 2],
            p95_ms=sorted_times[int(n * 0.95)],
            p99_ms=sorted_times[int(n * 0.99)],
            samples=n,
        )

    def _benchmark_throughput(
        self, run_fn: Callable[[], Any], batch_size: int, duration_seconds: float = 5.0,
    ) -> ThroughputStats:
        """Benchmark throughput over a fixed duration."""
        total_samples = 0
        start_time = time.perf_counter()
        end_time = start_time + duration_seconds

        while time.perf_counter() < end_time:
            run_fn()
            total_samples += batch_size

        elapsed = time.perf_counter() - start_time

        return ThroughputStats(
            samples_per_second=total_samples / elapsed,
            total_samples=total_samples,
            total_time_seconds=elapsed,
            batch_size=batch_size,
        )

    def compare(
        self,
        pytorch_result: BenchmarkResult,
        mlx_result: BenchmarkResult,
    ) -> dict[str, Any]:
        """
        Compare PyTorch and MLX benchmark results.

        Args:
            pytorch_result: PyTorch benchmark result
            mlx_result: MLX benchmark result

        Returns:
            Dictionary with comparison metrics
        """
        return {
            "latency_speedup": pytorch_result.latency.mean_ms
            / mlx_result.latency.mean_ms,
            "throughput_speedup": mlx_result.throughput.samples_per_second
            / pytorch_result.throughput.samples_per_second,
            "pytorch_latency_ms": pytorch_result.latency.mean_ms,
            "mlx_latency_ms": mlx_result.latency.mean_ms,
            "pytorch_throughput": pytorch_result.throughput.samples_per_second,
            "mlx_throughput": mlx_result.throughput.samples_per_second,
            "pytorch_device": pytorch_result.metadata.get("device", "unknown"),
            "mlx_device": mlx_result.metadata.get("device", "unknown"),
        }

    def generate_report(
        self,
        results: list[BenchmarkResult],
        comparison: dict[str, Any] | None = None,
        parallel_results: list[ParallelStats] | None = None,
    ) -> str:
        """
        Generate human-readable benchmark report.

        Args:
            results: List of benchmark results
            comparison: Optional comparison dictionary
            parallel_results: Optional parallel scaling results

        Returns:
            Formatted report string
        """
        lines = [
            "=" * 70,
            "BENCHMARK REPORT",
            "=" * 70,
            "",
        ]

        for result in results:
            lines.extend(
                [
                    f"Model: {result.model_name} ({result.framework.upper()})",
                    "-" * 50,
                    "",
                    "Latency:",
                    f"  Mean:   {result.latency.mean_ms:.2f} ms",
                    f"  Std:    {result.latency.std_ms:.2f} ms",
                    f"  Min:    {result.latency.min_ms:.2f} ms",
                    f"  Max:    {result.latency.max_ms:.2f} ms",
                    f"  P50:    {result.latency.p50_ms:.2f} ms",
                    f"  P95:    {result.latency.p95_ms:.2f} ms",
                    f"  P99:    {result.latency.p99_ms:.2f} ms",
                    "",
                    "Throughput:",
                    f"  {result.throughput.samples_per_second:.1f} samples/second",
                    f"  Batch size: {result.throughput.batch_size}",
                    "",
                ],
            )

            if result.memory:
                lines.extend(
                    [
                        "Memory:",
                        f"  Peak:      {result.memory.peak_mb:.1f} MB",
                        f"  Allocated: {result.memory.allocated_mb:.1f} MB",
                        "",
                    ],
                )

            lines.append("")

        if comparison:
            lines.extend(
                [
                    "COMPARISON (MLX vs PyTorch)",
                    "-" * 50,
                    f"  Latency speedup:    {comparison['latency_speedup']:.2f}x",
                    f"  Throughput speedup: {comparison['throughput_speedup']:.2f}x",
                    "",
                ],
            )

        if parallel_results:
            lines.extend(
                [
                    "PARALLEL SCALING (MLX)",
                    "-" * 50,
                ],
            )
            lines.extend(
                f"  {p.num_threads} threads: {p.throughput:.1f} samples/sec "
                f"(efficiency: {p.scaling_efficiency * 100:.1f}%)"
                for p in parallel_results
            )
            lines.append("")

        lines.append("=" * 70)

        return "\n".join(lines)

    @staticmethod
    def create_test_input(
        shape: tuple[int, ...], dtype: str = "float32", seed: int = 42,
    ) -> dict[str, np.ndarray]:
        """
        Create a random test input.

        Args:
            shape: Input shape
            dtype: Data type
            seed: Random seed

        Returns:
            Input dictionary with 'input' key
        """
        rng = np.random.default_rng(seed)

        dtype_map = {
            "float32": np.float32,
            "float16": np.float16,
            "int32": np.int32,
            "int64": np.int64,
        }
        np_dtype = dtype_map.get(dtype, np.float32)

        if np.issubdtype(np_dtype, np.floating):
            arr = rng.standard_normal(shape).astype(np_dtype)
        else:
            arr = rng.integers(0, 100, size=shape, dtype=np_dtype)

        return {"input": arr}
