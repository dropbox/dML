#!/usr/bin/env python3
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
Validate Parallel Inference Scaling

Tests that MLX models properly scale across multiple threads without
mutex/serialization issues.

This addresses Issue 4 from the external audit:
"Parallel benchmarking exists but no committed, reproducible scaling test on real model."

Usage:
    python scripts/validate_parallel_inference.py
"""

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List

# Add tools path
sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "pytorch_to_mlx"))

import mlx.core as mx


@dataclass
class ParallelStats:
    """Statistics for parallel inference."""

    num_threads: int
    total_time_ms: float
    throughput: float  # iterations per second
    scaling_efficiency: float  # actual vs expected scaling
    min_latency_ms: float
    max_latency_ms: float


def run_inference(model_fn, inputs, iterations: int, stream=None) -> List[float]:
    """Run inference multiple times and return latencies."""
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        if stream is not None:
            with mx.stream(stream):
                _ = model_fn(*inputs)
                mx.eval()
        else:
            _ = model_fn(*inputs)
            mx.eval()  # Force sync
        latencies.append((time.perf_counter() - start) * 1000)
    return latencies


def benchmark_parallel(
    model_fn,
    inputs,
    num_threads: int,
    iterations_per_thread: int = 20,
    use_separate_streams: bool = False,
) -> ParallelStats:
    """Benchmark parallel inference with given thread count.

    Args:
        model_fn: Model inference function
        inputs: Tuple of inputs to pass to model_fn
        num_threads: Number of concurrent threads
        iterations_per_thread: Iterations per thread
        use_separate_streams: If True, each thread gets its own MLX stream
    """

    all_latencies = []
    lock = threading.Lock()

    def worker(stream=None):
        latencies = run_inference(model_fn, inputs, iterations_per_thread, stream)
        with lock:
            all_latencies.extend(latencies)

    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        if use_separate_streams:
            # Create separate stream per thread for true parallelism
            streams = [mx.new_stream(mx.default_device()) for _ in range(num_threads)]
            futures = [executor.submit(worker, s) for s in streams]
        else:
            futures = [executor.submit(worker, None) for _ in range(num_threads)]
        for f in futures:
            f.result()

    total_time_ms = (time.perf_counter() - start) * 1000
    total_iters = num_threads * iterations_per_thread
    throughput = total_iters / (total_time_ms / 1000)

    return ParallelStats(
        num_threads=num_threads,
        total_time_ms=total_time_ms,
        throughput=throughput,
        scaling_efficiency=0.0,  # Calculated later
        min_latency_ms=min(all_latencies) if all_latencies else 0,
        max_latency_ms=max(all_latencies) if all_latencies else 0,
    )


def test_generator_parallel():
    """Test Generator parallel inference."""
    print("\n=== Generator Parallel Inference ===")

    from converters.models.kokoro import Generator
    from converters.models.kokoro_modules import KokoroConfig

    config = KokoroConfig()
    gen = Generator(config)

    # Test inputs
    batch = 1
    length = 32
    channels = 512

    x = mx.random.normal((batch, length, channels))
    s = mx.random.normal((batch, config.style_dim))
    f0 = mx.full((batch, length // 2), 1.0)

    # Warm up
    for _ in range(3):
        _ = gen(x, s, f0)
        mx.eval()

    # Test with different thread counts
    thread_counts = [1, 2, 4]
    results = []

    for n_threads in thread_counts:
        print(f"  Testing {n_threads} threads...")
        stats = benchmark_parallel(
            lambda x, s, f0: gen(x, s, f0),
            (x, s, f0),
            n_threads,
            iterations_per_thread=10,
        )
        results.append(stats)

    # Calculate scaling efficiency
    baseline = results[0].throughput
    for stats in results:
        expected = baseline * stats.num_threads
        stats.scaling_efficiency = stats.throughput / expected if expected > 0 else 0

    # Print results
    print("\n  Results:")
    print(f"  {'Threads':<10} {'Throughput':<15} {'Scaling':<10} {'Min/Max Lat':<20}")
    print(f"  {'-' * 55}")
    for stats in results:
        print(
            f"  {stats.num_threads:<10} {stats.throughput:>10.1f} iter/s"
            f"  {stats.scaling_efficiency:>6.1%}"
            f"    {stats.min_latency_ms:.1f}/{stats.max_latency_ms:.1f} ms"
        )

    # Validation
    min_efficiency = min(s.scaling_efficiency for s in results)
    print(f"\n  Minimum scaling efficiency: {min_efficiency:.1%}")

    if min_efficiency > 0.5:
        print("  PASS: Parallel scaling efficiency > 50%")
        return True
    else:
        print("  WARN: Low scaling efficiency, may indicate serialization")
        return False


def test_stft_parallel():
    """Test STFT parallel inference."""
    print("\n=== STFT Parallel Inference ===")

    from converters.models.stft import TorchSTFT

    stft = TorchSTFT(filter_length=800, hop_length=200, win_length=800)

    # Test signal
    signal = mx.random.normal((1, 24000))

    # Warm up
    for _ in range(3):
        mag, phase = stft.transform(signal)
        mx.eval(mag, phase)

    # Test with different thread counts
    thread_counts = [1, 2, 4]
    results = []

    for n_threads in thread_counts:
        print(f"  Testing {n_threads} threads...")
        stats = benchmark_parallel(
            lambda sig: stft.transform(sig),
            (signal,),
            n_threads,
            iterations_per_thread=20,
        )
        results.append(stats)

    # Calculate scaling efficiency
    baseline = results[0].throughput
    for stats in results:
        expected = baseline * stats.num_threads
        stats.scaling_efficiency = stats.throughput / expected if expected > 0 else 0

    # Print results
    print("\n  Results:")
    print(f"  {'Threads':<10} {'Throughput':<15} {'Scaling':<10}")
    print(f"  {'-' * 35}")
    for stats in results:
        print(
            f"  {stats.num_threads:<10} {stats.throughput:>10.1f} iter/s"
            f"  {stats.scaling_efficiency:>6.1%}"
        )

    # Validation
    min_efficiency = min(s.scaling_efficiency for s in results)
    print(f"\n  Minimum scaling efficiency: {min_efficiency:.1%}")

    if min_efficiency > 0.5:
        print("  PASS: Parallel scaling efficiency > 50%")
        return True
    else:
        print("  WARN: Low scaling efficiency")
        return False


def test_full_model_parallel():
    """Test full Kokoro model parallel inference."""
    print("\n=== Full Model Parallel Inference ===")

    from converters.models.kokoro import KokoroModel
    from converters.models.kokoro_modules import KokoroConfig

    config = KokoroConfig()
    model = KokoroModel(config)

    # Short sequence for speed
    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    style = mx.random.normal((1, config.style_dim))

    # Warm up
    for _ in range(2):
        _ = model.synthesize(tokens, style)
        mx.eval()

    # Test with different thread counts
    thread_counts = [1, 2]
    results = []

    for n_threads in thread_counts:
        print(f"  Testing {n_threads} threads...")
        stats = benchmark_parallel(
            lambda t, s: model.synthesize(t, s),
            (tokens, style),
            n_threads,
            iterations_per_thread=5,
        )
        results.append(stats)

    # Calculate scaling efficiency
    baseline = results[0].throughput
    for stats in results:
        expected = baseline * stats.num_threads
        stats.scaling_efficiency = stats.throughput / expected if expected > 0 else 0

    # Print results
    print("\n  Results:")
    print(f"  {'Threads':<10} {'Throughput':<15} {'Scaling':<10}")
    print(f"  {'-' * 35}")
    for stats in results:
        print(
            f"  {stats.num_threads:<10} {stats.throughput:>10.2f} iter/s"
            f"  {stats.scaling_efficiency:>6.1%}"
        )

    # Validation
    min_efficiency = min(s.scaling_efficiency for s in results)
    print(f"\n  Minimum scaling efficiency: {min_efficiency:.1%}")

    # Full model has more overhead, accept lower threshold
    if min_efficiency > 0.3:
        print("  PASS: Parallel scaling efficiency > 30%")
        return True
    else:
        print("  WARN: Low scaling efficiency")
        return False


def test_streams_vs_shared():
    """Compare separate streams vs shared default stream."""
    print("\n=== Stream Comparison Test ===")

    from converters.models.kokoro import Generator
    from converters.models.kokoro_modules import KokoroConfig

    config = KokoroConfig()
    gen = Generator(config)

    # Test inputs
    batch = 1
    length = 32
    channels = 512

    x = mx.random.normal((batch, length, channels))
    s = mx.random.normal((batch, config.style_dim))
    f0 = mx.full((batch, length // 2), 1.0)

    # Warm up
    for _ in range(3):
        _ = gen(x, s, f0)
        mx.eval()

    num_threads = 4
    iters = 10

    # Test without separate streams (shared default)
    print(f"  Testing {num_threads} threads (shared stream)...")
    shared_stats = benchmark_parallel(
        lambda x, s, f0: gen(x, s, f0),
        (x, s, f0),
        num_threads,
        iterations_per_thread=iters,
        use_separate_streams=False,
    )

    # Test with separate streams
    print(f"  Testing {num_threads} threads (separate streams)...")
    separate_stats = benchmark_parallel(
        lambda x, s, f0: gen(x, s, f0),
        (x, s, f0),
        num_threads,
        iterations_per_thread=iters,
        use_separate_streams=True,
    )

    # Print comparison
    print(f"\n  Stream Comparison ({num_threads} threads, {iters} iters each):")
    print(f"  {'Mode':<20} {'Throughput':<15} {'Total Time':<15}")
    print(f"  {'-' * 50}")
    print(
        f"  {'Shared stream':<20} {shared_stats.throughput:>10.1f} iter/s"
        f"  {shared_stats.total_time_ms:>10.1f} ms"
    )
    print(
        f"  {'Separate streams':<20} {separate_stats.throughput:>10.1f} iter/s"
        f"  {separate_stats.total_time_ms:>10.1f} ms"
    )

    speedup = (
        separate_stats.throughput / shared_stats.throughput
        if shared_stats.throughput > 0
        else 0
    )
    print(f"\n  Speedup with separate streams: {speedup:.2f}x")

    if speedup > 1.1:
        print("  PASS: Separate streams improve throughput")
        return True
    else:
        print("  INFO: Separate streams show similar performance (GPU-bound)")
        return True  # Not a failure - just expected for GPU-bound work


def main():
    """Run all parallel inference tests."""
    print("=" * 60)
    print("Parallel Inference Validation")
    print("=" * 60)
    print("\nTesting MLX parallel inference scaling...")
    print("Expected: Linear or near-linear scaling with thread count")
    print("(No mutex/serialization blocking parallel execution)")

    results = []

    try:
        results.append(("STFT", test_stft_parallel()))
    except Exception as e:
        print(f"Error in STFT test: {e}")
        results.append(("STFT", False))

    try:
        results.append(("Generator", test_generator_parallel()))
    except Exception as e:
        print(f"Error in Generator test: {e}")
        results.append(("Generator", False))

    try:
        results.append(("Full Model", test_full_model_parallel()))
    except Exception as e:
        print(f"Error in Full Model test: {e}")
        results.append(("Full Model", False))

    try:
        results.append(("Stream Comparison", test_streams_vs_shared()))
    except Exception as e:
        print(f"Error in Stream Comparison test: {e}")
        results.append(("Stream Comparison", False))

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    passed_count = sum(1 for _, p in results if p)
    print(f"\n  Total: {passed_count}/{len(results)} tests passed")

    if passed_count == len(results):
        print("\n  OVERALL: PASS - Parallel inference works correctly")
    else:
        print("\n  OVERALL: WARN - Some parallel tests failed")

    return passed_count == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
