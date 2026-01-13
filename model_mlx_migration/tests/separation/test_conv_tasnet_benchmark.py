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
Benchmark tests for Conv-TasNet MLX implementation.

Validates:
1. Latency: <25ms for 100ms audio chunks
2. Throughput: Real-time factor (RTF)
"""

import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.whisper_mlx.separation.conv_tasnet import ConvTasNet, ConvTasNetConfig


def load_model():
    """Load Conv-TasNet model with pretrained weights."""
    weights_path = Path(__file__).parent.parent.parent / "models" / "conv_tasnet" / "conv_tasnet_16k.safetensors"
    config_path = weights_path.with_suffix(".json")

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found at {weights_path}")

    # Load config
    import json
    with open(config_path) as f:
        config_dict = json.load(f)

    # Create config from Asteroid model_args
    model_args = config_dict.get("model_args", config_dict)
    config = ConvTasNetConfig.from_asteroid(model_args)

    # Create model
    model = ConvTasNet(config)

    # Load weights
    from mlx.utils import tree_unflatten
    weights = mx.load(str(weights_path))
    model.update(tree_unflatten(list(weights.items())))

    return model


def benchmark_latency(model, chunk_duration_ms=100, sample_rate=16000, n_warmup=5, n_runs=50):
    """
    Benchmark inference latency for a given chunk duration.

    Args:
        model: Conv-TasNet model
        chunk_duration_ms: Chunk duration in milliseconds
        sample_rate: Audio sample rate
        n_warmup: Number of warmup runs
        n_runs: Number of timed runs

    Returns:
        Dictionary with latency statistics
    """
    # Calculate chunk size
    chunk_samples = int(chunk_duration_ms * sample_rate / 1000)

    # Generate random test audio
    rng = np.random.default_rng(42)
    test_audio = mx.array(rng.standard_normal((1, chunk_samples)).astype(np.float32))

    # Warmup
    print(f"Warming up ({n_warmup} runs)...")
    for _ in range(n_warmup):
        output = model(test_audio)
        mx.eval(output)

    # Timed runs
    print(f"Running benchmark ({n_runs} runs)...")
    latencies = []
    for _i in range(n_runs):
        # Generate fresh random audio each run to avoid caching
        test_audio = mx.array(rng.standard_normal((1, chunk_samples)).astype(np.float32))

        start = time.perf_counter()
        output = model(test_audio)
        mx.eval(output)
        end = time.perf_counter()

        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)

    latencies = np.array(latencies)

    return {
        "chunk_duration_ms": chunk_duration_ms,
        "chunk_samples": chunk_samples,
        "n_runs": n_runs,
        "mean_latency_ms": float(np.mean(latencies)),
        "std_latency_ms": float(np.std(latencies)),
        "min_latency_ms": float(np.min(latencies)),
        "max_latency_ms": float(np.max(latencies)),
        "p50_latency_ms": float(np.percentile(latencies, 50)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "p99_latency_ms": float(np.percentile(latencies, 99)),
        "rtf": float(np.mean(latencies) / chunk_duration_ms),  # Real-time factor
    }


def benchmark_throughput(model, total_duration_s=10, sample_rate=16000):
    """
    Benchmark throughput for processing longer audio.

    Args:
        model: Conv-TasNet model
        total_duration_s: Total audio duration to process
        sample_rate: Audio sample rate

    Returns:
        Dictionary with throughput statistics
    """
    total_samples = int(total_duration_s * sample_rate)

    # Generate random test audio
    rng = np.random.default_rng(42)
    test_audio = mx.array(rng.standard_normal((1, total_samples)).astype(np.float32))

    # Warmup
    warmup_audio = mx.array(rng.standard_normal((1, sample_rate)).astype(np.float32))
    for _ in range(3):
        output = model(warmup_audio)
        mx.eval(output)

    # Time the full processing
    start = time.perf_counter()
    output = model(test_audio)
    mx.eval(output)
    end = time.perf_counter()

    processing_time = end - start

    return {
        "audio_duration_s": total_duration_s,
        "total_samples": total_samples,
        "processing_time_s": processing_time,
        "samples_per_second": total_samples / processing_time,
        "rtf": processing_time / total_duration_s,
    }


def test_latency_target():
    """Test that latency meets the <25ms target for 100ms chunks."""
    print("\n" + "=" * 60)
    print("Conv-TasNet Latency Benchmark")
    print("=" * 60)

    print("\nLoading model...")
    model = load_model()
    print("Model loaded.")

    # Benchmark 100ms chunks (target: <25ms)
    results = benchmark_latency(model, chunk_duration_ms=100)

    print("\n100ms Chunk Results:")
    print(f"  Mean latency: {results['mean_latency_ms']:.2f} ms")
    print(f"  Std latency:  {results['std_latency_ms']:.2f} ms")
    print(f"  Min latency:  {results['min_latency_ms']:.2f} ms")
    print(f"  Max latency:  {results['max_latency_ms']:.2f} ms")
    print(f"  P50 latency:  {results['p50_latency_ms']:.2f} ms")
    print(f"  P95 latency:  {results['p95_latency_ms']:.2f} ms")
    print(f"  P99 latency:  {results['p99_latency_ms']:.2f} ms")
    print(f"  RTF:          {results['rtf']:.3f}x")

    # Check target
    target_ms = 25.0
    if results['p95_latency_ms'] < target_ms:
        print(f"\n✓ PASS: P95 latency ({results['p95_latency_ms']:.2f} ms) < {target_ms} ms target")
    else:
        print(f"\n✗ FAIL: P95 latency ({results['p95_latency_ms']:.2f} ms) >= {target_ms} ms target")

    return results


def test_various_chunk_sizes():
    """Benchmark latency across various chunk sizes."""
    print("\n" + "=" * 60)
    print("Conv-TasNet Latency vs Chunk Size")
    print("=" * 60)

    print("\nLoading model...")
    model = load_model()
    print("Model loaded.")

    chunk_sizes_ms = [50, 100, 200, 500, 1000]

    print(f"\n{'Chunk (ms)':<12} {'Mean (ms)':<12} {'P95 (ms)':<12} {'RTF':<10}")
    print("-" * 46)

    for chunk_ms in chunk_sizes_ms:
        results = benchmark_latency(model, chunk_duration_ms=chunk_ms, n_warmup=3, n_runs=20)
        print(f"{chunk_ms:<12} {results['mean_latency_ms']:<12.2f} {results['p95_latency_ms']:<12.2f} {results['rtf']:<10.3f}")


def test_throughput():
    """Benchmark throughput for longer audio."""
    print("\n" + "=" * 60)
    print("Conv-TasNet Throughput Benchmark")
    print("=" * 60)

    print("\nLoading model...")
    model = load_model()
    print("Model loaded.")

    results = benchmark_throughput(model, total_duration_s=10)

    print("\n10s Audio Throughput:")
    print(f"  Processing time: {results['processing_time_s']:.3f} s")
    print(f"  Samples/second:  {results['samples_per_second']:,.0f}")
    print(f"  RTF:             {results['rtf']:.3f}x")

    if results['rtf'] < 1.0:
        print(f"\n✓ Faster than real-time ({1/results['rtf']:.1f}x speed)")
    else:
        print("\n✗ Slower than real-time")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Conv-TasNet benchmark")
    parser.add_argument("--latency", action="store_true", help="Run latency benchmark")
    parser.add_argument("--chunk-sizes", action="store_true", help="Run chunk size comparison")
    parser.add_argument("--throughput", action="store_true", help="Run throughput benchmark")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")

    args = parser.parse_args()

    if args.all or (not args.latency and not args.chunk_sizes and not args.throughput):
        test_latency_target()
        test_various_chunk_sizes()
        test_throughput()
    else:
        if args.latency:
            test_latency_target()
        if args.chunk_sizes:
            test_various_chunk_sizes()
        if args.throughput:
            test_throughput()
