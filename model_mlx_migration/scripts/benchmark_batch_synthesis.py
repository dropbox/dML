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
Benchmark script for Kokoro batch synthesis throughput.

Compares sequential vs batched synthesis to measure throughput improvement.

Usage:
    python scripts/benchmark_batch_synthesis.py [--batch-sizes 1,2,4,8]
"""

import argparse
import time
from typing import List, Tuple

import mlx.core as mx


def generate_test_inputs(
    num_samples: int,
    min_tokens: int = 5,
    max_tokens: int = 50,
    seed: int = 42,
) -> Tuple[List[mx.array], mx.array]:
    """Generate random test inputs with varying lengths."""
    mx.random.seed(seed)

    # Generate random token sequences
    input_ids_list = []
    for i in range(num_samples):
        # Vary length across samples
        length = min_tokens + (i * (max_tokens - min_tokens)) // max(num_samples - 1, 1)
        # Random tokens (1-177 to avoid token 0 which is typically padding)
        tokens = mx.random.randint(1, 178, shape=(length,), dtype=mx.int32)
        input_ids_list.append(tokens)

    # Generate voice embeddings
    voice = mx.zeros((num_samples, 256))

    return input_ids_list, voice


def benchmark_sequential(
    model,
    input_ids_list: List[mx.array],
    voice: mx.array,
    num_iterations: int = 3,
) -> dict:
    """Benchmark sequential synthesis (one at a time)."""
    # Warmup - use __call__ directly which has validate_output parameter
    for i, ids in enumerate(input_ids_list[:2]):
        ids_2d = ids[None, :]
        v = voice[i : i + 1, :]
        audio = model(ids_2d, v, validate_output=False)
        mx.eval(audio)

    # Benchmark
    times = []
    total_samples = 0
    for _ in range(num_iterations):
        start = time.perf_counter()
        for i, ids in enumerate(input_ids_list):
            ids_2d = ids[None, :]
            v = voice[i : i + 1, :]
            audio = model(ids_2d, v, validate_output=False)
            mx.eval(audio)
            total_samples += audio.shape[1]
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    return {
        "mode": "sequential",
        "num_samples": len(input_ids_list),
        "total_time_ms": avg_time * 1000,
        "time_per_sample_ms": avg_time * 1000 / len(input_ids_list),
        "throughput_samples_per_sec": len(input_ids_list) / avg_time,
    }


def benchmark_batched(
    model,
    input_ids_list: List[mx.array],
    voice: mx.array,
    num_iterations: int = 3,
) -> dict:
    """Benchmark batched synthesis."""
    # Warmup
    audio, lengths = model.synthesize_batch(
        input_ids_list[:2], voice[:2], validate_output=False
    )
    mx.eval(audio, lengths)

    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        audio, lengths = model.synthesize_batch(
            input_ids_list, voice, validate_output=False
        )
        mx.eval(audio, lengths)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    return {
        "mode": "batched",
        "num_samples": len(input_ids_list),
        "total_time_ms": avg_time * 1000,
        "time_per_sample_ms": avg_time * 1000 / len(input_ids_list),
        "throughput_samples_per_sec": len(input_ids_list) / avg_time,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Kokoro batch synthesis throughput"
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,2,4,8",
        help="Comma-separated batch sizes to test (default: 1,2,4,8)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of iterations per benchmark (default: 3)",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=10,
        help="Minimum tokens per sequence (default: 10)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum tokens per sequence (default: 50)",
    )
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    print("=" * 70)
    print("Kokoro Batch Synthesis Benchmark")
    print("=" * 70)
    print(f"Batch sizes: {batch_sizes}")
    print(f"Token range: {args.min_tokens}-{args.max_tokens}")
    print(f"Iterations per test: {args.iterations}")
    print()

    # Load model
    print("Loading model...")
    from tools.pytorch_to_mlx.converters.models import KokoroConfig, KokoroModel

    config = KokoroConfig()
    model = KokoroModel(config)
    model.set_deterministic(True)
    print("Model loaded (using random weights for benchmarking)")
    print()

    # Results table
    print(f"{'Batch':<6} {'Mode':<12} {'Total (ms)':<12} {'Per Sample':<12} {'Throughput':<15} {'Speedup':<10}")
    print("-" * 70)

    for batch_size in batch_sizes:
        # Generate test data
        input_ids_list, voice = generate_test_inputs(
            batch_size, args.min_tokens, args.max_tokens
        )

        # Benchmark sequential
        seq_results = benchmark_sequential(
            model, input_ids_list, voice, args.iterations
        )

        # Benchmark batched
        batch_results = benchmark_batched(
            model, input_ids_list, voice, args.iterations
        )

        # Calculate speedup
        speedup = seq_results["throughput_samples_per_sec"] / batch_results["throughput_samples_per_sec"]
        # Invert: we want batched throughput / sequential throughput
        speedup = batch_results["throughput_samples_per_sec"] / seq_results["throughput_samples_per_sec"]

        # Print results
        print(
            f"{batch_size:<6} {'sequential':<12} "
            f"{seq_results['total_time_ms']:<12.1f} "
            f"{seq_results['time_per_sample_ms']:<12.1f} "
            f"{seq_results['throughput_samples_per_sec']:<15.2f} "
            f"{'1.00x (baseline)':<10}"
        )
        print(
            f"{'':<6} {'batched':<12} "
            f"{batch_results['total_time_ms']:<12.1f} "
            f"{batch_results['time_per_sample_ms']:<12.1f} "
            f"{batch_results['throughput_samples_per_sec']:<15.2f} "
            f"{speedup:.2f}x"
        )
        print()

    print("=" * 70)
    print("Benchmark complete")
    print()
    print("Notes:")
    print("- Speedup > 1.0x means batched is faster")
    print("- Throughput = samples processed per second")
    print("- Per Sample = average time per utterance")


if __name__ == "__main__":
    main()
