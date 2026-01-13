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
Kokoro TTS End-to-End Benchmark

Compares full synthesis performance:
1. Regular __call__ (dynamic sizing with sync)
2. synthesize_bucketed (fixed buckets, no sync)
3. Impact of output validation (mx.eval for NaN/Inf checks)
"""

import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.pytorch_to_mlx.converters.kokoro_converter import KokoroConverter


def benchmark_synthesis(model, input_ids, voice, name, num_runs=20, warmup=5, validate_output=True):
    """Benchmark synthesis with given parameters."""
    # Warmup
    for _ in range(warmup):
        audio = model(input_ids, voice, validate_output=validate_output)
        mx.eval(audio)  # Ensure complete

    # Benchmark
    times = []
    for _ in range(num_runs):
        mx.synchronize()  # Ensure clean start
        start = time.perf_counter()
        audio = model(input_ids, voice, validate_output=validate_output)
        mx.eval(audio)  # Force completion
        mx.synchronize()  # Ensure done
        end = time.perf_counter()
        times.append(end - start)

    times = np.array(times)
    print(f"\n{name}:")
    print(f"  Mean: {times.mean() * 1000:.2f} ms")
    print(f"  Std:  {times.std() * 1000:.2f} ms")
    print(f"  Min:  {times.min() * 1000:.2f} ms")
    print(f"  Max:  {times.max() * 1000:.2f} ms")

    return times


def benchmark_bucketed(model, input_ids, voice, bucket, name, num_runs=20, warmup=5):
    """Benchmark bucketed synthesis."""
    # Warmup
    for _ in range(warmup):
        audio, valid = model.synthesize_bucketed(input_ids, voice, frame_bucket=bucket)
        mx.eval(audio)

    # Benchmark
    times = []
    for _ in range(num_runs):
        mx.synchronize()
        start = time.perf_counter()
        audio, valid = model.synthesize_bucketed(input_ids, voice, frame_bucket=bucket)
        mx.eval(audio)
        mx.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    times = np.array(times)
    print(f"\n{name}:")
    print(f"  Mean: {times.mean() * 1000:.2f} ms")
    print(f"  Std:  {times.std() * 1000:.2f} ms")
    print(f"  Min:  {times.min() * 1000:.2f} ms")
    print(f"  Max:  {times.max() * 1000:.2f} ms")

    return times


def main():
    print("=" * 70)
    print("Kokoro TTS End-to-End Benchmark")
    print("=" * 70)

    # Load model
    print("\nLoading model from Hugging Face...")
    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf("hexgrad/Kokoro-82M")
    print("Model loaded.")

    # Load voice embedding
    voice_path = Path(__file__).parent.parent / "src" / "kokoro" / "voices" / "af_bella.npy"
    if voice_path.exists():
        voice_data = np.load(voice_path)
        voice = mx.array(voice_data)[None, :]  # [1, 256]
        print(f"Voice loaded: {voice.shape}")
    else:
        print(f"Voice file not found: {voice_path}")
        print("Using random voice embedding...")
        voice = mx.random.normal((1, 256))

    # Test sequences of different lengths
    test_cases = [
        ("Short (16 tokens)", list(range(1, 17))),
        ("Medium (64 tokens)", list(range(1, 65))),
        ("Long (128 tokens)", list(range(1, 129))),
        ("Very Long (256 tokens)", list(range(1, 257))),
    ]

    print("\n" + "=" * 70)
    print("Benchmark Configuration")
    print("=" * 70)
    print("Runs: 20, Warmup: 5")
    print(f"Frame buckets available: {model.FRAME_BUCKETS}")

    results = {}

    for case_name, tokens in test_cases:
        print("\n" + "=" * 70)
        print(f"{case_name}: {len(tokens)} tokens")
        print("=" * 70)

        input_ids = mx.array([tokens])

        # Regular synthesis with validation (default, safe)
        regular_times = benchmark_synthesis(
            model, input_ids, voice,
            "Regular (validate_output=True)",
            validate_output=True
        )

        # Regular synthesis without validation (fast, for trusted inputs)
        no_validate_times = benchmark_synthesis(
            model, input_ids, voice,
            "Regular (validate_output=False)",
            validate_output=False
        )

        # Calculate validation overhead
        validation_speedup = regular_times.mean() / no_validate_times.mean()
        print(f"\n  Speedup (no validation vs with): {validation_speedup:.2f}x")

        results[case_name] = {
            "tokens": len(tokens),
            "with_validation_ms": regular_times.mean() * 1000,
            "no_validation_ms": no_validate_times.mean() * 1000,
            "validation_speedup": validation_speedup,
        }

    # Summary
    print("\n" + "=" * 70)
    print("Summary: validate_output Impact")
    print("=" * 70)
    print(f"\n{'Case':<25} {'Tokens':>7} {'With Valid':>12} {'No Valid':>12} {'Speedup':>8}")
    print("-" * 70)
    for case_name, data in results.items():
        print(f"{case_name:<25} {data['tokens']:>7} {data['with_validation_ms']:>11.1f}ms {data['no_validation_ms']:>11.1f}ms {data['validation_speedup']:>7.2f}x")

    # Check if validation matters
    avg_speedup = np.mean([d["validation_speedup"] for d in results.values()])
    print(f"\nAverage speedup from skipping validation: {avg_speedup:.2f}x")

    if avg_speedup > 1.05:
        print("=> Recommendation: Use validate_output=False for high-throughput trusted pipelines")
    else:
        print("=> Validation overhead is minimal; keep validate_output=True for safety")

    return 0


if __name__ == "__main__":
    sys.exit(main())
