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
Kokoro TTS Quantization Benchmark

Tests INT8 and INT4 quantization impact on:
1. Full model inference speed
2. Memory usage
3. Output quality (numerical difference)

Motivation: Previous work (#1316) found that WhisperMLX encoder shows no speedup
from quantization (non-linear ops dominate). Kokoro is also non-autoregressive,
so we expect similar results.
"""

import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.pytorch_to_mlx.converters.kokoro_converter import KokoroConverter
from tools.pytorch_to_mlx.converters.models.kokoro import (
    estimate_memory_savings,
    quantize_kokoro_model,
)


def benchmark_synthesis(model, input_ids, voice, num_runs=20, warmup=5):
    """Benchmark synthesis and return times and audio output."""
    # Warmup
    for _ in range(warmup):
        audio = model(input_ids, voice, validate_output=False)
        mx.eval(audio)

    # Benchmark
    times = []
    for _ in range(num_runs):
        mx.synchronize()
        start = time.perf_counter()
        audio = model(input_ids, voice, validate_output=False)
        mx.eval(audio)
        mx.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    return np.array(times), audio


def run_quantization_benchmark():
    print("=" * 70)
    print("Kokoro TTS Quantization Benchmark")
    print("=" * 70)

    # Load model
    print("\nLoading model from Hugging Face...")
    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf("hexgrad/Kokoro-82M")
    model.set_deterministic(True)  # For reproducible output comparison
    print("Model loaded.")

    # Load voice embedding
    voice_path = Path(__file__).parent.parent / "src" / "kokoro" / "voices" / "af_bella.npy"
    if voice_path.exists():
        voice_data = np.load(voice_path)
        voice = mx.array(voice_data)[None, :]
        print(f"Voice loaded: {voice.shape}")
    else:
        print(f"Voice file not found: {voice_path}")
        print("Using random voice embedding...")
        voice = mx.random.normal((1, 256))

    # Test sequences
    test_sequences = [
        ("Short (32 tokens)", list(range(1, 33))),
        ("Medium (64 tokens)", list(range(1, 65))),
        ("Long (128 tokens)", list(range(1, 129))),
    ]

    # Memory estimation before quantization
    print("\n" + "=" * 70)
    print("Memory Estimation (before quantization)")
    print("=" * 70)
    for bits in [8, 4]:
        est = estimate_memory_savings(model, bits=bits, mode="full")
        print(f"\nINT{bits} Full Mode:")
        print(f"  Original:   {est['original_mb']:.1f} MB")
        print(f"  Quantized:  {est['quantized_mb']:.1f} MB")
        print(f"  Savings:    {est['savings_mb']:.1f} MB ({est['savings_percent']:.1f}%)")
        print(f"  Quantizable params: {est['quantizable_params']:,}")

    results = {}

    # Benchmark FP16 (baseline)
    print("\n" + "=" * 70)
    print("Benchmarking FP16 (baseline)")
    print("=" * 70)

    for name, tokens in test_sequences:
        input_ids = mx.array([tokens])
        times_fp16, audio_fp16 = benchmark_synthesis(model, input_ids, voice)
        print(f"\n{name}:")
        print(f"  Mean: {times_fp16.mean() * 1000:.2f} ms")
        print(f"  Std:  {times_fp16.std() * 1000:.2f} ms")
        results[f"FP16_{len(tokens)}"] = {
            "mean_ms": times_fp16.mean() * 1000,
            "std_ms": times_fp16.std() * 1000,
            "audio": audio_fp16,
        }

    # Test quantization modes
    for bits in [8, 4]:
        print("\n" + "=" * 70)
        print(f"Benchmarking INT{bits} (full mode)")
        print("=" * 70)

        # Reload fresh model for quantization
        print("Reloading model for quantization...")
        model, config, _ = converter.load_from_hf("hexgrad/Kokoro-82M")
        model.set_deterministic(True)

        # IMPORTANT: Do a warmup forward pass to trigger lazy-built fused layers
        # (e.g., fused QKV in BERT attention) BEFORE quantization
        print("Warming up model to build fused layers...")
        dummy_input = mx.array([[1, 2, 3, 4, 5]])
        dummy_voice = mx.zeros((1, 256))
        _ = model(dummy_input, dummy_voice, validate_output=False)
        mx.eval(_)

        # Quantize
        print(f"Quantizing to INT{bits}...")
        stats = quantize_kokoro_model(model, bits=bits, mode="full")
        print(f"  Quantized: {stats['total_quantized']} layers")
        print(f"  Skipped:   {stats['total_skipped']} layers")

        for name, tokens in test_sequences:
            input_ids = mx.array([tokens])
            times_quant, audio_quant = benchmark_synthesis(model, input_ids, voice)

            # Compare with FP16 output
            fp16_key = f"FP16_{len(tokens)}"
            audio_fp16 = results[fp16_key]["audio"]

            # Compute error metrics
            min_len = min(audio_fp16.shape[1], audio_quant.shape[1])
            diff = np.abs(np.array(audio_fp16[0, :min_len]) - np.array(audio_quant[0, :min_len]))
            max_diff = float(diff.max())
            mean_diff = float(diff.mean())

            fp16_time = results[fp16_key]["mean_ms"]
            quant_time = times_quant.mean() * 1000
            speedup = fp16_time / quant_time

            print(f"\n{name}:")
            print(f"  Mean: {quant_time:.2f} ms (FP16: {fp16_time:.2f} ms)")
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  Max diff vs FP16: {max_diff:.6f}")
            print(f"  Mean diff vs FP16: {mean_diff:.6f}")

            results[f"INT{bits}_{len(tokens)}"] = {
                "mean_ms": quant_time,
                "std_ms": times_quant.std() * 1000,
                "speedup": speedup,
                "max_diff": max_diff,
                "mean_diff": mean_diff,
            }

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Kokoro Quantization Benchmark Results")
    print("=" * 70)
    print(f"\n{'Config':<15} {'32 tok (ms)':>12} {'64 tok (ms)':>12} {'128 tok (ms)':>13}")
    print("-" * 55)

    for quant in ["FP16", "INT8", "INT4"]:
        row = f"{quant:<15}"
        for tokens in [32, 64, 128]:
            key = f"{quant}_{tokens}"
            if key in results:
                row += f" {results[key]['mean_ms']:>11.1f}"
            else:
                row += f" {'N/A':>11}"
        print(row)

    print("\n" + "-" * 55)
    print(f"\n{'Speedup vs FP16':<15} {'32 tok':>12} {'64 tok':>12} {'128 tok':>13}")
    print("-" * 55)

    for quant in ["INT8", "INT4"]:
        row = f"{quant:<15}"
        for tokens in [32, 64, 128]:
            key = f"{quant}_{tokens}"
            if key in results:
                row += f" {results[key]['speedup']:>11.2f}x"
            else:
                row += f" {'N/A':>11}"
        print(row)

    print("\n" + "-" * 55)
    print(f"\n{'Max Diff vs FP16':<15} {'32 tok':>12} {'64 tok':>12} {'128 tok':>13}")
    print("-" * 55)

    for quant in ["INT8", "INT4"]:
        row = f"{quant:<15}"
        for tokens in [32, 64, 128]:
            key = f"{quant}_{tokens}"
            if key in results:
                row += f" {results[key]['max_diff']:>11.4f}"
            else:
                row += f" {'N/A':>11}"
        print(row)

    # Conclusion
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    avg_int8_speedup = np.mean([results[f"INT8_{t}"]["speedup"] for t in [32, 64, 128]])
    avg_int4_speedup = np.mean([results[f"INT4_{t}"]["speedup"] for t in [32, 64, 128]])

    print(f"\nAverage INT8 speedup: {avg_int8_speedup:.2f}x")
    print(f"Average INT4 speedup: {avg_int4_speedup:.2f}x")

    if avg_int8_speedup < 1.05:
        print("\nConclusion: Quantization provides NO significant speedup for Kokoro.")
        print("This matches WhisperMLX encoder findings - non-autoregressive models")
        print("are dominated by non-linear ops (Conv1d, LayerNorm, GELU, etc.)")
    elif avg_int8_speedup < 1.15:
        print("\nConclusion: Quantization provides MODEST speedup (~10%) for Kokoro.")
        print("May be worth using for memory savings.")
    else:
        print("\nConclusion: Quantization provides SIGNIFICANT speedup for Kokoro!")
        print("Recommend using INT8 for production deployments.")

    return 0


if __name__ == "__main__":
    sys.exit(run_quantization_benchmark())
