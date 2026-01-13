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
Benchmark script for Kokoro MLX model quantization.

Measures:
- Memory usage before/after quantization
- Inference latency before/after quantization
- Quality impact (if weights available)
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import mlx.core as mx


def get_memory_usage() -> dict:
    """Get current MLX memory usage."""
    try:
        stats = mx.metal.get_memory_info()
        return {
            "peak_allocated_mb": stats.get("peak_allocated", 0) / (1024 * 1024),
            "allocated_mb": stats.get("allocated", 0) / (1024 * 1024),
            "cache_mb": stats.get("cache", 0) / (1024 * 1024),
        }
    except Exception:
        return {"peak_allocated_mb": 0, "allocated_mb": 0, "cache_mb": 0}


def benchmark_inference(model, input_ids, voice, warmup=3, runs=10) -> dict:
    """Benchmark inference latency."""
    # Warmup runs - call __call__ directly to pass validate_output=False
    # since random weights produce NaN
    for _ in range(warmup):
        audio = model(input_ids, voice, validate_output=False)
        mx.eval(audio)

    # Timed runs
    latencies = []
    for _ in range(runs):
        try:
            mx.reset_peak_memory()
        except Exception:
            pass  # Ignore if not available
        start = time.perf_counter()
        audio = model(input_ids, voice, validate_output=False)
        mx.eval(audio)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms

    return {
        "mean_latency_ms": sum(latencies) / len(latencies),
        "min_latency_ms": min(latencies),
        "max_latency_ms": max(latencies),
        "samples": len(latencies),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Kokoro MLX quantization")
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128],
        help="Token counts to benchmark",
    )
    parser.add_argument(
        "--bits", type=int, choices=[4, 8], default=8, help="Quantization bits"
    )
    parser.add_argument(
        "--group-size", type=int, default=64, help="Quantization group size"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "encoder_only", "no_adain"],
        default="full",
        help="Quantization mode",
    )
    parser.add_argument(
        "--warmup", type=int, default=2, help="Warmup iterations"
    )
    parser.add_argument(
        "--runs", type=int, default=5, help="Benchmark iterations"
    )
    args = parser.parse_args()

    from tools.pytorch_to_mlx.converters.models.kokoro import (
        KokoroConfig,
        KokoroModel,
        estimate_memory_savings,
        quantize_kokoro_model,
    )

    print("=" * 60)
    print("Kokoro MLX Quantization Benchmark")
    print("=" * 60)
    print(f"Bits: {args.bits}, Group Size: {args.group_size}, Mode: {args.mode}")
    print(f"Warmup: {args.warmup}, Runs: {args.runs}")
    print()

    # Create model
    config = KokoroConfig()

    # Memory estimation
    print("=== Memory Estimation ===")
    model_est = KokoroModel(config)
    est = estimate_memory_savings(model_est, bits=args.bits, mode=args.mode)
    print(f"Original size (float32): {est['original_mb']:.1f} MB")
    print(f"Estimated quantized:     {est['quantized_mb']:.1f} MB")
    print(f"Estimated savings:       {est['savings_mb']:.1f} MB ({est['savings_percent']:.1f}%)")
    print(f"Total params:            {est['total_params']:,}")
    print(f"Quantizable params:      {est['quantizable_params']:,}")
    del model_est
    print()

    # Benchmark results table header
    print("=== Inference Benchmarks ===")
    print()
    print(
        f"{'Tokens':>8} | {'FP32 (ms)':>10} | {'Quant (ms)':>10} | "
        f"{'Speedup':>8} | {'FP32 Mem':>10} | {'Quant Mem':>10}"
    )
    print("-" * 75)

    # Voice embedding (zeros for benchmark)
    voice = mx.zeros((1, 256))

    results = []
    for num_tokens in args.tokens:
        input_ids = mx.zeros((1, num_tokens), dtype=mx.int32)

        # Benchmark FP32 model
        model_fp32 = KokoroModel(config)
        model_fp32.set_deterministic(True)
        mx.eval(model_fp32.parameters())

        try:
            mx.reset_peak_memory()
        except Exception:
            pass
        fp32_stats = benchmark_inference(
            model_fp32, input_ids, voice, warmup=args.warmup, runs=args.runs
        )
        fp32_mem = get_memory_usage()
        del model_fp32

        # Benchmark quantized model
        model_quant = KokoroModel(config)
        model_quant.set_deterministic(True)
        quant_stats = quantize_kokoro_model(
            model_quant, bits=args.bits, mode=args.mode, group_size=args.group_size
        )
        mx.eval(model_quant.parameters())

        try:
            mx.reset_peak_memory()
        except Exception:
            pass
        quant_bench = benchmark_inference(
            model_quant, input_ids, voice, warmup=args.warmup, runs=args.runs
        )
        quant_mem = get_memory_usage()
        del model_quant

        speedup = fp32_stats["mean_latency_ms"] / quant_bench["mean_latency_ms"]

        print(
            f"{num_tokens:>8} | "
            f"{fp32_stats['mean_latency_ms']:>10.1f} | "
            f"{quant_bench['mean_latency_ms']:>10.1f} | "
            f"{speedup:>7.2f}x | "
            f"{fp32_mem['peak_allocated_mb']:>9.1f}M | "
            f"{quant_mem['peak_allocated_mb']:>9.1f}M"
        )

        results.append({
            "tokens": num_tokens,
            "fp32_latency_ms": fp32_stats["mean_latency_ms"],
            "quant_latency_ms": quant_bench["mean_latency_ms"],
            "speedup": speedup,
            "fp32_peak_mem_mb": fp32_mem["peak_allocated_mb"],
            "quant_peak_mem_mb": quant_mem["peak_allocated_mb"],
        })

    print()
    print("=== Quantization Statistics ===")
    print(f"Layers quantized: {quant_stats['total_quantized']}")
    print(f"Layers skipped:   {quant_stats['total_skipped']}")
    if quant_stats["layers_skipped"]:
        print(f"Skipped layers:   {quant_stats['layers_skipped']}")

    # Summary
    print()
    print("=== Summary ===")
    avg_speedup = sum(r["speedup"] for r in results) / len(results)
    print(f"Average speedup: {avg_speedup:.2f}x")

    if results[0]["fp32_peak_mem_mb"] > 0 and results[0]["quant_peak_mem_mb"] > 0:
        mem_reduction = (
            1 - results[0]["quant_peak_mem_mb"] / results[0]["fp32_peak_mem_mb"]
        ) * 100
        print(f"Memory reduction: {mem_reduction:.1f}%")


if __name__ == "__main__":
    main()
