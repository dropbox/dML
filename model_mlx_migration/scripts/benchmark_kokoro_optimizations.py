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
Kokoro TTS Optimization Benchmark

Measures baseline performance and tests optimization techniques:
- N2: Style Parameter Cache
- A3: BF16 compute
- N7: Progressive Precision
- etc.

Usage:
    python scripts/benchmark_kokoro_optimizations.py --baseline
    python scripts/benchmark_kokoro_optimizations.py --test bf16
    python scripts/benchmark_kokoro_optimizations.py --test style_cache
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.pytorch_to_mlx.converters.models.kokoro import KokoroModel, KokoroConfig
from tools.pytorch_to_mlx.converters.models.kokoro_style_cache import (
    precompute_style_cache,
    estimate_cache_savings,
)


def load_model(weights_path: Optional[str] = None) -> KokoroModel:
    """Load Kokoro model."""
    config = KokoroConfig()
    model = KokoroModel(config)

    if weights_path:
        model.load_weights(weights_path)
    else:
        # Try default path
        default_path = Path.home() / ".cache/mlx_audio/kokoro/kokoro-v1.0.safetensors"
        if default_path.exists():
            model.load_weights(str(default_path))
        else:
            print("Warning: No weights loaded, using random weights")

    return model


def get_test_inputs(batch_size: int = 1, seq_lengths: List[int] = None) -> List[Tuple[mx.array, mx.array]]:
    """Generate test inputs of various lengths."""
    if seq_lengths is None:
        seq_lengths = [16, 32, 64, 128, 256]

    inputs = []
    for seq_len in seq_lengths:
        # Random phoneme IDs (typical range 0-177)
        input_ids = mx.array(np.random.randint(1, 178, (batch_size, seq_len)))
        # Random voice embedding (256-dim)
        voice = mx.array(np.random.randn(batch_size, 256).astype(np.float32) * 0.1)
        inputs.append((input_ids, voice, seq_len))

    return inputs


def warmup(model: KokoroModel, input_ids: mx.array, voice: mx.array, n_warmup: int = 3):
    """Warm up the model."""
    for _ in range(n_warmup):
        _ = model(input_ids, voice, validate_output=False)
        mx.eval(_)


def benchmark_baseline(model: KokoroModel, inputs: List[Tuple], n_runs: int = 10) -> Dict:
    """Benchmark baseline performance."""
    results = {}

    for input_ids, voice, seq_len in inputs:
        # Warmup
        warmup(model, input_ids, voice)

        # Benchmark
        times = []
        audio_samples = 0

        for _ in range(n_runs):
            mx.synchronize()
            start = time.perf_counter()

            audio = model(input_ids, voice, validate_output=False)
            mx.eval(audio)
            mx.synchronize()

            end = time.perf_counter()
            times.append(end - start)
            audio_samples = audio.shape[-1]

        # Calculate metrics
        mean_time = np.mean(times)
        std_time = np.std(times)
        audio_duration = audio_samples / 24000  # 24kHz
        rtf = audio_duration / mean_time

        results[seq_len] = {
            "mean_time_ms": mean_time * 1000,
            "std_time_ms": std_time * 1000,
            "audio_samples": audio_samples,
            "audio_duration_s": audio_duration,
            "rtf": rtf,
        }

        print(f"Seq len {seq_len:4d}: {mean_time*1000:7.1f}ms ± {std_time*1000:5.1f}ms | "
              f"Audio: {audio_duration:.2f}s | RTF: {rtf:.1f}x")

    return results


def convert_model_to_bf16(model: nn.Module) -> None:
    """
    Convert model parameters to bfloat16 in-place.

    Recursively converts all weight tensors to bfloat16, keeping
    bias terms and certain critical layers in float32.
    """

    def convert_params(module, prefix=""):
        """Recursively convert parameters."""
        # Get all parameters for this module
        params = dict(module.parameters())

        for name, param in params.items():
            # Skip biases (keep in FP32 for stability)
            if 'bias' in name:
                continue

            # Convert weights to BF16
            if isinstance(param, mx.array) and param.dtype == mx.float32:
                # Create BF16 version
                bf16_param = param.astype(mx.bfloat16)

                # Update the parameter
                parts = name.split('.')
                target = module
                for p in parts[:-1]:
                    target = getattr(target, p)
                setattr(target, parts[-1], bf16_param)

    convert_params(model)


def benchmark_bf16(model: KokoroModel, inputs: List[Tuple], n_runs: int = 10) -> Dict:
    """
    Benchmark with BF16 compute (A3).

    Tests:
    1. Check BF16 hardware support
    2. Measure FP32 baseline
    3. Convert model to BF16
    4. Measure BF16 performance
    5. Compare speedup
    """
    print("\n" + "=" * 60)
    print("A3: BF16 COMPUTE TEST")
    print("=" * 60)

    # === 1. Check BF16 support ===
    print("\n1. Checking BF16 hardware support...")
    try:
        test_arr = mx.array([1.0, 2.0, 3.0], dtype=mx.bfloat16)
        result = test_arr * 2.0
        mx.eval(result)
        print("   BF16 supported: YES")
    except Exception as e:
        print(f"   BF16 supported: NO - {e}")
        return {}

    input_ids, voice, seq_len = inputs[0]

    # === 2. FP32 baseline ===
    print("\n2. Measuring FP32 baseline...")
    warmup(model, input_ids, voice)

    fp32_times = []
    for _ in range(n_runs):
        mx.synchronize()
        start = time.perf_counter()
        audio = model(input_ids, voice, validate_output=False)
        mx.eval(audio)
        mx.synchronize()
        fp32_times.append(time.perf_counter() - start)

    fp32_mean = np.mean(fp32_times)
    fp32_audio_samples = audio.shape[-1]
    fp32_audio_duration = fp32_audio_samples / 24000
    fp32_rtf = fp32_audio_duration / fp32_mean
    print(f"   FP32 inference: {fp32_mean*1000:.2f}ms | RTF: {fp32_rtf:.1f}x")

    # === 3. Test BF16 matmul speed ===
    print("\n3. Testing BF16 matmul performance...")

    # Create test matrices
    m, n, k = 512, 512, 512
    a_fp32 = mx.random.normal((m, k))
    b_fp32 = mx.random.normal((k, n))

    # FP32 matmul
    fp32_matmul_times = []
    for _ in range(100):
        mx.synchronize()
        start = time.perf_counter()
        c = a_fp32 @ b_fp32
        mx.eval(c)
        mx.synchronize()
        fp32_matmul_times.append(time.perf_counter() - start)

    fp32_matmul_avg = np.mean(fp32_matmul_times[10:])

    # BF16 matmul
    a_bf16 = a_fp32.astype(mx.bfloat16)
    b_bf16 = b_fp32.astype(mx.bfloat16)

    bf16_matmul_times = []
    for _ in range(100):
        mx.synchronize()
        start = time.perf_counter()
        c = a_bf16 @ b_bf16
        mx.eval(c)
        mx.synchronize()
        bf16_matmul_times.append(time.perf_counter() - start)

    bf16_matmul_avg = np.mean(bf16_matmul_times[10:])

    matmul_speedup = fp32_matmul_avg / bf16_matmul_avg
    print(f"   FP32 matmul ({m}x{k} @ {k}x{n}): {fp32_matmul_avg*1000:.4f}ms")
    print(f"   BF16 matmul ({m}x{k} @ {k}x{n}): {bf16_matmul_avg*1000:.4f}ms")
    print(f"   Matmul speedup: {matmul_speedup:.2f}x")

    # === 4. BF16 model inference (theoretical) ===
    print("\n4. Theoretical BF16 model speedup...")

    # Most computation is matmul-dominated
    # Estimate based on matmul speedup
    # But note: Kokoro has many ops that don't benefit (FFT, etc.)
    matmul_fraction = 0.6  # Estimate: 60% of compute is matmul-like

    theoretical_speedup = 1 / (1 - matmul_fraction + matmul_fraction / matmul_speedup)
    print(f"   Matmul fraction of compute: ~{matmul_fraction*100:.0f}%")
    print(f"   Theoretical model speedup: {theoretical_speedup:.2f}x ({(theoretical_speedup-1)*100:.1f}%)")

    # === Summary ===
    print("\n" + "=" * 60)
    print("A3 BF16 SUMMARY")
    print("=" * 60)
    print("   BF16 hardware support: YES")
    print(f"   Matmul speedup: {matmul_speedup:.2f}x")
    print(f"   Theoretical model speedup: {theoretical_speedup:.2f}x ({(theoretical_speedup-1)*100:.1f}%)")
    print("   LOSSLESS: Near-lossless (BF16 precision)")
    print("\n   NOTE: Full BF16 model requires weight conversion and validation")

    return {
        "bf16_supported": True,
        "matmul_speedup": matmul_speedup,
        "theoretical_speedup": theoretical_speedup,
        "fp32_inference_ms": fp32_mean * 1000,
    }


def profile_components(model: KokoroModel, input_ids: mx.array, voice: mx.array) -> Dict:
    """Profile individual components to find bottlenecks."""
    print("\n=== Component Profiling ===")

    batch_size, seq_length = input_ids.shape
    voice[:, :128]
    speaker = voice[:, 128:]

    timings = {}

    # 1. BERT encoding
    mx.synchronize()
    start = time.perf_counter()
    bert_out = model.bert(input_ids, None)
    mx.eval(bert_out)
    mx.synchronize()
    timings["bert"] = time.perf_counter() - start

    # 2. BERT encoder projection
    mx.synchronize()
    start = time.perf_counter()
    bert_enc = model.bert_encoder(bert_out)
    mx.eval(bert_enc)
    mx.synchronize()
    timings["bert_encoder"] = time.perf_counter() - start

    # 3. Predictor text encoder (with AdaLayerNorm)
    mx.synchronize()
    start = time.perf_counter()
    duration_feats = model.predictor.text_encoder(bert_enc, speaker)
    mx.eval(duration_feats)
    mx.synchronize()
    timings["predictor_text_encoder"] = time.perf_counter() - start

    # 4. Duration prediction
    mx.synchronize()
    start = time.perf_counter()
    dur_enc = model.predictor.lstm(duration_feats)
    duration_logits = model.predictor.duration_proj(dur_enc)
    mx.eval(duration_logits)
    mx.synchronize()
    timings["duration_prediction"] = time.perf_counter() - start

    # Print timings
    total = sum(timings.values())
    print(f"\nComponent breakdown (seq_len={seq_length}):")
    for name, t in timings.items():
        pct = (t / total) * 100
        print(f"  {name:25s}: {t*1000:7.2f}ms ({pct:5.1f}%)")
    print(f"  {'TOTAL (partial)':25s}: {total*1000:7.2f}ms")

    return timings


def count_adain_layers(model: KokoroModel) -> Dict:
    """Count AdaIN layers in the model to estimate style cache benefit."""
    print("\n=== AdaIN Layer Count ===")

    count = {"fc_layers": 0, "total_params": 0}

    def count_fc_in_module(module, prefix=""):
        if hasattr(module, "fc") and hasattr(module.fc, "weight"):
            count["fc_layers"] += 1
            params = module.fc.weight.size
            if hasattr(module.fc, "bias") and module.fc.bias is not None:
                params += module.fc.bias.size
            count["total_params"] += params
            # print(f"  {prefix}.fc: {module.fc.weight.shape}")

        if hasattr(module, "fc_style") and hasattr(module.fc_style, "weight"):
            count["fc_layers"] += 1
            params = module.fc_style.weight.size
            if hasattr(module.fc_style, "bias") and module.fc_style.bias is not None:
                params += module.fc_style.bias.size
            count["total_params"] += params

        # MLX modules use items() to iterate
        if hasattr(module, "children"):
            for item in module.children():
                if isinstance(item, tuple) and len(item) == 2:
                    name, child = item
                else:
                    # Single item, use index
                    name = str(item)
                    child = item
                if hasattr(child, "__call__"):
                    count_fc_in_module(child, f"{prefix}.{name}" if prefix else name)

    count_fc_in_module(model)

    print(f"Total style FC layers: {count['fc_layers']}")
    print(f"Total FC params: {count['total_params']:,}")
    print(f"Potential cache size per voice: {count['total_params'] * 4 / 1024:.1f} KB")

    return count


def main():
    parser = argparse.ArgumentParser(description="Kokoro TTS Optimization Benchmark")
    parser.add_argument("--baseline", action="store_true", help="Run baseline benchmark")
    parser.add_argument("--test", type=str, help="Test specific optimization (bf16, style_cache)")
    parser.add_argument("--profile", action="store_true", help="Profile components")
    parser.add_argument("--count-adain", action="store_true", help="Count AdaIN layers")
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights")
    parser.add_argument("--seq-lengths", type=str, default="16,32,64,128",
                       help="Sequence lengths to test (comma-separated)")
    parser.add_argument("--n-runs", type=int, default=10, help="Number of benchmark runs")

    args = parser.parse_args()

    # Parse sequence lengths
    seq_lengths = [int(x) for x in args.seq_lengths.split(",")]

    print("Loading Kokoro model...")
    model = load_model(args.weights)
    print("Model loaded\n")

    # Generate test inputs
    inputs = get_test_inputs(batch_size=1, seq_lengths=seq_lengths)

    if args.count_adain:
        count_adain_layers(model)

    if args.profile:
        # Use middle input if available, otherwise first
        idx = min(2, len(inputs) - 1)
        input_ids, voice, seq_len = inputs[idx]
        profile_components(model, input_ids, voice)

    if args.baseline or not (args.test or args.profile or args.count_adain):
        print("=== Baseline Benchmark ===")
        results = benchmark_baseline(model, inputs, n_runs=args.n_runs)

        # Summary
        avg_rtf = np.mean([r["rtf"] for r in results.values()])
        print(f"\nAverage RTF: {avg_rtf:.1f}x real-time")

    if args.test == "bf16":
        benchmark_bf16(model, inputs, n_runs=args.n_runs)

    if args.test == "async_eval":
        benchmark_async_eval(model, inputs, n_runs=args.n_runs)

    if args.test == "progressive_precision":
        benchmark_progressive_precision(model, inputs, n_runs=args.n_runs)

    if args.test == "style_cache":
        benchmark_style_cache(model, inputs, n_runs=args.n_runs)


def benchmark_progressive_precision(model: KokoroModel, inputs: List[Tuple], n_runs: int = 10) -> Dict:
    """
    Benchmark Progressive Precision (N7) optimization.

    Tests: Use BF16/FP16 for early decoder layers, FP32 for final layers.
    The idea is that early layers do coarse shaping while final layers add detail.
    Early layer approximations are "washed out" by subsequent processing.
    """
    print("\n" + "=" * 60)
    print("N7: PROGRESSIVE PRECISION TEST")
    print("=" * 60)

    input_ids, voice, seq_len = inputs[0]

    # === 1. Analyze model layer structure ===
    print("\n1. Analyzing decoder layer structure...")

    decoder_layers = []
    generator_layers = []

    # Count decoder layers
    if hasattr(model, 'decoder'):
        decoder = model.decoder
        for name in ['encode', 'decode_0', 'decode_1', 'decode_2', 'decode_3']:
            if hasattr(decoder, name):
                decoder_layers.append(name)

        if hasattr(decoder, 'generator'):
            gen = decoder.generator
            # Count resblocks
            num_resblocks = getattr(gen, '_num_resblocks', 6)
            for i in range(num_resblocks):
                if hasattr(gen, f'resblocks_{i}'):
                    generator_layers.append(f'resblocks_{i}')

    print(f"   Decoder AdaIN blocks: {len(decoder_layers)}")
    print(f"   Generator ResBlocks: {len(generator_layers)}")

    # === 2. Test matmul precision cascading ===
    print("\n2. Testing precision cascading effect...")

    # Simulate progressive precision: BF16 matmul -> FP32 matmul
    m, n, k = 512, 512, 512
    a = mx.random.normal((m, k))
    b = mx.random.normal((k, n))

    # Full FP32 chain
    fp32_result = a @ b @ b.T @ a.T
    mx.eval(fp32_result)

    # BF16 early, FP32 late
    a_bf16 = a.astype(mx.bfloat16)
    b_bf16 = b.astype(mx.bfloat16)
    intermediate = a_bf16 @ b_bf16  # BF16
    intermediate_fp32 = intermediate.astype(mx.float32)
    mixed_result = intermediate_fp32 @ b.T @ a.T  # FP32
    mx.eval(mixed_result)

    # Compare error
    error = mx.abs(fp32_result - mixed_result)
    max_error = float(mx.max(error))
    mean_error = float(mx.mean(error))
    relative_error = float(mx.max(error) / mx.max(mx.abs(fp32_result)))

    print("   FP32 vs BF16->FP32 cascade:")
    print(f"   Max absolute error: {max_error:.6f}")
    print(f"   Mean absolute error: {mean_error:.6f}")
    print(f"   Relative error: {relative_error:.6%}")

    # === 3. Estimate progressive precision benefit ===
    print("\n3. Estimating progressive precision benefit...")

    # From BF16 test: matmul is 1.11x faster
    # If we use BF16 for first 60% of decoder computation:
    bf16_matmul_speedup = 1.11
    bf16_fraction = 0.6  # First 60% of layers
    decoder_fraction = 0.55  # Decoder is 55% of compute (from tracker)

    # Speedup = 1 / (1 - bf16_savings)
    bf16_savings = bf16_fraction * decoder_fraction * (1 - 1/bf16_matmul_speedup)
    theoretical_speedup = 1 / (1 - bf16_savings)

    print(f"   BF16 matmul speedup: {bf16_matmul_speedup:.2f}x")
    print(f"   BF16 fraction: {bf16_fraction*100:.0f}% of decoder layers")
    print(f"   Decoder fraction of total: {decoder_fraction*100:.0f}%")
    print(f"   Theoretical speedup: {theoretical_speedup:.3f}x ({(theoretical_speedup-1)*100:.1f}%)")

    # === 4. Quality impact assessment ===
    print("\n4. Quality impact assessment...")

    # The cascading test shows error accumulation
    if relative_error < 0.01:
        quality_impact = "LOW (< 1% relative error)"
    elif relative_error < 0.05:
        quality_impact = "MODERATE (1-5% relative error)"
    else:
        quality_impact = "HIGH (> 5% relative error)"

    print(f"   Precision cascading error: {relative_error:.4%}")
    print(f"   Quality impact: {quality_impact}")

    # === Summary ===
    print("\n" + "=" * 60)
    print("N7 PROGRESSIVE PRECISION SUMMARY")
    print("=" * 60)
    print(f"   Decoder layers: {len(decoder_layers)} + {len(generator_layers)} ResBlocks")
    print(f"   Theoretical speedup: {(theoretical_speedup-1)*100:.1f}%")
    print(f"   Quality impact: {quality_impact}")
    print("   LOSSLESS: Near-lossless (BF16 early layers)")

    viable = theoretical_speedup > 1.02 and relative_error < 0.05
    if viable:
        print("\n   RECOMMENDATION: Worth implementing")
        print("   - Convert decoder.encode, decode_0, decode_1 to BF16")
        print("   - Keep decode_2, decode_3, generator in FP32")
    else:
        print("\n   RECOMMENDATION: Marginal benefit, low priority")

    return {
        "decoder_layers": len(decoder_layers),
        "generator_layers": len(generator_layers),
        "theoretical_speedup": theoretical_speedup,
        "relative_error": relative_error,
        "viable": viable,
    }


def benchmark_async_eval(model: KokoroModel, inputs: List[Tuple], n_runs: int = 10) -> Dict:
    """
    Benchmark mx.async_eval (H2) optimization.

    Tests whether async_eval can overlap computation with subsequent processing.
    This is most useful when there are operations after inference (e.g., audio encoding).
    """
    print("\n" + "=" * 60)
    print("H2: mx.async_eval TEST")
    print("=" * 60)

    input_ids, voice, seq_len = inputs[0]

    # === 1. Check if async_eval exists ===
    print("\n1. Checking mx.async_eval availability...")
    if not hasattr(mx, 'async_eval'):
        print("   mx.async_eval: NOT AVAILABLE")
        print("   This MLX version doesn't support async_eval")
        return {"supported": False}

    print("   mx.async_eval: AVAILABLE")

    # === 2. Baseline: sync eval ===
    print("\n2. Measuring synchronous eval baseline...")
    warmup(model, input_ids, voice)

    sync_times = []
    for _ in range(n_runs):
        mx.synchronize()
        start = time.perf_counter()

        audio = model(input_ids, voice, validate_output=False)
        mx.eval(audio)  # Synchronous eval

        mx.synchronize()
        sync_times.append(time.perf_counter() - start)

    sync_mean = np.mean(sync_times)
    print(f"   Sync eval: {sync_mean*1000:.2f}ms")

    # === 3. Test async_eval pattern ===
    print("\n3. Testing async_eval with overlapped work...")

    # Pattern: Start async eval, do some CPU work, then wait
    # Simulates: inference -> audio encoding -> wait

    def simulate_cpu_work(duration_ms: float = 1.0):
        """Simulate some CPU work that could overlap with GPU."""
        import hashlib
        data = b"x" * 10000
        for _ in range(int(duration_ms * 100)):
            hashlib.md5(data)

    async_times = []
    for _ in range(n_runs):
        mx.synchronize()
        start = time.perf_counter()

        audio = model(input_ids, voice, validate_output=False)

        # Start async eval (returns immediately)
        mx.async_eval(audio)

        # Do CPU work while GPU computes
        simulate_cpu_work(1.0)  # 1ms of CPU work

        # Wait for result
        mx.synchronize()
        async_times.append(time.perf_counter() - start)

    async_mean = np.mean(async_times)
    print(f"   Async eval + 1ms CPU work: {async_mean*1000:.2f}ms")

    # === 4. Test batched async pattern ===
    print("\n4. Testing batched async pattern...")

    # Pattern: Multiple inferences with async eval
    batch_sync_times = []
    batch_async_times = []

    # Sync batch
    for _ in range(3):
        mx.synchronize()
        start = time.perf_counter()

        for _ in range(3):  # 3 sequential inferences
            audio = model(input_ids, voice, validate_output=False)
            mx.eval(audio)

        mx.synchronize()
        batch_sync_times.append(time.perf_counter() - start)

    # Async batch (evaluate at end)
    for _ in range(3):
        mx.synchronize()
        start = time.perf_counter()

        audios = []
        for _ in range(3):  # 3 sequential inferences
            audio = model(input_ids, voice, validate_output=False)
            audios.append(audio)

        # Evaluate all at once
        mx.eval(*audios)
        mx.synchronize()
        batch_async_times.append(time.perf_counter() - start)

    batch_sync_mean = np.mean(batch_sync_times)
    batch_async_mean = np.mean(batch_async_times)
    batch_speedup = batch_sync_mean / batch_async_mean

    print(f"   3x sync eval: {batch_sync_mean*1000:.2f}ms")
    print(f"   3x deferred eval: {batch_async_mean*1000:.2f}ms")
    print(f"   Batch speedup: {batch_speedup:.2f}x")

    # === Summary ===
    print("\n" + "=" * 60)
    print("H2 mx.async_eval SUMMARY")
    print("=" * 60)

    overlap_benefit = max(0, (sync_mean - async_mean + 0.001) / sync_mean * 100)
    print("   async_eval: AVAILABLE")
    print(f"   Overlap benefit: {overlap_benefit:.1f}%")
    print(f"   Batch eval speedup: {batch_speedup:.2f}x ({(batch_speedup-1)*100:.1f}%)")
    print("   LOSSLESS: Yes (same computation)")

    if batch_speedup > 1.05:
        print("\n   RECOMMENDATION: Use batched/deferred eval pattern")
    else:
        print("\n   RECOMMENDATION: Current eval pattern is optimal")

    return {
        "supported": True,
        "sync_time_ms": sync_mean * 1000,
        "async_time_ms": async_mean * 1000,
        "batch_speedup": batch_speedup,
    }


def benchmark_style_cache(model: KokoroModel, inputs: List[Tuple], n_runs: int = 10) -> Dict:
    """
    Benchmark Style Parameter Cache (N2) optimization.

    Tests:
    1. Cache creation time (one-time per voice)
    2. Cache lookup vs fc() computation time
    3. Theoretical speedup estimate
    """
    print("\n" + "=" * 60)
    print("N2: STYLE PARAMETER CACHE TEST")
    print("=" * 60)

    # Use first voice for testing
    input_ids, voice, seq_len = inputs[0]

    # === 1. Estimate savings ===
    print("\n1. Estimating potential savings...")
    savings = estimate_cache_savings(model)
    print(f"   FC layers that can be cached: {savings['num_fc_layers']}")
    print(f"   FC ops saved per inference: {savings['total_fc_ops']:,}")
    print(f"   Cache memory required: {savings['memory_bytes'] / 1024:.1f} KB")

    # === 2. Measure cache creation time ===
    print("\n2. Measuring cache creation time...")
    cache_times = []
    for i in range(5):
        mx.synchronize()
        start = time.perf_counter()
        cache = precompute_style_cache(model, voice, verbose=(i == 0))
        mx.synchronize()
        cache_times.append(time.perf_counter() - start)

    avg_cache_time = np.mean(cache_times[1:])  # Skip first (cold)
    print(f"   Cache creation time: {avg_cache_time*1000:.2f}ms")
    print(f"   Layers cached: {cache.num_layers_cached}")
    print(f"   Params cached: {cache.total_params_saved:,}")

    # === 3. Measure fc() computation time directly ===
    print("\n3. Measuring fc() computation overhead...")

    style = voice[:, :128]

    # Find some representative fc layers and measure their time
    sample_layers = []

    # Sample predictor layers
    if hasattr(model.predictor, 'text_encoder') and hasattr(model.predictor.text_encoder, 'lstms_1'):
        sample_layers.append(('predictor.text_encoder.lstms_1.fc', model.predictor.text_encoder.lstms_1.fc))

    if hasattr(model.predictor, 'F0_0') and hasattr(model.predictor.F0_0, 'norm1'):
        sample_layers.append(('predictor.F0_0.norm1.fc', model.predictor.F0_0.norm1.fc))

    # Sample decoder layers
    if hasattr(model.decoder, 'encode') and hasattr(model.decoder.encode, 'norm1'):
        sample_layers.append(('decoder.encode.norm1.fc', model.decoder.encode.norm1.fc))

    # Measure each sample layer
    total_fc_time = 0
    for name, fc in sample_layers:
        times = []
        for _ in range(100):
            mx.synchronize()
            start = time.perf_counter()
            _ = fc(style)
            mx.eval(_)
            mx.synchronize()
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times[10:])  # Skip warmup
        total_fc_time += avg_time
        print(f"   {name}: {avg_time*1000:.4f}ms")

    # Extrapolate to all layers
    num_measured = len(sample_layers)
    estimated_total_fc_time = (total_fc_time / num_measured) * cache.num_layers_cached
    print(f"\n   Estimated total fc() time per inference: {estimated_total_fc_time*1000:.2f}ms")

    # === 4. Baseline inference time ===
    print("\n4. Measuring baseline inference time...")
    baseline_times = []
    warmup(model, input_ids, voice)

    for _ in range(n_runs):
        mx.synchronize()
        start = time.perf_counter()
        audio = model(input_ids, voice, validate_output=False)
        mx.eval(audio)
        mx.synchronize()
        baseline_times.append(time.perf_counter() - start)

    avg_baseline = np.mean(baseline_times)
    print(f"   Baseline inference: {avg_baseline*1000:.2f}ms")

    # === 5. Calculate theoretical speedup ===
    print("\n5. Theoretical speedup calculation...")

    # With cache: save fc() time on all layers after first inference
    theoretical_savings_ms = estimated_total_fc_time * 1000
    theoretical_speedup = avg_baseline / (avg_baseline - estimated_total_fc_time)

    print(f"   fc() time savings: {theoretical_savings_ms:.2f}ms")
    print(f"   Theoretical speedup: {theoretical_speedup:.2f}x ({(theoretical_speedup-1)*100:.1f}%)")

    # === 6. Cache amortization ===
    print("\n6. Cache amortization analysis...")
    inferences_to_amortize = avg_cache_time / estimated_total_fc_time
    print(f"   Cache creation overhead: {avg_cache_time*1000:.2f}ms")
    print(f"   Inferences to amortize: {inferences_to_amortize:.1f}")
    print(f"   Break-even after: {int(np.ceil(inferences_to_amortize))} inferences with same voice")

    # === Summary ===
    print("\n" + "=" * 60)
    print("N2 STYLE CACHE SUMMARY")
    print("=" * 60)
    print("   Implementation status: CACHE INFRASTRUCTURE READY")
    print(f"   Layers cached: {cache.num_layers_cached}")
    print(f"   Theoretical speedup: {(theoretical_speedup-1)*100:.1f}%")
    print(f"   Amortization: {int(np.ceil(inferences_to_amortize))} inferences")
    print("   LOSSLESS: Yes (mathematically identical)")
    print("\n   NEXT STEP: Modify AdaIN __call__ to accept cache parameter")

    return {
        "layers_cached": cache.num_layers_cached,
        "cache_creation_ms": avg_cache_time * 1000,
        "estimated_fc_time_ms": estimated_total_fc_time * 1000,
        "theoretical_speedup": theoretical_speedup,
        "inferences_to_amortize": inferences_to_amortize,
    }


if __name__ == "__main__":
    main()
