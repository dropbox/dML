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
Benchmark: Token Processing Time in ODE Inference

Measures time spent on token processing (embedding + repeat + pre_lookahead)
vs total DiT forward pass to determine if caching is worthwhile.

This is related to Q8 (Token-to-Mel Expansion Fusion) but implemented as
simple caching rather than kernel fusion.
"""

import time
import mlx.core as mx
import sys
sys.path.insert(0, '/Users/ayates/model_mlx_migration')

from tools.pytorch_to_mlx.converters.models.cosyvoice3_dit import (
    CausalMaskedDiffWithDiT,
    create_cosyvoice3_flow_config
)


def benchmark_token_processing():
    """Benchmark token processing vs total forward pass time."""
    print("=" * 60)
    print("Token Processing Benchmark")
    print("=" * 60)

    # Create model
    config = create_cosyvoice3_flow_config()
    model = CausalMaskedDiffWithDiT(config)

    # Test inputs
    batch_size = 1
    num_tokens = 50
    num_steps = 5

    tokens = mx.zeros((batch_size, num_tokens), dtype=mx.int32)
    spk_emb = mx.random.normal((batch_size, 192))
    x = mx.random.normal((batch_size, num_tokens * 2, 80))  # mel_dim=80
    t = mx.array([0.5])

    # Warmup
    print("\nWarming up...")
    for _ in range(3):
        _ = model.forward(x, tokens, t, spk_emb)
        mx.eval(_)

    num_runs = 10

    # Benchmark 1: Token processing only
    print(f"\nBenchmarking token processing ({num_runs} runs)...")
    token_times = []
    for _ in range(num_runs):
        start = time.perf_counter()

        # These ops happen every forward pass
        token_emb = model.input_embedding(tokens)
        token_emb = mx.repeat(token_emb, config.token_mel_ratio, axis=1)
        mu = model.pre_lookahead_layer(token_emb)
        mx.eval(mu)

        token_times.append(time.perf_counter() - start)

    avg_token = sum(token_times) / len(token_times) * 1000

    # Benchmark 2: Full forward pass
    print(f"Benchmarking full forward pass ({num_runs} runs)...")
    forward_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        out = model.forward(x, tokens, t, spk_emb)
        mx.eval(out)
        forward_times.append(time.perf_counter() - start)

    avg_forward = sum(forward_times) / len(forward_times) * 1000

    # Benchmark 3: Full ODE inference
    print(f"Benchmarking full ODE inference ({num_steps} steps, {num_runs} runs)...")
    inference_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        out = model.inference(tokens, spk_emb, num_steps=num_steps)
        mx.eval(out)
        inference_times.append(time.perf_counter() - start)

    avg_inference = sum(inference_times) / len(inference_times) * 1000

    # Calculate savings
    token_per_inference = avg_token * num_steps
    savings_if_cached = (num_steps - 1) * avg_token
    savings_pct = savings_if_cached / avg_inference * 100

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print("Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num tokens: {num_tokens}")
    print(f"  ODE steps:  {num_steps}")
    print()
    print("Timing (ms):")
    print(f"  Token processing (1 call):    {avg_token:.2f}ms")
    print(f"  Full forward pass (1 call):   {avg_forward:.2f}ms")
    print(f"  Full ODE inference:           {avg_inference:.2f}ms")
    print()
    print("Analysis:")
    print(f"  Token proc % of forward:      {avg_token/avg_forward*100:.1f}%")
    print(f"  Token proc per inference:     {token_per_inference:.2f}ms ({num_steps}x)")
    print(f"  Token proc % of inference:    {token_per_inference/avg_inference*100:.1f}%")
    print()
    print("If we cache token processing:")
    print(f"  Saved computation:            {savings_if_cached:.2f}ms")
    print(f"  **Theoretical speedup:        {savings_pct:.1f}%**")
    print()

    if savings_pct >= 3.0:
        print("VERDICT: WORTH IMPLEMENTING (>3% speedup)")
    elif savings_pct >= 1.0:
        print("VERDICT: MARGINAL (<3% speedup)")
    else:
        print("VERDICT: NOT WORTH (<1% speedup)")

    return {
        'token_processing_ms': avg_token,
        'forward_pass_ms': avg_forward,
        'inference_ms': avg_inference,
        'savings_pct': savings_pct
    }


def benchmark_with_optimizations():
    """Benchmark with Q18+Q6 optimizations enabled."""
    print("\n" + "=" * 60)
    print("With Q18 (QKV Fusion) + Q6 (Speaker Cache) Enabled")
    print("=" * 60)

    config = create_cosyvoice3_flow_config()
    model = CausalMaskedDiffWithDiT(config)

    # Apply optimizations
    fused = model.fuse_qkv_weights()
    print(f"Q18: Fused {fused} QKV attention blocks")

    model.enable_speaker_cache(True)

    batch_size = 1
    num_tokens = 50
    num_steps = 5

    tokens = mx.zeros((batch_size, num_tokens), dtype=mx.int32)
    spk_emb = mx.random.normal((batch_size, 192))

    # Pre-cache speaker projection
    model.cache_speaker_projection(spk_emb, "test")

    # Warmup
    print("\nWarming up...")
    for _ in range(3):
        _ = model.inference(tokens, spk_emb, num_steps=num_steps, spk_cache_id="test")
        mx.eval(_)

    num_runs = 10

    # Benchmark full ODE inference with optimizations
    print(f"Benchmarking optimized ODE inference ({num_runs} runs)...")
    inference_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        out = model.inference(tokens, spk_emb, num_steps=num_steps, spk_cache_id="test")
        mx.eval(out)
        inference_times.append(time.perf_counter() - start)

    avg_inference = sum(inference_times) / len(inference_times) * 1000

    # Benchmark token processing (unchanged by Q18/Q6)
    token_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        token_emb = model.input_embedding(tokens)
        token_emb = mx.repeat(token_emb, config.token_mel_ratio, axis=1)
        mu = model.pre_lookahead_layer(token_emb)
        mx.eval(mu)
        token_times.append(time.perf_counter() - start)

    avg_token = sum(token_times) / len(token_times) * 1000

    savings_if_cached = (num_steps - 1) * avg_token
    savings_pct = savings_if_cached / avg_inference * 100

    print(f"\nOptimized inference:        {avg_inference:.2f}ms")
    print(f"Token processing (1 call):  {avg_token:.2f}ms")
    print(f"Token proc % of optimized:  {avg_token * num_steps / avg_inference * 100:.1f}%")
    print(f"**Potential further gain:   {savings_pct:.1f}%**")

    return {
        'optimized_inference_ms': avg_inference,
        'token_processing_ms': avg_token,
        'potential_gain_pct': savings_pct
    }


if __name__ == "__main__":
    results_base = benchmark_token_processing()
    results_opt = benchmark_with_optimizations()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Baseline potential:   {results_base['savings_pct']:.1f}%")
    print(f"With Q18+Q6 enabled:  {results_opt['potential_gain_pct']:.1f}%")
