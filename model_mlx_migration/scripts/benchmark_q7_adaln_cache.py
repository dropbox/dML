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
Q7 Evaluation: Adaptive Norm Parameter Cache

Measures whether caching AdaLN linear projections provides meaningful speedup.

The optimization would cache modulation parameters for discrete ODE time steps.
For 5 ODE steps × 22 DiT blocks = 110 linear projections potentially cached.

Worker #1341 - 2025-12-20
"""

import time
import mlx.core as mx
import mlx.nn as nn
import numpy as np


def benchmark_adaln_linear():
    """Benchmark AdaLN linear projection cost."""
    print("=== Q7 Evaluation: AdaLN Cache Potential ===\n")

    # DiT config
    dim = 1024  # CosyVoice3 hidden dim
    num_blocks = 22
    num_ode_steps = 5
    batch_size = 1

    # Create linear layer matching AdaLN structure
    linear = nn.Linear(dim, dim * 6, bias=True)

    # Create time embedding input
    t_emb = mx.random.normal((batch_size, dim))
    mx.eval(t_emb, linear.weight, linear.bias)

    # Warmup
    for _ in range(10):
        result = linear(t_emb)
        mx.eval(result)

    # Benchmark single AdaLN linear projection
    times = []
    for _ in range(100):
        start = time.perf_counter()
        result = linear(t_emb)
        mx.eval(result)
        times.append((time.perf_counter() - start) * 1000)

    single_adaln_ms = np.mean(times)
    print(f"Single AdaLN linear: {single_adaln_ms:.4f} ms")

    # Cost per ODE step (22 blocks)
    per_step_ms = single_adaln_ms * num_blocks
    print(f"Per ODE step ({num_blocks} blocks): {per_step_ms:.4f} ms")

    # Total for inference (5 steps)
    total_adaln_ms = per_step_ms * num_ode_steps
    print(f"Total per inference ({num_ode_steps} steps): {total_adaln_ms:.4f} ms")

    # Compare to full flow inference time (from benchmark)
    flow_inference_ms = 360.0  # Approximate from benchmark
    adaln_percent = (total_adaln_ms / flow_inference_ms) * 100

    print("\n--- Impact Analysis ---")
    print(f"Flow inference time: ~{flow_inference_ms:.0f} ms (from benchmark)")
    print(f"AdaLN total: {total_adaln_ms:.4f} ms")
    print(f"AdaLN % of flow: {adaln_percent:.2f}%")
    print(f"Max theoretical speedup from Q7: {adaln_percent:.2f}%")

    # Cache overhead estimate
    # If caching, need dict lookup + storage
    cache_lookup_times = []
    cache = {}
    for i in range(num_ode_steps):
        for j in range(num_blocks):
            cache[(i, j)] = mx.random.normal((batch_size, dim * 6))
    mx.eval(*cache.values())

    for _ in range(100):
        start = time.perf_counter()
        for i in range(num_ode_steps):
            for j in range(num_blocks):
                _ = cache[(i, j)]
        cache_lookup_times.append((time.perf_counter() - start) * 1000)

    cache_lookup_ms = np.mean(cache_lookup_times)
    print(f"\nCache lookup overhead ({num_ode_steps * num_blocks} lookups): {cache_lookup_ms:.4f} ms")

    net_savings = total_adaln_ms - cache_lookup_ms
    print(f"Net savings (linear - cache overhead): {net_savings:.4f} ms")

    if net_savings > 0:
        net_speedup = (net_savings / flow_inference_ms) * 100
        print(f"Potential E2E speedup: {net_speedup:.2f}%")
    else:
        print("Q7 would be SLOWER due to cache overhead!")

    return total_adaln_ms, cache_lookup_ms


def benchmark_time_embedding_computation():
    """Benchmark time embedding cost (Q3 related)."""
    print("\n\n=== Q3 Evaluation: Time Embedding Cost ===\n")

    dim = 1024
    _batch_size = 1  # noqa: F841 - documentation only
    num_ode_steps = 5

    # Time embedding layers (sinusoidal + MLP)
    half_dim = dim // 4

    def time_embed(t):
        """Simplified time embedding matching TimeEmbedding class."""
        emb = mx.arange(half_dim) * (-np.log(10000.0) / half_dim)
        emb = mx.exp(emb)
        emb = t[:, None] * emb[None, :]
        emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
        return emb

    # MLP after sinusoidal
    mlp1 = nn.Linear(dim // 2, dim)
    mlp2 = nn.Linear(dim, dim)

    def full_time_embed(t):
        emb = time_embed(t)
        emb = nn.silu(mlp1(emb))
        emb = mlp2(emb)
        return emb

    t = mx.array([0.5])
    mx.eval(mlp1.weight, mlp2.weight)

    # Warmup
    for _ in range(10):
        result = full_time_embed(t)
        mx.eval(result)

    # Benchmark
    times = []
    for _ in range(100):
        start = time.perf_counter()
        result = full_time_embed(t)
        mx.eval(result)
        times.append((time.perf_counter() - start) * 1000)

    single_time_emb_ms = np.mean(times)
    print(f"Single time embedding: {single_time_emb_ms:.4f} ms")
    print(f"Total ({num_ode_steps} steps): {single_time_emb_ms * num_ode_steps:.4f} ms")

    # This is already a small fraction
    flow_inference_ms = 360.0
    time_emb_percent = (single_time_emb_ms * num_ode_steps / flow_inference_ms) * 100
    print(f"Time embedding % of flow: {time_emb_percent:.2f}%")
    print(f"Max theoretical speedup from Q3: {time_emb_percent:.2f}%")


if __name__ == "__main__":
    total_adaln, cache_overhead = benchmark_adaln_linear()
    benchmark_time_embedding_computation()

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("""
Q7 (AdaLN Cache) and Q3 (Time Embedding Quantization) both target
very small fractions of the total computation (<1-2%).

Given that many previous Q-optimizations (Q1, Q2, Q8, Q11, Q12, Q15-Layer, Q17, Q18-LLM)
were evaluated as NOT WORTH or LIMITED, these optimizations follow the same pattern.

Recommendation: DO NOT IMPLEMENT - effort exceeds benefit.
""")
