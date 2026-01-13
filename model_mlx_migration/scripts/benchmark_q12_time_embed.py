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
Benchmark script for Q12: Batched ODE Time Embedding Computation.

Tests the speedup from precomputing all time embeddings before the ODE loop.

Expected: 5-10% speedup on flow inference.
"""

import time
import mlx.core as mx
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.pytorch_to_mlx.converters.models.cosyvoice3_dit import (
    CausalMaskedDiffWithDiT,
    create_cosyvoice3_flow_config,
    TimeEmbedding,
)


def benchmark_time_embedding(
    time_embed: TimeEmbedding,
    batch_size: int = 1,
    num_steps: int = 10,
    warmup: int = 3,
    runs: int = 10,
) -> dict:
    """
    Benchmark time embedding computation approaches.

    Returns:
        dict with sequential and batched timings
    """
    dt = 1.0 / num_steps

    # Warmup
    for _ in range(warmup):
        for i in range(num_steps):
            t = mx.array([1.0 - i * dt] * batch_size)
            t_emb = time_embed(t)
            mx.eval(t_emb)

    # Sequential: One time embedding per step
    seq_times = []
    for _ in range(runs):
        start = time.perf_counter()
        for i in range(num_steps):
            t = mx.array([1.0 - i * dt] * batch_size)
            t_emb = time_embed(t)
            mx.eval(t_emb)
        end = time.perf_counter()
        seq_times.append(end - start)

    # Batched: All time embeddings at once
    batch_times = []
    for _ in range(runs):
        start = time.perf_counter()
        # Create all time values [num_steps, B]
        all_t = mx.stack([mx.array([1.0 - i * dt] * batch_size) for i in range(num_steps)])
        # Compute all time embeddings
        all_t_embs = time_embed(all_t.reshape(-1))  # [num_steps*B, dim]
        all_t_embs = all_t_embs.reshape(num_steps, batch_size, -1)  # [num_steps, B, dim]
        mx.eval(all_t_embs)
        end = time.perf_counter()
        batch_times.append(end - start)

    return {
        "sequential_ms": sum(seq_times) / len(seq_times) * 1000,
        "batched_ms": sum(batch_times) / len(batch_times) * 1000,
    }


def benchmark_flow_inference(
    num_steps: int = 10,
    num_tokens: int = 50,
    warmup: int = 3,
    runs: int = 5,
) -> dict:
    """
    Benchmark full flow inference.
    """
    config = create_cosyvoice3_flow_config()
    model = CausalMaskedDiffWithDiT(config)
    mx.eval(model.parameters())

    batch_size = 1

    # Create test inputs
    tokens = mx.random.randint(0, config.vocab_size, (batch_size, num_tokens))
    spk_emb = mx.random.normal((batch_size, 192))

    # Warmup
    for _ in range(warmup):
        out = model.inference(tokens, spk_emb, num_steps=num_steps)
        mx.eval(out)

    # Timed runs
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        out = model.inference(tokens, spk_emb, num_steps=num_steps)
        mx.eval(out)
        end = time.perf_counter()
        times.append(end - start)

    avg_time = sum(times) / len(times)
    return {
        "total_time_ms": avg_time * 1000,
        "per_step_ms": (avg_time * 1000) / num_steps,
    }


def main():
    print("=" * 60)
    print("Q12 Benchmark: Batched ODE Time Embedding Computation")
    print("=" * 60)

    config = create_cosyvoice3_flow_config()
    time_embed = TimeEmbedding(config.dim)
    mx.eval(time_embed.parameters())

    # Test parameters
    num_steps = 10
    batch_size = 1
    warmup = 3
    runs = 10

    print(f"\nTime Embedding Test: {num_steps} steps, batch={batch_size}")
    print("-" * 60)

    result = benchmark_time_embedding(
        time_embed,
        batch_size=batch_size,
        num_steps=num_steps,
        warmup=warmup,
        runs=runs,
    )

    print(f"\nSequential (current): {result['sequential_ms']:.3f}ms")
    print(f"Batched (proposed):   {result['batched_ms']:.3f}ms")

    speedup = result['sequential_ms'] / result['batched_ms']
    print(f"Speedup: {speedup:.2f}x")

    # Full flow inference baseline
    print("\n" + "-" * 60)
    print("Full Flow Inference Baseline:")
    print("-" * 60)

    flow_result = benchmark_flow_inference(
        num_steps=num_steps,
        num_tokens=50,
        warmup=warmup,
        runs=5,
    )

    print(f"Total: {flow_result['total_time_ms']:.2f}ms")
    print(f"Per step: {flow_result['per_step_ms']:.2f}ms")

    # Calculate theoretical max speedup from time embedding optimization
    time_embed_pct = result['sequential_ms'] / flow_result['total_time_ms'] * 100
    saved_ms = result['sequential_ms'] - result['batched_ms']
    theoretical_speedup = flow_result['total_time_ms'] / (flow_result['total_time_ms'] - saved_ms)

    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    print(f"\nTime embedding is {time_embed_pct:.1f}% of total flow inference")
    print(f"Potential time saved: {saved_ms:.3f}ms")
    print(f"Theoretical max speedup: {(theoretical_speedup-1)*100:.1f}%")

    if speedup >= 1.05 and time_embed_pct > 1.0:
        print("\nVERDICT: WORTH INVESTIGATING (time embed is meaningful % of total)")
    else:
        print(f"\nVERDICT: LIMITED BENEFIT (time embed is {time_embed_pct:.1f}% of total)")


if __name__ == "__main__":
    main()
