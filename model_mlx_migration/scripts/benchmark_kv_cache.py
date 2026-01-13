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
Benchmark KV cache performance for CosyVoice3 LLM.

Compares:
1. Current dynamic cache (concatenation) - tuple format
2. D1 optimized KVCache (step-based preallocation)

Goal: Determine if D1 optimization provides meaningful speedup.
"""

import sys
import time
from pathlib import Path
from typing import List, Tuple

import mlx.core as mx

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.pytorch_to_mlx.converters.models.cosyvoice2_llm import KVCache, make_kv_cache


def benchmark_dynamic_cache(
    batch: int = 1,
    num_kv_heads: int = 1,
    head_dim: int = 128,
    num_tokens: int = 100,
    num_runs: int = 5,
) -> float:
    """
    Benchmark current dynamic (concatenation) cache approach.

    Simulates autoregressive generation where each step concatenates new K/V.
    """
    times = []

    for _ in range(num_runs):
        # Simulate generation with concatenation
        k_cache = None
        v_cache = None

        start = time.perf_counter()

        for step in range(num_tokens):
            # Simulate new K/V for this token
            new_k = mx.random.normal((batch, num_kv_heads, 1, head_dim))
            new_v = mx.random.normal((batch, num_kv_heads, 1, head_dim))

            if k_cache is None:
                k_cache = new_k
                v_cache = new_v
            else:
                k_cache = mx.concatenate([k_cache, new_k], axis=2)
                v_cache = mx.concatenate([v_cache, new_v], axis=2)

            # Force evaluation
            mx.eval(k_cache, v_cache)

        end = time.perf_counter()
        times.append(end - start)

    return sum(times) / len(times)


def benchmark_preallocated_cache(
    batch: int = 1,
    num_kv_heads: int = 1,
    head_dim: int = 128,
    num_tokens: int = 100,
    max_seq_len: int = 1024,
    num_runs: int = 5,
) -> float:
    """
    Benchmark preallocated cache approach.

    Preallocates buffers and uses position tracking.
    """
    times = []

    for _ in range(num_runs):
        # Preallocate buffers
        k_cache = mx.zeros((batch, num_kv_heads, max_seq_len, head_dim))
        v_cache = mx.zeros((batch, num_kv_heads, max_seq_len, head_dim))
        position = 0

        start = time.perf_counter()

        for step in range(num_tokens):
            # Simulate new K/V for this token
            new_k = mx.random.normal((batch, num_kv_heads, 1, head_dim))
            new_v = mx.random.normal((batch, num_kv_heads, 1, head_dim))

            # Update at position using slice assignment
            # Note: MLX uses copy-on-write, so this creates views efficiently
            k_slice = k_cache[:, :, :position + 1, :]
            v_slice = v_cache[:, :, :position + 1, :]

            # Concatenate to existing slice (still creates new array)
            if position == 0:
                k_slice = new_k
                v_slice = new_v
            else:
                old_k = k_cache[:, :, :position, :]
                old_v = v_cache[:, :, :position, :]
                k_slice = mx.concatenate([old_k, new_k], axis=2)
                v_slice = mx.concatenate([old_v, new_v], axis=2)

            # Force evaluation
            mx.eval(k_slice, v_slice)
            position += 1

        end = time.perf_counter()
        times.append(end - start)

    return sum(times) / len(times)


def benchmark_list_cache(
    batch: int = 1,
    num_kv_heads: int = 1,
    head_dim: int = 128,
    num_tokens: int = 100,
    num_runs: int = 5,
) -> float:
    """
    Benchmark list-based cache with final concatenation.

    Accumulates K/V in Python lists and concatenates only when needed.
    """
    times = []

    for _ in range(num_runs):
        k_list: List[mx.array] = []
        v_list: List[mx.array] = []

        start = time.perf_counter()

        for step in range(num_tokens):
            # Simulate new K/V for this token
            new_k = mx.random.normal((batch, num_kv_heads, 1, head_dim))
            new_v = mx.random.normal((batch, num_kv_heads, 1, head_dim))

            k_list.append(new_k)
            v_list.append(new_v)

            # Simulate attention: need full K/V
            k_cache = mx.concatenate(k_list, axis=2)
            v_cache = mx.concatenate(v_list, axis=2)

            # Force evaluation
            mx.eval(k_cache, v_cache)

        end = time.perf_counter()
        times.append(end - start)

    return sum(times) / len(times)


def benchmark_full_llm_simulation(
    batch: int = 1,
    num_layers: int = 24,
    num_kv_heads: int = 1,
    head_dim: int = 128,
    num_tokens: int = 100,
    num_runs: int = 3,
) -> Tuple[float, float]:
    """
    Benchmark simulating full LLM with 24 layers.

    Returns (dynamic_time, list_time).
    """
    # Dynamic approach
    times_dynamic = []
    for _ in range(num_runs):
        caches = [None] * num_layers

        start = time.perf_counter()
        for step in range(num_tokens):
            for layer in range(num_layers):
                new_k = mx.random.normal((batch, num_kv_heads, 1, head_dim))
                new_v = mx.random.normal((batch, num_kv_heads, 1, head_dim))

                if caches[layer] is None:
                    caches[layer] = (new_k, new_v)
                else:
                    k_cache, v_cache = caches[layer]
                    k_cache = mx.concatenate([k_cache, new_k], axis=2)
                    v_cache = mx.concatenate([v_cache, new_v], axis=2)
                    caches[layer] = (k_cache, v_cache)

            # Evaluate after each token (simulating actual generation)
            mx.eval(*[c for c in caches if c is not None for _ in c])

        times_dynamic.append(time.perf_counter() - start)

    avg_dynamic = sum(times_dynamic) / len(times_dynamic)

    # List approach with deferred concatenation
    times_list = []
    for _ in range(num_runs):
        k_lists = [[] for _ in range(num_layers)]
        v_lists = [[] for _ in range(num_layers)]

        start = time.perf_counter()
        for step in range(num_tokens):
            for layer in range(num_layers):
                new_k = mx.random.normal((batch, num_kv_heads, 1, head_dim))
                new_v = mx.random.normal((batch, num_kv_heads, 1, head_dim))

                k_lists[layer].append(new_k)
                v_lists[layer].append(new_v)

                # For attention: concatenate on demand
                k_cache = mx.concatenate(k_lists[layer], axis=2)
                v_cache = mx.concatenate(v_lists[layer], axis=2)

            # Evaluate after each token
            mx.eval(k_cache, v_cache)

        times_list.append(time.perf_counter() - start)

    avg_list = sum(times_list) / len(times_list)

    return avg_dynamic, avg_list


def benchmark_kvcache_optimized(
    batch: int = 1,
    num_layers: int = 24,
    num_kv_heads: int = 1,
    head_dim: int = 128,
    num_tokens: int = 100,
    num_runs: int = 3,
) -> float:
    """
    Benchmark D1 optimized KVCache with step-based preallocation.

    Uses the actual KVCache class from cosyvoice2_llm.
    """
    times = []

    for _ in range(num_runs):
        # Create KVCache for each layer
        caches = make_kv_cache(num_layers)

        start = time.perf_counter()
        for step in range(num_tokens):
            for layer in range(num_layers):
                new_k = mx.random.normal((batch, num_kv_heads, 1, head_dim))
                new_v = mx.random.normal((batch, num_kv_heads, 1, head_dim))

                # Use KVCache update_and_fetch (step-based preallocation)
                k, v = caches[layer].update_and_fetch(new_k, new_v)

            # Evaluate after each token
            mx.eval(k, v)

        times.append(time.perf_counter() - start)

    return sum(times) / len(times)


def benchmark_tuple_cache(
    batch: int = 1,
    num_layers: int = 24,
    num_kv_heads: int = 1,
    head_dim: int = 128,
    num_tokens: int = 100,
    num_runs: int = 3,
) -> float:
    """
    Benchmark original tuple-based cache (concatenation per token).

    This simulates the old approach before D1 optimization.
    """
    times = []

    for _ in range(num_runs):
        caches = [None] * num_layers

        start = time.perf_counter()
        for step in range(num_tokens):
            for layer in range(num_layers):
                new_k = mx.random.normal((batch, num_kv_heads, 1, head_dim))
                new_v = mx.random.normal((batch, num_kv_heads, 1, head_dim))

                if caches[layer] is None:
                    caches[layer] = (new_k, new_v)
                else:
                    k_cache, v_cache = caches[layer]
                    k = mx.concatenate([k_cache, new_k], axis=2)
                    v = mx.concatenate([v_cache, new_v], axis=2)
                    caches[layer] = (k, v)

            # Evaluate after each token
            mx.eval(*[c for c in caches if c is not None for _ in c])

        times.append(time.perf_counter() - start)

    return sum(times) / len(times)


def main():
    print("=" * 60)
    print("KV Cache Benchmark for CosyVoice3 LLM (D1 Optimization)")
    print("=" * 60)
    print()
    print("CosyVoice3 LLM Config:")
    print("  - num_layers: 24")
    print("  - num_kv_heads: 1 (GQA)")
    print("  - head_dim: 128")
    print("  - hidden_size: 896")
    print()

    # Warm up
    print("Warming up...")
    _ = mx.random.normal((1, 1, 1, 128))
    mx.eval(_)

    # Test KVCache import
    print("Testing KVCache import...")
    test_cache = KVCache()
    test_k = mx.random.normal((1, 1, 1, 128))
    test_v = mx.random.normal((1, 1, 1, 128))
    k, v = test_cache.update_and_fetch(test_k, test_v)
    mx.eval(k, v)
    print(f"KVCache test passed: offset={test_cache.offset}, k.shape={k.shape}")
    print()

    # D1 Benchmark: KVCache vs Tuple (100 tokens)
    print("=" * 60)
    print("D1 Optimization Benchmark (24 layers, 100 tokens)")
    print("=" * 60)

    tuple_time = benchmark_tuple_cache(num_tokens=100)
    print(f"Tuple cache (concatenation):    {tuple_time*1000:.2f}ms")

    kvcache_time = benchmark_kvcache_optimized(num_tokens=100)
    print(f"KVCache (D1 step-prealloc):     {kvcache_time*1000:.2f}ms")

    speedup_100 = tuple_time / kvcache_time
    print(f"Speedup: {speedup_100:.2f}x")
    print()

    # Longer sequences (500 tokens)
    print("=" * 60)
    print("Longer Sequences (24 layers, 500 tokens)")
    print("=" * 60)

    tuple_long = benchmark_tuple_cache(num_tokens=500, num_runs=2)
    print(f"Tuple cache (concatenation):    {tuple_long*1000:.2f}ms")

    kvcache_long = benchmark_kvcache_optimized(num_tokens=500, num_runs=2)
    print(f"KVCache (D1 step-prealloc):     {kvcache_long*1000:.2f}ms")

    speedup_500 = tuple_long / kvcache_long
    print(f"Speedup: {speedup_500:.2f}x")
    print()

    # Analysis
    print("=" * 60)
    print("Analysis")
    print("=" * 60)

    # Per-token overhead comparison
    per_token_tuple_100 = tuple_time / 100 * 1000  # ms
    per_token_kvcache_100 = kvcache_time / 100 * 1000  # ms
    per_token_tuple_500 = tuple_long / 500 * 1000  # ms
    per_token_kvcache_500 = kvcache_long / 500 * 1000  # ms

    print("Per-token overhead:")
    print(f"  Tuple (100 tokens):   {per_token_tuple_100:.3f}ms/token")
    print(f"  KVCache (100 tokens): {per_token_kvcache_100:.3f}ms/token")
    print(f"  Tuple (500 tokens):   {per_token_tuple_500:.3f}ms/token")
    print(f"  KVCache (500 tokens): {per_token_kvcache_500:.3f}ms/token")
    print()

    # Overhead scaling
    tuple_scaling = per_token_tuple_500 / per_token_tuple_100
    kvcache_scaling = per_token_kvcache_500 / per_token_kvcache_100

    print("Overhead scaling (100 -> 500 tokens):")
    print(f"  Tuple:   {tuple_scaling:.2f}x (O(n) concatenation)")
    print(f"  KVCache: {kvcache_scaling:.2f}x (O(1) step-based)")
    print()

    # Summary
    print("=" * 60)
    print("D1 Optimization Summary")
    print("=" * 60)
    print(f"Speedup at 100 tokens: {speedup_100:.2f}x")
    print(f"Speedup at 500 tokens: {speedup_500:.2f}x")
    print(f"KVCache overhead is {'constant' if kvcache_scaling < 1.3 else 'growing'} with sequence length")
    print(f"Tuple overhead grows {tuple_scaling:.2f}x from 100->500 tokens")
    print()

    # Estimate real-world impact
    # LLM forward pass is ~70% of total time
    # Cache operations are a fraction of LLM time
    cache_impact_estimate = (tuple_time - kvcache_time) * 0.1  # Rough estimate
    print("Estimated Real-World Impact:")
    print(f"  Cache overhead savings: {(tuple_time - kvcache_time)*1000:.0f}ms (100 tokens)")
    print("  Note: Cache ops are ~10% of LLM forward pass time")
    print(f"  Expected E2E speedup: ~{cache_impact_estimate/tuple_time*100:.0f}% for LLM component")


if __name__ == "__main__":
    main()
