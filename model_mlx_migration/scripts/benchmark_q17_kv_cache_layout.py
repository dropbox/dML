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
Q17 Benchmark: Contiguous vs Per-Layer KV Cache Layout

Compares:
1. Per-layer KVCache (current implementation, mlx-lm pattern)
2. Contiguous KV Cache (all layers in single tensor)

For Qwen2/CosyVoice2 LLM with GQA (1 KV head, 128 head_dim).
"""

import time
from typing import Tuple

import mlx.core as mx


# Configuration matching CosyVoice2 LLM
NUM_LAYERS = 24
N_KV_HEADS = 1  # GQA: only 1 KV head
HEAD_DIM = 128
BATCH_SIZE = 1
STEP = 256  # Preallocation step


class PerLayerKVCache:
    """Current implementation - per-layer KVCache (D1 optimized)."""

    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> Tuple[mx.array, mx.array]:
        prev = self.offset
        num_new = keys.shape[2]

        if self.keys is None or (prev + num_new) > self.keys.shape[2]:
            B, n_kv_heads, _, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            n_steps = (STEP + num_new - 1) // STEP
            k_shape = (B, n_kv_heads, n_steps * STEP, k_head_dim)
            v_shape = (B, n_kv_heads, n_steps * STEP, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                if prev % STEP != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self.offset += num_new
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]


class ContiguousKVCache:
    """
    Q17: Contiguous KV Cache Layout.

    All layers share a single contiguous allocation:
    [num_layers, 2, max_seq, num_heads, head_dim]
    where cache[:, 0, ...] = K, cache[:, 1, ...] = V
    """

    def __init__(self, num_layers: int, batch_size: int, n_kv_heads: int, head_dim: int):
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.cache = None  # [num_layers, 2, batch, n_kv_heads, max_seq, head_dim]
        self.offset = 0

    def update_and_fetch(
        self, layer_idx: int, keys: mx.array, values: mx.array
    ) -> Tuple[mx.array, mx.array]:
        """Update cache for a specific layer and return full cache."""
        B, n_kv_heads, num_new, head_dim = keys.shape
        prev = self.offset if layer_idx == 0 else self.offset  # All layers share offset

        # Initialize or expand cache
        if self.cache is None or (prev + num_new) > self.cache.shape[4]:
            n_steps = (STEP + num_new - 1) // STEP
            new_shape = (
                self.num_layers,
                2,  # K and V
                B,
                n_kv_heads,
                n_steps * STEP,
                head_dim,
            )
            new_cache = mx.zeros(new_shape, keys.dtype)
            if self.cache is not None:
                if prev % STEP != 0:
                    self.cache = self.cache[..., :prev, :]
                self.cache = mx.concatenate([self.cache, new_cache], axis=4)
            else:
                self.cache = new_cache

        # Update only for layer 0 (others use same offset)
        if layer_idx == 0:
            self.offset += num_new

        # In-place update
        self.cache[layer_idx, 0, :, :, prev : prev + num_new, :] = keys
        self.cache[layer_idx, 1, :, :, prev : prev + num_new, :] = values

        # Return K, V for this layer
        k = self.cache[layer_idx, 0, :, :, : prev + num_new, :]
        v = self.cache[layer_idx, 1, :, :, : prev + num_new, :]
        return k, v


def benchmark_per_layer(num_tokens: int, warmup_iters: int = 5, bench_iters: int = 20):
    """Benchmark per-layer KVCache (current implementation)."""
    caches = [PerLayerKVCache() for _ in range(NUM_LAYERS)]

    # Warmup
    for _ in range(warmup_iters):
        for i in range(num_tokens):
            k = mx.random.normal((BATCH_SIZE, N_KV_HEADS, 1, HEAD_DIM))
            v = mx.random.normal((BATCH_SIZE, N_KV_HEADS, 1, HEAD_DIM))
            for layer_idx, cache in enumerate(caches):
                cache.update_and_fetch(k, v)
            mx.eval([c.keys for c in caches])
        # Reset
        caches = [PerLayerKVCache() for _ in range(NUM_LAYERS)]

    # Benchmark
    times = []
    for _ in range(bench_iters):
        caches = [PerLayerKVCache() for _ in range(NUM_LAYERS)]
        start = time.perf_counter()
        for i in range(num_tokens):
            k = mx.random.normal((BATCH_SIZE, N_KV_HEADS, 1, HEAD_DIM))
            v = mx.random.normal((BATCH_SIZE, N_KV_HEADS, 1, HEAD_DIM))
            for layer_idx, cache in enumerate(caches):
                k_full, v_full = cache.update_and_fetch(k, v)
            # Force evaluation
            mx.eval(k_full, v_full)
        times.append(time.perf_counter() - start)

    return times


def benchmark_contiguous(num_tokens: int, warmup_iters: int = 5, bench_iters: int = 20):
    """Benchmark contiguous KVCache (Q17 proposal)."""
    cache = ContiguousKVCache(NUM_LAYERS, BATCH_SIZE, N_KV_HEADS, HEAD_DIM)

    # Warmup
    for _ in range(warmup_iters):
        cache = ContiguousKVCache(NUM_LAYERS, BATCH_SIZE, N_KV_HEADS, HEAD_DIM)
        for i in range(num_tokens):
            k = mx.random.normal((BATCH_SIZE, N_KV_HEADS, 1, HEAD_DIM))
            v = mx.random.normal((BATCH_SIZE, N_KV_HEADS, 1, HEAD_DIM))
            for layer_idx in range(NUM_LAYERS):
                cache.update_and_fetch(layer_idx, k, v)
            mx.eval(cache.cache)

    # Benchmark
    times = []
    for _ in range(bench_iters):
        cache = ContiguousKVCache(NUM_LAYERS, BATCH_SIZE, N_KV_HEADS, HEAD_DIM)
        start = time.perf_counter()
        for i in range(num_tokens):
            k = mx.random.normal((BATCH_SIZE, N_KV_HEADS, 1, HEAD_DIM))
            v = mx.random.normal((BATCH_SIZE, N_KV_HEADS, 1, HEAD_DIM))
            for layer_idx in range(NUM_LAYERS):
                k_full, v_full = cache.update_and_fetch(layer_idx, k, v)
            # Force evaluation
            mx.eval(k_full, v_full)
        times.append(time.perf_counter() - start)

    return times


def main():
    print("=" * 70)
    print("Q17 Benchmark: Contiguous vs Per-Layer KV Cache Layout")
    print("=" * 70)
    print("Configuration:")
    print(f"  Layers: {NUM_LAYERS}")
    print(f"  KV Heads: {N_KV_HEADS} (GQA)")
    print(f"  Head Dim: {HEAD_DIM}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Prealloc Step: {STEP}")
    print()

    for num_tokens in [50, 100, 200]:
        print(f"--- {num_tokens} tokens (autoregressive, seq_len=1 per step) ---")

        per_layer_times = benchmark_per_layer(num_tokens)
        contiguous_times = benchmark_contiguous(num_tokens)

        per_layer_mean = sum(per_layer_times) / len(per_layer_times) * 1000
        contiguous_mean = sum(contiguous_times) / len(contiguous_times) * 1000

        per_layer_per_token = per_layer_mean / num_tokens
        contiguous_per_token = contiguous_mean / num_tokens

        speedup = per_layer_mean / contiguous_mean

        print(f"Per-Layer KVCache:   {per_layer_mean:.2f}ms total, {per_layer_per_token:.3f}ms/token")
        print(f"Contiguous KVCache:  {contiguous_mean:.2f}ms total, {contiguous_per_token:.3f}ms/token")
        print(f"Speedup:             {speedup:.2f}x {'(contiguous faster)' if speedup > 1 else '(per-layer faster)'}")
        print()

    print("=" * 70)
    print("ANALYSIS:")
    print("=" * 70)
    print("The Q17 'Contiguous KV Cache Layout' optimization proposes storing")
    print("all layers' K/V in a single tensor vs separate per-layer tensors.")
    print()
    print("Key observations:")
    print("1. mlx-lm (Apple's official library) uses per-layer approach")
    print("2. D1 optimization already provides O(1) updates via step-based prealloc")
    print("3. GQA with 1 KV head makes cache very small (128 elements/token)")
    print("4. MLX's lazy evaluation may already optimize allocation patterns")
    print()
    if speedup < 1.05:
        print("VERDICT: Q17 is NOT WORTH implementing.")
        print("The contiguous layout provides no significant benefit over per-layer.")
    elif speedup > 1.1:
        print("VERDICT: Q17 MAY BE WORTH implementing.")
        print(f"Contiguous layout shows {speedup:.2f}x improvement.")
    else:
        print("VERDICT: Q17 provides marginal benefit (< 10%).")
        print("Consider implementation complexity vs small gain.")


if __name__ == "__main__":
    main()
