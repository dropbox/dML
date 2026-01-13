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
CosyVoice3 Attention Detailed Profiler

Deep dive into why attention is taking 10ms per layer.
For reference, attention should be ~1-2ms on Apple Silicon.
"""

import sys
from pathlib import Path
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import mlx.core.fast as fast
import mlx.nn as nn


def profile_attention():
    """Profile attention components in detail."""
    print("=" * 70)
    print("CosyVoice3 Attention Detailed Profiler")
    print("=" * 70)

    # Model parameters
    batch = 1
    seq_len = 1  # Single token for autoregressive
    hidden_size = 896
    num_heads = 7  # Query heads
    num_kv_heads = 1  # KV heads (GQA)
    head_dim = 128
    cache_len = 100  # Simulating 100 cached tokens

    print("\nModel config:")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Query heads: {num_heads}")
    print(f"  KV heads: {num_kv_heads}")
    print(f"  Head dim: {head_dim}")
    print(f"  Cache length: {cache_len}")

    # ==========================================================================
    # Test 1: Raw SDPA performance
    # ==========================================================================
    print("\n1. Testing raw SDPA...")

    q = mx.random.normal((batch, num_heads, seq_len, head_dim))
    k = mx.random.normal((batch, num_kv_heads, cache_len, head_dim))
    v = mx.random.normal((batch, num_kv_heads, cache_len, head_dim))
    mx.eval(q, k, v)

    # Warmup
    for _ in range(5):
        out = fast.scaled_dot_product_attention(q, k, v, scale=head_dim**-0.5)
        mx.eval(out)

    times = []
    for _ in range(20):
        start = time.time()
        out = fast.scaled_dot_product_attention(q, k, v, scale=head_dim**-0.5)
        mx.eval(out)
        times.append(time.time() - start)

    avg = np.mean(times[5:])
    print(f"   SDPA only: {avg*1000:.2f}ms")

    # ==========================================================================
    # Test 2: Q,K,V projections
    # ==========================================================================
    print("\n2. Testing Q,K,V projections...")

    hidden = mx.random.normal((batch, seq_len, hidden_size))
    w_q = mx.random.normal((num_heads * head_dim, hidden_size))  # 896, 896
    w_k = mx.random.normal((num_kv_heads * head_dim, hidden_size))  # 128, 896
    w_v = mx.random.normal((num_kv_heads * head_dim, hidden_size))  # 128, 896
    b_q = mx.random.normal((num_heads * head_dim,))
    b_k = mx.random.normal((num_kv_heads * head_dim,))
    b_v = mx.random.normal((num_kv_heads * head_dim,))
    mx.eval(hidden, w_q, w_k, w_v, b_q, b_k, b_v)

    # Warmup
    for _ in range(5):
        q = mx.matmul(hidden, w_q.T) + b_q
        k = mx.matmul(hidden, w_k.T) + b_k
        v = mx.matmul(hidden, w_v.T) + b_v
        mx.eval(q, k, v)

    times = []
    for _ in range(20):
        start = time.time()
        q = mx.matmul(hidden, w_q.T) + b_q
        k = mx.matmul(hidden, w_k.T) + b_k
        v = mx.matmul(hidden, w_v.T) + b_v
        mx.eval(q, k, v)
        times.append(time.time() - start)

    avg = np.mean(times[5:])
    print(f"   Q,K,V projections: {avg*1000:.2f}ms")

    # ==========================================================================
    # Test 3: Output projection
    # ==========================================================================
    print("\n3. Testing output projection...")

    attn_out = mx.random.normal((batch, seq_len, num_heads * head_dim))
    w_o = mx.random.normal((hidden_size, num_heads * head_dim))
    mx.eval(attn_out, w_o)

    times = []
    for _ in range(20):
        start = time.time()
        out = mx.matmul(attn_out, w_o.T)
        mx.eval(out)
        times.append(time.time() - start)

    avg = np.mean(times[5:])
    print(f"   Output projection: {avg*1000:.2f}ms")

    # ==========================================================================
    # Test 4: RoPE
    # ==========================================================================
    print("\n4. Testing RoPE...")

    rope = nn.RoPE(dims=head_dim, base=1000000.0)
    q = mx.random.normal((batch, num_heads, seq_len, head_dim))
    mx.eval(q)

    times = []
    for _ in range(20):
        start = time.time()
        q_rot = rope(q, offset=cache_len)
        mx.eval(q_rot)
        times.append(time.time() - start)

    avg = np.mean(times[5:])
    print(f"   RoPE: {avg*1000:.2f}ms")

    # ==========================================================================
    # Test 5: KV Cache update
    # ==========================================================================
    print("\n5. Testing KV cache update...")

    from tools.pytorch_to_mlx.converters.models.cosyvoice2_llm import KVCache

    # Initialize cache with content
    cache = KVCache()
    initial_k = mx.random.normal((batch, num_kv_heads, cache_len, head_dim))
    initial_v = mx.random.normal((batch, num_kv_heads, cache_len, head_dim))
    cache.update_and_fetch(initial_k, initial_v)
    mx.eval(cache.keys, cache.values)

    new_k = mx.random.normal((batch, num_kv_heads, seq_len, head_dim))
    new_v = mx.random.normal((batch, num_kv_heads, seq_len, head_dim))
    mx.eval(new_k, new_v)

    times = []
    for _ in range(20):
        # Reset offset to simulate continuous update
        cache.offset = cache_len
        start = time.time()
        k_full, v_full = cache.update_and_fetch(new_k, new_v)
        mx.eval(k_full, v_full)
        times.append(time.time() - start)

    avg = np.mean(times[5:])
    print(f"   KV cache update: {avg*1000:.2f}ms")

    # ==========================================================================
    # Test 6: Complete attention flow
    # ==========================================================================
    print("\n6. Testing complete attention flow...")

    # Simulate full attention: hidden -> Q,K,V proj -> reshape -> RoPE -> cache update -> SDPA -> reshape -> O proj
    hidden = mx.random.normal((batch, seq_len, hidden_size))
    cache = KVCache()
    initial_k = mx.random.normal((batch, num_kv_heads, cache_len, head_dim))
    initial_v = mx.random.normal((batch, num_kv_heads, cache_len, head_dim))
    cache.update_and_fetch(initial_k, initial_v)
    mx.eval(hidden, cache.keys, cache.values)

    times = []
    for _ in range(20):
        cache.offset = cache_len  # Reset for consistent testing
        start = time.time()

        # Q,K,V projection
        q = mx.matmul(hidden, w_q.T) + b_q
        k = mx.matmul(hidden, w_k.T) + b_k
        v = mx.matmul(hidden, w_v.T) + b_v

        # Reshape
        q = q.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, seq_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, seq_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)

        # RoPE
        q = rope(q, offset=cache_len)
        k = rope(k, offset=cache_len)

        # Cache update
        k, v = cache.update_and_fetch(k, v)

        # SDPA
        out = fast.scaled_dot_product_attention(q, k, v, scale=head_dim**-0.5)

        # Reshape and output projection
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
        out = mx.matmul(out, w_o.T)

        mx.eval(out)
        times.append(time.time() - start)

    avg = np.mean(times[5:])
    print(f"   Full attention flow: {avg*1000:.2f}ms")

    # ==========================================================================
    # Test 7: Compare to actual model attention
    # ==========================================================================
    print("\n7. Comparing to actual model attention...")

    from tools.pytorch_to_mlx.converters.models.cosyvoice2_llm import (
        CosyVoice2LLM, Qwen2Config
    )

    llm_config = Qwen2Config(
        hidden_size=896,
        num_hidden_layers=24,
        num_attention_heads=7,
        num_key_value_heads=1,
        head_dim=128,
        intermediate_size=4864,
        vocab_size=151936,
        speech_vocab_size=6564,
        rope_theta=1000000.0,
    )
    llm = CosyVoice2LLM(llm_config)

    # Load weights
    weights = mx.load('models/cosyvoice3_mlx/model.safetensors')
    llm_weights = {k[4:]: v for k, v in weights.items() if k.startswith('llm.')}
    llm.load_weights(list(llm_weights.items()))
    mx.eval(llm.parameters())

    # Get first layer's attention
    attn = llm.llm.layers[0].self_attn

    # Create cache
    layer_cache = KVCache()
    initial_k = mx.random.normal((batch, num_kv_heads, cache_len, head_dim))
    initial_v = mx.random.normal((batch, num_kv_heads, cache_len, head_dim))
    layer_cache.update_and_fetch(initial_k, initial_v)
    mx.eval(layer_cache.keys, layer_cache.values)

    hidden = mx.random.normal((batch, seq_len, hidden_size))
    mx.eval(hidden)

    times = []
    for _ in range(20):
        layer_cache.offset = cache_len  # Reset
        start = time.time()
        out, _ = attn(hidden, cache=layer_cache)
        mx.eval(out)
        times.append(time.time() - start)

    avg_model = np.mean(times[5:])
    print(f"   Model attention: {avg_model*1000:.2f}ms")

    # ==========================================================================
    # Test 8: Check for mask overhead
    # ==========================================================================
    print("\n8. Testing mask overhead...")

    q = mx.random.normal((batch, num_heads, seq_len, head_dim))
    k = mx.random.normal((batch, num_kv_heads, cache_len + seq_len, head_dim))
    v = mx.random.normal((batch, num_kv_heads, cache_len + seq_len, head_dim))
    mx.eval(q, k, v)

    # Without mask
    times_no_mask = []
    for _ in range(20):
        start = time.time()
        out = fast.scaled_dot_product_attention(q, k, v, scale=head_dim**-0.5)
        mx.eval(out)
        times_no_mask.append(time.time() - start)

    # With causal mask
    times_causal = []
    for _ in range(20):
        start = time.time()
        out = fast.scaled_dot_product_attention(q, k, v, scale=head_dim**-0.5, mask="causal")
        mx.eval(out)
        times_causal.append(time.time() - start)

    # With explicit mask
    mask = mx.zeros((seq_len, cache_len + seq_len))
    mask = mask[None, None, :, :]
    mx.eval(mask)

    times_explicit = []
    for _ in range(20):
        start = time.time()
        out = fast.scaled_dot_product_attention(q, k, v, scale=head_dim**-0.5, mask=mask)
        mx.eval(out)
        times_explicit.append(time.time() - start)

    print(f"   SDPA no mask: {np.mean(times_no_mask[5:])*1000:.2f}ms")
    print(f"   SDPA causal mask: {np.mean(times_causal[5:])*1000:.2f}ms")
    print(f"   SDPA explicit mask: {np.mean(times_explicit[5:])*1000:.2f}ms")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nExpected vs Actual:")
    print("  SDPA expected: ~0.5-1ms")
    print("  Full attention expected: ~2-3ms")
    print(f"  Model attention actual: {avg_model*1000:.2f}ms")

    if avg_model > 5:
        print(f"\n[WARNING] Attention is {avg_model*1000/3:.1f}x slower than expected")


if __name__ == "__main__":
    profile_attention()
