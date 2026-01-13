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
CosyVoice3 LLM Layer-by-Layer Profiler

Profiles each component of the forward pass to find the slowest parts.
"""

import sys
from pathlib import Path
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx


def profile_layers():
    """Profile each layer of the LLM."""
    print("=" * 70)
    print("CosyVoice3 LLM Layer-by-Layer Profiler")
    print("=" * 70)

    # Load models
    print("\n1. Loading LLM...")

    from tools.pytorch_to_mlx.converters.models.cosyvoice2_llm import (
        CosyVoice2LLM, Qwen2Config, make_kv_cache, KVCache
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

    print("   LLM loaded")

    # ==========================================================================
    # Profile single-token forward pass in detail
    # ==========================================================================
    print("\n2. Profiling single-token forward pass components...")

    batch = 1
    hidden_size = 896
    seq_len = 1

    # Create dummy hidden states
    hidden_states = mx.random.normal((batch, seq_len, hidden_size))
    mx.eval(hidden_states)

    # Create cache with some content (simulate generation in progress)
    cache = make_kv_cache(24)

    # Simulate having processed a prompt first
    # Process 10 tokens to populate cache
    dummy_prompt = mx.zeros((batch, 10), dtype=mx.int32)
    _, _, cache = llm(dummy_prompt, cache=cache)
    mx.eval([c.keys for c in cache if c.keys is not None])

    print(f"   Cache populated with {cache[0].offset} tokens")

    # Now profile a single-token forward pass

    # Profile embedding lookup
    single_token = mx.zeros((batch, 1), dtype=mx.int32)
    times = []
    for _ in range(10):
        start = time.time()
        emb = llm.llm.embed_tokens(single_token)
        mx.eval(emb)
        times.append(time.time() - start)
    print(f"   Embedding lookup: {np.mean(times[3:])*1000:.2f}ms")

    # Profile single transformer layer
    layer = llm.llm.layers[0]
    hidden = mx.random.normal((batch, seq_len, hidden_size))
    layer_cache = KVCache()
    # Populate layer cache
    _, layer_cache = layer(mx.random.normal((batch, 10, hidden_size)), cache=layer_cache)
    mx.eval(hidden, layer_cache.keys, layer_cache.values)

    times = []
    for _ in range(10):
        start = time.time()
        out, _ = layer(hidden, cache=layer_cache)
        mx.eval(out)
        times.append(time.time() - start)
    print(f"   Single layer (cached): {np.mean(times[3:])*1000:.2f}ms")

    # Profile attention only
    attn = layer.self_attn
    norm_hidden = layer.input_layernorm(hidden)
    mx.eval(norm_hidden)

    times = []
    for _ in range(10):
        start = time.time()
        out, _ = attn(norm_hidden, cache=layer_cache)
        mx.eval(out)
        times.append(time.time() - start)
    print(f"   Attention only (cached): {np.mean(times[3:])*1000:.2f}ms")

    # Profile MLP only
    post_norm = layer.post_attention_layernorm(hidden)
    mx.eval(post_norm)

    times = []
    for _ in range(10):
        start = time.time()
        out = layer.mlp(post_norm)
        mx.eval(out)
        times.append(time.time() - start)
    print(f"   MLP only: {np.mean(times[3:])*1000:.2f}ms")

    # Profile RMSNorm
    times = []
    for _ in range(10):
        start = time.time()
        out = layer.input_layernorm(hidden)
        mx.eval(out)
        times.append(time.time() - start)
    print(f"   RMSNorm: {np.mean(times[3:])*1000:.2f}ms")

    # Profile LM head
    final_hidden = mx.random.normal((batch, seq_len, hidden_size))
    mx.eval(final_hidden)

    times = []
    for _ in range(10):
        start = time.time()
        logits = llm.llm_decoder(final_hidden)
        mx.eval(logits)
        times.append(time.time() - start)
    print(f"   LLM decoder head: {np.mean(times[3:])*1000:.2f}ms")

    # ==========================================================================
    # Profile all 24 layers together
    # ==========================================================================
    print("\n3. Profiling all layers...")

    # Fresh cache
    cache = make_kv_cache(24)

    # Populate with 10 tokens
    dummy_prompt = mx.zeros((batch, 10), dtype=mx.int32)
    _, _, cache = llm(dummy_prompt, cache=cache)
    mx.eval([c.keys for c in cache if c.keys is not None])

    single_token = mx.zeros((batch, 1), dtype=mx.int32)

    # Profile full forward
    times = []
    for _ in range(10):
        start = time.time()
        _, speech_logits, cache = llm(single_token, cache=cache)
        mx.eval(speech_logits)
        times.append(time.time() - start)

    avg = np.mean(times[3:])
    print(f"   Full forward (24 layers): {avg*1000:.2f}ms ({1/avg:.1f} tok/sec)")

    # ==========================================================================
    # Compare to direct matmul equivalent
    # ==========================================================================
    print("\n4. Comparing to raw matmul...")

    # Roughly, each layer has:
    # - Q,K,V projections: 3 matmuls of (896, 896), (896, 128), (896, 128)
    # - O projection: (896, 896)
    # - Gate, Up, Down MLP: (896, 4864), (896, 4864), (4864, 896)
    # = 7 matmuls per layer, 24 layers = 168 matmuls

    # Simulate just the matmuls
    x = mx.random.normal((1, 1, 896))
    w_small = mx.random.normal((896, 128))
    w_large = mx.random.normal((896, 896))
    w_mlp = mx.random.normal((896, 4864))
    mx.eval(x, w_small, w_large, w_mlp)

    times = []
    for _ in range(10):
        start = time.time()
        # Simulate 24 layers of matmuls
        for _ in range(24):
            # Q, K, V
            q = mx.matmul(x, w_large.T)
            k = mx.matmul(x, w_small.T)
            v = mx.matmul(x, w_small.T)
            # O proj
            o = mx.matmul(x, w_large.T)
            # MLP (simplified)
            g = mx.matmul(x, w_mlp.T)
            u = mx.matmul(x, w_mlp.T)
        mx.eval(q, k, v, o, g, u)
        times.append(time.time() - start)

    avg_matmul = np.mean(times[3:])
    print(f"   Raw matmuls (24 layers, no attention): {avg_matmul*1000:.2f}ms")

    # ==========================================================================
    # Profile KV cache operations
    # ==========================================================================
    print("\n5. Profiling KV cache operations...")

    cache = KVCache()

    # Initial fill
    k = mx.random.normal((1, 1, 10, 128))  # 10 tokens
    v = mx.random.normal((1, 1, 10, 128))
    mx.eval(k, v)

    times = []
    for _ in range(10):
        cache = KVCache()
        start = time.time()
        cache.update_and_fetch(k, v)
        mx.eval(cache.keys, cache.values)
        times.append(time.time() - start)
    print(f"   KV cache initial fill (10 tokens): {np.mean(times[3:])*1000:.2f}ms")

    # Single token update
    cache = KVCache()
    cache.update_and_fetch(k, v)
    mx.eval(cache.keys, cache.values)

    k_new = mx.random.normal((1, 1, 1, 128))
    v_new = mx.random.normal((1, 1, 1, 128))
    mx.eval(k_new, v_new)

    times = []
    for _ in range(10):
        start = time.time()
        cache.update_and_fetch(k_new, v_new)
        mx.eval(cache.keys, cache.values)
        times.append(time.time() - start)
    print(f"   KV cache update (1 token): {np.mean(times[3:])*1000:.2f}ms")

    # ==========================================================================
    # Profile SDPA
    # ==========================================================================
    print("\n6. Profiling scaled_dot_product_attention...")
    import mlx.core.fast as fast

    # Single query attending to 100 cached K,V
    q = mx.random.normal((1, 7, 1, 128))  # 7 heads, 1 token
    k = mx.random.normal((1, 1, 100, 128))  # 1 KV head, 100 cached
    v = mx.random.normal((1, 1, 100, 128))
    mx.eval(q, k, v)

    times = []
    for _ in range(10):
        start = time.time()
        out = fast.scaled_dot_product_attention(q, k, v, scale=128**-0.5)
        mx.eval(out)
        times.append(time.time() - start)
    print(f"   SDPA (7 Q heads, 1 KV head, 100 cache): {np.mean(times[3:])*1000:.2f}ms")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nSingle token generation:")
    print(f"  Full forward pass: {avg*1000:.1f}ms")
    print(f"  Theoretical max (raw matmuls): {avg_matmul*1000:.1f}ms")
    print(f"  Overhead: {(avg - avg_matmul)/avg_matmul*100:.0f}%")

    if avg > 0.1:  # > 100ms per token
        print("\n[CRITICAL] Forward pass is extremely slow (>100ms)")
        print("Expected: ~10-30ms for model this size on Apple Silicon")
        print("Possible causes:")
        print("  1. Model weights not on GPU (check mx.default_device())")
        print("  2. Excessive graph rebuilding (check compilation)")
        print("  3. Memory pressure (check GPU memory)")


if __name__ == "__main__":
    print(f"MLX default device: {mx.default_device()}")
    profile_layers()
