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
CosyVoice3 LLM Profiler

Profiles the LLM generation loop to identify specific bottlenecks.
Tests:
1. Forward pass time
2. Sampling time (compiled vs ras_sampling)
3. mx.eval() overhead
4. .item() sync overhead
"""

import sys
from pathlib import Path
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx


def profile_llm_forward():
    """Profile the LLM forward pass alone."""
    print("=" * 70)
    print("CosyVoice3 LLM Profiler")
    print("=" * 70)

    # Load models
    print("\n1. Loading LLM...")
    load_start = time.time()

    from tools.pytorch_to_mlx.converters.models.cosyvoice2_llm import (
        CosyVoice2LLM, Qwen2Config, make_kv_cache,
        _compiled_sample_no_rep_penalty, ras_sampling
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

    print(f"   LLM loaded in {time.time() - load_start:.2f}s")

    # ==========================================================================
    # Test 1: Raw forward pass latency
    # ==========================================================================
    print("\n2. Profiling forward pass...")

    batch = 1
    seq_len = 10

    # Create dummy inputs
    input_ids = mx.zeros((batch, seq_len), dtype=mx.int32)
    cache = make_kv_cache(24)

    # Warmup
    for _ in range(3):
        _, speech_logits, cache = llm(input_ids, cache=cache)
        mx.eval(speech_logits)

    # Reset cache
    cache = make_kv_cache(24)

    # Profile prompt encoding
    start = time.time()
    _, speech_logits, cache = llm(input_ids, cache=cache)
    mx.eval(speech_logits)
    prompt_time = time.time() - start
    print(f"   Prompt encoding ({seq_len} tokens): {prompt_time*1000:.1f}ms")

    # Profile single token generation (autoregressive)
    single_token = mx.zeros((batch, 1), dtype=mx.int32)

    times = []
    for _ in range(20):
        start = time.time()
        _, speech_logits, cache = llm(single_token, cache=cache)
        mx.eval(speech_logits)
        times.append(time.time() - start)

    avg_time = np.mean(times[5:])  # Skip first 5 for warmup
    print(f"   Single token forward: {avg_time*1000:.1f}ms ({1/avg_time:.1f} tokens/sec)")

    # ==========================================================================
    # Test 2: Sampling latency comparison
    # ==========================================================================
    print("\n3. Profiling sampling methods...")

    # Create dummy logits
    logits = mx.random.normal((batch, 6564))
    mx.eval(logits)

    # Test compiled sampling
    times_compiled = []
    for _ in range(50):
        start = time.time()
        token = _compiled_sample_no_rep_penalty(logits, 0.8, 25, 0.8)
        mx.eval(token)
        times_compiled.append(time.time() - start)

    avg_compiled = np.mean(times_compiled[10:])
    print(f"   Compiled sampling: {avg_compiled*1000:.2f}ms")

    # Test RAS sampling (the slow one)
    decoded_tokens = list(range(100))  # Simulate 100 past tokens
    times_ras = []
    for _ in range(20):
        start = time.time()
        token = ras_sampling(logits[0], decoded_tokens, top_p=0.8, top_k=25)
        times_ras.append(time.time() - start)

    avg_ras = np.mean(times_ras[5:])
    print(f"   RAS sampling (Python): {avg_ras*1000:.2f}ms")
    print(f"   RAS vs Compiled: {avg_ras/avg_compiled:.1f}x SLOWER")

    # ==========================================================================
    # Test 3: mx.eval() overhead
    # ==========================================================================
    print("\n4. Profiling mx.eval() overhead...")

    # Multiple ops without eval
    start = time.time()
    for _ in range(100):
        x = mx.random.normal((1, 896))
        y = mx.matmul(x, mx.random.normal((896, 896)))
    mx.eval(y)
    batched_time = time.time() - start

    # Multiple ops with eval each time
    start = time.time()
    for _ in range(100):
        x = mx.random.normal((1, 896))
        y = mx.matmul(x, mx.random.normal((896, 896)))
        mx.eval(y)
    sync_time = time.time() - start

    print(f"   100 matmuls batched: {batched_time*1000:.1f}ms")
    print(f"   100 matmuls with per-op eval: {sync_time*1000:.1f}ms")
    print(f"   Overhead: {sync_time/batched_time:.1f}x")

    # ==========================================================================
    # Test 4: .item() overhead
    # ==========================================================================
    print("\n5. Profiling .item() overhead...")

    # Create array
    arr = mx.random.normal((1000,))
    mx.eval(arr)

    # Test without .item()
    start = time.time()
    total = mx.sum(arr)
    mx.eval(total)
    no_item_time = time.time() - start

    # Test with .item() in loop
    start = time.time()
    total = 0
    for i in range(100):
        total += arr[i].item()
    item_loop_time = time.time() - start

    print(f"   mx.sum() + eval: {no_item_time*1000:.2f}ms")
    print(f"   100x .item() loop: {item_loop_time*1000:.1f}ms")
    print(f"   Overhead: {item_loop_time/no_item_time:.1f}x")

    # ==========================================================================
    # Test 5: Full generation comparison
    # ==========================================================================
    print("\n6. Full generation comparison...")

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "models/cosyvoice3/CosyVoice-BlankEN",
        trust_remote_code=True
    )

    text = "Hello world."
    tokens = tokenizer(text, return_tensors="pt")
    input_ids = mx.array(tokens["input_ids"].numpy())

    # Standard generation (with per-token eval)
    cache = make_kv_cache(24)
    start = time.time()
    speech_tokens_std = llm.generate_speech_tokens(
        input_ids, max_length=50, temperature=0
    )
    mx.eval(speech_tokens_std)
    std_time = time.time() - start
    std_tokens = speech_tokens_std.shape[1]

    print("\n   Standard generate_speech_tokens:")
    print(f"     Time: {std_time:.2f}s for {std_tokens} tokens")
    print(f"     Rate: {std_tokens/std_time:.1f} tokens/sec")

    # Test RAS generation
    cache = make_kv_cache(24)
    start = time.time()
    speech_tokens_ras = llm.generate_speech_tokens_ras(
        input_ids, max_length=50
    )
    mx.eval(speech_tokens_ras)
    ras_time = time.time() - start
    ras_tokens = speech_tokens_ras.shape[1]

    print("\n   RAS generate_speech_tokens_ras:")
    print(f"     Time: {ras_time:.2f}s for {ras_tokens} tokens")
    print(f"     Rate: {ras_tokens/ras_time:.1f} tokens/sec")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nForward pass: {avg_time*1000:.1f}ms ({1/avg_time:.0f} tok/sec theoretical)")
    print(f"Compiled sampling: {avg_compiled*1000:.2f}ms")
    print(f"RAS sampling: {avg_ras*1000:.2f}ms ({avg_ras/avg_compiled:.0f}x slower)")
    print(f"\nActual generation: {std_tokens/std_time:.1f} tok/sec (std) / {ras_tokens/ras_time:.1f} tok/sec (ras)")

    # Calculate theoretical max
    theoretical_max = 1 / avg_time
    actual = std_tokens / std_time
    efficiency = (actual / theoretical_max) * 100

    print(f"\nEfficiency: {efficiency:.1f}% of theoretical max")
    print(f"Gap: {theoretical_max - actual:.1f} tok/sec lost to overhead")

    print("\n\nBOTTLENECKS:")
    if avg_ras / avg_compiled > 10:
        print("  1. [CRITICAL] RAS sampling is extremely slow due to Python loops + .item()")
    if sync_time / batched_time > 2:
        print("  2. [MAJOR] Per-token mx.eval() adds significant overhead")
    print("  3. [INVESTIGATE] Compare to mlx-lm generate() for best practices")


if __name__ == "__main__":
    profile_llm_forward()
