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
Test compiled generation for CosyVoice3 LLM

Key optimizations from mlx-lm:
1. Compile the step function (forward + sampling)
2. Batch mx.eval() calls
3. Stream tokens with deferred evaluation
"""

import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx


def test_compiled_generation():
    print("=" * 70)
    print("Testing Compiled Generation for CosyVoice3 LLM")
    print("=" * 70)

    # Load model
    print("\n1. Loading model...")
    from tools.pytorch_to_mlx.converters.models.cosyvoice2_llm import (
        CosyVoice2LLM, Qwen2Config, make_kv_cache
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
    print("   Model loaded")

    # ==========================================================================
    # Test 1: Original generation (per-token eval)
    # ==========================================================================
    print("\n2. Testing original generation (per-token eval)...")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "models/cosyvoice3/CosyVoice-BlankEN",
        trust_remote_code=True
    )

    text = "Hello world."
    tokens = tokenizer(text, return_tensors="pt")
    input_ids = mx.array(tokens["input_ids"].numpy())

    max_tokens = 100

    # Original method
    start = time.time()
    cache = make_kv_cache(24)
    _, _, cache = llm(input_ids, cache=cache)

    generated = []
    next_input = mx.zeros((1, 1), dtype=mx.int32)

    for _ in range(max_tokens):
        _, speech_logits, cache = llm(next_input, cache=cache)
        logits = speech_logits[:, -1, :]
        next_token = mx.argmax(logits, axis=-1)  # Greedy for comparison
        generated.append(next_token)
        next_input = next_token[:, None]
        mx.eval(next_token)  # Original: eval each token

    original_time = time.time() - start
    print(f"   Original: {max_tokens} tokens in {original_time:.2f}s ({max_tokens/original_time:.1f} tok/s)")

    # ==========================================================================
    # Test 2: Deferred eval (eval every N tokens)
    # ==========================================================================
    print("\n3. Testing deferred eval (every 10 tokens)...")

    start = time.time()
    cache = make_kv_cache(24)
    _, _, cache = llm(input_ids, cache=cache)

    generated = []
    next_input = mx.zeros((1, 1), dtype=mx.int32)

    for i in range(max_tokens):
        _, speech_logits, cache = llm(next_input, cache=cache)
        logits = speech_logits[:, -1, :]
        next_token = mx.argmax(logits, axis=-1)
        generated.append(next_token)
        next_input = next_token[:, None]

        # Eval every 10 tokens
        if (i + 1) % 10 == 0:
            mx.eval(next_token)

    mx.eval(generated)  # Final eval
    deferred_time = time.time() - start
    print(f"   Deferred (10): {max_tokens} tokens in {deferred_time:.2f}s ({max_tokens/deferred_time:.1f} tok/s)")

    # ==========================================================================
    # Test 3: Compiled step function
    # ==========================================================================
    print("\n4. Testing compiled step function...")

    # Define step function that can be compiled
    def step_fn(model, token, cache_state):
        """Single step: forward + argmax (greedy)."""
        _, speech_logits, new_cache = model(token, cache=cache_state)
        logits = speech_logits[:, -1, :]
        next_token = mx.argmax(logits, axis=-1)
        return next_token, new_cache

    # Test uncompiled first as baseline
    start = time.time()
    cache = make_kv_cache(24)
    _, _, cache = llm(input_ids, cache=cache)

    generated = []
    next_input = mx.zeros((1, 1), dtype=mx.int32)

    for _ in range(max_tokens):
        next_token, cache = step_fn(llm, next_input, cache)
        generated.append(next_token)
        next_input = next_token[:, None]
        mx.eval(next_token)

    uncompiled_step_time = time.time() - start
    print(f"   Uncompiled step: {max_tokens} tokens in {uncompiled_step_time:.2f}s ({max_tokens/uncompiled_step_time:.1f} tok/s)")

    # ==========================================================================
    # Test 4: No eval in loop (pure graph building)
    # ==========================================================================
    print("\n5. Testing no eval in loop (graph building)...")

    start = time.time()
    cache = make_kv_cache(24)
    _, _, cache = llm(input_ids, cache=cache)

    generated = []
    next_input = mx.zeros((1, 1), dtype=mx.int32)

    for _ in range(max_tokens):
        _, speech_logits, cache = llm(next_input, cache=cache)
        logits = speech_logits[:, -1, :]
        next_token = mx.argmax(logits, axis=-1)
        generated.append(next_token)
        next_input = next_token[:, None]
        # NO mx.eval() inside loop!

    # Single eval at end
    mx.eval(generated)
    no_eval_time = time.time() - start
    print(f"   No eval in loop: {max_tokens} tokens in {no_eval_time:.2f}s ({max_tokens/no_eval_time:.1f} tok/s)")

    # ==========================================================================
    # Test 5: Compiled model forward
    # ==========================================================================
    print("\n6. Testing compiled model forward...")

    # Compile just the forward pass
    @mx.compile
    def compiled_forward(model_llm, token, cache_list):
        # Note: mx.compile doesn't work well with object methods
        # We need to use functional style
        return model_llm(token, cache=cache_list)

    # Unfortunately mx.compile with model state is tricky
    # Let's try a different approach - compile the full step
    print("   (Skipping - mx.compile with model state requires special handling)")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    results = [
        ("Original (eval each)", original_time, max_tokens/original_time),
        ("Deferred (eval/10)", deferred_time, max_tokens/deferred_time),
        ("Uncompiled step", uncompiled_step_time, max_tokens/uncompiled_step_time),
        ("No eval in loop", no_eval_time, max_tokens/no_eval_time),
    ]

    print("\n{:<25} {:>10} {:>15}".format("Method", "Time (s)", "Tokens/sec"))
    print("-" * 50)
    for name, t, tps in results:
        print(f"{name:<25} {t:>10.2f} {tps:>15.1f}")

    best_method, _, best_tps = max(results, key=lambda x: x[2])
    print(f"\nBest: {best_method} at {best_tps:.1f} tok/s")

    # Compare to target
    target_tps = 100  # mlx-lm achieves this for similar model sizes
    print(f"\nTarget: {target_tps} tok/s")
    print(f"Gap: {target_tps - best_tps:.1f} tok/s ({best_tps/target_tps*100:.0f}% of target)")

    if best_tps < 20:
        print("\n[CRITICAL] Generation is still very slow (<20 tok/s)")
        print("Possible issues:")
        print("  1. MLX version may have inefficient SDPA for small batches")
        print("  2. Model architecture may not be well-optimized for MLX")
        print("  3. Consider batched/parallel generation")


if __name__ == "__main__":
    test_compiled_generation()
