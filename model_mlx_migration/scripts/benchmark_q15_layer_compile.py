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
Benchmark script for Q15-Layer: LLM Decoder Layer Compilation.

Tests the speedup from compiling individual decoder layer forward passes.

Expected: 10-20% speedup during autoregressive generation.
"""

import time
import mlx.core as mx
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.pytorch_to_mlx.converters.models.cosyvoice2_llm import (
    CosyVoice2LLM,
    Qwen2Config,
    make_kv_cache,
)


def benchmark_autoregressive_generation(
    model: CosyVoice2LLM,
    num_tokens: int = 50,
    warmup: int = 3,
    runs: int = 5,
) -> dict:
    """
    Benchmark autoregressive token generation.

    Returns:
        dict with total_time, per_token_time, tokens_per_second
    """
    config = model.config
    batch = 1

    # Initial prompt
    prompt_len = 5
    prompt_ids = mx.random.randint(0, config.vocab_size, (batch, prompt_len))

    # Warmup
    for _ in range(warmup):
        cache = make_kv_cache(config.num_hidden_layers)
        _, _, cache = model(prompt_ids, cache=cache)
        mx.eval(cache)  # Force computation

        # Generate a few tokens
        for _ in range(3):
            next_input = mx.random.randint(0, config.vocab_size, (batch, 1))
            _, _, cache = model(next_input, cache=cache)
            mx.eval(cache)

    # Timed runs
    times = []
    for _ in range(runs):
        cache = make_kv_cache(config.num_hidden_layers)

        # Initial prompt forward
        _, _, cache = model(prompt_ids, cache=cache)
        mx.eval(cache)

        start = time.perf_counter()

        # Generate tokens autoregressively
        for _ in range(num_tokens):
            next_input = mx.random.randint(0, config.vocab_size, (batch, 1))
            _, _, cache = model(next_input, cache=cache)
            mx.eval(cache)

        end = time.perf_counter()
        times.append(end - start)

    avg_time = sum(times) / len(times)
    return {
        "total_time_ms": avg_time * 1000,
        "per_token_ms": (avg_time * 1000) / num_tokens,
        "tokens_per_second": num_tokens / avg_time,
    }


def main():
    print("=" * 60)
    print("Q15-Layer Benchmark: LLM Decoder Layer Compilation")
    print("=" * 60)

    # Use default config for CosyVoice2 LLM
    # Qwen2Config defaults: 24 layers, 896 hidden, GQA (7/1 heads)
    config = Qwen2Config()

    print(f"\nConfig: {config.num_hidden_layers} layers, "
          f"{config.hidden_size} hidden, "
          f"GQA ({config.num_attention_heads}/{config.num_key_value_heads} heads)")

    # Test parameters
    num_tokens = 50
    warmup = 3
    runs = 5

    print(f"\nTest: {num_tokens} tokens, {warmup} warmup, {runs} runs")
    print("-" * 60)

    # Baseline: Uncompiled
    print("\n[1/2] Baseline (uncompiled)...")
    model = CosyVoice2LLM(config, early_exit_layer=6)
    mx.eval(model.parameters())  # Initialize weights

    baseline = benchmark_autoregressive_generation(
        model, num_tokens=num_tokens, warmup=warmup, runs=runs
    )
    print(f"  Total: {baseline['total_time_ms']:.2f}ms")
    print(f"  Per token: {baseline['per_token_ms']:.2f}ms")
    print(f"  Tokens/sec: {baseline['tokens_per_second']:.1f}")

    # Compiled: Q15-Layer
    print("\n[2/2] Q15-Layer (compiled layers)...")
    model_compiled = CosyVoice2LLM(config, early_exit_layer=6)
    mx.eval(model_compiled.parameters())

    # Apply Q15-Layer compilation
    num_compiled = model_compiled.compile_layers()
    print(f"  Compiled {num_compiled} layers")

    compiled = benchmark_autoregressive_generation(
        model_compiled, num_tokens=num_tokens, warmup=warmup, runs=runs
    )
    print(f"  Total: {compiled['total_time_ms']:.2f}ms")
    print(f"  Per token: {compiled['per_token_ms']:.2f}ms")
    print(f"  Tokens/sec: {compiled['tokens_per_second']:.1f}")

    # Results
    speedup = baseline['total_time_ms'] / compiled['total_time_ms']
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nBaseline:  {baseline['per_token_ms']:.2f}ms/token")
    print(f"Compiled:  {compiled['per_token_ms']:.2f}ms/token")
    print(f"Speedup:   {speedup:.2f}x")

    if speedup >= 1.05:
        print(f"\nVERDICT: WORTH IT ({(speedup-1)*100:.1f}% improvement)")
    elif speedup >= 0.95:
        print(f"\nVERDICT: MARGINAL ({(speedup-1)*100:.1f}% difference)")
    else:
        print(f"\nVERDICT: NOT WORTH ({(speedup-1)*100:.1f}% slower)")


if __name__ == "__main__":
    main()
