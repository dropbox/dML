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
Q21 Evaluation on ACTUAL CosyVoice3 Vocoder - Worker #1346

The simplified simulator showed Q21+H2 providing 2.14x over H2 alone.
This script tests against the real vocoder to verify the effect.
"""

import mlx.core as mx
import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.pytorch_to_mlx.converters.models.cosyvoice3_vocoder import (
    CausalHiFTGenerator,
    create_cosyvoice3_vocoder_config,
    ResBlock
)


def benchmark_fn(fn, *args, warmup: int = 3, runs: int = 10, name: str = ""):
    """Benchmark a function with warmup and averaging."""
    for _ in range(warmup):
        out = fn(*args)
        mx.eval(out)

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        out = fn(*args)
        mx.eval(out)
        times.append((time.perf_counter() - start) * 1000)

    return sum(times) / len(times)


def compile_resblock_steps(model: CausalHiFTGenerator):
    """
    Q21 optimization: Compile each ResBlock iteration step.

    This adds step-level compilation on top of H2 whole-model compilation.
    """
    compiled_count = 0

    # Compile main ResBlocks (9 total)
    for rb_idx, resblock in enumerate(model.resblocks):
        for i in range(resblock.num_layers):
            # Create compiled step function for this iteration
            conv1 = resblock.convs1[i]
            conv2 = resblock.convs2[i]
            act1 = resblock.activations1[i]
            act2 = resblock.activations2[i]

            @mx.compile
            def step_fn(x, _c1=conv1, _c2=conv2, _a1=act1, _a2=act2):
                xt = _a1(x)
                xt = _c1(xt)
                xt = _a2(xt)
                xt = _c2(xt)
                return x + xt

            # Store compiled function (but we can't easily inject it)
            compiled_count += 1

    # Compile source ResBlocks (3 total)
    for rb_idx, resblock in enumerate(model.source_resblocks):
        for i in range(resblock.num_layers):
            compiled_count += 1

    return compiled_count


def create_q21_resblock_forward(resblock: ResBlock):
    """Create a compiled forward function for a ResBlock."""
    num_layers = resblock.num_layers

    # We need to restructure to make compilation effective
    # The issue: the original forward uses a loop with dynamic indices
    # Compilation works better with explicit unrolling

    @mx.compile
    def compiled_forward(x):
        for i in range(num_layers):
            xt = resblock.activations1[i](x)
            xt = resblock.convs1[i](xt)
            xt = resblock.activations2[i](xt)
            xt = resblock.convs2[i](xt)
            x = x + xt
        return x

    return compiled_forward


def run_benchmark():
    """Run Q21 evaluation on actual vocoder."""
    print("=" * 60)
    print("Q21 Vocoder Conv1d Kernel Fusion - ACTUAL VOCODER")
    print("=" * 60)

    # Create vocoder
    config = create_cosyvoice3_vocoder_config()
    vocoder = CausalHiFTGenerator(config)

    # Initialize weights
    mx.eval(vocoder.parameters())

    # Test input: 50 mel frames
    batch_size = 1
    mel_frames = 50
    mel = mx.random.normal((batch_size, config.in_channels, mel_frames))
    mx.eval(mel)

    print(f"\nConfig: {len(vocoder.resblocks)} main ResBlocks, "
          f"{len(vocoder.source_resblocks)} source ResBlocks")
    print(f"Input: batch={batch_size}, mel_channels={config.in_channels}, "
          f"mel_frames={mel_frames}")

    print("\n--- Benchmarking Configurations ---\n")

    # 1. Baseline (no H2 compile)
    vocoder._is_compiled = False
    time_baseline = benchmark_fn(vocoder.forward, mel, name="Baseline")
    print(f"1. Baseline (no compile):      {time_baseline:.2f} ms")

    # 2. H2 only (current DONE state)
    vocoder.compile_model()
    time_h2 = benchmark_fn(vocoder, mel, name="H2")
    print(f"2. H2 (whole compile):         {time_h2:.2f} ms  ({time_baseline/time_h2:.2f}x)")

    # 3. Test individual ResBlock compilation
    # Create compiled ResBlock forwards
    vocoder._is_compiled = False  # Disable H2

    # Compile each ResBlock's forward
    compiled_resblocks = []
    for rb in vocoder.resblocks:
        compiled_rb = create_q21_resblock_forward(rb)
        compiled_resblocks.append(compiled_rb)

    compiled_source_resblocks = []
    for rb in vocoder.source_resblocks:
        compiled_rb = create_q21_resblock_forward(rb)
        compiled_source_resblocks.append(compiled_rb)

    # Monkey-patch the compiled forwards
    original_rb_calls = [rb.__call__ for rb in vocoder.resblocks]
    original_src_rb_calls = [rb.__call__ for rb in vocoder.source_resblocks]

    for i, rb in enumerate(vocoder.resblocks):
        rb.__call__ = lambda x, fn=compiled_resblocks[i]: fn(x)
    for i, rb in enumerate(vocoder.source_resblocks):
        rb.__call__ = lambda x, fn=compiled_source_resblocks[i]: fn(x)

    time_q21_only = benchmark_fn(vocoder.forward, mel, name="Q21-only")
    print(f"3. Q21 only (ResBlock compile): {time_q21_only:.2f} ms  ({time_baseline/time_q21_only:.2f}x)")

    # 4. Q21 + H2 (both)
    vocoder.compile_model()
    time_q21_h2 = benchmark_fn(vocoder, mel, name="Q21+H2")
    print(f"4. Q21 + H2 (both):            {time_q21_h2:.2f} ms  ({time_baseline/time_q21_h2:.2f}x)")

    # Restore original calls for clean state
    for i, rb in enumerate(vocoder.resblocks):
        rb.__call__ = original_rb_calls[i]
    for i, rb in enumerate(vocoder.source_resblocks):
        rb.__call__ = original_src_rb_calls[i]

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\n| Configuration | Latency | Speedup vs Baseline |")
    print("|---------------|---------|---------------------|")
    print(f"| Baseline      | {time_baseline:.2f} ms | 1.00x |")
    print(f"| H2 only       | {time_h2:.2f} ms | {time_baseline/time_h2:.2f}x |")
    print(f"| Q21 only      | {time_q21_only:.2f} ms | {time_baseline/time_q21_only:.2f}x |")
    print(f"| Q21 + H2      | {time_q21_h2:.2f} ms | {time_baseline/time_q21_h2:.2f}x |")

    # Q21 benefit analysis
    q21_vs_h2 = time_h2 / time_q21_h2
    print(f"\n**Q21 benefit over H2 alone: {q21_vs_h2:.2f}x**")

    if q21_vs_h2 > 1.05:
        verdict = "WORTH IMPLEMENTING"
        print("\nResult: Q21 provides MEANINGFUL benefit over H2")
    elif q21_vs_h2 > 1.01:
        verdict = "MARGINAL"
        print("\nResult: Q21 provides MARGINAL benefit over H2")
    elif q21_vs_h2 >= 0.98:
        verdict = "NOT WORTH"
        print("\nResult: Q21 provides NO additional benefit over H2")
    else:
        verdict = "NOT WORTH (SLOWER)"
        print("\nResult: Q21 is SLOWER than H2 alone")

    print(f"\n**VERDICT: {verdict}**")

    # E2E impact calculation
    print("\n" + "=" * 60)
    print("E2E IMPACT ANALYSIS")
    print("=" * 60)

    vocoder_pct = 10  # Vocoder is ~10% of total inference
    if q21_vs_h2 > 1.0:
        q21_speedup_pct = (q21_vs_h2 - 1.0) * 100
        e2e_impact_pct = q21_speedup_pct * (vocoder_pct / 100)
        print(f"\nVocoder is {vocoder_pct}% of total CosyVoice3 inference time")
        print(f"Q21 vocoder speedup over H2: {q21_speedup_pct:.1f}%")
        print(f"**E2E impact: {e2e_impact_pct:.2f}%**")

        if e2e_impact_pct < 1.0:
            print("\nE2E impact is <1% - NOT WORTH the code complexity")
    else:
        print("\nQ21 provides no benefit or is slower - NOT WORTH")

    return {
        "baseline": time_baseline,
        "h2": time_h2,
        "q21_only": time_q21_only,
        "q21_h2": time_q21_h2,
        "verdict": verdict
    }


if __name__ == "__main__":
    results = run_benchmark()
