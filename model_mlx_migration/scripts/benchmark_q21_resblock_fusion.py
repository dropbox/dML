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
Q21 Vocoder Conv1d Kernel Fusion Evaluation - Worker #1346

Test whether additional ResBlock-level compilation provides benefit
on top of H2 (whole vocoder mx.compile).

H2 already provides 1.44x speedup by compiling the entire forward pass.
Q21 proposes fusing conv1 -> activation -> conv2 sequences within ResBlocks.

This benchmark tests:
1. Baseline (no compilation)
2. H2 (whole vocoder mx.compile) - current DONE state
3. Q21 attempt: ResBlock-level compiled functions

Expected result: Q21 provides NO ADDITIONAL benefit since H2 already
captures these patterns at the graph level.
"""

import mlx.core as mx
import mlx.nn as nn
import time


def snake_activation(x: mx.array, alpha: mx.array) -> mx.array:
    """Snake activation: x + (1/alpha) * sin^2(alpha * x)"""
    alpha_exp = alpha[None, :, None]
    return x + (1.0 / (alpha_exp + 1e-9)) * mx.power(mx.sin(alpha_exp * x), 2)


class Conv1dWrapper(nn.Module):
    """Simple Conv1d wrapper for PyTorch format [B, C, L]."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        padding = (kernel_size * dilation - dilation) // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)

    def __call__(self, x: mx.array) -> mx.array:
        x = x.transpose(0, 2, 1)
        x = self.conv(x)
        return x.transpose(0, 2, 1)


class BaselineResBlock(nn.Module):
    """Baseline ResBlock - no compilation."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.channels = channels

        # 3 iterations of (conv1 -> snake -> conv2 -> snake -> residual)
        self.convs1 = [
            Conv1dWrapper(channels, channels, kernel_size, dilation=d)
            for d in [1, 3, 5]
        ]
        self.convs2 = [
            Conv1dWrapper(channels, channels, kernel_size, dilation=1)
            for _ in range(3)
        ]
        self.alphas1 = [mx.ones((channels,)) for _ in range(3)]
        self.alphas2 = [mx.ones((channels,)) for _ in range(3)]

    def __call__(self, x: mx.array) -> mx.array:
        for i in range(3):
            xt = snake_activation(x, self.alphas1[i])
            xt = self.convs1[i](xt)
            xt = snake_activation(xt, self.alphas2[i])
            xt = self.convs2[i](xt)
            x = x + xt
        return x


class Q21CompiledResBlock(nn.Module):
    """Q21 attempt: ResBlock with compiled inner loop."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.channels = channels

        self.convs1 = [
            Conv1dWrapper(channels, channels, kernel_size, dilation=d)
            for d in [1, 3, 5]
        ]
        self.convs2 = [
            Conv1dWrapper(channels, channels, kernel_size, dilation=1)
            for _ in range(3)
        ]
        self.alphas1 = [mx.ones((channels,)) for _ in range(3)]
        self.alphas2 = [mx.ones((channels,)) for _ in range(3)]

        self._compiled_steps = None

    def compile_steps(self):
        """Pre-compile each iteration step."""
        self._compiled_steps = []

        for i in range(3):
            # Capture loop variables
            conv1 = self.convs1[i]
            conv2 = self.convs2[i]
            alpha1 = self.alphas1[i]
            alpha2 = self.alphas2[i]

            @mx.compile
            def step_fn(x, _conv1=conv1, _conv2=conv2, _a1=alpha1, _a2=alpha2):
                xt = snake_activation(x, _a1)
                xt = _conv1(xt)
                xt = snake_activation(xt, _a2)
                xt = _conv2(xt)
                return x + xt

            self._compiled_steps.append(step_fn)

    def __call__(self, x: mx.array) -> mx.array:
        if self._compiled_steps:
            for step_fn in self._compiled_steps:
                x = step_fn(x)
        else:
            for i in range(3):
                xt = snake_activation(x, self.alphas1[i])
                xt = self.convs1[i](xt)
                xt = snake_activation(xt, self.alphas2[i])
                xt = self.convs2[i](xt)
                x = x + xt
        return x


class VocoderSimulator(nn.Module):
    """Simplified vocoder for benchmarking ResBlock behavior."""

    def __init__(self, num_resblocks: int = 9, channels: int = 64, use_q21: bool = False):
        super().__init__()

        if use_q21:
            self.resblocks = [Q21CompiledResBlock(channels) for _ in range(num_resblocks)]
        else:
            self.resblocks = [BaselineResBlock(channels) for _ in range(num_resblocks)]

        self.use_q21 = use_q21
        self._is_compiled = False

    def compile_q21(self):
        """Compile Q21 ResBlock steps (pre-H2)."""
        if self.use_q21:
            for rb in self.resblocks:
                rb.compile_steps()

    def compile_h2(self):
        """Compile entire forward (H2 optimization)."""
        self._compiled_forward = mx.compile(self.forward)
        self._is_compiled = True

    def forward(self, x: mx.array) -> mx.array:
        for rb in self.resblocks:
            x = rb(x)
        return x

    def __call__(self, x: mx.array) -> mx.array:
        if self._is_compiled:
            return self._compiled_forward(x)
        return self.forward(x)


def benchmark_fn(fn, x, warmup: int = 3, runs: int = 10, name: str = ""):
    """Benchmark a function with warmup and averaging."""
    # Warmup
    for _ in range(warmup):
        out = fn(x)
        mx.eval(out)

    # Timed runs
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        out = fn(x)
        mx.eval(out)
        times.append((time.perf_counter() - start) * 1000)

    avg = sum(times) / len(times)
    return avg


def run_benchmark():
    """Run Q21 evaluation benchmark."""
    print("=" * 60)
    print("Q21 Vocoder Conv1d Kernel Fusion Evaluation")
    print("=" * 60)

    # Simulate vocoder ResBlocks
    # Real vocoder: 9 resblocks, channels=64 (final stage)
    batch_size = 1
    channels = 64
    seq_len = 400  # 50 mel frames * 8 (first upsample)
    num_resblocks = 9

    print(f"\nConfig: {num_resblocks} ResBlocks, channels={channels}, seq_len={seq_len}")

    # Create test input
    x = mx.random.normal((batch_size, channels, seq_len))
    mx.eval(x)

    print("\n--- Benchmarking Configurations ---\n")

    # 1. Baseline (no compilation)
    model_baseline = VocoderSimulator(num_resblocks, channels, use_q21=False)
    mx.eval(model_baseline.parameters())

    time_baseline = benchmark_fn(model_baseline, x, name="Baseline")
    print(f"1. Baseline (no compile):      {time_baseline:.2f} ms")

    # 2. H2 only (whole model compile)
    model_h2 = VocoderSimulator(num_resblocks, channels, use_q21=False)
    mx.eval(model_h2.parameters())
    model_h2.compile_h2()

    time_h2 = benchmark_fn(model_h2, x, name="H2")
    print(f"2. H2 (whole compile):         {time_h2:.2f} ms  ({time_baseline/time_h2:.2f}x)")

    # 3. Q21 only (ResBlock-level compile, no H2)
    model_q21_only = VocoderSimulator(num_resblocks, channels, use_q21=True)
    mx.eval(model_q21_only.parameters())
    model_q21_only.compile_q21()

    time_q21_only = benchmark_fn(model_q21_only, x, name="Q21-only")
    print(f"3. Q21 only (step compile):    {time_q21_only:.2f} ms  ({time_baseline/time_q21_only:.2f}x)")

    # 4. Q21 + H2 (both compilations)
    model_q21_h2 = VocoderSimulator(num_resblocks, channels, use_q21=True)
    mx.eval(model_q21_h2.parameters())
    model_q21_h2.compile_q21()
    model_q21_h2.compile_h2()

    time_q21_h2 = benchmark_fn(model_q21_h2, x, name="Q21+H2")
    print(f"4. Q21 + H2 (both):            {time_q21_h2:.2f} ms  ({time_baseline/time_q21_h2:.2f}x)")

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
        print("\nResult: Q21 provides MEANINGFUL benefit over H2")
        verdict = "WORTH INVESTIGATING"
    elif q21_vs_h2 > 1.01:
        print("\nResult: Q21 provides MARGINAL benefit over H2")
        verdict = "MARGINAL"
    elif q21_vs_h2 >= 0.99:
        print("\nResult: Q21 provides NO additional benefit over H2")
        verdict = "NOT WORTH"
    else:
        print("\nResult: Q21 is SLOWER than H2 alone")
        verdict = "NOT WORTH (SLOWER)"

    print(f"\n**VERDICT: {verdict}**")

    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    print("""
Q21 proposes compiling ResBlock inner loops (conv -> snake -> conv -> snake)
separately before H2 whole-model compilation.

Key insight: mx.compile operates at the graph level. When H2 compiles the
entire forward pass, it already captures and fuses these patterns.

Pre-compiling ResBlock steps (Q21) before H2:
- Creates nested compiled functions
- May add overhead from function call boundaries
- Does NOT provide graph-level fusion benefits beyond H2

This follows the pattern seen in:
- Q23 (Static Shape Compilation): 0.72-0.88x - compilation overhead
- Q15-Layer (LLM Layer Compile): 0.98x - no benefit
- H2 already captures the optimization opportunity
""")

    return {
        "baseline": time_baseline,
        "h2": time_h2,
        "q21_only": time_q21_only,
        "q21_h2": time_q21_h2,
        "verdict": verdict
    }


if __name__ == "__main__":
    results = run_benchmark()
