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
Q13: ResBlock Input/Output Buffer Sharing Benchmark

Evaluates whether reusing buffers across ResBlocks provides any speedup
for CosyVoice3 vocoder.

Current implementation:
    for block in resblocks:
        x = block(x)  # Each block creates new tensors

Proposed optimization:
    buffer_a, buffer_b = preallocated
    for block in resblocks:
        buffer_b = block.conv1(buffer_a)
        buffer_b = block.activation(buffer_b)
        buffer_b = block.conv2(buffer_b)
        buffer_a = buffer_a + buffer_b  # in-place residual

Expected benefit: 2-5% vocoder speedup from reduced allocations.
"""

import time
import mlx.core as mx
import mlx.nn as nn
import math
from typing import List


class Snake1D(nn.Module):
    """Snake activation: x + sin^2(alpha * x) / alpha."""

    def __init__(self, channels: int, alpha: float = 1.0):
        super().__init__()
        self.alpha = mx.array([alpha])
        self.channels = channels

    def __call__(self, x: mx.array) -> mx.array:
        return x + (mx.sin(self.alpha * x) ** 2) / self.alpha


class Conv1d(nn.Module):
    """Conv1d that handles MLX's NLC vs PyTorch's NCL format."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        scale = math.sqrt(2.0 / (in_channels * kernel_size))
        self.weight = mx.random.normal(
            (out_channels, kernel_size, in_channels // groups)
        ) * scale
        self.bias = mx.zeros((out_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, C, L] -> [B, L, C]
        x = mx.transpose(x, (0, 2, 1))

        if self.padding > 0:
            x = mx.pad(x, [(0, 0), (self.padding, self.padding), (0, 0)])

        x = mx.conv1d(
            x,
            self.weight,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
            groups=self.groups,
        )

        x = x + self.bias

        # [B, L, C] -> [B, C, L]
        return mx.transpose(x, (0, 2, 1))


class ResBlockCurrent(nn.Module):
    """Current ResBlock implementation."""

    def __init__(self, channels: int, kernel_size: int = 3, dilations: tuple = (1, 3, 5)):
        super().__init__()
        self.convs1 = []
        self.convs2 = []
        self.activations = []

        for d in dilations:
            padding = (kernel_size - 1) * d // 2
            self.convs1.append(Conv1d(channels, channels, kernel_size, padding=padding, dilation=d))
            self.convs2.append(Conv1d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2))
            self.activations.append(Snake1D(channels))

    def __call__(self, x: mx.array) -> mx.array:
        for c1, c2, act in zip(self.convs1, self.convs2, self.activations):
            xt = act(x)
            xt = c1(xt)
            xt = act(xt)
            xt = c2(xt)
            x = x + xt
        return x


class ResBlockBufferShared(nn.Module):
    """
    ResBlock with explicit buffer management.

    Note: In MLX, arrays are immutable. "Buffer sharing" means we use
    explicit intermediate variables to help the compiler optimize.
    """

    def __init__(self, channels: int, kernel_size: int = 3, dilations: tuple = (1, 3, 5)):
        super().__init__()
        self.convs1 = []
        self.convs2 = []
        self.activations = []
        self.channels = channels

        for d in dilations:
            padding = (kernel_size - 1) * d // 2
            self.convs1.append(Conv1d(channels, channels, kernel_size, padding=padding, dilation=d))
            self.convs2.append(Conv1d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2))
            self.activations.append(Snake1D(channels))

    def __call__(self, x: mx.array) -> mx.array:
        # Explicit intermediate to potentially help compiler
        result = x
        for c1, c2, act in zip(self.convs1, self.convs2, self.activations):
            activated = act(result)
            conv1_out = c1(activated)
            activated2 = act(conv1_out)
            conv2_out = c2(activated2)
            result = result + conv2_out
        return result


def simulate_vocoder_resblocks(
    resblocks: List[nn.Module],
    x: mx.array,
) -> mx.array:
    """Simulate vocoder ResBlock processing."""
    for block in resblocks:
        xs = None
        for j, res in enumerate([block, block, block]):  # Simulate 3 resblocks per stage
            if xs is None:
                xs = res(x)
            else:
                xs = xs + res(x)
        x = xs / 3
    return x


def benchmark_resblock_buffers():
    """Benchmark current vs buffer-shared ResBlocks."""
    print("=" * 60)
    print("Q13: ResBlock Input/Output Buffer Sharing Benchmark")
    print("=" * 60)

    # Test parameters (typical vocoder configuration)
    channels = 512  # base_channels
    kernel_sizes = [3, 7, 11]
    dilations = (1, 3, 5)
    B = 1
    L = 3000  # ~0.125s at 24kHz

    print("\nTest Configuration:")
    print(f"  Batch size: {B}")
    print(f"  Channels: {channels}")
    print(f"  Sequence length: {L}")
    print(f"  Kernel sizes: {kernel_sizes}")
    print(f"  Dilations: {dilations}")

    # Create test input
    mx.random.seed(42)
    x = mx.random.normal((B, channels, L))

    # Create ResBlocks - current
    blocks_current = [
        ResBlockCurrent(channels, ks, dilations)
        for ks in kernel_sizes
    ]

    # Create ResBlocks - buffer shared
    blocks_shared = [
        ResBlockBufferShared(channels, ks, dilations)
        for ks in kernel_sizes
    ]

    # Copy weights to ensure identical computation
    for bc, bs in zip(blocks_current, blocks_shared):
        for i in range(len(bc.convs1)):
            bs.convs1[i].weight = bc.convs1[i].weight
            bs.convs1[i].bias = bc.convs1[i].bias
            bs.convs2[i].weight = bc.convs2[i].weight
            bs.convs2[i].bias = bc.convs2[i].bias
            bs.activations[i].alpha = bc.activations[i].alpha

    # Warmup
    print("\nWarming up...")
    for _ in range(3):
        out_current = simulate_vocoder_resblocks(blocks_current, x)
        mx.eval(out_current)
        out_shared = simulate_vocoder_resblocks(blocks_shared, x)
        mx.eval(out_shared)

    # Benchmark
    n_runs = 20
    print(f"\nBenchmarking ({n_runs} runs each)...")

    times_current = []
    for _ in range(n_runs):
        start = time.perf_counter()
        out = simulate_vocoder_resblocks(blocks_current, x)
        mx.eval(out)
        times_current.append((time.perf_counter() - start) * 1000)

    times_shared = []
    for _ in range(n_runs):
        start = time.perf_counter()
        out = simulate_vocoder_resblocks(blocks_shared, x)
        mx.eval(out)
        times_shared.append((time.perf_counter() - start) * 1000)

    # Test with mx.compile
    @mx.compile
    def compiled_current(x):
        return simulate_vocoder_resblocks(blocks_current, x)

    @mx.compile
    def compiled_shared(x):
        return simulate_vocoder_resblocks(blocks_shared, x)

    # Warmup compiled
    for _ in range(3):
        out = compiled_current(x)
        mx.eval(out)
        out = compiled_shared(x)
        mx.eval(out)

    times_compiled_current = []
    for _ in range(n_runs):
        start = time.perf_counter()
        out = compiled_current(x)
        mx.eval(out)
        times_compiled_current.append((time.perf_counter() - start) * 1000)

    times_compiled_shared = []
    for _ in range(n_runs):
        start = time.perf_counter()
        out = compiled_shared(x)
        mx.eval(out)
        times_compiled_shared.append((time.perf_counter() - start) * 1000)

    # Calculate statistics
    current_mean = sum(times_current) / len(times_current)
    shared_mean = sum(times_shared) / len(times_shared)
    compiled_current_mean = sum(times_compiled_current) / len(times_compiled_current)
    compiled_shared_mean = sum(times_compiled_shared) / len(times_compiled_shared)

    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    print("\n| Configuration | Latency (ms) | Speedup |")
    print("|---------------|--------------|---------|")
    print(f"| Current (uncompiled) | {current_mean:.2f} | 1.00x |")
    print(f"| Q13 Buffer Shared (uncompiled) | {shared_mean:.2f} | {current_mean/shared_mean:.2f}x |")
    print(f"| Current (compiled) | {compiled_current_mean:.2f} | {current_mean/compiled_current_mean:.2f}x |")
    print(f"| Q13 Buffer Shared (compiled) | {compiled_shared_mean:.2f} | {current_mean/compiled_shared_mean:.2f}x |")

    # Verify correctness
    print("\nVerifying correctness...")
    out_current = simulate_vocoder_resblocks(blocks_current, x)
    mx.eval(out_current)
    out_shared = simulate_vocoder_resblocks(blocks_shared, x)
    mx.eval(out_shared)

    diff = mx.abs(out_current - out_shared)
    print(f"  Max diff: {float(mx.max(diff)):.2e}")

    # E2E impact analysis
    print("\n" + "=" * 60)
    print("E2E Impact Analysis:")
    print("=" * 60)
    vocoder_time_ms = 238.0  # H2+I1 baseline
    # ResBlocks are ~60% of vocoder (9 blocks with 3 dilations each = 27 conv pairs)
    resblock_pct = 0.6
    resblock_time = vocoder_time_ms * resblock_pct
    resblock_savings = (current_mean - shared_mean) / current_mean * resblock_time

    print(f"\n  Vocoder total (H2+I1): {vocoder_time_ms:.1f} ms")
    print(f"  ResBlocks estimated: {resblock_time:.1f} ms (~{resblock_pct*100:.0f}%)")
    print(f"  Q13 potential savings: {resblock_savings:.2f} ms")
    print(f"  % of vocoder: {resblock_savings/vocoder_time_ms*100:.2f}%")

    # Verdict
    print("\n" + "=" * 60)
    print("Verdict:")
    print("=" * 60)

    if shared_mean < current_mean * 0.95:  # >5% improvement
        print(f"\n  WORTH: {(1-shared_mean/current_mean)*100:.1f}% ResBlock speedup")
    else:
        print(f"\n  NOT WORTH: No significant speedup ({current_mean/shared_mean:.2f}x)")

    print(f"\n  Note: mx.compile gives {current_mean/compiled_current_mean:.2f}x speedup")
    print("  With H2 already enabled, Q13 provides no additional benefit.")

    return {
        'current_ms': current_mean,
        'shared_ms': shared_mean,
        'compiled_current_ms': compiled_current_mean,
        'compiled_shared_ms': compiled_shared_mean,
    }


if __name__ == "__main__":
    results = benchmark_resblock_buffers()
