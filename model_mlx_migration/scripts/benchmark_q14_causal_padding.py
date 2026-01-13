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
Q14: Causal Padding Preallocation Evaluation

Tests whether pre-allocating causal padding arrays improves performance.
CosyVoice3's PreLookaheadLayer uses mx.pad() for causal convolutions.

Expected: 1-2% improvement (per tracker)
"""

import time
import mlx.core as mx
import mlx.nn as nn


class PreLookaheadLayerOriginal(nn.Module):
    """Original implementation - creates padding specs on each call."""

    def __init__(self, in_channels: int = 80, hidden_channels: int = 1024):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=4, padding=0, bias=True)
        self.conv1_pad = 3
        self.conv2 = nn.Conv1d(hidden_channels, in_channels, kernel_size=3, padding=0, bias=True)
        self.conv2_pad = 2

    def __call__(self, x: mx.array) -> mx.array:
        # Original: creates padding spec each call
        x = mx.pad(x, [(0, 0), (self.conv1_pad, 0), (0, 0)])
        x = self.conv1(x)
        x = nn.relu(x)
        x = mx.pad(x, [(0, 0), (self.conv2_pad, 0), (0, 0)])
        x = self.conv2(x)
        x = nn.relu(x)
        return x


class PreLookaheadLayerQ14(nn.Module):
    """Q14 optimization - pre-allocates padding spec tuples."""

    def __init__(self, in_channels: int = 80, hidden_channels: int = 1024):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=4, padding=0, bias=True)
        self.conv2 = nn.Conv1d(hidden_channels, in_channels, kernel_size=3, padding=0, bias=True)
        # Q14: Pre-allocate padding specs as class attributes
        self._pad1 = [(0, 0), (3, 0), (0, 0)]  # conv1_pad = 3
        self._pad2 = [(0, 0), (2, 0), (0, 0)]  # conv2_pad = 2

    def __call__(self, x: mx.array) -> mx.array:
        # Q14: Use pre-allocated padding specs
        x = mx.pad(x, self._pad1)
        x = self.conv1(x)
        x = nn.relu(x)
        x = mx.pad(x, self._pad2)
        x = self.conv2(x)
        x = nn.relu(x)
        return x


def benchmark_layer(layer, x, num_iters=100, warmup=10):
    """Benchmark a layer."""
    # Warmup
    for _ in range(warmup):
        _ = layer(x)
        mx.eval(_)

    # Benchmark
    times = []
    for _ in range(num_iters):
        start = time.perf_counter()
        out = layer(x)
        mx.eval(out)
        times.append(time.perf_counter() - start)

    return sum(times) / len(times) * 1000  # ms


def main():
    print("=" * 60)
    print("Q14: Causal Padding Preallocation Evaluation")
    print("=" * 60)

    # Create test input (typical mel spectrogram shape)
    B, L, C = 1, 200, 80  # batch=1, 200 mel frames, 80 channels
    x = mx.random.normal((B, L, C))
    mx.eval(x)

    # Create layers
    layer_orig = PreLookaheadLayerOriginal(C, 1024)
    layer_q14 = PreLookaheadLayerQ14(C, 1024)

    # Initialize weights
    mx.eval(layer_orig.parameters())
    mx.eval(layer_q14.parameters())

    # Benchmark
    print(f"\nInput shape: {x.shape}")
    print("Iterations: 100 (10 warmup)")
    print()

    time_orig = benchmark_layer(layer_orig, x)
    time_q14 = benchmark_layer(layer_q14, x)

    speedup = time_orig / time_q14

    print(f"{'Configuration':<30} {'Latency (ms)':<15} {'Speedup':<10}")
    print("-" * 55)
    print(f"{'Original (padding each call)':<30} {time_orig:>12.4f}   {'1.00x':<10}")
    print(f"{'Q14 (pre-allocated padding)':<30} {time_q14:>12.4f}   {f'{speedup:.2f}x':<10}")

    # Test numerical equivalence
    out_orig = layer_orig(x)
    out_q14 = layer_q14(x)
    mx.eval(out_orig, out_q14)
    max_diff = mx.max(mx.abs(out_orig - out_q14)).item()
    print(f"\nNumerical verification: max diff = {max_diff}")

    # Also test in context of full flow (PreLookaheadLayer is small part)
    print("\n" + "=" * 60)
    print("FLOW CONTEXT ANALYSIS")
    print("=" * 60)

    # Estimate PreLookaheadLayer % of flow
    # Flow inference is ~189ms (optimized), PreLookaheadLayer runs once per inference
    pre_lookahead_time = time_orig
    flow_time_ms = 189.0  # From tracker benchmarks
    pre_lookahead_pct = (pre_lookahead_time / flow_time_ms) * 100

    print(f"\nPreLookaheadLayer latency: {pre_lookahead_time:.4f} ms")
    print(f"Flow inference latency: {flow_time_ms:.1f} ms")
    print(f"PreLookaheadLayer % of flow: {pre_lookahead_pct:.2f}%")

    savings_ms = time_orig - time_q14
    e2e_pct = (savings_ms / flow_time_ms) * 100
    print(f"\nPotential savings: {savings_ms:.4f} ms ({e2e_pct:.3f}% E2E)")

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    if speedup < 1.02:
        print("Q14: NOT WORTH - Speedup < 2%")
        print("Reason: Python tuple creation overhead is negligible")
        print("        mx.pad() dominates timing, not the padding spec")
    else:
        print(f"Q14: {speedup:.2f}x speedup - Consider implementation")


if __name__ == "__main__":
    main()
