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
Profile FFN to understand where time is spent.
Tests whether memory bandwidth or compute is the bottleneck.
"""

import time
import mlx.core as mx
import mlx.nn as nn
import numpy as np

WARMUP = 5
ITERATIONS = 50


def benchmark_op(name: str, fn, *args):
    """Benchmark a single operation"""
    # Warmup
    for _ in range(WARMUP):
        out = fn(*args)
        mx.eval(out)

    # Benchmark
    times = []
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        out = fn(*args)
        mx.eval(out)
        times.append(time.perf_counter() - start)

    avg_ms = np.mean(times) * 1000
    return avg_ms, out


def main():
    print("=" * 60)
    print("FFN OPERATION BREAKDOWN")
    print("=" * 60)

    # Whisper large-v3 dimensions
    n_state = 1280
    n_mlp = n_state * 4  # 5120
    batch = 1
    seq_len = 1500  # 30s audio

    print(f"\nDimensions: batch={batch}, seq={seq_len}, n_state={n_state}, n_mlp={n_mlp}")

    mx.random.seed(42)
    x = mx.random.normal((batch, seq_len, n_state))
    mx.eval(x)

    # Create layers
    mlp1 = nn.Linear(n_state, n_mlp)
    mlp2 = nn.Linear(n_mlp, n_state)

    # Get the GELU output intermediate for profiling
    intermediate = mlp1(x)
    mx.eval(intermediate)

    print("\n--- Individual Operation Timing ---")

    # 1. First linear (n_state -> n_mlp)
    time_mlp1, out_mlp1 = benchmark_op("mlp1", mlp1, x)
    print(f"mlp1 (Linear {n_state}->{n_mlp}): {time_mlp1:.3f} ms")

    # 2. GELU activation
    time_gelu, out_gelu = benchmark_op("GELU", nn.gelu, out_mlp1)
    print(f"GELU ({n_mlp}):                   {time_gelu:.3f} ms")

    # 3. Second linear (n_mlp -> n_state)
    time_mlp2, out_mlp2 = benchmark_op("mlp2", mlp2, out_gelu)
    print(f"mlp2 (Linear {n_mlp}->{n_state}): {time_mlp2:.3f} ms")

    total_breakdown = time_mlp1 + time_gelu + time_mlp2
    print(f"\n--- Total from breakdown: {total_breakdown:.3f} ms ---")

    # 4. Combined FFN (single call, sequential execution)
    def ffn_combined(x):
        return mlp2(nn.gelu(mlp1(x)))

    time_combined, _ = benchmark_op("FFN combined", ffn_combined, x)
    print(f"FFN combined (single call):     {time_combined:.3f} ms")

    # 5. Analysis
    print("\n--- Analysis ---")
    print(f"Sum of parts:    {total_breakdown:.3f} ms")
    print(f"Combined:        {time_combined:.3f} ms")
    print(f"Overhead ratio:  {total_breakdown / time_combined:.2f}x")

    # Memory bandwidth analysis
    print("\n--- Memory Bandwidth Analysis ---")

    # Theoretical data movement (float16 = 2 bytes)
    bytes_per_elem = 2  # float16
    input_size = batch * seq_len * n_state * bytes_per_elem
    mlp1_output_size = batch * seq_len * n_mlp * bytes_per_elem
    mlp2_output_size = batch * seq_len * n_state * bytes_per_elem

    # mlp1: read input + weights, write output
    mlp1_weight_size = n_state * n_mlp * bytes_per_elem
    mlp1_bytes = input_size + mlp1_weight_size + mlp1_output_size

    # gelu: read intermediate, write intermediate (in-place?)
    gelu_bytes = mlp1_output_size * 2  # read + write

    # mlp2: read intermediate + weights, write output
    mlp2_weight_size = n_mlp * n_state * bytes_per_elem
    mlp2_bytes = mlp1_output_size + mlp2_weight_size + mlp2_output_size

    total_bytes = mlp1_bytes + gelu_bytes + mlp2_bytes

    print(f"Input size:     {input_size / 1e6:.2f} MB")
    print(f"MLP1 data:      {mlp1_bytes / 1e6:.2f} MB (weights + I/O)")
    print(f"GELU data:      {gelu_bytes / 1e6:.2f} MB")
    print(f"MLP2 data:      {mlp2_bytes / 1e6:.2f} MB (weights + I/O)")
    print(f"Total movement: {total_bytes / 1e6:.2f} MB")

    # M4 Max memory bandwidth: ~400 GB/s
    # Theoretical minimum time
    mem_bandwidth_gbps = 400  # GB/s
    theoretical_min_ms = (total_bytes / 1e9) / mem_bandwidth_gbps * 1000
    print(f"\nTheoretical min (400 GB/s): {theoretical_min_ms:.3f} ms")
    print(f"Actual time:                {time_combined:.3f} ms")
    print(f"Memory efficiency:          {theoretical_min_ms / time_combined * 100:.1f}%")

    # Test if fused kernel could help - compare memory-bound vs compute-bound
    print("\n--- Fusion Potential Analysis ---")

    # With fusion, GELU intermediate doesn't go to DRAM
    fused_bytes = mlp1_bytes + mlp2_bytes - gelu_bytes  # Save GELU round-trip
    fused_theoretical_ms = (fused_bytes / 1e9) / mem_bandwidth_gbps * 1000

    time_combined / fused_theoretical_ms
    actual_possible_speedup = (total_bytes) / (fused_bytes)

    print(f"Without fusion: {total_bytes / 1e6:.2f} MB data movement")
    print(f"With fusion:    {fused_bytes / 1e6:.2f} MB data movement")
    print(f"Theoretical max speedup from fusion: {actual_possible_speedup:.2f}x")
    print("Note: GELU is a small fraction of total time")

    # Test GELU as fraction of total
    print("\n--- Op Time Breakdown ---")
    print(f"MLP1: {time_mlp1 / total_breakdown * 100:.1f}%")
    print(f"GELU: {time_gelu / total_breakdown * 100:.1f}%")
    print(f"MLP2: {time_mlp2 / total_breakdown * 100:.1f}%")


if __name__ == "__main__":
    main()
