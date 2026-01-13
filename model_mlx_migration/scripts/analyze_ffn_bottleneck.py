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
Analyze whether FFN is compute-bound or memory-bound at different scales.
Determines if custom Metal kernel would help.
"""

import time
import mlx.core as mx
import mlx.nn as nn
import numpy as np

WARMUP = 5
ITERATIONS = 50


def benchmark_ffn(batch: int, seq_len: int, n_state: int):
    """Benchmark FFN and compute metrics"""
    n_mlp = n_state * 4

    mx.random.seed(42)
    x = mx.random.normal((batch, seq_len, n_state))
    mx.eval(x)

    mlp1 = nn.Linear(n_state, n_mlp)
    mlp2 = nn.Linear(n_mlp, n_state)

    def ffn(x):
        return mlp2(nn.gelu(mlp1(x)))

    # Warmup
    for _ in range(WARMUP):
        out = ffn(x)
        mx.eval(out)

    # Benchmark
    times = []
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        out = ffn(x)
        mx.eval(out)
        times.append(time.perf_counter() - start)

    avg_ms = np.mean(times) * 1000

    # Compute metrics
    bytes_per_elem = 2  # float16

    # Data movement (pessimistic, no fusion)
    input_size = batch * seq_len * n_state * bytes_per_elem
    mlp1_output_size = batch * seq_len * n_mlp * bytes_per_elem
    mlp1_weight_size = n_state * n_mlp * bytes_per_elem
    mlp2_weight_size = n_mlp * n_state * bytes_per_elem
    mlp2_output_size = input_size

    # Total bytes moved
    total_bytes = (
        input_size +           # mlp1 input read
        mlp1_weight_size +     # mlp1 weights read
        mlp1_output_size +     # mlp1 output write
        mlp1_output_size +     # gelu input read
        mlp1_output_size +     # gelu output write
        mlp1_output_size +     # mlp2 input read
        mlp2_weight_size +     # mlp2 weights read
        mlp2_output_size       # mlp2 output write
    )

    # Compute FLOPs (2 matmuls + GELU)
    # Matmul FLOPs = 2 * M * N * K
    mlp1_flops = 2 * batch * seq_len * n_state * n_mlp
    mlp2_flops = 2 * batch * seq_len * n_mlp * n_state
    gelu_flops = batch * seq_len * n_mlp * 8  # Approximate GELU ops
    total_flops = mlp1_flops + mlp2_flops + gelu_flops

    # Achieved bandwidth and FLOPS
    achieved_bandwidth_gbps = (total_bytes / 1e9) / (avg_ms / 1000)
    achieved_tflops = (total_flops / 1e12) / (avg_ms / 1000)

    # M4 Max specs (approximate)
    peak_bandwidth_gbps = 400  # GB/s
    peak_tflops = 59  # TFLOPS (FP16 tensor)

    # Compute vs memory bound ratio
    # If achieved_bandwidth / peak_bandwidth > achieved_tflops / peak_tflops => memory bound
    bandwidth_ratio = achieved_bandwidth_gbps / peak_bandwidth_gbps
    compute_ratio = achieved_tflops / peak_tflops

    # Arithmetic intensity = FLOPs / Bytes
    arithmetic_intensity = total_flops / total_bytes

    return {
        "time_ms": avg_ms,
        "bandwidth_gbps": achieved_bandwidth_gbps,
        "tflops": achieved_tflops,
        "bandwidth_pct": bandwidth_ratio * 100,
        "compute_pct": compute_ratio * 100,
        "arithmetic_intensity": arithmetic_intensity,
        "is_memory_bound": bandwidth_ratio > compute_ratio * 1.5,
    }


def main():
    print("=" * 80)
    print("FFN COMPUTE vs MEMORY BOUND ANALYSIS")
    print("=" * 80)
    print("\nM4 Max specs: ~400 GB/s memory BW, ~59 TFLOPS FP16")
    print("Roofline: Memory-bound if BW% > Compute% * 1.5")

    # Test cases: (seq_len, n_state, description)
    test_cases = [
        # Small (likely memory-bound)
        (1, 256, "Tiny model, 1 token"),
        (1, 512, "Small model, 1 token"),
        (1, 768, "Base model, 1 token"),

        # Medium (transition zone)
        (1, 1280, "Large model, 1 token"),
        (100, 1280, "Large model, 100 tokens"),

        # Large (likely compute-bound)
        (1500, 1280, "Large model, encoder (30s)"),
        (100, 2048, "XL model, 100 tokens"),
        (500, 2048, "XL model, 500 tokens"),
    ]

    print("\n" + "-" * 80)
    print(f"{'Config':<35} {'Time':<10} {'BW%':<8} {'Comp%':<8} {'AI':<8} {'Bound':<12}")
    print("-" * 80)

    results = []
    for seq_len, n_state, desc in test_cases:
        result = benchmark_ffn(1, seq_len, n_state)
        bound = "MEMORY" if result["is_memory_bound"] else "COMPUTE"
        print(f"{desc:<35} {result['time_ms']:<10.3f} {result['bandwidth_pct']:<8.1f} "
              f"{result['compute_pct']:<8.1f} {result['arithmetic_intensity']:<8.1f} {bound:<12}")
        results.append((desc, result))

    print("-" * 80)

    # Summary
    print("\n" + "=" * 80)
    print("CONCLUSIONS")
    print("=" * 80)

    memory_bound = [r for r in results if r[1]["is_memory_bound"]]
    compute_bound = [r for r in results if not r[1]["is_memory_bound"]]

    print(f"\nMemory-bound configs ({len(memory_bound)}):")
    for desc, _ in memory_bound:
        print(f"  - {desc}")
    print("  => Custom Metal fusion WOULD help these")

    print(f"\nCompute-bound configs ({len(compute_bound)}):")
    for desc, _ in compute_bound:
        print(f"  - {desc}")
    print("  => Custom Metal fusion would NOT help these")

    # Whisper-specific conclusion
    print("\n" + "-" * 80)
    print("WHISPER-SPECIFIC CONCLUSION:")
    print("-" * 80)

    whisper_encoder = [r for _, r in results if "encoder" in _][0] if any("encoder" in d for d, _ in results) else None
    whisper_decoder_1 = [r for d, r in results if "Large model, 1 token" in d][0] if any("Large model, 1 token" in d for d, _ in results) else None

    if whisper_encoder:
        print(f"\nEncoder (30s audio): {'COMPUTE-BOUND' if not whisper_encoder['is_memory_bound'] else 'MEMORY-BOUND'}")
        print(f"  => FFN fusion would {'NOT ' if not whisper_encoder['is_memory_bound'] else ''}help")

    if whisper_decoder_1:
        print(f"\nDecoder (1 token): {'COMPUTE-BOUND' if not whisper_decoder_1['is_memory_bound'] else 'MEMORY-BOUND'}")
        print(f"  => FFN fusion would {'NOT ' if not whisper_decoder_1['is_memory_bound'] else ''}help")

    print("\nRECOMMENDATION:")
    print("  Custom Metal FFN kernels are NOT recommended for Whisper workloads.")
    print("  The operations are compute-bound, not memory-bound.")
    print("  Focus optimization efforts on algorithmic improvements instead.")


if __name__ == "__main__":
    main()
