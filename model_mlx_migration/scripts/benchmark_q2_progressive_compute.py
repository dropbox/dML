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
Benchmark Q2: DiT Block Progressive Compute

Tests the hypothesis that running early DiT blocks (0-10) in BF16
and later blocks (11-21) in FP32 provides speedup.

Theory:
- Early blocks do coarse flow estimation (can tolerate lower precision)
- Later blocks refine details (need full precision)
- BF16 matmuls are faster than FP32 on Apple Silicon

Worker #1333, 2025-12-20
"""

import time
from dataclasses import dataclass
from typing import List

import mlx.core as mx
import mlx.nn as nn


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    name: str
    latency_ms: float
    dtype_info: str


def create_dummy_dit_block(dim: int = 1024, heads: int = 16, dtype=mx.float32):
    """Create a simplified DiT block for benchmarking."""
    class SimpleDiTBlock(nn.Module):
        def __init__(self, dim, heads, dtype):
            super().__init__()
            self.dim = dim
            self.heads = heads
            self.dim_head = dim // heads
            self.dtype = dtype

            # QKV projection
            self.qkv = nn.Linear(dim, dim * 3, bias=True)
            # Output projection
            self.proj_out = nn.Linear(dim, dim, bias=True)
            # FFN
            self.ffn1 = nn.Linear(dim, dim * 4, bias=True)
            self.ffn2 = nn.Linear(dim * 4, dim, bias=True)

        def __call__(self, x: mx.array, t_emb: mx.array) -> mx.array:
            B, L, D = x.shape

            # Cast to block dtype
            x = x.astype(self.dtype)
            t_emb = t_emb.astype(self.dtype)

            # Self-attention
            qkv = self.qkv(x)
            q, k, v = mx.split(qkv, 3, axis=-1)

            # Reshape for attention
            q = q.reshape(B, L, self.heads, self.dim_head).transpose(0, 2, 1, 3)
            k = k.reshape(B, L, self.heads, self.dim_head).transpose(0, 2, 1, 3)
            v = v.reshape(B, L, self.heads, self.dim_head).transpose(0, 2, 1, 3)

            # Attention
            attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.dim_head ** -0.5)
            attn = attn.transpose(0, 2, 1, 3).reshape(B, L, D)

            # Residual
            x = x + self.proj_out(attn)

            # FFN
            h = self.ffn1(x)
            h = nn.gelu(h)
            h = self.ffn2(h)
            x = x + h

            return x

    return SimpleDiTBlock(dim, heads, dtype)


def benchmark_uniform_precision(
    blocks: List[nn.Module],
    x: mx.array,
    t_emb: mx.array,
    num_runs: int = 10,
    warmup: int = 3
) -> float:
    """Benchmark with uniform precision across all blocks."""
    # Warmup
    for _ in range(warmup):
        h = x
        for block in blocks:
            h = block(h, t_emb)
        mx.eval(h)

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        h = x
        for block in blocks:
            h = block(h, t_emb)
        mx.eval(h)
        times.append((time.perf_counter() - start) * 1000)

    return sum(times) / len(times)


def benchmark_progressive_precision(
    bf16_blocks: List[nn.Module],
    fp32_blocks: List[nn.Module],
    x: mx.array,
    t_emb: mx.array,
    num_runs: int = 10,
    warmup: int = 3
) -> float:
    """Benchmark with progressive precision (BF16 early, FP32 later)."""
    # Warmup
    for _ in range(warmup):
        h = x
        # BF16 early blocks
        for block in bf16_blocks:
            h = block(h, t_emb)
        # Cast back to FP32 for later blocks
        h = h.astype(mx.float32)
        # FP32 later blocks
        for block in fp32_blocks:
            h = block(h, t_emb)
        mx.eval(h)

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        h = x
        # BF16 early blocks
        for block in bf16_blocks:
            h = block(h, t_emb)
        # Cast back to FP32
        h = h.astype(mx.float32)
        # FP32 later blocks
        for block in fp32_blocks:
            h = block(h, t_emb)
        mx.eval(h)
        times.append((time.perf_counter() - start) * 1000)

    return sum(times) / len(times)


def test_bf16_availability():
    """Check if BF16 operations work correctly on this system."""
    print("=" * 60)
    print("BF16 AVAILABILITY CHECK")
    print("=" * 60)

    # Test basic BF16 matmul
    try:
        a = mx.random.normal((32, 64)).astype(mx.bfloat16)
        b = mx.random.normal((64, 32)).astype(mx.bfloat16)
        c = a @ b
        mx.eval(c)
        print(f"✓ BF16 matmul works: shape={c.shape}, dtype={c.dtype}")
    except Exception as e:
        print(f"✗ BF16 matmul failed: {e}")
        return False

    # Test BF16 vs FP32 speed
    sizes = [(512, 1024), (1024, 2048)]
    for m, n in sizes:
        a_fp32 = mx.random.normal((m, n))
        b_fp32 = mx.random.normal((n, m))
        a_bf16 = a_fp32.astype(mx.bfloat16)
        b_bf16 = b_fp32.astype(mx.bfloat16)

        # Warmup
        for _ in range(3):
            mx.eval(a_fp32 @ b_fp32)
            mx.eval(a_bf16 @ b_bf16)

        # Benchmark
        times_fp32 = []
        times_bf16 = []
        for _ in range(10):
            start = time.perf_counter()
            c = a_fp32 @ b_fp32
            mx.eval(c)
            times_fp32.append(time.perf_counter() - start)

            start = time.perf_counter()
            c = a_bf16 @ b_bf16
            mx.eval(c)
            times_bf16.append(time.perf_counter() - start)

        fp32_ms = sum(times_fp32) / len(times_fp32) * 1000
        bf16_ms = sum(times_bf16) / len(times_bf16) * 1000
        speedup = fp32_ms / bf16_ms

        print(f"  Matmul {m}x{n}: FP32={fp32_ms:.3f}ms, BF16={bf16_ms:.3f}ms, BF16 speedup={speedup:.2f}x")

    return True


def run_q2_benchmark():
    """Run the Q2 progressive compute benchmark."""
    print("\n" + "=" * 60)
    print("Q2: DiT PROGRESSIVE COMPUTE BENCHMARK")
    print("=" * 60)

    # Configuration matching CosyVoice3 DiT
    dim = 1024
    heads = 16
    depth = 22
    batch_size = 1
    seq_len = 100  # 50 tokens * 2

    print("\nConfiguration:")
    print(f"  dim={dim}, heads={heads}, depth={depth}")
    print(f"  batch_size={batch_size}, seq_len={seq_len}")

    # Create test inputs
    mx.random.seed(42)
    x = mx.random.normal((batch_size, seq_len, dim))
    t_emb = mx.random.normal((batch_size, dim))

    # Create blocks
    print(f"\nCreating {depth} DiT blocks...")

    # Config 1: All FP32 (baseline)
    fp32_blocks = [create_dummy_dit_block(dim, heads, mx.float32) for _ in range(depth)]

    # Config 2: All BF16
    bf16_blocks = [create_dummy_dit_block(dim, heads, mx.bfloat16) for _ in range(depth)]

    # Config 3: Progressive (first 11 BF16, last 11 FP32)
    split_point = depth // 2
    early_bf16_blocks = [create_dummy_dit_block(dim, heads, mx.bfloat16) for _ in range(split_point)]
    later_fp32_blocks = [create_dummy_dit_block(dim, heads, mx.float32) for _ in range(depth - split_point)]

    print("\nRunning benchmarks...")

    # Benchmark 1: All FP32
    print("\n  [1/3] Baseline: All FP32")
    latency_fp32 = benchmark_uniform_precision(fp32_blocks, x, t_emb)
    print(f"        Latency: {latency_fp32:.2f}ms")

    # Benchmark 2: All BF16
    print("\n  [2/3] All BF16")
    latency_bf16 = benchmark_uniform_precision(bf16_blocks, x, t_emb)
    speedup_bf16 = latency_fp32 / latency_bf16
    print(f"        Latency: {latency_bf16:.2f}ms")
    print(f"        Speedup: {speedup_bf16:.2f}x vs FP32")

    # Benchmark 3: Progressive
    print(f"\n  [3/3] Progressive: First {split_point} BF16, Last {depth - split_point} FP32")
    latency_progressive = benchmark_progressive_precision(
        early_bf16_blocks, later_fp32_blocks, x, t_emb
    )
    speedup_progressive = latency_fp32 / latency_progressive
    print(f"        Latency: {latency_progressive:.2f}ms")
    print(f"        Speedup: {speedup_progressive:.2f}x vs FP32")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print("\n| Configuration | Latency (ms) | Speedup vs FP32 |")
    print("|---------------|--------------|-----------------|")
    print(f"| All FP32 (baseline) | {latency_fp32:.2f} | 1.00x |")
    print(f"| All BF16 | {latency_bf16:.2f} | {speedup_bf16:.2f}x |")
    print(f"| Progressive (Q2) | {latency_progressive:.2f} | {speedup_progressive:.2f}x |")

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)

    # Determine if Q2 is worth it
    if speedup_progressive > 1.05:
        print(f"\n✓ Q2 POTENTIALLY VIABLE: {speedup_progressive:.2f}x speedup")
        print("  Note: Requires quality validation (near-lossless, not lossless)")
    elif speedup_progressive > 1.01:
        print(f"\n⚠ Q2 MARGINAL: {speedup_progressive:.2f}x speedup")
        print("  Benefit too small to justify code complexity")
    else:
        print(f"\n✗ Q2 NOT WORTH: {speedup_progressive:.2f}x (no meaningful speedup)")
        print("  BF16 precision change provides no compute benefit on this hardware")

    # Additional notes
    if speedup_bf16 < 1.1:
        print("\n  Note: BF16 matmuls show minimal speedup on this hardware.")
        print("  Apple Silicon GPU may not have dedicated BF16 units.")

    return {
        'fp32_ms': latency_fp32,
        'bf16_ms': latency_bf16,
        'progressive_ms': latency_progressive,
        'speedup_bf16': speedup_bf16,
        'speedup_progressive': speedup_progressive
    }


if __name__ == "__main__":
    # First check BF16 availability
    bf16_available = test_bf16_availability()

    if bf16_available:
        # Run the main Q2 benchmark
        results = run_q2_benchmark()
    else:
        print("\n✗ BF16 not available, cannot benchmark Q2")
