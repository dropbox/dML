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
Fused FFN Benchmark - Test whether mx.compile fuses Linear + GELU + Linear

Tests:
1. Unfused FFN (3 separate ops)
2. mx.compile'd FFN (potential automatic fusion)
3. Custom Metal kernel (if needed)

Expected outcome: mx.compile may already fuse these ops, providing 10-15% speedup
"""

import time
import mlx.core as mx
import mlx.nn as nn
import numpy as np

# Warmup iterations
WARMUP = 5
# Benchmark iterations
ITERATIONS = 100


class UnfusedFFN(nn.Module):
    """Standard unfused FFN: Linear -> GELU -> Linear"""
    def __init__(self, n_state: int, n_mlp: int):
        super().__init__()
        self.mlp1 = nn.Linear(n_state, n_mlp)
        self.mlp2 = nn.Linear(n_mlp, n_state)

    def __call__(self, x: mx.array) -> mx.array:
        return self.mlp2(nn.gelu(self.mlp1(x)))


class GatedFFN(nn.Module):
    """Gated FFN (LLaMA-style): Linear -> SILU * Gate -> Linear"""
    def __init__(self, n_state: int, n_mlp: int):
        super().__init__()
        self.gate = nn.Linear(n_state, n_mlp, bias=False)
        self.up = nn.Linear(n_state, n_mlp, bias=False)
        self.down = nn.Linear(n_mlp, n_state, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down(nn.silu(self.gate(x)) * self.up(x))


def benchmark_module(module, x, name: str, compiled: bool = False):
    """Run benchmark for a module"""
    if compiled:
        forward = mx.compile(module.__call__)
    else:
        forward = module.__call__

    # Warmup
    for _ in range(WARMUP):
        out = forward(x)
        mx.eval(out)

    # Benchmark
    times = []
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        out = forward(x)
        mx.eval(out)
        times.append(time.perf_counter() - start)

    avg_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000
    return avg_ms, std_ms


def test_numerical_equivalence():
    """Verify compiled and uncompiled produce identical results"""
    print("=" * 60)
    print("NUMERICAL EQUIVALENCE TEST")
    print("=" * 60)

    n_state = 1280  # Whisper large-v3
    n_mlp = n_state * 4
    batch = 1
    seq_len = 100

    mx.random.seed(42)
    x = mx.random.normal((batch, seq_len, n_state))

    # Test UnfusedFFN
    ffn = UnfusedFFN(n_state, n_mlp)

    out_unfused = ffn(x)
    mx.eval(out_unfused)

    compiled_fn = mx.compile(ffn.__call__)
    out_compiled = compiled_fn(x)
    mx.eval(out_compiled)

    diff = mx.abs(out_unfused - out_compiled)
    max_diff = mx.max(diff).item()
    mean_diff = mx.mean(diff).item()

    print("UnfusedFFN vs Compiled:")
    print(f"  Max diff:  {max_diff:.2e}")
    print(f"  Mean diff: {mean_diff:.2e}")
    print(f"  Status:    {'PASS' if max_diff < 1e-5 else 'FAIL'}")
    print()

    return max_diff < 1e-5


def benchmark_whisper_ffn():
    """Benchmark FFN with Whisper dimensions"""
    print("=" * 60)
    print("WHISPER FFN BENCHMARK (n_state=1280, n_mlp=5120)")
    print("=" * 60)

    n_state = 1280  # Whisper large-v3
    n_mlp = n_state * 4

    # Test different sequence lengths
    test_cases = [
        (1, 1, "Decoder (1 token)"),
        (1, 100, "Decoder (100 tokens)"),
        (1, 1500, "Encoder (30s audio)"),
    ]

    results = {}

    for batch, seq_len, desc in test_cases:
        print(f"\n{desc} - batch={batch}, seq_len={seq_len}")
        print("-" * 50)

        mx.random.seed(42)
        x = mx.random.normal((batch, seq_len, n_state))

        ffn = UnfusedFFN(n_state, n_mlp)

        # Unfused
        avg_unfused, std_unfused = benchmark_module(ffn, x, "Unfused")
        print(f"  Unfused:   {avg_unfused:.3f} ± {std_unfused:.3f} ms")

        # Compiled
        avg_compiled, std_compiled = benchmark_module(ffn, x, "Compiled", compiled=True)
        print(f"  Compiled:  {avg_compiled:.3f} ± {std_compiled:.3f} ms")

        speedup = avg_unfused / avg_compiled
        print(f"  Speedup:   {speedup:.2f}x")

        results[desc] = {
            "unfused_ms": avg_unfused,
            "compiled_ms": avg_compiled,
            "speedup": speedup
        }

    return results


def benchmark_gated_ffn():
    """Benchmark gated FFN (LLaMA-style)"""
    print("\n" + "=" * 60)
    print("GATED FFN BENCHMARK (LLaMA-style)")
    print("=" * 60)

    n_state = 1280
    n_mlp = n_state * 4

    test_cases = [
        (1, 1, "Decoder (1 token)"),
        (1, 100, "Decoder (100 tokens)"),
    ]

    for batch, seq_len, desc in test_cases:
        print(f"\n{desc} - batch={batch}, seq_len={seq_len}")
        print("-" * 50)

        mx.random.seed(42)
        x = mx.random.normal((batch, seq_len, n_state))

        ffn = GatedFFN(n_state, n_mlp)

        avg_unfused, std_unfused = benchmark_module(ffn, x, "Unfused")
        print(f"  Unfused:   {avg_unfused:.3f} ± {std_unfused:.3f} ms")

        avg_compiled, std_compiled = benchmark_module(ffn, x, "Compiled", compiled=True)
        print(f"  Compiled:  {avg_compiled:.3f} ± {std_compiled:.3f} ms")

        speedup = avg_unfused / avg_compiled
        print(f"  Speedup:   {speedup:.2f}x")


def benchmark_full_transformer_block():
    """Benchmark a full transformer block (attention + FFN)"""
    print("\n" + "=" * 60)
    print("FULL TRANSFORMER BLOCK BENCHMARK")
    print("=" * 60)

    n_state = 1280
    n_head = 20
    n_mlp = n_state * 4

    class TransformerBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn_ln = nn.LayerNorm(n_state)
            self.mlp_ln = nn.LayerNorm(n_state)
            # Simplified self-attention
            self.q = nn.Linear(n_state, n_state)
            self.k = nn.Linear(n_state, n_state)
            self.v = nn.Linear(n_state, n_state)
            self.out = nn.Linear(n_state, n_state)
            # FFN
            self.mlp1 = nn.Linear(n_state, n_mlp)
            self.mlp2 = nn.Linear(n_mlp, n_state)

            self.n_head = n_head
            self.head_dim = n_state // n_head

        def __call__(self, x):
            # Self-attention with residual
            h = mx.fast.layer_norm(x, self.attn_ln.weight, self.attn_ln.bias, eps=1e-5)
            B, L, _ = h.shape
            q = self.q(h).reshape(B, L, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
            k = self.k(h).reshape(B, L, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
            v = self.v(h).reshape(B, L, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
            attn_out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim**-0.5)
            attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, L, -1)
            x = x + self.out(attn_out)

            # FFN with residual
            h = mx.fast.layer_norm(x, self.mlp_ln.weight, self.mlp_ln.bias, eps=1e-5)
            x = x + self.mlp2(nn.gelu(self.mlp1(h)))

            return x

    test_cases = [
        (1, 100, "100 tokens"),
        (1, 500, "500 tokens"),
    ]

    for batch, seq_len, desc in test_cases:
        print(f"\n{desc} - batch={batch}, seq_len={seq_len}")
        print("-" * 50)

        mx.random.seed(42)
        x = mx.random.normal((batch, seq_len, n_state))

        block = TransformerBlock()

        avg_unfused, std_unfused = benchmark_module(block, x, "Unfused")
        print(f"  Unfused:   {avg_unfused:.3f} ± {std_unfused:.3f} ms")

        avg_compiled, std_compiled = benchmark_module(block, x, "Compiled", compiled=True)
        print(f"  Compiled:  {avg_compiled:.3f} ± {std_compiled:.3f} ms")

        speedup = avg_unfused / avg_compiled
        print(f"  Speedup:   {speedup:.2f}x")


def benchmark_memory_layout():
    """Test if memory layout affects FFN performance"""
    print("\n" + "=" * 60)
    print("MEMORY LAYOUT TEST (contiguous vs non-contiguous)")
    print("=" * 60)

    n_state = 1280
    n_mlp = n_state * 4
    batch = 1
    seq_len = 100

    mx.random.seed(42)
    x_contiguous = mx.random.normal((batch, seq_len, n_state))

    # Create non-contiguous tensor via transpose
    x_noncontig = mx.random.normal((batch, n_state, seq_len)).transpose(0, 2, 1)

    ffn = UnfusedFFN(n_state, n_mlp)

    print("\nContiguous input:")
    avg_contig, std_contig = benchmark_module(ffn, x_contiguous, "Contiguous")
    print(f"  Time: {avg_contig:.3f} ± {std_contig:.3f} ms")

    print("\nNon-contiguous input:")
    avg_noncontig, std_noncontig = benchmark_module(ffn, x_noncontig, "Non-contiguous")
    print(f"  Time: {avg_noncontig:.3f} ± {std_noncontig:.3f} ms")

    ratio = avg_noncontig / avg_contig
    print(f"\nNon-contiguous / Contiguous: {ratio:.2f}x")


def main():
    print("=" * 60)
    print("FUSED FFN BENCHMARK")
    print("=" * 60)
    print(f"Device: {mx.default_device()}")
    print(f"Warmup iterations: {WARMUP}")
    print(f"Benchmark iterations: {ITERATIONS}")
    print()

    # Numerical equivalence test
    if not test_numerical_equivalence():
        print("FAILED: Numerical equivalence test")
        return

    # Whisper FFN benchmark
    whisper_results = benchmark_whisper_ffn()

    # Gated FFN benchmark
    benchmark_gated_ffn()

    # Full transformer block
    benchmark_full_transformer_block()

    # Memory layout test
    benchmark_memory_layout()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nmx.compile FFN Fusion Results (Whisper dimensions):")
    print("-" * 50)
    for desc, result in whisper_results.items():
        status = "BENEFICIAL" if result["speedup"] > 1.05 else "NEUTRAL"
        print(f"  {desc}:")
        print(f"    Unfused:  {result['unfused_ms']:.3f} ms")
        print(f"    Compiled: {result['compiled_ms']:.3f} ms")
        print(f"    Speedup:  {result['speedup']:.2f}x ({status})")


if __name__ == "__main__":
    main()
