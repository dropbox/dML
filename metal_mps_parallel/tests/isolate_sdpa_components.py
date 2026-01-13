#!/usr/bin/env python3
"""
Isolate which SDPA component causes the parallel race condition.

SDPA consists of:
1. QKV projections (Linear layers)
2. Attention scores = Q @ K.T / sqrt(d_k)  [matmul + scale]
3. Softmax
4. Value aggregation = softmax @ V  [matmul]

Goal: Identify the SMALLEST component that reproduces the race.
"""

import argparse
import os
import threading
import sys
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

os.environ["MPS_FORCE_GRAPH_PATH"] = "1"


@dataclass
class TestResult:
    name: str
    passed: int
    total: int
    max_diff: float
    failures: list


def run_parallel_test(
    name: str,
    op: Callable[[torch.Tensor], torch.Tensor],
    input_factory: Callable[[int], torch.Tensor],
    num_threads: int = 8,
    iterations: int = 30,
    tolerance: float = 1e-3,
) -> TestResult:
    """Run a parallel test on the given operation."""
    torch.mps.synchronize()

    passed = 0
    max_diff_overall = 0.0
    all_failures = []

    for it in range(iterations):
        # Create fresh inputs each iteration
        inputs = [input_factory(tid).to("mps") for tid in range(num_threads)]

        # Compute golden outputs sequentially
        expected = []
        with torch.no_grad():
            for inp in inputs:
                expected.append(op(inp).clone())
        torch.mps.synchronize()

        # Run in parallel
        results = [None] * num_threads
        errors = []

        def worker(tid):
            try:
                with torch.no_grad():
                    results[tid] = op(inputs[tid])
                torch.mps.synchronize()
            except Exception as e:
                errors.append((tid, e))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if errors:
            all_failures.append(f"iter {it}: {len(errors)} errors: {errors[0]}")
            continue

        # Check results
        iteration_ok = True
        for tid in range(num_threads):
            if results[tid] is None:
                iteration_ok = False
                all_failures.append(f"iter {it} tid {tid}: None")
                continue
            diff = (results[tid] - expected[tid]).abs().max().item()
            max_diff_overall = max(max_diff_overall, diff)
            if diff > tolerance:
                iteration_ok = False
                all_failures.append(f"iter {it} tid {tid}: diff={diff:.2e}")

        if iteration_ok:
            passed += 1

    return TestResult(
        name=name,
        passed=passed,
        total=iterations,
        max_diff=max_diff_overall,
        failures=all_failures[:5],
    )


def main():
    parser = argparse.ArgumentParser(description="Isolate failing SDPA component")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        print("SKIP: MPS not available")
        return 0

    print("=" * 70)
    print("SDPA Component Isolation Test")
    print(f"Threads: {args.threads}, Iterations: {args.iterations}, Tolerance: {args.tolerance}")
    print("=" * 70)

    # Dimensions matching typical attention
    batch_size = 4
    seq_len = 128
    embed_dim = 256
    num_heads = 4
    head_dim = embed_dim // num_heads

    def make_input_3d(tid):
        """Input shape: (batch, seq, embed)"""
        torch.manual_seed(tid * 1000 + 42)
        return torch.randn(batch_size, seq_len, embed_dim)

    def make_input_4d(tid):
        """Input shape: (batch, heads, seq, head_dim) - attention format"""
        torch.manual_seed(tid * 1000 + 42)
        return torch.randn(batch_size, num_heads, seq_len, head_dim)

    def make_attn_scores(tid):
        """Input shape: (batch, heads, seq, seq) - attention scores"""
        torch.manual_seed(tid * 1000 + 42)
        return torch.randn(batch_size, num_heads, seq_len, seq_len)

    results = []

    # Test 1: Large matmul alone (QK^T style)
    print("\nTesting: Large matmul (Q @ K.T style)...")
    def matmul_qk(x):
        # x is (B, H, L, D), compute (B, H, L, L)
        return torch.matmul(x, x.transpose(-2, -1))
    result = run_parallel_test("Matmul Q@K.T", matmul_qk, make_input_4d, args.threads, args.iterations, args.tolerance)
    results.append(result)
    print(f"  {'PASS' if result.passed == result.total else 'FAIL'}: {result.passed}/{result.total}")

    # Test 2: Softmax on attention scores
    print("\nTesting: Softmax on attention scores...")
    def softmax_attn(x):
        return F.softmax(x, dim=-1)
    result = run_parallel_test("Softmax (attn)", softmax_attn, make_attn_scores, args.threads, args.iterations, args.tolerance)
    results.append(result)
    print(f"  {'PASS' if result.passed == result.total else 'FAIL'}: {result.passed}/{result.total}")

    # Test 3: Scale operation (1/sqrt(d_k))
    print("\nTesting: Scale (1/sqrt(d_k))...")
    scale = 1.0 / math.sqrt(head_dim)
    def scale_op(x):
        return x * scale
    result = run_parallel_test("Scale op", scale_op, make_attn_scores, args.threads, args.iterations, args.tolerance)
    results.append(result)
    print(f"  {'PASS' if result.passed == result.total else 'FAIL'}: {result.passed}/{result.total}")

    # Test 4: Matmul + Scale (Q @ K.T / sqrt(d_k))
    print("\nTesting: Matmul + Scale (Q @ K.T / sqrt(d))...")
    def matmul_scale(x):
        scores = torch.matmul(x, x.transpose(-2, -1))
        return scores * scale
    result = run_parallel_test("Matmul+Scale", matmul_scale, make_input_4d, args.threads, args.iterations, args.tolerance)
    results.append(result)
    print(f"  {'PASS' if result.passed == result.total else 'FAIL'}: {result.passed}/{result.total}")

    # Test 5: Full attention score path (Q @ K.T / sqrt(d) + softmax)
    print("\nTesting: Full attention scores (matmul + scale + softmax)...")
    def full_attn_scores(x):
        scores = torch.matmul(x, x.transpose(-2, -1)) * scale
        return F.softmax(scores, dim=-1)
    result = run_parallel_test("Full attn scores", full_attn_scores, make_input_4d, args.threads, args.iterations, args.tolerance)
    results.append(result)
    print(f"  {'PASS' if result.passed == result.total else 'FAIL'}: {result.passed}/{result.total}")

    # Test 6: Attention output (softmax @ V)
    print("\nTesting: Attention output (attn @ V)...")
    # Need both attention scores and values
    def make_attn_and_v(tid):
        torch.manual_seed(tid * 1000 + 42)
        # Return concatenated tensor - will split in op
        attn = torch.randn(batch_size, num_heads, seq_len, seq_len)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim)
        # Combine into one tensor for simpler API (hacky but works)
        # Flatten and concatenate
        attn_flat = attn.reshape(batch_size, num_heads, -1)  # (B, H, L*L)
        v_flat = v.reshape(batch_size, num_heads, -1)  # (B, H, L*D)
        combined = torch.cat([attn_flat, v_flat], dim=-1)  # (B, H, L*L + L*D)
        return combined

    def attn_output(combined):
        # Split back
        attn_flat = combined[..., :seq_len * seq_len]
        v_flat = combined[..., seq_len * seq_len:]
        attn = attn_flat.reshape(batch_size, num_heads, seq_len, seq_len)
        v = v_flat.reshape(batch_size, num_heads, seq_len, head_dim)
        attn_probs = F.softmax(attn, dim=-1)
        return torch.matmul(attn_probs, v)

    result = run_parallel_test("Attn output (softmax@V)", attn_output, make_attn_and_v, args.threads, args.iterations, args.tolerance)
    results.append(result)
    print(f"  {'PASS' if result.passed == result.total else 'FAIL'}: {result.passed}/{result.total}")

    # Test 7: F.scaled_dot_product_attention directly
    print("\nTesting: F.scaled_dot_product_attention...")
    def make_qkv(tid):
        torch.manual_seed(tid * 1000 + 42)
        q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim)
        # Stack along a new dimension
        return torch.stack([q, k, v], dim=0)  # (3, B, H, L, D)

    def sdpa_direct(qkv):
        q, k, v = qkv[0], qkv[1], qkv[2]
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

    result = run_parallel_test("F.scaled_dot_product_attention", sdpa_direct, make_qkv, args.threads, args.iterations, args.tolerance)
    results.append(result)
    print(f"  {'PASS' if result.passed == result.total else 'FAIL'}: {result.passed}/{result.total}")

    # Test 8: nn.MultiheadAttention (full)
    print("\nTesting: nn.MultiheadAttention...")
    mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True).to("mps").eval()
    torch.mps.synchronize()

    def mha_forward(x):
        out, _ = mha(x, x, x, need_weights=False)
        return out

    result = run_parallel_test("nn.MultiheadAttention", mha_forward, make_input_3d, args.threads, args.iterations, args.tolerance)
    results.append(result)
    print(f"  {'PASS' if result.passed == result.total else 'FAIL'}: {result.passed}/{result.total}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Component':<35} {'Pass/Total':<15} {'Max Diff':<12} Status")
    print("-" * 70)

    for r in results:
        status = "OK" if r.passed == r.total else "RACE"
        print(f"{r.name:<35} {r.passed}/{r.total:<12} {r.max_diff:<12.2e} {status}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    failing = [r for r in results if r.passed < r.total]
    passing = [r for r in results if r.passed == r.total]

    if not failing:
        print("All components passed! Race may be in combination or intermittent.")
        print("Try running with more iterations.")
    else:
        print("Components with race conditions:")
        for r in failing:
            failure_rate = (r.total - r.passed) / r.total * 100
            print(f"  - {r.name}: {failure_rate:.0f}% failure rate")
            if r.failures:
                for f in r.failures[:2]:
                    print(f"    {f}")

    if passing:
        print("\nComponents that passed:")
        for r in passing:
            print(f"  - {r.name}")

    return 0 if not failing else 1


if __name__ == "__main__":
    sys.exit(main())
