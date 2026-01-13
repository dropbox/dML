#!/usr/bin/env python3
"""
Isolate which part of nn.MultiheadAttention causes the race.

nn.MultiheadAttention internally does:
1. QKV in_projection (Linear)
2. Reshape to (B*H, L, D) format
3. F.scaled_dot_product_attention
4. Reshape back
5. out_projection (Linear)

Since F.scaled_dot_product_attention passes alone, the bug is in the
projection or reshape operations around it.
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
        inputs = [input_factory(tid).to("mps") for tid in range(num_threads)]

        expected = []
        with torch.no_grad():
            for inp in inputs:
                expected.append(op(inp).clone())
        torch.mps.synchronize()

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
            all_failures.append(f"iter {it}: {len(errors)} errors")
            continue

        iteration_ok = True
        for tid in range(num_threads):
            if results[tid] is None:
                iteration_ok = False
                continue
            diff = (results[tid] - expected[tid]).abs().max().item()
            max_diff_overall = max(max_diff_overall, diff)
            if diff > tolerance:
                iteration_ok = False
                all_failures.append(f"iter {it} tid {tid}: diff={diff:.2e}")

        if iteration_ok:
            passed += 1

    return TestResult(name=name, passed=passed, total=iterations, max_diff=max_diff_overall, failures=all_failures[:5])


def main():
    parser = argparse.ArgumentParser(description="Isolate failing MHA internal")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        print("SKIP: MPS not available")
        return 0

    print("=" * 70)
    print("MultiheadAttention Internals Isolation Test")
    print(f"Threads: {args.threads}, Iterations: {args.iterations}, Tolerance: {args.tolerance}")
    print("=" * 70)

    batch_size = 4
    seq_len = 128
    embed_dim = 256
    num_heads = 4
    head_dim = embed_dim // num_heads

    def make_input(tid):
        torch.manual_seed(tid * 1000 + 42)
        return torch.randn(batch_size, seq_len, embed_dim)

    results = []

    # Create MHA module for weight access
    mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True).to("mps").eval()
    torch.mps.synchronize()

    # Test 1: QKV projection alone (packed linear)
    print("\nTesting: QKV in_projection (packed Linear)...")
    in_proj_weight = mha.in_proj_weight.clone()
    in_proj_bias = mha.in_proj_bias.clone() if mha.in_proj_bias is not None else None

    def qkv_projection(x):
        return F.linear(x, in_proj_weight, in_proj_bias)

    result = run_parallel_test("QKV projection", qkv_projection, make_input, args.threads, args.iterations, args.tolerance)
    results.append(result)
    print(f"  {'PASS' if result.passed == result.total else 'FAIL'}: {result.passed}/{result.total}")

    # Test 2: Out projection alone (single linear)
    print("\nTesting: Out projection (Linear)...")
    out_proj_weight = mha.out_proj.weight.clone()
    out_proj_bias = mha.out_proj.bias.clone() if mha.out_proj.bias is not None else None

    def out_projection(x):
        return F.linear(x, out_proj_weight, out_proj_bias)

    result = run_parallel_test("Out projection", out_projection, make_input, args.threads, args.iterations, args.tolerance)
    results.append(result)
    print(f"  {'PASS' if result.passed == result.total else 'FAIL'}: {result.passed}/{result.total}")

    # Test 3: QKV projection + head reshape (the strided view)
    print("\nTesting: QKV projection + head reshape...")

    def qkv_with_reshape(x):
        # Project
        qkv = F.linear(x, in_proj_weight, in_proj_bias)  # (B, L, 3*E)
        # Split
        q, k, v = qkv.chunk(3, dim=-1)  # Each: (B, L, E)
        # Reshape to (B, H, L, D)
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        # Return just q for simplicity
        return q

    result = run_parallel_test("QKV proj + reshape", qkv_with_reshape, make_input, args.threads, args.iterations, args.tolerance)
    results.append(result)
    print(f"  {'PASS' if result.passed == result.total else 'FAIL'}: {result.passed}/{result.total}")

    # Test 4: QKV + SDPA (no out projection)
    print("\nTesting: QKV + SDPA (no out proj)...")

    def qkv_sdpa(x):
        # Project
        qkv = F.linear(x, in_proj_weight, in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        # Reshape
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        # SDPA
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        return out

    result = run_parallel_test("QKV + SDPA", qkv_sdpa, make_input, args.threads, args.iterations, args.tolerance)
    results.append(result)
    print(f"  {'PASS' if result.passed == result.total else 'FAIL'}: {result.passed}/{result.total}")

    # Test 5: QKV + SDPA + reshape back (before out proj)
    print("\nTesting: QKV + SDPA + reshape back...")

    def qkv_sdpa_reshape(x):
        qkv = F.linear(x, in_proj_weight, in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        # Reshape back: (B, H, L, D) -> (B, L, E)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return out

    result = run_parallel_test("QKV + SDPA + reshape", qkv_sdpa_reshape, make_input, args.threads, args.iterations, args.tolerance)
    results.append(result)
    print(f"  {'PASS' if result.passed == result.total else 'FAIL'}: {result.passed}/{result.total}")

    # Test 6: Full manual MHA (QKV + SDPA + reshape + out_proj)
    print("\nTesting: Full manual MHA (QKV + SDPA + out)...")

    def full_manual_mha(x):
        qkv = F.linear(x, in_proj_weight, in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        out = F.linear(out, out_proj_weight, out_proj_bias)
        return out

    result = run_parallel_test("Full manual MHA", full_manual_mha, make_input, args.threads, args.iterations, args.tolerance)
    results.append(result)
    print(f"  {'PASS' if result.passed == result.total else 'FAIL'}: {result.passed}/{result.total}")

    # Test 7: nn.MultiheadAttention (PyTorch's implementation)
    print("\nTesting: nn.MultiheadAttention...")

    def nn_mha(x):
        out, _ = mha(x, x, x, need_weights=False)
        return out

    result = run_parallel_test("nn.MultiheadAttention", nn_mha, make_input, args.threads, args.iterations, args.tolerance)
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
        print("All components passed! PyTorch's MHA implementation may have")
        print("special code paths that trigger the race.")
    else:
        print("Components with race conditions:")
        for r in failing:
            rate = (r.total - r.passed) / r.total * 100
            print(f"  - {r.name}: {rate:.0f}% failure rate")

    if passing:
        print("\nComponents that passed:")
        for r in passing:
            print(f"  - {r.name}")

    # If manual MHA passes but nn.MHA fails, the issue is in PyTorch's
    # specific implementation (e.g., C++ codepath, memory layout, etc.)
    manual_result = next((r for r in results if r.name == "Full manual MHA"), None)
    nn_result = next((r for r in results if r.name == "nn.MultiheadAttention"), None)

    if manual_result and nn_result:
        print("\n" + "=" * 70)
        print("KEY FINDING")
        print("=" * 70)
        if manual_result.passed == manual_result.total and nn_result.passed < nn_result.total:
            print("Manual MHA passes, but nn.MultiheadAttention fails!")
            print("This means the bug is in PyTorch's specific MHA implementation,")
            print("likely in the C++ backend or memory layout choices.")
        elif manual_result.passed < manual_result.total:
            print("Manual MHA also fails - bug is in the composition of operations.")
        else:
            print("Both pass - need more iterations to trigger race.")

    return 0 if not failing else 1


if __name__ == "__main__":
    sys.exit(main())
