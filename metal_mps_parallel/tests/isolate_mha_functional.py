#!/usr/bin/env python3
"""
Test PyTorch's F.multi_head_attention_forward to pinpoint the bug.

Manual MHA passes but nn.MultiheadAttention fails.
This test checks if F.multi_head_attention_forward also fails.
"""

import argparse
import os
import threading
import sys
from dataclasses import dataclass

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


def run_parallel_test(name, op, input_factory, num_threads=8, iterations=30, tolerance=1e-3):
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
            all_failures.append(f"iter {it}: errors: {errors[0]}")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        print("SKIP: MPS not available")
        return 0

    print("=" * 70)
    print("F.multi_head_attention_forward vs Manual MHA")
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

    # Create MHA module for weight access
    mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True).to("mps").eval()
    torch.mps.synchronize()

    in_proj_weight = mha.in_proj_weight.clone()
    in_proj_bias = mha.in_proj_bias.clone() if mha.in_proj_bias is not None else None
    out_proj_weight = mha.out_proj.weight.clone()
    out_proj_bias = mha.out_proj.bias.clone() if mha.out_proj.bias is not None else None

    results = []

    # Test 1: Manual MHA with explicit SDPA call
    print("\nTesting: Manual MHA (explicit SDPA)...")

    def manual_mha_explicit(x):
        # This matches PyTorch's implementation more closely
        qkv = F.linear(x, in_proj_weight, in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return F.linear(out, out_proj_weight, out_proj_bias)

    result = run_parallel_test("Manual MHA (explicit)", manual_mha_explicit, make_input, args.threads, args.iterations, args.tolerance)
    results.append(result)
    print(f"  {'PASS' if result.passed == result.total else 'FAIL'}: {result.passed}/{result.total}")

    # Test 2: F.multi_head_attention_forward (PyTorch's low-level functional)
    print("\nTesting: F.multi_head_attention_forward...")

    def functional_mha(x):
        # Need to transpose for non-batch-first format
        # F.multi_head_attention_forward expects (L, B, E)
        x_t = x.transpose(0, 1)  # (L, B, E)
        out, _ = F.multi_head_attention_forward(
            x_t, x_t, x_t,
            embed_dim_to_check=embed_dim,
            num_heads=num_heads,
            in_proj_weight=in_proj_weight,
            in_proj_bias=in_proj_bias,
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.0,
            out_proj_weight=out_proj_weight,
            out_proj_bias=out_proj_bias,
            training=False,
            key_padding_mask=None,
            need_weights=False,
            attn_mask=None,
            use_separate_proj_weight=False,
            q_proj_weight=None,
            k_proj_weight=None,
            v_proj_weight=None,
        )
        return out.transpose(0, 1)  # Back to (B, L, E)

    result = run_parallel_test("F.multi_head_attention_forward", functional_mha, make_input, args.threads, args.iterations, args.tolerance)
    results.append(result)
    print(f"  {'PASS' if result.passed == result.total else 'FAIL'}: {result.passed}/{result.total}")

    # Test 3: nn.MultiheadAttention (original)
    print("\nTesting: nn.MultiheadAttention...")

    def nn_mha(x):
        out, _ = mha(x, x, x, need_weights=False)
        return out

    result = run_parallel_test("nn.MultiheadAttention", nn_mha, make_input, args.threads, args.iterations, args.tolerance)
    results.append(result)
    print(f"  {'PASS' if result.passed == result.total else 'FAIL'}: {result.passed}/{result.total}")

    # Test 4: Manual MHA with PyTorch's striding pattern
    # PyTorch uses a specific layout: (L, B*H, D) not (B, H, L, D)
    print("\nTesting: Manual MHA (PyTorch stride pattern)...")

    def manual_mha_pytorch_strides(x):
        # This uses PyTorch's internal striding: (L, B*H, D)
        qkv = F.linear(x, in_proj_weight, in_proj_bias)  # (B, L, 3*E)
        # PyTorch does: view(L, B, 3, H, D) then permute/reshape
        # Let me try their exact pattern
        qkv = qkv.transpose(0, 1)  # (L, B, 3*E)
        qkv = qkv.view(seq_len, batch_size, 3, num_heads, head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # Each: (L, B, H, D)
        # Reshape to (L, B*H, D)
        q = q.reshape(seq_len, batch_size * num_heads, head_dim)
        k = k.reshape(seq_len, batch_size * num_heads, head_dim)
        v = v.reshape(seq_len, batch_size * num_heads, head_dim)
        # Transpose for SDPA: (B*H, L, D) - then reshape to (B, H, L, D)
        q = q.transpose(0, 1).view(batch_size, num_heads, seq_len, head_dim)
        k = k.transpose(0, 1).view(batch_size, num_heads, seq_len, head_dim)
        v = v.transpose(0, 1).view(batch_size, num_heads, seq_len, head_dim)
        # SDPA
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        # Reshape back: (B, H, L, D) -> (B, L, E)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return F.linear(out, out_proj_weight, out_proj_bias)

    result = run_parallel_test("Manual MHA (PT strides)", manual_mha_pytorch_strides, make_input, args.threads, args.iterations, args.tolerance)
    results.append(result)
    print(f"  {'PASS' if result.passed == result.total else 'FAIL'}: {result.passed}/{result.total}")

    # Test 5: Test with non-contiguous tensors (strided views)
    print("\nTesting: SDPA with strided (non-contiguous) inputs...")

    def sdpa_strided(x):
        qkv = F.linear(x, in_proj_weight, in_proj_bias)
        qkv = qkv.transpose(0, 1)  # (L, B, 3*E)
        qkv = qkv.view(seq_len, batch_size, 3, num_heads, head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # Strided views!
        # Reshape but don't call .contiguous()
        q = q.reshape(seq_len, batch_size * num_heads, head_dim).transpose(0, 1)
        k = k.reshape(seq_len, batch_size * num_heads, head_dim).transpose(0, 1)
        v = v.reshape(seq_len, batch_size * num_heads, head_dim).transpose(0, 1)
        # These may be non-contiguous
        q = q.view(batch_size, num_heads, seq_len, head_dim)
        k = k.view(batch_size, num_heads, seq_len, head_dim)
        v = v.view(batch_size, num_heads, seq_len, head_dim)
        # SDPA with potentially non-contiguous inputs
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        return out

    result = run_parallel_test("SDPA strided inputs", sdpa_strided, make_input, args.threads, args.iterations, args.tolerance)
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

    failing = [r for r in results if r.passed < r.total]
    if failing:
        print("\n" + "=" * 70)
        print("BUG LOCATION")
        print("=" * 70)
        for r in failing:
            rate = (r.total - r.passed) / r.total * 100
            print(f"  - {r.name}: {rate:.0f}% failure rate")
            if r.failures:
                for f in r.failures[:2]:
                    print(f"    {f}")

    return 0 if not failing else 1


if __name__ == "__main__":
    sys.exit(main())
