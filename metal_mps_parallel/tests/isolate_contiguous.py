#!/usr/bin/env python3
"""
Test if .contiguous() is causing the race condition.

Hypothesis: PyTorch's _in_projection_packed calls .contiguous() which
triggers a race in MPS memory allocation or copy.
"""

import os
import threading
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["MPS_FORCE_GRAPH_PATH"] = "1"
SYNC_AFTER_PROJECTION = os.environ.get("MPS_TEST_SYNC_AFTER_PROJECTION") == "1"


def run_parallel_test(name, op, input_factory, num_threads=8, iterations=30, tolerance=1e-3):
    torch.mps.synchronize()
    passed = 0
    max_diff = 0.0
    failures = []

    for it in range(iterations):
        inputs = [input_factory(tid).to("mps") for tid in range(num_threads)]
        expected = []
        with torch.no_grad():
            for inp in inputs:
                expected.append(op(inp).clone())
        torch.mps.synchronize()

        results = [None] * num_threads

        def worker(tid):
            with torch.no_grad():
                results[tid] = op(inputs[tid])
            torch.mps.synchronize()

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        ok = True
        for tid in range(num_threads):
            if results[tid] is None:
                ok = False
                continue
            diff = (results[tid] - expected[tid]).abs().max().item()
            max_diff = max(max_diff, diff)
            if diff > tolerance:
                ok = False
                failures.append(f"iter {it} tid {tid}: diff={diff:.2e}")
        if ok:
            passed += 1

    return passed, iterations, max_diff, failures[:3]


def main():
    if not torch.backends.mps.is_available():
        print("SKIP: MPS not available")
        return 0

    print("=" * 70)
    print("Testing if .contiguous() causes the race")
    print(f"MPS_TEST_SYNC_AFTER_PROJECTION={int(SYNC_AFTER_PROJECTION)}")
    print("=" * 70)

    batch_size = 4
    seq_len = 128
    embed_dim = 256
    num_heads = 4
    head_dim = embed_dim // num_heads

    mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True).to("mps").eval()
    torch.mps.synchronize()

    in_proj_weight = mha.in_proj_weight.clone()
    in_proj_bias = mha.in_proj_bias.clone()

    def make_input(tid):
        torch.manual_seed(tid * 1000 + 42)
        return torch.randn(batch_size, seq_len, embed_dim)

    # Test 1: PyTorch pattern WITH .contiguous()
    print("\nTesting: With .contiguous() (PyTorch's approach)...")

    def proj_with_contiguous(x):
        proj = F.linear(x, in_proj_weight, in_proj_bias)
        proj = (
            proj.unflatten(-1, (3, embed_dim))
            .unsqueeze(0)
            .transpose(0, -2)
            .squeeze(-2)
            .contiguous()  # <-- This might be the culprit
        )
        if SYNC_AFTER_PROJECTION:
            torch.mps.synchronize()
        q, k, v = proj[0], proj[1], proj[2]
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

    result = run_parallel_test("With contiguous()", proj_with_contiguous, make_input)
    print(f"  {'PASS' if result[0] == result[1] else 'FAIL'}: {result[0]}/{result[1]}, max_diff={result[2]:.2e}")

    # Test 2: Same pattern WITHOUT .contiguous()
    print("\nTesting: Without .contiguous() (reshape only)...")

    def proj_without_contiguous(x):
        proj = F.linear(x, in_proj_weight, in_proj_bias)
        proj = (
            proj.unflatten(-1, (3, embed_dim))
            .unsqueeze(0)
            .transpose(0, -2)
            .squeeze(-2)
            # NO .contiguous()
        )
        # Need to use reshape instead of view since tensor is not contiguous
        q, k, v = proj[0], proj[1], proj[2]
        q = q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

    result = run_parallel_test("Without contiguous()", proj_without_contiguous, make_input)
    print(f"  {'PASS' if result[0] == result[1] else 'FAIL'}: {result[0]}/{result[1]}, max_diff={result[2]:.2e}")

    # Test 3: Simple chunk (no complex reshaping)
    print("\nTesting: Simple chunk (baseline)...")

    def proj_chunk(x):
        qkv = F.linear(x, in_proj_weight, in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

    result = run_parallel_test("Simple chunk", proj_chunk, make_input)
    print(f"  {'PASS' if result[0] == result[1] else 'FAIL'}: {result[0]}/{result[1]}, max_diff={result[2]:.2e}")

    # Test 4: Just .contiguous() on output
    print("\nTesting: Just .contiguous() on qkv (no reshaping)...")

    def proj_contiguous_qkv(x):
        qkv = F.linear(x, in_proj_weight, in_proj_bias)
        qkv = qkv.contiguous()  # Just make contiguous
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.contiguous().view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.contiguous().view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.contiguous().view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

    result = run_parallel_test("Extra contiguous()", proj_contiguous_qkv, make_input)
    print(f"  {'PASS' if result[0] == result[1] else 'FAIL'}: {result[0]}/{result[1]}, max_diff={result[2]:.2e}")

    # Test 5: Clone instead of contiguous (explicit copy)
    print("\nTesting: Clone instead of contiguous...")

    def proj_clone(x):
        proj = F.linear(x, in_proj_weight, in_proj_bias)
        proj = (
            proj.unflatten(-1, (3, embed_dim))
            .unsqueeze(0)
            .transpose(0, -2)
            .squeeze(-2)
            .clone()  # Use clone instead of contiguous
        )
        if SYNC_AFTER_PROJECTION:
            torch.mps.synchronize()
        q, k, v = proj[0], proj[1], proj[2]
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

    result = run_parallel_test("Clone instead", proj_clone, make_input)
    print(f"  {'PASS' if result[0] == result[1] else 'FAIL'}: {result[0]}/{result[1]}, max_diff={result[2]:.2e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
