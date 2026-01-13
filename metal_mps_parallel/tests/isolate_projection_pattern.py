#!/usr/bin/env python3
"""
Test if PyTorch's _in_projection_packed reshaping pattern causes the race.

PyTorch does: linear -> unflatten -> unsqueeze -> transpose -> squeeze -> contiguous -> slice
vs our manual: linear -> chunk

These create different stride patterns that may trigger the MPS bug.
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
    max_diff_overall = 0.0
    failures = []

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
            failures.append(f"iter {it}: errors: {errors[0]}")
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
                failures.append(f"iter {it} tid {tid}: diff={diff:.2e}")

        if iteration_ok:
            passed += 1

    return passed, iterations, max_diff_overall, failures[:5]


def main():
    if not torch.backends.mps.is_available():
        print("SKIP: MPS not available")
        return 0

    print("=" * 70)
    print("Testing _in_projection_packed reshape pattern")
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
    out_proj_weight = mha.out_proj.weight.clone()
    out_proj_bias = mha.out_proj.bias.clone()

    def make_input(tid):
        torch.manual_seed(tid * 1000 + 42)
        return torch.randn(batch_size, seq_len, embed_dim)

    # Test 1: Manual projection with chunk (our simple approach)
    print("\nTesting: Manual projection (chunk)...")

    def manual_proj_chunk(x):
        qkv = F.linear(x, in_proj_weight, in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        return q, k, v

    def test_proj_chunk(x):
        q, k, v = manual_proj_chunk(x)
        # Reshape and SDPA
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

    result = run_parallel_test("Manual (chunk)", test_proj_chunk, make_input)
    status = "PASS" if result[0] == result[1] else "FAIL"
    print(f"  {status}: {result[0]}/{result[1]}, max_diff={result[2]:.2e}")

    # Test 2: PyTorch's _in_projection_packed pattern
    print("\nTesting: PyTorch _in_projection_packed pattern...")

    def pytorch_proj_pattern(x):
        # Exactly as PyTorch does it
        proj = F.linear(x, in_proj_weight, in_proj_bias)  # (B, L, 3*E)
        E = embed_dim
        # reshape to 3, E and not E, 3 for better memory coalescing
        proj = (
            proj.unflatten(-1, (3, E))
            .unsqueeze(0)
            .transpose(0, -2)
            .squeeze(-2)
            .contiguous()
        )
        if SYNC_AFTER_PROJECTION:
            torch.mps.synchronize()
        q, k, v = proj[0], proj[1], proj[2]
        return q, k, v

    def test_pytorch_proj(x):
        q, k, v = pytorch_proj_pattern(x)
        # PyTorch's projection returns (B, L, E) format
        # Need to reshape to (B, H, L, D) for SDPA
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

    result = run_parallel_test("PyTorch proj pattern", test_pytorch_proj, make_input)
    status = "PASS" if result[0] == result[1] else "FAIL"
    print(f"  {status}: {result[0]}/{result[1]}, max_diff={result[2]:.2e}")

    # Test 3: Full PyTorch MHA path (need_weights=False)
    print("\nTesting: F.multi_head_attention_forward (need_weights=False)...")

    def test_functional_mha(x):
        # Transpose to (L, B, E) for non-batch-first
        x_t = x.transpose(0, 1)
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
        )
        return out.transpose(0, 1)

    result = run_parallel_test("F.multi_head_attention_forward", test_functional_mha, make_input)
    status = "PASS" if result[0] == result[1] else "FAIL"
    print(f"  {status}: {result[0]}/{result[1]}, max_diff={result[2]:.2e}")

    # Test 4: nn.MultiheadAttention
    print("\nTesting: nn.MultiheadAttention...")

    def test_nn_mha(x):
        out, _ = mha(x, x, x, need_weights=False)
        return out

    result = run_parallel_test("nn.MultiheadAttention", test_nn_mha, make_input)
    status = "PASS" if result[0] == result[1] else "FAIL"
    print(f"  {status}: {result[0]}/{result[1]}, max_diff={result[2]:.2e}")

    # Test 5: Check stride patterns
    print("\n" + "=" * 70)
    print("STRIDE ANALYSIS")
    print("=" * 70)

    x_test = torch.randn(batch_size, seq_len, embed_dim, device="mps")

    # Manual chunk
    qkv_chunk = F.linear(x_test, in_proj_weight, in_proj_bias)
    q_chunk, k_chunk, v_chunk = qkv_chunk.chunk(3, dim=-1)
    print(f"\nManual chunk strides:")
    print(f"  qkv: {qkv_chunk.stride()}, contiguous: {qkv_chunk.is_contiguous()}")
    print(f"  q:   {q_chunk.stride()}, contiguous: {q_chunk.is_contiguous()}")

    # PyTorch pattern
    proj = F.linear(x_test, in_proj_weight, in_proj_bias)
    proj = proj.unflatten(-1, (3, embed_dim)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
    q_pt, k_pt, v_pt = proj[0], proj[1], proj[2]
    print(f"\nPyTorch pattern strides:")
    print(f"  proj: {proj.stride()}, contiguous: {proj.is_contiguous()}")
    print(f"  q:    {q_pt.stride()}, contiguous: {q_pt.is_contiguous()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
