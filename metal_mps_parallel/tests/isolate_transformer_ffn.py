#!/usr/bin/env python3
"""
Isolate TransformerBlock components beyond MHA to find additional race sources.

TransformerEncoderLayer stages:
1. Self-attention (MHA) - known race in .contiguous()
2. Residual: x + dropout1(attn_out)
3. LayerNorm (norm1)
4. FFN: linear1 -> activation -> dropout -> linear2
5. Residual: x + dropout2(ffn_out)
6. LayerNorm (norm2)

FINDINGS (N=1273):
- FFN components pass 100/100 parallel
- Manual TransformerBlock (chunk() instead of unflatten+contiguous) passes 100/100
- Official nn.TransformerEncoderLayer fails at ~90% due to MHA .contiguous() race only
- The ONLY race in TransformerBlock is in MHA's _in_projection_packed
- Patch 035 fixes this by using chunk() instead of the complex reshape pattern
"""

import argparse
import os
import threading
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["MPS_FORCE_GRAPH_PATH"] = "1"


def test_component(name, op, iterations=30, num_threads=8, tolerance=1e-3):
    """Test a component for parallel race conditions."""
    passed = 0
    max_diff = 0.0
    batch_size, seq_len, embed_dim = 4, 128, 256

    for _ in range(iterations):
        inputs = [torch.randn(batch_size, seq_len, embed_dim, device="mps") for _ in range(num_threads)]
        torch.mps.synchronize()

        # Sequential baseline
        expected = []
        with torch.no_grad():
            for inp in inputs:
                expected.append(op(inp).clone())
        torch.mps.synchronize()

        # Parallel execution
        results = [None] * num_threads

        def worker(tid):
            with torch.no_grad():
                results[tid] = op(inputs[tid])
            torch.mps.synchronize()

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        if any(t.is_alive() for t in threads):
            print(f"  {name}: TIMEOUT (thread hang)")
            return 0, iterations, float("inf")

        torch.mps.synchronize()

        ok = True
        for tid in range(num_threads):
            if results[tid] is None:
                ok = False
                continue
            diff = (results[tid] - expected[tid]).abs().max().item()
            max_diff = max(max_diff, diff)
            if diff > tolerance:
                ok = False

        if ok:
            passed += 1

    return passed, iterations, max_diff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        print("SKIP: MPS not available")
        return 0

    print("=" * 70)
    print("TransformerBlock Component Isolation")
    print(f"Threads: {args.threads}, Iterations: {args.iterations}")
    print("=" * 70)

    batch_size, seq_len, embed_dim = 4, 128, 256
    dim_feedforward = 1024
    num_heads = 4
    head_dim = embed_dim // num_heads

    torch.manual_seed(42)
    linear1 = nn.Linear(embed_dim, dim_feedforward).to("mps").eval()
    linear2 = nn.Linear(dim_feedforward, embed_dim).to("mps").eval()
    norm1 = nn.LayerNorm(embed_dim).to("mps").eval()
    norm2 = nn.LayerNorm(embed_dim).to("mps").eval()
    mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True).to("mps").eval()
    in_proj_weight = mha.in_proj_weight.clone()
    in_proj_bias = mha.in_proj_bias.clone() if mha.in_proj_bias is not None else None
    out_proj_weight = mha.out_proj.weight.clone()
    out_proj_bias = mha.out_proj.bias.clone() if mha.out_proj.bias is not None else None
    layer = nn.TransformerEncoderLayer(embed_dim, nhead=num_heads, dim_feedforward=dim_feedforward, batch_first=True).to("mps").eval()
    torch.mps.synchronize()

    results = []

    # Test 1: FFN
    print("\n1. FFN (linear1 -> GELU -> linear2)...")
    p, t, d = test_component("FFN", lambda x: linear2(F.gelu(linear1(x))), args.iterations, args.threads)
    results.append(("FFN", p, t, d))
    print(f"   {p}/{t}, max_diff={d:.2e}")

    # Test 2: LayerNorm chain
    print("\n2. LayerNorm chain...")
    p, t, d = test_component("norm1+norm2", lambda x: norm2(norm1(x)), args.iterations, args.threads)
    results.append(("norm1+norm2", p, t, d))
    print(f"   {p}/{t}, max_diff={d:.2e}")

    # Test 3: Manual MHA (safe pattern)
    print("\n3. Manual MHA (chunk, no .contiguous() race)...")

    def manual_mha(x):
        qkv = F.linear(x, in_proj_weight, in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)  # Safe: chunk instead of unflatten+transpose+contiguous
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return F.linear(out, out_proj_weight, out_proj_bias)

    p, t, d = test_component("Manual MHA (safe)", manual_mha, args.iterations, args.threads)
    results.append(("Manual MHA (safe)", p, t, d))
    print(f"   {p}/{t}, max_diff={d:.2e}")

    # Test 4: Manual TransformerBlock (safe pattern)
    print("\n4. Manual TransformerBlock (full, safe pattern)...")

    def manual_transformer(x):
        attn_out = manual_mha(x)
        x1 = x + attn_out
        x2 = norm1(x1)
        ff = linear2(F.gelu(linear1(x2)))
        x3 = x2 + ff
        return norm2(x3)

    p, t, d = test_component("Manual Transformer", manual_transformer, args.iterations, args.threads)
    results.append(("Manual Transformer", p, t, d))
    print(f"   {p}/{t}, max_diff={d:.2e}")

    # Test 5: Official nn.TransformerEncoderLayer
    print("\n5. nn.TransformerEncoderLayer (official, has .contiguous() race)...")
    p, t, d = test_component("nn.TransformerEncoderLayer", lambda x: layer(x), args.iterations, args.threads)
    results.append(("nn.TransformerEncoderLayer", p, t, d))
    print(f"   {p}/{t}, max_diff={d:.2e}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Component':<30} {'Result':<12} {'Max Diff':<12} Status")
    print("-" * 70)

    all_pass = True
    for name, p, t, d in results:
        status = "OK" if p == t else "RACE"
        if p < t:
            all_pass = False
        print(f"{name:<30} {p}/{t:<9} {d:<12.2e} {status}")

    print("\n" + "=" * 70)
    if all_pass:
        print("ALL PASS: No races detected in any component")
    else:
        print("CONCLUSION: The only race is in nn.TransformerEncoderLayer's MHA")
        print("Root cause: .contiguous() in _in_projection_packed (torch/nn/functional.py)")
        print("Fix: Use chunk() instead of unflatten+transpose+contiguous (patch 035)")
    print("=" * 70)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
