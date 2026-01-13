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
CosyVoice3 PyTorch LLM Benchmark

Compare PyTorch MPS performance to MLX to understand if the slowness
is architecture-specific or MLX-specific.
"""

import sys
from pathlib import Path
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def benchmark_pytorch():
    import torch

    print("=" * 70)
    print("CosyVoice3 PyTorch LLM Benchmark")
    print("=" * 70)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\nDevice: {device}")
    print(f"PyTorch version: {torch.__version__}")

    # Model dimensions (same as CosyVoice3)
    batch = 1
    hidden_size = 896
    num_heads = 7
    num_kv_heads = 1
    head_dim = 128
    num_layers = 24
    intermediate_size = 4864

    # ==========================================================================
    # Test 1: Raw SDPA performance
    # ==========================================================================
    print("\n1. Testing PyTorch SDPA...")

    q = torch.randn(batch, num_heads, 1, head_dim, device=device)
    k = torch.randn(batch, num_kv_heads, 100, head_dim, device=device)
    v = torch.randn(batch, num_kv_heads, 100, head_dim, device=device)

    # PyTorch needs to expand KV for GQA
    k_expanded = k.expand(batch, num_heads, -1, head_dim)
    v_expanded = v.expand(batch, num_heads, -1, head_dim)

    # Warmup
    for _ in range(5):
        out = torch.nn.functional.scaled_dot_product_attention(q, k_expanded, v_expanded)
        torch.mps.synchronize()

    times = []
    for _ in range(20):
        start = time.time()
        out = torch.nn.functional.scaled_dot_product_attention(q, k_expanded, v_expanded)
        torch.mps.synchronize()
        times.append(time.time() - start)

    avg = np.mean(times[5:])
    print(f"   PyTorch SDPA: {avg*1000:.2f}ms")

    # ==========================================================================
    # Test 2: Single transformer layer simulation
    # ==========================================================================
    print("\n2. Testing single layer forward (simulated)...")

    # Create weights
    w_q = torch.randn(num_heads * head_dim, hidden_size, device=device)
    w_k = torch.randn(num_kv_heads * head_dim, hidden_size, device=device)
    w_v = torch.randn(num_kv_heads * head_dim, hidden_size, device=device)
    w_o = torch.randn(hidden_size, num_heads * head_dim, device=device)

    w_gate = torch.randn(intermediate_size, hidden_size, device=device)
    w_up = torch.randn(intermediate_size, hidden_size, device=device)
    w_down = torch.randn(hidden_size, intermediate_size, device=device)

    hidden = torch.randn(batch, 1, hidden_size, device=device)

    # Cached KV
    k_cache = torch.randn(batch, num_kv_heads, 100, head_dim, device=device)
    v_cache = torch.randn(batch, num_kv_heads, 100, head_dim, device=device)

    def single_layer(x, k_cache, v_cache):
        # Attention
        q = torch.matmul(x, w_q.T)
        k = torch.matmul(x, w_k.T)
        v = torch.matmul(x, w_v.T)

        q = q.view(batch, 1, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch, 1, num_kv_heads, head_dim).transpose(1, 2)
        v = v.view(batch, 1, num_kv_heads, head_dim).transpose(1, 2)

        # Update cache
        k = torch.cat([k_cache, k], dim=2)
        v = torch.cat([v_cache, v], dim=2)

        # Expand for GQA
        k_exp = k.expand(batch, num_heads, -1, head_dim)
        v_exp = v.expand(batch, num_heads, -1, head_dim)

        # SDPA
        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k_exp, v_exp)
        attn_out = attn_out.transpose(1, 2).reshape(batch, 1, -1)
        attn_out = torch.matmul(attn_out, w_o.T)

        # MLP (simplified)
        gate = torch.matmul(x + attn_out, w_gate.T)
        up = torch.matmul(x + attn_out, w_up.T)
        mlp_out = torch.matmul(torch.nn.functional.silu(gate) * up, w_down.T)

        return x + attn_out + mlp_out, k, v

    # Warmup
    for _ in range(5):
        out, _, _ = single_layer(hidden, k_cache, v_cache)
        torch.mps.synchronize()

    times = []
    for _ in range(20):
        start = time.time()
        out, _, _ = single_layer(hidden, k_cache, v_cache)
        torch.mps.synchronize()
        times.append(time.time() - start)

    avg_layer = np.mean(times[5:])
    print(f"   Single layer: {avg_layer*1000:.2f}ms")
    print(f"   Estimated 24 layers: {avg_layer*24*1000:.1f}ms ({1/(avg_layer*24):.1f} tok/s)")

    # ==========================================================================
    # Test 3: Full 24-layer forward
    # ==========================================================================
    print("\n3. Testing 24-layer forward...")

    # Create all layer weights
    layers = []
    for _ in range(num_layers):
        layers.append({
            'w_q': torch.randn(num_heads * head_dim, hidden_size, device=device),
            'w_k': torch.randn(num_kv_heads * head_dim, hidden_size, device=device),
            'w_v': torch.randn(num_kv_heads * head_dim, hidden_size, device=device),
            'w_o': torch.randn(hidden_size, num_heads * head_dim, device=device),
            'w_gate': torch.randn(intermediate_size, hidden_size, device=device),
            'w_up': torch.randn(intermediate_size, hidden_size, device=device),
            'w_down': torch.randn(hidden_size, intermediate_size, device=device),
        })

    # Create caches
    caches = [(
        torch.randn(batch, num_kv_heads, 100, head_dim, device=device),
        torch.randn(batch, num_kv_heads, 100, head_dim, device=device)
    ) for _ in range(num_layers)]

    def full_forward(x, caches):
        for i, (layer, (k_c, v_c)) in enumerate(zip(layers, caches)):
            # Attention
            q = torch.matmul(x, layer['w_q'].T)
            k = torch.matmul(x, layer['w_k'].T)
            v = torch.matmul(x, layer['w_v'].T)

            q = q.view(batch, 1, num_heads, head_dim).transpose(1, 2)
            k = k.view(batch, 1, num_kv_heads, head_dim).transpose(1, 2)
            v = v.view(batch, 1, num_kv_heads, head_dim).transpose(1, 2)

            k = torch.cat([k_c, k], dim=2)
            v = torch.cat([v_c, v], dim=2)

            k_exp = k.expand(batch, num_heads, -1, head_dim)
            v_exp = v.expand(batch, num_heads, -1, head_dim)

            attn_out = torch.nn.functional.scaled_dot_product_attention(q, k_exp, v_exp)
            attn_out = attn_out.transpose(1, 2).reshape(batch, 1, -1)
            attn_out = torch.matmul(attn_out, layer['w_o'].T)

            x = x + attn_out

            # MLP
            gate = torch.matmul(x, layer['w_gate'].T)
            up = torch.matmul(x, layer['w_up'].T)
            mlp_out = torch.matmul(torch.nn.functional.silu(gate) * up, layer['w_down'].T)
            x = x + mlp_out

        return x

    # Warmup
    for _ in range(5):
        out = full_forward(hidden, caches)
        torch.mps.synchronize()

    times = []
    for _ in range(20):
        start = time.time()
        out = full_forward(hidden, caches)
        torch.mps.synchronize()
        times.append(time.time() - start)

    avg_full = np.mean(times[5:])
    print(f"   24-layer forward: {avg_full*1000:.1f}ms ({1/avg_full:.1f} tok/s)")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: PyTorch MPS vs Expected")
    print("=" * 70)

    print(f"\nPyTorch SDPA: {np.mean(times[5:])*1000:.2f}ms")
    print(f"PyTorch 24-layer: {avg_full*1000:.1f}ms ({1/avg_full:.1f} tok/s)")

    # Compare to MLX
    print("\nFor comparison (from earlier MLX profiling):")
    print("  MLX SDPA: ~3-6ms")
    print("  MLX 24-layer: ~140ms (7 tok/s)")

    if avg_full < 0.1:  # < 100ms
        print("\n[PyTorch MPS is FASTER]")
        print("This suggests MLX SDPA/GQA implementation needs optimization")
    else:
        print(f"\n[PyTorch MPS is also slow: {avg_full*1000:.0f}ms]")
        print("This model architecture is inherently slow for batch=1 generation")


if __name__ == "__main__":
    benchmark_pytorch()
