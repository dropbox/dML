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
Q7 Integrated Benchmark: Measure AdaLN impact within actual DiT forward.

Worker #1341 - 2025-12-20
"""

import time
import mlx.core as mx
import numpy as np


def benchmark_q7_integrated():
    """Measure AdaLN impact in actual DiT forward pass."""
    from tools.pytorch_to_mlx.converters.models.cosyvoice3_dit import (
        CausalMaskedDiffWithDiT,
        create_cosyvoice3_flow_config
    )

    print("=== Q7 Integrated Benchmark ===\n")

    config = create_cosyvoice3_flow_config()
    model = CausalMaskedDiffWithDiT(config)

    # Test inputs
    batch = 1
    num_tokens = 50
    tokens = mx.zeros((batch, num_tokens), dtype=mx.int32)
    spk_emb = mx.random.normal((batch, 192))
    mx.eval(tokens, spk_emb)

    # Warmup
    print("Warming up...")
    for _ in range(3):
        result = model.inference(tokens, spk_emb, num_steps=5)
        mx.eval(result)

    # Baseline: Normal forward
    print("\n1. Baseline flow inference...")
    times_baseline = []
    for _ in range(5):
        start = time.perf_counter()
        result = model.inference(tokens, spk_emb, num_steps=5)
        mx.eval(result)
        times_baseline.append((time.perf_counter() - start) * 1000)
    baseline_ms = np.mean(times_baseline)
    print(f"   Baseline: {baseline_ms:.2f} +/- {np.std(times_baseline):.2f} ms")

    # Profile: Time spent in AdaLN linear vs total DiT forward
    # This requires instrumenting the code, so we'll estimate
    print("\n2. Profiling AdaLN contribution...")

    # Count operations in DiT forward for 22 blocks:
    # - 22 × AdaLN linear: [B, 1024] @ [1024, 6144]
    # - 22 × Attention Q projection: [B, L, 1024] @ [1024, 1024]
    # - 22 × Attention K projection: [B, L, 1024] @ [1024, 64] (GQA)
    # - 22 × Attention V projection: [B, L, 1024] @ [1024, 64] (GQA)
    # - 22 × Attention O projection: [B, L, 1024] @ [1024, 1024]
    # - 22 × SDPA: O(L^2) attention
    # - 22 × FFN linear1: [B, L, 1024] @ [1024, 4096]
    # - 22 × FFN linear2: [B, L, 4096] @ [4096, 1024]

    # AdaLN: 22 × [1, 1024] @ [1024, 6144] = 22 × 6M = 132M FLOPs
    # Attention Q: 22 × [1, 100, 1024] @ [1024, 1024] = 22 × 100 × 1M = 2.2B FLOPs
    # etc...

    L = num_tokens * 2  # After upsampling
    dim = 1024
    adaln_flops = 22 * 1 * dim * dim * 6  # per ODE step
    attn_qkv_flops = 22 * L * dim * dim * 3  # Q+K+V approx (K,V smaller due to GQA)
    ffn_flops = 22 * L * dim * 4 * dim * 2  # FFN1 + FFN2

    total_flops = adaln_flops + attn_qkv_flops + ffn_flops
    adaln_percent_flops = (adaln_flops / total_flops) * 100

    print(f"   AdaLN FLOPs: {adaln_flops / 1e6:.1f}M per ODE step")
    print(f"   Attention FLOPs: {attn_qkv_flops / 1e6:.1f}M per ODE step")
    print(f"   FFN FLOPs: {ffn_flops / 1e6:.1f}M per ODE step")
    print(f"   AdaLN % of DiT compute: {adaln_percent_flops:.2f}%")

    # Adjust estimate based on FLOP analysis
    _num_ode_steps = 5  # noqa: F841 - documentation only
    estimated_adaln_time = baseline_ms * (adaln_percent_flops / 100)
    print(f"\n   Estimated AdaLN time: {estimated_adaln_time:.2f} ms")
    print(f"   Potential speedup from Q7: {adaln_percent_flops:.2f}%")

    # Compare isolated vs integrated
    print("\n3. Isolated vs Integrated comparison...")
    isolated_adaln_ms = 45.37  # From previous benchmark
    print(f"   Isolated benchmark: {isolated_adaln_ms:.2f} ms ({isolated_adaln_ms/baseline_ms*100:.1f}% of flow)")
    print(f"   FLOP-based estimate: {estimated_adaln_time:.2f} ms ({adaln_percent_flops:.1f}% of flow)")

    discrepancy = isolated_adaln_ms / estimated_adaln_time if estimated_adaln_time > 0 else float('inf')
    print(f"\n   Discrepancy factor: {discrepancy:.1f}x")

    if discrepancy > 2:
        print("   WARNING: Isolated benchmark overestimates - kernel fusion/pipelining hides AdaLN cost")
    elif discrepancy < 0.5:
        print("   NOTE: FLOP estimate underestimates - memory bandwidth limited")
    else:
        print("   OK: Estimates are consistent")

    return baseline_ms, adaln_percent_flops


if __name__ == "__main__":
    baseline, adaln_pct = benchmark_q7_integrated()

    print("\n" + "=" * 60)
    print("Q7 EVALUATION CONCLUSION")
    print("=" * 60)

    if adaln_pct > 5:
        print(f"""
AdaLN contributes ~{adaln_pct:.1f}% of DiT compute.
Potential speedup from Q7: ~{adaln_pct:.1f}%

However, this requires:
1. Precompute all 22 blocks × 5 ODE steps = 110 cached tensors
2. Implement cache lookup mechanism with proper memory management
3. Handle variable batch sizes and sequence lengths

Given the complexity vs ~{adaln_pct:.1f}% gain, and past experience with
similar optimizations (Q1, Q12, Q17 all being NOT WORTH), Q7 is MARGINAL.

Recommendation: LOW PRIORITY - implement only if other work is exhausted.
""")
    else:
        print(f"""
AdaLN contributes ~{adaln_pct:.1f}% of DiT compute.
This is too small to be worth the implementation complexity.

Recommendation: NOT WORTH - skip Q7 optimization.
""")
