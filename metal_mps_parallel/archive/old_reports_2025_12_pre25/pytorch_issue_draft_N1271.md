# PyTorch GitHub Issue Draft

**Title**: [MPS] Race condition in nn.MultiheadAttention when .contiguous() called from multiple threads

## Bug Description

When using `nn.MultiheadAttention` on MPS with multiple threads, intermittent numerical errors occur. The root cause is in `_in_projection_packed` (torch/nn/functional.py) which calls `.contiguous()` on a tensor with complex stride patterns from `unflatten/transpose` operations. This `.contiguous()` call triggers a race condition in the MPS backend's memory allocation or copy operations.

## Reproduction

```python
#!/usr/bin/env python3
"""Minimal reproduction: MPS .contiguous() race condition"""
import os
import threading
os.environ["MPS_FORCE_GRAPH_PATH"] = "1"

import torch
import torch.nn.functional as F

def test_contiguous_race(use_contiguous: bool, iterations: int = 30, threads: int = 8):
    embed_dim, batch_size, seq_len = 256, 4, 128
    num_heads, head_dim = 4, embed_dim // 4

    torch.manual_seed(42)
    weight = torch.randn(3 * embed_dim, embed_dim, device="mps")
    bias = torch.randn(3 * embed_dim, device="mps")
    torch.mps.synchronize()

    def projection_op(x):
        proj = F.linear(x, weight, bias)
        proj = proj.unflatten(-1, (3, embed_dim)).unsqueeze(0).transpose(0, -2).squeeze(-2)

        if use_contiguous:
            proj = proj.contiguous()  # <-- RACE CONDITION

        q, k, v = proj[0], proj[1], proj[2]
        if use_contiguous:
            q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        else:
            q = q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            k = k.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            v = v.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

    passed, max_diff = 0, 0.0
    for iteration in range(iterations):
        inputs = [torch.randn(batch_size, seq_len, embed_dim, device="mps") for _ in range(threads)]
        torch.mps.synchronize()

        with torch.no_grad():
            expected = [projection_op(inp).clone() for inp in inputs]
        torch.mps.synchronize()

        results = [None] * threads
        def worker(tid):
            with torch.no_grad():
                results[tid] = projection_op(inputs[tid])
            torch.mps.synchronize()

        worker_threads = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
        for t in worker_threads: t.start()
        for t in worker_threads: t.join()

        ok = all(results[i] is not None and (results[i] - expected[i]).abs().max().item() <= 1e-3
                 for i in range(threads))
        if ok: passed += 1
        for i in range(threads):
            if results[i] is not None:
                max_diff = max(max_diff, (results[i] - expected[i]).abs().max().item())

    return passed, iterations, max_diff

print(f"PyTorch: {torch.__version__}")
print(f"WITHOUT .contiguous(): {test_contiguous_race(False)}")
print(f"WITH .contiguous(): {test_contiguous_race(True)}")
```

## Expected Behavior

Both tests should pass 30/30 iterations with max_diff near 0.

## Actual Behavior

```
PyTorch: 2.9.1a0+git9a4518e
WITHOUT .contiguous(): (30, 30, 0.0)
WITH .contiguous(): (24, 30, 112.45)
```

The test WITH `.contiguous()` fails ~6-15 iterations with large numerical differences (up to 100+).

## Root Cause Analysis

The bug is in `_in_projection_packed` (torch/nn/functional.py:5706):

```python
proj = proj.unflatten(-1, (3, E)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
#                                                                          ^^^^^^^^^^^
# This .contiguous() triggers the race
```

**Evidence:**
1. Individual MPS operations (matmul, softmax, linear, layernorm) all pass 30/30 in parallel
2. `F.scaled_dot_product_attention` passes 30/30 in parallel
3. Manual Python implementation of MHA with identical operations passes 30/30
4. Only PyTorch's `F.multi_head_attention_forward` / `nn.MultiheadAttention` fails
5. Removing `.contiguous()` and using `.reshape()` instead of `.view()` → 30/30 PASS

## Proposed Fix

Remove `.contiguous()` from `_in_projection_packed` and use `.reshape()` downstream:

```python
# Before (races on MPS):
proj = proj.unflatten(-1, (3, E)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()

# After (correct):
proj = proj.unflatten(-1, (3, E)).unsqueeze(0).transpose(0, -2).squeeze(-2)
# Use .reshape() instead of .view() where q, k, v are extracted
```

This avoids the memory allocation race while producing identical results.

## Environment

- macOS 15.7.2 (24G325)
- Apple M4 Max (40 GPU cores)
- PyTorch 2.9.1a0+git9a4518e
- Python 3.x

## Additional Context

- `.clone()` also triggers the race (17/30 FAIL)
- Simple `.contiguous()` on small tensors works fine
- The race appears to be in MPS memory allocation during parallel tensor copies
- Setting `MPS_FORCE_GRAPH_PATH=0` does not change behavior

## Workaround

Serialize MPS inference through a batch queue with num_workers=1, or avoid `nn.MultiheadAttention` and implement MHA manually using `.reshape()` instead of `.view()`.

---

## Labels
- module: mps
- triaged

## cc
@kulinseth @malfet @DenisVieriu97 (MPS maintainers)
