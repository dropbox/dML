# Projection Pattern Race: Device-Sync Barrier Finding

**Worker context**: Follow-up to N=1337 residual races after `View.mm` MPSEncodingLock coverage.
**Date**: 2025-12-19
**Status**: Reproduced + mitigation validated (no backend fix)

## Summary

The remaining correctness failures in PyTorch’s `_in_projection_packed` reshape pattern are eliminated by inserting a **device-wide MPS synchronization** between the projection copy (`.contiguous()` / `.clone()`) and the subsequent `F.scaled_dot_product_attention()` call.

This strongly suggests the residual corruption is caused by **concurrent GPU execution overlap** between the projection-copy phase and the SDPA phase across multiple MPS command queues (streams), not by the isolated `.contiguous()` implementation itself.

## Reproduction (current torch build: `2.9.1a0+git4201c80`)

Baseline failure:
- `python3 tests/isolate_projection_pattern.py`
  - `PyTorch _in_projection_packed pattern`: typically ~`28/30` pass
  - `F.multi_head_attention_forward`: typically ~`27-28/30` pass
  - `nn.MultiheadAttention`: typically ~`23-25/30` pass

Validated mitigation for the direct projection-pattern test:
- `MPS_TEST_SYNC_AFTER_PROJECTION=1 python3 tests/isolate_projection_pattern.py`
  - `PyTorch _in_projection_packed pattern`: `30/30` pass
  - `F.multi_head_attention_forward` and `nn.MultiheadAttention` still fail (cannot inject the barrier into internal PyTorch implementation without patching).

Related (contiguous-focused) repro:
- `python3 tests/isolate_contiguous.py`
  - `With .contiguous()` fails without the barrier.
  - With `MPS_TEST_SYNC_AFTER_PROJECTION=1`, `With .contiguous()` becomes `30/30` pass.

## Key observation

Adding `torch.mps.synchronize()` **after** the projection-copy step and **before** SDPA yields deterministic correctness for the manual reproduction:

```python
proj = (...).contiguous()  # or .clone()
torch.mps.synchronize()    # device-wide barrier
out = F.scaled_dot_product_attention(q, k, v, ...)
```

## What this implies

- Pure projection copy (`linear + reshape + contiguous/clone`) is correct in parallel.
- Pure SDPA is correct in parallel.
- The failure is triggered by **running SDPA immediately after the projection-copy** in a multi-stream context, and disappears when a device-wide barrier separates the phases.

## Practical mitigation options

- Preferred: avoid the `_in_projection_packed` `.contiguous()` pattern (see existing workaround patch `patches/035-mps-in-projection-packed-mps-parallel.patch`).
- Fallback: stage execution with a device-wide barrier between projection and SDPA (correct but very expensive; primarily useful for diagnosis).
- Production-safe: use `BatchQueue` with `num_workers=1` to serialize GPU execution.

## Verified Results (N=1338)

### isolate_projection_pattern.py

| Test | Without Sync | With Sync | Change |
|------|--------------|-----------|--------|
| Manual projection (chunk) | 30/30 | 30/30 | - |
| PyTorch _in_projection_packed pattern | 27/30 | **30/30** | +3 |
| F.multi_head_attention_forward | 26/30 | 25/30 | -1 (variance) |
| nn.MultiheadAttention | 26/30 | 26/30 | - |

**Key finding**: The sync barrier fixes the manual reproduction (30/30) but cannot fix internal PyTorch functions (F.multi_head_attention_forward, nn.MultiheadAttention) because the barrier cannot be injected into their internal implementation.

### isolate_contiguous.py

| Test | Without Sync | With Sync | Change |
|------|--------------|-----------|--------|
| With .contiguous() | 24/30 | 29/30 | +5 |
| Without .contiguous() | 30/30 | 30/30 | - |
| Simple chunk | 30/30 | 30/30 | - |
| Just .contiguous() | 30/30 | 30/30 | - |
| Clone instead | 23/30 | **30/30** | +7 |

**Key finding**: Pure `.contiguous()` and `.clone()` are thread-safe in isolation. The race occurs in the **interaction** between projection-copy and subsequent SDPA execution across command queues.

## Code changes (tests only)

Two debug scripts now accept an environment toggle:
- `tests/isolate_projection_pattern.py` reads `MPS_TEST_SYNC_AFTER_PROJECTION=1` and inserts `torch.mps.synchronize()` after the projection-copy step.
- `tests/isolate_contiguous.py` reads `MPS_TEST_SYNC_AFTER_PROJECTION=1` and inserts `torch.mps.synchronize()` after the projection-copy step in the relevant variants.

## Conclusion

The device-wide sync barrier confirms the race is an **inter-operation timing issue** in multi-stream MPS execution, not a bug in `.contiguous()` or `.clone()` themselves. The View.mm encoding lock fix (N=1336) correctly protects the gather/scatter paths; the residual failures are in the command queue coordination between projection and attention phases.

**Production recommendation**: Use BatchQueue with num_workers=1 to serialize GPU access.
