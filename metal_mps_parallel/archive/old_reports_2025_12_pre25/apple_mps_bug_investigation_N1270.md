# Apple MPS Bug Investigation Report

**Worker**: N=1270
**Date**: 2025-12-18
**Phase**: 3.2 - Identify the exact failing operation

> ⚠️ **UPDATE (N=1337-1338)**: The conclusions below have been superseded by more detailed
> investigation. The root cause is now understood to be:
> 1. Pure `.contiguous()` is **fixed** after the View.mm `MPSEncodingLock` fix (N=1336)
> 2. The remaining race is in **inter-operation timing** when SDPA runs immediately after
>    projection-copy across multiple MPS command queues
> 3. This is an **Apple MPS framework limitation** in multi-stream command queue coordination
> 4. Production solution: BatchQueue with `num_workers=1`
>
> See: `reports/main/view_mm_fix_verification_N1337.md` and
> `reports/main/projection_sdpa_device_sync_barrier_N1338.md` for accurate conclusions.

## Executive Summary (Original N=1270 Analysis)

The parallel MPS race condition is NOT in Apple's Metal framework or SDPA algorithm.
**The bug is in PyTorch's C++ implementation of `F.multi_head_attention_forward`**, specifically in how it interfaces with the MPS backend.

## Methodology

Created isolation tests to systematically identify which operation causes the race:

1. `tests/isolate_mps_operation.py` - Tests individual operations in parallel
2. `tests/isolate_sdpa_components.py` - Tests SDPA sub-components
3. `tests/isolate_mha_internals.py` - Tests MHA internal operations
4. `tests/isolate_mha_functional.py` - Tests PyTorch's functional vs manual implementation

Test parameters: 8 threads, 30 iterations, 1e-3 tolerance

## Results

### Phase 1: Operation-Level Isolation

| Operation | Pass Rate | Max Diff | Status |
|-----------|-----------|----------|--------|
| LayerNorm only | 30/30 | 0.00e+00 | OK |
| Linear only | 30/30 | 0.00e+00 | OK |
| MLP (Linear+GELU+Linear) | 30/30 | 0.00e+00 | OK |
| SDPA (MultiheadAttention) | 28/30 | 3.34e-01 | RACE |
| LayerNorm + Linear | 30/30 | 0.00e+00 | OK |
| Full TransformerBlock | 2/30 | 3.52e-01 | RACE |

**Finding**: Only SDPA (via MultiheadAttention) fails. Individual components pass.

### Phase 2: SDPA Component Isolation

| Component | Pass Rate | Max Diff | Status |
|-----------|-----------|----------|--------|
| Matmul Q@K.T | 30/30 | 0.00e+00 | OK |
| Softmax (attn) | 30/30 | 0.00e+00 | OK |
| Scale op | 30/30 | 0.00e+00 | OK |
| Matmul+Scale | 30/30 | 0.00e+00 | OK |
| Full attn scores | 30/30 | 0.00e+00 | OK |
| Attn output (softmax@V) | 30/30 | 0.00e+00 | OK |
| **F.scaled_dot_product_attention** | **30/30** | 0.00e+00 | **OK** |
| nn.MultiheadAttention | 26/30 | 2.67e-01 | RACE |

**Critical Finding**: `F.scaled_dot_product_attention` passes, but `nn.MultiheadAttention` fails.
The bug is NOT in SDPA itself.

### Phase 3: MHA Internal Isolation

| Component | Pass Rate | Max Diff | Status |
|-----------|-----------|----------|--------|
| QKV projection | 30/30 | 0.00e+00 | OK |
| Out projection | 30/30 | 0.00e+00 | OK |
| QKV proj + reshape | 30/30 | 0.00e+00 | OK |
| QKV + SDPA | 30/30 | 0.00e+00 | OK |
| QKV + SDPA + reshape | 30/30 | 0.00e+00 | OK |
| **Full manual MHA** | **30/30** | 0.00e+00 | **OK** |
| nn.MultiheadAttention | 28/30 | 2.33e-01 | RACE |

**Critical Finding**: Manual MHA implementation with identical operations passes 100%.
Only PyTorch's `nn.MultiheadAttention` fails.

### Phase 4: Functional vs Manual

| Component | Pass Rate | Max Diff | Status |
|-----------|-----------|----------|--------|
| Manual MHA (explicit) | 30/30 | 0.00e+00 | OK |
| **F.multi_head_attention_forward** | **28/30** | 3.49e-01 | **RACE** |
| nn.MultiheadAttention | 29/30 | 2.74e-01 | RACE |
| Manual MHA (PT strides) | 30/30 | 0.00e+00 | OK |
| SDPA strided inputs | 30/30 | 0.00e+00 | OK |

**Root Cause Identified**: The bug is in PyTorch's C++ `F.multi_head_attention_forward` function.

## Root Cause Analysis

### What PASSES (not the bug)
- All individual Metal/MPS operations (matmul, softmax, linear, layernorm)
- `F.scaled_dot_product_attention` (Apple's SDPA implementation)
- Manual Python implementation of MHA with same striding patterns
- Non-contiguous/strided tensor inputs to SDPA

### What FAILS (the bug)
- `F.multi_head_attention_forward` (PyTorch's C++ functional)
- `nn.MultiheadAttention` (calls the above)

### Conclusion

The bug is **NOT in Apple's MPS framework**. It is in **PyTorch's C++ implementation** of `multi_head_attention_forward`, specifically in how it:

1. Manages tensor memory allocation/reuse in parallel contexts
2. Interfaces with the MPS backend for kernel dispatch
3. Handles graph compilation caching across threads

The race condition occurs somewhere in PyTorch's C++ code path that differs from the equivalent Python implementation. Both use the same underlying MPS kernels, but PyTorch's C++ version has thread-safety issues.

### Evidence
- Setting/unsetting `MPS_FORCE_GRAPH_PATH` does not change behavior (28/30 either way)
- The exact same mathematical operations pass when implemented in Python
- The exact same striding patterns pass when implemented in Python
- Only PyTorch's C++ codepath fails

## Implications

1. **Apple MPS is thread-safe**: Individual MPS operations are correct in parallel
2. **PyTorch MPS backend has bugs**: The C++ wrapper for MHA has race conditions
3. **Workaround confirmed**: Our BatchQueue with num_workers=1 is correct solution

## Deep Dive: The `.contiguous()` Bug

### Additional Tests

Created deeper investigation tests:
- `tests/isolate_projection_pattern.py`
- `tests/isolate_contiguous.py`

| Test | Pass Rate | Status |
|------|-----------|--------|
| With .contiguous() (PyTorch pattern) | 28/30 | FAIL |
| **Without .contiguous()** | **30/30** | **PASS** |
| Simple chunk | 30/30 | PASS |
| Extra contiguous() on q,k,v | 30/30 | PASS |
| Clone instead of contiguous | 17/30 | FAIL |

### Precise Root Cause

The bug is triggered by:
1. `.contiguous()` on complex reshaped 4D tensors (3, B, L, E)
2. `.clone()` operations on similar tensors

The bug is **NOT** triggered by:
- Simple `.chunk()` operations (views into existing storage)
- `.contiguous()` on smaller individual tensors
- The reshape operations themselves

### Why PyTorch's MHA Fails

PyTorch's `_in_projection_packed` in `torch/nn/functional.py:5698-5708`:

```python
proj = linear(q, w, b)
proj = proj.unflatten(-1, (3, E)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
#                                                                          ^^^^^^^^^^^
# This .contiguous() triggers the race!
```

The `.contiguous()` call creates a new tensor copy, and multiple threads
racing to allocate/copy triggers the bug.

### Proposed Fix

Remove `.contiguous()` from `_in_projection_packed` and use `.reshape()` instead of `.view()`:

```python
proj = proj.unflatten(-1, (3, E)).unsqueeze(0).transpose(0, -2).squeeze(-2)
# Use .reshape() downstream instead of .view()
```

This avoids the memory allocation race while maintaining correct results.

## Recommended Next Steps

1. **File PyTorch issue**: Report bug in `_in_projection_packed` - the `.contiguous()` call
2. **Propose fix**: Remove `.contiguous()` for MPS device, use `.reshape()` instead
3. **File Apple Radar**: Report MPS memory allocation race in parallel tensor copies

## Files Created

- `tests/isolate_mps_operation.py` - Operation-level isolation
- `tests/isolate_sdpa_components.py` - SDPA component isolation
- `tests/isolate_mha_internals.py` - MHA internal isolation
- `tests/isolate_mha_functional.py` - Functional vs manual comparison
- `tests/isolate_projection_pattern.py` - PyTorch projection pattern test
- `tests/isolate_contiguous.py` - `.contiguous()` bug isolation

## Test Commands

```bash
# Reproduce findings
python3 tests/isolate_mps_operation.py --threads 8 --iterations 30
python3 tests/isolate_sdpa_components.py --threads 8 --iterations 30
python3 tests/isolate_mha_internals.py --threads 8 --iterations 30
python3 tests/isolate_mha_functional.py --threads 8 --iterations 30

# Minimal standalone reproduction (suitable for bug report)
python3 tests/minimal_mps_contiguous_race.py
```

## Framework Analysis (N=1271 Update)

### Apple Framework Threading Model

Per Apple's Metal documentation:
- `MTLDevice` is thread-safe (can be shared across threads)
- `MTLCommandQueue` is thread-safe (multiple queues can execute in parallel)
- `MTLCommandBuffer` is NOT thread-safe (must be encoded from single thread)
- MPS operations use internal command buffers and should be thread-safe

### Framework Symbol Analysis

Apple's Metal and MPS frameworks on macOS 15 are in the dyld shared cache,
making direct symbol analysis difficult. The frameworks export Objective-C
classes (MTLDebug*, MTLCounters*, MPS*) but internal synchronization
primitives are not exported.

### Conclusion

The race condition is NOT in Apple's Metal compute operations. Individual
MPS kernels (matmul, softmax, linear, etc.) are thread-safe. The bug is
in PyTorch's memory allocation/copy path when `.contiguous()` is called
from multiple threads.
