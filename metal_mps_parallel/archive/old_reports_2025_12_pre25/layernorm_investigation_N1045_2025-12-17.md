# LayerNorm Non-Determinism Investigation Report

**Worker:** N=1045
**Date:** 2025-12-17
**Status:** CRITICAL FINDING

## UPDATE (2025-12-17): Root Cause + Fix

The observed LayerNorm non-determinism and large numerical mismatches were caused by a bug in our
`layer_norm_mps_graph()` implementation (not an inherent Apple/MPS non-determinism):

- `mean`/`rstd` output tensors were allocated without the keep-dims shape produced by MPSGraph reductions,
  causing a shape mismatch when writing results (partial/zeroed outputs and apparent non-determinism).
- Fix: allocate `mean`/`rstd` with the keep-dims `stat_shape` and broadcast stats correctly.

Verification after fix (patched PyTorch):
- `python3 tests/verify_layernorm_fix.py`: PASS (diff=0.0)
- `python3 tests/correctness_benchmark.py --parallel`: 40/40 PASS
- `MPS_FORCE_GRAPH_PATH=1` + `aten.native_layer_norm`: matches CPU (max_abs_diff ≈ 2e-06)

The remainder of this report is historical context from before the fix and is superseded by the update above.

---

## Summary

Investigation reveals that MPS LayerNorm is **fundamentally non-deterministic** on Apple Silicon.
Both the Metal kernel path AND the MPSGraph path produce different results on consecutive calls,
even within the same thread with identical inputs.

## Test Results

### Same Thread, Same Input, Consecutive Calls

| Operation | Max Diff | Status |
|-----------|----------|--------|
| Linear    | 0.0      | PASS   |
| Softmax   | 0.0      | PASS   |
| GELU      | 0.0      | PASS   |
| LayerNorm | ~0.14    | FAIL   |

### Test Code

```python
# With MPS_FORCE_GRAPH_PATH=1 (confirmed graph path used via debug output)
# Same input tensor, same LayerNorm layer, same thread
# 5 consecutive calls produce different results (max_diff=0.11-0.14)
```

### Debug Output

```
[DEBUG LayerNorm] force_graph_path_env=1, parallel_streams_active=0, use_graph_path=1
```

This confirms the MPSGraph path IS being used, yet non-determinism persists.

## Key Finding

**The N=931 fix hypothesis was incorrect.**

The hypothesis was:
- Metal kernel path has thread-affinity bug
- MPSGraph path is thread-safe and deterministic
- Solution: Use MPSGraph path for parallel execution

The reality is:
- **Both paths are non-deterministic**
- Non-determinism occurs even in single-threaded execution
- Non-determinism occurs on consecutive calls with identical inputs
- This appears to be a fundamental Apple MPS bug, not a threading issue

## Implications

1. **The LayerNorm fix (N=931) does not solve the correctness issue**
2. **The 32/40 correctness benchmark failures are expected and cannot be fixed with current approach**
3. **This is likely an Apple MPS framework bug that needs to be reported to Apple**

## Potential Root Causes

1. **Floating-point accumulation order:** LayerNorm involves reduction operations (mean, variance) that may use non-deterministic GPU parallel reduction strategies
2. **Metal shader compilation non-determinism:** JIT compilation may produce different code paths
3. **GPU workgroup scheduling:** Different workgroup scheduling may affect accumulation order

## Recommendations

1. **File Apple bug report** with minimal reproduction case
2. **Document in README** that LayerNorm has known non-determinism on MPS
3. **Consider CPU fallback** for applications requiring deterministic LayerNorm
4. **Investigate torch.use_deterministic_algorithms()** - may not help for MPS

## Test Artifacts

- `tests/correctness_benchmark.py --parallel`: 32/40 pass (LayerNorm/TransformerBlock fail)
- `tests/verify_layernorm_fix.py`: Would fail due to non-determinism
- Debug output confirms MPSGraph path selection

## Evidence

Tested on:
- Hardware: Apple M4 Max
- PyTorch: 2.9.1a0+git9a7876e (patched fork)
- Metal: Metal 3
- macOS: 15.7.2
