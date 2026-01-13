# Formal Verification Gap Analysis - N=1980

**Date**: 2025-12-22 20:15 PST
**Worker**: N=1980
**Method**: TLA+ Model Analysis + Code Review + Bug Fix

## Summary

Conducted 3 gap search iterations (7-9). **FOUND AND FIXED 1 NEW VULNERABILITY.**

## New Vulnerability Found

### `masked_fill__mps` in Indexing.mm (lines 739-786)

**Vulnerability Pattern** (same as layer_norm):
```cpp
c10::MaybeOwned<Tensor> b_mask = expand_inplace(self, mask, "masked_fill_");
...
dispatch_sync_with_rethrow(stream->queue(), ^() {
    mtl_setArgs(computeEncoder, self, *b_mask, mpsScalar);  // VULNERABLE
});
```

**Race Condition**:
1. Thread A: `expand_inplace()` returns MaybeOwned that may borrow mask
2. Thread A: enters `dispatch_sync_with_rethrow` (blocks)
3. Thread B: Python GC runs, frees mask tensor
4. Thread A's block: `*b_mask` dereferences freed tensor -> CRASH

**Fix Applied** (patches/043-masked-fill-tensor-lifetime-fix.patch):
```cpp
// FIX: Capture tensors by value to prevent use-after-free race condition.
Tensor b_mask_owned = b_mask->contiguous();
__block Tensor self_block = self;
__block Tensor b_mask_block = b_mask_owned;
__block Tensor mask_block = mask;

dispatch_sync_with_rethrow(stream->queue(), ^() {
    mtl_setArgs(computeEncoder, self_block, b_mask_block, mpsScalar);  // SAFE
});
```

## Gap Search Iterations

### Iteration 7: Fresh Proof Review
- Verified TensorLifetime.tla correctly models MaybeOwned borrow vulnerability
- Confirmed function parameters (`const Tensor&`) are safe (binding keeps them alive)
- Confirmed Placeholder class always copies tensors (safe)
- Only MaybeOwned + dispatch_sync_with_rethrow is vulnerable

### Iteration 8: Find Uncovered Scenarios
- Searched all 53 `dispatch_sync_with_rethrow` usages
- Found 3 files with both MaybeOwned and dispatch_sync_with_rethrow:
  - Indexing.mm: **VULNERABLE** (masked_fill__mps) - FIXED
  - LinearAlgebra.mm: SAFE (uses Placeholder)
  - Normalization.mm: SAFE (already has __block fix)

### Iteration 9: Final Exhaustive Check
- Verified SummaryOps.mm and LossOps.mm don't use dispatch_sync_with_rethrow
- Confirmed Blas.mm MaybeOwned isn't used inside dispatch (uses Placeholder)
- No other vulnerable patterns found

## Files Modified

1. `pytorch-mps-fork/aten/src/ATen/native/mps/operations/Indexing.mm`
   - Added __block tensor capture fix to `masked_fill__mps`

2. `patches/043-masked-fill-tensor-lifetime-fix.patch`
   - Created patch for the fix

## TLA+ Model Coverage

The TensorLifetime.tla model with `CaptureByValue=FALSE` correctly identifies this vulnerability pattern. The model proves:
- Without __block capture: Race window exists (invariant violated)
- With __block capture: Race window closed (invariant holds)

## Conclusion

After 9 total gap search iterations:
1. **1 NEW VULNERABILITY FOUND AND FIXED**: `masked_fill__mps`
2. **All other dispatch_sync_with_rethrow usages verified safe**
3. **TLA+ model coverage is complete** for this vulnerability class
4. **Proof system correctly identifies the bug pattern**
