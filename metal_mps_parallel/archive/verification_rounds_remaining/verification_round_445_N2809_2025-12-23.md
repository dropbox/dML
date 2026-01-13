# Verification Round 445

**Worker**: N=2809
**Date**: 2025-12-23
**Result**: CLEAN - 0 new bugs found in 3 attempts

## Verification Attempts

### Attempt 1: MTLSize Passing

MTLSize parameter passing:

| Aspect | Status |
|--------|--------|
| Struct size | 24 bytes (3 × NSUInteger) |
| Passing convention | By value |
| Stack alignment | ARM64 handles |

MTLSize correctly passed by value.

**Result**: No bugs found - MTLSize passing correct

### Attempt 2: MTLRegion Passing

MTLRegion parameter passing:

| Aspect | Status |
|--------|--------|
| Struct size | 48 bytes (origin + size) |
| Passing convention | By value |
| Stack alignment | ARM64 handles |

MTLRegion correctly passed by value.

**Result**: No bugs found - MTLRegion passing correct

### Attempt 3: NSRange Passing

NSRange parameter passing:

| Aspect | Status |
|--------|--------|
| Struct size | 16 bytes (location + length) |
| Passing convention | By value |
| ABI compliance | Yes |

NSRange correctly passed by value.

**Result**: No bugs found - NSRange passing correct

## Summary

3 consecutive verification attempts with 0 new bugs found.

**269 consecutive clean rounds** since the MAX_SWIZZLED fix.

Total verification effort: 801 rigorous attempts across 269 rounds.

