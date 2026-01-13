# Verification Rounds N=2995-2997

**Date**: 2025-12-23 **Result**: 1 NEW BUG FOUND
3 rounds: 58T+55T+60T, 9521 ops, 1 new error
**Consecutive clean**: RESET (new bug found)

## Round 175 Details (6 Attempts)

| # | Category | Result |
|---|----------|--------|
| 1 | SWIZZLE macro expansion | CORRECT |
| 2 | TLA+ state transitions | CORRECT |
| 3 | Array bounds in store_original_imp | **NEW LOW BUG** |
| 4 | Binary patch file I/O | CORRECT |
| 5 | Namespace static initialization | SAFE |
| 6 | Exception safety in retain_encoder_on_creation | CONFIRMED R20 |

## NEW BUG: MAX_SWIZZLED Array Overflow

**Location**: agx_fix_v2_3.mm lines 81-99

**Issue**: MAX_SWIZZLED = 64, but total swizzle calls = 76

**Impact**:
- Last 12 entries silently dropped from g_swizzled_sels/g_original_imps arrays
- Affected methods call get_original_imp(), get NULL, silently skip original
- Affected encoder types: render, resource_state, accel_struct

**Severity**: LOW
- Compute and blit encoders (PyTorch MPS) swizzled first (within 64 limit)
- Affected encoders not used by PyTorch MPS

**Recommended Fix**: Increase MAX_SWIZZLED to 128

## Known Bugs Re-confirmed
- Round 20: OOM exception safety (CFRetain before insert)
