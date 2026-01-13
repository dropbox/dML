# Verification Report N=1978

**Date**: 2025-12-22 18:11 PST
**Worker**: N=1978

## Bug Fixed

**Issue**: `run_mps_test.sh` was using `libagx_fix.dylib` (v1), which crashes at 8 threads.

**Fix**: Updated wrapper to use `libagx_fix_v2_3.dylib` (the stable, recommended version).

**Root cause**: The Makefile builds multiple versions. v1 is the original mutex-per-call approach, v2.3 is the stable version combining retain-from-creation with mutex protection.

## Test Results

### Complete Story Test Suite
| Test | Result |
|------|--------|
| Thread safety (8T x 20) | PASS (160/160) |
| Efficiency | 14.2% at 8T |
| Batching advantage | PASS |
| Correctness | PASS (max diff 1e-6) |

### TLA+ Verification
| Spec | States | Result |
|------|--------|--------|
| AGXDylibFix.tla | 13 | PASS |
| AGXRaceFix.tla | 10 | PASS |

### Stress Tests
| Test | Ops/s | Result |
|------|-------|--------|
| LayerNorm 8T x 50 | 4218 | PASS |
| Transformer 8T x 20 | 1085 | PASS |
| Extended 8T x 100 | 5072 | PASS |
| Max threads 16T x 50 | 5146 | PASS |
| Large tensor 4T x 20 | 2431 | PASS |

### 60-Second Soak Test
- Total ops: 494,629
- Throughput: 8,243 ops/s
- Errors: 0
- Result: PASS

## Summary

Fixed wrapper script bug that caused crashes when using the default AGX fix library. All tests now pass with v2.3.
