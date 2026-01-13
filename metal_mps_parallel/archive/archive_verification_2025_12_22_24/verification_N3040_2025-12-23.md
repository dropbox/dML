# Verification Report N=3040

**Date**: 2025-12-23
**Worker**: N=3040
**Platform**: Apple M4 Max, macOS 15.7.3

## Test Results

All tests executed with crash-checking wrapper (`run_test_with_crash_check.sh`)
and AGX fix dylib (v2.5 + Semaphore(2) throttling via `MPS_USE_AGX_FIX=1`).

### test_semaphore_recommended

| Throttle Type | Completed | Elapsed | Ops/s | Speedup | Status |
|---------------|-----------|---------|-------|---------|--------|
| Lock          | 400/400   | 0.43s   | 920   | 1.00x   | PASS   |
| Semaphore(2)  | 400/400   | 0.38s   | 1052  | 1.14x   | PASS   |

**Result**: Semaphore(2) provides 14% throughput improvement over Lock

### complete_story_test_suite

| Chapter | Status | Details |
|---------|--------|---------|
| thread_safety | PASS | 160/160 ops, 8 threads, 0 crashes |
| efficiency_ceiling | PASS | 15.9% at 8 threads |
| batching_advantage | PASS | 6537 samples/s batched vs 1030 threaded |
| correctness | PASS | max diff < 1e-6 |

### soak_test_quick

| Metric | Value |
|--------|-------|
| Duration | 60s |
| Total ops | 492,847 |
| Throughput | 8213 ops/s |
| Errors | 0 |
| Status | PASS |

## Crash Status

- **Crashes before tests**: 259
- **Crashes after tests**: 259
- **New crashes**: 0

## Dylib Verification

- **Dylib**: libagx_fix_v2_5.dylib
- **MD5**: 9768f99c81a898d3ffbadf483af9776e

## Summary

All verification tests pass with 0 new crashes. The Semaphore(2) throttling
approach continues to provide stable 0-crash operation with measurable
throughput improvement over full serialization.

Binary patch deployment (for full parallelism) awaits user action to disable SIP.
