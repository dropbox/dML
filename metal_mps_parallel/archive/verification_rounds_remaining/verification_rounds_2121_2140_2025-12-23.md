# Verification Report: Rounds 2121-2140

**Worker**: N=2952
**Date**: 2025-12-23
**AGX Fix Version**: v2.4 NR (Never-Release)

## Summary

Completed 20 verification rounds with 100% pass rate.

| Metric | Result |
|--------|--------|
| Rounds | 2121-2140 (20 total) |
| Pass Rate | 100% (20/20) |
| Thread Safety | 160/160 ops per round |
| 8-Thread Efficiency | ~14.9% (consistent with ~13% ceiling) |
| Correctness | All outputs match CPU reference |

## Round Details

All 20 rounds passed all 4 test chapters:
- CHAPTER 1: Thread Safety (160/160 operations, no crashes)
- CHAPTER 2: Efficiency Ceiling (14.9% at 8 threads)
- CHAPTER 3: Batching Advantage (~7600 samples/s batched vs ~770 threaded)
- CHAPTER 4: Correctness (max diff < 0.000001, tolerance 0.001)

## Final Round (2140) Detailed Results

```
Thread Count | Throughput | Speedup | Efficiency
-------------------------------------------------------
     1       |   632.4    |  1.00x  |  100.0%
     2       |  1035.5    |  1.64x  |  81.9%
     4       |   967.9    |  1.53x  |  38.3%
     8       |   754.2    |  1.19x  |  14.9%

Batched (1 thread, batch=8)  |    7602.8 |  baseline
Threaded (8 threads, batch=1)|     768.5 |  0.10x
Threaded (4 threads, batch=2)|    2190.8 |  0.29x
```

## Cumulative v2.4 NR Results

| Session | Worker | Rounds | Passed | Pass Rate |
|---------|--------|--------|--------|-----------|
| N=2949 | 2949 | 10 | 10 | 100% |
| N=2950 | 2950 | 20 | 20 | 100% |
| N=2951 | 2951 | 10 | 10 | 100% |
| N=2952 | 2952 | 20 | 20 | 100% |
| **Total** | - | **60** | **60** | **100%** |

## Conclusion

v2.4 NR (Never-Release) maintains 100% reliability over 60 cumulative verification rounds.
The efficiency ceiling remains at ~14% at 8 threads, which is a hardware/driver limitation.

## Test Command

```bash
DYLD_INSERT_LIBRARIES="/Users/ayates/metal_mps_parallel/agx_fix/build/libagx_fix_v2_4_nr.dylib" \
python3 tests/complete_story_test_suite.py
```
