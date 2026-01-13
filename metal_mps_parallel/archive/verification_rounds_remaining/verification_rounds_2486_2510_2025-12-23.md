# Verification Rounds 2486-2510

**Date**: 2025-12-23
**Worker**: N=2967
**v2.3 NR (No Regression)**

## Summary

| Metric | Value |
|--------|-------|
| Rounds | 25 |
| Passed | 25 |
| Pass Rate | **100%** |
| Thread Safety Ops | 4000/4000 (160 ops/round x 25 rounds) |
| Crashes | 0 |

## Efficiency Metrics (Final Round)

| Thread Count | Throughput (ops/s) | Efficiency |
|--------------|-------------------|------------|
| 1 | 579.4 | 100.0% |
| 2 | 818.6 | 70.7% |
| 4 | 861.2 | 37.2% |
| 8 | 670.1 | 14.5% |

8-thread efficiency: **14.5%** (matches documented ~13% ceiling)

## Correctness

- Max diff from CPU reference: 0.000002
- Tolerance: 0.001
- Status: **PASS**

## Batching Advantage

- Batched (batch=8): 5992 samples/s
- Threaded (8 threads): 743 samples/s
- Batching advantage: **8.1x**

## Cumulative v2.3 NR Results

| Sessions | Rounds | Passed | Pass Rate |
|----------|--------|--------|-----------|
| 19 | 430 | 430 | **100%** |

## Notes

- All tests run with `DYLD_INSERT_LIBRARIES=agx_fix/build/libagx_fix_v2_3.dylib`
- Platform: macOS 15.7.3, Apple M4 Max (40 GPU cores)
- PyTorch: 2.9.1a0+gitf44c036
