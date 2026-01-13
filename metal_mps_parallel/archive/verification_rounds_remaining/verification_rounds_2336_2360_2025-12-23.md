# Verification Rounds 2336-2360 (N=2961)

**Date**: 2025-12-23
**Worker**: N=2961
**Version**: v2.4 NR (No Reentrant)

## Results

| Metric | Value |
|--------|-------|
| Rounds | 2336-2360 |
| Passed | 25 |
| Failed | 0 |
| Pass Rate | 100% |

## Metrics from Representative Runs

### Thread Safety
- Operations: 160/160 completed (per round)
- Total operations: 4000 (25 rounds x 160 ops)
- Threads: 8
- Crashes: 0
- Elapsed: ~0.25s per round

### Efficiency
| Run | Efficiency |
|-----|------------|
| 2336 | 15.1% |
| 2337 | 14.6% |
| 2338 | 15.0% |
| 2339 | 14.3% |
| 2340 | 14.0% |
| 2341 | 14.7% |
| 2342 | 13.3% |
| 2343 | 13.7% |
| 2344 | 15.2% |
| 2345 | 15.7% |
| 2346 | 14.9% |
| 2347 | 14.5% |
| 2348 | 13.9% |
| 2349 | 14.3% |
| 2350 | 15.3% |
| 2351 | 15.0% |
| 2352 | 15.3% |
| 2353 | 15.1% |
| 2354 | 13.9% |
| 2355 | 15.2% |
| 2356 | 14.7% |
| 2357 | 14.7% |
| 2358 | 13.7% |
| 2359 | 15.0% |
| 2360 | 15.0% |

**Mean**: 14.6%
**Range**: 13.3% - 15.7%

### Correctness
- All outputs match CPU reference (max diff < 0.000002)
- Tolerance: 0.001
- Status: PASS

## Cumulative v2.4 NR Statistics

| Sessions | Rounds | Passed | Pass Rate |
|----------|--------|--------|-----------|
| 13 | 280 | 280 | 100% |

## Notes

All claims verified:
1. Thread safety: 8 threads without crashes
2. Efficiency ceiling: ~14.6% at 8 threads (matches documented ceiling)
3. Batching advantage: Batched throughput > threaded throughput
4. Correctness: Outputs match CPU reference
