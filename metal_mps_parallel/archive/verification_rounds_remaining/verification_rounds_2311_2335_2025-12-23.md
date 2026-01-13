# Verification Rounds 2311-2335 (N=2960)

**Date**: 2025-12-23
**Worker**: N=2960
**Version**: v2.4 NR (No Reentrant)

## Results

| Metric | Value |
|--------|-------|
| Rounds | 2311-2335 |
| Passed | 25 |
| Failed | 0 |
| Pass Rate | 100% |

## Metrics from Representative Run

### Thread Safety
- Operations: 160/160 completed
- Threads: 8
- Crashes: 0
- Elapsed: 0.25s

### Efficiency
| Threads | Throughput (ops/s) | Speedup | Efficiency |
|---------|-------------------|---------|------------|
| 1 | 610.1 | 1.00x | 100.0% |
| 2 | 825.8 | 1.35x | 67.7% |
| 4 | 908.3 | 1.49x | 37.2% |
| 8 | 728.6 | 1.19x | 14.9% |

### Correctness
- Avg max diff: 0.000001
- Max diff: 0.000001
- Tolerance: 0.001
- Status: PASS

## Cumulative v2.4 NR Statistics

| Sessions | Rounds | Passed | Pass Rate |
|----------|--------|--------|-----------|
| 12 | 255 | 255 | 100% |

## Notes

All claims verified:
1. Thread safety: 8 threads without crashes
2. Efficiency ceiling: ~14.9% at 8 threads (matches documented ceiling)
3. Batching advantage: Batched throughput > threaded throughput
4. Correctness: Outputs match CPU reference
