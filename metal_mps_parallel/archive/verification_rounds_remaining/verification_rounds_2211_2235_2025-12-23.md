# Verification Rounds 2211-2235

**Date**: 2025-12-23
**Worker**: N=2956
**Result**: ALL PASS (25/25)

## Session Summary

| Metric | Value |
|--------|-------|
| Rounds | 25 (2211-2235) |
| Pass rate | 100% |
| Operations | 4000 (160/round x 25) |
| Crashes | 0 |
| 8-thread efficiency | 13.2% - 16.8% (mean ~14.9%) |
| Max diff from CPU | < 0.000001 |

## Round Details

All 25 rounds:
- Thread safety: PASS (160/160 operations per round)
- Efficiency ceiling: CONFIRMED (~14-17% at 8 threads)
- Batching advantage: CONFIRMED
- Correctness: PASS (outputs match CPU reference)

Efficiency samples:
- Round 1: 14.3%
- Round 2: 13.8%
- Round 3: 13.2%
- Round 4: 15.7%
- Round 5: 14.7%
- Round 6: 15.3%
- Round 7: 14.6%
- Round 8-10: ~13.8-14%
- Round 11-25: 14.0% - 16.8%

## Cumulative v2.4 NR Results

| Session | Worker | Rounds | Passed | Pass Rate |
|---------|--------|--------|--------|-----------|
| 1 | N=2949 | 10 | 10 | 100% |
| 2 | N=2950 | 20 | 20 | 100% |
| 3 | N=2951 | 10 | 10 | 100% |
| 4 | N=2952 | 20 | 20 | 100% |
| 5 | N=2953 | 20 | 20 | 100% |
| 6 | N=2954 | 25 | 25 | 100% |
| 7 | N=2955 | 25 | 25 | 100% |
| 8 | N=2956 | 25 | 25 | 100% |
| **Total** | - | **155** | **155** | **100%** |

## Conclusion

v2.4 NR maintains perfect reliability over 155 cumulative verification rounds.
The efficiency ceiling (~14-17% at 8 threads) remains hardware-bound.
