# Verification Rounds 2161-2185

**Worker**: N=2954
**Date**: 2025-12-23 12:10-12:15 PST
**AGX Fix Version**: v2.4 NR (Never-Release)
**System**: Apple M4 Max, macOS 15.7.3, Metal 3

## Results

| Round Range | Passed | Failed | Pass Rate |
|-------------|--------|--------|-----------|
| 2161-2180   | 20     | 0      | 100%      |
| 2181-2185   | 5      | 0      | 100%      |
| **Total**   | **25** | **0**  | **100%**  |

## 8-Thread Efficiency Measurements (Rounds 2181-2185)

| Run | Efficiency |
|-----|------------|
| 1   | 14.3%      |
| 2   | 14.0%      |
| 3   | 15.7%      |
| 4   | 14.5%      |
| 5   | 14.4%      |
| **Mean** | **14.6%** |
| **Range** | 14.0-15.7% |

## Thread Safety

All 25 rounds completed:
- 4000 thread-safe operations (160 ops/round × 25 rounds)
- 0 crashes
- 0 errors
- All outputs match CPU reference (max diff < 0.000001)

## Cumulative v2.4 NR Results

| Session | Worker | Rounds | Passed | Pass Rate |
|---------|--------|--------|--------|-----------|
| N=2949  | Initial | 10 | 10 | 100% |
| N=2950  | -       | 20 | 20 | 100% |
| N=2951  | -       | 10 | 10 | 100% |
| N=2952  | -       | 20 | 20 | 100% |
| N=2953  | -       | 20 | 20 | 100% |
| N=2954  | Current | 25 | 25 | 100% |
| **Total** | - | **105** | **105** | **100%** |

## Conclusion

v2.4 NR maintains 100% reliability over 105 cumulative verification rounds.
The 8-thread efficiency ceiling (~14-16%) remains consistent with documented behavior.
