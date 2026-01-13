# Verification Rounds 2186-2210

**Worker**: N=2955
**Date**: 2025-12-23 12:12-12:14 PST
**AGX Fix Version**: v2.4 NR (Never-Release)
**System**: Apple M4 Max, macOS 15.7.3, Metal 3

## Results

| Round Range | Passed | Failed | Pass Rate |
|-------------|--------|--------|-----------|
| 2186-2200   | 15     | 0      | 100%      |
| 2201-2210   | 10     | 0      | 100%      |
| **Total**   | **25** | **0**  | **100%**  |

## 8-Thread Efficiency Measurements

| Round | Efficiency |
|-------|------------|
| 2186  | 15.0%      |
| 2187  | 16.1%      |
| 2188  | 16.3%      |
| 2189  | 14.6%      |
| 2190  | 14.8%      |
| 2191  | 15.1%      |
| 2192  | 13.5%      |
| 2193  | 15.5%      |
| 2194  | 15.1%      |
| 2195  | 14.3%      |
| 2196  | 15.2%      |
| 2197  | 16.1%      |
| 2198  | 15.1%      |
| 2199  | 14.6%      |
| 2200  | 14.7%      |
| 2201  | 14.9%      |
| 2202  | 13.6%      |
| 2203  | 14.6%      |
| 2204  | 15.4%      |
| 2205  | 16.2%      |
| 2206  | 15.5%      |
| 2207  | 14.0%      |
| 2208  | 15.1%      |
| 2209  | 14.8%      |
| 2210  | 14.6%      |
| **Mean** | **15.0%** |
| **Range** | 13.5-16.3% |

## Thread Safety

All 25 rounds completed:
- 4000 thread-safe operations (160 ops/round x 25 rounds)
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
| N=2954  | -       | 25 | 25 | 100% |
| N=2955  | Current | 25 | 25 | 100% |
| **Total** | - | **130** | **130** | **100%** |

## Conclusion

v2.4 NR maintains 100% reliability over 130 cumulative verification rounds.
The 8-thread efficiency ceiling (~13-16%) remains consistent with documented behavior.
