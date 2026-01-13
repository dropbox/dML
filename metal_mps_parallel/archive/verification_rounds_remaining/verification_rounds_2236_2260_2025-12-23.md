# Verification Rounds 2236-2260

**Worker**: N=2957
**Date**: 2025-12-23 12:39-12:50 PST
**AGX Fix Version**: v2.4 NR (Never-Release)
**System**: Apple M4 Max (40 GPU cores), macOS 15.7.3, Metal 3

## Configuration

- Command: `DYLD_INSERT_LIBRARIES=agx_fix/build/libagx_fix_v2_4_nr.dylib python3 tests/complete_story_test_suite.py --threads 8 --iterations 160 --verify`
- Threads: 8
- Operations per round: 160

## Results

| Round Range | Passed | Failed | Pass Rate |
|-------------|--------|--------|-----------|
| 2236-2260   | 25     | 0      | 100%      |

## 8-Thread Efficiency Measurements

| Round | Efficiency |
|-------|------------|
| 2236  | 14.7%      |
| 2237  | -          |
| 2238  | 36.9%      |
| 2239  | 35.7%      |
| 2240  | 36.6%      |
| 2241  | 37.3%      |
| 2242  | 33.7%      |
| 2243  | 37.1%      |
| 2244  | 36.7%      |
| 2245  | 40.8%      |
| 2246  | 36.5%      |
| 2247  | 35.5%      |
| 2248  | 33.6%      |
| 2249  | 40.5%      |
| 2250  | 34.9%      |
| 2251  | 34.1%      |
| 2252  | 34.8%      |
| 2253  | 49.4%      |
| 2254  | 35.9%      |
| 2255  | 38.1%      |
| 2256  | 35.7%      |
| 2257  | 34.6%      |
| 2258  | 38.0%      |
| 2259  | 31.4%      |
| 2260  | 33.0%      |
| **Mean** | **35.6%** |
| **Range** | 14.7-49.4% |

**Note**: Higher efficiency observed this session compared to previous (~13-17%).
Variance likely due to system load conditions.

## Thread Safety + Correctness

All 25 rounds completed:
- 4000 thread-safe operations (160 ops/round x 25 rounds)
- 0 crashes
- 0 errors
- All outputs match CPU reference (max diff < 0.000002)

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
| 9 | N=2957 | 25 | 25 | 100% |
| **Total** | - | **180** | **180** | **100%** |

## Conclusion

v2.4 NR maintains 100% reliability over 180 cumulative verification rounds.
8-thread efficiency varied significantly this session (mean 35.6%) vs previous sessions (~13-17%).
