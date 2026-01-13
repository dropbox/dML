# Verification Rounds 2286-2310

**Worker**: N=2959
**Date**: 2025-12-23 12:48-13:02 PST
**AGX Fix Version**: v2.4 NR (Never-Release)
**System**: Apple M4 Max (40 GPU cores), macOS 15.7.3, Metal 3

## Configuration

- Command: `DYLD_INSERT_LIBRARIES=agx_fix/build/libagx_fix_v2_4_nr.dylib python3 tests/complete_story_test_suite.py --threads 8 --iterations 160 --verify`
- Threads: 8
- Operations per round: 160

## Results

| Round Range | Passed | Failed | Pass Rate |
|-------------|--------|--------|-----------|
| 2286-2310   | 25     | 0      | 100%      |

## 8-Thread Efficiency Measurements

| Round | Efficiency |
|-------|------------|
| 2286  | 15.5%      |
| 2287  | 14.7%      |
| 2288  | 14.8%      |
| 2289  | 15.1%      |
| 2290  | 14.9%      |
| 2291  | 14.0%      |
| 2292  | 15.1%      |
| 2293  | 15.0%      |
| 2294  | 15.1%      |
| 2295  | 14.5%      |
| 2296  | 15.0%      |
| 2297  | 14.9%      |
| 2298  | 13.8%      |
| 2299  | 14.2%      |
| 2300  | 14.3%      |
| 2301  | 14.8%      |
| 2302  | 13.9%      |
| 2303  | 14.8%      |
| 2304  | 14.6%      |
| 2305  | 16.5%      |
| 2306  | 14.1%      |
| 2307  | 14.3%      |
| 2308  | 15.0%      |
| 2309  | 14.0%      |
| 2310  | 14.2%      |
| **Mean** | **14.7%** |
| **Range** | 13.8-16.5% |

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
| 10 | N=2958 | 25 | 25 | 100% |
| 11 | N=2959 | 25 | 25 | 100% |
| **Total** | - | **230** | **230** | **100%** |

## Observations

Efficiency this session (mean 14.7%) is consistent with the documented ~13-15% ceiling.
v2.4 NR maintains rock-solid reliability with 230 cumulative verification rounds at 100% pass rate.
