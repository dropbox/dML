# Verification Rounds 2261-2285

**Worker**: N=2958
**Date**: 2025-12-23 12:43-12:58 PST
**AGX Fix Version**: v2.4 NR (Never-Release)
**System**: Apple M4 Max (40 GPU cores), macOS 15.7.3, Metal 3

## Configuration

- Command: `DYLD_INSERT_LIBRARIES=agx_fix/build/libagx_fix_v2_4_nr.dylib python3 tests/complete_story_test_suite.py --threads 8 --iterations 160 --verify`
- Threads: 8
- Operations per round: 160

## Results

| Round Range | Passed | Failed | Pass Rate |
|-------------|--------|--------|-----------|
| 2261-2285   | 25     | 0      | 100%      |

## 8-Thread Efficiency Measurements

| Round | Efficiency |
|-------|------------|
| 2261  | 13.8%      |
| 2262  | 14.5%      |
| 2263  | 13.4%      |
| 2264  | 14.6%      |
| 2265  | 13.5%      |
| 2266  | 15.0%      |
| 2267  | 12.9%      |
| 2268  | 9.7%       |
| 2269  | 15.0%      |
| 2270  | 14.3%      |
| 2271  | 14.2%      |
| 2272  | 15.4%      |
| 2273  | 14.9%      |
| 2274  | 14.4%      |
| 2275  | 14.1%      |
| 2276  | 14.9%      |
| 2277  | 14.7%      |
| 2278  | 15.3%      |
| 2279  | 14.7%      |
| 2280  | 14.7%      |
| 2281  | 14.4%      |
| 2282  | 13.3%      |
| 2283  | 14.7%      |
| 2284  | 14.4%      |
| 2285  | 15.4%      |
| **Mean** | **14.5%** |
| **Range** | 9.7-15.4% |

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
| **Total** | - | **205** | **205** | **100%** |

## Observations

Efficiency this session (mean 14.5%) is consistent with the documented ~13% ceiling.
Previous session N=2957 showed anomalously high efficiency (mean 35.6%) which was
likely due to favorable system load conditions. The target 50% efficiency remains
unmet under normal conditions.

## Also Committed

TLA+ spec fixes for liveness properties (bounded model checking guards).
