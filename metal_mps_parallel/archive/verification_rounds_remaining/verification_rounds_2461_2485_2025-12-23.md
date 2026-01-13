# Verification Rounds 2461-2485

**Worker**: N=2966
**Date**: 2025-12-23
**Dylib**: v2.4 NR (No Retain)

## Results

| Metric | Value |
|--------|-------|
| Rounds | 25 |
| Passed | 25 |
| Pass Rate | 100% |
| Thread Safety Ops | 4,000 (160/round x 25) |
| Crashes | 0 |

## 8-Thread Efficiency

| Round | Efficiency |
|-------|------------|
| 2461 | 14.9% |
| 2462 | 14.9% |
| 2463 | 15.3% |
| 2464 | 16.0% |
| 2465 | 15.6% |
| 2466 | 15.1% |
| 2467 | 15.5% |
| 2468 | 14.0% |
| 2469 | 15.3% |
| 2470 | 15.3% |
| 2471 | 14.8% |
| 2472 | 13.9% |
| 2473 | 15.3% |
| 2474 | 15.1% |
| 2475 | 13.8% |
| 2476 | 14.3% |
| 2477 | 15.0% |
| 2478 | 14.8% |
| 2479 | 15.3% |
| 2480 | 15.2% |
| 2481 | 15.1% |
| 2482 | 13.4% |
| 2483 | 15.4% |
| 2484 | 14.5% |
| 2485 | 14.8% |

**Mean**: ~14.8%
**Range**: 13.4% - 16.0%

## Cumulative v2.4 NR Results

| Sessions | Rounds | Passed | Pass Rate |
|----------|--------|--------|-----------|
| 18 | 405 | 405 | 100% |

## Correctness

All outputs match CPU reference (max diff < 0.000002).

## Test Configuration

- Hardware: Apple M4 Max (40 GPU cores, Metal 3)
- macOS: 15.7.3
- Test suite: `tests/complete_story_test_suite.py`
- Threads: 8
- Operations per round: 160
