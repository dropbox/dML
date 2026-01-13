# Verification Report N=1973

**Date**: 2025-12-22 19:37 PST
**Worker**: N=1973
**PyTorch Version**: 2.9.1a0+git8cfbcc8
**Hardware**: Apple M4 Max (40 cores)

## Summary

Fresh verification of native PyTorch 2.9.1 MPS threading stability. All tests pass except for intermittent crashes in complex sequential test runs.

## Test Results

### Simple Matmul Soak Test (20s, 8 threads)
| Metric | Result |
|--------|--------|
| Total ops | 127,818 |
| Duration | 20.5s |
| Throughput | 6,232 ops/s |
| Errors | 0 |
| Result | **PASS (100%)** |

### TransformerEncoderLayer (10 isolated rounds)
| Metric | Result |
|--------|--------|
| Passed | 9/10 |
| Config | 8 threads x 20 iterations |
| Pass Rate | **90%** |

### Efficiency Ceiling Test
| Threads | Throughput | Speedup | Efficiency |
|---------|------------|---------|------------|
| 1 | 618.3 | 1.00x | 100.0% |
| 2 | 1025.7 | 1.66x | 83.0% |
| 4 | 959.9 | 1.55x | 38.8% |
| 8 | 726.1 | 1.17x | **14.7%** |

Efficiency at 8 threads: 14.7% (matches documented ~13% ceiling)

### Multi-Model Test (8 models, 8 threads)
| Metric | Result |
|--------|--------|
| Completed | 160/160 |
| Time | 0.22s |
| Throughput | 727.5 ops/s |
| Result | **PASS** |

## Findings

1. **Simple operations stable**: Matmul at 8 threads shows 100% stability (127K ops, 0 errors)

2. **TransformerEncoderLayer ~90% stable**: 9/10 rounds pass when isolated in separate processes

3. **Efficiency ceiling confirmed**: 14.7% at 8 threads matches documented ~13% ceiling

4. **Complete story suite has intermittent crashes**: Running all chapters sequentially causes crashes (~40% pass rate from N=1972)

## Blockers

- **SIP enabled**: Binary patch deployment (Tasks 3-4) requires user to disable SIP
- **Layer norm patch unapplied**: `patches/040-layer-norm-tensor-lifetime-fix.patch` requires PyTorch rebuild

## Recommendations

1. **DO NOT use v2.3 dylib** - counterproductive on PyTorch 2.9.1+
2. **Binary patch blocked** - waiting for user to disable SIP
3. **Layer norm patch** - requires PyTorch source rebuild to test
4. **Use isolated tests** - running chapters in separate processes is more stable
