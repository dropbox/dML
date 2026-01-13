# Verification Report N=1976

**Date**: 2025-12-22 20:00 PST
**Worker**: N=1976
**PyTorch Version**: 2.9.1a0+git8cfbcc8
**Hardware**: Apple M4 Max (40 cores)

## Summary

Verified native PyTorch 2.9.1 stability and confirmed v2.3 dylib is counterproductive. Native PyTorch is 100% stable; v2.3 dylib causes 10% crash rate.

## Test Results

### Quick Soak Test (15s, 8 threads)
| Metric | Result |
|--------|--------|
| Total ops | 123,101 |
| Errors | 0 |
| Throughput | 8,204 ops/s |
| Result | **PASS** |

### LayerNorm/Transformer Stress Test
| Test | Threads | Iterations | Success |
|------|---------|------------|---------|
| LayerNorm | 8 | 50 | 400/400 (100%) |
| Transformer | 8 | 20 | 160/160 (100%) |

### Scaling Curves
| Threads | Ops/s | Efficiency |
|---------|-------|------------|
| 1 | 2,362 | 23.3% |
| 4 | 3,915 | 9.6% |
| 8 | 4,285 | 5.3% |

Batching scales to 149x at batch 256.

### Complete Story Test Suite
| Chapter | Result |
|---------|--------|
| Thread Safety | PASS |
| Efficiency Ceiling | PASS (14.3% at 8T) |
| Batching Advantage | PASS |
| Correctness | PASS |

### v2.3 Dylib Impact (20 rounds each)
| Environment | TransformerEncoderLayer (8T x 20) | Pass Rate |
|-------------|-----------------------------------|-----------|
| Native PyTorch | 20/20 | **100%** |
| v2.3 with blit encoder | 18/20 | 90% |

**Conclusion**: v2.3 dylib introduces instability (10% crash rate vs 0% native).

### Blit Encoder Implementation Test
- Rebuilt v2.3 with blit encoder swizzles (uncommitted changes exist in working tree)
- Tested with full blit encoder coverage (blitCommandEncoder factory, fillBuffer, copyFromBuffer, etc.)
- Result: No improvement (still 90% vs native 100%)

## Key Findings

1. **Native PyTorch 2.9.1 is 100% stable** at 8 threads for eval-mode TransformerEncoderLayer (20/20 rounds)
2. **v2.3 dylib is counterproductive** - causes 10% crash rate vs 0% native, even with blit encoder coverage
3. **`.eval()` mode is REQUIRED** - training mode triggers MTLCommandBuffer race condition
4. **Binary patch deployment blocked** - SIP is enabled

## Critical Warning

**DO NOT use v2.3 dylib on PyTorch 2.9.1**. The dylib causes crashes that don't occur with native PyTorch. The blit encoder implementation in working tree was tested and provides no improvement.

## Blockers

- SIP enabled - binary patch deployment blocked

## Recommendations

1. Use native PyTorch 2.9.1 without dylib
2. Always use `.eval()` mode for multi-threaded inference
3. Use batching instead of threading for parallelism
4. Wait for user to disable SIP for binary patch testing
