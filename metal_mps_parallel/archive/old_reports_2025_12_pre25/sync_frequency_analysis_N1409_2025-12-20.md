# Synchronization Frequency Analysis

**Worker N=1409**
**Date: 2025-12-20**

> **CORRECTION (N=1410)**: The 6.8x speedup below used a simpler model than the standard
> 3-layer benchmark. The benchmark model achieves only **2.4x** improvement. See
> `reports/main/sync_frequency_correction_N1410_2025-12-20.md` for details.

## Executive Summary

**Key Finding**: Synchronization frequency dominates MPS performance more than threading pattern or serialization.

Reducing sync frequency from every-op to every-256-ops improves throughput by **6.8x** (1-layer model) or **2.4x** (3-layer benchmark model).

## Measurements

| Sync Frequency | Ops/s | Speedup vs every-op |
|----------------|-------|---------------------|
| Every 1 op | 5,610 | 1.0x (baseline) |
| Every 2 ops | 10,433 | 1.9x |
| Every 4 ops | 14,343 | 2.6x |
| Every 8 ops | 19,785 | 3.5x |
| Every 16 ops | 27,164 | 4.8x |
| Every 32 ops | 31,078 | 5.5x |
| Every 64 ops | 34,968 | 6.2x |
| Every 128 ops | 37,044 | 6.6x |
| Every 256 ops | 38,329 | **6.8x** |

**Hardware**: Apple M4 Max, macOS 15.7.3, PyTorch 2.9.1

## Implications

### Why Batching is 33x Better Than Threading

The "batching is 33x better" finding is primarily due to sync frequency:

- **Threading benchmark**: Sync after every op (5,610 ops/s)
- **Batching benchmark**: Batch 256, sync once (equivalent to 38K ops/s per batch)

### The "93% Threading Overhead" is Misleading

The claimed "93% threading overhead" compared:
- Single-op without sync: ~35,000 ops/s
- Threaded with per-op sync: ~6,000 ops/s

This isn't threading overhead - it's **synchronization overhead**. A direct loop with per-op sync shows the same ~6,000 ops/s.

### Raw Metal vs MPS Gap Explained

The gap between raw Metal (62% efficiency) and MPS (34% efficiency) at 8 threads likely includes:
1. MPS kernel dispatch overhead
2. PyTorch tensor management overhead
3. Python/GIL overhead

But the **dominant factor** in absolute throughput is sync frequency, not parallelism efficiency.

## Recommendations

### For Maximum Throughput

```python
# DON'T sync every op
for x in inputs:
    y = model(x)
    torch.mps.synchronize()  # BAD - 5,610 ops/s

# DO batch operations
outputs = model(batched_inputs)  # GOOD - benefits from batched sync
torch.mps.synchronize()  # One sync for entire batch
```

### For Multi-Threaded Inference

```python
# If you must sync per result, use event sync
event = torch.mps.Event(enable_timing=False)
output = model(input)
event.record()
event.synchronize()  # Still ~6K ops/s but better than device sync

# Better: accumulate N results then sync
batch = []
for i, input in enumerate(inputs):
    batch.append(model(input))
    if (i + 1) % 32 == 0:  # Sync every 32 ops
        torch.mps.synchronize()
        # Process batch results
```

## Conclusion

**Synchronization frequency is the dominant performance factor.**

The serialization in MPS (34% vs 62% efficiency) is real but secondary. Users get much more benefit from reducing sync frequency than from optimizing thread patterns.

Batching remains the best approach because it naturally reduces sync frequency.

## Files

- Test script: Inline Python in N=1409 commit
- Related: `docs/PYTORCH_UPSTREAM_ISSUE.md` - Updated with raw Metal comparison
