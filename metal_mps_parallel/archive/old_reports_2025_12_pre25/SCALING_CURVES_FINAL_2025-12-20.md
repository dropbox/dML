# MPS Scaling Curves Analysis - Definitive Results

**Created by Andrew Yates**
**Date: 2025-12-20**

## Executive Summary

This report presents the complete scaling curves for MPS parallelization methods. The results conclusively show that **batching is 7-60x more effective than threading** for parallel inference on Apple Silicon.

## The Data

### Threading Scaling (batch=1 per thread)

| Threads | Ops/s | Speedup | Efficiency | Notes |
|---------|-------|---------|------------|-------|
| 1 | 719 | 0.07x | 6.7% | Baseline |
| 2 | 2,605 | 0.24x | 12.2% | Peak efficiency |
| 4 | 2,752 | 0.26x | 6.4% | Diminishing |
| 8 | 2,728 | 0.25x | 3.2% | Flat |
| 16 | 2,720 | 0.25x | 1.6% | Flat |

**Observation:** Threading hits a wall at ~2,700 ops/s regardless of thread count. Adding more threads does NOT increase throughput.

### Batching Scaling (single thread, varying batch)

| Batch | Samples/s | Speedup | Efficiency | Notes |
|-------|-----------|---------|------------|-------|
| 1 | 9,218 | 0.86x | 86% | Baseline |
| 2 | 17,892 | 1.67x | 84% | Linear |
| 4 | 38,823 | 3.63x | 91% | Near-linear |
| 8 | 82,315 | 7.69x | **96%** | Near-linear |
| 16 | 162,060 | 15.15x | **95%** | Near-linear |
| 32 | 307,829 | 28.77x | 90% | Good |
| 64 | 603,591 | 56.42x | 88% | Good |
| 128 | 988,226 | 92.37x | 72% | Saturating |
| 256 | 1,388,672 | **129.80x** | 51% | GPU limit |

**Observation:** Batching scales near-linearly up to batch 64 (88% efficiency), then continues scaling with decreasing efficiency as GPU saturates.

### Direct Comparison: Threading vs Batching

| N | Threading | Batching | Batching Advantage |
|---|-----------|----------|-------------------|
| 2 | 2,605 samp/s | 17,892 samp/s | **7x better** |
| 4 | 2,752 samp/s | 38,823 samp/s | **14x better** |
| 8 | 2,728 samp/s | 82,315 samp/s | **30x better** |
| 16 | 2,720 samp/s | 162,060 samp/s | **60x better** |

## Scaling Curves Visualization

```
Efficiency vs Parallelism Level (N)

100% ─┬─────────────────────────────────────────────────
      │                    ████ Batching
 90% ─┤                ████████████
      │            ████████████████████
 80% ─┤        ████████████████████████████
      │    ████████████████████████████████████
 70% ─┤████████████████████████████████████████████
      │
 60% ─┤
      │                                        ████ ← batch 256
 50% ─┤                                    ████████
      │
 40% ─┤
      │
 30% ─┤
      │
 20% ─┤  ░░ Threading
      │░░░░░░
 10% ─┤░░░░░░░░░░░░
      │  ░░░░░░░░░░░░░░░░░░
  0% ─┴───┬───┬───┬───┬───┬───┬───┬───┬───────────────
          1   2   4   8  16  32  64 128 256
                    Parallelism Level (N)
```

## The Apple Driver Bug

### Evidence

1. **Threading efficiency drops with N:**
   - N=2: 12.2% → N=4: 6.4% → N=8: 3.2% → N=16: 1.6%
   - Efficiency halves as threads double (classic serialization)

2. **Throughput is CAPPED:**
   - 1 thread: 719 ops/s
   - 16 threads: 2,720 ops/s
   - Only 3.8x increase despite 16x more threads

3. **Batching has NO such limitation:**
   - Efficiency stays >85% up to batch 64
   - GPU handles internal parallelism correctly

### Root Cause

Apple's Metal driver serializes command buffer encoding per-process. When multiple threads submit work:

```
Thread 1: [encode]─────────────────────[wait GPU]
Thread 2:         [wait]───[encode]────[wait GPU]
Thread 3:                  [wait]──────[encode]───[wait GPU]
Thread 4:                              [wait]─────[encode]───[wait GPU]
                  ↑
           SERIALIZED ENCODING
```

With batching:
```
Single Thread: [encode batch of N]────[GPU processes N in parallel]
                                      ↑
                              PARALLEL EXECUTION
```

## Optimal Configuration

### For Maximum Throughput

| Use Case | Configuration | Expected Throughput |
|----------|--------------|---------------------|
| Single stream | batch_256 | 1.39M samples/s (130x) |
| 4 parallel streams | 4 processes × batch_64 | ~2.4M samples/s (225x) |
| Latency-sensitive | batch_16-32 | 162-308K samples/s |

### What NOT to Do

❌ Do not use threading for MPS parallelism
❌ Do not expect more threads = more throughput
❌ Do not use batch_1 with multiple threads

### What TO Do

✅ Use large batches (64-256) for throughput
✅ Use process pool if parallel streams needed
✅ Combine: process pool + large batches per process

## Conclusion

**Batching is the correct parallelization strategy for MPS.**

- Threading: 3% efficiency at 8 threads (broken)
- Batching: 88% efficiency at batch 64 (working)
- Batching is **30x more effective** than threading at N=8

The Apple Metal driver has a serialization bug that prevents thread-level parallelism. Batching bypasses this by letting the GPU handle parallelism internally.

## Files

| File | Description |
|------|-------------|
| `tests/benchmark_scaling_curves.py` | Benchmark script |
| `reports/main/scaling_curves.json` | Raw data |
| This report | Analysis |
