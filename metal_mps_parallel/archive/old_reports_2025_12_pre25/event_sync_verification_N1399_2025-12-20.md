# Event Sync vs Device Sync Verification (N=1399)

**Date:** 2025-12-20
**Hardware:** Apple M4 Max (40 GPU cores), macOS 15.7.3
**Worker:** N=1399

> **UPDATE (2025-12-20):** The 1-thread event sync throughput in the "Small Matrix (512x512 FP16)" table below (2,911 ops/s; 0.45x vs device) was later re-run and found to be ~1.0x vs device. This makes the derived "78.3% efficiency" figure obsolete. See `reports/main/mps_sync_comparison_update_2025-12-20-15-35.md`.

## Executive Summary

Verified the claim from `metal_bottleneck_proof_2025-12-20.md` that the bottleneck is NOT the Metal driver but `torch.mps.synchronize()` device-wide semantics.

**Key Result:** Per-stream event sync achieves **2.79x throughput** vs device sync baseline at 8 threads for small matrices.

## Test Results

### Small Matrix (512x512 FP16)

| Threads | Device Sync (ops/s) | Event Sync (ops/s) | Event/Device Ratio |
|---------|--------------------|--------------------|-------------------|
| 1       | 6,525              | 2,911              | 0.45x             |
| 2       | 5,687              | 13,005             | 2.29x             |
| 4       | 6,563              | 17,830             | 2.72x             |
| 8       | 6,773              | 18,222             | 2.69x             |

**Analysis:**
- Device sync: NO scaling (1.04x at 8T vs 1T)
- Event sync: 6.26x speedup (relative to event sync 1T)
- Event sync efficiency: 6.26/8 = **78.3%**
- Event sync vs device sync baseline: **2.79x throughput at 8T**

### Large Matrix (2048x2048 FP16)

| Threads | Device Sync (ops/s) | Event Sync (ops/s) | Scaling |
|---------|--------------------|--------------------|---------|
| 1       | 719                | 717                | 1.00x   |
| 8       | 719                | 849                | 1.18x   |

**Analysis:**
- GPU saturated - large matrix dominates GPU
- Neither sync mode helps significantly
- Event sync provides marginal 18% improvement

## Critical Finding

The "Metal driver bottleneck ceiling" claim from previous iterations is **partially incorrect**.

| Bottleneck | Affected By |
|------------|-------------|
| Device-wide sync | `torch.mps.synchronize()` synchronizes ALL streams |
| GPU saturation | Large kernels (2048x2048) saturate GPU |
| Metal driver | NOT the primary bottleneck for most workloads |

## Implications

1. **For non-GPU-saturated workloads:** Use per-stream event synchronization
2. **For GPU-saturated workloads:** Batching is the only solution
3. **DYLD Interposition (TASK 2):** May be unnecessary - the issue is API usage, not driver

## Recommendation

Update WORKER_DIRECTIVE.md priorities:
1. **HIGH:** Document correct sync mode usage
2. **MEDIUM:** Update benchmarks to use event sync where appropriate
3. **LOW:** DYLD interposition - may be unnecessary

## Complete Comparison: All Methods (NN Inference Workload)

Tested with same workload: `nn.Linear(512→1024→512)` with batch=32

| Method                    | Ops/s    | Scaling | Efficiency |
|---------------------------|----------|---------|------------|
| Single-thread baseline    | 3,005    | 1.00x   | 100%       |
| 8 threads + device sync   | 4,607    | 1.53x   | 19.2%      |
| 8 threads + event sync    | 7,065    | 2.35x   | 29.4%      |
| **Process pool 8 workers**| **23,035**| **7.67x**| **95.8%** |

**KEY FINDINGS:**
1. Process pool achieves near-linear scaling (95.8% efficiency)
2. Event sync is 1.53x better than device sync
3. Process pool is 3.26x better than threading+event sync
4. **BEST METHOD: Process pool for maximum throughput**

## Updated Recommendations

Based on comprehensive testing:

| Use Case | Best Method | Why |
|----------|-------------|-----|
| Maximum throughput | Process pool | 95.8% efficiency, near-linear scaling |
| Threading required | Event sync | 1.53x better than device sync |
| GPU-saturated ops | Batching | Neither sync mode helps at saturation |
| Single-stream | Device sync | Lower overhead for single thread |

## Repro Commands

```bash
# Small matrix - shows scaling improvement
python3 tests/profile_metal_trace.py --op matmul --size 512 --dtype float16 --threads 8 --iters 2000 --sync-mode device
python3 tests/profile_metal_trace.py --op matmul --size 512 --dtype float16 --threads 8 --iters 2000 --sync-mode event

# Large matrix - shows GPU saturation
python3 tests/profile_metal_trace.py --op matmul --size 2048 --dtype float16 --threads 8 --iters 500 --sync-mode device
python3 tests/profile_metal_trace.py --op matmul --size 2048 --dtype float16 --threads 8 --iters 500 --sync-mode event
```
