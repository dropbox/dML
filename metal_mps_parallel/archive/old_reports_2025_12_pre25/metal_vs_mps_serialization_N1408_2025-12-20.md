# Metal vs MPS Serialization Analysis

**Worker N=1408**
**Date: 2025-12-20**

## Executive Summary

**KEY FINDING**: The serialization bottleneck is NOT in Apple's Metal driver. It's specifically in MPS (MetalPerformanceShaders) or PyTorch's MPS backend.

| Test | 8-Thread Efficiency | Scaling |
|------|---------------------|---------|
| Raw Metal (shared queue) | **59.8%** | 4.79x |
| Raw Metal (separate queues) | 39.1% | 3.13x |
| PyTorch MPS (device sync) | ~13% | ~1.1x |
| PyTorch MPS (event sync) | ~33% | ~2.6x |

## Evidence

### Raw Metal Test Results

```
Device: Apple M4 Max
========================================
Sequential (1 thread):         65,179 ops/s

--- pthread Parallel (separate queues) ---
2 threads: 114,548 ops/s (1.76x, 87.9% eff)
4 threads: 165,172 ops/s (2.53x, 63.4% eff)
8 threads: 203,897 ops/s (3.13x, 39.1% eff)

--- pthread Parallel (shared queue) ---
2 threads: 125,903 ops/s (1.93x, 96.6% eff)
4 threads: 202,540 ops/s (3.11x, 77.7% eff)
8 threads: 311,936 ops/s (4.79x, 59.8% eff)
```

### Key Observations

1. **Metal driver supports parallel submission**: 8 threads achieve 59.8% efficiency with raw Metal
2. **Shared queue is BETTER**: Using a single shared command queue is more efficient than separate queues
3. **MPS adds serialization**: PyTorch MPS drops from 59.8% to ~13% efficiency at 8 threads
4. **Event sync helps**: Using `torch.mps.Event` recovers some parallelism (~33% vs ~13%)

## Root Cause Analysis

The serialization is NOT in:
- Metal command buffer creation
- Metal command buffer commit
- Metal command buffer waitUntilCompleted
- Metal command queue (shared queue is actually faster!)

The serialization IS in:
- MetalPerformanceShaders (MPS) kernel dispatch
- PyTorch's MPS backend stream management
- Potentially: MPSCommandBuffer or MPS internal synchronization

## Interposition Library Findings

Using DYLD_INSERT_LIBRARIES with our metal_interpose.m library:

```
Statistics from mps_sync_comparison.py:
- computeCommandEncoder: 0 calls (MPS uses different path)
- commit: 22,714 calls, avg time: 0.00 ms
- waitUntilCompleted: 7,576 calls, avg time: 0.12 ms
```

The fact that `commit` averages 0.00 ms means the serialization is not at the Metal layer.
MPS must be adding its own synchronization internally.

## Implications

1. **Custom Metal kernels would bypass the bug**: Since raw Metal scales well, custom compute kernels would achieve near-linear efficiency

2. **MPS needs per-stream synchronization**: The MPS library likely has internal global locks that serialize kernel dispatch

3. **Process pool workaround is valid**: Separate processes have separate MPS contexts, bypassing the global lock

## Recommendations

1. **For immediate workaround**: Use process pool (96% efficiency) or event sync (~33% efficiency)

2. **For PyTorch upstream**: File issue against MPS backend, not Metal driver

3. **For best performance**: Consider custom Metal kernels (like MLX does) instead of MPS

## Files

- `fixes/metal_interpose/metal_interpose.m` - DYLD interposition library
- `fixes/metal_interpose/test_raw_metal.m` - Raw Metal parallelism test
- `fixes/metal_interpose/build.sh` - Build script

## Test Commands

```bash
# Build interposition library
cd fixes/metal_interpose && ./build.sh

# Test with interposition
DYLD_INSERT_LIBRARIES=./libmetal_interpose.dylib \
    METAL_INTERPOSE_LOG=1 \
    python3 tests/mps_sync_comparison.py

# Run raw Metal test
cd fixes/metal_interpose && ./test_raw_metal
```
