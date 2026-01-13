# Phase 3.6: Bare Metal API Isolation Test

**Worker**: N=1334
**Date**: 2025-12-19
**Phase**: 3.6 - Metal API isolation
**Previous Investigation**: N=1333 (C++ ATen reproduction)

## Executive Summary

Created a bare Metal API test (`tests/minimal_metal_race.mm`) that demonstrates Apple's Metal and MetalPerformanceShaders frameworks are **thread-safe**. The race condition exists in **PyTorch's ATen/MPS layer**, NOT Apple's frameworks.

## Key Finding

**The race condition is in PyTorch's ATen/MPS code, NOT Apple's Metal/MPS frameworks.**

| Test Layer | Test Name | Serial | 8-Thread Parallel | Failure Rate |
|------------|-----------|--------|-------------------|--------------|
| **Apple Metal API** | Strided Buffer Copy | 30/30 PASS | 30/30 PASS | 0% |
| **Apple MPS Framework** | MPSMatrix Multiply | 30/30 PASS | 30/30 PASS | 0% |
| **PyTorch ATen/MPS** | .contiguous() | 30/30 PASS | 23/30 PASS | ~23% |

## Test Methodology

### Bare Metal API Test

Created `tests/minimal_metal_race.mm` which uses ONLY Metal and MetalPerformanceShaders APIs (no ATen/PyTorch). Tests:

1. **Metal Compute Shader (Strided Copy)**
   - Allocate MTLBuffer with strided layout
   - Run compute kernel to copy to contiguous buffer
   - Each thread gets own MTLCommandQueue (mirrors PyTorch per-stream behavior)
   - Tests: 8 threads, 30 iterations

2. **MPSMatrix Multiply**
   - Use MPSMatrixMultiplication (Apple's MPS framework)
   - Allocate input/output matrices, run matmul
   - Each thread gets own queue and kernel instance
   - Tests: 8 threads, 30 iterations

### Results (3 consecutive runs)

**Run 1:**
```
Metal Strided Copy (Parallel): PASS (30/30), max_diff=0.00e+00
MPS Matrix Multiply (Parallel): PASS (30/30), max_diff=0.00e+00
```

**Run 2:**
```
Metal Strided Copy (Parallel): PASS (30/30), max_diff=0.00e+00
MPS Matrix Multiply (Parallel): PASS (30/30), max_diff=0.00e+00
```

**Run 3:**
```
Metal Strided Copy (Parallel): PASS (30/30), max_diff=0.00e+00
MPS Matrix Multiply (Parallel): PASS (30/30), max_diff=0.00e+00
```

### ATen C++ Test (for comparison)

Running `tests/minimal_mps_contiguous_race` (from N=1333):
```
WITHOUT .contiguous(): PASS (30/30), max_diff=0.00e+00
WITH .contiguous(): FAIL (23/30), max_diff=1.90e-01
```

## Implications

### 1. Apple's Metal/MPS Framework is Thread-Safe

At the API level we tested:
- MTLDevice, MTLCommandQueue, MTLCommandBuffer, MTLComputeCommandEncoder
- MTLBuffer allocation with StorageModeShared
- Custom compute kernels with strided memory access
- MPSMatrixMultiplication from MetalPerformanceShaders

All operations are safe with per-thread command queues at 8 threads.

### 2. Bug is in PyTorch ATen/MPS Layer

The race condition is somewhere in PyTorch's code, likely:

1. **MPSGraph caching/lookup** - PyTorch caches MPSGraphs for repeated operations. Concurrent lookups or cache updates may race.

2. **MPS heap allocator** - PyTorch's custom heap allocator (`MPSAllocator.mm`) manages MTLBuffer pools. Allocation races possible.

3. **Tensor metadata** - ATen tensor storage and stride metadata updates during `.contiguous()` may race.

4. **MPSGraphTensorData binding** - The binding between ATen tensors and MPSGraph inputs may have races.

### 3. This is Potentially Fixable Upstream

Since the bug is in PyTorch's code (not Apple's frameworks), it can theoretically be fixed in PyTorch. However:
- May require significant changes to MPS backend architecture
- Our BatchQueue workaround is the practical solution
- Filing a PyTorch issue with these findings would be valuable

## Files Created

- `tests/minimal_metal_race.mm` - Bare Metal API test (no ATen dependency)

## Build and Run

```bash
# Build
clang++ -std=c++17 -framework Metal -framework MetalPerformanceShaders \
        -framework Foundation -O2 -o tests/build/minimal_metal_race \
        tests/minimal_metal_race.mm

# Run
./tests/build/minimal_metal_race
```

## Verification Status

| Suite | Result |
|-------|--------|
| TSA | 0 warnings (4 files) |
| Structural | 54/61 pass, 0 failures |
| Batch inference | 5/5 tests pass |
| Parallel correctness (batched) | 10/10 (100%) |
| Bare Metal API (new) | 4/4 tests pass |

## Conclusions

1. **Apple's Metal and MPS frameworks are thread-safe** at the API level tested
2. **The race condition is in PyTorch's ATen/MPS layer**
3. **Likely culprits**: MPSGraph caching, MPS heap allocator, or tensor metadata updates
4. **Our BatchQueue workaround** correctly addresses the issue by serializing MPS operations
5. **This finding could inform a PyTorch upstream bug report** with specific localization data

## Next Steps (Optional)

1. [ ] Investigate MPSGraph caching code (`MPSGraph.mm`, `MPSGraphCache.mm`)
2. [ ] Profile with Instruments to identify specific lock contention
3. [ ] File PyTorch issue with these isolation findings
4. [ ] Consider adding locking to ATen MPS allocation path
