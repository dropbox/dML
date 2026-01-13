# MANAGER DIRECTIVE N=1039

**Date**: 2025-12-17 09:55 PST
**Author**: Manager AI

## Work Completed

1. **m_mutex Sharding Implementation** (pytorch-mps-fork commit 9a7876e6)
   - Sharded `m_allocated_buffers` map across 8 shards
   - Each shard has its own mutex (`AllocatedBufferShard`)
   - Updated all buffer lookup functions:
     - `alloc_buffer()`, `get_allocated_buffer_block()`, `release_buffer()`
     - `isSharedBuffer()`, `getSharedBufferPtr()`, `recordStream()`
     - `getBufferId()`, `getUnalignedBufferSize()`
     - `setBufferShape()`, `getBufferShape()`
     - `recordEvents()`, `waitForEvents()`

2. **Fixed MPS_CAPABILITY Build Errors**
   - Removed incorrect `MPS_CAPABILITY` annotations from mutex members
   - The `capability` attribute only applies to type definitions, not variable declarations
   - Fixed in: MPSAllocator.h, MPSStream.h, MPSEvent.h

3. **Test Results**
   - All 24 Python tests: PASS
   - Build: SUCCESS (168/168 ninja targets)

4. **Benchmark Results**
   | Model | 8-Thread Efficiency | Previous | Change |
   |-------|---------------------|----------|--------|
   | nn.Linear | 30.6% | 29.3% | +1.3% |
   | MLP | 30.6% | ~29% | +1.6% |
   | TransformerEncoderLayer | 21.7% | 19.6% | +2.1% |

## Analysis

The sharding implementation is correct and builds cleanly. However, the efficiency improvement is modest (~1-2%) because:

1. **GPU Saturation**: At 8 threads, the M4 Max GPU may already be saturated with work, making CPU-side lock contention less relevant.

2. **Other Bottlenecks**: The `pool_mutex` per BufferPool may still cause contention for operations on the same pool.

3. **Lock Ordering Constraint**: The double-check pattern (shard_mutex -> release -> pool_mutex -> shard_mutex) still requires 2 shard_mutex acquisitions per operation.

## WORKER DIRECTIVE

### Priority 1: Generate Updated Cumulative Patch
The current `cumulative-v2.9.1-to-mps-stream-pool.patch` is stale. Generate a new cumulative patch that includes the sharding fix:

```bash
cd pytorch-mps-fork
git diff v2.9.1 HEAD > ../patches/cumulative-v2.9.1-to-mps-stream-pool-v2.patch
```

### Priority 2: Investigate Alternative Bottlenecks
The 8-thread efficiency of 30% suggests the GPU is the bottleneck, not CPU locks. Investigate:

1. **GPU Queue Depth**: Are 8 threads overwhelming the GPU command queue?
2. **Metal Profiler**: Use Instruments to check if GPU is at 100% utilization
3. **Workload Size**: Larger workloads may benefit more from parallelism

### Priority 3: Document Scalability Ceiling
Write a report documenting:
- Maximum practical parallelism for MPS on M4 Max
- When CPU contention vs GPU saturation is the bottleneck
- Recommendations for users on optimal thread counts

### Files Modified (pytorch-mps-fork)
- `aten/src/ATen/mps/MPSAllocator.h` - Sharding infrastructure
- `aten/src/ATen/mps/MPSAllocator.mm` - Sharded mutex usage
- `aten/src/ATen/mps/MPSStream.h` - Removed bad annotations
- `aten/src/ATen/mps/MPSEvent.h` - Removed bad annotations

### Patch
- `patches/1040-allocator-sharding.patch` - Contains all changes

## Success Criteria
- Updated cumulative patch generated
- Scalability ceiling documented
- Tests continue to pass
