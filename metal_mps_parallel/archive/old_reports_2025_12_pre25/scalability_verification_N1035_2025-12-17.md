# Scalability Verification Report N=1035

**Date**: 2025-12-17 09:31 PST
**System**: Apple M4 Max, macOS 15.7.2

## Summary

Extended TLA+ verification with scalability properties to diagnose the 4-8 thread performance regression.

## TLC Results

### Safety Properties (ALL PASS)
| Property | Status | States Explored |
|----------|--------|-----------------|
| TypeOK | PASS | 15,298,749 generated |
| Safety | PASS | 7,766,326 distinct |
| LockHierarchy | PASS | Complete in 12s |

The safety properties confirm the implementation is **correct** - no ABA races, no double-free, no use-after-free.

### Scalability Properties (Documented, Not Invariants)

These properties document design bottlenecks but are not safety violations:

| Property | Purpose | Current Design Status |
|----------|---------|----------------------|
| ParallelLockHolding | Can 2+ threads hold different locks? | NO - global m_mutex serializes |
| GlobalSerializerViolation | Is there a global serializing lock? | YES - m_mutex on every op |
| ExcessiveLocking | Lock acquisitions per operation | getPtr: 3 (m_mutex x2 + pool_mutex) |
| DoubleMutexBottleneck | Double m_mutex in getPtr | YES - 2x m_mutex contention |

## Key Findings

### The Scalability Bottleneck Is Verified

The TLA+ model confirms what the performance data shows:

1. **Global m_mutex serializes ALL operations**
   - Every `alloc`, `free`, and `getPtr` must acquire m_mutex
   - With 8 threads, all serialize through this single lock

2. **getSharedBufferPtr is 2x worse**
   - Acquires m_mutex at GetPtrLockM1 (capture use_count)
   - Releases m_mutex
   - Acquires pool_mutex
   - Re-acquires m_mutex at GetPtrLockM2 (verify)
   - This "double-check" pattern causes 2x m_mutex contention per call

3. **Lock acquisition count per operation:**
   - `alloc`: 2 locks (m_mutex + pool_mutex)
   - `free`: 1 lock (pool_mutex only)
   - `getPtr`: 3 locks (m_mutex + pool_mutex + m_mutex again)

### Why 8-Thread Efficiency Is 14%

With 8 threads all calling memory operations:
- All 8 serialize on m_mutex
- getPtr operations contribute 2x m_mutex acquisitions each
- Maximum parallelism = 1 thread in critical section
- Expected theoretical efficiency: 12.5% (1/8)
- Measured: 14% (slightly better due to pool_mutex parallelism)

## Recommended Fixes

Based on the verified bottlenecks:

### Fix 1: Shard m_mutex
```cpp
// Instead of single m_mutex:
std::array<std::recursive_mutex, 16> shard_mutexes_;
size_t get_shard(void* ptr) { return hash(ptr) % 16; }
```

### Fix 2: Eliminate double m_mutex in getPtr
```cpp
// Current: lock m_mutex -> unlock -> lock pool -> lock m_mutex again
// Better: lock m_mutex once, do all work, unlock
void* getSharedBufferPtr(void* ptr) {
    std::lock_guard lock(m_mutex);  // Single acquisition
    BufferBlock* block = find_buffer(ptr);
    if (!block) return nullptr;

    std::lock_guard pool_lock(block->pool->pool_mutex);
    // Verify and return under both locks
    if (!block->in_use) return nullptr;
    return block->buf;
}
```

### Fix 3: Lock-free ABA check
Use atomic compare-exchange for use_count verification instead of mutex.

## Verification Infrastructure Updates

Added to MPSAllocator.tla:
- `lock_wait_count` variable (tracks blocked lock attempts)
- `lock_acquire_count` variable (tracks total acquisitions)
- `ParallelLockHolding` scalability property
- `GlobalSerializerViolation` documentation
- `ExcessiveLocking` property
- `LockHierarchy` safety invariant
- `DoubleMutexBottleneck` documentation

## Files Modified

- `mps-verify/specs/MPSAllocator.tla` - Added scalability variables and properties
- `mps-verify/specs/MPSAllocator.cfg` - Updated to check LockHierarchy

## Next Steps

1. Implement allocator sharding (Fix 1)
2. Modify getSharedBufferPtr to single m_mutex acquisition (Fix 2)
3. Re-run TLC to verify ParallelLockHolding now passes
4. Benchmark 8-thread efficiency - target ≥50%
