# MPSAllocator Lock Order Analysis N=1039

**Date**: 2025-12-17 09:40 PST
**Purpose**: Analyze lock ordering constraints for scalability improvements

## Current Lock Ordering

The MPSAllocator has TWO different lock orderings depending on code path:

### Path 1: Allocation (pool_mutex → m_mutex)
```cpp
// malloc() → alloc_buffer() at line 566, 429-432
std::unique_lock<std::mutex> pool_lock(pool.pool_mutex);  // FIRST
...
{
    std::lock_guard<std::recursive_mutex> lock(m_mutex);   // SECOND (nested)
    m_allocated_buffers[buffer_block->buffer] = buffer_block;
}
```

### Path 2: Double-Check (m_mutex → release → pool_mutex → m_mutex)
```cpp
// getSharedBufferPtr(), recordStream(), setBufferShape(), etc.
{
    std::lock_guard lock(m_mutex);           // FIRST
    buffer_block = find_buffer(ptr);
    saved_use_count = buffer_block->use_count;
}                                            // RELEASE
std::lock_guard pool_lock(pool->pool_mutex); // SECOND
{
    std::lock_guard lock(m_mutex);           // THIRD (re-acquire)
    // verify buffer unchanged
}
```

## Why Double-Check Pattern Exists

The double-check pattern exists to **prevent deadlock**:
- If Path 2 held m_mutex while acquiring pool_mutex, and Path 1 holds pool_mutex while acquiring m_mutex:
  - Thread A (allocation): holds pool_mutex, waiting for m_mutex
  - Thread B (recordStream): holds m_mutex, waiting for pool_mutex
  - **DEADLOCK**

The current pattern avoids this by NEVER holding m_mutex when acquiring pool_mutex.

## Scalability Cost

The double-check pattern requires **2 m_mutex acquisitions per operation** for:
- getSharedBufferPtr() (used for shared memory access)
- recordStream() (used for cross-stream sync)
- recordEvents() (used for cross-stream sync)
- waitForEvents() (used for cross-stream sync)
- setBufferShape() (used for tensor metadata)
- getBufferShape() (used for tensor metadata)

With 8 threads all calling these functions, contention on m_mutex causes:
- Measured 8-thread efficiency: **29.3%** (target: ≥50%)
- TransformerEncoderLayer: **19.6%** (8-thread ops/s LOWER than 4-thread!)

## Safe Optimization Options

### Option 1: Shard m_mutex by Buffer Address (Recommended)
```cpp
static constexpr size_t NUM_SHARDS = 16;
std::array<std::recursive_mutex, NUM_SHARDS> m_mutex_shards;
ska::flat_hash_map<void*, BufferBlock*> m_allocated_buffers_shards[NUM_SHARDS];

size_t get_shard(const void* ptr) {
    return std::hash<const void*>{}(ptr) % NUM_SHARDS;
}

// Lookup becomes:
auto shard = get_shard(ptr);
std::lock_guard lock(m_mutex_shards[shard]);
auto it = m_allocated_buffers_shards[shard].find(ptr);
```

**Benefits**:
- Reduces contention by ~16x (buffers distribute across shards)
- Maintains existing lock ordering (safe)
- Double-check pattern still works per-shard

**Risk**: LOW - lock ordering unchanged, just finer granularity

### Option 2: Unify Lock Order (High Risk)
Change allocation path to use m_mutex → pool_mutex ordering:
```cpp
// In alloc_buffer():
{
    std::lock_guard lock(m_mutex);  // FIRST
    std::lock_guard pool_lock(pool_mutex);  // SECOND (nested)
    m_allocated_buffers[buffer_block->buffer] = buffer_block;
}
```

Then change double-check to hold m_mutex throughout:
```cpp
// In getSharedBufferPtr():
std::lock_guard lock(m_mutex);  // Hold throughout
buffer_block = find_buffer(ptr);
std::lock_guard pool_lock(pool_mutex);  // Nested under m_mutex
// No re-verification needed - we never released m_mutex
```

**Benefits**:
- Reduces from 2 to 1 m_mutex acquisition per operation
- Cleaner code

**Risk**: HIGH - requires auditing ALL code paths for lock order consistency

### Option 3: Lock-Free Buffer Lookup
Use atomic operations for buffer lookup with version stamps:
```cpp
struct AtomicBufferEntry {
    std::atomic<BufferBlock*> block;
    std::atomic<uint64_t> version;
};
```

**Benefits**: Eliminates m_mutex for read-only lookups

**Risk**: MEDIUM - complex correctness, needs careful memory ordering

## Recommendation

**Option 1 (Sharding)** is the safest path to improved scalability:
1. Split m_allocated_buffers into 16 shards
2. Each shard has its own mutex
3. Existing lock ordering preserved
4. Expected efficiency improvement: 29% → ~50%+

## Files to Modify

1. `MPSAllocator.mm` - Add shard array, modify all m_mutex usages
2. `MPSAllocator.tla` - Update TLA+ model to reflect sharding
3. Re-verify with TLC that safety properties still hold

## Next Steps

1. Implement Option 1 (m_mutex sharding)
2. Run full test suite to verify correctness
3. Re-run benchmarks to measure improvement
4. Update TLA+ ParallelLockHolding property - should now PASS
