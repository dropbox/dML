// buffer_block.h - Simplified BufferBlock model for CBMC verification
//
// This models the key fields and invariants of PyTorch MPS BufferBlock
// for bounded model checking. Focus is on:
// - Atomic in_use flag (prevents double-free)
// - Atomic use_count (ABA detection)
// - Memory safety of the double-check locking pattern

#ifndef MPS_VERIFY_BUFFER_BLOCK_H
#define MPS_VERIFY_BUFFER_BLOCK_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>

#ifdef __cplusplus
#include <atomic>
#define ATOMIC_BOOL std::atomic<bool>
#define ATOMIC_UINT32 std::atomic<uint32_t>
#define ATOMIC_UINT64 std::atomic<uint64_t>
#define ATOMIC_LOAD(x) (x).load()
#define ATOMIC_STORE(x, v) (x).store(v)
#define ATOMIC_FETCH_ADD(x, v) (x).fetch_add(v)
#define ATOMIC_COMPARE_EXCHANGE(x, expected, desired) \
    (x).compare_exchange_strong(expected, desired)
#else
// For pure C CBMC verification
#define ATOMIC_BOOL _Bool
#define ATOMIC_UINT32 uint32_t
#define ATOMIC_UINT64 uint64_t
#define ATOMIC_LOAD(x) (x)
#define ATOMIC_STORE(x, v) ((x) = (v))
#define ATOMIC_FETCH_ADD(x, v) ((x) += (v), (x) - (v))
#define ATOMIC_COMPARE_EXCHANGE(x, expected, desired) \
    ((x) == (expected) ? ((x) = (desired), true) : (*(expected) = (x), false))
#endif

// ============================================================================
// BufferBlock Model
// ============================================================================

// Simplified BufferBlock focusing on synchronization-critical fields
typedef struct BufferBlock {
    // Pointer to underlying buffer (modeled as opaque for CBMC)
    void* buffer;

    // Size after alignment
    size_t size;

    // Requested size before alignment
    size_t requested_size;

    // Atomic: Is this block currently allocated?
    // Read without lock in getSharedBufferPtr(), written under pool_mutex
    ATOMIC_BOOL in_use;

    // Atomic: ABA generation counter
    // Incremented on each allocation, used to detect stale references
    ATOMIC_UINT32 use_count;

    // Unique block ID (for debugging/tracking)
    uint64_t buf_id;

    // GC counter for eviction heuristics
    uint32_t gc_count;

    // Stream that allocated this buffer (-1 = none)
    int64_t alloc_stream_id;

} BufferBlock;

// Global buffer ID counter (simplified)
static ATOMIC_UINT64 g_buffer_counter = 0;

// ============================================================================
// BufferBlock Operations
// ============================================================================

// Initialize a new BufferBlock
static inline void BufferBlock_init(BufferBlock* block, size_t size, void* buffer) {
    assert(block != NULL);
    block->buffer = buffer;
    block->size = size;
    block->requested_size = size;
    ATOMIC_STORE(block->in_use, false);
    ATOMIC_STORE(block->use_count, 0);
    block->buf_id = ATOMIC_FETCH_ADD(g_buffer_counter, 1) + 1;
    block->gc_count = 0;
    block->alloc_stream_id = -1;
}

// Mark block as in-use (during allocation)
// PRE: Caller holds pool_mutex
// POST: in_use=true, use_count incremented
static inline void BufferBlock_acquire(BufferBlock* block) {
    assert(block != NULL);
    // Must not already be in use
    assert(!ATOMIC_LOAD(block->in_use));

    ATOMIC_STORE(block->in_use, true);
    ATOMIC_FETCH_ADD(block->use_count, 1);
}

// Mark block as free (during deallocation)
// PRE: Caller holds pool_mutex
// POST: in_use=false
static inline void BufferBlock_release(BufferBlock* block) {
    assert(block != NULL);
    // Must be in use
    assert(ATOMIC_LOAD(block->in_use));

    ATOMIC_STORE(block->in_use, false);
}

// Check if block is in use (can be called without lock)
// NOTE: This is a point-in-time check - value may change immediately after
static inline bool BufferBlock_isInUse(const BufferBlock* block) {
    assert(block != NULL);
    return ATOMIC_LOAD(block->in_use);
}

// Get current use_count for ABA detection
// Called once before acquiring lock, once after
static inline uint32_t BufferBlock_getUseCount(const BufferBlock* block) {
    assert(block != NULL);
    return ATOMIC_LOAD(block->use_count);
}

// ============================================================================
// Size Alignment (from actual code)
// ============================================================================

// Align size up to given alignment (must be power of 2)
static inline size_t alignUp(size_t size, size_t alignment) {
    // Check alignment is power of 2
    assert((alignment & (alignment - 1)) == 0);

    // Overflow check (from 32.256 fix)
    if (size > SIZE_MAX - (alignment - 1)) {
        // Would overflow - return SIZE_MAX as sentinel
        return SIZE_MAX;
    }

    return (size + alignment - 1) & ~(alignment - 1);
}

// Typical MPS alignment (512 bytes for SSE, can be larger for GPU)
#define MPS_ALIGNMENT 512

// ============================================================================
// Pool Model (simplified)
// ============================================================================

// MAX_POOL_BUFFERS can be overridden with -DMAX_POOL_BUFFERS=N
// Default 64 for full verification, use 8 for faster CBMC runs
#ifndef MAX_POOL_BUFFERS
#define MAX_POOL_BUFFERS 64
#endif

typedef struct BufferPool {
    BufferBlock blocks[MAX_POOL_BUFFERS];
    size_t num_blocks;
    bool is_alive;
} BufferPool;

static inline void BufferPool_init(BufferPool* pool) {
    assert(pool != NULL);
    pool->num_blocks = 0;
    pool->is_alive = true;
}

// Add a new block to the pool
static inline BufferBlock* BufferPool_addBlock(BufferPool* pool, size_t size, void* buffer) {
    assert(pool != NULL);
    assert(pool->is_alive);

    if (pool->num_blocks >= MAX_POOL_BUFFERS) {
        return NULL; // Pool full
    }

    BufferBlock* block = &pool->blocks[pool->num_blocks++];
    BufferBlock_init(block, size, buffer);
    return block;
}

// Find a free block of suitable size
// Returns NULL if none available
// NOTE: Caller must hold pool_mutex
static inline BufferBlock* BufferPool_findFreeBlock(BufferPool* pool, size_t size) {
    assert(pool != NULL);

    for (size_t i = 0; i < pool->num_blocks; i++) {
        BufferBlock* block = &pool->blocks[i];
        if (!ATOMIC_LOAD(block->in_use) && block->size >= size) {
            return block;
        }
    }
    return NULL;
}

#endif // MPS_VERIFY_BUFFER_BLOCK_H
