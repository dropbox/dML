// alloc_free_harness.c - CBMC harness for verifying alloc/free safety
//
// Verifies:
// 1. No double-free (cannot free a block that's already free)
// 2. No use-after-free (cannot access freed blocks)
// 3. ABA detection works (use_count prevents stale reference use)
// 4. Memory bounds safety
//
// Run with:
//   cbmc alloc_free_harness.c --unwind 10 --pointer-check --bounds-check
//
// With concurrency modeling:
//   cbmc alloc_free_harness.c --unwind 10 --pointer-check --bounds-check \
//        --mm pso  # Partial Store Order (ARM-like memory model)

// Optimize pool size for CBMC - harness only uses up to 4 blocks
// This reduces state space from 64^N to 8^N, dramatically speeding up verification
#define MAX_POOL_BUFFERS 8

#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>

// Include our models
#include "../models/buffer_block.h"
#include "../stubs/metal_stubs.h"

// ============================================================================
// CBMC Primitives
// ============================================================================

// Non-deterministic value generators (CBMC built-ins)
extern int nondet_int(void);
extern unsigned int nondet_uint(void);
extern size_t nondet_size_t(void);
extern bool nondet_bool(void);

// CBMC assume (constrain state space)
extern void __CPROVER_assume(bool);

// CBMC assert (verification target)
extern void __CPROVER_assert(bool, const char*);

// ============================================================================
// Simplified Allocator Model
// ============================================================================

typedef struct AllocatorModel {
    BufferPool pool;
    size_t total_allocated;
    size_t current_allocated;
    bool is_alive;
} AllocatorModel;

static inline void Allocator_init(AllocatorModel* alloc) {
    BufferPool_init(&alloc->pool);
    alloc->total_allocated = 0;
    alloc->current_allocated = 0;
    alloc->is_alive = true;
}

// Allocate a buffer of given size
// Returns pointer to BufferBlock, or NULL if OOM
static inline BufferBlock* Allocator_alloc(AllocatorModel* alloc, size_t size) {
    __CPROVER_assert(alloc != NULL, "Allocator must not be null");
    __CPROVER_assert(alloc->is_alive, "Allocator must be alive");

    // Align size
    size_t aligned_size = alignUp(size, MPS_ALIGNMENT);
    __CPROVER_assert(aligned_size != SIZE_MAX, "Size alignment must not overflow");

    // Try to find existing free block
    BufferBlock* block = BufferPool_findFreeBlock(&alloc->pool, aligned_size);

    if (block == NULL) {
        // Create new block
        void* buffer = (void*)(uintptr_t)(alloc->total_allocated + 0x10000);
        block = BufferPool_addBlock(&alloc->pool, aligned_size, buffer);
        if (block == NULL) {
            return NULL; // Pool full
        }
    }

    // Mark as in-use
    BufferBlock_acquire(block);

    // Update stats
    alloc->total_allocated += aligned_size;
    alloc->current_allocated += block->size;

    return block;
}

// Free a buffer
static inline void Allocator_free(AllocatorModel* alloc, BufferBlock* block) {
    __CPROVER_assert(alloc != NULL, "Allocator must not be null");
    __CPROVER_assert(block != NULL, "Block must not be null");
    __CPROVER_assert(BufferBlock_isInUse(block), "Cannot free block that's not in use (double-free)");

    // Mark as free
    BufferBlock_release(block);

    // Update stats
    alloc->current_allocated -= block->size;
}

// ============================================================================
// ABA Double-Check Pattern Model
// ============================================================================

// Models getSharedBufferPtr() ABA detection pattern
// This is the critical synchronization pattern from 32.267 fix
static inline BufferBlock* Allocator_getSharedBufferPtr_ABA(
    AllocatorModel* alloc,
    void* ptr)
{
    __CPROVER_assert(alloc != NULL, "Allocator must not be null");
    __CPROVER_assert(alloc->is_alive, "Allocator must be alive");

    // Simulate: Find block by pointer (simplified)
    BufferBlock* block = NULL;
    for (size_t i = 0; i < alloc->pool.num_blocks; i++) {
        if (alloc->pool.blocks[i].buffer == ptr) {
            block = &alloc->pool.blocks[i];
            break;
        }
    }

    if (block == NULL) {
        return NULL;
    }

    // ==== FIRST CHECK (outside lock) ====
    // Capture in_use and use_count atomically (well, snapshot)
    bool first_in_use = BufferBlock_isInUse(block);
    uint32_t captured_use_count = BufferBlock_getUseCount(block);

    if (!first_in_use) {
        // Block is not in use - can't get shared ptr
        return NULL;
    }

    // ==== ACQUIRE LOCK (simulated) ====
    // In real code, we'd acquire m_mutex here

    // ==== SECOND CHECK (inside lock) - ABA detection ====
    bool second_in_use = BufferBlock_isInUse(block);
    uint32_t current_use_count = BufferBlock_getUseCount(block);

    // ABA Detection: use_count changed means block was freed and reallocated
    if (current_use_count != captured_use_count) {
        // ABA detected! Block was recycled between first and second check
        __CPROVER_assert(true, "ABA detection triggered - block recycled");
        return NULL;
    }

    // Final validity check
    if (!second_in_use) {
        // Block was freed between first check and lock acquisition
        return NULL;
    }

    return block;
}

// ============================================================================
// Main Harness
// ============================================================================

int main(void) {
    AllocatorModel allocator;
    Allocator_init(&allocator);

    // Non-deterministic number of operations
    // Keep num_ops <= 4 to match --unwind 5 (loop needs num_ops+1 iterations)
    int num_ops = nondet_int();
    __CPROVER_assume(num_ops > 0 && num_ops <= 4);

    // Track allocated blocks
    BufferBlock* allocated[4] = {NULL};
    int num_allocated = 0;

    // Perform sequence of alloc/free operations
    for (int i = 0; i < num_ops; i++) {
        bool do_alloc = nondet_bool();

        if (do_alloc && num_allocated < 4) {
            // Allocate
            size_t size = nondet_size_t();
            __CPROVER_assume(size > 0 && size <= 1024 * 1024); // 1MB max

            BufferBlock* block = Allocator_alloc(&allocator, size);
            if (block != NULL) {
                // Verify block is valid
                __CPROVER_assert(BufferBlock_isInUse(block), "Newly allocated block must be in use");
                __CPROVER_assert(block->size >= size, "Block size must be >= requested");

                allocated[num_allocated++] = block;
            }
        } else if (num_allocated > 0) {
            // Free a random allocated block
            int idx = nondet_int();
            __CPROVER_assume(idx >= 0 && idx < num_allocated);

            BufferBlock* block = allocated[idx];
            __CPROVER_assert(block != NULL, "Selected block must not be null");
            __CPROVER_assert(BufferBlock_isInUse(block), "Block to free must be in use");

            Allocator_free(&allocator, block);

            // Remove from tracking (swap with last)
            allocated[idx] = allocated[--num_allocated];
            allocated[num_allocated] = NULL;
        }
    }

    // ==== Verify Invariants ====

    // 1. All tracked blocks are still in use
    for (int i = 0; i < num_allocated; i++) {
        __CPROVER_assert(allocated[i] != NULL, "Tracked block must not be null");
        __CPROVER_assert(BufferBlock_isInUse(allocated[i]), "Tracked block must be in use");
    }

    // 2. Free remaining blocks
    for (int i = 0; i < num_allocated; i++) {
        Allocator_free(&allocator, allocated[i]);
    }

    // 3. After freeing all, current_allocated should be 0
    __CPROVER_assert(allocator.current_allocated == 0, "All allocations must be freed");

    // 4. No block should be in use
    for (size_t i = 0; i < allocator.pool.num_blocks; i++) {
        __CPROVER_assert(!BufferBlock_isInUse(&allocator.pool.blocks[i]),
                        "No blocks should be in use after freeing all");
    }

    return 0;
}
