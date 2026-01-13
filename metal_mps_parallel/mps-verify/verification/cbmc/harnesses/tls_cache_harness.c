// tls_cache_harness.c - CBMC harness for verifying TLS block cache
//
// The TLS (Thread-Local Storage) block cache provides per-thread caching
// of recently freed BufferBlocks to avoid pool mutex contention.
//
// Properties verified:
// 1. No double-caching (block can only be in cache once)
// 2. Cache bounds respected (max blocks, max size)
// 3. Shutdown safety (flush must work during allocator shutdown)
// 4. Invariant: cached blocks are not in_use
//
// Run with:
//   cbmc tls_cache_harness.c -I ../models -I ../stubs --unwind 5

#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>

#include "../models/buffer_block.h"
#include "../stubs/metal_stubs.h"

// CBMC Primitives
extern int nondet_int(void);
extern unsigned int nondet_uint(void);
extern size_t nondet_size_t(void);
extern bool nondet_bool(void);
extern void __CPROVER_assume(bool);
extern void __CPROVER_assert(bool, const char*);

// ============================================================================
// TLS Cache Configuration (reduced for CBMC verification with --unwind 5)
// Real values: MAX_BLOCKS=16, MAX_SIZE=32MB, MAX_BLOCK=4MB
// ============================================================================

#define TLS_CACHE_MAX_BLOCKS 4  // Reduced for CBMC (original: 16)
#define TLS_CACHE_MAX_SIZE (32 * 1024 * 1024)  // 32MB max total
#define TLS_CACHE_MAX_BLOCK_SIZE (4 * 1024 * 1024)  // 4MB max per block

// ============================================================================
// TLS Cache Model
// ============================================================================

typedef struct TLSBlockCache {
    BufferBlock* blocks[TLS_CACHE_MAX_BLOCKS];
    size_t num_blocks;
    size_t total_size;
} TLSBlockCache;

// Global allocator alive flag (models s_allocator_alive)
static bool g_allocator_alive = true;

// Initialize cache
static void TLSCache_init(TLSBlockCache* cache) {
    __CPROVER_assert(cache != NULL, "Cache must not be null");
    cache->num_blocks = 0;
    cache->total_size = 0;
    for (size_t i = 0; i < TLS_CACHE_MAX_BLOCKS; i++) {
        cache->blocks[i] = NULL;
    }
}

// Try to get a cached block that fits the requested size
// Returns NULL if no suitable block found
static BufferBlock* TLSCache_tryGet(TLSBlockCache* cache, size_t size) {
    __CPROVER_assert(cache != NULL, "Cache must not be null");

    // Fast-fail on shutdown
    if (!g_allocator_alive) {
        return NULL;
    }

    for (size_t i = 0; i < cache->num_blocks; i++) {
        BufferBlock* block = cache->blocks[i];

        // Match: size within 2x of request (actual heuristic)
        if (block != NULL &&
            block->size >= size &&
            block->size <= size * 2) {

            // Remove from cache
            cache->total_size -= block->size;

            // Compact: move last element to this position
            cache->num_blocks--;
            if (i < cache->num_blocks) {
                cache->blocks[i] = cache->blocks[cache->num_blocks];
            }
            cache->blocks[cache->num_blocks] = NULL;

            // Mark as in-use before returning
            BufferBlock_acquire(block);

            return block;
        }
    }

    return NULL;
}

// Try to cache a block for later reuse
// Returns true if cached, false if cache limits would be exceeded
// PRE: block must be marked as NOT in_use
static bool TLSCache_tryPut(TLSBlockCache* cache, BufferBlock* block) {
    __CPROVER_assert(cache != NULL, "Cache must not be null");
    __CPROVER_assert(block != NULL, "Block must not be null");

    // Block must not be in use when cached
    __CPROVER_assert(!BufferBlock_isInUse(block),
        "Block must be released before caching");

    // Reject blocks too large for TLS cache
    if (block->size > TLS_CACHE_MAX_BLOCK_SIZE) {
        return false;
    }

    // Reject if cache is full (by count)
    if (cache->num_blocks >= TLS_CACHE_MAX_BLOCKS) {
        return false;
    }

    // Reject if would exceed total size limit
    if (cache->total_size + block->size > TLS_CACHE_MAX_SIZE) {
        return false;
    }

    // Check for double-caching (block already in cache)
    for (size_t i = 0; i < cache->num_blocks; i++) {
        __CPROVER_assert(cache->blocks[i] != block,
            "Double-caching detected: block already in cache");
    }

    // Add to cache
    cache->blocks[cache->num_blocks++] = block;
    cache->total_size += block->size;

    return true;
}

// Flush all cached blocks (e.g., on thread exit or shutdown)
// Returns blocks to pool (modeled as just clearing the cache)
static void TLSCache_flush(TLSBlockCache* cache, BufferPool* pool) {
    __CPROVER_assert(cache != NULL, "Cache must not be null");

    // During shutdown, we still need to flush but be careful
    // not to access destroyed pool

    for (size_t i = 0; i < cache->num_blocks; i++) {
        BufferBlock* block = cache->blocks[i];
        if (block != NULL) {
            // In real code: return block to pool
            // Here: just verify block state is consistent
            __CPROVER_assert(!BufferBlock_isInUse(block),
                "Cached block should not be in use");

            cache->blocks[i] = NULL;
        }
    }

    cache->num_blocks = 0;
    cache->total_size = 0;
}

// ============================================================================
// Main Harness
// ============================================================================

int main(void) {
    // Create a pool with some blocks
    BufferPool pool;
    BufferPool_init(&pool);

    // Add some initial blocks to pool
    BufferBlock* block1 = BufferPool_addBlock(&pool, 1024, (void*)0x1000);
    BufferBlock* block2 = BufferPool_addBlock(&pool, 2048, (void*)0x2000);
    BufferBlock* block3 = BufferPool_addBlock(&pool, 4096, (void*)0x3000);

    __CPROVER_assume(block1 != NULL && block2 != NULL && block3 != NULL);

    // Create TLS cache
    TLSBlockCache cache;
    TLSCache_init(&cache);

    // ==== PHASE 1: Normal operation ====

    // Allocate blocks
    BufferBlock_acquire(block1);
    BufferBlock_acquire(block2);
    BufferBlock_acquire(block3);

    // Release block1 and try to cache it
    BufferBlock_release(block1);
    bool cached1 = TLSCache_tryPut(&cache, block1);
    __CPROVER_assert(cached1, "First block should be cached");

    // Release block2 and try to cache it
    BufferBlock_release(block2);
    bool cached2 = TLSCache_tryPut(&cache, block2);
    __CPROVER_assert(cached2, "Second block should be cached");

    // ==== PHASE 2: Verify cache state ====
    __CPROVER_assert(cache.num_blocks == 2, "Cache should have 2 blocks");
    __CPROVER_assert(cache.total_size == 1024 + 2048, "Cache size should be 3072");

    // ==== PHASE 3: Try to get a cached block ====
    size_t request_size = nondet_size_t();
    __CPROVER_assume(request_size > 0 && request_size <= 2048);

    BufferBlock* retrieved = TLSCache_tryGet(&cache, request_size);

    if (retrieved != NULL) {
        // Verify block is now in use
        __CPROVER_assert(BufferBlock_isInUse(retrieved),
            "Retrieved block must be in use");

        // Verify it was one of our cached blocks
        __CPROVER_assert(retrieved == block1 || retrieved == block2,
            "Retrieved block must be from cache");

        // Release it back
        BufferBlock_release(retrieved);
    }

    // ==== PHASE 4: Shutdown scenario ====

    // Simulate shutdown
    g_allocator_alive = false;

    // Flush cache (should work even during shutdown)
    TLSCache_flush(&cache, &pool);

    // Verify cache is empty
    __CPROVER_assert(cache.num_blocks == 0, "Cache must be empty after flush");
    __CPROVER_assert(cache.total_size == 0, "Cache size must be 0 after flush");

    // ==== PHASE 5: Operations should fail during shutdown ====
    BufferBlock_release(block3);

    // tryGet should return NULL during shutdown
    BufferBlock* should_be_null = TLSCache_tryGet(&cache, 1024);
    __CPROVER_assert(should_be_null == NULL,
        "tryGet should fail during shutdown");

    return 0;
}
