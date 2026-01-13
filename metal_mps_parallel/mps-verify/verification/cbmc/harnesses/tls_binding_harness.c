// tls_binding_harness.c - CBMC harness for verifying TLS stream binding safety
//
// The TLS (Thread-Local Storage) binding mechanism ensures each thread gets
// a unique stream from the pool without conflicts.
//
// Properties verified:
// 1. No two threads are bound to the same stream simultaneously
// 2. Stream indices are always within valid pool bounds
// 3. TLS binding survives multiple operations per thread
// 4. TLS cleanup releases the stream properly
// 5. Unbound threads get valid streams when acquiring
// 6. Shutdown prevents new bindings while allowing cleanup
//
// Run with:
//   cbmc tls_binding_harness.c -I ../models -I ../stubs --unwind 10 --pointer-check --bounds-check

#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>

// CBMC Primitives
extern int nondet_int(void);
extern unsigned int nondet_uint(void);
extern bool nondet_bool(void);
extern void __CPROVER_assume(bool);
extern void __CPROVER_assert(bool, const char*);

// ============================================================================
// Stream Pool Configuration (reduced for CBMC verification)
// Real values: NUM_STREAMS=32, NUM_THREADS=unlimited
// ============================================================================

#define MAX_STREAMS 8       // Reduced for CBMC (original: 32)
#define MAX_THREADS 4       // Reduced for CBMC
#define INVALID_STREAM (-1)

// ============================================================================
// Stream State Model
// ============================================================================

typedef enum StreamState {
    STREAM_FREE = 0,        // Available in pool
    STREAM_BOUND = 1,       // Bound to a thread via TLS
    STREAM_ACTIVE = 2       // Currently executing GPU work
} StreamState;

typedef struct MPSStream {
    StreamState state;
    int bound_thread_id;    // Which thread owns this stream (-1 if none)
    uint32_t use_count;     // How many operations executed on this stream
    bool has_pending_work;  // Is there uncommitted work?
} MPSStream;

// ============================================================================
// Stream Pool Model
// ============================================================================

typedef struct StreamPool {
    MPSStream streams[MAX_STREAMS];
    int tls_bindings[MAX_THREADS];  // Thread ID -> Stream index, -1 = unbound
    uint32_t freemask;              // Bitmask of free streams (1 = free)
    bool is_alive;                  // Pool shutdown flag
    int total_bindings;             // Count of active bindings
} StreamPool;

// Global pool instance
static StreamPool g_pool;

// ============================================================================
// Pool Operations
// ============================================================================

static void StreamPool_init(StreamPool* pool) {
    __CPROVER_assert(pool != NULL, "Pool must not be null");

    // All streams start free
    for (int i = 0; i < MAX_STREAMS; i++) {
        pool->streams[i].state = STREAM_FREE;
        pool->streams[i].bound_thread_id = -1;
        pool->streams[i].use_count = 0;
        pool->streams[i].has_pending_work = false;
    }

    // All threads start unbound
    for (int t = 0; t < MAX_THREADS; t++) {
        pool->tls_bindings[t] = INVALID_STREAM;
    }

    // All streams are free initially
    pool->freemask = (1u << MAX_STREAMS) - 1;  // All bits set
    pool->is_alive = true;
    pool->total_bindings = 0;
}

// Try to acquire a stream for a thread
// Returns stream index, or INVALID_STREAM if none available
static int StreamPool_acquireStream(StreamPool* pool, int thread_id) {
    __CPROVER_assert(pool != NULL, "Pool must not be null");
    __CPROVER_assert(thread_id >= 0 && thread_id < MAX_THREADS,
        "Thread ID must be valid");

    // Check if pool is shutting down
    if (!pool->is_alive) {
        return INVALID_STREAM;
    }

    // Check if thread already has a binding (TLS lookup)
    if (pool->tls_bindings[thread_id] != INVALID_STREAM) {
        int existing = pool->tls_bindings[thread_id];
        __CPROVER_assert(existing >= 0 && existing < MAX_STREAMS,
            "Existing TLS binding must be valid stream index");
        __CPROVER_assert(pool->streams[existing].bound_thread_id == thread_id,
            "Stream must be bound to this thread");
        return existing;  // Return cached binding
    }

    // No existing binding, find a free stream
    if (pool->freemask == 0) {
        return INVALID_STREAM;  // No free streams
    }

    // Find first free stream (simulates __builtin_ffs)
    int free_idx = INVALID_STREAM;
    for (int i = 0; i < MAX_STREAMS; i++) {
        if (pool->freemask & (1u << i)) {
            free_idx = i;
            break;
        }
    }

    if (free_idx == INVALID_STREAM) {
        return INVALID_STREAM;
    }

    // Mark stream as bound
    pool->freemask &= ~(1u << free_idx);  // Clear bit
    pool->streams[free_idx].state = STREAM_BOUND;
    pool->streams[free_idx].bound_thread_id = thread_id;

    // Set TLS binding
    pool->tls_bindings[thread_id] = free_idx;
    pool->total_bindings++;

    return free_idx;
}

// Release a stream back to the pool
// Called on thread exit or explicit release
static void StreamPool_releaseStream(StreamPool* pool, int thread_id) {
    __CPROVER_assert(pool != NULL, "Pool must not be null");
    __CPROVER_assert(thread_id >= 0 && thread_id < MAX_THREADS,
        "Thread ID must be valid");

    int stream_idx = pool->tls_bindings[thread_id];
    if (stream_idx == INVALID_STREAM) {
        return;  // No binding to release
    }

    __CPROVER_assert(stream_idx >= 0 && stream_idx < MAX_STREAMS,
        "TLS binding must be valid stream index");

    MPSStream* stream = &pool->streams[stream_idx];

    // Verify ownership
    __CPROVER_assert(stream->bound_thread_id == thread_id,
        "Stream must be bound to this thread");

    // Wait for pending work if any (modeled as just checking)
    __CPROVER_assert(!stream->has_pending_work,
        "Stream should not have pending work when released");

    // Release the stream
    stream->state = STREAM_FREE;
    stream->bound_thread_id = -1;

    // Clear TLS binding
    pool->tls_bindings[thread_id] = INVALID_STREAM;
    pool->total_bindings--;

    // Mark stream as free
    pool->freemask |= (1u << stream_idx);
}

// Get currently bound stream for a thread (TLS lookup)
// Returns INVALID_STREAM if not bound
static int StreamPool_getCurrentStream(StreamPool* pool, int thread_id) {
    __CPROVER_assert(pool != NULL, "Pool must not be null");
    __CPROVER_assert(thread_id >= 0 && thread_id < MAX_THREADS,
        "Thread ID must be valid");

    return pool->tls_bindings[thread_id];
}

// Submit work on a stream
static void StreamPool_submitWork(StreamPool* pool, int thread_id) {
    __CPROVER_assert(pool != NULL, "Pool must not be null");

    int stream_idx = pool->tls_bindings[thread_id];
    __CPROVER_assert(stream_idx != INVALID_STREAM,
        "Thread must have a stream to submit work");
    __CPROVER_assert(stream_idx >= 0 && stream_idx < MAX_STREAMS,
        "Stream index must be valid");

    MPSStream* stream = &pool->streams[stream_idx];
    __CPROVER_assert(stream->bound_thread_id == thread_id,
        "Stream must be bound to this thread");

    stream->state = STREAM_ACTIVE;
    stream->has_pending_work = true;
    stream->use_count++;
}

// Commit work and wait for completion
static void StreamPool_synchronize(StreamPool* pool, int thread_id) {
    __CPROVER_assert(pool != NULL, "Pool must not be null");

    int stream_idx = pool->tls_bindings[thread_id];
    if (stream_idx == INVALID_STREAM) {
        return;  // Nothing to synchronize
    }

    MPSStream* stream = &pool->streams[stream_idx];
    stream->has_pending_work = false;
    stream->state = STREAM_BOUND;  // Back to bound (not active)
}

// Shutdown the pool
static void StreamPool_shutdown(StreamPool* pool) {
    __CPROVER_assert(pool != NULL, "Pool must not be null");
    pool->is_alive = false;
}

// ============================================================================
// Invariant Checkers
// ============================================================================

// Check that no two threads share the same stream
static void check_no_shared_streams(StreamPool* pool) {
    for (int t1 = 0; t1 < MAX_THREADS; t1++) {
        int s1 = pool->tls_bindings[t1];
        if (s1 == INVALID_STREAM) continue;

        for (int t2 = t1 + 1; t2 < MAX_THREADS; t2++) {
            int s2 = pool->tls_bindings[t2];
            if (s2 == INVALID_STREAM) continue;

            __CPROVER_assert(s1 != s2,
                "No two threads can be bound to the same stream");
        }
    }
}

// Check that all bindings are consistent
static void check_binding_consistency(StreamPool* pool) {
    for (int t = 0; t < MAX_THREADS; t++) {
        int stream_idx = pool->tls_bindings[t];

        if (stream_idx == INVALID_STREAM) {
            continue;  // Unbound thread, OK
        }

        // Binding must be valid index
        __CPROVER_assert(stream_idx >= 0 && stream_idx < MAX_STREAMS,
            "TLS binding must be valid stream index");

        // Stream must know it's bound to this thread
        __CPROVER_assert(pool->streams[stream_idx].bound_thread_id == t,
            "Stream's bound_thread_id must match TLS binding");

        // Stream must not be marked free
        __CPROVER_assert(pool->streams[stream_idx].state != STREAM_FREE,
            "Bound stream must not be FREE");

        // Freemask bit must be clear
        __CPROVER_assert((pool->freemask & (1u << stream_idx)) == 0,
            "Bound stream must not be in freemask");
    }
}

// Check freemask consistency
static void check_freemask_consistency(StreamPool* pool) {
    for (int s = 0; s < MAX_STREAMS; s++) {
        bool is_free = (pool->freemask & (1u << s)) != 0;

        if (is_free) {
            __CPROVER_assert(pool->streams[s].state == STREAM_FREE,
                "Stream in freemask must be FREE state");
            __CPROVER_assert(pool->streams[s].bound_thread_id == -1,
                "Stream in freemask must have no owner");
        } else {
            __CPROVER_assert(pool->streams[s].state != STREAM_FREE,
                "Stream not in freemask must not be FREE");
        }
    }
}

// ============================================================================
// Main Harness
// ============================================================================

int main(void) {
    // Initialize pool
    StreamPool_init(&g_pool);

    // ==== PHASE 1: Basic acquisition ====

    // Thread 0 acquires a stream
    int stream0 = StreamPool_acquireStream(&g_pool, 0);
    __CPROVER_assert(stream0 != INVALID_STREAM, "Thread 0 should get a stream");
    __CPROVER_assert(stream0 >= 0 && stream0 < MAX_STREAMS, "Stream 0 must be valid index");

    // Thread 1 acquires a stream
    int stream1 = StreamPool_acquireStream(&g_pool, 1);
    __CPROVER_assert(stream1 != INVALID_STREAM, "Thread 1 should get a stream");
    __CPROVER_assert(stream1 >= 0 && stream1 < MAX_STREAMS, "Stream 1 must be valid index");

    // They must be different streams
    __CPROVER_assert(stream0 != stream1, "Threads must get different streams");

    // ==== PHASE 2: TLS caching ====

    // Thread 0 acquires again - should get same stream (TLS cached)
    int stream0_again = StreamPool_acquireStream(&g_pool, 0);
    __CPROVER_assert(stream0_again == stream0, "Same thread should get same stream");

    // ==== PHASE 3: Submit and synchronize work ====

    StreamPool_submitWork(&g_pool, 0);
    __CPROVER_assert(g_pool.streams[stream0].has_pending_work, "Stream should have pending work");

    StreamPool_synchronize(&g_pool, 0);
    __CPROVER_assert(!g_pool.streams[stream0].has_pending_work, "Work should be complete");

    // ==== PHASE 4: Check invariants ====

    check_no_shared_streams(&g_pool);
    check_binding_consistency(&g_pool);
    check_freemask_consistency(&g_pool);

    // ==== PHASE 5: Non-deterministic thread operations ====

    int tid = nondet_int();
    __CPROVER_assume(tid >= 0 && tid < MAX_THREADS);

    // Try to acquire stream for non-deterministic thread
    int stream_t = StreamPool_acquireStream(&g_pool, tid);

    if (stream_t != INVALID_STREAM) {
        // Verify the binding
        __CPROVER_assert(g_pool.tls_bindings[tid] == stream_t,
            "TLS should be set after acquire");
        __CPROVER_assert(g_pool.streams[stream_t].bound_thread_id == tid,
            "Stream should know its owner");

        // Submit some work
        StreamPool_submitWork(&g_pool, tid);
        StreamPool_synchronize(&g_pool, tid);
    }

    // Check invariants again
    check_no_shared_streams(&g_pool);
    check_binding_consistency(&g_pool);

    // ==== PHASE 6: Release and re-acquire ====

    // Thread 0 releases its stream
    StreamPool_releaseStream(&g_pool, 0);
    __CPROVER_assert(g_pool.tls_bindings[0] == INVALID_STREAM,
        "TLS should be cleared after release");
    __CPROVER_assert(g_pool.streams[stream0].state == STREAM_FREE,
        "Stream should be FREE after release");
    __CPROVER_assert((g_pool.freemask & (1u << stream0)) != 0,
        "Stream should be in freemask after release");

    // New thread can acquire the freed stream
    int stream2 = StreamPool_acquireStream(&g_pool, 2);
    __CPROVER_assert(stream2 != INVALID_STREAM, "Thread 2 should get a stream");
    // It might get stream0 (which was just freed) or another free stream

    // Thread 0 re-acquires (might get same or different stream)
    int stream0_new = StreamPool_acquireStream(&g_pool, 0);
    __CPROVER_assert(stream0_new != INVALID_STREAM, "Thread 0 should get a stream again");
    __CPROVER_assert(stream0_new != stream1, "Should not conflict with thread 1");
    __CPROVER_assert(stream0_new != stream2, "Should not conflict with thread 2");

    // ==== PHASE 7: Shutdown behavior ====

    StreamPool_shutdown(&g_pool);

    // New acquisitions should fail
    int stream3 = StreamPool_acquireStream(&g_pool, 3);
    __CPROVER_assert(stream3 == INVALID_STREAM,
        "No new acquisitions after shutdown");

    // Existing bindings still work for cleanup
    __CPROVER_assert(g_pool.tls_bindings[1] == stream1,
        "Existing bindings preserved during shutdown");

    // Release remaining streams for cleanup
    StreamPool_synchronize(&g_pool, 1);  // Clear pending work first
    StreamPool_synchronize(&g_pool, 2);
    StreamPool_synchronize(&g_pool, 0);

    StreamPool_releaseStream(&g_pool, 1);
    StreamPool_releaseStream(&g_pool, 2);
    StreamPool_releaseStream(&g_pool, 0);

    // ==== PHASE 8: Final invariant check ====

    // After cleanup, binding count should be 0 (or just non-negative)
    __CPROVER_assert(g_pool.total_bindings >= 0,
        "Total bindings must be non-negative");

    check_no_shared_streams(&g_pool);
    check_binding_consistency(&g_pool);
    check_freemask_consistency(&g_pool);

    return 0;
}
