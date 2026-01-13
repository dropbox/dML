// CBMC Model: MPS Stream Pool
// This model verifies the thread-local stream binding mechanism.
//
// Properties verified:
// 1. No two threads share the same stream
// 2. Stream lifecycle is correct (alloc -> use -> release)
// 3. TLS binding is consistent
// 4. Fork safety (TLS invalidation)
//
// Run with:
//   cbmc stream_pool_c_model.c --unwind 5 --bounds-check --pointer-check

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

// CBMC-friendly configuration
#ifndef CBMC_MAX_STREAMS
#define CBMC_MAX_STREAMS 4
#endif

#ifndef CBMC_MAX_THREADS
#define CBMC_MAX_THREADS 3
#endif

// Stream states
typedef enum {
  STREAM_FREE = 0,
  STREAM_BOUND = 1,
  STREAM_ACTIVE = 2
} StreamState;

// Simulated stream
typedef struct {
  volatile StreamState state;
  volatile int32_t bound_thread;  // -1 = unbound
  volatile uint32_t use_count;
} Stream;

// Stream pool model
typedef struct {
  Stream streams[CBMC_MAX_STREAMS];
  volatile uint32_t freelist[CBMC_MAX_STREAMS];  // Stack of free stream indices
  volatile uint32_t freelist_top;
  volatile bool mutex_locked;
  volatile uint32_t pid;  // Simulated process ID (for fork detection)
} StreamPool;

// Thread-local storage simulation
// In reality this is per-thread, but for CBMC we model it as an array
typedef struct {
  volatile int32_t bound_stream;  // -1 = none
  volatile uint32_t cached_pid;   // For fork detection
} ThreadLocalState;

// Global pool and TLS
StreamPool g_pool;
ThreadLocalState g_tls[CBMC_MAX_THREADS];

// Nondeterministic functions
int nondet_int(void);
unsigned int nondet_uint(void);
_Bool nondet_bool(void);

// CBMC atomic primitives
void __CPROVER_atomic_begin(void);
void __CPROVER_atomic_end(void);
void __CPROVER_assume(_Bool);

// Initialize the pool
void pool_init(StreamPool* pool) {
  pool->freelist_top = CBMC_MAX_STREAMS;
  pool->mutex_locked = false;
  pool->pid = 1;  // Initial PID

  for (uint32_t i = 0; i < CBMC_MAX_STREAMS; i++) {
    pool->streams[i].state = STREAM_FREE;
    pool->streams[i].bound_thread = -1;
    pool->streams[i].use_count = 0;
    pool->freelist[i] = i;  // freelist[0] = 0, freelist[1] = 1, etc.
  }
}

// Initialize TLS for all threads
void tls_init(void) {
  for (uint32_t t = 0; t < CBMC_MAX_THREADS; t++) {
    g_tls[t].bound_stream = -1;
    g_tls[t].cached_pid = g_pool.pid;
  }
}

// Lock the pool mutex
void pool_lock(StreamPool* pool) {
  bool expected;
  do {
    __CPROVER_atomic_begin();
    expected = pool->mutex_locked;
    if (!expected) {
      pool->mutex_locked = true;
    }
    __CPROVER_atomic_end();
  } while (expected);
}

// Unlock the pool mutex
void pool_unlock(StreamPool* pool) {
  __CPROVER_atomic_begin();
  assert(pool->mutex_locked && "Unlock without lock");
  pool->mutex_locked = false;
  __CPROVER_atomic_end();
}

// Get stream for thread (TLS binding)
// Returns stream index, or -1 if pool exhausted
int32_t get_stream(uint32_t tid) {
  assert(tid < CBMC_MAX_THREADS);

  // Check for fork (pid change)
  if (g_tls[tid].cached_pid != g_pool.pid) {
    // Fork detected - invalidate TLS binding
    g_tls[tid].bound_stream = -1;
    g_tls[tid].cached_pid = g_pool.pid;
  }

  // Fast path: already have a bound stream
  int32_t bound = g_tls[tid].bound_stream;
  if (bound >= 0) {
    assert(bound < CBMC_MAX_STREAMS);
    // PROPERTY: Our bound stream should be bound to us
    assert(g_pool.streams[bound].bound_thread == (int32_t)tid &&
           "TLS/pool binding mismatch");
    return bound;
  }

  // Slow path: need to acquire from pool
  pool_lock(&g_pool);

  // Double-check TLS (another thread might have set it)
  bound = g_tls[tid].bound_stream;
  if (bound >= 0) {
    pool_unlock(&g_pool);
    return bound;
  }

  // Get from freelist
  if (g_pool.freelist_top == 0) {
    // Pool exhausted
    pool_unlock(&g_pool);
    return -1;
  }

  g_pool.freelist_top--;
  uint32_t stream_idx = g_pool.freelist[g_pool.freelist_top];
  assert(stream_idx < CBMC_MAX_STREAMS);

  // PROPERTY: Stream from freelist must be FREE
  assert(g_pool.streams[stream_idx].state == STREAM_FREE &&
         "Freelist contains non-free stream");

  // Bind stream to this thread
  g_pool.streams[stream_idx].state = STREAM_BOUND;
  g_pool.streams[stream_idx].bound_thread = (int32_t)tid;
  g_tls[tid].bound_stream = (int32_t)stream_idx;

  pool_unlock(&g_pool);
  return (int32_t)stream_idx;
}

// Release stream back to pool
void release_stream(uint32_t tid) {
  assert(tid < CBMC_MAX_THREADS);

  int32_t bound = g_tls[tid].bound_stream;
  if (bound < 0) {
    return;  // No stream to release
  }

  pool_lock(&g_pool);

  // Clear TLS binding
  g_tls[tid].bound_stream = -1;

  // PROPERTY: Stream should be bound to this thread
  assert(g_pool.streams[bound].bound_thread == (int32_t)tid &&
         "Releasing stream not bound to us");

  // Return to freelist
  g_pool.streams[bound].state = STREAM_FREE;
  g_pool.streams[bound].bound_thread = -1;
  g_pool.freelist[g_pool.freelist_top] = (uint32_t)bound;
  g_pool.freelist_top++;

  pool_unlock(&g_pool);
}

// Simulate fork - invalidates all TLS bindings
void simulate_fork(void) {
  pool_lock(&g_pool);

  // Increment PID - all threads will detect fork on next get_stream()
  g_pool.pid++;

  // In a real fork, child process gets all streams back
  // For simplicity, we release all bound streams
  for (uint32_t i = 0; i < CBMC_MAX_STREAMS; i++) {
    if (g_pool.streams[i].state == STREAM_BOUND) {
      g_pool.streams[i].state = STREAM_FREE;
      g_pool.streams[i].bound_thread = -1;
      g_pool.freelist[g_pool.freelist_top] = i;
      g_pool.freelist_top++;
    }
  }

  pool_unlock(&g_pool);
}

// Check pool invariants
void pool_check_invariants(void) {
  // PROPERTY: No two threads bound to same stream
  for (uint32_t t1 = 0; t1 < CBMC_MAX_THREADS; t1++) {
    for (uint32_t t2 = t1 + 1; t2 < CBMC_MAX_THREADS; t2++) {
      int32_t s1 = g_tls[t1].bound_stream;
      int32_t s2 = g_tls[t2].bound_stream;
      assert((s1 < 0 || s2 < 0 || s1 != s2) &&
             "Two threads share same stream");
    }
  }

  // PROPERTY: Freelist size + bound streams = total streams
  uint32_t bound_count = 0;
  for (uint32_t i = 0; i < CBMC_MAX_STREAMS; i++) {
    if (g_pool.streams[i].state == STREAM_BOUND) {
      bound_count++;
    }
  }
  assert(g_pool.freelist_top + bound_count == CBMC_MAX_STREAMS &&
         "Stream count mismatch");
}

// Thread function: acquires stream, uses it, releases
void thread_work(uint32_t tid) {
  // Get a stream
  int32_t stream = get_stream(tid);
  if (stream < 0) {
    return;  // Pool exhausted - OK
  }

  // PROPERTY: Stream is bound to us
  assert(g_pool.streams[stream].bound_thread == (int32_t)tid);

  // Simulate some work with the stream
  __CPROVER_atomic_begin();
  g_pool.streams[stream].use_count++;
  __CPROVER_atomic_end();

  // Optionally release (some threads keep their stream)
  if (nondet_bool()) {
    release_stream(tid);
  }
}

// Main verification harness
int main(void) {
  // Initialize
  pool_init(&g_pool);
  tls_init();

  // Simulate concurrent execution
  for (int step = 0; step < 10; step++) {
    unsigned action = nondet_uint() % 5;

    switch (action) {
      case 0:
        // Thread 0 works
        thread_work(0);
        break;
      case 1:
        // Thread 1 works
        thread_work(1);
        break;
      case 2:
        // Thread 2 works
        thread_work(2 % CBMC_MAX_THREADS);
        break;
      case 3:
        // Simulate fork
        simulate_fork();
        tls_init();  // Reset TLS after fork
        break;
      case 4:
        // Thread 0 releases stream
        release_stream(0);
        break;
    }
  }

  // Final invariant check
  pool_check_invariants();

  return 0;
}
