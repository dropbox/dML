// CBMC Model: MPS Heap Allocator with ABA Detection
// This model verifies the buffer allocation/free mechanism.
//
// Properties verified:
// 1. ABA counter monotonicity (prevents ABA bugs)
// 2. No buffer ID reuse within an execution
// 3. Buffer state machine correctness
// 4. No use-after-free
// 5. Pool invariants maintained
//
// Run with:
//   cbmc allocator_c_model.c --unwind 15 --bounds-check --pointer-check

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

// CBMC-friendly configuration
#ifndef CBMC_MAX_BUFFERS
#define CBMC_MAX_BUFFERS 4
#endif

#ifndef CBMC_MAX_THREADS
#define CBMC_MAX_THREADS 2
#endif

#ifndef CBMC_MAX_STREAMS
#define CBMC_MAX_STREAMS 2
#endif

// Buffer states (matching TLA+ spec)
typedef enum {
  BUFFER_FREE = 0,
  BUFFER_ALLOCATED = 1,
  BUFFER_IN_USE = 2,
  BUFFER_PENDING_FREE = 3
} BufferState;

// Buffer block model
typedef struct {
  volatile BufferState state;
  volatile uint64_t buf_id;          // Unique ID assigned by buffer_counter
  volatile uint32_t alloc_stream;    // Stream that allocated this buffer (0 = none)
  volatile uint32_t stream_mask;     // Bitmask of streams that have used this buffer
  volatile uint32_t use_count;       // For LRU tracking
} BufferBlock;

// Heap allocator model
typedef struct {
  BufferBlock buffers[CBMC_MAX_BUFFERS];

  // ABA counter - KEY invariant: only increases, never wraps/resets
  volatile uint64_t buffer_counter;

  // Freelist of available buffer indices
  volatile uint32_t freelist[CBMC_MAX_BUFFERS];
  volatile uint32_t freelist_size;

  // Pending free buffers (waiting for GPU completion)
  volatile bool pending[CBMC_MAX_BUFFERS];
  volatile uint32_t pending_count;

  // Pool mutex (single pool for simplicity)
  volatile bool pool_mutex_locked;

  // Statistics
  volatile uint32_t total_allocated;
  volatile uint32_t total_freed;
} HeapAllocator;

// Global allocator
HeapAllocator g_alloc;

// Track all IDs ever assigned (for verifying no reuse)
volatile uint64_t g_assigned_ids[CBMC_MAX_BUFFERS * 4];  // Generous size
volatile uint32_t g_assigned_count;

// Previous value of buffer_counter (for monotonicity check)
volatile uint64_t g_prev_counter;

// Nondeterministic functions (CBMC built-ins)
int nondet_int(void);
unsigned int nondet_uint(void);
_Bool nondet_bool(void);

// CBMC atomic primitives
void __CPROVER_atomic_begin(void);
void __CPROVER_atomic_end(void);
void __CPROVER_assume(_Bool);

// Initialize allocator
void alloc_init(HeapAllocator* a) {
  a->buffer_counter = 0;
  a->freelist_size = CBMC_MAX_BUFFERS;
  a->pool_mutex_locked = false;
  a->pending_count = 0;
  a->total_allocated = 0;
  a->total_freed = 0;

  for (uint32_t i = 0; i < CBMC_MAX_BUFFERS; i++) {
    a->buffers[i].state = BUFFER_FREE;
    a->buffers[i].buf_id = 0;
    a->buffers[i].alloc_stream = 0;
    a->buffers[i].stream_mask = 0;
    a->buffers[i].use_count = 0;
    a->freelist[i] = i;  // freelist[0] = 0, freelist[1] = 1, etc.
    a->pending[i] = false;
  }

  g_assigned_count = 0;
  g_prev_counter = 0;
}

// Lock pool mutex
void pool_lock(HeapAllocator* a) {
  bool expected;
  do {
    __CPROVER_atomic_begin();
    expected = a->pool_mutex_locked;
    if (!expected) {
      a->pool_mutex_locked = true;
    }
    __CPROVER_atomic_end();
  } while (expected);
}

// Unlock pool mutex
void pool_unlock(HeapAllocator* a) {
  __CPROVER_atomic_begin();
  assert(a->pool_mutex_locked && "Unlock without lock");
  a->pool_mutex_locked = false;
  __CPROVER_atomic_end();
}

// Allocate a buffer on a given stream
// Returns buffer index, or -1 on failure
int alloc_buffer(HeapAllocator* a, uint32_t stream) {
  assert(stream > 0 && stream <= CBMC_MAX_STREAMS);

  pool_lock(a);

  // Check freelist
  if (a->freelist_size == 0) {
    // Pool exhausted
    pool_unlock(a);
    return -1;
  }

  // Pop from freelist
  a->freelist_size--;
  uint32_t buf_idx = a->freelist[a->freelist_size];
  assert(buf_idx < CBMC_MAX_BUFFERS);

  // PROPERTY: Buffer from freelist must be FREE
  assert(a->buffers[buf_idx].state == BUFFER_FREE &&
         "Freelist contains non-free buffer");

  // Increment ABA counter atomically
  __CPROVER_atomic_begin();
  uint64_t old_counter = a->buffer_counter;
  a->buffer_counter = old_counter + 1;
  uint64_t new_id = a->buffer_counter;

  // PROPERTY: ABA monotonicity - counter only increases
  assert(a->buffer_counter > old_counter &&
         "ABA counter did not increase");
  assert(a->buffer_counter >= g_prev_counter &&
         "ABA counter decreased (should never happen)");
  g_prev_counter = a->buffer_counter;
  __CPROVER_atomic_end();

  // Record the assigned ID for uniqueness checking
  if (g_assigned_count < CBMC_MAX_BUFFERS * 4) {
    g_assigned_ids[g_assigned_count] = new_id;
    g_assigned_count++;
  }

  // Initialize buffer
  a->buffers[buf_idx].state = BUFFER_ALLOCATED;
  a->buffers[buf_idx].buf_id = new_id;
  a->buffers[buf_idx].alloc_stream = stream;
  a->buffers[buf_idx].stream_mask = (1u << (stream - 1));  // Mark stream as user
  a->buffers[buf_idx].use_count = 1;

  a->total_allocated++;

  pool_unlock(a);
  return (int)buf_idx;
}

// Mark buffer as in-use (GPU work submitted)
void use_buffer(HeapAllocator* a, uint32_t buf_idx) {
  assert(buf_idx < CBMC_MAX_BUFFERS);

  __CPROVER_atomic_begin();
  // PROPERTY: Can only use allocated buffer
  assert(a->buffers[buf_idx].state == BUFFER_ALLOCATED &&
         "Using non-allocated buffer");
  a->buffers[buf_idx].state = BUFFER_IN_USE;
  a->buffers[buf_idx].use_count++;
  __CPROVER_atomic_end();
}

// Record cross-stream usage (CUDA pattern: recordStream)
void record_stream(HeapAllocator* a, uint32_t buf_idx, uint32_t stream) {
  assert(buf_idx < CBMC_MAX_BUFFERS);
  assert(stream > 0 && stream <= CBMC_MAX_STREAMS);

  pool_lock(a);

  // PROPERTY: Can only record on allocated/in-use buffer
  assert(a->buffers[buf_idx].state == BUFFER_ALLOCATED ||
         a->buffers[buf_idx].state == BUFFER_IN_USE &&
         "Recording stream on invalid buffer");

  // Add stream to usage mask
  a->buffers[buf_idx].stream_mask |= (1u << (stream - 1));

  pool_unlock(a);
}

// Count number of streams that used a buffer
uint32_t count_stream_uses(HeapAllocator* a, uint32_t buf_idx) {
  uint32_t mask = a->buffers[buf_idx].stream_mask;
  uint32_t count = 0;
  while (mask) {
    count += (mask & 1);
    mask >>= 1;
  }
  return count;
}

// Free a buffer
void free_buffer(HeapAllocator* a, uint32_t buf_idx) {
  assert(buf_idx < CBMC_MAX_BUFFERS);

  pool_lock(a);

  // PROPERTY: Can only free allocated/in-use buffer
  assert(a->buffers[buf_idx].state == BUFFER_ALLOCATED ||
         a->buffers[buf_idx].state == BUFFER_IN_USE &&
         "Freeing invalid buffer");

  uint32_t stream_count = count_stream_uses(a, buf_idx);

  if (stream_count > 1) {
    // Cross-stream usage: mark pending until GPU completion
    a->buffers[buf_idx].state = BUFFER_PENDING_FREE;
    a->pending[buf_idx] = true;
    a->pending_count++;
  } else {
    // Single stream: return directly to freelist
    a->buffers[buf_idx].state = BUFFER_FREE;
    a->buffers[buf_idx].alloc_stream = 0;
    a->buffers[buf_idx].stream_mask = 0;
    // Note: buf_id is NOT cleared - it remains for debugging

    // Push to freelist
    a->freelist[a->freelist_size] = buf_idx;
    a->freelist_size++;
  }

  a->total_freed++;

  pool_unlock(a);
}

// Process pending buffer (GPU completion handler)
void process_pending(HeapAllocator* a, uint32_t buf_idx) {
  assert(buf_idx < CBMC_MAX_BUFFERS);

  // Check if pending (quick check without lock)
  bool is_pending;
  __CPROVER_atomic_begin();
  is_pending = a->pending[buf_idx];
  __CPROVER_atomic_end();

  if (!is_pending) {
    return;  // Not pending
  }

  pool_lock(a);

  // Re-check under lock
  if (!a->pending[buf_idx]) {
    pool_unlock(a);
    return;
  }

  // PROPERTY: Pending buffer must be in PENDING_FREE state
  assert(a->buffers[buf_idx].state == BUFFER_PENDING_FREE &&
         "Processing non-pending buffer");

  // Return to freelist
  a->buffers[buf_idx].state = BUFFER_FREE;
  a->buffers[buf_idx].alloc_stream = 0;
  a->buffers[buf_idx].stream_mask = 0;
  a->pending[buf_idx] = false;
  a->pending_count--;

  // Push to freelist
  a->freelist[a->freelist_size] = buf_idx;
  a->freelist_size++;

  pool_unlock(a);
}

// Check allocator invariants
void check_invariants(HeapAllocator* a) {
  // PROPERTY: ABA counter only increases
  assert(a->buffer_counter >= g_prev_counter &&
         "ABA counter regression");

  // PROPERTY: No ID reuse among assigned IDs
  for (uint32_t i = 0; i < g_assigned_count; i++) {
    for (uint32_t j = i + 1; j < g_assigned_count; j++) {
      assert(g_assigned_ids[i] != g_assigned_ids[j] &&
             "Buffer ID reused (ABA bug!)");
    }
  }

  // PROPERTY: Freelist + allocated + pending = total buffers
  uint32_t free_count = a->freelist_size;
  uint32_t allocated_count = 0;
  uint32_t pending_count = 0;

  for (uint32_t i = 0; i < CBMC_MAX_BUFFERS; i++) {
    if (a->buffers[i].state == BUFFER_ALLOCATED ||
        a->buffers[i].state == BUFFER_IN_USE) {
      allocated_count++;
    }
    if (a->buffers[i].state == BUFFER_PENDING_FREE) {
      pending_count++;
    }
  }

  assert(free_count + allocated_count + pending_count == CBMC_MAX_BUFFERS &&
         "Buffer count mismatch");
  assert(pending_count == a->pending_count &&
         "Pending count mismatch");

  // PROPERTY: Free buffers are in freelist
  for (uint32_t i = 0; i < CBMC_MAX_BUFFERS; i++) {
    if (a->buffers[i].state == BUFFER_FREE) {
      bool in_freelist = false;
      for (uint32_t j = 0; j < a->freelist_size; j++) {
        if (a->freelist[j] == i) {
          in_freelist = true;
          break;
        }
      }
      assert(in_freelist && "Free buffer not in freelist");
    }
  }

  // PROPERTY: No use-after-free (free buffers have no alloc_stream)
  for (uint32_t i = 0; i < CBMC_MAX_BUFFERS; i++) {
    if (a->buffers[i].state == BUFFER_FREE) {
      assert(a->buffers[i].alloc_stream == 0 &&
             "Free buffer has alloc_stream (use-after-free risk)");
      assert(a->buffers[i].stream_mask == 0 &&
             "Free buffer has stream_mask (use-after-free risk)");
    }
  }
}

// Thread function: allocate, use, and free buffers
void thread_work(HeapAllocator* a, uint32_t tid, uint32_t stream) {
  // Allocate a buffer
  int buf = alloc_buffer(a, stream);
  if (buf < 0) {
    return;  // Pool exhausted - OK
  }

  // PROPERTY: Got a valid buffer with unique ID
  assert(a->buffers[buf].buf_id > 0 && "Allocated buffer has no ID");

  // Optionally use it
  if (nondet_bool()) {
    use_buffer(a, (uint32_t)buf);
  }

  // Optionally record cross-stream usage
  if (nondet_bool() && CBMC_MAX_STREAMS > 1) {
    uint32_t other_stream = (stream % CBMC_MAX_STREAMS) + 1;
    record_stream(a, (uint32_t)buf, other_stream);
  }

  // Free it
  free_buffer(a, (uint32_t)buf);
}

// Process any pending buffer
void background_completion(HeapAllocator* a) {
  for (uint32_t i = 0; i < CBMC_MAX_BUFFERS; i++) {
    if (a->pending[i]) {
      process_pending(a, i);
      break;  // Process one at a time
    }
  }
}

// Main verification harness
int main(void) {
  // Initialize
  alloc_init(&g_alloc);

  // Simulate concurrent execution
  for (int step = 0; step < 10; step++) {
    unsigned action = nondet_uint() % 5;

    switch (action) {
      case 0:
        // Thread 0, stream 1
        thread_work(&g_alloc, 0, 1);
        break;
      case 1:
        // Thread 1, stream 2
        thread_work(&g_alloc, 1, (2 <= CBMC_MAX_STREAMS) ? 2 : 1);
        break;
      case 2:
        // Thread 0, stream 2 (cross-stream)
        thread_work(&g_alloc, 0, (2 <= CBMC_MAX_STREAMS) ? 2 : 1);
        break;
      case 3:
        // Background completion
        background_completion(&g_alloc);
        break;
      case 4:
        // Another allocation on stream 1
        {
          int buf = alloc_buffer(&g_alloc, 1);
          if (buf >= 0) {
            // Verify ID is unique
            uint64_t id = g_alloc.buffers[buf].buf_id;
            for (uint32_t i = 0; i < CBMC_MAX_BUFFERS; i++) {
              if (i != (uint32_t)buf &&
                  g_alloc.buffers[i].state != BUFFER_FREE &&
                  g_alloc.buffers[i].buf_id > 0) {
                assert(g_alloc.buffers[i].buf_id != id &&
                       "Active buffers share ID (ABA bug!)");
              }
            }
            free_buffer(&g_alloc, (uint32_t)buf);
          }
        }
        break;
    }
  }

  // Final invariant check
  check_invariants(&g_alloc);

  // PROPERTY: ABA counter equals total allocations
  // (Each allocation increments counter exactly once)
  assert(g_alloc.buffer_counter == g_alloc.total_allocated &&
         "ABA counter != total allocations");

  return 0;
}
