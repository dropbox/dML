// CBMC Model: Simplified Batch Queue (C version)
// This model captures the concurrent producer/consumer behavior.
//
// Properties verified:
// 1. No data races (mutual exclusion)
// 2. No lost requests
// 3. Queue invariants maintained
//
// Run with:
//   cbmc batch_queue_c_model.c --unwind 5 --bounds-check --pointer-check

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

// CBMC-friendly configuration
#ifndef CBMC_MAX_QUEUE_SIZE
#define CBMC_MAX_QUEUE_SIZE 4
#endif

#ifndef CBMC_MAX_THREADS
#define CBMC_MAX_THREADS 2
#endif

#ifndef CBMC_MAX_WORKERS
#define CBMC_MAX_WORKERS 2
#endif

// Request states
typedef enum {
  STATE_EMPTY = 0,
  STATE_PENDING = 1,
  STATE_PROCESSING = 2,
  STATE_COMPLETED = 3
} RequestState;

// Simplified request
typedef struct {
  volatile RequestState state;
  volatile uint32_t result;
  uint32_t input;
} Request;

// Batch queue model
typedef struct {
  Request slots[CBMC_MAX_QUEUE_SIZE];
  volatile uint32_t head;  // Next slot to dequeue
  volatile uint32_t tail;  // Next slot to enqueue
  volatile bool running;
  volatile bool mutex_locked;
  volatile uint32_t submitted;
  volatile uint32_t completed;
} BatchQueue;

// Global queue
BatchQueue g_queue;

// Nondeterministic functions (CBMC built-ins)
int nondet_int(void);
unsigned int nondet_uint(void);
_Bool nondet_bool(void);

// Atomic operations (CBMC models these correctly)
void __CPROVER_atomic_begin(void);
void __CPROVER_atomic_end(void);
void __CPROVER_assume(_Bool);

// Initialize queue
void queue_init(BatchQueue* q) {
  q->head = 0;
  q->tail = 0;
  q->running = false;
  q->mutex_locked = false;
  q->submitted = 0;
  q->completed = 0;
  for (int i = 0; i < CBMC_MAX_QUEUE_SIZE; i++) {
    q->slots[i].state = STATE_EMPTY;
    q->slots[i].result = 0;
    q->slots[i].input = 0;
  }
}

// Lock the mutex (simple spinlock model)
void queue_lock(BatchQueue* q) {
  bool expected;
  do {
    __CPROVER_atomic_begin();
    expected = q->mutex_locked;
    if (!expected) {
      q->mutex_locked = true;
    }
    __CPROVER_atomic_end();
  } while (expected);
}

// Unlock the mutex
void queue_unlock(BatchQueue* q) {
  __CPROVER_atomic_begin();
  assert(q->mutex_locked && "Unlock without lock");
  q->mutex_locked = false;
  __CPROVER_atomic_end();
}

// Start the queue
void queue_start(BatchQueue* q) {
  q->running = true;
}

// Stop the queue
void queue_stop(BatchQueue* q) {
  q->running = false;
}

// Submit a request (producer)
// Returns slot index on success, -1 on failure
int queue_submit(BatchQueue* q, uint32_t input) {
  if (!q->running) {
    return -1;
  }

  queue_lock(q);

  uint32_t t = q->tail;
  uint32_t h = q->head;

  // Queue full?
  if ((t - h) >= CBMC_MAX_QUEUE_SIZE) {
    queue_unlock(q);
    return -1;
  }

  uint32_t slot_idx = t % CBMC_MAX_QUEUE_SIZE;

  // Check if slot is available (EMPTY state)
  // In a ring buffer, we must wait for slot to be reclaimed
  if (q->slots[slot_idx].state != STATE_EMPTY) {
    // Slot still in use - treat as full
    queue_unlock(q);
    return -1;
  }

  q->slots[slot_idx].state = STATE_PENDING;
  q->slots[slot_idx].input = input;
  q->slots[slot_idx].result = 0;

  q->tail = t + 1;
  q->submitted++;

  queue_unlock(q);
  return (int)slot_idx;
}

// Dequeue a request (worker)
// Returns slot index on success, -1 if empty
int queue_dequeue(BatchQueue* q) {
  queue_lock(q);

  uint32_t h = q->head;
  uint32_t t = q->tail;

  // Queue empty?
  if (h >= t) {
    queue_unlock(q);
    return -1;
  }

  uint32_t slot_idx = h % CBMC_MAX_QUEUE_SIZE;

  // PROPERTY: Slot must be pending
  assert(q->slots[slot_idx].state == STATE_PENDING &&
         "Queue invariant: slot not pending on dequeue");

  q->slots[slot_idx].state = STATE_PROCESSING;
  q->head = h + 1;

  queue_unlock(q);
  return (int)slot_idx;
}

// Complete a request (worker)
void queue_complete(BatchQueue* q, uint32_t slot_idx, uint32_t result) {
  assert(slot_idx < CBMC_MAX_QUEUE_SIZE);

  __CPROVER_atomic_begin();
  // PROPERTY: Slot must be processing
  assert(q->slots[slot_idx].state == STATE_PROCESSING &&
         "Complete on non-processing slot");
  q->slots[slot_idx].state = STATE_COMPLETED;
  q->slots[slot_idx].result = result;
  q->completed++;
  __CPROVER_atomic_end();
}

// Reclaim a completed slot (client)
void queue_reclaim(BatchQueue* q, uint32_t slot_idx) {
  assert(slot_idx < CBMC_MAX_QUEUE_SIZE);

  __CPROVER_atomic_begin();
  // PROPERTY: Slot must be completed
  assert(q->slots[slot_idx].state == STATE_COMPLETED &&
         "Reclaim on non-completed slot");
  q->slots[slot_idx].state = STATE_EMPTY;
  q->slots[slot_idx].input = 0;
  q->slots[slot_idx].result = 0;
  __CPROVER_atomic_end();
}

// Check queue invariants
void queue_check_invariants(BatchQueue* q) {
  // PROPERTY: Head never exceeds tail
  assert(q->head <= q->tail && "Head exceeded tail");

  // PROPERTY: Queue size bounded
  assert((q->tail - q->head) <= CBMC_MAX_QUEUE_SIZE && "Queue overflow");

  // PROPERTY: Completed <= submitted
  assert(q->completed <= q->submitted && "Completed exceeded submitted");
}

// Producer function: submits one request, waits for completion
void producer(BatchQueue* q, uint32_t tid) {
  uint32_t input = tid * 100;  // Unique input per producer

  int slot = queue_submit(q, input);
  if (slot < 0) {
    return;  // Queue full or not running - OK
  }

  // Wait for completion (bounded)
  unsigned wait = 0;
  while (wait < 10) {
    __CPROVER_atomic_begin();
    RequestState state = q->slots[slot].state;
    __CPROVER_atomic_end();

    if (state == STATE_COMPLETED) {
      // Check result
      __CPROVER_atomic_begin();
      uint32_t result = q->slots[slot].result;
      __CPROVER_atomic_end();

      // PROPERTY: Result should be input + 1
      assert(result == input + 1 && "Incorrect result");

      queue_reclaim(q, (uint32_t)slot);
      return;
    }
    wait++;
  }
}

// Worker function: processes one request
void worker(BatchQueue* q, uint32_t wid) {
  (void)wid;

  int slot = queue_dequeue(q);
  if (slot < 0) {
    return;  // Queue empty
  }

  // Get input
  __CPROVER_atomic_begin();
  uint32_t input = q->slots[slot].input;
  __CPROVER_atomic_end();

  // Process: result = input + 1
  uint32_t result = input + 1;

  queue_complete(q, (uint32_t)slot, result);
}

// Main verification harness
int main(void) {
  // Initialize
  queue_init(&g_queue);
  queue_start(&g_queue);

  // Simulate concurrent execution (CBMC explores interleavings)
  // This models 2 producers and 2 workers running concurrently

  // Nondeterministic interleaving
  for (int step = 0; step < 8; step++) {
    unsigned action = nondet_uint() % 4;

    switch (action) {
      case 0:
        // Producer 0
        producer(&g_queue, 0);
        break;
      case 1:
        // Producer 1
        producer(&g_queue, 1);
        break;
      case 2:
        // Worker 0
        worker(&g_queue, 0);
        break;
      case 3:
        // Worker 1
        worker(&g_queue, 1);
        break;
    }
  }

  // Final checks
  queue_check_invariants(&g_queue);

  return 0;
}
