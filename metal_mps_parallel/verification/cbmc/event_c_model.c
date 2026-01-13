// CBMC Model: MPS Event Pool with Callback Survival
// Based on: pytorch-mps-fork/aten/src/ATen/mps/MPSEvent.h
// Based on: specs/MPSEvent.tla
//
// Properties verified:
// 1. Event ID uniqueness (no two in-use events share same ID)
// 2. Callback survival (events with pending callbacks can't be released)
// 3. Signal counter monotonicity (per-event counter only increases)
// 4. Pool/InUse partition (events are in pool XOR in use)
// 5. No use-after-release (pooled events have consistent state)
// 6. Reference counting correctness
//
// Run with:
//   cbmc event_c_model.c --unwind 15 --bounds-check --pointer-check

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

// CBMC-friendly configuration
#ifndef CBMC_MAX_EVENTS
#define CBMC_MAX_EVENTS 4
#endif

#ifndef CBMC_MAX_THREADS
#define CBMC_MAX_THREADS 2
#endif

#ifndef CBMC_MAX_STREAMS
#define CBMC_MAX_STREAMS 2
#endif

// Event states (matching TLA+ spec)
typedef enum {
  EVENT_POOLED = 0,
  EVENT_ACQUIRED = 1,
  EVENT_RECORDED = 2,
  EVENT_SIGNALED = 3,
  EVENT_PENDING_CALLBACK = 4
} EventState;

// Event model (matching MPSEvent class structure)
typedef struct {
  volatile EventState state;
  volatile uint64_t event_id;        // Unique ID from event_counter
  volatile uint32_t signal_counter;  // MTLSharedEvent signal counter
  volatile uint32_t recording_stream;// Stream that recorded this event (0 = none)
  volatile uint32_t ref_count;       // Reference count for callback survival
  volatile bool has_pending_callback;// True if callback is pending
  volatile int event_mutex_holder;   // Thread holding event mutex (-1 = none)
} Event;

// Event pool model (matching MPSEventPool class structure)
typedef struct {
  Event events[CBMC_MAX_EVENTS];

  // Global event ID counter (atomic, monotonically increasing)
  volatile uint64_t event_counter;

  // Pool freelist
  volatile uint32_t freelist[CBMC_MAX_EVENTS];
  volatile uint32_t freelist_size;

  // In-use events (tracked by state)
  volatile uint32_t in_use_count;

  // Pool mutex holder (-1 = none)
  volatile int pool_mutex_holder;

  // Statistics
  volatile uint32_t total_acquired;
  volatile uint32_t total_released;
  volatile uint32_t total_callbacks_scheduled;
  volatile uint32_t total_callbacks_completed;
} EventPool;

// Global pool
EventPool g_pool;

// Track all IDs ever assigned (for uniqueness verification)
volatile uint64_t g_assigned_ids[CBMC_MAX_EVENTS * 4];
volatile uint32_t g_assigned_count;

// Previous signal counters (for monotonicity verification)
volatile uint32_t g_prev_signal_counters[CBMC_MAX_EVENTS];

// Nondeterministic functions (CBMC built-ins)
int nondet_int(void);
unsigned int nondet_uint(void);
_Bool nondet_bool(void);

// CBMC atomic primitives
void __CPROVER_atomic_begin(void);
void __CPROVER_atomic_end(void);
void __CPROVER_assume(_Bool);

// Initialize event pool
void pool_init(EventPool* p) {
  p->event_counter = 0;
  p->freelist_size = CBMC_MAX_EVENTS;
  p->in_use_count = 0;
  p->pool_mutex_holder = -1;
  p->total_acquired = 0;
  p->total_released = 0;
  p->total_callbacks_scheduled = 0;
  p->total_callbacks_completed = 0;

  for (uint32_t i = 0; i < CBMC_MAX_EVENTS; i++) {
    p->events[i].state = EVENT_POOLED;
    p->events[i].event_id = 0;
    p->events[i].signal_counter = 0;
    p->events[i].recording_stream = 0;
    p->events[i].ref_count = 0;
    p->events[i].has_pending_callback = false;
    p->events[i].event_mutex_holder = -1;
    p->freelist[i] = i;
    g_prev_signal_counters[i] = 0;
  }

  g_assigned_count = 0;
}

// Lock pool mutex
void pool_lock(EventPool* p, int tid) {
  __CPROVER_atomic_begin();
  __CPROVER_assume(p->pool_mutex_holder == -1);
  p->pool_mutex_holder = tid;
  __CPROVER_atomic_end();
}

// Unlock pool mutex
void pool_unlock(EventPool* p, int tid) {
  __CPROVER_atomic_begin();
  assert(p->pool_mutex_holder == tid && "Unlock without holding lock");
  p->pool_mutex_holder = -1;
  __CPROVER_atomic_end();
}

// Lock event mutex
void event_lock(EventPool* p, uint32_t event_idx, int tid) {
  assert(event_idx < CBMC_MAX_EVENTS);
  __CPROVER_atomic_begin();
  __CPROVER_assume(p->events[event_idx].event_mutex_holder == -1);
  p->events[event_idx].event_mutex_holder = tid;
  __CPROVER_atomic_end();
}

// Unlock event mutex
void event_unlock(EventPool* p, uint32_t event_idx, int tid) {
  assert(event_idx < CBMC_MAX_EVENTS);
  __CPROVER_atomic_begin();
  assert(p->events[event_idx].event_mutex_holder == tid &&
         "Unlock event without holding lock");
  p->events[event_idx].event_mutex_holder = -1;
  __CPROVER_atomic_end();
}

// Acquire event from pool
// Returns event index, or -1 on failure
int acquire_event(EventPool* p, int tid) {
  pool_lock(p, tid);

  if (p->freelist_size == 0) {
    pool_unlock(p, tid);
    return -1;  // Pool exhausted
  }

  // Pop from freelist
  p->freelist_size--;
  uint32_t event_idx = p->freelist[p->freelist_size];
  assert(event_idx < CBMC_MAX_EVENTS);

  // PROPERTY: Event from freelist must be pooled
  assert(p->events[event_idx].state == EVENT_POOLED &&
         "Freelist contains non-pooled event");
  assert(p->events[event_idx].ref_count == 0 &&
         "Pooled event has non-zero ref_count");

  // Assign new ID (atomic increment)
  __CPROVER_atomic_begin();
  uint64_t old_counter = p->event_counter;
  p->event_counter = old_counter + 1;
  uint64_t new_id = p->event_counter;

  // PROPERTY: Event counter only increases
  assert(p->event_counter > old_counter &&
         "Event counter did not increase");
  __CPROVER_atomic_end();

  // Track assigned ID for uniqueness check
  if (g_assigned_count < CBMC_MAX_EVENTS * 4) {
    g_assigned_ids[g_assigned_count] = new_id;
    g_assigned_count++;
  }

  // Initialize event
  p->events[event_idx].state = EVENT_ACQUIRED;
  p->events[event_idx].event_id = new_id;
  p->events[event_idx].signal_counter = 0;
  p->events[event_idx].recording_stream = 0;
  p->events[event_idx].ref_count = 1;  // in_use_events reference
  p->events[event_idx].has_pending_callback = false;

  p->in_use_count++;
  p->total_acquired++;

  pool_unlock(p, tid);
  return (int)event_idx;
}

// Release event back to pool
// CALLBACK SURVIVAL: Only releases when ref_count == 1 (no pending callbacks)
bool release_event(EventPool* p, uint32_t event_idx, int tid) {
  assert(event_idx < CBMC_MAX_EVENTS);

  pool_lock(p, tid);

  // PROPERTY: Cannot release pooled event
  if (p->events[event_idx].state == EVENT_POOLED) {
    pool_unlock(p, tid);
    return false;
  }

  // CALLBACK SURVIVAL: Check if there are pending callbacks
  // ref_count == 1 means only in_use_events holds reference
  // ref_count > 1 means callbacks are pending
  if (p->events[event_idx].ref_count > 1) {
    // Cannot release - callbacks still pending
    pool_unlock(p, tid);
    return false;
  }

  // PROPERTY: ref_count must be exactly 1 for release
  assert(p->events[event_idx].ref_count == 1 &&
         "Releasing event with invalid ref_count");

  // PROPERTY: No pending callbacks on release
  assert(!p->events[event_idx].has_pending_callback &&
         "Releasing event with pending callback (callback survival violated)");

  // Reset event
  p->events[event_idx].state = EVENT_POOLED;
  p->events[event_idx].event_id = 0;
  p->events[event_idx].signal_counter = 0;
  p->events[event_idx].recording_stream = 0;
  p->events[event_idx].ref_count = 0;

  // Push to freelist
  p->freelist[p->freelist_size] = event_idx;
  p->freelist_size++;

  p->in_use_count--;
  p->total_released++;

  pool_unlock(p, tid);
  return true;
}

// Record event on a stream
void record_event(EventPool* p, uint32_t event_idx, uint32_t stream, int tid) {
  assert(event_idx < CBMC_MAX_EVENTS);
  assert(stream > 0 && stream <= CBMC_MAX_STREAMS);

  event_lock(p, event_idx, tid);

  // PROPERTY: Can only record acquired or signaled events
  assert(p->events[event_idx].state == EVENT_ACQUIRED ||
         p->events[event_idx].state == EVENT_SIGNALED &&
         "Recording on invalid event");

  // Store previous signal counter for monotonicity check
  uint32_t prev_counter = p->events[event_idx].signal_counter;

  // Increment signal counter
  p->events[event_idx].signal_counter++;

  // PROPERTY: Signal counter monotonicity (per event)
  assert(p->events[event_idx].signal_counter > prev_counter &&
         "Signal counter did not increase");
  assert(p->events[event_idx].signal_counter >= g_prev_signal_counters[event_idx] &&
         "Signal counter decreased");
  g_prev_signal_counters[event_idx] = p->events[event_idx].signal_counter;

  p->events[event_idx].state = EVENT_RECORDED;
  p->events[event_idx].recording_stream = stream;

  event_unlock(p, event_idx, tid);
}

// Schedule callback notification (increases ref_count for survival)
void notify_event(EventPool* p, uint32_t event_idx, int tid) {
  assert(event_idx < CBMC_MAX_EVENTS);

  event_lock(p, event_idx, tid);

  // PROPERTY: Can only notify recorded events
  assert(p->events[event_idx].state == EVENT_RECORDED &&
         "Notifying non-recorded event");

  // CALLBACK SURVIVAL: Increment ref_count to keep event alive
  __CPROVER_atomic_begin();
  p->events[event_idx].ref_count++;
  p->events[event_idx].has_pending_callback = true;
  __CPROVER_atomic_end();

  p->events[event_idx].state = EVENT_PENDING_CALLBACK;
  p->total_callbacks_scheduled++;

  event_unlock(p, event_idx, tid);
}

// Signal event completion (GPU async)
void signal_event(EventPool* p, uint32_t event_idx) {
  assert(event_idx < CBMC_MAX_EVENTS);

  // PROPERTY: Can only signal recorded events
  // Note: This is an async operation, no mutex needed
  __CPROVER_atomic_begin();
  if (p->events[event_idx].state == EVENT_RECORDED) {
    p->events[event_idx].state = EVENT_SIGNALED;
  }
  __CPROVER_atomic_end();
}

// Callback completion (decrements ref_count)
bool callback_complete(EventPool* p, uint32_t event_idx) {
  assert(event_idx < CBMC_MAX_EVENTS);

  bool result = false;

  // Check pending status atomically, perform completion if pending
  __CPROVER_atomic_begin();
  bool has_pending = p->events[event_idx].has_pending_callback;
  if (has_pending) {
    // PROPERTY: Ref count must be > 1 if callback pending
    assert(p->events[event_idx].ref_count > 1 &&
           "Callback completing with ref_count <= 1");

    // Decrement ref_count
    p->events[event_idx].ref_count--;
    p->events[event_idx].has_pending_callback = false;

    // Transition to signaled state if was pending_callback
    if (p->events[event_idx].state == EVENT_PENDING_CALLBACK) {
      p->events[event_idx].state = EVENT_SIGNALED;
    }

    p->total_callbacks_completed++;
    result = true;
  }
  __CPROVER_atomic_end();

  return result;
}

// Wait for event to be signaled
bool wait_event(EventPool* p, uint32_t event_idx, int tid) {
  assert(event_idx < CBMC_MAX_EVENTS);

  event_lock(p, event_idx, tid);

  bool signaled = (p->events[event_idx].state == EVENT_SIGNALED);

  event_unlock(p, event_idx, tid);
  return signaled;
}

// Check pool invariants
void check_invariants(EventPool* p) {
  // PROPERTY: Event ID uniqueness among in-use events
  for (uint32_t i = 0; i < CBMC_MAX_EVENTS; i++) {
    if (p->events[i].state != EVENT_POOLED && p->events[i].event_id > 0) {
      for (uint32_t j = i + 1; j < CBMC_MAX_EVENTS; j++) {
        if (p->events[j].state != EVENT_POOLED && p->events[j].event_id > 0) {
          assert(p->events[i].event_id != p->events[j].event_id &&
                 "Two in-use events share same ID");
        }
      }
    }
  }

  // PROPERTY: Callback survival - events with pending callbacks have ref_count > 1
  for (uint32_t i = 0; i < CBMC_MAX_EVENTS; i++) {
    if (p->events[i].has_pending_callback) {
      assert(p->events[i].ref_count > 1 &&
             "Event with pending callback has ref_count <= 1 (callback survival violated)");
    }
  }

  // PROPERTY: Pool/InUse partition - events are in freelist XOR in-use
  uint32_t pooled_count = 0;
  uint32_t in_use_count = 0;

  for (uint32_t i = 0; i < CBMC_MAX_EVENTS; i++) {
    if (p->events[i].state == EVENT_POOLED) {
      pooled_count++;
    } else {
      in_use_count++;
    }
  }

  assert(pooled_count == p->freelist_size &&
         "Pooled count != freelist size");
  assert(in_use_count == p->in_use_count &&
         "In-use count mismatch");
  assert(pooled_count + in_use_count == CBMC_MAX_EVENTS &&
         "Total event count mismatch");

  // PROPERTY: Pooled events have consistent state
  for (uint32_t i = 0; i < CBMC_MAX_EVENTS; i++) {
    if (p->events[i].state == EVENT_POOLED) {
      assert(p->events[i].event_id == 0 &&
             "Pooled event has non-zero ID");
      assert(p->events[i].ref_count == 0 &&
             "Pooled event has non-zero ref_count");
      assert(p->events[i].recording_stream == 0 &&
             "Pooled event has recording stream");
      assert(!p->events[i].has_pending_callback &&
             "Pooled event has pending callback");
    }
  }

  // PROPERTY: Acquired events have ref_count >= 1
  for (uint32_t i = 0; i < CBMC_MAX_EVENTS; i++) {
    if (p->events[i].state != EVENT_POOLED) {
      assert(p->events[i].ref_count >= 1 &&
             "In-use event has ref_count < 1");
    }
  }

  // PROPERTY: Freelist contains only pooled events
  for (uint32_t i = 0; i < p->freelist_size; i++) {
    uint32_t idx = p->freelist[i];
    assert(idx < CBMC_MAX_EVENTS &&
           "Freelist contains invalid index");
    assert(p->events[idx].state == EVENT_POOLED &&
           "Freelist contains non-pooled event");
  }

  // PROPERTY: No ID reuse (among all ever-assigned IDs)
  for (uint32_t i = 0; i < g_assigned_count; i++) {
    for (uint32_t j = i + 1; j < g_assigned_count; j++) {
      assert(g_assigned_ids[i] != g_assigned_ids[j] &&
             "Event ID reused");
    }
  }
}

// Thread work: acquire, record, notify, signal, release
void thread_work(EventPool* p, int tid, uint32_t stream) {
  // Acquire an event
  int event_idx = acquire_event(p, tid);
  if (event_idx < 0) {
    return;  // Pool exhausted - OK
  }

  // PROPERTY: Acquired event has unique ID
  assert(p->events[event_idx].event_id > 0 &&
         "Acquired event has no ID");

  // Record on stream
  record_event(p, (uint32_t)event_idx, stream, tid);

  // Optionally schedule callback
  if (nondet_bool()) {
    notify_event(p, (uint32_t)event_idx, tid);

    // PROPERTY: After notify, ref_count must be > 1
    assert(p->events[event_idx].ref_count > 1 &&
           "ref_count <= 1 after notify");

    // Optionally complete callback (simulating GPU completion)
    if (nondet_bool()) {
      callback_complete(p, (uint32_t)event_idx);
    }
  } else {
    // No callback - directly signal
    signal_event(p, (uint32_t)event_idx);
  }

  // Try to release
  // If callbacks pending, this should fail
  bool released = release_event(p, (uint32_t)event_idx, tid);

  if (!released) {
    // Still has pending callbacks - complete them first
    while (p->events[event_idx].has_pending_callback) {
      callback_complete(p, (uint32_t)event_idx);
    }

    // Now release should succeed
    released = release_event(p, (uint32_t)event_idx, tid);
    assert(released && "Failed to release after completing callbacks");
  }
}

// GPU signal work (async)
void gpu_signal_work(EventPool* p) {
  for (uint32_t i = 0; i < CBMC_MAX_EVENTS; i++) {
    if (p->events[i].state == EVENT_RECORDED) {
      signal_event(p, i);
      break;  // Signal one at a time
    }
  }
}

// Callback completion work (async)
void callback_work(EventPool* p) {
  for (uint32_t i = 0; i < CBMC_MAX_EVENTS; i++) {
    if (p->events[i].has_pending_callback) {
      callback_complete(p, i);
      break;  // Complete one at a time
    }
  }
}

// Main verification harness
int main(void) {
  // Initialize
  pool_init(&g_pool);

  // Simulate concurrent execution (8 steps for reasonable verification time)
  for (int step = 0; step < 8; step++) {
    unsigned action = nondet_uint() % 6;

    switch (action) {
      case 0:
        // Thread 0, stream 1
        thread_work(&g_pool, 0, 1);
        break;
      case 1:
        // Thread 1, stream 2
        thread_work(&g_pool, 1, (2 <= CBMC_MAX_STREAMS) ? 2 : 1);
        break;
      case 2:
        // GPU signal
        gpu_signal_work(&g_pool);
        break;
      case 3:
        // Callback completion
        callback_work(&g_pool);
        break;
      case 4:
        // Thread 0, acquire and hold
        {
          int e = acquire_event(&g_pool, 0);
          if (e >= 0) {
            // Record and immediately try to release
            record_event(&g_pool, (uint32_t)e, 1, 0);
            signal_event(&g_pool, (uint32_t)e);
            release_event(&g_pool, (uint32_t)e, 0);
          }
        }
        break;
      case 5:
        // Callback survival test: acquire, notify, try release (should fail)
        {
          int e = acquire_event(&g_pool, 1);
          if (e >= 0) {
            record_event(&g_pool, (uint32_t)e, 1, 1);
            notify_event(&g_pool, (uint32_t)e, 1);

            // PROPERTY: Release should fail while callback pending
            bool released = release_event(&g_pool, (uint32_t)e, 1);
            assert(!released && "Released event with pending callback!");

            // Complete callback
            callback_complete(&g_pool, (uint32_t)e);

            // Now release should succeed
            released = release_event(&g_pool, (uint32_t)e, 1);
            assert(released && "Failed to release after callback completion");
          }
        }
        break;
    }
  }

  // Final invariant check
  check_invariants(&g_pool);

  // PROPERTY: Event counter equals total acquisitions
  assert(g_pool.event_counter == g_pool.total_acquired &&
         "Event counter != total acquired");

  // PROPERTY: Released <= acquired
  assert(g_pool.total_released <= g_pool.total_acquired &&
         "Released more than acquired");

  // PROPERTY: Completed callbacks <= scheduled callbacks
  assert(g_pool.total_callbacks_completed <= g_pool.total_callbacks_scheduled &&
         "Completed more callbacks than scheduled");

  return 0;
}
