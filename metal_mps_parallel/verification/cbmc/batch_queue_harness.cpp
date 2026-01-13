// CBMC Harness: Batch Queue Concurrent Verification
//
// This harness verifies the batch queue model under concurrent execution:
// - Multiple producer threads submit requests
// - Multiple worker threads process requests
// - Verifies no data races, lost requests, or deadlocks
//
// Run with:
//   cbmc batch_queue_harness.cpp --unwind 5 --bounds-check --pointer-check \
//        --slice-formula --drop-unused-functions

#include "batch_queue_model.h"
#include <pthread.h>
#include <cstdlib>

// Global queue instance for all threads
cbmc_model::BatchQueueModel g_queue;

// Thread synchronization
std::atomic<bool> g_producers_done{false};
std::atomic<size_t> g_active_producers{0};

// Nondeterministic choice for CBMC
extern "C" int nondet_int();
extern "C" unsigned nondet_uint();
extern "C" bool nondet_bool();

// Producer thread: submits requests
void* producer_thread(void* arg) {
  size_t tid = reinterpret_cast<size_t>(arg);

  // Each producer submits 1-2 requests (bounded for CBMC)
  unsigned num_requests = 1 + (nondet_uint() % 2);

  for (unsigned i = 0; i < num_requests; ++i) {
    if (!g_queue.running.load(std::memory_order_acquire)) {
      break;
    }

    size_t input = tid * 100 + i;  // Unique input per producer/request
    int slot = g_queue.submit(input);

    if (slot >= 0) {
      // Successfully submitted - wait for completion (bounded spin)
      unsigned wait_count = 0;
      while (wait_count < 10) {
        cbmc_model::RequestState state =
            g_queue.slots[slot].state.load(std::memory_order_acquire);

        if (state == cbmc_model::RequestState::COMPLETED) {
          // Verify result is correct transformation of input
          size_t result = g_queue.slots[slot].result.load(std::memory_order_acquire);

          // PROPERTY: Result should be input + 1 (our simple operation)
          assert(result == input + 1 && "Incorrect result");

          // Reclaim the slot
          g_queue.reclaim(static_cast<size_t>(slot));
          break;
        }

        if (state == cbmc_model::RequestState::ERROR) {
          // Error state - reclaim and continue
          g_queue.slots[slot].state.store(
              cbmc_model::RequestState::EMPTY, std::memory_order_release);
          break;
        }

        ++wait_count;
      }
    }
    // If slot < 0, queue was full - that's allowed
  }

  g_active_producers.fetch_sub(1, std::memory_order_release);
  return nullptr;
}

// Worker thread: processes requests
void* worker_thread(void* arg) {
  size_t wid = reinterpret_cast<size_t>(arg);
  (void)wid;  // Worker ID for debugging

  // Process requests until shutdown
  unsigned idle_count = 0;
  while (idle_count < 5) {  // Bounded for CBMC
    // Check for shutdown
    if (g_queue.shutdown_requested.load(std::memory_order_acquire)) {
      // Drain remaining requests
      int slot = g_queue.dequeue();
      if (slot < 0) {
        break;  // Queue empty and shutdown - exit
      }
      // Process this last request
      size_t input = g_queue.slots[slot].input_value;
      g_queue.complete(static_cast<size_t>(slot), input + 1);
      continue;
    }

    // Check if all producers done and queue empty
    if (g_producers_done.load(std::memory_order_acquire)) {
      int slot = g_queue.dequeue();
      if (slot < 0) {
        ++idle_count;
        continue;  // Queue empty but might have more coming
      }
      // Process request
      size_t input = g_queue.slots[slot].input_value;
      g_queue.complete(static_cast<size_t>(slot), input + 1);
      idle_count = 0;
      continue;
    }

    // Normal processing
    int slot = g_queue.dequeue();
    if (slot >= 0) {
      // Got a request - process it
      size_t input = g_queue.slots[slot].input_value;

      // Simple operation: result = input + 1
      g_queue.complete(static_cast<size_t>(slot), input + 1);
      idle_count = 0;
    } else {
      ++idle_count;
    }
  }

  return nullptr;
}

// Main verification harness
int main() {
  // Initialize queue
  g_queue.start();

  // Create threads
  pthread_t producers[CBMC_MAX_THREADS];
  pthread_t workers[CBMC_MAX_WORKERS];

  // Start workers first
  for (size_t w = 0; w < CBMC_MAX_WORKERS; ++w) {
    pthread_create(&workers[w], nullptr, worker_thread, reinterpret_cast<void*>(w));
  }

  // Start producers
  g_active_producers.store(CBMC_MAX_THREADS, std::memory_order_release);
  for (size_t p = 0; p < CBMC_MAX_THREADS; ++p) {
    pthread_create(&producers[p], nullptr, producer_thread, reinterpret_cast<void*>(p));
  }

  // Wait for producers to finish
  for (size_t p = 0; p < CBMC_MAX_THREADS; ++p) {
    pthread_join(producers[p], nullptr);
  }

  g_producers_done.store(true, std::memory_order_release);

  // Signal shutdown
  g_queue.stop();

  // Wait for workers
  for (size_t w = 0; w < CBMC_MAX_WORKERS; ++w) {
    pthread_join(workers[w], nullptr);
  }

  // Final invariant check
  g_queue.check_invariants();

  // PROPERTY: All submitted requests should complete
  size_t submitted = g_queue.submitted.load(std::memory_order_acquire);
  size_t completed = g_queue.completed.load(std::memory_order_acquire);

  // Note: Some requests may fail to submit if queue is full - that's OK
  // But all submitted must complete
  assert(completed == submitted && "Lost requests detected");

  return 0;
}
