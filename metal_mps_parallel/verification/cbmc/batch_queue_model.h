// CBMC Model: Simplified MPSBatchQueue for bounded model checking
// This model captures the concurrent behavior without ObjC++ dependencies.
//
// Properties verified:
// 1. No data races on queue state
// 2. No lost requests (all submitted requests eventually complete)
// 3. No deadlock between producers and consumers
// 4. No double-completion of requests

#pragma once

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstddef>

// CBMC-friendly configuration (small bounds for tractable verification)
#ifndef CBMC_MAX_QUEUE_SIZE
#define CBMC_MAX_QUEUE_SIZE 4
#endif

#ifndef CBMC_MAX_THREADS
#define CBMC_MAX_THREADS 3
#endif

#ifndef CBMC_MAX_WORKERS
#define CBMC_MAX_WORKERS 2
#endif

namespace cbmc_model {

// Simplified request state
enum class RequestState : uint8_t {
  EMPTY = 0,
  PENDING = 1,
  PROCESSING = 2,
  COMPLETED = 3,
  ERROR = 4
};

// Simplified request (no tensors, just state)
struct Request {
  std::atomic<RequestState> state{RequestState::EMPTY};
  std::atomic<size_t> result{0};  // 0 = not set, non-zero = result value
  size_t input_value{0};           // Simulated input

  void reset() {
    state.store(RequestState::EMPTY, std::memory_order_release);
    result.store(0, std::memory_order_release);
    input_value = 0;
  }
};

// Simplified queue model
class BatchQueueModel {
public:
  // Queue slots (fixed array for CBMC)
  Request slots[CBMC_MAX_QUEUE_SIZE];

  // Queue head/tail indices
  std::atomic<size_t> head{0};  // Next slot to dequeue
  std::atomic<size_t> tail{0};  // Next slot to enqueue

  // State
  std::atomic<bool> running{false};
  std::atomic<bool> shutdown_requested{false};

  // Statistics
  std::atomic<size_t> submitted{0};
  std::atomic<size_t> completed{0};

  // Mutex simulation (CBMC models pthread_mutex)
  std::atomic<bool> mutex_locked{false};

  void lock() {
    bool expected = false;
    // Spin until we acquire (CBMC will explore all interleavings)
    while (!mutex_locked.compare_exchange_weak(
        expected, true, std::memory_order_acquire, std::memory_order_relaxed)) {
      expected = false;
    }
  }

  void unlock() {
    mutex_locked.store(false, std::memory_order_release);
  }

  // Submit a request (producer side)
  // Returns slot index if successful, -1 if queue full or not running
  int submit(size_t input) {
    if (!running.load(std::memory_order_acquire)) {
      return -1;
    }

    lock();

    size_t t = tail.load(std::memory_order_relaxed);
    size_t h = head.load(std::memory_order_relaxed);

    // Queue full check
    if ((t - h) >= CBMC_MAX_QUEUE_SIZE) {
      unlock();
      return -1;
    }

    size_t slot_idx = t % CBMC_MAX_QUEUE_SIZE;

    // Verify slot is empty (invariant)
    RequestState expected_state = RequestState::EMPTY;
    bool slot_was_empty = slots[slot_idx].state.compare_exchange_strong(
        expected_state, RequestState::PENDING,
        std::memory_order_acq_rel, std::memory_order_acquire);

    // PROPERTY: Slot must be empty when we enqueue
    assert(slot_was_empty && "Queue invariant violated: slot not empty");

    slots[slot_idx].input_value = input;
    slots[slot_idx].result.store(0, std::memory_order_release);

    tail.store(t + 1, std::memory_order_release);
    submitted.fetch_add(1, std::memory_order_relaxed);

    unlock();
    return static_cast<int>(slot_idx);
  }

  // Dequeue a request (consumer/worker side)
  // Returns slot index if successful, -1 if queue empty
  int dequeue() {
    lock();

    size_t h = head.load(std::memory_order_relaxed);
    size_t t = tail.load(std::memory_order_relaxed);

    // Queue empty check
    if (h >= t) {
      unlock();
      return -1;
    }

    size_t slot_idx = h % CBMC_MAX_QUEUE_SIZE;

    // Verify slot is pending
    RequestState expected_state = RequestState::PENDING;
    bool transitioned = slots[slot_idx].state.compare_exchange_strong(
        expected_state, RequestState::PROCESSING,
        std::memory_order_acq_rel, std::memory_order_acquire);

    // PROPERTY: Dequeued slot must be pending
    assert(transitioned && "Queue invariant violated: slot not pending");

    head.store(h + 1, std::memory_order_release);

    unlock();
    return static_cast<int>(slot_idx);
  }

  // Complete a request (worker side)
  void complete(size_t slot_idx, size_t result_value) {
    assert(slot_idx < CBMC_MAX_QUEUE_SIZE);

    // Verify slot is in processing state
    RequestState expected_state = RequestState::PROCESSING;
    bool transitioned = slots[slot_idx].state.compare_exchange_strong(
        expected_state, RequestState::COMPLETED,
        std::memory_order_acq_rel, std::memory_order_acquire);

    // PROPERTY: No double-completion
    assert(transitioned && "Double completion detected");

    slots[slot_idx].result.store(result_value, std::memory_order_release);
    completed.fetch_add(1, std::memory_order_relaxed);
  }

  // Reclaim a completed slot (client side after reading result)
  void reclaim(size_t slot_idx) {
    assert(slot_idx < CBMC_MAX_QUEUE_SIZE);

    RequestState expected_state = RequestState::COMPLETED;
    bool transitioned = slots[slot_idx].state.compare_exchange_strong(
        expected_state, RequestState::EMPTY,
        std::memory_order_acq_rel, std::memory_order_acquire);

    // PROPERTY: Only completed slots can be reclaimed
    assert(transitioned && "Reclaim on non-completed slot");

    slots[slot_idx].reset();
  }

  // Start the queue
  void start() {
    running.store(true, std::memory_order_release);
    shutdown_requested.store(false, std::memory_order_release);
  }

  // Request shutdown
  void stop() {
    shutdown_requested.store(true, std::memory_order_release);
    running.store(false, std::memory_order_release);
  }

  // Check invariants (for CBMC verification)
  void check_invariants() {
    size_t h = head.load(std::memory_order_acquire);
    size_t t = tail.load(std::memory_order_acquire);

    // PROPERTY: Head never exceeds tail
    assert(h <= t && "Head exceeded tail");

    // PROPERTY: Queue size never exceeds max
    assert((t - h) <= CBMC_MAX_QUEUE_SIZE && "Queue overflow");

    // PROPERTY: Completed count never exceeds submitted
    size_t sub = submitted.load(std::memory_order_acquire);
    size_t comp = completed.load(std::memory_order_acquire);
    assert(comp <= sub && "Completed exceeded submitted");
  }
};

} // namespace cbmc_model
