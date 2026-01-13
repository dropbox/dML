// Standalone repro for nested dispatch_sync on the same serial queue (deadlock),
// plus a queue-specific inline workaround used by the MPS patch.

#include <dispatch/dispatch.h>

#include <exception>
#include <optional>
#include <cstdio>

static char kReproQueueSpecificKey;

static void dispatch_sync_with_rethrow_unsafe(dispatch_queue_t queue, void (^block)()) {
  __block std::optional<std::exception_ptr> block_exception;
  dispatch_sync(queue, ^() {
    try {
      block();
    } catch (...) {
      block_exception = std::current_exception();
    }
  });
  if (block_exception) {
    std::rethrow_exception(*block_exception);
  }
}

static void dispatch_sync_with_rethrow_safe(dispatch_queue_t queue, const void* key, void (^block)()) {
  void* queue_specific_value = dispatch_queue_get_specific(queue, key);
  if (queue_specific_value != nullptr && queue_specific_value == dispatch_get_specific(key)) {
    block();
    return;
  }

  __block std::optional<std::exception_ptr> block_exception;
  dispatch_sync(queue, ^() {
    try {
      block();
    } catch (...) {
      block_exception = std::current_exception();
    }
  });
  if (block_exception) {
    std::rethrow_exception(*block_exception);
  }
}

static bool expect_unsafe_deadlock_via_timeout() {
  dispatch_queue_t queue = dispatch_queue_create("repro.dispatch_sync_with_rethrow.unsafe", DISPATCH_QUEUE_SERIAL);
  dispatch_queue_set_specific(queue, &kReproQueueSpecificKey, reinterpret_cast<void*>(0x1), nullptr);

  dispatch_semaphore_t started = dispatch_semaphore_create(0);
  dispatch_semaphore_t finished = dispatch_semaphore_create(0);

  dispatch_async(queue, ^() {
    dispatch_semaphore_signal(started);
    dispatch_sync_with_rethrow_unsafe(queue, ^() {});
    dispatch_semaphore_signal(finished);
  });

  if (dispatch_semaphore_wait(started, dispatch_time(DISPATCH_TIME_NOW, 500 * NSEC_PER_MSEC)) != 0) {
    std::fprintf(stderr, "ERROR: unsafe test did not start\n");
    return false;
  }

  return dispatch_semaphore_wait(finished, dispatch_time(DISPATCH_TIME_NOW, 200 * NSEC_PER_MSEC)) != 0;
}

static bool expect_safe_reentrant_completes() {
  dispatch_queue_t queue = dispatch_queue_create("repro.dispatch_sync_with_rethrow.safe", DISPATCH_QUEUE_SERIAL);
  dispatch_queue_set_specific(queue, &kReproQueueSpecificKey, reinterpret_cast<void*>(0x2), nullptr);

  dispatch_semaphore_t finished = dispatch_semaphore_create(0);

  dispatch_async(queue, ^() {
    dispatch_sync_with_rethrow_safe(queue, &kReproQueueSpecificKey, ^() {});
    dispatch_semaphore_signal(finished);
  });

  return dispatch_semaphore_wait(finished, dispatch_time(DISPATCH_TIME_NOW, 500 * NSEC_PER_MSEC)) == 0;
}

int main() {
  const bool unsafe_deadlocks = expect_unsafe_deadlock_via_timeout();
  const bool safe_completes = expect_safe_reentrant_completes();

  if (!unsafe_deadlocks || !safe_completes) {
    std::fprintf(
        stderr,
        "FAIL: unsafe_deadlocks=%d safe_completes=%d\n",
        unsafe_deadlocks ? 1 : 0,
        safe_completes ? 1 : 0);
    return 1;
  }

  std::puts("PASS");
  return 0;
}

