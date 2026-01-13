// aba_detection_harness.c - CBMC harness for verifying ABA detection
//
// Tests the double-check locking pattern with use_count that prevents
// the ABA problem in getSharedBufferPtr() (issue 32.267).
//
// The ABA problem:
// 1. Thread A reads in_use=true, use_count=5, releases lock
// 2. Thread B frees block, use_count stays 5
// 3. Thread C allocates same block, in_use=true, use_count=6
// 4. Thread A re-acquires lock, sees in_use=true
// 5. WITHOUT use_count check: Thread A uses block (which is different allocation!)
// 6. WITH use_count check: Thread A sees use_count=6 != 5, aborts (correct)
//
// Run with:
//   cbmc aba_detection_harness.c -I ../models -I ../stubs --unwind 5

#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>

#include "../models/buffer_block.h"
#include "../stubs/metal_stubs.h"

// CBMC Primitives
extern int nondet_int(void);
extern unsigned int nondet_uint(void);
extern bool nondet_bool(void);
extern void __CPROVER_assume(bool);
extern void __CPROVER_assert(bool, const char*);

// ============================================================================
// ABA Scenario Model
// ============================================================================

typedef enum {
    THREAD_IDLE,
    THREAD_FIRST_CHECK_PASSED,
    THREAD_WAITING_FOR_LOCK,
    THREAD_HAS_LOCK,
    THREAD_SECOND_CHECK_PASSED,
    THREAD_DONE
} ThreadState;

typedef struct {
    BufferBlock block;
    bool mutex_held;
    int mutex_holder; // -1 = none, 0-N = thread id

    // Thread A state (doing getSharedBufferPtr)
    ThreadState thread_a_state;
    bool thread_a_first_in_use;
    uint32_t thread_a_captured_use_count;

    // Generation tracker: how many times has block been recycled?
    int allocation_generation;
} ABATestState;

static void ABATestState_init(ABATestState* state) {
    BufferBlock_init(&state->block, 1024, (void*)0x1000);
    state->mutex_held = false;
    state->mutex_holder = -1;
    state->thread_a_state = THREAD_IDLE;
    state->thread_a_first_in_use = false;
    state->thread_a_captured_use_count = 0;
    state->allocation_generation = 0;

    // Start with block allocated
    BufferBlock_acquire(&state->block);
    state->allocation_generation = 1;
}

// Thread A: First check (outside lock)
static void thread_a_first_check(ABATestState* state) {
    __CPROVER_assert(state->thread_a_state == THREAD_IDLE, "Thread A must be idle");

    // Atomic reads (can happen without lock)
    state->thread_a_first_in_use = BufferBlock_isInUse(&state->block);
    state->thread_a_captured_use_count = BufferBlock_getUseCount(&state->block);

    if (state->thread_a_first_in_use) {
        state->thread_a_state = THREAD_FIRST_CHECK_PASSED;
    } else {
        state->thread_a_state = THREAD_DONE; // Early exit - block not in use
    }
}

// Thread A: Acquire lock
static bool thread_a_acquire_lock(ABATestState* state) {
    __CPROVER_assert(state->thread_a_state == THREAD_FIRST_CHECK_PASSED,
                    "Thread A must have passed first check");

    if (!state->mutex_held) {
        state->mutex_held = true;
        state->mutex_holder = 0; // Thread A = 0
        state->thread_a_state = THREAD_HAS_LOCK;
        return true;
    }
    state->thread_a_state = THREAD_WAITING_FOR_LOCK;
    return false;
}

// Thread A: Second check with ABA detection
static bool thread_a_second_check_aba(ABATestState* state, bool* aba_detected) {
    __CPROVER_assert(state->thread_a_state == THREAD_HAS_LOCK,
                    "Thread A must hold lock");
    __CPROVER_assert(state->mutex_held && state->mutex_holder == 0,
                    "Thread A must actually hold mutex");

    bool current_in_use = BufferBlock_isInUse(&state->block);
    uint32_t current_use_count = BufferBlock_getUseCount(&state->block);

    // ABA Detection: use_count changed means block was recycled
    if (current_use_count != state->thread_a_captured_use_count) {
        *aba_detected = true;
        state->thread_a_state = THREAD_DONE;

        // Release lock
        state->mutex_held = false;
        state->mutex_holder = -1;
        return false; // Abort - ABA detected
    }

    *aba_detected = false;

    if (!current_in_use) {
        // Block freed between first check and lock acquisition
        state->thread_a_state = THREAD_DONE;
        state->mutex_held = false;
        state->mutex_holder = -1;
        return false;
    }

    state->thread_a_state = THREAD_SECOND_CHECK_PASSED;
    return true;
}

// Thread A: Complete operation and release lock
static void thread_a_complete(ABATestState* state) {
    __CPROVER_assert(state->thread_a_state == THREAD_SECOND_CHECK_PASSED,
                    "Thread A must have passed second check");
    __CPROVER_assert(state->mutex_held && state->mutex_holder == 0,
                    "Thread A must hold mutex");

    // Here Thread A would use the block - verify it's the same allocation
    // In buggy code (without ABA detection), this could be a different allocation

    state->thread_a_state = THREAD_DONE;
    state->mutex_held = false;
    state->mutex_holder = -1;
}

// Thread B: Free the block (simulating concurrent free)
static bool thread_b_free(ABATestState* state) {
    if (state->mutex_held) {
        return false; // Can't acquire lock
    }

    state->mutex_held = true;
    state->mutex_holder = 1; // Thread B = 1

    if (BufferBlock_isInUse(&state->block)) {
        BufferBlock_release(&state->block);
    }

    state->mutex_held = false;
    state->mutex_holder = -1;
    return true;
}

// Thread C: Reallocate the block (simulating ABA completion)
static bool thread_c_realloc(ABATestState* state) {
    if (state->mutex_held) {
        return false; // Can't acquire lock
    }

    state->mutex_held = true;
    state->mutex_holder = 2; // Thread C = 2

    if (!BufferBlock_isInUse(&state->block)) {
        BufferBlock_acquire(&state->block);
        state->allocation_generation++;
    }

    state->mutex_held = false;
    state->mutex_holder = -1;
    return true;
}

// ============================================================================
// Main Harness
// ============================================================================

int main(void) {
    ABATestState state;
    ABATestState_init(&state);

    // Record initial state
    int initial_generation = state.allocation_generation;

    // ========== PHASE 1: Thread A does first check ==========
    thread_a_first_check(&state);

    if (state.thread_a_state == THREAD_DONE) {
        // Block wasn't in use, nothing to test
        return 0;
    }

    __CPROVER_assert(state.thread_a_state == THREAD_FIRST_CHECK_PASSED,
                    "Thread A should have passed first check");

    // ========== PHASE 2: Non-deterministic interleaving ==========
    // Between first check and lock acquisition, other threads may act

    int interleave = nondet_int();
    __CPROVER_assume(interleave >= 0 && interleave <= 3);

    bool aba_scenario = false;

    if (interleave == 1) {
        // Thread B frees the block
        thread_b_free(&state);
    } else if (interleave == 2) {
        // Thread B frees, Thread C reallocates (ABA!)
        thread_b_free(&state);
        thread_c_realloc(&state);
        aba_scenario = (state.allocation_generation > initial_generation);
    } else if (interleave == 3) {
        // Multiple free/realloc cycles
        thread_b_free(&state);
        thread_c_realloc(&state);
        thread_b_free(&state);
        thread_c_realloc(&state);
        aba_scenario = (state.allocation_generation > initial_generation);
    }
    // interleave == 0: no interference

    // ========== PHASE 3: Thread A acquires lock ==========
    bool got_lock = thread_a_acquire_lock(&state);
    __CPROVER_assume(got_lock); // For this test, assume we eventually get lock

    // ========== PHASE 4: Thread A does second check with ABA detection ==========
    bool aba_detected = false;
    bool passed = thread_a_second_check_aba(&state, &aba_detected);

    // ========== VERIFICATION ==========

    if (aba_scenario) {
        // If ABA happened, we MUST detect it
        // The key property: if allocation_generation changed, use_count changed too
        // Therefore aba_detected should be true
        __CPROVER_assert(aba_detected || !passed,
            "ABA scenario must be detected or operation must fail");
    }

    if (passed) {
        // If we passed the second check, verify we're operating on same allocation
        __CPROVER_assert(state.allocation_generation == initial_generation,
            "Passed check must mean same allocation (no ABA)");

        thread_a_complete(&state);
    }

    // Final state check
    __CPROVER_assert(!state.mutex_held,
        "Mutex must be released at end");

    return 0;
}
