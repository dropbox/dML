// fork_safety_harness.c - CBMC harness for verifying fork handler correctness
//
// The MPS fork handler ensures GPU resources are properly handled across fork():
// - Child process must reinitialize (GPU contexts don't survive fork)
// - Parent process state should be preserved
// - Pending GPU work must be flushed before fork
// - pthread_atfork handlers must execute in correct order
//
// Properties verified:
// 1. Fork preparation flushes all pending work
// 2. Child process correctly resets to uninitialized state
// 3. Parent process preserves all state after fork
// 4. No resource leaks in child (resources marked invalid, not duplicated)
// 5. Fork handlers execute in correct order (prepare -> parent/child)
// 6. Multiple fork() calls handled correctly
//
// Run with:
//   cbmc fork_safety_harness.c -I ../models -I ../stubs --unwind 10 --pointer-check --bounds-check

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
// Configuration (reduced for CBMC verification)
// ============================================================================

#define MAX_STREAMS 4
#define MAX_PENDING_OPS 8
#define MAX_FORK_DEPTH 2  // Nested forks

// ============================================================================
// Process State Model
// ============================================================================

typedef enum ProcessRole {
    ROLE_PARENT = 0,    // Original parent process
    ROLE_CHILD = 1      // Child after fork()
} ProcessRole;

typedef enum InitState {
    INIT_UNINITIALIZED = 0,
    INIT_IN_PROGRESS = 1,
    INIT_COMPLETE = 2,
    INIT_FAILED = 3
} InitState;

// ============================================================================
// GPU Context Model (doesn't survive fork)
// ============================================================================

typedef struct GPUContext {
    bool is_valid;              // Context is usable
    uint64_t context_id;        // Unique ID (invalid after fork in child)
    int device_id;              // Which GPU
} GPUContext;

// ============================================================================
// Stream with Pending Work
// ============================================================================

typedef struct Stream {
    bool is_active;
    int pending_ops;            // Number of uncommitted operations
    bool needs_sync;            // Has work that needs GPU sync
    GPUContext* context;        // Associated GPU context
} Stream;

// ============================================================================
// MPS State Model
// ============================================================================

typedef struct MPSState {
    InitState init_state;
    ProcessRole role;
    int fork_count;             // How many times this process has forked

    GPUContext gpu_context;
    Stream streams[MAX_STREAMS];
    int num_active_streams;

    // Fork handler state
    bool atfork_registered;
    bool in_fork_prepare;       // Currently in prepare phase
    bool fork_prepare_done;     // Prepare phase completed
} MPSState;

// Global state
static MPSState g_mps_state;
static uint64_t g_next_context_id = 1;

// ============================================================================
// Initialization
// ============================================================================

static void MPSState_init(MPSState* state) {
    __CPROVER_assert(state != NULL, "State must not be null");

    state->init_state = INIT_UNINITIALIZED;
    state->role = ROLE_PARENT;
    state->fork_count = 0;

    state->gpu_context.is_valid = false;
    state->gpu_context.context_id = 0;
    state->gpu_context.device_id = -1;

    for (int i = 0; i < MAX_STREAMS; i++) {
        state->streams[i].is_active = false;
        state->streams[i].pending_ops = 0;
        state->streams[i].needs_sync = false;
        state->streams[i].context = NULL;
    }
    state->num_active_streams = 0;

    state->atfork_registered = false;
    state->in_fork_prepare = false;
    state->fork_prepare_done = false;
}

// Initialize GPU context (lazy init pattern)
static bool MPSState_initGPU(MPSState* state) {
    __CPROVER_assert(state != NULL, "State must not be null");

    if (state->init_state == INIT_COMPLETE) {
        return true;  // Already initialized
    }

    if (state->init_state == INIT_FAILED) {
        return false;  // Previously failed
    }

    state->init_state = INIT_IN_PROGRESS;

    // Simulate GPU context creation
    state->gpu_context.context_id = g_next_context_id++;
    state->gpu_context.device_id = 0;
    state->gpu_context.is_valid = true;

    state->init_state = INIT_COMPLETE;
    return true;
}

// ============================================================================
// Stream Operations
// ============================================================================

static int MPSState_acquireStream(MPSState* state) {
    __CPROVER_assert(state != NULL, "State must not be null");

    // Must be initialized
    if (state->init_state != INIT_COMPLETE) {
        return -1;
    }

    // GPU context must be valid
    if (!state->gpu_context.is_valid) {
        return -1;
    }

    // Find free stream
    for (int i = 0; i < MAX_STREAMS; i++) {
        if (!state->streams[i].is_active) {
            state->streams[i].is_active = true;
            state->streams[i].pending_ops = 0;
            state->streams[i].needs_sync = false;
            state->streams[i].context = &state->gpu_context;
            state->num_active_streams++;
            return i;
        }
    }
    return -1;  // No free streams
}

static void MPSState_submitWork(MPSState* state, int stream_idx) {
    __CPROVER_assert(state != NULL, "State must not be null");
    __CPROVER_assert(stream_idx >= 0 && stream_idx < MAX_STREAMS,
        "Stream index must be valid");
    __CPROVER_assert(state->streams[stream_idx].is_active,
        "Stream must be active");

    state->streams[stream_idx].pending_ops++;
    state->streams[stream_idx].needs_sync = true;
}

static void MPSState_syncStream(MPSState* state, int stream_idx) {
    __CPROVER_assert(state != NULL, "State must not be null");
    __CPROVER_assert(stream_idx >= 0 && stream_idx < MAX_STREAMS,
        "Stream index must be valid");

    if (state->streams[stream_idx].is_active) {
        state->streams[stream_idx].pending_ops = 0;
        state->streams[stream_idx].needs_sync = false;
    }
}

// ============================================================================
// Fork Handlers (pthread_atfork callbacks)
// ============================================================================

// Called BEFORE fork() in parent
static void fork_prepare(MPSState* state) {
    __CPROVER_assert(state != NULL, "State must not be null");
    __CPROVER_assert(!state->in_fork_prepare, "Cannot nest fork_prepare");

    state->in_fork_prepare = true;

    // Flush all pending GPU work (critical for fork safety)
    for (int i = 0; i < MAX_STREAMS; i++) {
        if (state->streams[i].is_active && state->streams[i].needs_sync) {
            MPSState_syncStream(state, i);
        }
    }

    state->fork_prepare_done = true;
}

// Called AFTER fork() in parent
static void fork_parent(MPSState* state) {
    __CPROVER_assert(state != NULL, "State must not be null");
    __CPROVER_assert(state->fork_prepare_done,
        "fork_parent requires fork_prepare to have run");

    // Parent continues normally
    state->in_fork_prepare = false;
    state->fork_prepare_done = false;
    state->fork_count++;

    // Parent's GPU context remains valid
    __CPROVER_assert(state->gpu_context.is_valid,
        "Parent GPU context should remain valid");
}

// Called AFTER fork() in child
static void fork_child(MPSState* state) {
    __CPROVER_assert(state != NULL, "State must not be null");
    __CPROVER_assert(state->fork_prepare_done,
        "fork_child requires fork_prepare to have run");

    // Child must reinitialize - GPU context doesn't survive fork
    state->role = ROLE_CHILD;
    state->in_fork_prepare = false;
    state->fork_prepare_done = false;

    // Mark GPU context as INVALID (doesn't survive fork)
    state->gpu_context.is_valid = false;
    state->gpu_context.context_id = 0;  // Invalid

    // Mark all streams as invalid
    for (int i = 0; i < MAX_STREAMS; i++) {
        state->streams[i].is_active = false;
        state->streams[i].pending_ops = 0;
        state->streams[i].needs_sync = false;
        state->streams[i].context = NULL;
    }
    state->num_active_streams = 0;

    // Child must reinitialize before using MPS
    state->init_state = INIT_UNINITIALIZED;
}

// Simulate fork() - returns 0 in child, >0 in parent, -1 on error
static int simulate_fork(MPSState* state, bool* is_child_out) {
    __CPROVER_assert(state != NULL, "State must not be null");

    // Prepare phase
    fork_prepare(state);

    // Verify all pending work was flushed
    for (int i = 0; i < MAX_STREAMS; i++) {
        if (state->streams[i].is_active) {
            __CPROVER_assert(!state->streams[i].needs_sync,
                "All pending work must be flushed before fork");
            __CPROVER_assert(state->streams[i].pending_ops == 0,
                "All pending ops must be zero before fork");
        }
    }

    // Non-deterministic: are we the child or parent?
    bool is_child = nondet_bool();
    *is_child_out = is_child;

    if (is_child) {
        fork_child(state);
        return 0;  // Child gets 0
    } else {
        fork_parent(state);
        return 1234;  // Parent gets child PID (non-zero)
    }
}

// ============================================================================
// Invariant Checkers
// ============================================================================

// Check post-fork invariants for parent
static void check_parent_invariants(MPSState* state) {
    __CPROVER_assert(state->role == ROLE_PARENT,
        "Parent role should be preserved");
    __CPROVER_assert(state->gpu_context.is_valid,
        "Parent GPU context should remain valid");
    __CPROVER_assert(state->init_state == INIT_COMPLETE,
        "Parent init state should remain complete");
}

// Check post-fork invariants for child
static void check_child_invariants(MPSState* state) {
    __CPROVER_assert(state->role == ROLE_CHILD,
        "Child role should be CHILD");
    __CPROVER_assert(!state->gpu_context.is_valid,
        "Child GPU context must be invalid");
    __CPROVER_assert(state->init_state == INIT_UNINITIALIZED,
        "Child must be uninitialized");
    __CPROVER_assert(state->num_active_streams == 0,
        "Child must have no active streams");

    // All streams must be deactivated
    for (int i = 0; i < MAX_STREAMS; i++) {
        __CPROVER_assert(!state->streams[i].is_active,
            "Child streams must be inactive");
        __CPROVER_assert(state->streams[i].context == NULL,
            "Child streams must have NULL context");
    }
}

// ============================================================================
// Main Harness
// ============================================================================

int main(void) {
    // Initialize state
    MPSState_init(&g_mps_state);

    // ==== PHASE 1: Normal initialization ====

    bool init_ok = MPSState_initGPU(&g_mps_state);
    __CPROVER_assert(init_ok, "GPU init should succeed");
    __CPROVER_assert(g_mps_state.init_state == INIT_COMPLETE,
        "Init state should be complete");
    __CPROVER_assert(g_mps_state.gpu_context.is_valid,
        "GPU context should be valid");

    uint64_t original_context_id = g_mps_state.gpu_context.context_id;
    __CPROVER_assert(original_context_id > 0, "Context ID should be non-zero");

    // ==== PHASE 2: Acquire streams and submit work ====

    int stream0 = MPSState_acquireStream(&g_mps_state);
    __CPROVER_assert(stream0 >= 0, "Should acquire stream 0");

    int stream1 = MPSState_acquireStream(&g_mps_state);
    __CPROVER_assert(stream1 >= 0, "Should acquire stream 1");
    __CPROVER_assert(stream0 != stream1, "Streams must be different");

    // Submit some work
    MPSState_submitWork(&g_mps_state, stream0);
    MPSState_submitWork(&g_mps_state, stream0);
    MPSState_submitWork(&g_mps_state, stream1);

    __CPROVER_assert(g_mps_state.streams[stream0].pending_ops == 2,
        "Stream 0 should have 2 pending ops");
    __CPROVER_assert(g_mps_state.streams[stream1].pending_ops == 1,
        "Stream 1 should have 1 pending op");

    // ==== PHASE 3: Fork with pending work ====

    bool is_child = false;
    int fork_result = simulate_fork(&g_mps_state, &is_child);

    if (is_child) {
        // ==== CHILD PATH ====

        __CPROVER_assert(fork_result == 0, "Child should get 0 from fork");
        check_child_invariants(&g_mps_state);

        // Child must reinitialize before using MPS
        bool reinit_ok = MPSState_initGPU(&g_mps_state);
        __CPROVER_assert(reinit_ok, "Child reinit should succeed");

        // New context should have different ID
        __CPROVER_assert(g_mps_state.gpu_context.context_id != original_context_id,
            "Child should get new context ID");

        // Child can now acquire streams again
        int child_stream = MPSState_acquireStream(&g_mps_state);
        __CPROVER_assert(child_stream >= 0, "Child should acquire stream");

        // Submit work in child
        MPSState_submitWork(&g_mps_state, child_stream);
        MPSState_syncStream(&g_mps_state, child_stream);

    } else {
        // ==== PARENT PATH ====

        __CPROVER_assert(fork_result > 0, "Parent should get positive PID");
        check_parent_invariants(&g_mps_state);

        // Parent's context ID unchanged
        __CPROVER_assert(g_mps_state.gpu_context.context_id == original_context_id,
            "Parent context ID should be unchanged");

        // Parent's streams still active (but synced by prepare)
        __CPROVER_assert(g_mps_state.streams[stream0].is_active,
            "Parent stream 0 should still be active");
        __CPROVER_assert(g_mps_state.streams[stream0].pending_ops == 0,
            "Stream 0 should have been synced");

        // Parent can continue working
        MPSState_submitWork(&g_mps_state, stream0);
        __CPROVER_assert(g_mps_state.streams[stream0].pending_ops == 1,
            "New work should be tracked");
    }

    // ==== PHASE 4: Non-deterministic fork scenario ====

    // Reset to known state for another test scenario
    MPSState_init(&g_mps_state);
    MPSState_initGPU(&g_mps_state);

    // Non-deterministic number of pending ops
    int ops_to_submit = nondet_int();
    __CPROVER_assume(ops_to_submit >= 0 && ops_to_submit <= MAX_PENDING_OPS);

    int test_stream = MPSState_acquireStream(&g_mps_state);
    if (test_stream >= 0) {
        for (int i = 0; i < ops_to_submit; i++) {
            MPSState_submitWork(&g_mps_state, test_stream);
        }

        // Fork should flush all pending work regardless of count
        bool is_child2 = false;
        simulate_fork(&g_mps_state, &is_child2);

        // Regardless of parent/child, pending work should be 0 now
        // (either synced in prepare, or reset in child)
        __CPROVER_assert(g_mps_state.streams[test_stream].pending_ops == 0 ||
                         !g_mps_state.streams[test_stream].is_active,
            "Pending ops should be 0 or stream inactive");
    }

    // ==== PHASE 5: Multiple forks (nested/sequential) ====

    MPSState_init(&g_mps_state);
    MPSState_initGPU(&g_mps_state);

    int first_fork_count = g_mps_state.fork_count;
    bool child1 = false;
    simulate_fork(&g_mps_state, &child1);

    if (!child1) {
        // Parent: fork again
        __CPROVER_assert(g_mps_state.fork_count == first_fork_count + 1,
            "Parent fork count should increment");

        bool child2 = false;
        simulate_fork(&g_mps_state, &child2);

        if (!child2) {
            // Still parent
            __CPROVER_assert(g_mps_state.fork_count == first_fork_count + 2,
                "Parent fork count should increment again");
        } else {
            // Second-level child
            check_child_invariants(&g_mps_state);
        }
    } else {
        // First-level child
        check_child_invariants(&g_mps_state);
    }

    return 0;
}
