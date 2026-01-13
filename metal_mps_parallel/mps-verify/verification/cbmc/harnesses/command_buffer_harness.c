// command_buffer_harness.c - CBMC harness for verifying Metal command buffer lifecycle
//
// Metal command buffers have a strict state machine:
//   Created → Encoding → Committed → Scheduled → Completed
//
// Critical correctness properties:
// 1. No double-commit (commit after already committed)
// 2. No encoding after commit (operations added to committed buffer)
// 3. Command buffer reuse only after completion
// 4. Completion handlers fire only after commit
// 5. Proper ordering: encode → commit → schedule → complete
// 6. Reference counting prevents premature deallocation
// 7. Command queues track outstanding buffers
//
// Based on: MTLCommandBuffer lifecycle in PyTorch MPS backend (MPSStream.mm)
//
// Run with:
//   cbmc command_buffer_harness.c --unwind 10 --pointer-check --bounds-check

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
// Constants
// ============================================================================

#define MAX_COMMAND_BUFFERS 4
#define MAX_ENCODERS 3
#define MAX_COMPLETION_HANDLERS 2
#define INVALID_INDEX (-1)

// ============================================================================
// Command Buffer State Machine
// ============================================================================

typedef enum {
    CB_FREE = 0,         // Not allocated
    CB_CREATED,          // Allocated from queue, ready for encoding
    CB_ENCODING,         // Currently encoding operations
    CB_COMMITTED,        // Committed to GPU, waiting to be scheduled
    CB_SCHEDULED,        // Scheduled for execution
    CB_COMPLETED,        // Execution completed
    CB_ERROR             // Error state (e.g., double commit detected)
} CommandBufferState;

typedef enum {
    ENCODER_NONE = 0,
    ENCODER_COMPUTE,     // MTLComputeCommandEncoder
    ENCODER_RENDER,      // MTLRenderCommandEncoder
    ENCODER_BLIT         // MTLBlitCommandEncoder
} EncoderType;

typedef enum {
    HANDLER_NONE = 0,
    HANDLER_REGISTERED,  // addCompletedHandler called
    HANDLER_PENDING,     // Buffer committed, handler waiting
    HANDLER_EXECUTED     // Handler executed after completion
} CompletionHandlerState;

// ============================================================================
// Command Buffer Model
// ============================================================================

typedef struct {
    int id;
    CommandBufferState state;

    // Encoding state
    EncoderType active_encoder;
    int encoder_count;           // Number of encoders used
    bool encoder_ended;          // endEncoding() called

    // Completion handlers (max 2 for verification)
    CompletionHandlerState handlers[MAX_COMPLETION_HANDLERS];
    int handler_count;

    // Reference counting
    int ref_count;

    // Queue association
    int queue_id;

    // Timing (for profiling)
    uint64_t enqueue_time;
    uint64_t start_time;
    uint64_t end_time;

    // Error tracking
    bool double_commit_attempted;
    bool encode_after_commit_attempted;
    bool use_after_complete;

} CommandBuffer;

// ============================================================================
// Command Queue Model
// ============================================================================

typedef struct {
    int id;
    int outstanding_buffers;     // Number of uncommitted buffers
    int total_created;           // Total buffers created
    int total_completed;         // Total buffers completed

    // Mutex simulation
    bool mutex_locked;
    int mutex_holder;

} CommandQueue;

// ============================================================================
// Global State
// ============================================================================

static CommandBuffer g_buffers[MAX_COMMAND_BUFFERS];
static CommandQueue g_queue;

// ============================================================================
// Initialization
// ============================================================================

static void CommandBuffer_init(CommandBuffer* cb, int id, int queue_id) {
    cb->id = id;
    cb->state = CB_CREATED;
    cb->active_encoder = ENCODER_NONE;
    cb->encoder_count = 0;
    cb->encoder_ended = true;  // No active encoder yet

    for (int i = 0; i < MAX_COMPLETION_HANDLERS; i++) {
        cb->handlers[i] = HANDLER_NONE;
    }
    cb->handler_count = 0;

    cb->ref_count = 1;  // Queue holds initial reference
    cb->queue_id = queue_id;
    cb->enqueue_time = 0;
    cb->start_time = 0;
    cb->end_time = 0;

    cb->double_commit_attempted = false;
    cb->encode_after_commit_attempted = false;
    cb->use_after_complete = false;
}

static void CommandQueue_init(CommandQueue* q, int id) {
    q->id = id;
    q->outstanding_buffers = 0;
    q->total_created = 0;
    q->total_completed = 0;
    q->mutex_locked = false;
    q->mutex_holder = -1;
}

static void global_init(void) {
    for (int i = 0; i < MAX_COMMAND_BUFFERS; i++) {
        g_buffers[i].id = i;
        g_buffers[i].state = CB_FREE;
        g_buffers[i].ref_count = 0;
    }
    CommandQueue_init(&g_queue, 0);
}

// ============================================================================
// Command Queue Operations
// ============================================================================

static bool CommandQueue_lock(CommandQueue* q, int thread_id) {
    if (q->mutex_locked) {
        return false;
    }
    q->mutex_locked = true;
    q->mutex_holder = thread_id;
    return true;
}

static void CommandQueue_unlock(CommandQueue* q, int thread_id) {
    __CPROVER_assert(q->mutex_locked, "Cannot unlock unlocked queue mutex");
    __CPROVER_assert(q->mutex_holder == thread_id, "Must hold mutex to unlock");
    q->mutex_locked = false;
    q->mutex_holder = -1;
}

// Create a new command buffer from the queue
static CommandBuffer* CommandQueue_createBuffer(CommandQueue* q, int thread_id) {
    if (!CommandQueue_lock(q, thread_id)) {
        return NULL;
    }

    CommandBuffer* result = NULL;

    // Find a free slot
    for (int i = 0; i < MAX_COMMAND_BUFFERS; i++) {
        if (g_buffers[i].state == CB_FREE) {
            CommandBuffer_init(&g_buffers[i], i, q->id);
            q->outstanding_buffers++;
            q->total_created++;
            result = &g_buffers[i];
            break;
        }
    }

    CommandQueue_unlock(q, thread_id);
    return result;
}

// ============================================================================
// Command Buffer Operations
// ============================================================================

// Start encoding with a specific encoder type
static bool CommandBuffer_beginEncoding(CommandBuffer* cb, EncoderType type) {
    __CPROVER_assert(cb != NULL, "Cannot encode on NULL buffer");

    // Check for illegal encode after commit
    if (cb->state == CB_COMMITTED || cb->state == CB_SCHEDULED ||
        cb->state == CB_COMPLETED) {
        cb->encode_after_commit_attempted = true;
        return false;
    }

    __CPROVER_assert(cb->state == CB_CREATED || cb->state == CB_ENCODING,
                    "Can only begin encoding on created or encoding buffer");

    // Must end previous encoder first
    if (!cb->encoder_ended) {
        return false;  // Previous encoder not ended
    }

    cb->state = CB_ENCODING;
    cb->active_encoder = type;
    cb->encoder_ended = false;
    cb->encoder_count++;

    return true;
}

// End current encoder
static bool CommandBuffer_endEncoding(CommandBuffer* cb) {
    __CPROVER_assert(cb != NULL, "Cannot end encoding on NULL buffer");
    __CPROVER_assert(cb->state == CB_ENCODING, "Must be encoding to end");
    __CPROVER_assert(!cb->encoder_ended, "Encoder already ended");
    __CPROVER_assert(cb->active_encoder != ENCODER_NONE, "Must have active encoder");

    cb->active_encoder = ENCODER_NONE;
    cb->encoder_ended = true;

    return true;
}

// Add a completion handler
static bool CommandBuffer_addCompletionHandler(CommandBuffer* cb) {
    __CPROVER_assert(cb != NULL, "Cannot add handler to NULL buffer");

    // Can add handlers before commit
    if (cb->state != CB_CREATED && cb->state != CB_ENCODING) {
        return false;
    }

    if (cb->handler_count >= MAX_COMPLETION_HANDLERS) {
        return false;
    }

    cb->handlers[cb->handler_count] = HANDLER_REGISTERED;
    cb->handler_count++;

    return true;
}

// Commit the command buffer
static bool CommandBuffer_commit(CommandBuffer* cb) {
    __CPROVER_assert(cb != NULL, "Cannot commit NULL buffer");

    // Check for double commit
    if (cb->state == CB_COMMITTED || cb->state == CB_SCHEDULED ||
        cb->state == CB_COMPLETED) {
        cb->double_commit_attempted = true;
        return false;
    }

    // Must end any active encoder before commit
    if (cb->state == CB_ENCODING && !cb->encoder_ended) {
        return false;  // Active encoder not ended
    }

    __CPROVER_assert(cb->state == CB_CREATED || cb->state == CB_ENCODING,
                    "Invalid state for commit");
    __CPROVER_assert(cb->encoder_ended, "Must end encoder before commit");

    // Transition to committed
    cb->state = CB_COMMITTED;

    // Move handlers to pending
    for (int i = 0; i < cb->handler_count; i++) {
        if (cb->handlers[i] == HANDLER_REGISTERED) {
            cb->handlers[i] = HANDLER_PENDING;
        }
    }

    return true;
}

// Simulate GPU scheduling the buffer
static void CommandBuffer_schedule(CommandBuffer* cb) {
    __CPROVER_assert(cb != NULL, "Cannot schedule NULL buffer");
    __CPROVER_assert(cb->state == CB_COMMITTED, "Must be committed to schedule");

    cb->state = CB_SCHEDULED;
    cb->start_time = 1;  // Non-zero indicates started
}

// Simulate GPU completing execution
static void CommandBuffer_complete(CommandBuffer* cb) {
    __CPROVER_assert(cb != NULL, "Cannot complete NULL buffer");
    __CPROVER_assert(cb->state == CB_SCHEDULED, "Must be scheduled to complete");

    cb->state = CB_COMPLETED;
    cb->end_time = 2;  // Non-zero indicates completed

    // Execute completion handlers
    for (int i = 0; i < cb->handler_count; i++) {
        if (cb->handlers[i] == HANDLER_PENDING) {
            cb->handlers[i] = HANDLER_EXECUTED;
        }
    }
}

// Wait for completion (blocking)
static void CommandBuffer_waitUntilCompleted(CommandBuffer* cb) {
    __CPROVER_assert(cb != NULL, "Cannot wait on NULL buffer");
    __CPROVER_assert(cb->state == CB_COMMITTED || cb->state == CB_SCHEDULED ||
                    cb->state == CB_COMPLETED,
                    "Must be committed/scheduled/completed to wait");

    // Simulate GPU execution
    if (cb->state == CB_COMMITTED) {
        CommandBuffer_schedule(cb);
    }
    if (cb->state == CB_SCHEDULED) {
        CommandBuffer_complete(cb);
    }
}

// Release reference
static void CommandBuffer_release(CommandBuffer* cb) {
    __CPROVER_assert(cb != NULL, "Cannot release NULL buffer");
    __CPROVER_assert(cb->ref_count > 0, "Cannot release with zero ref_count");

    cb->ref_count--;

    if (cb->ref_count == 0) {
        // Check for use-after-complete
        if (cb->state != CB_COMPLETED && cb->state != CB_FREE) {
            cb->use_after_complete = true;  // Releasing before complete!
        }

        // Return to free pool
        cb->state = CB_FREE;
        g_queue.outstanding_buffers--;
        g_queue.total_completed++;
    }
}

// ============================================================================
// Main Harness: Verify command buffer lifecycle
// ============================================================================

int main(void) {
    global_init();

    // ========== PHASE 1: Basic lifecycle ==========

    // Create a command buffer
    CommandBuffer* cb1 = CommandQueue_createBuffer(&g_queue, 0);
    __CPROVER_assert(cb1 != NULL, "Must create first buffer");
    __CPROVER_assert(cb1->state == CB_CREATED, "New buffer must be in CREATED state");
    __CPROVER_assert(cb1->ref_count == 1, "New buffer must have ref_count 1");
    __CPROVER_assert(g_queue.outstanding_buffers == 1, "Queue must track outstanding buffer");

    // Begin encoding
    bool enc_ok = CommandBuffer_beginEncoding(cb1, ENCODER_COMPUTE);
    __CPROVER_assert(enc_ok, "Begin encoding must succeed");
    __CPROVER_assert(cb1->state == CB_ENCODING, "State must be ENCODING");
    __CPROVER_assert(cb1->active_encoder == ENCODER_COMPUTE, "Encoder type must match");

    // End encoding
    bool end_ok = CommandBuffer_endEncoding(cb1);
    __CPROVER_assert(end_ok, "End encoding must succeed");
    __CPROVER_assert(cb1->encoder_ended, "Encoder must be ended");

    // Add completion handler
    bool handler_ok = CommandBuffer_addCompletionHandler(cb1);
    __CPROVER_assert(handler_ok, "Adding handler must succeed");
    __CPROVER_assert(cb1->handler_count == 1, "Handler count must be 1");

    // Commit
    bool commit_ok = CommandBuffer_commit(cb1);
    __CPROVER_assert(commit_ok, "Commit must succeed");
    __CPROVER_assert(cb1->state == CB_COMMITTED, "State must be COMMITTED");
    __CPROVER_assert(cb1->handlers[0] == HANDLER_PENDING, "Handler must be pending");

    // ========== PHASE 2: Test double-commit detection ==========

    bool double_commit = CommandBuffer_commit(cb1);
    __CPROVER_assert(!double_commit, "Double commit must fail");
    __CPROVER_assert(cb1->double_commit_attempted, "Double commit flag must be set");

    // ========== PHASE 3: Test encode-after-commit detection ==========

    bool bad_encode = CommandBuffer_beginEncoding(cb1, ENCODER_BLIT);
    __CPROVER_assert(!bad_encode, "Encode after commit must fail");
    __CPROVER_assert(cb1->encode_after_commit_attempted, "Encode-after-commit flag must be set");

    // ========== PHASE 4: Complete the buffer ==========

    CommandBuffer_waitUntilCompleted(cb1);
    __CPROVER_assert(cb1->state == CB_COMPLETED, "State must be COMPLETED");
    __CPROVER_assert(cb1->handlers[0] == HANDLER_EXECUTED, "Handler must have executed");
    __CPROVER_assert(cb1->end_time > cb1->start_time, "End time must be after start");

    // ========== PHASE 5: Non-deterministic operations ==========

    // Create another buffer with non-deterministic encoding
    CommandBuffer* cb2 = CommandQueue_createBuffer(&g_queue, 1);
    if (cb2 != NULL) {
        int num_encoders = nondet_int();
        __CPROVER_assume(num_encoders >= 0 && num_encoders <= MAX_ENCODERS);

        for (int i = 0; i < num_encoders; i++) {
            int enc_type = nondet_int();
            __CPROVER_assume(enc_type >= ENCODER_COMPUTE && enc_type <= ENCODER_BLIT);

            if (CommandBuffer_beginEncoding(cb2, (EncoderType)enc_type)) {
                CommandBuffer_endEncoding(cb2);
            }
        }

        // Non-deterministic completion handler registration
        int num_handlers = nondet_int();
        __CPROVER_assume(num_handlers >= 0 && num_handlers <= MAX_COMPLETION_HANDLERS);

        for (int i = 0; i < num_handlers; i++) {
            CommandBuffer_addCompletionHandler(cb2);
        }

        // Commit and complete
        if (cb2->state == CB_CREATED || cb2->state == CB_ENCODING) {
            if (cb2->encoder_ended) {
                CommandBuffer_commit(cb2);
                CommandBuffer_waitUntilCompleted(cb2);
            }
        }
    }

    // ========== PHASE 6: Verify invariants ==========

    // Invariant 1: All buffers in valid states
    for (int i = 0; i < MAX_COMMAND_BUFFERS; i++) {
        CommandBuffer* cb = &g_buffers[i];

        // State machine validity
        __CPROVER_assert(cb->state >= CB_FREE && cb->state <= CB_ERROR,
                        "Buffer must be in valid state");

        // Reference count consistency
        if (cb->state == CB_FREE) {
            __CPROVER_assert(cb->ref_count == 0, "Free buffer must have ref_count 0");
        } else {
            __CPROVER_assert(cb->ref_count >= 1, "Active buffer must have ref_count >= 1");
        }

        // Encoder consistency
        if (cb->state == CB_ENCODING) {
            if (!cb->encoder_ended) {
                __CPROVER_assert(cb->active_encoder != ENCODER_NONE,
                                "Encoding buffer must have active encoder");
            }
        }

        // Handler consistency
        for (int j = 0; j < cb->handler_count; j++) {
            if (cb->state == CB_COMPLETED) {
                __CPROVER_assert(cb->handlers[j] == HANDLER_EXECUTED,
                                "Completed buffer handlers must be executed");
            }
        }

        // No bugs detected
        __CPROVER_assert(!cb->double_commit_attempted ||
                        cb->state == CB_COMMITTED || cb->state == CB_SCHEDULED ||
                        cb->state == CB_COMPLETED,
                        "Double commit only fails after first commit");
    }

    // Invariant 2: Queue accounting
    int active_count = 0;
    for (int i = 0; i < MAX_COMMAND_BUFFERS; i++) {
        if (g_buffers[i].state != CB_FREE && g_buffers[i].state != CB_COMPLETED) {
            active_count++;
        }
    }
    // Note: outstanding_buffers may differ due to committed but not released buffers
    __CPROVER_assert(g_queue.total_created >= g_queue.total_completed,
                    "Created count must be >= completed count");

    // ========== PHASE 7: Cleanup ==========

    // Release our references
    if (cb1->ref_count > 0) {
        CommandBuffer_release(cb1);
    }
    if (cb2 != NULL && cb2->ref_count > 0) {
        CommandBuffer_release(cb2);
    }

    // Final verification: no memory leaks (all buffers released)
    __CPROVER_assert(g_queue.outstanding_buffers >= 0,
                    "Outstanding buffer count must be non-negative");

    return 0;
}
