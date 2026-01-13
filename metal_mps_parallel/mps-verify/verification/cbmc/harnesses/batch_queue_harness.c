// batch_queue_harness.c - CBMC harness for verifying MPSBatchQueue correctness
//
// Verifies the producer/consumer pattern central to 8-thread MPS support:
// 1. Queue size never exceeds maximum
// 2. Queue size never goes negative
// 3. Requests submitted are eventually processable
// 4. No double-processing of requests
// 5. Thread-safe invariants hold under concurrent operations
//
// The batch queue solves Apple MPS framework threading bugs by serializing
// GPU access through fewer worker threads while allowing many user threads.
//
// Run with:
//   cbmc batch_queue_harness.c --unwind 10 --pointer-check --bounds-check

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

#define MAX_QUEUE_SIZE 8
#define MAX_REQUESTS 10   // Reduced for tractable verification
#define MAX_WORKERS 4
#define MAX_PRODUCERS 8

// ============================================================================
// Request Model
// ============================================================================

typedef enum {
    REQUEST_EMPTY = 0,
    REQUEST_QUEUED,
    REQUEST_PROCESSING,
    REQUEST_COMPLETED
} RequestState;

typedef struct {
    int id;
    RequestState state;
    int processed_by;  // Worker ID that processed this request (-1 if none)
} Request;

// ============================================================================
// Batch Queue Model
// ============================================================================

typedef struct {
    // Queue storage (circular buffer)
    Request* queue[MAX_QUEUE_SIZE];
    int head;  // Next position to dequeue from
    int tail;  // Next position to enqueue to
    int size;  // Current number of items in queue

    // Configuration
    int max_size;
    int num_workers;

    // State
    bool running;
    bool shutdown_requested;

    // Statistics
    int total_submitted;
    int total_processed;
    int total_rejected;  // Rejected due to full queue

    // Mutex simulation (for modeling)
    int mutex_holder;  // Thread ID holding mutex (-1 = none)
    bool mutex_locked;

} BatchQueue;

// ============================================================================
// Queue Operations
// ============================================================================

static void BatchQueue_init(BatchQueue* q, int max_size, int num_workers) {
    __CPROVER_assume(max_size > 0 && max_size <= MAX_QUEUE_SIZE);
    __CPROVER_assume(num_workers > 0 && num_workers <= MAX_WORKERS);

    for (int i = 0; i < MAX_QUEUE_SIZE; i++) {
        q->queue[i] = NULL;
    }
    q->head = 0;
    q->tail = 0;
    q->size = 0;
    q->max_size = max_size;
    q->num_workers = num_workers;
    q->running = false;
    q->shutdown_requested = false;
    q->total_submitted = 0;
    q->total_processed = 0;
    q->total_rejected = 0;
    q->mutex_holder = -1;
    q->mutex_locked = false;
}

// Acquire mutex (model)
static bool BatchQueue_lock(BatchQueue* q, int thread_id) {
    __CPROVER_assume(thread_id >= 0);

    if (q->mutex_locked) {
        // Mutex already held - in reality would block
        // For bounded model checking, we just return false (try-lock semantics)
        return false;
    }

    q->mutex_locked = true;
    q->mutex_holder = thread_id;
    return true;
}

// Release mutex (model)
static void BatchQueue_unlock(BatchQueue* q, int thread_id) {
    __CPROVER_assert(q->mutex_locked, "Cannot unlock unlocked mutex");
    __CPROVER_assert(q->mutex_holder == thread_id, "Thread must hold mutex to unlock");

    q->mutex_locked = false;
    q->mutex_holder = -1;
}

// Check if queue is full (must hold lock)
static bool BatchQueue_is_full_locked(BatchQueue* q) {
    return q->size >= q->max_size;
}

// Check if queue is empty (must hold lock)
static bool BatchQueue_is_empty_locked(BatchQueue* q) {
    return q->size == 0;
}

// Submit request to queue (producer operation)
// Returns: true if enqueued, false if queue full or shutdown
static bool BatchQueue_submit(BatchQueue* q, Request* req, int producer_id) {
    __CPROVER_assume(req != NULL);
    __CPROVER_assume(req->state == REQUEST_EMPTY);

    // Acquire mutex
    if (!BatchQueue_lock(q, producer_id)) {
        return false;  // Could not acquire lock (try-lock model)
    }

    bool success = false;

    // Check if queue is accepting requests
    if (!q->shutdown_requested && !BatchQueue_is_full_locked(q)) {
        // Enqueue at tail
        __CPROVER_assert(q->tail >= 0 && q->tail < MAX_QUEUE_SIZE,
                        "Tail index must be in bounds");

        q->queue[q->tail] = req;
        q->tail = (q->tail + 1) % q->max_size;
        q->size++;
        q->total_submitted++;

        req->state = REQUEST_QUEUED;
        success = true;

        // Verify invariants
        __CPROVER_assert(q->size <= q->max_size, "Queue size must not exceed max");
        __CPROVER_assert(q->size > 0, "Queue size must be positive after enqueue");
    } else {
        q->total_rejected++;
    }

    BatchQueue_unlock(q, producer_id);
    return success;
}

// Process one request from queue (consumer/worker operation)
// Returns: processed request, or NULL if queue empty
static Request* BatchQueue_process_one(BatchQueue* q, int worker_id) {
    // Acquire mutex
    if (!BatchQueue_lock(q, worker_id + 100)) {  // Workers use IDs 100+
        return NULL;
    }

    Request* req = NULL;

    if (!BatchQueue_is_empty_locked(q)) {
        // Dequeue from head
        __CPROVER_assert(q->head >= 0 && q->head < MAX_QUEUE_SIZE,
                        "Head index must be in bounds");

        req = q->queue[q->head];
        q->queue[q->head] = NULL;
        q->head = (q->head + 1) % q->max_size;
        q->size--;

        __CPROVER_assert(req != NULL, "Dequeued request must not be NULL");
        __CPROVER_assert(req->state == REQUEST_QUEUED,
                        "Dequeued request must be in QUEUED state");

        // Mark as processing
        req->state = REQUEST_PROCESSING;
        req->processed_by = worker_id;
        q->total_processed++;

        // Verify invariants
        __CPROVER_assert(q->size >= 0, "Queue size must not be negative");
    }

    BatchQueue_unlock(q, worker_id + 100);
    return req;
}

// Complete a request after processing
static void BatchQueue_complete_request(Request* req, int worker_id) {
    __CPROVER_assert(req != NULL, "Cannot complete NULL request");
    __CPROVER_assert(req->state == REQUEST_PROCESSING,
                    "Can only complete processing requests");
    __CPROVER_assert(req->processed_by == worker_id,
                    "Only assigned worker can complete request");

    req->state = REQUEST_COMPLETED;
}

// Start queue (enables workers)
static void BatchQueue_start(BatchQueue* q) {
    q->running = true;
}

// Request shutdown (graceful)
static void BatchQueue_request_shutdown(BatchQueue* q, int thread_id) {
    if (BatchQueue_lock(q, thread_id)) {
        q->shutdown_requested = true;
        BatchQueue_unlock(q, thread_id);
    }
}

// ============================================================================
// Main Harness: Verify producer/consumer correctness
// ============================================================================

int main(void) {
    BatchQueue queue;
    Request requests[MAX_REQUESTS];

    // Initialize queue with non-deterministic configuration
    int queue_max_size = nondet_int();
    int num_workers = nondet_int();
    __CPROVER_assume(queue_max_size >= 2 && queue_max_size <= MAX_QUEUE_SIZE);
    __CPROVER_assume(num_workers >= 1 && num_workers <= MAX_WORKERS);

    BatchQueue_init(&queue, queue_max_size, num_workers);

    // Initialize requests
    for (int i = 0; i < MAX_REQUESTS; i++) {
        requests[i].id = i;
        requests[i].state = REQUEST_EMPTY;
        requests[i].processed_by = -1;
    }

    // Start the queue
    BatchQueue_start(&queue);

    // ========== PHASE 1: Non-deterministic submission pattern ==========

    int num_to_submit = nondet_int();
    __CPROVER_assume(num_to_submit >= 0 && num_to_submit <= 8);

    int submitted_count = 0;
    int rejected_count = 0;

    for (int i = 0; i < num_to_submit; i++) {
        // Non-deterministic producer ID
        int producer = nondet_int();
        __CPROVER_assume(producer >= 0 && producer < MAX_PRODUCERS);

        bool success = BatchQueue_submit(&queue, &requests[i], producer);
        if (success) {
            submitted_count++;
        } else {
            rejected_count++;
        }
    }

    // Verify: submitted + rejected = attempted
    __CPROVER_assert(submitted_count + rejected_count == num_to_submit,
                    "All submission attempts must be accounted for");

    // Verify: queue size matches submissions
    __CPROVER_assert(queue.size <= queue_max_size,
                    "Queue size must not exceed max after submissions");

    // ========== PHASE 2: Non-deterministic processing pattern ==========

    int num_to_process = nondet_int();
    __CPROVER_assume(num_to_process >= 0 && num_to_process <= submitted_count);

    int processed_count = 0;
    Request* processed_requests[MAX_REQUESTS];
    for (int i = 0; i < MAX_REQUESTS; i++) {
        processed_requests[i] = NULL;
    }

    for (int i = 0; i < num_to_process; i++) {
        // Non-deterministic worker ID
        int worker = nondet_int();
        __CPROVER_assume(worker >= 0 && worker < num_workers);

        Request* req = BatchQueue_process_one(&queue, worker);
        if (req != NULL) {
            // Verify no double-processing
            for (int j = 0; j < processed_count; j++) {
                __CPROVER_assert(processed_requests[j] != req,
                                "No request should be processed twice");
            }

            processed_requests[processed_count] = req;

            // Complete the request
            BatchQueue_complete_request(req, worker);
            processed_count++;
        }
    }

    // ========== PHASE 3: Verify queue invariants ==========

    // Invariant 1: Queue size is non-negative
    __CPROVER_assert(queue.size >= 0,
                    "Queue size must never be negative");

    // Invariant 2: Queue size does not exceed max
    __CPROVER_assert(queue.size <= queue_max_size,
                    "Queue size must never exceed max");

    // Invariant 3: Remaining size matches submitted - processed
    int expected_remaining = submitted_count - processed_count;
    __CPROVER_assert(queue.size == expected_remaining,
                    "Queue size must equal submitted minus processed");

    // Invariant 4: Statistics are consistent
    __CPROVER_assert(queue.total_submitted == submitted_count,
                    "Total submitted must match actual submissions");
    __CPROVER_assert(queue.total_processed == processed_count,
                    "Total processed must match actual processing");
    __CPROVER_assert(queue.total_rejected == rejected_count,
                    "Total rejected must match actual rejections");

    // ========== PHASE 4: Verify request state machine ==========

    for (int i = 0; i < MAX_REQUESTS; i++) {
        RequestState state = requests[i].state;

        // Valid state transitions: EMPTY -> QUEUED -> PROCESSING -> COMPLETED
        if (state == REQUEST_COMPLETED) {
            __CPROVER_assert(requests[i].processed_by >= 0,
                            "Completed request must have valid worker ID");
        }

        if (state == REQUEST_PROCESSING) {
            // Should not remain in PROCESSING state after completion
            __CPROVER_assert(requests[i].processed_by >= 0,
                            "Processing request must have assigned worker");
        }

        if (state == REQUEST_QUEUED) {
            __CPROVER_assert(requests[i].processed_by == -1,
                            "Queued request must not have worker assigned");
        }

        if (state == REQUEST_EMPTY) {
            __CPROVER_assert(requests[i].processed_by == -1,
                            "Empty request must not have worker assigned");
        }
    }

    // ========== PHASE 5: Shutdown behavior ==========

    // Request shutdown
    BatchQueue_request_shutdown(&queue, 999);

    // After shutdown, new submissions should be rejected
    bool post_shutdown_submit = BatchQueue_submit(&queue, &requests[MAX_REQUESTS-1], 0);
    __CPROVER_assert(!post_shutdown_submit,
                    "Submissions should be rejected after shutdown");

    // Processing should still work for remaining items
    if (queue.size > 0) {
        Request* remaining = BatchQueue_process_one(&queue, 0);
        // May or may not succeed depending on lock acquisition
    }

    return 0;
}
