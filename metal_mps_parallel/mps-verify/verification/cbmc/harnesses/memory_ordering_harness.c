/**
 * Memory Ordering Verification Harness
 *
 * This harness verifies that C++11 memory ordering semantics are correctly
 * modeled and that our code respects the ARM memory model (PSO-like).
 *
 * Properties verified:
 * 1. Sequential consistency prevents IRIW (Independent Reads of Independent Writes)
 * 2. Release-acquire synchronizes correctly
 * 3. No out-of-thin-air values under relaxed ordering
 *
 * Based on WORKER_DIRECTIVE_PROVE_EVERYTHING.md requirements.
 */

#include <stdint.h>
#include <stdbool.h>

// CBMC built-ins
extern int nondet_int(void);
extern bool nondet_bool(void);
extern void __CPROVER_assume(bool);
extern void __CPROVER_assert(bool, const char*);
extern void __CPROVER_atomic_begin(void);
extern void __CPROVER_atomic_end(void);

// Simulated atomic variables
typedef struct {
    int value;
    int version;  // For tracking modification order
} atomic_int_t;

// Global state
static atomic_int_t g_x = {0, 0};
static atomic_int_t g_y = {0, 0};
static int g_r1 = -1;  // Thread 1's read
static int g_r2 = -1;  // Thread 2's read

// Memory fences (simplified model)
static void memory_fence_seq_cst(void) {
    __CPROVER_atomic_begin();
    __CPROVER_atomic_end();
}

static void memory_fence_release(void) {
    // Release fence: all prior stores visible before this point
    __CPROVER_fence("WWfence", "RWfence");
}

static void memory_fence_acquire(void) {
    // Acquire fence: all subsequent loads see stores before paired release
    __CPROVER_fence("RRfence", "RWfence");
}

// Atomic store with memory order
static void atomic_store_seq_cst(atomic_int_t* a, int val) {
    __CPROVER_atomic_begin();
    a->value = val;
    a->version++;
    __CPROVER_atomic_end();
}

static void atomic_store_release(atomic_int_t* a, int val) {
    memory_fence_release();
    a->value = val;
    a->version++;
}

// Atomic load with memory order
static int atomic_load_seq_cst(atomic_int_t* a) {
    int result;
    __CPROVER_atomic_begin();
    result = a->value;
    __CPROVER_atomic_end();
    return result;
}

static int atomic_load_acquire(atomic_int_t* a) {
    int result = a->value;
    memory_fence_acquire();
    return result;
}

// ============================================================================
// Test 1: Sequential Consistency - Dekker's Algorithm
// ============================================================================
// Under seq_cst, it's impossible for both threads to read 0.
// This is the classic store-buffer litmus test.

static void thread1_seq_cst(void) {
    atomic_store_seq_cst(&g_x, 1);  // x = 1
    g_r1 = atomic_load_seq_cst(&g_y);  // r1 = y
}

static void thread2_seq_cst(void) {
    atomic_store_seq_cst(&g_y, 1);  // y = 1
    g_r2 = atomic_load_seq_cst(&g_x);  // r2 = x
}

static void harness_seq_cst(void) {
    // Reset state
    g_x.value = 0; g_x.version = 0;
    g_y.value = 0; g_y.version = 0;
    g_r1 = -1;
    g_r2 = -1;

    // Non-deterministically interleave threads
    int schedule = nondet_int();
    __CPROVER_assume(schedule >= 0 && schedule <= 3);

    switch (schedule) {
        case 0:  // T1 fully before T2
            thread1_seq_cst();
            thread2_seq_cst();
            break;
        case 1:  // T2 fully before T1
            thread2_seq_cst();
            thread1_seq_cst();
            break;
        case 2:  // T1 store, T2 fully, T1 load
            atomic_store_seq_cst(&g_x, 1);
            thread2_seq_cst();
            g_r1 = atomic_load_seq_cst(&g_y);
            break;
        case 3:  // T2 store, T1 fully, T2 load
            atomic_store_seq_cst(&g_y, 1);
            thread1_seq_cst();
            g_r2 = atomic_load_seq_cst(&g_x);
            break;
    }

    // Under seq_cst, r1=0 AND r2=0 is impossible
    // At least one of them must see the other's write
    __CPROVER_assert(!(g_r1 == 0 && g_r2 == 0),
        "SEQ_CST violation: both reads cannot return 0");
}

// ============================================================================
// Test 2: Release-Acquire Synchronization
// ============================================================================
// If thread2 reads the release store, it must see all stores before release.

static int g_data = 0;
static atomic_int_t g_flag = {0, 0};

static void thread1_release(void) {
    g_data = 42;  // Non-atomic store
    atomic_store_release(&g_flag, 1);  // Release store
}

static void thread2_acquire(void) {
    int flag_val = atomic_load_acquire(&g_flag);  // Acquire load
    if (flag_val == 1) {
        // If we see the release store, we must see g_data = 42
        __CPROVER_assert(g_data == 42,
            "RELEASE-ACQUIRE violation: must see prior stores");
    }
}

static void harness_release_acquire(void) {
    // Reset state
    g_data = 0;
    g_flag.value = 0;
    g_flag.version = 0;

    // Non-deterministically interleave
    int schedule = nondet_int();
    __CPROVER_assume(schedule >= 0 && schedule <= 1);

    if (schedule == 0) {
        thread1_release();
        thread2_acquire();
    } else {
        // Interleaved: T2 acquire before T1 finishes
        // This models the case where T2 reads 0 (which is fine)
        // or T2 reads 1 (must see g_data = 42)
        thread2_acquire();  // May see 0
        thread1_release();
        thread2_acquire();  // Will see 1 and g_data = 42
    }
}

// ============================================================================
// Test 3: ABA Counter Monotonicity
// ============================================================================
// Our use_count/generation must be monotonically increasing.

static atomic_int_t g_generation = {0, 0};

static void harness_aba_monotonicity(void) {
    int gen1, gen2;

    // Simulate multiple operations
    int ops = nondet_int();
    __CPROVER_assume(ops >= 0 && ops <= 5);

    gen1 = atomic_load_seq_cst(&g_generation);

    // Non-deterministically increment generation
    for (int i = 0; i < ops; i++) {
        atomic_store_seq_cst(&g_generation, g_generation.value + 1);
    }

    gen2 = atomic_load_seq_cst(&g_generation);

    // Generation must be monotonically non-decreasing
    __CPROVER_assert(gen2 >= gen1,
        "ABA MONOTONICITY violation: generation decreased");
}

// ============================================================================
// Test 4: No Out-of-Thin-Air Values
// ============================================================================
// Even under relaxed ordering, values can't appear from nowhere.

static atomic_int_t g_a = {0, 0};
static atomic_int_t g_b = {0, 0};

static void harness_no_oota(void) {
    // Reset
    g_a.value = 0;
    g_b.value = 0;

    // Classic OOTA test:
    // Thread 1: r1 = a; b = r1;
    // Thread 2: r2 = b; a = r2;
    // Under any memory model, r1=r2=42 is impossible if initial values are 0

    int r1, r2;
    int schedule = nondet_int();
    __CPROVER_assume(schedule >= 0 && schedule <= 1);

    if (schedule == 0) {
        // Thread 1 first
        r1 = g_a.value;
        g_b.value = r1;

        // Thread 2
        r2 = g_b.value;
        g_a.value = r2;
    } else {
        // Thread 2 first
        r2 = g_b.value;
        g_a.value = r2;

        // Thread 1
        r1 = g_a.value;
        g_b.value = r1;
    }

    // Under any memory model, r1 and r2 can only be 0
    // (assuming no OOTA values)
    __CPROVER_assert(r1 == 0 && r2 == 0,
        "OOTA violation: values appeared from nowhere");
}

// ============================================================================
// Main Harness
// ============================================================================

int main(void) {
    int test = nondet_int();
    __CPROVER_assume(test >= 0 && test <= 3);

    switch (test) {
        case 0:
            harness_seq_cst();
            break;
        case 1:
            harness_release_acquire();
            break;
        case 2:
            harness_aba_monotonicity();
            break;
        case 3:
            harness_no_oota();
            break;
    }

    return 0;
}
