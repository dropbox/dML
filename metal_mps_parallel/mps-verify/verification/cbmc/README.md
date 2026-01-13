# CBMC Bounded Model Checking for MPS

This directory contains verification infrastructure for bounded model checking
of the MPS parallel inference implementation using [CBMC](https://www.cprover.org/cbmc/).

## Purpose

CBMC (C Bounded Model Checker) verifies actual C/C++ code by:
1. Unrolling loops up to a specified bound
2. Converting the program to a SAT/SMT formula
3. Checking for property violations (bounds, pointers, assertions)

Unlike TLA+ (which verifies designs), CBMC verifies implementation code.

## Directory Structure

```
cbmc/
├── README.md           # This file
├── stubs/
│   └── metal_stubs.h   # Stub definitions for Metal/ObjC types
├── models/
│   └── buffer_block.h  # Simplified BufferBlock model
└── harnesses/
    ├── aba_detection_harness.c     # ABA detection pattern verification
    ├── alloc_free_harness.c        # Buffer allocation/free cycle safety
    ├── batch_queue_harness.c       # Batch queue producer-consumer safety
    ├── command_buffer_harness.c    # Metal command buffer lifecycle
    ├── event_pool_harness.c        # Event pool management safety
    ├── fork_safety_harness.c       # Fork handler cleanup safety
    ├── graph_cache_harness.c       # MPSGraph cache thread safety
    ├── memory_ordering_harness.c   # C++11 memory ordering verification (45 assertions)
    ├── stream_pool_harness.c       # Stream pool lifecycle verification
    ├── tls_binding_harness.c       # TLS binding consistency
    └── tls_cache_harness.c         # TLS cache eviction safety
```

## Prerequisites

Install CBMC:

```bash
# macOS (Homebrew)
brew install cbmc

# Ubuntu/Debian
apt-get install cbmc

# From source
git clone https://github.com/diffblue/cbmc.git
cd cbmc && cmake -B build && cmake --build build
```

Verify installation:
```bash
cbmc --version
```

## Running Verification

### Basic Verification (Memory Safety)

```bash
cd mps-verify/verification/cbmc

# Run with pointer and bounds checking
cbmc harnesses/alloc_free_harness.c \
    -I models -I stubs \
    --unwind 10 \
    --pointer-check \
    --bounds-check \
    --unwinding-assertions

# Expected output (if verification passes):
# ** 0 of N failed (N step)
# VERIFICATION SUCCESSFUL
```

### Running All Harnesses

Most harnesses work with `--unwind 10`, but `batch_queue_harness.c` requires
`--unwind 12` due to loop iteration count (MAX_REQUESTS=10 + loop exit check).

```bash
cd mps-verify/verification/cbmc

# Run all harnesses (individual commands for clarity)
cbmc harnesses/aba_detection_harness.c -I models -I stubs --unwind 10 --pointer-check --bounds-check --unwinding-assertions
cbmc harnesses/alloc_free_harness.c -I models -I stubs --unwind 10 --pointer-check --bounds-check --unwinding-assertions
cbmc harnesses/batch_queue_harness.c -I models -I stubs --unwind 12 --pointer-check --bounds-check --unwinding-assertions
cbmc harnesses/command_buffer_harness.c -I models -I stubs --unwind 10 --pointer-check --bounds-check --unwinding-assertions
cbmc harnesses/event_pool_harness.c -I models -I stubs --unwind 10 --pointer-check --bounds-check --unwinding-assertions
cbmc harnesses/fork_safety_harness.c -I models -I stubs --unwind 10 --pointer-check --bounds-check --unwinding-assertions
cbmc harnesses/graph_cache_harness.c -I models -I stubs --unwind 10 --pointer-check --bounds-check --unwinding-assertions
cbmc harnesses/stream_pool_harness.c -I models -I stubs --unwind 10 --pointer-check --bounds-check --unwinding-assertions
cbmc harnesses/tls_binding_harness.c -I models -I stubs --unwind 10 --pointer-check --bounds-check --unwinding-assertions
cbmc harnesses/tls_cache_harness.c -I models -I stubs --unwind 10 --pointer-check --bounds-check --unwinding-assertions

# Expected: All 10 harnesses should report VERIFICATION SUCCESSFUL
```

### With Concurrency Modeling

```bash
# ARM-like memory model (Partial Store Order)
cbmc harnesses/alloc_free_harness.c \
    -I models -I stubs \
    --unwind 10 \
    --pointer-check \
    --bounds-check \
    --mm pso

# x86-like memory model (Total Store Order)
cbmc harnesses/alloc_free_harness.c \
    -I models -I stubs \
    --unwind 10 \
    --pointer-check \
    --bounds-check \
    --mm tso
```

### Finding All Violations

```bash
# Stop at first violation (default)
cbmc harnesses/alloc_free_harness.c -I models -I stubs --unwind 10

# Find all violations (slower but comprehensive)
cbmc harnesses/alloc_free_harness.c -I models -I stubs --unwind 10 --all-properties
```

### Generating Counterexamples

```bash
# With trace output
cbmc harnesses/alloc_free_harness.c -I models -I stubs --unwind 10 --trace

# JSON output for tooling
cbmc harnesses/alloc_free_harness.c -I models -I stubs --unwind 10 --json-ui
```

## What We Verify

### 1. No Double-Free
The harness asserts that `free()` is never called on a block that's already free.

### 2. No Use-After-Free
Blocks are tracked; accessing freed memory would be caught by pointer checks.

### 3. ABA Detection Correctness
The `Allocator_getSharedBufferPtr_ABA` function models the double-check pattern
with use_count verification (from issue 32.267).

### 4. Memory Bounds Safety
`--bounds-check` flag ensures all array accesses are within bounds.

### 5. Alignment Overflow Protection
`alignUp()` includes overflow check (from issue 32.256).

## Extending the Verification

### Adding New Harnesses

1. Create a new `.c` file in `harnesses/`
2. Include necessary models from `models/`
3. Use `nondet_*()` functions for non-deterministic inputs
4. Use `__CPROVER_assume()` to constrain inputs
5. Use `__CPROVER_assert()` for verification properties

### Example: Verify New Function

```c
#include "../models/buffer_block.h"

extern int nondet_int(void);
extern void __CPROVER_assume(bool);
extern void __CPROVER_assert(bool, const char*);

int main(void) {
    // Setup
    BufferBlock block;
    BufferBlock_init(&block, 1024, (void*)0x1000);

    // Non-deterministic scenario
    int scenario = nondet_int();
    __CPROVER_assume(scenario >= 0 && scenario <= 2);

    // Test different scenarios
    if (scenario == 0) {
        BufferBlock_acquire(&block);
        __CPROVER_assert(BufferBlock_isInUse(&block), "Block must be in use after acquire");
    }

    return 0;
}
```

## Limitations

1. **Bounded Unrolling**: CBMC only checks up to N loop iterations. Bugs at iteration N+1 won't be found.

2. **No Real Concurrency**: CBMC's memory models approximate concurrent behavior but don't truly execute threads in parallel.

3. **Objective-C Not Supported**: Actual MPS code uses Objective-C++. We use stub types to model the Metal API.

4. **State Space**: Complex state spaces may cause CBMC to timeout or run out of memory.

## Integration with mps-verify

The Lean 4 verification platform can invoke CBMC:

```bash
# From mps-verify root
mpsverify cbmc --harness alloc_free --unwind 10
```

(Requires implementing CBMCRunner.lean bridge - planned for Phase 3.2)

## References

- [CBMC Manual](https://www.cprover.org/cprover-manual/)
- [CBMC Tutorial](https://www.cprover.org/cbmc/doc/)
- [Memory Models in CBMC](https://www.cprover.org/cbmc/doc/memory-models.html)
