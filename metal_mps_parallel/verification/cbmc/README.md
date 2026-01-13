# CBMC Verification for MPS Parallel Inference

Bounded model checking of concurrent data structures using [CBMC](https://www.cprover.org/cbmc/).

## Overview

CBMC (C Bounded Model Checker) verifies C/C++ code by exploring all possible execution paths up to a specified bound. It finds:
- Memory safety violations (buffer overflows, use-after-free)
- Assertion violations
- Data races under specific memory models
- Undefined behavior

## Verified Models

### 1. Batch Queue (`batch_queue_c_model.c`)

Models the `MPSBatchQueue` producer/consumer pattern:

**Properties Verified**:
- No data races on queue state
- No lost requests (all submitted requests eventually complete)
- Queue invariants maintained (head <= tail, bounded size)
- No double-completion of requests
- Mutex safety (no unlock without lock)

**Configuration**:
- Queue size: 4 slots
- Producers: 2 threads
- Workers: 2 threads
- Unwind bound: 15

**Results** (N=1254):
```
** 0 of 282 failed (1 iterations)
VERIFICATION SUCCESSFUL
```

### 2. Stream Pool (`stream_pool_c_model.c`)

Models the `MPSStreamPool` TLS binding mechanism:

**Properties Verified**:
- No two threads share the same stream
- Stream lifecycle correct (alloc -> use -> release)
- TLS binding consistency (TLS matches pool state)
- Fork safety (TLS invalidation on PID change)
- Stream count conservation (freelist + bound = total)

**Configuration**:
- Stream pool size: 4 streams
- Threads: 3
- Includes fork simulation

**Results** (N=1254):
```
** 0 of 100 failed (1 iterations)
VERIFICATION SUCCESSFUL
```

### 3. Allocator (`allocator_c_model.c`)

Models the `MPSHeapAllocator` ABA detection mechanism:

**Properties Verified**:
- ABA counter monotonicity (counter only increases, never wraps)
- No buffer ID reuse (prevents ABA bugs)
- Buffer state machine correctness (free -> allocated -> in_use -> pending_free -> free)
- No use-after-free (free buffers have no stream associations)
- Pool invariants maintained (freelist + allocated + pending = total)
- Cross-stream buffer tracking (recordStream pattern)

**Configuration**:
- Buffer pool size: 4 buffers
- Threads: 2
- Streams: 2
- Includes pending buffer completion simulation

**Results** (N=1255):
```
** 0 of 524 failed (1 iterations)
VERIFICATION SUCCESSFUL
```

### 4. Event Pool (`event_c_model.c`)

Models the `MPSEventPool` callback survival mechanism:

**Properties Verified**:
- Event ID uniqueness (no two in-use events share same ID)
- Callback survival (events with pending callbacks cannot be released)
- Signal counter monotonicity (per-event counter only increases)
- Pool/InUse partition (events are in pool XOR in use)
- No use-after-release (pooled events have consistent state)
- Reference counting correctness (ref_count >= 1 for acquired events)

**Configuration**:
- Event pool size: 4 events
- Threads: 2
- Streams: 2
- Simulates: acquire, record, notify, signal, callback completion, release

**Results** (N=1256):
```
** 0 of 657 failed (1 iterations)
VERIFICATION SUCCESSFUL
```

## Summary

| Model | Properties | Status |
|-------|------------|--------|
| batch_queue | 282 | VERIFIED |
| stream_pool | 100 | VERIFIED |
| allocator | 524 | VERIFIED |
| event | 657 | VERIFIED |
| **Total** | **1563** | **ALL PASS** |

## Running Verification

### Prerequisites

```bash
# Install CBMC (macOS)
brew install cbmc

# Verify installation
cbmc --version  # Should be 6.8.0 or later
```

### Run Verification

```bash
cd verification/cbmc

# Batch queue model
cbmc batch_queue_c_model.c --unwind 15 --bounds-check --pointer-check

# Stream pool model
cbmc stream_pool_c_model.c --unwind 15 --bounds-check --pointer-check

# Allocator model (ABA detection)
cbmc allocator_c_model.c --unwind 15 --bounds-check --pointer-check

# Run all models
./run_cbmc.sh all

# With counterexample trace on failure
cbmc batch_queue_c_model.c --unwind 15 --trace
```

### Options

| Option | Description |
|--------|-------------|
| `--unwind N` | Loop unwinding bound (higher = more thorough, slower) |
| `--bounds-check` | Check array bounds |
| `--pointer-check` | Check pointer validity |
| `--div-by-zero-check` | Check for division by zero |
| `--trace` | Show execution trace on failure |
| `--slice-formula` | Optimize formula (faster) |

## Model Design Principles

### Why C Models Instead of C++?

CBMC 6.8 has limited support for modern C++ features:
- `std::atomic<T>` - partial support
- C++17 structured bindings - not supported
- Lambda expressions - limited support

Using C with CBMC's built-in atomic primitives provides:
- Complete control over memory model
- Deterministic verification
- Faster analysis

### Nondeterminism

The models use nondeterministic choices to simulate all possible thread interleavings:

```c
unsigned action = nondet_uint() % 4;
switch (action) {
    case 0: producer(&queue, 0); break;
    case 1: producer(&queue, 1); break;
    case 2: worker(&queue, 0); break;
    case 3: worker(&queue, 1); break;
}
```

CBMC explores ALL paths through these choices, finding any property violations.

### Bounds

CBMC uses bounded model checking - loops are unrolled up to `--unwind N` iterations. The "unwinding assertions" verify that the bound is sufficient. If an unwinding assertion fails, increase the bound.

## Relationship to TLA+ Specs

The CBMC models complement the TLA+ specifications in `specs/`:

| Aspect | TLA+ | CBMC |
|--------|------|------|
| Abstraction | High-level protocol | Low-level C implementation |
| Verification | Temporal logic (liveness) | Safety properties only |
| Memory model | Abstract | C11 memory model |
| Code | Specification language | Actual C code |

TLA+ verifies the algorithm is correct. CBMC verifies the implementation doesn't have bugs.

## Future Work

1. ~~**Allocator Model**: Add `allocator_c_model.c` for ABA detection verification~~ DONE (N=1255)
2. ~~**Event Model**: Add `event_c_model.c` for callback survival verification~~ DONE (N=1256)
3. **Increase Bounds**: Run with larger unwind bounds for higher confidence
4. **Property Coverage**: Add more assertions for edge cases

## Files

```
verification/cbmc/
├── README.md                     # This file
├── run_cbmc.sh                   # Run script for all models
├── batch_queue_c_model.c         # BatchQueue verification (282 properties)
├── stream_pool_c_model.c         # StreamPool TLS verification (100 properties)
├── allocator_c_model.c           # Allocator ABA detection verification (524 properties)
├── event_c_model.c               # Event callback survival verification (657 properties)
├── batch_queue_model.h           # C++ model (for reference)
└── batch_queue_harness.cpp       # C++ harness (for reference)
```

## References

- [CBMC User Manual](https://www.cprover.org/cbmc/doc/manual.html)
- [CBMC GitHub](https://github.com/diffblue/cbmc)
- [Formal Methods for MPS Plan](~/.claude/plans/whimsical-popping-beaver.md)
