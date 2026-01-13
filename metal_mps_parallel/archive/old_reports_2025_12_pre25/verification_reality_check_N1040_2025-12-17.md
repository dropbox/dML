# Verification Infrastructure Reality Check N=1040

**Date**: 2025-12-17 10:15 PST
**Purpose**: Honest assessment of what formal verification provides and what's actually working

## Executive Summary

The formal verification infrastructure provides **design insight** but is not yet a **reliable gate**. Key findings:

1. TLA+ specs exist and reveal bottlenecks, but CLI discovery is broken
2. Lean builds with 1 sorry (unproven conjecture)
3. CBMC passes but uses simplified models
4. Static analysis annotations exist but aren't enforced in build

## What TLA+ Actually Tells Us

### 1. ABA Double-Check Pattern Is Correct But Expensive

The MPSAllocator.tla spec models the exact pattern used in getSharedBufferPtr:

```
getPtr requires 3 lock acquisitions:
  m_mutex (capture) → pool_mutex → m_mutex (verify)

This pattern exists to PREVENT DEADLOCK:
  - Allocation uses: pool_mutex → m_mutex
  - If getPtr used: m_mutex → pool_mutex continuously
  - That would create lock-order inversion → deadlock risk

The double-check pattern breaks the lock cycle by releasing m_mutex
before acquiring pool_mutex. Safe, but causes 2x m_mutex contention.
```

### 2. Safety Properties Verified (15.8M states)

| Property | Status | Meaning |
|----------|--------|---------|
| ABADetectionSound | PASS | use_count correctly detects buffer reuse |
| NoDoubleFree | PASS | Buffers can't be freed twice |
| NoUseAfterFree | PASS | Freed buffers aren't accessed |
| MutexExclusivity | PASS | No two threads hold same mutex |
| BufferConsistency | PASS | in_use=TRUE implies allocated |

### 3. Scalability Properties (Informational, Not Invariants)

| Property | Status | Meaning |
|----------|--------|---------|
| ParallelLockHolding | DOCUMENTED | Only 1 thread in critical section at a time |
| GlobalSerializerViolation | DOCUMENTED | m_mutex serializes all operations |
| ExcessiveLocking | DOCUMENTED | getPtr: 3 locks, alloc: 2 locks, free: 1 lock |

## What Sharding Fixed (and Didn't Fix)

### Fixed: m_mutex Contention in Allocator
- Split `m_allocated_buffers` into 8 shards
- Each shard has own mutex
- Result: 29.3% → 30.6% efficiency (+1.3%)

### NOT Fixed: Other Global Mutexes
The benchmark workloads (nn.Linear, TransformerEncoderLayer) likely hit:

```cpp
// pytorch-mps-fork/aten/src/ATen/native/mps/operations/Linear.mm:19
static std::mutex s_linear_nograph_mutex;

// pytorch-mps-fork/aten/src/ATen/native/mps/operations/Normalization.mm:31
static std::mutex s_layer_norm_mutex;

// pytorch-mps-fork/aten/src/ATen/native/mps/operations/LinearAlgebra.mm:49
static std::mutex s_bmm_tiled_mutex;

// pytorch-mps-fork/aten/src/ATen/native/mps/OperationUtils.mm:484
static std::mutex s_ndarray_identity_mutex;
```

These are **operation-level global mutexes** that serialize ALL threads doing the same operation type!

## Verification Infrastructure Gaps

### 1. TLC Discovery Broken
`mpsverify tla --all` fails because it doesn't find `tools/tla2tools.jar`.

**Fix needed**: Update `TLCRunner.lean:62` to search `tools/` subdirectory.

### 2. Lean Has Unproven Conjecture
```lean
-- mps-verify/MPSVerify/Core/MemoryModel.lean:238
theorem seq_cst_race_free_conjecture : ... := by sorry
```

**Fix needed**: Either prove it or move to a separate "conjectures" module with explicit warning.

### 3. CBMC Uses Simplified Models
The CBMC harnesses verify simplified C models, not the actual ObjC++ code.

**Value**: Still catches logic errors in the verification model itself.

### 4. Static Analysis Not Enforced
TSA annotations exist but `mpsverify static` doesn't actually run Clang with compile_commands.json.

**Fix needed**: Wire compile-db-driven TSA into the CLI.

## Recommended Next Steps

### Priority 1: Attribute the Real Bottleneck
Before more code changes, instrument to measure:
1. Time spent in global operation mutexes (s_linear_nograph_mutex, etc.)
2. Time spent in allocator m_mutex (now sharded)
3. Time spent in stream queue waits
4. Time spent in PSO cache misses/compiles

### Priority 2: Fix Verification Infrastructure
1. Fix TLC jar discovery in TLCRunner.lean
2. Address or isolate the sorry in MemoryModel.lean
3. Wire compile-db TSA into CLI

### Priority 3: Fix Actual Bottleneck
Once attribution is complete:
- If global op mutexes dominate: Shard them or use lock-free paths
- If stream queue waits dominate: Investigate GCD overhead
- If PSO cache misses dominate: Investigate shader compilation patterns

## Key Insight

**The allocator sharding was the RIGHT fix for the allocator, but the allocator may not be the bottleneck.** The 1-2% improvement suggests other serialization points (likely the global operation mutexes) are more significant.

## Files Referenced

- `mps-verify/specs/MPSAllocator.tla` - TLA+ spec with scalability properties
- `mps-verify/MPSVerify/Bridges/TLCRunner.lean` - Broken TLC discovery
- `mps-verify/MPSVerify/Core/MemoryModel.lean:238` - sorry conjecture
- `pytorch-mps-fork/aten/src/ATen/native/mps/operations/Linear.mm` - s_linear_nograph_mutex
- `pytorch-mps-fork/aten/src/ATen/native/mps/operations/Normalization.mm` - s_layer_norm_mutex
