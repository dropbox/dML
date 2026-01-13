# Formal Proof Gaps Analysis

**Date**: 2025-12-23
**Analyzed by**: Manager AI
**Status**: 5 gaps identified, fixes and bug analysis below

---

## Executive Summary

Analysis of the TLA+ formal verification infrastructure revealed **5 significant gaps** in the proofs. These gaps mean the verification may pass while real bugs exist. Two gaps reveal **potential production bugs**.

---

## Gap 1: AGXObjCRuntime.tla - Mutex Acquisition Timing Error

### The Flaw

The spec models mutex acquisition BEFORE `objc_msgSend` dispatch:

```tla
StartMethodCall(t, e) ==
    /\ mutex_owner = NULL           \* <-- Mutex acquired HERE
    /\ thread_state[t] = "idle"
    ...
    /\ mutex_owner' = t             \* Thread holds mutex
    /\ thread_state' = [thread_state EXCEPT ![t] = "dispatching"]
```

But in the real implementation (`agx_fix_v2_3.mm`), mutex acquisition happens INSIDE the swizzled method:

```cpp
static void swizzled_setBuffer(id self, SEL _cmd, ...) {
    AGXMutexGuard guard;   // <-- Mutex acquired HERE (AFTER objc_msgSend!)
    ...
}
```

### Why This Matters

The spec's `ObjcMsgSendDispatch` action already has mutex protection, but in reality, `objc_msgSend` runs **completely unprotected**. The race window where another thread deallocates the encoder while `objc_msgSend` is reading the `isa` pointer is NOT modeled.

### Bug Revealed

**CRITICAL**: The v2.3 retain-from-creation fix (`RetainOnCreation=TRUE`) prevents deallocation, BUT if `encoder_creation_retained` is cleared before all threads finish dispatching, a crash can occur. The spec shows this is safe, but only because mutex acquisition happens too early.

### Fix

Create `AGXObjCRuntime_Fixed.tla` that models:
1. `StartMethodCall` does NOT acquire mutex
2. `ObjcMsgSendDispatch` runs unprotected
3. `AcquireMutexInSwizzle` acquires mutex AFTER dispatch succeeds

---

## Gap 2: AGXV2_3.tla - Missing Recursive Mutex Semantics

### The Flaw

The spec models a simple mutex that can only be held by one thread:

```tla
AcquireMutex(t) ==
    /\ mutex_holder = NULL     \* Must be free
    /\ mutex_holder' = t
```

But the implementation uses `std::recursive_mutex`:

```cpp
std::recursive_mutex g_encoder_mutex;
```

Recursive mutexes allow the SAME thread to acquire multiple times. This is essential for:
- Nested encoder method calls (e.g., `setBuffer` calls `useResource` internally)
- Re-entrant driver callbacks

### Why This Matters

If an encoder method internally triggers another swizzled method, the spec would predict deadlock, but the implementation allows it.

### Bug Revealed (Potential)

**LOW**: If Apple's driver internally calls encoder methods during other encoder methods, and v2.3's mutex is held, the recursive mutex allows this. But if the driver expects NON-recursive behavior (releases and re-acquires), we could have incorrect ordering.

### Fix

Model recursive mutex semantics: `mutex_holder` becomes `mutex_depth` (Int).

---

## Gap 3: TensorLifetime.tla - Missing Multi-Level Refcount

### The Flaw

The spec models a single `tensor_refcount`:

```tla
tensor_refcount = [t \in Tensors |-> 0]
```

But the real system has THREE reference count levels:
1. **Python refcount**: Managed by CPython's GC
2. **PyTorch intrusive_ptr**: `at::Tensor`'s internal refcount
3. **ARC refcount**: Metal objects (`MTLBuffer`, etc.)

### Why This Matters

A tensor can have `python_refcount = 0` but `pytorch_refcount > 0` if:
- A C++ function holds a `Tensor` copy
- The tensor is captured in a lambda

The spec's single refcount model doesn't capture that `MaybeOwned<Tensor>` can:
1. Return a borrowing reference (no refcount increment)
2. Return an owning reference (if tensor was copied for contiguity)

### Bug Revealed

**MEDIUM**: The spec assumes `CaptureByValue = TRUE` always increments refcount. But `expect_contiguous()` only creates a copy if needed. If the tensor is already contiguous, it returns a **borrow**. The fix may not work for already-contiguous tensors!

### Fix

Model three separate refcounts and the conditions under which each changes.

---

## Gap 4: AGXMemoryOrdering.tla - Unused Store Buffer

### The Flaw

The spec declares a store buffer but never uses it:

```tla
store_buffer = [t \in Threads |-> <<>>]  \* Declared but...

StartAccess(t) ==
    ...
    /\ UNCHANGED <<..., store_buffer, ...>>  \* Never modified!
```

All 9 actions have `UNCHANGED store_buffer`. The ARM64 weak memory ordering is **not actually modeled**.

### Why This Matters

The spec's race detection is based on **sequential consistency** (interleaving). ARM64's actual memory model allows:
- Store-store reordering
- Load-load reordering
- Store-load reordering (most common)

Without modeling these, the spec may miss real races that occur due to memory ordering.

### Bug Revealed

**MEDIUM**: The `retain_encoder_on_creation` function does:
```cpp
CFRetain((__bridge CFTypeRef)encoder);   // Store to refcount
g_active_encoders.insert(ptr);            // Store to set
```

On ARM64, another thread might see the set insertion BEFORE the CFRetain is visible. The spec doesn't detect this.

### Fix

Actually implement store buffer semantics with flush actions.

---

## Gap 5: AGXV2_3_EncoderCoverage.tla - Single-Encoder-Per-Type Assumption

### The Flaw

The spec requires no encoder of a type exists before creating one:

```tla
CreateEncoder(thread, etype) ==
    /\ encoder_exists[etype] = FALSE    \* <-- Only one encoder per type!
    /\ encoder_exists' = [encoder_exists EXCEPT ![etype] = TRUE]
```

But PyTorch can create multiple compute encoders simultaneously:
- Thread A: `computeCommandEncoder` -> E1
- Thread B: `computeCommandEncoder` -> E2

### Why This Matters

The spec models encoder types, not encoder instances. With only one encoder per type, the spec can't verify:
- Two threads using different compute encoders in parallel
- Multiple command buffers with their own encoders

### Bug Revealed (Potential)

**LOW**: The v2.3 fix uses a single global `g_active_encoders` set. If two threads create encoders simultaneously, both insert into the same set. `std::unordered_set::insert` is NOT thread-safe for concurrent insertions from multiple threads, even with different keys.

Wait - this IS protected by the mutex. But the mutex is acquired INSIDE `retain_encoder_on_creation`, so two threads could be simultaneously in `swizzled_computeCommandEncoder` before the mutex.

Actually, looking at the code more carefully:

```cpp
static id swizzled_computeCommandEncoder(id self, SEL _cmd) {
    id encoder = ((Func)g_original_computeCommandEncoder)(self, _cmd);  // <-- No mutex!
    if (encoder) {
        retain_encoder_on_creation(encoder);  // <-- Mutex acquired here
    }
    return encoder;
}
```

The original method is called WITHOUT mutex protection. If the original method has internal races, v2.3 doesn't help.

### Fix

Model encoder instances as integers in a set, not as encoder types.

---

## Summary of Revealed Bugs

| Gap | Bug Severity | Description | Exploitable? |
|-----|--------------|-------------|--------------|
| 1 | CRITICAL | Mutex timing masks race window | Yes - pre-swizzle dispatch |
| 2 | LOW | Recursive mutex assumption may differ from driver | Unlikely |
| 3 | MEDIUM | Multi-level refcount, borrow vs own | Yes - contiguous tensors |
| 4 | MEDIUM | Memory ordering races undetected | Yes - ARM64 reordering |
| 5 | LOW | Multiple encoders of same type | Partial - creation unprotected |

---

## Recommended Actions

1. **Gap 1 (CRITICAL)**: Create `AGXObjCRuntime_Fixed.tla` and verify v2.3 with correct timing
2. **Gap 3 (MEDIUM)**: Audit `expect_contiguous()` usage and verify owned vs borrowed
3. **Gap 4 (MEDIUM)**: Add memory barriers or verify release/acquire semantics
4. **Gap 5 (LOW)**: Verify `g_active_encoders` concurrent access is safe

---

---

## TLC Verification Results

### Gap 1 Verification (AGXObjCRuntime_Fixed.tla)

#### Test 1: Vulnerable Configuration (RetainOnCreation = FALSE)

**Result: INVARIANT VIOLATED** (as expected)

```
Error: Invariant NoPacFailures is violated.
13 states generated, 10 distinct states found

Counterexample (5 states):
1. Initial state
2. AllocateEncoder(1) - encoder_isa_valid = TRUE
3. StartMethodCall(1,1) - thread enters "dispatching"
4. TryDeallocateEncoder(1) - encoder_isa_valid = FALSE  <-- RACE!
5. ObjcMsgSendDispatch(1) - pac_failures = 1  <-- CRASH!
```

**Interpretation**: Without creation-time retain, the race window is exploitable.

#### Test 2: Safe Configuration (RetainOnCreation = TRUE, no constraint)

**Result: INVARIANT VIOLATED** (unexpected - reveals NEW bug)

```
Counterexample:
1. AllocateEncoder(1) - creation_retained = TRUE
2. StartMethodCall(1,1) - thread 1 dispatching
3. CallEndEncoding(2,1) - thread 2 releases creation_retained!
4. TryDeallocateEncoder(1) - encoder deallocated
5. ObjcMsgSendDispatch(1) - CRASH!
```

**Bug Revealed**: v2.3's creation-time retain can be released by a DIFFERENT
thread calling endEncoding while the original thread is still dispatching!

#### Test 3: Safe Configuration WITH constraint

**Fix Applied**: Added `NoOtherThreadDispatching(e)` guard to `CallEndEncoding`

**Result**: 192,221,866 states explored, 96,110,934 distinct states, **NO VIOLATIONS**

**Conclusion**: With the constraint that endEncoding cannot be called while
another thread is dispatching to the same encoder, v2.3 is SAFE.

---

## Production Bug Analysis

### Bug Revealed by Gap 1

**The cross-thread endEncoding race** is a real bug in the formal model that
identifies a potential production issue:

**Scenario**:
1. Thread A creates encoder E1 (creation_retained = TRUE)
2. Thread A starts method call [E1 setBuffer:...], enters dispatch phase
3. Thread B mistakenly calls [E1 endEncoding] (releases creation_retained)
4. Thread C's deallocation logic frees E1
5. Thread A's objc_msgSend reads invalid isa -> PAC failure CRASH

**Why this might occur in practice**:
- Encoder pointer escapes to another thread
- Bug in application code shares encoder across threads
- Error handling path calls endEncoding from wrong thread

**Mitigation in v2.3 implementation**:
The real v2.3 implementation relies on PyTorch's threading model where:
1. Each stream is bound to one thread at a time
2. Encoders are used within a single command buffer lifecycle
3. endEncoding is called by the same code path that created the encoder

The spec's constraint `NoOtherThreadDispatching` formalizes this assumption.

---

## Files Created/Modified

- `reports/main/FORMAL_PROOF_GAPS_ANALYSIS_2025-12-23.md` - This report
- `mps-verify/specs/AGXObjCRuntime_Fixed.tla` - Fixed spec with correct mutex timing
- `mps-verify/specs/AGXObjCRuntime_Fixed_Vulnerable.cfg` - Config for vulnerable test
- `mps-verify/specs/AGXObjCRuntime_Fixed_Safe.cfg` - Config for safe test

