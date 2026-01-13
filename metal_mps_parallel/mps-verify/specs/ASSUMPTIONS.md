# TLA+ Spec Assumptions (Apple Silicon / ARM64)

This document enumerates the **explicit modeling assumptions** used across the TLA+ specs in `mps-verify/specs/`.
These assumptions matter because TLA+ explores **logical interleavings** under a simplified execution model, while the
real implementation runs on **ARM64 + Objective-C + Metal/MPS** with additional behaviors (weak memory ordering, async GPU
execution, private locks, etc.).

For assumptions about **Apple's closed-source frameworks** (Metal/MPS/objc runtime), see:
- `mps-verify/assumptions/apple_framework_assumptions.md`

For runtime verification of key platform assumptions on the current machine, run:
- `verification/run_platform_checks` (includes memory-order checks like Dekker + release/acquire message passing)

---

## What The Specs Model (High-Level)

Most specs model:
- A finite set of threads performing operations
- A shared state machine representing encoder/command-buffer lifecycle
- A mutex/critical section to serialize operations (where applicable)
- Safety properties like “no use-after-free”, “no commit race window”, “no timeout escape hatch”

The specs generally treat each action as an **atomic transition**.

---

## Key Assumptions

### A1: Atomic Spec Transitions
**Assumption**: Each TLA+ action executes atomically and sees a consistent global state.

**Implementation mapping**: We approximate this using a global mutex (`g_encoder_mutex`) to serialize state transitions in
userspace (plus `std::atomic<>` counters for stats).

### A2: Mutex Synchronizes Memory (Acquire/Release)
**Assumption**: Entering/leaving the critical section establishes a happens-before edge so state written by one thread is
visible to another thread after it acquires the same mutex.

**Implementation mapping**: `std::mutex` / `std::recursive_*mutex` lock/unlock have acquire/release semantics per the C++
memory model. This is treated as a required platform property; see `verification/run_platform_checks`.

### A3: `std::atomic` Honors Requested Memory Order
**Assumption**: `std::atomic<T>` operations behave according to the C++ memory model on Apple Silicon (including
`memory_order_seq_cst` total order and release/acquire synchronization).

**Implementation mapping**: We rely on `std::atomic` semantics for cross-thread counters/flags and for runtime assumption
tests; see `verification/run_platform_checks`.

### A4: No Torn Reads/Writes For Pointer-Sized Aligned Values
**Assumption**: Aligned word-sized loads/stores (pointers, `uint64_t`) are not torn.

**Implementation mapping**: Apple Silicon ARM64 provides naturally-aligned word access without tearing; additionally,
critical pointer state is protected by the mutex in the userspace fix.

### A5: Fairness Is Not Guaranteed
**Assumption**: The specs do not assume fairness or bounded scheduling delays unless explicitly modeled.

**Reality**: macOS scheduling can delay threads, and GPU completion is asynchronous; the fix must not rely on fairness.

---

## Behaviors Intentionally Not Modeled

These are real behaviors that may exist but are out of scope for most specs:
- **Weak-memory reorderings** in code paths not protected by mutex/atomics
- **Priority inversion / QoS effects / preemption timing**
- **Objective-C runtime internals** (IMP caching, message-send dispatch windows)
- **Metal/MPS internal locks** and possible lock-inversion deadlocks
- **GPU asynchronous execution details** (command buffer completion timing, firmware scheduling)
- **Crashes/hangs inside AGX driver** (modeled only indirectly via safety properties)

---

## Practical Guidance

- Treat TLA+ results as proving properties of the **modeled protocol**, not the entire hardware+driver stack.
- When a spec relies on atomicity or ordering, prefer a corresponding runtime check in `verification/run_platform_checks`
  and document the mapping (mutex / atomic ordering) in the spec header.

