# Benefits and Limits of Static Analysis & Formal Verification

**Project**: MPS Parallel Inference
**Date**: 2025-12-19
**Iterations**: N=0 through N=1305
**Author**: AI Workers under Manager direction

---

## Executive Summary

This project built one of the most comprehensive formal verification platforms for a concurrent systems project:

| Tool | Artifacts | Results |
|------|-----------|---------|
| TLA+ | 10 specifications | 15.3M+ states verified |
| CBMC | 10 harnesses | 3,856 assertions, 0 failures |
| Iris/Coq | 6 proof modules | All complete, no Admitted |
| Clang TSA | Full codebase | 210→92 warnings |
| Structural | 31 checks | 26 pass, 5 warnings |

**Yet a critical SIGABRT crash (N=1305) escaped all verification.**

This report analyzes what formal verification can and cannot do, using concrete examples from this project.

---

## Part 1: What Formal Verification Accomplished

### 1.1 Bugs Found by Formal Methods

| Bug | Found By | Severity | Status |
|-----|----------|----------|--------|
| MPSEvent callback UAF | TSA + Structural | CRITICAL | Fixed N=1275 |
| TSA lock violations in MPSStream | Clang TSA | HIGH | Fixed N=1275-1277 |
| ABA counter wrap risk | TLA+ model | MEDIUM | Documented |
| Missing lock in commitAndWait | Clang TSA | HIGH | Fixed N=1275 |
| Potential deadlock in graph encoding | TLA+ | HIGH | Fixed (encoding lock) |

### 1.2 Properties Successfully Proved

#### TLA+ Model Checking
```
✅ Deadlock freedom (bounded: 8 threads, 32 streams)
✅ No use-after-free in stream pool lifecycle
✅ ABA counter monotonicity
✅ TLS binding uniqueness (no two threads share stream)
✅ Fork handler invalidates TLS correctly
✅ Batch queue: no stuck futures, stop drains queue
```

#### CBMC Bounded Model Checking
```
✅ No double-free in allocator
✅ No buffer overrun in TLS cache
✅ ABA detection catches all recycled pointers
✅ Pool shutdown prevents new allocations
✅ Fork safety: pool_alive guards checked
```

#### Iris/Coq Separation Logic
```
✅ Mutex mutual exclusion (mutex_token_exclusive)
✅ Lock acquisition terminates (Löb induction)
✅ Release returns resources to invariant
✅ TLS slots are exclusive per thread
✅ Callback tokens track pending callbacks
✅ ABA generation agreement (gen_agree, gen_update)
```

### 1.3 Verification Artifacts Produced

```
specs/
├── MPSStreamPool.tla          # Stream lifecycle
├── MPSAllocator.tla           # Buffer allocation + ABA
├── MPSEvent.tla               # Event + callback survival
├── MPSBatchQueue.tla          # Producer-consumer queue
├── MPSStreamPoolBoundedWait.tla
├── MPSStreamPoolParallel.tla
├── MPSRecordStream.tla
├── MPSEncodingLock.tla
├── MPSStreamSlotAllocator.tla
└── MPSDispatchQueueContext.tla

verification/iris/theories/
├── mutex.v                    # Spin lock proofs
├── tls.v                      # TLS binding proofs
├── callback.v                 # Callback lifetime
├── aba.v                      # ABA detection
├── stream_pool.v              # Combined safety
└── prelude.v                  # Definitions

mps-verify/verification/cbmc/harnesses/
├── aba_detection_harness.c
├── alloc_free_harness.c
├── tls_cache_harness.c
├── stream_pool_harness.c
├── event_pool_harness.c
├── batch_queue_harness.c
├── command_buffer_harness.c
├── graph_cache_harness.c
├── tls_binding_harness.c
└── fork_safety_harness.c
```

---

## Part 2: What Formal Verification Missed

### 2.1 The N=1305 SIGABRT Crash

**Crash stack:**
```
HeapAllocator::free()
→ MPSStream::addCompletedHandler()
→ [_MTLCommandBuffer addCompletedHandler:]
→ MTLReportFailure → abort() → SIGABRT
```

**The bug (MPSStream.mm:294):**
```cpp
// BUG: Falls back to _prevCommandBuffer which is ALREADY COMMITTED
MPSCommandBuffer* cb = _commandBuffer ? _commandBuffer : _prevCommandBuffer;
```

**Why it crashed:**
- `_prevCommandBuffer` points to a command buffer that was already committed
- Metal's `addCompletedHandler:` can ONLY be called BEFORE commit
- Using a committed buffer causes Metal to assert and abort

**Why formal verification missed it:**

| Tool | Why It Missed |
|------|---------------|
| TLA+ | Models state transitions, not Apple API contracts |
| CBMC | Checks our C code, not Objective-C framework behavior |
| Iris/Coq | Proves properties of our abstractions, not external APIs |
| Clang TSA | Checks lock discipline, not API usage patterns |

### 2.2 Categories of Bugs That Escape Formal Verification

#### Category A: External API Constraints

We model OUR code. External APIs have constraints we don't model.

| API Constraint | Framework | Consequence if Violated |
|----------------|-----------|------------------------|
| `addCompletedHandler` before commit | Metal | SIGABRT |
| `waitUntilCompleted` after commit | Metal | Undefined behavior |
| `encodeToCommandBuffer` thread affinity | MPSGraph | Corruption |
| `@autoreleasepool` for Objective-C objects | Foundation | Memory leak |

**Lesson**: Document external API constraints in assumption ledger.

#### Category B: Platform-Specific Behavior

Our proofs assume uniform platform behavior. Reality differs.

| Platform Variable | Risk |
|-------------------|------|
| GPU core count (8 vs 76) | Different contention patterns |
| Memory bandwidth | Different race windows |
| Dynamic Caching (M3+) | Unknown interactions |
| macOS version | Metal framework changes |
| AGX firmware | Chip-specific scheduling |

**Lesson**: Test on multiple platforms, document untested configurations.

#### Category C: Timing-Dependent Bugs

TLA+ and CBMC explore state space, but don't model real timing.

```
Thread A                    Thread B
─────────                   ─────────
read ptr
                            free ptr
                            alloc new obj at ptr
use ptr (stale!)
```

Our proofs show this CAN'T happen with proper synchronization.
But they don't prove our synchronization is FAST ENOUGH.

**Lesson**: Stress tests and long-running soak tests are essential.

#### Category D: Resource Exhaustion

Formal models typically assume infinite resources.

| Resource | What If Exhausted |
|----------|-------------------|
| Stream pool (32 slots) | Graceful degradation? Crash? |
| GPU memory | OOM handling correct? |
| File descriptors | Leak detection? |
| Thread stack | Stack overflow? |

**Lesson**: Explicit resource exhaustion tests required.

#### Category E: Numerical/Floating-Point Issues

Formal verification of concurrent code doesn't cover:
- Floating-point precision
- Numerical stability
- NaN/Inf propagation

**Lesson**: Numerical correctness requires separate testing.

---

## Part 3: The Fundamental Limits

### 3.1 The Verification Boundary

```
┌─────────────────────────────────────────────────────────────┐
│                    What We Verify                            │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Our C++/Objective-C++ Code                          │    │
│  │  - State machines (TLA+)                             │    │
│  │  - Memory safety (CBMC)                              │    │
│  │  - Lock discipline (TSA, Iris)                       │    │
│  │  - Data structure invariants                         │    │
│  └─────────────────────────────────────────────────────┘    │
│                           │                                  │
│                           │ calls                            │
│                           ▼                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  UNVERIFIED: External Dependencies                   │    │
│  │  - Apple Metal framework                             │    │
│  │  - Apple MPS framework                               │    │
│  │  - AGX GPU driver                                    │    │
│  │  - macOS kernel                                      │    │
│  │  - libc++, libdispatch                               │    │
│  │  - Hardware behavior                                 │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 What Each Tool Actually Checks

| Tool | Actually Checks | Does NOT Check |
|------|-----------------|----------------|
| **TLA+** | State machine properties (safety, liveness) | Implementation matches model |
| **CBMC** | C code paths up to bound | Unbounded execution, external calls |
| **Iris/Coq** | Mathematical properties of abstractions | That code matches abstraction |
| **Clang TSA** | Lock acquisition patterns in annotations | That annotations are correct |
| **Structural** | Code patterns via regex | Semantic correctness |

### 3.3 The Assumption Stack

Every formal proof rests on assumptions:

```
Level 5: Our Properties (what we prove)
    ↓ assumes
Level 4: Our Code Model (TLA+, Lean, CBMC harnesses)
    ↓ assumes
Level 3: Language Semantics (C++ memory model, Obj-C ARC)
    ↓ assumes
Level 2: Framework Behavior (Metal, MPS, GCD)
    ↓ assumes
Level 1: OS/Driver Behavior (macOS, AGX)
    ↓ assumes
Level 0: Hardware Behavior (Apple Silicon)
```

**We only verify Level 5. Levels 0-4 are ASSUMED.**

---

## Part 4: Recommendations

### 4.1 For This Project

| Gap | Recommendation | Priority |
|-----|----------------|----------|
| Apple API constraints | Document in AF.* assumptions | HIGH |
| Platform variations | Test on M2, M3, M4; document gaps | HIGH |
| Timing-dependent bugs | 1-hour soak test, TSan CI | HIGH |
| Resource exhaustion | Pool exhaustion, OOM tests | MEDIUM |
| External API modeling | Consider adding to TLA+ specs | LOW |

### 4.2 For Future Projects

1. **Document all external API constraints BEFORE coding**
   - Create assumption ledger at project start
   - Review API documentation for pre/post conditions
   - Add runtime assertions for critical constraints

2. **Combine formal methods with empirical testing**
   - Formal: Proves properties of your model
   - Testing: Validates model matches reality
   - Neither alone is sufficient

3. **Model external APIs when critical**
   ```tla
   \* Model Metal command buffer state machine
   CommandBufferState == {"created", "encoding", "committed", "completed"}

   \* Constraint: addCompletedHandler only in created/encoding
   AddCompletedHandler(cb) ==
       /\ cb.state \in {"created", "encoding"}
       /\ cb.handlers' = cb.handlers \cup {new_handler}
   ```

4. **Use defense in depth**
   - Formal proofs for core properties
   - Static analysis for code quality
   - Runtime assertions for API constraints
   - Stress tests for timing issues
   - Fuzzing for unexpected inputs

### 4.3 Verification Checklist for Concurrent Systems

```
□ Model core state machines in TLA+
□ Prove deadlock freedom
□ Prove key safety invariants
□ Add CBMC harnesses for memory safety
□ Add Clang TSA annotations
□ Document ALL external API constraints
□ Document platform assumptions
□ Run sanitizers (TSan, ASan, UBSan)
□ Create stress tests / soak tests
□ Test on multiple platforms
□ Test resource exhaustion scenarios
□ Review assumption ledger quarterly
```

---

## Part 5: Case Studies from This Project

### Case Study 1: Bug Found - MPSEvent Callback UAF

**Discovery**: Structural check ST.003 flagged lambda capturing `this`

**Analysis**:
```cpp
notifyLocked(^(id<MTLSharedEvent>, uint64_t) {
    notifyCpuSync(getTime());  // Captures 'this' implicitly!
});
```

If `~MPSEvent()` runs before callback fires → use-after-free.

**Fix**: Added `m_pending_callbacks` tracking, destructor waits.

**Why formal methods found it**: Pattern matching on known-dangerous code patterns.

### Case Study 2: Bug Found - TSA Lock Violations

**Discovery**: Clang TSA reported:
```
warning: reading variable '_commandBuffer' requires holding mutex '_streamMutex'
```

**Analysis**: `commitAndWait()` accessed protected members without lock.

**Fix**: Added lock acquisition to all accessor functions.

**Why formal methods found it**: TSA tracks lock state through control flow.

### Case Study 3: Bug Missed - addCompletedHandler Crash

**Discovery**: User crash report (SIGABRT in production)

**Analysis**: API constraint violation, not a concurrency bug.

**Root cause**: `_prevCommandBuffer` is committed, can't add handlers.

**Fix**: Never use `_prevCommandBuffer` for adding handlers.

**Why formal methods missed it**: External API constraint not modeled.

### Case Study 4: Bug Missed (Hypothetical) - M1 Ultra Memory Coherency

**Scenario**: Code works on M4 Max, fails on M1 Ultra.

**Potential cause**: UltraFusion interconnect has different memory coherency.

**Why formal methods would miss it**: Hardware behavior not modeled.

**Mitigation**: Platform matrix testing, document untested configs.

---

## Part 6: Quantitative Summary

### Verification Investment

| Activity | AI Commits | Human Hours (est.) |
|----------|------------|-------------------|
| TLA+ specifications | ~50 | ~100 |
| CBMC harnesses | ~30 | ~60 |
| Iris/Coq proofs | ~40 | ~80 |
| Clang TSA annotations | ~20 | ~40 |
| Structural checks | ~10 | ~20 |
| **Total** | **~150** | **~300** |

### Bug Detection Summary

| Source | Bugs Found | Bugs Missed |
|--------|------------|-------------|
| Formal methods | 5 | 1+ |
| Runtime crashes | 1 | ? |
| Code review | ? | ? |
| User reports | 1 | ? |

### Confidence Assessment

| Property | Confidence | Basis |
|----------|------------|-------|
| No deadlock (bounded) | 99% | TLA+ 15.3M states |
| No double-free | 99% | CBMC 3,856 checks |
| ABA detection correct | 99% | Iris proof + CBMC |
| TLS binding unique | 99% | TLA+ + Iris |
| No API constraint violations | 70% | Only documented constraints |
| Works on all platforms | 60% | Only tested M2/M3/M4 |
| No timing-dependent bugs | 80% | Tests pass, but no soak test |

---

## Conclusion

Formal verification is **powerful but bounded**. It proves properties of your **model**, not your **system**.

### The Value
- Found 5 real bugs before deployment
- Provides mathematical confidence in core properties
- Forces rigorous thinking about concurrency
- Documentation artifact for future maintainers

### The Limits
- Cannot verify external API constraints
- Cannot verify platform-specific behavior
- Cannot verify timing-dependent properties
- Model may not match implementation

### The Synthesis

**Formal methods + empirical testing + assumption documentation = production confidence**

None alone is sufficient. Together, they provide defense in depth.

---

---

## Part 7: Encoding API Constraints for Static Analysis

The N=1305 crash raises a critical question: **How can we make static analysis aware of external API constraints?**

### 7.1 Approaches to API Constraint Encoding

#### Approach A: Custom Clang Checker (Recommended)

Create a Clang Static Analyzer checker that tracks command buffer state:

```cpp
// MPSCommandBufferChecker.cpp - Custom Clang checker

class MPSCommandBufferChecker : public Checker<check::PreCall, check::PostCall> {
  // Track command buffer state: Created → Encoding → Committed → Completed
  enum BufferState { Created, Encoding, Committed, Completed };

  void checkPreCall(const CallEvent &Call, CheckerContext &C) const {
    // Detect addCompletedHandler calls
    if (Call.getCalleeIdentifier()->getName() == "addCompletedHandler") {
      auto *Buffer = getCommandBuffer(Call);
      auto State = C.getState()->get<BufferStateMap>(Buffer);

      if (State == Committed || State == Completed) {
        // BUG: addCompletedHandler on committed buffer!
        reportBug("addCompletedHandler called on committed MTLCommandBuffer. "
                  "This API can only be called before commit.", C);
      }
    }
  }

  void checkPostCall(const CallEvent &Call, CheckerContext &C) const {
    if (Call.getCalleeIdentifier()->getName() == "commit") {
      auto *Buffer = getCommandBuffer(Call);
      // Transition: * → Committed
      C.addTransition(C.getState()->set<BufferStateMap>(Buffer, Committed));
    }
  }
};
```

**Pros**: Deep integration with compiler, precise analysis
**Cons**: Requires Clang plugin infrastructure, maintenance burden

#### Approach B: Type-State Annotations

Encode state machine in types using phantom types or linear types:

```cpp
// Type-state encoding (C++ approximation)
template<typename State> class MTLCommandBuffer;

struct StateCreated {};
struct StateEncoding {};
struct StateCommitted {};

// Only Created and Encoding states have addCompletedHandler
template<>
class MTLCommandBuffer<StateCreated> {
public:
    void addCompletedHandler(Handler h);
    MTLCommandBuffer<StateEncoding> beginEncoding();
};

template<>
class MTLCommandBuffer<StateEncoding> {
public:
    void addCompletedHandler(Handler h);
    MTLCommandBuffer<StateCommitted> commit();
};

template<>
class MTLCommandBuffer<StateCommitted> {
    // NO addCompletedHandler method - compile error if called!
public:
    void waitUntilCompleted();
};
```

**Pros**: Compile-time enforcement, zero runtime cost
**Cons**: Requires wrapper types, doesn't work with Objective-C APIs

#### Approach C: Custom Attributes + clang-tidy

Define custom attributes and clang-tidy rules:

```cpp
// Header: MTLCommandBuffer+Constraints.h

// Custom attribute (requires Clang plugin or pragma)
#define MPS_REQUIRES_STATE(state) __attribute__((annotate("mps_requires_" #state)))
#define MPS_TRANSITIONS_TO(state) __attribute__((annotate("mps_transitions_" #state)))

@interface MTLCommandBuffer (StateConstraints)

// Requires: Created or Encoding
// Postcondition: State unchanged
- (void)addCompletedHandler:(MTLCommandBufferHandler)block
    MPS_REQUIRES_STATE(created, encoding);

// Requires: Any
// Transitions to: Committed
- (void)commit
    MPS_TRANSITIONS_TO(committed);

@end
```

Then write a clang-tidy check that reads these annotations:

```cpp
// MPSStateCheck.cpp - clang-tidy check
class MPSStateCheck : public ClangTidyCheck {
public:
  void check(const MatchFinder::MatchResult &Result) override {
    // Parse MPS_REQUIRES_STATE annotations
    // Track state through control flow
    // Report violations
  }
};
```

**Pros**: Declarative, documentation doubles as specification
**Cons**: Requires custom tooling to enforce

#### Approach D: Contract Comments + Custom Linter

Use structured comments that a linter can parse:

```objc
// MPSStream.mm

/// @api_constraint MTLCommandBuffer.addCompletedHandler
/// @requires state in {created, encoding}
/// @error Calling on committed buffer causes SIGABRT
/// @ref https://developer.apple.com/documentation/metal/...
void MPSStream::addCompletedHandler(MTLCommandBufferHandler block) {
    // ...
}
```

Then write a linter (Python, Rust, etc.) that:
1. Parses `@api_constraint` comments
2. Builds a state machine model
3. Performs dataflow analysis
4. Reports violations

```python
# mps_api_linter.py
class APIConstraintLinter:
    def __init__(self):
        self.constraints = self.parse_constraint_comments()

    def check_file(self, filepath):
        ast = parse_cpp(filepath)
        for call in find_api_calls(ast):
            constraint = self.constraints.get(call.api_name)
            if constraint:
                state = self.infer_state(call.receiver)
                if state not in constraint.required_states:
                    self.report_violation(call, constraint)
```

**Pros**: Language-agnostic, easy to extend
**Cons**: Less precise than compiler integration

#### Approach E: TLA+ Model with Code Correspondence

Extend TLA+ spec to model external API state machines:

```tla
--------------------------- MODULE MTLCommandBuffer ---------------------------

VARIABLES buffer_state  \* created, encoding, committed, completed

TypeOK ==
    buffer_state \in {"created", "encoding", "committed", "completed"}

\* API Constraint: addCompletedHandler requires pre-commit state
AddCompletedHandler ==
    /\ buffer_state \in {"created", "encoding"}  \* PRECONDITION
    /\ UNCHANGED buffer_state

\* Commit transitions to committed
Commit ==
    /\ buffer_state' = "committed"

\* Violation detection
APIViolation ==
    \E op \in Operations :
        /\ op.name = "addCompletedHandler"
        /\ buffer_state \in {"committed", "completed"}

Safety == ~APIViolation

=============================================================================
```

Then verify code matches the TLA+ model (manually or with tools like PlusCal).

**Pros**: Formal model includes external APIs
**Cons**: Model-code correspondence still manual

### 7.2 Recommended Implementation for This Project

**Phase 1: Document constraints (DONE)**
- Created `WORKER_DIRECTIVE_PLATFORM_ASSUMPTIONS.md`
- Added AF.007 for addCompletedHandler constraint

**Phase 2: Add runtime assertions**
```objc
void MPSStream::addCompletedHandler(MTLCommandBufferHandler block) {
    // Runtime check for API constraint
    TORCH_INTERNAL_ASSERT(
        _commandBuffer != nil || _prevCommandBuffer == nil,
        "Cannot add completed handler: no active command buffer. "
        "_prevCommandBuffer is committed and cannot have handlers added. "
        "This is a Metal API constraint (AF.007)."
    );
    // ... rest of implementation
}
```

**Phase 3: Create custom clang-tidy check**
```cpp
// checks/MPSCommandBufferStateCheck.cpp
// Tracks command buffer state through control flow
// Warns if addCompletedHandler called on potentially committed buffer
```

**Phase 4: Integrate into CI**
```yaml
# .github/workflows/static-analysis.yml
- name: Run MPS API constraint checks
  run: |
    clang-tidy -checks='-*,mps-*' pytorch-mps-fork/aten/src/ATen/mps/*.mm
```

### 7.3 Industry Examples

| Project | Approach | API Constraints Encoded |
|---------|----------|------------------------|
| Linux kernel | Sparse annotations | `__must_hold`, `__acquires`, `__releases` |
| Chrome | Clang plugins | Garbage collection safety |
| Firefox | Custom attributes | Pointers-to-pointers restrictions |
| Rust | Type system | Ownership, borrowing, lifetimes |
| Java | @Nullable/@NonNull | Null safety |
| Swift | Optional types | Null safety |

### 7.4 Future Work: AI-Assisted API Constraint Inference

An interesting research direction: **use LLMs to infer API constraints from documentation**.

```python
# Hypothetical: LLM-based constraint extraction
def extract_constraints(api_docs: str) -> List[APIConstraint]:
    prompt = f"""
    Extract API constraints from this documentation.
    For each method, identify:
    - Preconditions (what state must be true before calling)
    - Postconditions (what state is true after calling)
    - Error conditions (what causes failures)

    Documentation:
    {api_docs}
    """
    return llm.extract_structured(prompt, APIConstraint)

# Then generate checker code automatically
constraints = extract_constraints(metal_docs)
checker_code = generate_clang_checker(constraints)
```

This could help projects bootstrap API constraint checking without manual annotation.

---

## Appendix A: Assumption Ledger Summary

| ID | Assumption | Verified | Risk |
|----|------------|----------|------|
| A.001 | std::mutex provides mutual exclusion | Trusted | LOW |
| A.002 | std::atomic provides seq_cst | Trusted | LOW |
| A.003 | pthread_atfork registers correctly | Tested | LOW |
| A.004 | use_count never wraps (32-bit) | Analyzed | LOW |
| A.005 | Static destruction order defined | Documented | MEDIUM |
| A.006 | MTLCommandQueue thread-safe | Apple docs | MEDIUM |
| A.007 | addCompletedHandler pre-commit | Apple docs | **HIGH** |
| AF.X01 | MPS operations thread-safe | **FALSE** | N/A |

## Appendix B: Tools Reference

| Tool | Version | Documentation |
|------|---------|---------------|
| Apalache | 0.52.1 | https://apalache-mc.org/ |
| TLC | 2.20 | https://lamport.azurewebsites.net/tla/tools.html |
| CBMC | 5.x | https://www.cprover.org/cbmc/ |
| Coq + Iris | 9.1.0 / dev | https://iris-project.org/ |
| Clang TSA | 15+ | https://clang.llvm.org/docs/ThreadSafetyAnalysis.html |

## Appendix C: References

1. Lamport, L. "Specifying Systems" (TLA+ book)
2. Jung, R. et al. "Iris from the ground up" (Iris tutorial)
3. Kroening, D. & Strichman, O. "Decision Procedures" (CBMC theory)
4. Apple Metal Documentation (API constraints)
5. This project's VERIFICATION_TRACEABILITY.md
