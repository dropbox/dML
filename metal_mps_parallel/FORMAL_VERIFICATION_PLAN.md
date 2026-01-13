# Formal Verification Plan for MPS Parallel Inference

## Overview

Comprehensive formal verification for thread-safe parallel PyTorch MPS inference on Apple Silicon. The goal is to mathematically prove correctness properties and build reusable verification infrastructure.

---

## Tool Landscape

### Tier 1: Model Checking (State Machine Verification)

| Tool | Language | Best For | Maturity |
|------|----------|----------|----------|
| **TLA+** | Promela-like | High-level concurrent protocol design | Production |
| **SPIN** | Promela | Deadlock/livelock detection | Production |
| **CBMC** | C/C++ | Bounded model checking, memory safety | Production |
| **Kani** | Rust | Bit-precise verification | Emerging |

### Tier 2: Proof Assistants (Theorem Proving)

| Tool | Foundation | Best For | Extensibility |
|------|------------|----------|---------------|
| **Lean 4** | Type theory | Math proofs, custom DSLs | Excellent (metaprogramming) |
| **Coq/Rocq** | Type theory | Foundational proofs | Good |
| **Isabelle/HOL** | Higher-order logic | Large proofs | Good |

### Tier 3: Separation Logic / Concurrency Frameworks

| Tool | Built On | Best For | C/C++ Support |
|------|----------|----------|---------------|
| **Iris** | Coq | Concurrent separation logic | Via RefinedC |
| **RefinedC** | Iris/Coq | C with ownership types | Native |
| **VeriFast** | Custom | C/Java/Rust concurrent verification | Native |
| **VST** | Coq | C verification (CompCert semantics) | Native |

### Tier 4: Language-Integrated Verification

| Tool | Language | Best For | Extraction |
|------|----------|----------|------------|
| **Dafny** | Custom | Verified-by-construction code | C#, Go, Java |
| **F*** | ML-like | Cryptography, low-level verified code | C, OCaml, Wasm |
| **Verus** | Rust | Verified Rust systems code | Rust |

### Tier 5: Static Analysis (Lightweight)

| Tool | Target | Best For | Automation |
|------|--------|----------|------------|
| **Clang Thread Safety** | C/C++ | Lock annotations | Fully automatic |
| **Infer** | C/C++/Java | Race detection, null safety | Fully automatic |
| **Frama-C** | C | ACSL contracts, abstract interpretation | Semi-automatic |

---

## Recommended Strategy: Layered Verification

### Layer 1: TLA+ for Protocol Design (Immediate)
**Why**: Verify high-level concurrent protocol correctness before implementation details.

**Targets**:
- Stream pool lifecycle (creation, binding, destruction, fork)
- Allocator double-check locking with ABA detection
- Event callback survival semantics

**Deliverables**:
- `specs/MPSStreamPool.tla` - Stream pool state machine
- `specs/MPSAllocator.tla` - Buffer allocation protocol
- `specs/MPSEvent.tla` - Event lifecycle
- TLC model checking results proving deadlock freedom

### Layer 2: Lean 4 for Core Invariants (High Value)
**Why**: Extensible, excellent metaprogramming, can build domain-specific tactics.

**Approach**:
- Model C++ concurrency primitives in Lean
- Build MPS-specific tactics for common patterns
- Prove key invariants (ABA detection correctness, TLS safety)

**Extensibility Opportunity**:
```lean
-- Define domain-specific notation for C++ atomics
notation "atomic_load_acquire" x => Atomic.load x MemoryOrder.acquire

-- Custom tactic for double-check locking proofs
macro "prove_dcl_safe" : tactic => `(tactic| (
  apply dcl_safety_theorem
  · prove_first_check_sufficient
  · prove_second_check_consistent
  · prove_aba_detection_sound
))
```

**Deliverables**:
- `lean/MPSConcurrency.lean` - C++ atomic/mutex modeling
- `lean/DoubleCheckLocking.lean` - DCL pattern proofs
- `lean/ABADetection.lean` - ABA counter correctness
- Custom Lean tactics for MPS verification patterns

### Layer 3: Iris/RefinedC for C Code (Deep Verification)
**Why**: Separation logic is the gold standard for concurrent C verification.

**Approach**:
- Use RefinedC to annotate critical C functions
- Prove memory safety and data race freedom in Iris
- Get foundational proofs checked by Coq

**Targets** (priority order):
1. `MPSAllocator.mm::getSharedBufferPtr()` - ABA double-check
2. `MPSStream.mm::getCurrentStream()` - TLS binding safety
3. `MPSEvent.mm::~MPSEvent()` - Callback survival

**Deliverables**:
- RefinedC annotations for 3 critical functions
- Iris proofs of race freedom
- Coq proof certificates

### Layer 4: CBMC for Exhaustive Bug Finding
**Why**: Bounded model checking finds real bugs in actual C++ code.

**Targets**:
- Memory safety (buffer overflows, use-after-free)
- Assertion violations
- Undefined behavior

**Integration**:
```bash
# Bounded verification of allocator
cbmc aten/src/ATen/mps/MPSAllocator.mm \
  --unwind 10 \
  --pointer-check \
  --bounds-check \
  --memory-leak-check
```

### Layer 5: Static Analysis Integration
**Why**: Low-cost, continuous verification in CI.

**Tools**:
- Clang Thread Safety Annotations
- Facebook Infer (racerd checker)
- Clang Static Analyzer

---

## Custom Verifier Architecture (Long-Term)

Build a unified verification framework that combines multiple tools:

```
┌─────────────────────────────────────────────────────────────┐
│                    MPS Verification Suite                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   TLA+ Spec  │  │  Lean Proofs │  │ RefinedC/Iris│      │
│  │   (Protocol) │  │ (Invariants) │  │  (C Safety)  │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                 │                  │              │
│         ▼                 ▼                  ▼              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Unified Verification DSL               │   │
│  │         (Built in Lean 4 with metaprogramming)      │   │
│  └─────────────────────────────────────────────────────┘   │
│         │                 │                  │              │
│         ▼                 ▼                  ▼              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │    CBMC      │  │ Clang TSan   │  │    Infer     │      │
│  │  (Bounded)   │  │ (Annotations)│  │   (Races)    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Lean 4 as Integration Hub

Lean 4's metaprogramming allows building:
1. **TLA+ embeddings** - Write TLA+ specs in Lean, get Lean proofs
2. **C AST parsing** - Extract C code structure for verification
3. **Custom tactics** - Domain-specific automation for MPS patterns
4. **Proof certificates** - Export to other systems

---

## Implementation Phases

### Phase 1: TLA+ Specifications (5-8 AI commits)
- Create TLA+ models for stream pool, allocator, events
- Run TLC model checker
- Document verified properties

### Phase 2: Static Analysis (3-4 AI commits)
- Add Clang thread safety annotations to headers
- Configure Infer for CI
- Fix any warnings found

### Phase 3: Lean 4 Foundation (8-12 AI commits)
- Model C++ atomics and mutexes in Lean
- Prove ABA detection correctness
- Build MPS-specific tactics
- Create verification DSL prototype

### Phase 4: RefinedC/Iris (10-15 AI commits)
- Annotate critical C functions
- Prove separation logic properties
- Generate Coq proof certificates

### Phase 5: CBMC Integration (4-6 AI commits)
- Configure CBMC for MPS code
- Create verification harnesses
- Integrate with test suite

### Phase 6: Unified Verifier (15-20 AI commits)
- Build Lean 4 verification framework
- Integrate all tools
- Create CI pipeline
- Documentation

---

## Critical Files

**MPS Source (verification targets)**:
- `pytorch-mps-fork/aten/src/ATen/mps/MPSStream.mm`
- `pytorch-mps-fork/aten/src/ATen/mps/MPSAllocator.mm`
- `pytorch-mps-fork/aten/src/ATen/mps/MPSEvent.mm`
- `pytorch-mps-fork/aten/src/ATen/native/mps/MetalShaderLibrary.h`

**New verification artifacts**:
- `specs/*.tla` - TLA+ specifications
- `lean/` - Lean 4 proofs and framework
- `verification/refinedc/` - RefinedC annotations
- `verification/cbmc/` - CBMC harnesses

---

## Properties to Verify

| Property | Method | Priority |
|----------|--------|----------|
| Deadlock freedom | TLA+ | HIGH |
| No use-after-free in stream pool | Iris | HIGH |
| ABA detection prevents corruption | Lean + TLA+ | HIGH |
| TLS binding safety across fork | TLA+ | HIGH |
| Callback survival in event destructor | Iris | MEDIUM |
| Buffer lifecycle correctness | RefinedC | MEDIUM |
| No data races (beyond TSan) | Infer + CBMC | MEDIUM |
| Sharded cache thread safety | Clang annotations | LOW |

---

## Total Estimated Effort

| Phase | AI Commits | Cumulative |
|-------|------------|------------|
| TLA+ Specs | 5-8 | 5-8 |
| Static Analysis | 3-4 | 8-12 |
| Lean Foundation | 8-12 | 16-24 |
| RefinedC/Iris | 10-15 | 26-39 |
| CBMC | 4-6 | 30-45 |
| Unified Verifier | 15-20 | 45-65 |

**Total: 45-65 AI commits** for comprehensive formal verification suite.

---

## Selected Direction: Multi-Language Verification Platform

**Goal**: Build reusable verification infrastructure that integrates all tools into a unified platform.

**Primary Success Metric**: Create tooling that can be used for ongoing MPS development and potentially other concurrent C++ projects.

---

## Unified Verification Platform Architecture

### Core Design: Lean 4 as the Integration Hub

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MPS Verification Platform                            │
│                     (mps-verify command-line tool)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Lean 4 Core (Orchestration)                       │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌────────────┐  │   │
│  │  │  C++ Parser  │ │ TLA+ Embed   │ │ Iris Bridge  │ │ Report Gen │  │   │
│  │  │  (tree-sitter)│ │ (DSL in Lean)│ │ (Coq export) │ │ (HTML/MD)  │  │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘ └────────────┘  │   │
│  │                                                                      │   │
│  │  ┌─────────────────────────────────────────────────────────────┐    │   │
│  │  │              MPS Concurrency DSL (Domain-Specific)          │    │   │
│  │  │  • StreamPool patterns  • Allocator patterns  • Event patterns│   │   │
│  │  │  • Custom tactics       • Proof templates     • Invariants   │    │   │
│  │  └─────────────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│              ┌───────────────┼───────────────┐                              │
│              ▼               ▼               ▼                              │
│  ┌──────────────────┐ ┌──────────────┐ ┌──────────────────┐                │
│  │   Model Checking │ │  Separation  │ │  Static Analysis │                │
│  │      Layer       │ │ Logic Layer  │ │      Layer       │                │
│  ├──────────────────┤ ├──────────────┤ ├──────────────────┤                │
│  │ • TLA+/TLC       │ │ • Iris/Coq   │ │ • CBMC           │                │
│  │ • SPIN           │ │ • RefinedC   │ │ • Clang-SA       │                │
│  │ • Custom Lean    │ │ • VeriFast   │ │ • Infer          │                │
│  │   state machines │ │ • VST        │ │ • Thread Safety  │                │
│  └──────────────────┘ └──────────────┘ └──────────────────┘                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Platform Components

#### 1. Lean 4 Core Framework (`mps-verify/`)
```
mps-verify/
├── lakefile.lean          # Build configuration
├── MPSVerify/
│   ├── Core/
│   │   ├── Concurrency.lean    # C++ atomic/mutex models
│   │   ├── MemoryModel.lean    # C++ memory ordering
│   │   └── Types.lean          # Shared type definitions
│   ├── DSL/
│   │   ├── StreamPool.lean     # Stream pool patterns
│   │   ├── Allocator.lean      # Allocator patterns
│   │   └── Syntax.lean         # Custom notation
│   ├── Tactics/
│   │   ├── DCL.lean            # Double-check locking
│   │   ├── ABA.lean            # ABA detection
│   │   └── RaceCondition.lean  # Race freedom
│   ├── Bridges/
│   │   ├── TLAPlus.lean        # TLA+ embedding
│   │   ├── CBMC.lean           # CBMC harness generation
│   │   └── Coq.lean            # Coq/Iris export
│   └── CLI/
│       ├── Main.lean           # CLI entry point
│       └── Report.lean         # Report generation
└── specs/                      # Generated TLA+ specs
```

#### 2. TLA+ Embedding in Lean

```lean
-- Embed TLA+ state machines directly in Lean
namespace TLAPlus

structure State where
  variables : List (String × Value)

structure Action where
  name : String
  guard : State → Bool
  effect : State → State

structure Spec where
  init : State → Bool
  next : State → State → Bool
  fairness : List Action

-- Prove properties in Lean
theorem streamPoolDeadlockFree (spec : MPSStreamPool.spec) :
    ∀ s, Reachable spec s → ¬Deadlock s := by
  prove_temporal_property

-- Export to TLA+ for TLC model checking
#export_tla MPSStreamPool.spec "specs/MPSStreamPool.tla"
```

#### 3. C++ Analysis Bridge

```lean
-- Parse C++ using tree-sitter, extract concurrent primitives
namespace CppAnalysis

def extractAtomics (file : FilePath) : IO (List AtomicUsage) := do
  let ast ← TreeSitter.parse file
  pure $ ast.filter isAtomicOperation

def extractMutexPatterns (file : FilePath) : IO (List MutexPattern) := do
  let ast ← TreeSitter.parse file
  findDoubleCheckLocking ast ++ findRAIILocks ast

-- Generate verification conditions
def generateVCs (pattern : MutexPattern) : List VerificationCondition :=
  match pattern with
  | .doubleCheckLocking dcl =>
    [ VC.firstCheckSufficient dcl
    , VC.secondCheckConsistent dcl
    , VC.noTOCTOU dcl
    ]
  | .abaDetection aba =>
    [ VC.counterMonotonic aba
    , VC.detectionComplete aba
    ]
```

#### 4. Report Generation

```lean
-- Generate comprehensive verification reports
structure VerificationReport where
  timestamp : DateTime
  components : List ComponentReport
  properties : List PropertyStatus
  coverageMetrics : CoverageData

def generateReport (results : VerificationResults) : IO Unit := do
  let html := renderHTML results
  let markdown := renderMarkdown results
  IO.FS.writeFile "verification-report.html" html
  IO.FS.writeFile "verification-report.md" markdown
```

### CLI Interface

```bash
# Full verification suite
mps-verify check --all

# Specific verifiers
mps-verify tla --spec=StreamPool --model-check
mps-verify cbmc --file=MPSAllocator.mm --unwind=10
mps-verify iris --function=getSharedBufferPtr
mps-verify static --clang-tsa --infer

# Interactive proof development
mps-verify prove --component=ABADetection --tactic=auto

# Generate reports
mps-verify report --format=html --output=verification-report.html
```

---

## Detailed Implementation Plan

### Phase 1: Foundation (10-12 commits)

**1.1 Lean 4 Project Setup (2 commits)**
- Initialize Lake project structure
- Configure dependencies (Mathlib, std4)
- Set up CI for Lean builds

**1.2 C++ Concurrency Model (4 commits)**
- Model `std::atomic<T>` with memory orderings
- Model `std::mutex` and `std::recursive_mutex`
- Model `std::once_flag` and `std::call_once`
- Define thread-local storage semantics

**1.3 Core Tactics (4 commits)**
- Implement `prove_race_free` tactic
- Implement `prove_deadlock_free` tactic
- Implement `prove_aba_safe` tactic
- Create tactic combinators

### Phase 2: TLA+ Integration (8-10 commits)

**2.1 TLA+ DSL in Lean (3 commits)**
- Define TLA+ syntax as Lean DSL
- Implement state machine semantics
- Create spec combinators

**2.2 TLC Bridge (2 commits)**
- Export Lean specs to TLA+ files
- Parse TLC output back into Lean
- Counterexample visualization

**2.3 MPS Specifications (3-4 commits)**
- `MPSStreamPool.lean` → `MPSStreamPool.tla`
- `MPSAllocator.lean` → `MPSAllocator.tla`
- `MPSEvent.lean` → `MPSEvent.tla`
- Run TLC, verify properties

### Phase 3: CBMC Integration (6-8 commits)

**3.1 Harness Generation (3 commits)**
- Parse C++ to identify verification targets
- Generate CBMC harnesses from Lean specs
- Configure unwinding bounds

**3.2 Result Integration (2 commits)**
- Parse CBMC output
- Map counterexamples to source
- Integrate with report system

**3.3 MPS Verification (2-3 commits)**
- Verify MPSAllocator memory safety
- Verify MPSStream pointer validity
- Document findings

### Phase 4: Static Analysis Layer (5-6 commits)

**4.1 Clang Thread Safety (2 commits)**
- Add annotations to MPS headers
- Configure warning levels
- Create annotation helper macros

**4.2 Infer Integration (2 commits)**
- Configure .inferconfig
- Create CI scripts
- Filter MPS-relevant results

**4.3 Unified Static Runner (1-2 commits)**
- Single command for all static tools
- Aggregate results
- Deduplicate findings

### Phase 5: Iris/Coq Integration (12-15 commits)

**5.1 Coq Project Setup (2 commits)**
- Configure Coq with Iris
- Set up RefinedC

**5.2 Core Function Proofs (6-8 commits)**
- `getSharedBufferPtr` separation logic proof
- `getCurrentStream` race freedom proof
- `~MPSEvent` callback safety proof
- Document proof strategies

**5.3 Lean-Coq Bridge (4-5 commits)**
- Export proof obligations from Lean
- Import Coq proof results
- Unified proof status tracking

### Phase 6: Platform Integration (10-12 commits)

**6.1 CLI Development (4 commits)**
- Implement command parser
- Add subcommands for each verifier
- Progress reporting
- Error handling

**6.2 Report System (3 commits)**
- HTML report template
- Markdown output
- Verification badge generation

**6.3 Incremental Verification System (4 commits)**
- File-change detection (hash-based)
- Dependency tracking between specs and source
- Cache verification results locally
- `mps-verify --incremental` only re-verifies changed components
- Local verification scripts (no external CI)

**6.4 Documentation (2 commits)**
- User guide
- API documentation
- Extension guide

---

## Timeline Summary

| Phase | Description | Commits | Cumulative |
|-------|-------------|---------|------------|
| 1 | Lean Foundation | 10-12 | 10-12 |
| 2 | TLA+ Integration | 8-10 | 18-22 |
| 3 | CBMC Integration | 6-8 | 24-30 |
| 4 | Static Analysis | 5-6 | 29-36 |
| 5 | Iris/Coq | 12-15 | 41-51 |
| 6 | Platform | 10-12 | 51-63 |

**Total: 51-63 AI commits**

---

## Incremental Verification Design

### Core Concept
Only re-verify what changed. Track dependencies between:
- C++ source files → verification specs
- Specs → proof obligations
- Proof obligations → verification results

### Cache Structure
```
.mps-verify/
├── cache/
│   ├── hashes.json          # SHA256 of all tracked files
│   ├── dependencies.json    # File dependency graph
│   └── results/
│       ├── tla/             # TLC model checking results
│       ├── cbmc/            # CBMC verification results
│       ├── iris/            # Coq proof status
│       └── static/          # Static analysis results
├── reports/
│   └── latest/              # Most recent verification report
└── config.json              # User configuration
```

### Dependency Tracking
```lean
-- Automatically track which specs depend on which source files
structure Dependency where
  spec : FilePath           -- e.g., specs/MPSStreamPool.lean
  sources : List FilePath   -- e.g., [MPSStream.mm, MPSStream.h]
  verifiers : List String   -- e.g., ["tla", "cbmc"]

def needsReverification (dep : Dependency) : IO Bool := do
  let cachedHashes ← loadCachedHashes
  for src in dep.sources do
    let currentHash ← hashFile src
    if cachedHashes.get? src != some currentHash then
      return true
  return false
```

### CLI Usage
```bash
# Full verification (first run or after major changes)
mps-verify check --all

# Incremental (default - only changed components)
mps-verify check

# Force re-verify specific component
mps-verify check --force --component=allocator

# Show what would be verified without running
mps-verify check --dry-run

# Clear cache and reverify everything
mps-verify check --clean
```

---

## First Implementation Steps

Start with Phase 1.1 and 1.2:

1. Create `mps-verify/` directory with Lake project
2. Implement basic `Concurrency.lean` with atomic models
3. Create first tactic proof for simple invariant
4. Verify Lean toolchain works end-to-end
5. Set up incremental verification cache structure early
