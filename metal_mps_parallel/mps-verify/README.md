# MPS Verification Platform

**Created by Andrew Yates**

Multi-language formal verification platform for the MPS parallel inference implementation.

## Overview

This platform integrates multiple verification tools to ensure correctness of the concurrent MPS stream pool, allocator, and event subsystems:

- **Lean 4**: Core proofs and verification DSL
- **TLA+**: State machine model checking
- **CBMC**: Bounded model checking for C++
- **Static Analysis**: Clang Thread Safety, Infer

Assumptions and platform caveats:
- Spec modeling assumptions: `mps-verify/specs/ASSUMPTIONS.md`
- Apple/framework assumptions: `mps-verify/assumptions/apple_framework_assumptions.md` (runtime-checked via `verification/run_platform_checks`)

## Quick Start

```bash
# Build the platform
lake build

# Run verification status
./.lake/build/bin/mpsverify check

# Run the full suite (writes artifacts under mps-verify/states/<timestamp>/)
./.lake/build/bin/mpsverify check --all

# View help
./.lake/build/bin/mpsverify --help

# From repo root:
#   (cd mps-verify && lake build)
#   ./mps-verify/.lake/build/bin/mpsverify check
```

## Project Structure

```
mps-verify/
├── MPSVerify/
│   ├── Core/
│   │   ├── Types.lean         # Core type definitions
│   │   ├── MemoryModel.lean   # C++ memory model
│   │   └── Concurrency.lean   # Atomic/mutex models
│   ├── DSL/                   # Domain-specific language (planned)
│   ├── Tactics/               # Custom proof tactics (planned)
│   └── Bridges/               # Tool integrations
│       ├── TLAPlus.lean       # TLC output parser
│       └── TLCRunner.lean     # TLC process executor
├── specs/                        # 10 TLA+ specifications
│   ├── MPSStreamPool.tla         # Stream pool state machine
│   ├── MPSAllocator.tla          # Allocator ABA double-check pattern
│   ├── MPSEvent.tla              # Event callback lifetime safety
│   ├── MPSBatchQueue.tla         # Batch queue correctness
│   ├── MPSCommandBuffer.tla      # Command buffer lifecycle
│   ├── MPSForkHandler.tla        # Fork handler correctness
│   ├── MPSGraphCache.tla         # Graph cache coherency
│   ├── MPSKernelCache.tla        # Kernel cache thread safety
│   ├── MPSTLSBinding.tla         # TLS stream binding safety
│   └── MPSFullSystem.tla         # Composed system model
├── .mps-verify/
│   ├── config.json            # Verification configuration
│   └── cache/                 # Incremental verification cache
├── verification/
│   └── cbmc/                  # CBMC bounded model checking
│       ├── stubs/             # Metal API stubs
│       ├── models/            # C models of MPS types
│       └── harnesses/         # Verification harnesses
└── Main.lean                  # CLI entry point
```

## Core Modules

### MPSVerify.Core.Types

Core type definitions:
- `ThreadId`, `Location`, `Timestamp`
- `StreamId` (0-31 with default/worker distinction)
- `BufferId` with generation counter for ABA detection
- `VerificationStatus`, `Property`, `FileDependency`

### MPSVerify.Core.MemoryModel

C++11 memory model formalization:
- `MemoryOrder` (relaxed, acquire, release, seq_cst)
- `MemoryOp` (read, write, rmw, fence)
- `happensBefore` relation
- `hasDataRace` predicate
- `isRaceFree` property

### MPSVerify.Core.Concurrency

C++ concurrency primitives:
- `AtomicState<T>` with load, store, CAS, fetch_add
- `MutexState` with try_lock, unlock (recursive and non-recursive)
- `OnceState` for std::call_once semantics
- `ThreadLocal<T>` for TLS modeling
- `LockGuard` for RAII pattern

**Proven theorems:**
- `recursive_mutex_allows_recursion`: Same thread can re-lock
- `once_flag_exactly_once`: Only first caller executes

## TLA+ Specifications

Ten TLA+ specifications model the critical concurrent subsystems (~163 million states verified):

### Core Specs (Phase 2)

| Spec | Description | States |
|------|-------------|--------|
| **MPSStreamPool.tla** | 32-stream pool with TLS binding & fork safety | 1.96M |
| **MPSAllocator.tla** | ABA double-check pattern in buffer allocation | 12M |
| **MPSEvent.tla** | Callback lifetime safety with shared_ptr | 26K |

### Extended Specs (N=1392)

| Spec | Description | States |
|------|-------------|--------|
| **MPSBatchQueue.tla** | Batch queue correctness | 31K |
| **MPSCommandBuffer.tla** | Command buffer lifecycle | 3.5K |
| **MPSForkHandler.tla** | Fork handler correctness | 10K |
| **MPSGraphCache.tla** | Graph cache coherency | 995K |
| **MPSKernelCache.tla** | Kernel cache thread safety | 138M |
| **MPSTLSBinding.tla** | TLS stream binding safety | 49K |
| **MPSFullSystem.tla** | Composed system model | 8M |

### Running Model Checking

```bash
# Using CLI (runs all 10 specs)
./.lake/build/bin/mpsverify tla --all

# Or manually
cd specs
tlc MPSStreamPool.tla -config MPSStreamPool.cfg
```

## Verification Suite

```bash
# Full suite (TLA+, CBMC, TSA, Structural) with per-run artifacts
mpsverify check --all

# Best-effort run when some tools are missing
mpsverify check --all --allow-skip
```

Incremental hashing is planned (see `.mps-verify/config.json`), but `check` currently runs the selected tool(s) directly.

## Development Status

### Phase 1: Lean Foundation ✓ Complete (N=982)
- [x] Project structure
- [x] Core types and memory model
- [x] Concurrency primitives
- [x] Basic proofs

### Phase 2: TLA+ Integration ✓ Complete (N=989, expanded N=1392)
- [x] Stream pool specification (2.1)
- [x] Allocator specification (2.2, N=986)
- [x] Event specification (2.3, N=988)
- [x] TLC-Lean bridge (2.4, N=989)
- [x] 7 additional specs (N=1392): BatchQueue, CommandBuffer, ForkHandler, GraphCache, KernelCache, TLSBinding, FullSystem

### Phase 3: CBMC Integration ✓ Complete
- [x] Directory structure (`verification/cbmc/`)
- [x] Metal API stubs
- [x] BufferBlock model
- [x] Alloc/free harness
- [x] ABA detection harness
- [x] Run harnesses with CBMC (10 harnesses, 3,856 checks)

### Phase 4: Static Analysis ✓ Complete
- [x] Thread Safety Analysis (TSA) annotations applied

### Phase 5: Iris/Coq ✓ Complete (repo-level)
Proofs live under the repo root `verification/iris/` (outside this subproject).
### Phase 6: Unified Platform ✓ Complete

## Requirements

- Lean 4 (installed via elan)
- Java + TLA+ TLC (vendored in `tools/` for offline runs)
- CBMC (optional)

## License

MIT
