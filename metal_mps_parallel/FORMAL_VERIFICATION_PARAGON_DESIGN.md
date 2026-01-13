# World‑Class Formal Verification Design (Paragon Plan)

**Project:** MPS Parallel Inference (`metal_mps_parallel`)  
**Role:** MANAGER design report (documentation + directives; no code changes here)  
**Created:** 2025‑12‑17  
**Primary Goal:** Turn formal verification into a *useful, enforceable engineering gate* that enables aggressive threading/scaling refactors with high confidence.

---

## 0. Executive Summary (What “Paragon” Means Here)

“Paragon” does **not** mean “we wrote some specs.” It means:

1. **Executable gates** run by default (offline) and fail on regressions.  
2. **Traceability** exists from code → spec → property → evidence artifacts.  
3. **Assumptions** are explicit, monitored, and minimized (Apple framework limits are treated as assumptions, not swept under the rug).  
4. Verification covers the bug classes that actually matter for this project: **UAF, ABA, deadlocks, shutdown/fork hazards, stream/TLS liveness**, and **global serialization bottlenecks**.  
5. We add “aspirational” properties that **fail today** but would pass in a maximally optimized design—so verification becomes a **guide for performance/scaling** rather than only a safety checklist.

This report defines the full verification contract, the tooling architecture, the proof obligations, and the worker roadmap to make verification world‑class.

---

## 1. Scope, Threat Model, and Non‑Goals

### 1.1 System Scope (What we are verifying)

We verify concurrency‑critical parts of PyTorch MPS backend behavior as modified in this repo:

- **Stream pool**: TLS binding, pool lifetime, fork invalidation, TOCTOU checks.
  - Code: `pytorch-mps-fork/aten/src/ATen/mps/MPSStream.mm`, `.../MPSStream.h`
  - Spec: `mps-verify/specs/MPSStreamPool.tla`
  - CBMC harness: `mps-verify/verification/cbmc/harnesses/stream_pool_harness.c`
- **Allocator**: BufferBlock lifetime, ABA detection, lock ordering, shutdown safety.
  - Code: `pytorch-mps-fork/aten/src/ATen/mps/MPSAllocator.mm`, `.../MPSAllocator.h`
  - Spec: `mps-verify/specs/MPSAllocator.tla`
  - CBMC harnesses: `alloc_free_harness.c`, `aba_detection_harness.c`, `tls_cache_harness.c`
- **Events**: callback lifetime safety, pool reuse correctness, destructor behavior.
  - Code: `pytorch-mps-fork/aten/src/ATen/mps/MPSEvent.mm`, `.../MPSEvent.h`
  - Spec: `mps-verify/specs/MPSEvent.tla`

Additionally (if used in production), we treat these as **first-class verification targets** because they directly mediate concurrency and scaling:

- **Batch queue**: start/stop/submit semantics, request/future completion, shutdown races.
  - Code: `pytorch-mps-fork/aten/src/ATen/mps/MPSBatchQueue.h`, `.../MPSBatchQueue.mm`
  - (New) Spec: `mps-verify/specs/MPSBatchQueue.tla` (planned)
  - (New) CBMC harness: `mps-verify/verification/cbmc/harnesses/batch_queue_harness.c` (planned)

We also track **operation-level global serialization locks** that can dominate scaling:

- `pytorch-mps-fork/aten/src/ATen/native/mps/OperationUtils.mm` (`s_ndarray_identity_mutex`)
- `pytorch-mps-fork/aten/src/ATen/native/mps/operations/Linear.mm` (`s_linear_nograph_mutex`)
- `pytorch-mps-fork/aten/src/ATen/native/mps/operations/Normalization.mm` (`s_layer_norm_mutex`)
- `pytorch-mps-fork/aten/src/ATen/native/mps/operations/LinearAlgebra.mm` (`s_bmm_tiled_mutex`, `s_lu_*`, `s_solve_triangular_mutex`)

These locks are correctness workarounds for Apple framework thread-safety limitations; they must be treated as **explicit** assumptions (see §2.3).

### 1.2 Threat Model (What can go wrong)

We assume:

- Any user code can run **N threads** calling into MPS concurrently.
- Threads can appear/disappear (“thread churn”).
- `fork()` can occur after MPS initialization (must fail safely).
- Program shutdown can race with pending callbacks / TLS access.
- No trust in “it usually works” under GPU/Metal; we require proof‑guided discipline.

Bug classes in scope:

- **Memory safety:** use‑after‑free, double‑free, dangling TLS pointers, stale encoder use, stale pool pointers during shutdown.
- **Concurrency safety:** data races, lock order inversion, deadlocks, re‑entrant dispatch bugs.
- **Protocol safety:** TOCTOU windows, ABA reuse, cross-thread ownership changes.
- **Liveness / progress (bounded):** starvation due to global serialization; waits under locks; unbounded lock acquisition loops.
- **Scalability regressions:** avoidable global mutexes that serialize multi-thread inference.

### 1.3 Non‑Goals (what we do NOT claim)

We do **not** claim to formally verify:

- Apple’s internal MPS/MPSGraph/Metal implementations.
- GPU correctness (numerical equivalence beyond existing tests).
- Unbounded liveness (“will always finish” under arbitrary OS/GPU scheduling).
- Full C++11 memory model for all operations—unless explicitly proven.

Instead, we:

- Make assumptions explicit and enforce them with “assumption guards.”
- Prove our code uses safe protocols and does not create UB/dangling access under those assumptions.

---

## 2. Verification Contract (What We Promise)

### 2.1 Contract Levels (graded guarantees)

We adopt explicit **Verification Levels**. Each change must state what level it preserves or upgrades.

**Level 0 — Documented:** Only narrative reasoning; no machine checks. (Not acceptable for concurrency changes.)

**Level 1 — Checked Safety:**  
At least one of: TSan + stress tests OR static checks + grep audits; still weak against rare schedules.

**Level 2 — Protocol Verified:**  
TLA+ model checking validates the *algorithm/protocol* for concurrency-critical subsystems.

**Level 3 — Bounded Code Verified:**  
CBMC proves bounded safety properties for extracted concurrency kernels/harnesses.

**Level 4 — Statics Enforced:**  
Clang Thread Safety Analysis (TSA) runs on real build flags and warnings are gated.

**Level 5 — Theorem‑Verified Core:**  
Lean proves key lemmas for the core concurrency protocols and invariants (no `sorry` in gated modules).

“Paragon” means: for the concurrency kernel of this project, we achieve **Level 2–5** with traceability and enforceability.

### 2.2 Minimum Required Properties (must always hold)

These are non‑negotiable safety properties.

**Stream pool**
- No use of a stream when the pool is dead (prevents UAF via TLS stale pointers).
- Main thread stream policy consistent with implementation.
- Fork safety: post‑fork child cannot use MPS; TLS invalidated.

**Allocator**
- No double free (including during shutdown + callback completions).
- No use‑after‑free via stale pointers.
- ABA detection is sound (generation changes prevent stale references).
- Lock ordering must not allow deadlocks under any interleaving.

**Events**
- Callbacks never access dead state; callback state lifetime is safe.
- Pool reuse cannot cross-talk between generations/owners.
- Destructor behavior is safe even with pending callbacks (bounded wait/timeout semantics documented).

### 2.3 Assumption Ledger (Apple limitations become explicit obligations)

We treat Apple framework thread-safety limitations as **assumptions** with enforceable structure:

- Each global serialization mutex must be:
  1) explicitly documented as “assumption enforcement,”  
  2) as *narrow as possible* (critical section minimal),  
  3) monitored for performance impact, and  
  4) paired with a plan to avoid it in parallel mode when a safe alternative exists (often MPSGraph).

Any time we add or widen a global mutex, we must:
- update the ledger entry, and
- add a benchmark + proof obligation (§6.3) showing the scaling impact is understood.

Important: there are *two* classes of “global serialization” in this repo:

1) **Operation-level mutexes** (e.g., `s_linear_nograph_mutex`, `s_layer_norm_mutex`) that serialize a specific Apple API usage.  
2) **Cross-cutting global encoding serialization** via `at::mps::MPSEncodingLock` / `getGlobalMetalEncodingMutex()` (AGX driver workaround), which can serialize large portions of encoding across unrelated operations.

Both must be tracked in the assumption ledger, because both can cap scaling.

### 2.4 Property Catalog (required: name every guarantee)

Paragon verification requires a **named catalog** of properties. “We verified safety” is not an acceptable statement; every guarantee must have:

- **Property name** (stable identifier)
- **Subsystem** (stream pool / allocator / event / op-mutex policy / etc.)
- **Formal statement** (TLA+ invariant or temporal property; CBMC assertion family; Lean theorem)
- **Check location** (spec file, harness file, theorem file)
- **Evidence artifact** (log path + summary row in report)
- **Assumptions** (Apple limitations, boundedness limits, memory model abstractions)

Initial required property set (minimum bar):

**Stream pool**
- `SP.NoUseAfterFree` (TLA+): no thread uses stream when pool is dead.
- `SP.TLSBindingValid` (TLA+): TLS stream id always in bounds when pool alive.
- `SP.ForkInvalidatesTLS` (TLA+): fork child invalidates TLS and disables MPS.
- `SP.TOCTOUTripleCheck` (structural): code conforms to 3-check protocol.

**Allocator**
- `A.NoDoubleFree` (TLA+/CBMC): cannot free a block twice.
- `A.NoUseAfterFree` (TLA+/CBMC): freed buffers not accessed.
- `A.ABADetectionSound` (TLA+/CBMC): generation counter prevents ABA reuse.
- `A.LockOrderNoDeadlock` (TLA+): lock ordering does not admit deadlock cycles.

**Events**
- `E.CallbackNeverUAF` (TLA+): callback never accesses dead state.
- `E.PoolReuseNoCrosstalk` (TLA+): reset/reuse cannot mix generations.
- `E.DestructorWaitSafety` (TLA+): destructor sets “not alive” before waiting; timeout semantics documented.

**Scaling / policy**
- `S.NoAvoidableGlobalSerialization` (goal): in parallel mode, avoid global op mutex paths where a safe alternative exists.
- `S.BoundedWaitSignals` (goal): bounded lock/queue wait under representative workloads.

Additional “next” properties (high ROI targets for expanding verification coverage):

**Global Metal encoding lock**
- `GL.LockOrderEncodingThenStream` (static + model): no path acquires per-stream mutex while holding no encoding lock and then attempts to acquire encoding lock (prevents lock inversion deadlocks).
- `GL.NoReentrantDispatchDeadlock` (static): code must not `dispatch_sync` onto a queue it is already on (enforced via `dispatch_get_specific` pattern or equivalent).
- `GL.ShutdownSafe` (runtime + model): encoding lock “alive” flag prevents use during static destruction.

**Stream slot allocator / backpressure**
- `SP.SlotNoLeak` (model): every acquired worker slot is eventually released exactly once (including TLS destructor and `setCurrentStream` transitions).
- `SP.BackpressureNoLostWakeup` (model): when waiting is enabled, releasing a slot cannot leave all waiters blocked indefinitely if a slot exists.
- `SP.NoDoubleRelease` (runtime + model): double release is detected and does not corrupt freelist state.

**Batch queue**
- `BQ.NoStuckFutures` (model): every successful `submit()` returns a future that is eventually fulfilled (value or exception).
- `BQ.StopDrains` (model): after `stop()` completes, there are no queued requests that can never be processed.
- `BQ.SubmitStopRaceSafe` (model): `submit()` cannot enqueue work after the last worker exits (prevents “accepted request, no workers” hang).

**Shader/pipeline caches**
- `SL.CacheLinearizable` (model): concurrent requests for pipeline state behave like a linearizable cache (no corruption, no UAF).
- `SL.NoCompileUnderShardLock` (static): compilation happens outside shard locks; insertion under lock only.

**dispatch_sync_with_rethrow wrapper**
- `GCD.NoReentrantSync` (static + model): forbid calling `dispatch_sync_with_rethrow` when already on the target queue (or require a safe bypass path).

The traceability map (§4.1) is the canonical place where these are tracked and linked.

---

## 3. Tooling Architecture (What Each Tool Is For)

### 3.1 TLA+ (design/model level)

**Purpose:** Verify protocols against *all interleavings* at the algorithm level.

Strengths:
- Finds race windows, TOCTOU bugs, deadlocks from lock-order cycles.
- Great for “pool lifecycle + TLS” and “two-phase double-check patterns.”

Limitations:
- Abstraction gap: spec may drift from code unless we enforce traceability (§4).

Action items for “paragon”:
- Ensure TLC runs offline by default (see §5.1).
- Keep specs small but accurate; treat each spec as a “protocol contract.”

### 3.2 CBMC (bounded code-level model checking)

**Purpose:** Prove bounded safety properties (pointer safety, bounds, assertions) for concurrency kernels.

Strengths:
- Concrete C model; can catch subtle logic bugs in modeled code.
- Useful for verifying invariants like ABA detection and TOCTOU triple-check pattern.

Limitations:
- Bounded: depends on unwind limits.
- Our harnesses are simplified; must improve correspondence (§4.2).

### 3.3 Lean (theorem proving, reusable lemmas)

**Purpose:** Prove reusable invariants about core patterns and “why” a protocol is safe.

Strengths:
- Machine-checked reasoning; enables refactors where model checking is too expensive.

Limitations:
- High effort; must target only the highest-leverage lemmas.
- `sorry` is an unproven claim; cannot be in gated modules (§5.2).

### 3.4 Clang Thread Safety Analysis (TSA) + static analyzers

**Purpose:** Enforce lock discipline at compile-time using annotations in real compilation units.

Current reality:
- Annotations exist (`pytorch-mps-fork/aten/src/ATen/mps/MPSThreadSafety.h`) but aren’t currently enforced as a gate.

Paragon requirement:
- TSA must run using the real compile flags from `pytorch-mps-fork/build/compile_commands.json` (see §5.3).

Optional add-ons (world-class, if feasible):
- **Infer (RacerD/starvation)** with compile DB.
- **CodeQL** for concurrency antipattern scanning (lock-inversion patterns, raw shared state).
- **clang-tidy** rules for “no blocking wait under lock,” “no capture of encoder outside dispatch block,” etc.

### 3.5 Runtime verification (sanitizers + stress)

**Purpose:** Catch mismatches between models and real behavior; validate assumptions.

Must-have:
- TSan runs (already present).
- “Conformance stress tests”: fork/shutdown races, thread churn, stream churn.

Nice-to-have:
- ASan/UBSan builds for CPU-side logic.
- Deterministic stress harnesses (seeded schedule fuzzing; see §6.4).

### 3.6 Tool soundness / scope matrix (be explicit about what is proved)

This is the “no hand-waving” rule: we must be able to answer “what exactly do we know now?”.

| Tool | What it proves | What it does not prove | Primary risks | Mitigation |
|------|----------------|------------------------|---------------|------------|
| TLA+ (TLC) | Protocol correctness across all interleavings within the model | Code correctness; model-code drift | Spec divergence | Traceability map + structural conformance checks (§4) |
| CBMC | Bounded memory-safety/assertions on modeled code | Unbounded behaviors; ObjC++/Metal real behavior | Over-abstraction | Shared “concurrency kernel” extraction + harness correspondence notes (§4.2, §7.1) |
| Lean | Mathematical invariants about core patterns (no drift once proved) | Doesn’t execute the real program | Wrong theorem target; `sorry` | No-`sorry` gate + narrow lemma selection (§5.2) |
| TSA | Locking discipline in compiled translation units | Higher-level protocol bugs; runtime ordering | Build mismatch | Compile DB replay with gating (§5.3) |
| TSan | Dynamic race detection in executed tests | Misses rare schedules; can’t prove absence | Coverage gaps | Pair with TLC/CBMC; add schedule fuzzing (§6.4) |

### 3.7 “Toolbox expansion” inventory (world-class breadth, used selectively)

We should know the broader verification landscape, even if we only adopt a subset:

- **Alloy**: fast relational model finding; can encode resource lifecycle constraints.
- **SPIN/Promela**: protocol/liveness model checking; alternative to TLA+ for some specs.
- **Dafny / F★ / Why3 / Viper**: program verification for algorithmic kernels (if extracted into suitable form).
- **Frama‑C / Astrée**: C static analysis; potential for low-level allocator models (if we isolate C code).
- **KLEE / angr**: symbolic execution; useful for path exploration in extracted kernels.
- **SeaHorn / SMACK / Boogie**: bounded/unbounded verification pipelines (if we invest).
- **herd7 + cat models**: memory-order litmus testing; validates atomic order assumptions (§7.1).

Paragon rule: we only add a tool when it closes a real gap and can be made reproducible (offline, scripted).

---

## 4. Traceability: Preventing Spec/Code Drift

### 4.1 The Traceability Map (required artifact)

For each verified subsystem, create a small table mapping:

- **Code anchor(s)**: file + function + stable identifier (line ranges drift; use a comment anchor or a `constexpr` “VERIF_ANCHOR” string).
- **Spec(s)**: TLA module(s), Lean module(s), CBMC harness.
- **Properties**: invariants/assertions verified.
- **Assumptions**: Apple limitations, memory-order assumptions, runtime invariants.
- **Evidence artifacts**: TLC state count + run log, CBMC version + assertion counts, TSA warning count.

This traceability map must be updated when code changes.

### 4.2 “Design for Verifiability” refactor guideline (mandatory for future work)

Any concurrency refactor should follow:

1. **Extract the concurrency kernel** (pure C++ where possible) into a small unit.
2. Make invariants explicit: “pool alive,” “generation counters,” “lock order.”
3. Prove the kernel once (TLA+/CBMC/Lean), then reuse it.
4. Keep ObjC++/Metal calls outside the core protocol when possible (treat them as opaque actions in models).

This reduces the proof surface area and improves correspondence.

### 4.3 Model-based testing bridge (make specs generate regressions)

World-class practice: use formal models to generate **concrete tests**.

Minimum viable approach:
- From TLA+ counterexample traces, extract a minimal sequence of API-level operations (alloc/free/getPtr, acquire/reset/destroy event, stream pool init/destroy/fork).
- Implement that sequence as a deterministic runtime test/harness and keep it in `tests/` with the seed/trace attached.

Goal: when TLC finds a bug (or when a goal property is violated), we preserve a runnable reproducer so the suite becomes self-reinforcing.

---

## 5. Fixing Verification Infrastructure (make it executable)

### 5.1 TLC discovery and offline execution (must fix)

Current issue: `mpsverify tla --all` can fail to find TLC even though the repo contains `mps-verify/tools/tla2tools.jar`.

Paragon requirement:
- `mpsverify tla --all` must work offline on any machine that has Java.
- TLC jar search must include `mps-verify/tools/tla2tools.jar`.
- Store TLC outputs + state counts under `mps-verify/states/<timestamp>/` and summarize in the report.

### 5.2 Lean: zero `sorry` in gated modules

Current issue: Lean build reports a `sorry` in `mps-verify/MPSVerify/Core/MemoryModel.lean`.

Paragon requirement:
- Split “Conjectures / future work” into a separate module not included in `mpsverify check --all` gating.
- Add a “no-sorry” lint step for gated modules.

### 5.3 Static analysis must use the real compile DB

Current reality: `pytorch-mps-fork/build/compile_commands.json` exists, but `mpsverify static` doesn’t run TSA against it.

Paragon requirement:
- Provide a deterministic “TSA runner” that:
  - loads the compile DB,
  - selects relevant translation units (MPS files + touched deps),
  - replays compile commands with `-fsyntax-only -Wthread-safety*`,
  - outputs machine-readable results,
  - fails on new warnings (or fails on any warnings, depending on policy).

### 5.4 Unified reporting and regressions as first-class artifacts

`mpsverify check --all` must output:

- `verification_report.md`
- `verification_report.json` (machine-readable)
- logs for TLC/CBMC/TSA in a timestamped directory

The report must state **exactly** which tools ran and which did not, and why (e.g., “TLC not found” is unacceptable once jar is vendored).

### 5.5 Artifact integrity + reproducibility protocol (no “it passed on my machine”)

Paragon requirement: verification outputs are **artifacts**, not console noise.

Rules:
- Every `mpsverify check --all` run emits a timestamped directory under `mps-verify/states/<timestamp>/`.
- Artifacts include:
  - `tlc/*.log` (one per spec) and (if failure) the counterexample trace.
  - `cbmc/*.log` and a summary row including CBMC version + assertion counts.
  - `tsa/*.log` (compile-db replay) + JSON summary.
- The top-level `verification_report.json` stores:
  - tool versions,
  - pass/fail + counts,
  - paths to artifact logs,
  - git HEAD and patch hash.

### 5.6 Verification impact statement (required for concurrency-affecting commits)

Any commit that touches concurrency protocol code must include a short “verification impact statement” (in commit message or a report) answering:

1) Which properties from §2.4 changed, if any?  
2) Which tools were re-run? What were the results?  
3) Did any assumption ledger entries change (global mutexes, Apple limitations)?  
4) What new artifacts were produced (paths)?  

This is not bureaucracy—it’s how we avoid invisible regressions and spec drift.

---

## 6. “More Than Safety”: Verification That Drives Scaling

### 6.1 Aspirational properties (should pass in an optimized design)

These properties may FAIL today. That’s the point: they guide optimization and prevent regressions.

**A1. No Avoidable Global Serialization in Parallel Mode**
- If `parallel_streams_active` (or equivalent) is true, then operations must avoid global mutex paths when safe alternatives exist (usually MPSGraph paths).
- This becomes a *design rule*: “Parallel mode must not route through globally serialized correctness workarounds unless no alternative exists.”

**A2. Bounded Lock Wait / Bounded Queue Wait**
- Threads should not spin/wait unboundedly on a single lock/queue under typical steady-state inference.
- Verified via:
  - model-level bounded wait counters (TLA+),
  - runtime instrumentation thresholds (tests),
  - static prohibition of “wait under lock” patterns.

**A3. Existence of parallel critical sections**
- There exists an execution where two threads can proceed concurrently through memory-management code paths.
- If the design forces global serialization, we acknowledge it; otherwise we must prove it’s not accidental.

### 6.2 Measure → attribute → prove

For scaling regressions, we require a loop:

1) **Attribute** bottlenecks (instrument global mutexes, allocator locks, dispatch queue waits, shader compilation).  
2) **Refactor** to remove avoidable serialization.  
3) **Prove** the refactor preserves safety properties and improves aspirational properties.  
4) **Lock it in** with gates (TLA+/CBMC/TSA + benchmark thresholds).

### 6.3 Global mutex ledger (must be kept current)

Maintain a single list of global serialization points with:

- name + location,
- what it serializes,
- why it exists (Apple limitation evidence),
- scope reduction notes,
- parallel-mode alternative,
- benchmark impact.

This ledger should be referenced by the worker checklist (§8) and updated when any global mutex changes.

### 6.4 Deterministic schedule fuzzing (world-class add-on)

Add a “schedule fuzzer” mode for a subset of CPU-side concurrency kernels:

- Use seeded randomness to permute operation order and inject yields.
- Preserve seeds for reproduction.
- Goal: bridge the gap between “bounded model” and “real runtime.”

This is not a substitute for TLA+/CBMC, but it is a high-signal regression catcher.

### 6.5 “Assumption falsification” tests (prove the mutex is actually needed)

Where we rely on global mutexes due to Apple framework thread-safety limits, we should periodically test whether the assumption still holds on newer macOS/Metal:

- A controlled “unsafe mode” build/test that disables a given mutex (one at a time) and runs a high-stress reproduction suite.
- If it still crashes, we keep the mutex and preserve evidence.
- If it no longer crashes on supported OS versions, we can consider relaxing serialization (subject to upstream policy).

This converts “we assume Apple is broken” into a managed, evidence-backed assumption.

---

## 7. World-Class Extensions (optional, prioritized)

These are ambitious; they must be justified by ROI.

### 7.1 C++ concurrency model checking (CDSChecker / similar)

If we have a small extracted concurrency kernel that uses atomics/mutexes:
- run a C++ memory-model model checker (where practical) to validate memory-order assumptions.

Concrete target: validate that our atomic ordering assumptions are not accidental (e.g., store-store ordering reliance, acquire/release pairs). Where possible:
- write litmus tests for key patterns and check them with `herd7` (cat model approximations).
- ensure code does not rely on undocumented compiler/codegen behavior.

### 7.2 Separation logic (Iris/Coq) for one critical protocol

Pick exactly one high-value target (e.g., callback-state lifetime) and fully verify it in Iris/Coq to demonstrate world-class rigor.

### 7.3 PlusCal for protocol specs

Where a spec becomes too complex, write it in PlusCal and compile to TLA+ to reduce mistakes and improve maintainability.

### 7.4 Static ruleset “no footguns”

Encode known-dangerous patterns as static checks:
- no blocking waits while holding a capability lock,
- no encoder capture outside dispatch blocks,
- no global mutable state without atomic/lock annotation,
- no lock order inversion patterns (best-effort).

---

## 8. Implementation Roadmap (High-level)

1. **Make gates executable**: TLC discovery + compile-db TSA + no-sorry gating.  
2. **Traceability map**: code ↔ specs ↔ properties ↔ evidence.  
3. **Aspirational scalability properties**: add failing-now properties as tracked goals.  
4. **Correspondence strengthening**: extract concurrency kernels; upgrade CBMC harnesses.  
5. **Optional world-class extensions**: only if they pay off for actual refactors.

---

## 9. Success Criteria (what “done” looks like)

Minimum bar:
- `mpsverify check --all` runs offline and deterministically.
- TLC specs run and report state counts; failures store counterexample traces.
- 0 `sorry` in gated Lean modules.
- compile-db TSA runs and warnings are gated.
- A traceability map exists and is maintained.
- Verification outputs produce durable artifacts (JSON + logs) per run.

World-class bar:
- At least one real concurrency regression is caught by the suite (TLA+/CBMC/TSA) and preserved as a counterexample + test.
- At least two aspirational properties flip from FAIL→PASS due to real scaling improvements.
- The suite is fast enough for daily use (incremental <30s, full <10m on dev machine).
- Assumption ledger is actively managed and periodically falsified (§6.5).

---

## Appendix A: Current Known Gaps (ground truth)

**Update rule:** This appendix must be kept consistent with `git log --oneline -10` and the latest `mpsverify check --all` output. If it becomes stale, update it immediately; stale “ground truth” is worse than none.

As of current HEAD, the remaining gaps that block “paragon-grade” confidence are:

- **“One command” gate is incomplete:** `mpsverify check --all` is not yet a single honest, deterministic, offline gate that runs *and reports* (TLA+ + CBMC + TSA + structural) with durable artifacts (§5.4–§5.5).
- **TSA gate is clean for actionable warnings:** compile-db driven TSA (`mpsverify static`) passes with 0 warnings for `-Wthread-safety-analysis` (see `mps-verify/tsa_results.json`). Note: `-Wthread-safety-negative` is not gated on Apple clang due to non-actionable negative-capability noise for lock-taking APIs.
- **Lean has non-gated `sorry`s:** `sorry` remains in non-gated modules (e.g., `mps-verify/MPSVerify/Core/Conjectures.lean`, `mps-verify/MPSVerify/Tactics/RaceCondition.lean`). This is acceptable only if the gate never claims them as proven; the boundary must remain explicit (§3.3, §5.2).
- **Coverage gaps in what we model:** critical protocols are not yet specified/harnessed:
  - **CUDA-style `recordStream` cross-stream lifetime protocol** in `MPSAllocator` (Appendix B1 item 1).
  - **Batch queue shutdown/liveness** (`MPSBatchQueue`) (Appendix B1 item 2).
  - **Global encoding lock critical-section minimality** (avoid “wait under global lock”) (Appendix B1 item 3, §6.1).
- **Lock hierarchy documentation vs reality:** `pytorch-mps-fork/aten/src/ATen/mps/MPSThreadSafety.h` lock-order guidance must be reconciled with actual code paths (encoding lock sometimes acquired before per-stream locks) and then enforced via TSA + structural checks (Appendix B1 item 3).
- **Correspondence is still thin:** CBMC harnesses and TLA+ specs prove important properties, but the “why this matches the production kernel” story must be strengthened further before using proofs as refactor armor (§4.2, §7.1).
- **Aspirational/scaling properties are mostly unimplemented:** the project needs *tracked FAIL → PASS* goals that explicitly measure/flag global serialization and progress hazards (§6.1–§6.3).

## Appendix B: Opportunity Map — Where Formal Verification Can Help (exhaustive, curated)

This appendix is the “keep adding until exhausted” inventory. It enumerates **all** meaningful places in this repo where formal methods can add high-value guarantees, grouped by ROI and tool fit. Items are phrased as *verification applications*, not just “write another spec.”

### B1. High-ROI, concurrency-protocol targets (add next)

These are the most likely to (a) find real bugs and (b) de-risk future scaling refactors.

1) **CUDA-style `recordStream()` cross-stream lifetime protocol** (`MPSAllocator.{h,mm}`, `MPSEvent.{h,mm}`)
   - Why it matters: this is the core “no UAF / no early reuse” guarantee when tensors cross streams; regressions are catastrophic and hard to reproduce.
   - What to verify:
     - `RS.NoEarlyReuse`: a buffer cannot return to an allocatable pool state until all recorded cross-stream events are complete.
     - `RS.EventAccounting`: pending events are monotonically removed only when `query()` observes completion; no “lost event → premature reuse.”
     - `RS.NoUnboundedGrowth`: pending event lists cannot grow without bound under a bounded set of streams (bounded liveness/property check).
   - Tools:
     - TLA+ (primary): model `BufferBlock` state + `pending_events` + `recordStream` + `free_buffer` deferral.
     - CBMC (secondary): bounded interleavings of {alloc, recordStream, free, query} with 2–3 streams.
     - Structural checks: enforce “recordStream must create an event + record it under the right lock scope.”

2) **MPSBatchQueue protocol** (`MPSBatchQueue.h/.mm`)
   - Why it matters: mediates many-user-threads → few-worker-threads; correctness bugs become “stuck futures” or silent drops.
   - What to verify:
     - stop/submit race safety (`BQ.SubmitStopRaceSafe`, `BQ.NoStuckFutures`).
     - “stop drains” semantics (no accepted request can become unprocessable).
     - restart semantics (if supported).
     - global singleton reconfiguration safety (`configureMPSBatchQueue`) under concurrent access.
   - Tools:
     - TLA+ (primary): state machine for {running, shutdown_requested, workers_alive, queue}.
     - CBMC (secondary): bounded harness for one or two worker threads + submit/stop interleavings.

3) **Global encoding lock contract (deadlock freedom + minimality)** (`MPSEncodingLock`, `getGlobalMetalEncodingMutex`, call sites)
   - Why it matters: this lock is both (a) a correctness workaround for an Apple driver race and (b) a potential global scalability ceiling; it also introduces multi-lock ordering complexity.
   - What to verify:
     - `GL.DeadlockFree`: no lock-order cycle exists across {allocator locks, pool locks, stream locks, encoding lock, batch queue lock, shader cache locks}.
     - `GL.NoWaitUnderEncodingLock` (aspirational, likely FAIL today): no thread holds the global encoding lock while blocking on GPU completion (`waitUntilCompleted`) or other unbounded waits.
     - `GL.MetalAPICoverage`: every known-racy Metal API call is either under `MPSEncodingLock` or explicitly waived with a documented justification.
   - Tools:
     - TLA+ “lock graph” model + “blocking ops under lock” model (small but high yield).
     - TSA lock annotations + a structural “Metal API must be under encoding lock” checker.

4) **Stream slot allocator + backpressure protocol** (`MPSStreamPool::acquireSlot/releaseStreamSlot`, TLS destructor)
   - Why it matters: this is a lock-free-ish allocator with a CV fallback; lost wakeups or slot leaks cap concurrency.
   - What to verify:
     - no slot leak across TLS destructor, `setCurrentStream`, and explicit release APIs.
     - backpressure waiting correctness (no indefinite waiting when slot exists, under fairness).
     - double-release does not corrupt freelist state.
   - Tools:
     - TLA+ model (bitmask + waiters + notify).
     - CBMC harness for the atomic mask/CV logic (bounded).

5) **Dispatch/queue execution context safety (TLS hazards + reentrancy)**
   - Why it matters: re-entrant dispatch_sync on the same queue deadlocks; this is a recurring pattern across ops.
   - What to verify:
     - `DQ.NoReentrantDispatchSync`: any `dispatch_sync` to a stream queue must have an “already-on-queue” bypass (`dispatch_get_specific`) or a proven non-reentrancy argument.
     - `DQ.NoTLSLookupInsideDispatchedBlock`: blocks that may run on different threads must not use thread-local stream lookup; must use captured stream pointer.
     - `DQ.ExceptionPropagationSound`: `dispatch_sync_with_rethrow` rethrows exactly once and does not swallow exceptions.
   - Tools:
     - structural checks (primary) + small TLA+ model of {on-queue/off-queue} and “holds-lock/dispatch-sync” deadlocks.

6) **Per-stream command-buffer / encoder state machine** (`MPSStream::{commandBuffer,commandEncoder,endKernelCoalescing,synchronize}`)
   - Why it matters: subtle state-machine bugs can cause leaks, stale encoder use, or deadlocks when combined with global encoding lock and dispatch blocks.
   - What to verify:
     - `CB.NoStaleEncoder`: an encoder is never used after `endEncoding`.
     - `CB.ReleaseDiscipline`: command buffers are released exactly once across commit/flush paths.
     - `CB.NoWaitWhileHoldingWrongLocks` (aspirational): blocking waits do not occur while holding locks that serialize unrelated work.
   - Tools:
     - TLA+ state machine (abstract “encoded/committed/waited/released”).
     - CBMC harness on a simplified extracted state machine (or an audited “concurrency kernel” copy).

7) **Operation-level mutex policy and path selection (graph vs no-graph)**
   - Why it matters: global op mutexes are scaling ceilings; path selection heuristics can accidentally route parallel workloads into serialized paths.
   - What to verify:
     - correctness: serialized path prevents crashes (assumption enforcement).
     - policy goal: in parallel mode, use graph path where safe and beneficial (`S.NoAvoidableGlobalSerialization`).
   - Tools:
     - TLA+ “policy model” (threads choose path; no-graph serializes; graph parallelizes; compilation cost).
     - runtime instrumentation as “proof obligation” (measure actual path usage).

8) **Global singleton lifecycle (init/shutdown/fork) across subsystems**
   - Why it matters: most hard-to-debug crashes happen during shutdown or after fork; concurrency proofs must include “terminal states.”
   - What to verify:
     - `L.InitOnce`: singletons (device, stream pool, allocator, event pool, batch queue) initialize exactly once and publish fully-initialized state.
     - `L.ShutdownNoUAF`: post-shutdown paths fail fast without UAF (alive flags are honored).
     - `L.PostForkTerminal`: in forked child, MPS is disabled consistently and cannot partially run.
   - Tools:
     - TLA+ lifecycle model (tiny, very effective) + structural checks around alive-flag usage.
     - CBMC for simplified “alive flag + TLS destructor” patterns.

### B2. Medium-ROI targets (do after B1)

9) **MetalShaderLibrary cache correctness + contention control**
   - Why it matters: pipeline compilation is expensive; cache races can crash or serialize unexpectedly.
   - What to verify:
     - cache insertion linearizability under shard locks.
     - “compile outside lock” discipline stays true over refactors.
     - Metal API calls remain under encoding lock where required (see `GL.MetalAPICoverage`).
   - Tools:
     - structural checks + CBMC for simplified map protocol (or extract cache core).

10) **MPSGraphCache / MPSKernelCache correctness envelope** (`OperationUtils.{h,mm}`)
   - Why it matters: these caches are thread-local by design, but they still touch shared Metal/MPS resources and compile graphs/kernels.
   - What to verify:
     - thread-local isolation remains true (no sharing of non-thread-safe cached objects across threads).
     - graph/kernel creation always respects encoding-lock requirements.
     - cache growth is bounded or at least observable (operational proof obligation).
   - Tools:
     - structural checks (primary) + optional CBMC on extracted cache core.

11) **MPSProfiler destructor / dispatch safety**
   - Why it matters: shutdown paths often deadlock; profiler runs at exit.
   - What to verify:
     - no `dispatch_sync` deadlock if destructor runs on stream queue.
     - completion handler draining is safe.
   - Tools:
     - static check + small TLA+ model of “destructor invoked on/off queue.”

12) **Allocator GC / pending-free protocol (beyond recordStream)** (`MPSAllocator.mm`)
   - Why it matters: pending sets + completion handlers are where UAF/double-free bugs hide.
   - What to verify:
     - pending-free state machine safety (no block freed while in use).
     - shutdown safety with late callbacks.
   - Tools:
     - extend `MPSAllocator.tla` and/or add a dedicated “pending buffers” spec.

13) **Fork safety beyond “TORCH_CHECK in child”**
   - Why it matters: fork-after-Metal-init is notoriously dangerous; must remain fail-safe.
   - What to verify:
     - after fork handler, all APIs that could touch Metal reliably fail fast.
   - Tools:
     - structural checks + a small model for “forked child terminal state.”

14) **Env-var configuration as a verified contract**
   - Why it matters: env vars select safety-vs-performance modes (encoding mutex disable, graph path forcing, pool backpressure) and can silently invalidate assumptions.
   - What to verify:
     - parsing is total and robust (no UB / no weird partial states).
     - dangerous modes are loud and traceable (emit a single durable warning + require explicit opt-in).
   - Tools:
     - CBMC/clang analyzer for parsing code + structural checks for “dangerous mode must log.”

15) **Public API surface contract: hooks + “device-wide” sync semantics** (`MPSHooks.mm`, `MPSStreamPool::synchronizeAllStreams`)
   - Why it matters: these are the externally visible semantics that upstream users rely on; regressions here can silently break correctness or scaling.
   - What to verify:
     - `H.DeviceSynchronizeIsDeviceWide`: `deviceSynchronize()` waits for all active streams (matches CUDA semantics claim).
     - `H.NoDeadlockWithGlobalLocks`: calling hooks from arbitrary threads cannot deadlock with stream/allocator/event locks.
     - `H.NoWaitUnderGlobalEncodingLock` (aspirational, overlaps `GL.NoWaitUnderEncodingLock`): hooks do not hold global encoding lock while waiting.
   - Tools:
     - TLA+ small “device sync” model + structural checks for lock scopes.

16) **Parallel-mode detection signal correctness** (`g_active_stream_users`, stream/TLS transitions)
   - Why it matters: parallel-vs-serial mode signals drive safety fallbacks and path selection; if this counter is wrong, we can take the wrong (unsafe or slow) path.
   - What to verify:
     - `PM.CountAccurate`: each thread contributes at most once; counter never underflows; transitions (setCurrentStream, TLS destructor) preserve invariants.
     - `PM.NoStaleOnQueue`: queue-executed blocks use queue-specific stream pointer (not TLS) so “parallel mode” does not depend on GCD thread choice.
   - Tools:
     - TLA+ lifecycle model + CBMC bounded interleavings around TLS destructor and setCurrentStream.

17) **RAII correctness envelope for stream acquisition helpers** (`MPSStreamGuard`, acquire/release APIs)
   - Why it matters: multiple overlapping slot-management mechanisms (TLS destructor, explicit release, RAII guard) are fertile ground for leaks and double-free-style bugs.
   - What to verify:
     - `SG.NoSlotLeak`: any slot acquired through the guard is released exactly once.
     - `SG.NoDoubleRelease`: guard + TLS interactions do not corrupt freelist state (duplicate release is handled safely).
   - Tools:
     - TLA+ stream-slot model extended with RAII transitions + structural checks for safe usage patterns.

### B3. Low-ROI / “only if it pays” targets

18) **LRU eviction correctness for thread-local graph/kernel caches**
   - Why it matters: memory blow-ups are real, but concurrency risk is low because caches are thread-local.
   - Tools: unit tests + optional CBMC on extracted LRU core.

19) **Atomic memory-order assumptions via litmus testing**
   - Why it matters: subtle “it works on ARM codegen” assumptions can break with compilers/flags.
   - Tools: herd7-litmus tests for key patterns; keep scope small.

20) **Full op-by-op correctness envelopes**
   - Why it matters: tempting, but huge surface area; only do it when a specific op is implicated in a concurrency bug.
   - Tools: targeted spec/harness per-op; avoid blanket expansion.

## Appendix C: Pruned / Rejected Applications (documented to avoid re-adding bloat)

These were considered and **intentionally not pursued** right now because they are low ROI, high maintenance, or not realistically provable. Do not re-add them without a concrete bug/need.

1) **“Verify Metal/MPSGraph internals”**  
   Rejected: closed-source; cannot be proven; only assumptions + falsification tests are feasible (§6.5).

2) **Full numerical correctness / floating-point equivalence proofs**  
   Rejected: disproportionate effort; existing tests + upstream correctness expectations are the right tool.

3) **Prove the entire C++11 memory model for the whole backend**  
   Rejected: state explosion and low marginal benefit; focus on specific litmus tests for key assumptions (§7.1) instead.

4) **Iris/Coq for everything**  
   Rejected: only worth doing for *one* flagship protocol where it unlocks refactors (§7.2).

5) **Formal verification of all per-op Metal kernels**  
   Rejected: huge surface area; better served by structural concurrency checks + targeted bug repros.

6) **“Verify performance” in a purely formal sense**  
   Rejected: performance is empirical; we instead formalize *anti-serialization policies* and then enforce via benchmarks as proof obligations (§6.2–§6.3).

7) **Prove Objective‑C retain/release correctness end-to-end**  
   Rejected: too hard to model faithfully across ARC/MRR/autorelease pools; prefer targeted static analyzers + leak checks + small “lifetime protocol” models where they pay off.

8) **Prove hash-collision freedom for cache keys**  
   Rejected: you cannot prove collision freedom of `std::hash`; instead, treat collisions as a correctness risk and mitigate structurally (store full key, validate on lookup, or avoid hash-as-key patterns).

9) **Verify full GCD/dispatch semantics**  
   Rejected: GCD is external; we model only the minimal “queue execution context” assumptions we rely on (reentrancy, possible different thread execution) and enforce them structurally.

10) **Duplicate-model every protocol in multiple model checkers**  
   Rejected: running the same spec in TLC + SPIN + Alloy + Apalache is mostly bloat; pick *one* primary checker per protocol unless a specific limitation forces a second.

11) **Prove unbounded liveness under real OS/GPU scheduling**  
   Rejected: we cannot realistically model macOS scheduling + GCD + GPU driver progress guarantees end-to-end. Instead, we prove bounded progress properties (where feasible) and enforce empirical “bounded wait” obligations with stress tests (§6.2).

12) **Formally verify the entire Python test/benchmark harness layer**  
   Rejected: too wide and not the risk center; focus formal work on the C++/ObjC++ concurrency kernels and use the existing stress tests as assumption-falsification and regression reproduction vehicles.
