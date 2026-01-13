# Vision: World-Class GenAI Formal Verification

**Goal:** Establish this project as the definitive example of AI-assisted formal verification—a reference implementation that demonstrates what's possible when generative AI is applied rigorously to mathematical proof and program verification.

---

## Why This Matters

### The Current State of Formal Verification

Formal verification is powerful but underused:
- **seL4** took 11 person-years to verify a 10,000-line microkernel
- **CompCert** took 6 person-years for a verified C compiler
- Most software ships without any formal guarantees

The barrier isn't capability—it's **cost**. Writing proofs is labor-intensive, requiring rare expertise in both the domain and proof assistants.

### The AI Opportunity

Generative AI changes the economics:
- AI can explore proof search spaces tirelessly
- AI can translate between specification languages
- AI can learn domain patterns and apply them
- AI can work 24/7, iterating through failures

**If we can demonstrate AI producing production-quality formal proofs, we change what's economically feasible.**

---

## What "World-Class" Means

### 1. Prove Something Non-Trivial

Not toy examples. Not textbook exercises. Actual concurrent systems code with:
- Race conditions
- Memory management
- Subtle invariants
- Real-world complexity

**Target:** Prove the ABA detection in MPSAllocator is correct—that the use_count generation counter prevents corruption under all possible thread interleavings.

### 2. Discover Real Bugs

The highest value of formal verification is finding bugs that testing misses.

**Aspiration:** Use TLA+ or Lean to find an actual bug in the MPS code that wasn't caught by 24 tests and TSan. Even in "complete" code, formal methods often find edge cases.

### 3. Create Reusable Methodology

Document a repeatable process:
1. How to model concurrent C++ in TLA+
2. How to prove memory safety properties in Lean
3. How to integrate multiple verification tools
4. How AI assists at each step

**Deliverable:** A methodology paper or guide that others can follow.

### 4. Build Novel Tooling

Not just using existing tools—extending them:
- **Lean tactics** for common concurrent patterns (DCL, ABA, TLS)
- **TLA+ templates** for C++ memory model
- **Bridges** between specification languages
- **AI-assisted proof search**

### 5. Achieve Academic Rigor

Proofs that would be accepted in a peer-reviewed venue:
- Machine-checked (no hand-waving)
- Foundational (built on trusted kernels)
- Documented (clear proof strategies)

### 6. Maintain Industrial Applicability

Not just research artifacts—tools that work on real code:
- Handles Objective-C++ and Metal APIs
- Integrates with existing build systems
- Provides actionable feedback

---

## The Unique Contribution

### What Makes This Different

| Existing Work | This Project |
|---------------|--------------|
| Human-written proofs | AI-generated proofs |
| Single verification tool | Multi-tool integration |
| Toy examples | Real concurrent GPU code |
| One-off verification | Reproducible methodology |
| Academic focus | Industrial applicability |

### The Novel Thesis

> **Generative AI can produce formal proofs of concurrent systems code at a quality and speed that makes verification economically viable for mainstream software development.**

If we prove this thesis, we've contributed something significant to the field.

---

## Concrete Milestones

### Phase A: Foundation (Current)
- [x] Lean 4 project structure
- [x] TLA+ stream pool spec
- [ ] **TLC verification passing** ← BLOCKED HERE
- [ ] First non-trivial Lean proof

### Phase B: Deep Verification
- [ ] TLA+ allocator spec with ABA model
- [ ] Lean proof of ABA detection soundness
- [ ] CBMC verification of memory safety
- [ ] First Iris/Coq separation logic proof

### Phase C: Novel Contribution
- [ ] Custom Lean tactics for concurrent patterns
- [ ] AI-assisted proof search demonstration
- [ ] Bug discovery (if any exist)
- [ ] Methodology documentation

### Phase D: Publication-Ready
- [ ] Complete proof portfolio
- [ ] Reproducibility package
- [ ] Performance benchmarks
- [ ] Technical report or paper draft

---

## Quality Standards

### For TLA+ Specifications

1. **Fidelity:** Spec must model actual code, not simplified version
2. **Completeness:** All concurrent operations represented
3. **Verification:** TLC must run with explicit state counts
4. **Documentation:** Every action explained with code reference

Example quality bar:
```tla
(* GetCurrentStream models getCurrentMPSStream() from MPSStream.mm:714-765
 *
 * Key implementation details captured:
 * - Three g_pool_alive checks (lines 722, 736, 762) for TOCTOU safety
 * - pthread_main_np() detection for main thread (line 745)
 * - Round-robin stream selection via atomic counter (line 750)
 * - Fork handler invalidation (lines 575-585)
 *
 * Simplifications:
 * - dispatch_get_specific() GCD mechanism not modeled (orthogonal to safety)
 * - Stream initialization via std::call_once abstracted
 *)
GetCurrentStream(t) == ...
```

### For Lean Proofs

1. **No sorry:** Every theorem fully proved
2. **Documented strategy:** Comments explain proof approach
3. **Reusable tactics:** Extract common patterns
4. **Tested:** Proofs type-check with `lake build`

Example quality bar:
```lean
/--
ABA detection is sound: if use_count changed between first and second check,
we correctly detect potential buffer reuse and abort the operation.

Proof strategy:
1. Model the double-check pattern as state transitions
2. Show use_count is monotonically increasing (never reused)
3. Conclude that changed use_count implies different buffer instance

This corresponds to the pattern in MPSAllocator.mm getSharedBufferPtr()
lines 847-892, implementing fix for issue 32.267.
-/
theorem aba_detection_sound
    (s₁ s₂ : AllocatorState)
    (ptr : BufferPtr)
    (captured_count : Nat) :
    s₁.use_count ptr = captured_count →
    transition s₁ s₂ →
    s₂.use_count ptr ≠ captured_count →
    s₂.buffer ptr ≠ s₁.buffer ptr := by
  -- Proof proceeds by case analysis on transition type
  intro h_captured h_trans h_changed
  cases h_trans with
  | free_buffer h_free =>
    -- When buffer is freed, use_count increments
    simp [AllocatorState.use_count, h_free]
    -- New allocation gets fresh use_count
    exact fresh_use_count_different h_changed
  | reallocate h_realloc =>
    -- Reallocation at same address gets new use_count
    exact realloc_changes_identity h_realloc h_changed
```

### For Iris/Coq Proofs

1. **Foundational:** Built on Iris separation logic
2. **Modular:** One function per proof module
3. **Annotated:** RefinedC specifications inline with C
4. **Checked:** `coqc` compiles without warnings

### For Documentation

1. **Reproducible:** Anyone can clone and run
2. **Explained:** Not just what, but why
3. **Honest:** Limitations clearly stated
4. **Versioned:** Tied to specific code commits

---

## AI-Specific Practices

### What AI Does Well

1. **Exploration:** Try many proof approaches quickly
2. **Translation:** Convert between formalisms
3. **Pattern matching:** Recognize similar proofs
4. **Persistence:** Work through tedious case splits
5. **Documentation:** Generate clear explanations

### What AI Needs Help With

1. **Creativity:** Novel proof strategies
2. **Debugging:** Understanding why a proof fails
3. **Judgment:** When to simplify vs. be precise
4. **Validation:** Checking work is actually correct

### Best Practices

1. **Run the tools:** Never claim verification without execution
2. **Check outputs:** Parse and understand tool results
3. **Iterate:** Proof development is non-linear
4. **Document failures:** Failed attempts are valuable data
5. **Be skeptical:** Question every claim, including your own

---

## Success Metrics

### Quantitative

| Metric | Target |
|--------|--------|
| TLA+ specs | 3 (StreamPool, Allocator, Event) |
| TLC states explored | >100,000 per spec |
| Lean theorems proved | 10+ non-trivial |
| Lean lines of proof | 500+ |
| Coq/Iris proofs | 1+ separation logic proof |
| CBMC harnesses | 5+ |
| Bugs discovered | Any is valuable |

### Qualitative

- [ ] External reviewer can reproduce all verification
- [ ] Proofs are understandable, not just correct
- [ ] Methodology is generalizable to other projects
- [ ] Work could be submitted to a verification venue

---

## The Stakes

This isn't just about MPS parallel inference. It's about demonstrating a new paradigm:

**AI-assisted formal verification as a practical engineering tool.**

If we succeed:
- Other projects will adopt our methodology
- The economics of verification shift
- More software gets formally verified
- Fewer bugs reach production

If we fail:
- We learn what doesn't work
- We document the limitations
- We contribute negative results (also valuable)

Either way, we do it rigorously, honestly, and thoroughly.

---

## Call to Action

**Workers:** Every commit should move toward verification results, not just infrastructure.

**Standard:** If you can't show TLC output or a complete proof, you haven't verified anything.

**Ambition:** We're not building a toy. We're building a reference implementation that changes what people think is possible.

---

*"The purpose of formal verification is not to prove programs correct. It is to find bugs that no other method can find."* — paraphrased from Tony Hoare

*"Move fast and prove things."* — This project's motto

---

*Vision document created: N=982*
*Last updated: 2025-12-16*
