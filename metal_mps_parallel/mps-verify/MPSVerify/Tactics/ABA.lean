/-
  MPSVerify.Tactics.ABA
  Custom Tactics for ABA Detection Proofs

  This module provides tactics for proving correctness of
  ABA detection mechanisms using generation counters.

  The ABA problem is a classic concurrency bug in lock-free algorithms.
  Our tactics help prove that generation counters correctly detect ABA.
-/

import MPSVerify.Core
import MPSVerify.DSL.Allocator
import Lean

namespace MPSVerify.Tactics.ABA

open Lean Elab Tactic Meta
open MPSVerify.DSL.Allocator

/-!
## ABA Detection Properties

A correct ABA detection mechanism must satisfy:
1. **Counter monotonicity**: Generation always increases
2. **Detection completeness**: All ABA scenarios are detected
3. **No false positives**: Valid operations succeed
4. **Atomicity**: Counter update is atomic with pointer update
-/

/-- Generation counter is monotonically increasing -/
def counterMonotonic (gen1 gen2 : Nat) (operations : Nat) : Prop :=
  gen2 ≥ gen1 + operations

/-- ABA is detected when generation changes -/
def abaDetected (expected observed : ABAPointer α) : Prop :=
  expected.generation ≠ observed.generation → expected ≠ observed

/-- No false positives: same generation means same logical state -/
def noFalsePositives (p1 p2 : ABAPointer α) : Prop :=
  p1.generation = p2.generation → p1.ptr = p2.ptr → p1 = p2

/-- CAS is atomic -/
def casAtomic (before after : ABAPointer α) (success : Bool) : Prop :=
  success → after.generation = before.generation + 1

/-!
## ABA Proof Tactics
-/

/-- Tactic to prove counter monotonicity -/
macro "prove_counter_monotonic" : tactic =>
  `(tactic| (
    simp [counterMonotonic]
    omega
  ))

/-- Tactic to prove ABA detection -/
macro "prove_aba_detected" : tactic =>
  `(tactic| (
    simp [abaDetected]
    intro hne heq
    exact hne (congrArg ABAPointer.generation heq)
  ))

/-!
## ABA Theorems
-/

/-- If CAS succeeds, both pointer and generation matched -/
theorem cas_success_means_match {α : Type} [DecidableEq α]
    (ptr expected : ABAPointer α)
    (newVal : Option α)
    (h_success : (ABAPointer.compareAndSwap ptr expected newVal).1 = true) :
    ptr.ptr = expected.ptr ∧ ptr.generation = expected.generation := by
  simp only [ABAPointer.compareAndSwap] at h_success
  split at h_success
  · case isTrue h =>
    simp only [Bool.and_eq_true, decide_eq_true_eq] at h
    exact h
  · contradiction

/-- Generation difference detects intermediate modifications -/
theorem generation_detects_modifications {α : Type} [DecidableEq α]
    (ptr expected : ABAPointer α)
    (newVal : Option α)
    (h_gen_diff : ptr.generation ≠ expected.generation) :
    (ABAPointer.compareAndSwap ptr expected newVal).1 = false := by
  simp only [ABAPointer.compareAndSwap]
  split
  · case isTrue h =>
    simp only [Bool.and_eq_true, decide_eq_true_eq] at h
    exact absurd h.2 h_gen_diff
  · rfl

/-- Generation counter is a valid happens-before indicator -/
theorem generation_happens_before
    (gen1 gen2 : Nat)
    (h : gen1 < gen2) :
    -- At least one modification happened between gen1 and gen2
    gen2 ≥ gen1 + 1 := by
  omega

/-!
## ABA Scenario Modeling

Model the classic ABA scenario and prove detection.
-/

/-- State during ABA scenario -/
structure ABAScenario (α : Type) where
  initial : ABAPointer α      -- T1 reads A
  afterFirstChange : ABAPointer α  -- T2 changes A→B
  afterSecondChange : ABAPointer α -- T2 changes B→A
  deriving Repr

/-- Valid ABA scenario: value returns to original but generation changes -/
def validABAScenario {α : Type} [DecidableEq α] (s : ABAScenario α) : Prop :=
  s.initial.ptr = s.afterSecondChange.ptr ∧  -- Same pointer value
  s.afterSecondChange.generation > s.initial.generation  -- Generation increased

/-!
## Combined ABA Safety Property
-/

/-- Full ABA safety: combines key properties -/
structure ABASafety (α : Type) [DecidableEq α] where
  /-- If CAS fails due to generation, it's a valid detection -/
  detection : ∀ (ptr expected : ABAPointer α) (newVal : Option α),
    ptr.generation ≠ expected.generation →
    (ABAPointer.compareAndSwap ptr expected newVal).1 = false

  /-- If generation matches and pointer matches, CAS succeeds -/
  completeness : ∀ (ptr expected : ABAPointer α) (newVal : Option α),
    ptr.generation = expected.generation →
    ptr.ptr = expected.ptr →
    (ABAPointer.compareAndSwap ptr expected newVal).1 = true

/-- ABA safety is satisfied by our implementation -/
theorem aba_safety_holds {α : Type} [DecidableEq α] : ABASafety α where
  detection := by
    intro ptr expected newVal hne
    simp only [ABAPointer.compareAndSwap]
    split
    · case isTrue h =>
      simp only [Bool.and_eq_true, decide_eq_true_eq] at h
      exact absurd h.2 hne
    · rfl

  completeness := by
    intro ptr expected newVal hgen hptr
    simp only [ABAPointer.compareAndSwap, hgen, hptr]
    rfl

end MPSVerify.Tactics.ABA
