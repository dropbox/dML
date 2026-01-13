/-
  MPSVerify.Tactics.DCL
  Custom Tactics for Double-Check Locking Proofs

  This module provides tactics for proving correctness of
  double-check locking (DCL) patterns used in the MPS allocator.

  DCL is notoriously tricky to get right without proper memory barriers.
  These tactics help prove that our implementation is correct.
-/

import MPSVerify.Core
import MPSVerify.DSL.Allocator
import Lean

namespace MPSVerify.Tactics.DCL

open Lean Elab Tactic Meta
open MPSVerify.Core.MemoryModel
open MPSVerify.DSL.Allocator

/-!
## DCL Safety Properties

A correct DCL implementation must satisfy:
1. **First check sufficient**: If initialized flag is set, value is valid
2. **Second check consistent**: Lock acquisition synchronizes properly
3. **No TOCTOU**: Time-of-check-time-of-use race prevented
4. **Publication safety**: Initialized value visible to all threads
-/

/-- First check in DCL is sufficient when true -/
def firstCheckSufficient {α : Type} (state : DCLState α) : Prop :=
  state.isInitialized → state.getValue?.isSome

/-- Second check under lock is consistent -/
def secondCheckConsistent {α : Type} (state : DCLState α) : Prop :=
  ∀ v, state = .initialized v → state.getValue? = some v

/-- No TOCTOU vulnerability -/
def noTOCTOU {α : Type} (state1 state2 : DCLState α) : Prop :=
  state1.isInitialized → state2.isInitialized

/-- Publication is safe (value visible after release) -/
def publicationSafe {α : Type} (state : DCLState α) (order : MemoryOrder) : Prop :=
  state.isInitialized → order.hasRelease

/-!
## DCL Proof Tactics
-/

/-- Tactic to prove first check sufficiency -/
macro "prove_first_check_sufficient" : tactic =>
  `(tactic| (
    intro h
    simp [DCLState.isInitialized, DCLState.getValue?] at *
    cases ‹DCLState _› <;> simp_all
  ))

/-- Tactic to prove second check consistency -/
macro "prove_second_check_consistent" : tactic =>
  `(tactic| (
    intro v heq
    simp [DCLState.getValue?, heq]
  ))

/-- Combined tactic for DCL safety proof -/
macro "prove_dcl_safe" : tactic =>
  `(tactic| (
    constructor
    · prove_first_check_sufficient
    · prove_second_check_consistent
  ))

/-!
## DCL Theorems
-/

/-- First check is sufficient for initialized state -/
theorem dcl_first_check_ok {α : Type} (state : DCLState α) :
    firstCheckSufficient state := by
  intro h
  simp [DCLState.isInitialized] at h
  cases state with
  | uninitialized => contradiction
  | initializing t => contradiction
  | initialized v => simp [DCLState.getValue?]

/-- Second check is consistent -/
theorem dcl_second_check_ok {α : Type} (state : DCLState α) :
    secondCheckConsistent state := by
  intro v heq
  simp [heq, DCLState.getValue?]

/-- DCL never loses an initialized value -/
theorem dcl_preserves_initialization {α : Type} [Inhabited α]
    (state : DCLState α)
    (thread : ThreadId)
    (init : Unit → α)
    (h_init : state.isInitialized) :
    (dclAcquire state thread init).2.isInitialized := by
  simp [dclAcquire, DCLState.isInitialized] at *
  cases state with
  | uninitialized => contradiction
  | initializing t => contradiction
  | initialized v => rfl

/-- After DCL acquire returns gotExisting, the state is still initialized -/
theorem dcl_gotExisting_means_initialized {α : Type} [Inhabited α]
    (state : DCLState α)
    (thread : ThreadId)
    (init : Unit → α)
    (v : α)
    (h : (dclAcquire state thread init).1 = .gotExisting v) :
    state.isInitialized := by
  simp [dclAcquire] at h
  split at h
  · simp [DCLState.isInitialized]
  · split at h <;> contradiction
  · contradiction

/-!
## Memory Barrier Verification

DCL requires proper memory barriers to work correctly.
-/

/-- Memory barrier configuration for DCL -/
structure DCLBarriers where
  loadOrder : MemoryOrder    -- For first check (should be acquire or seq_cst)
  storeOrder : MemoryOrder   -- For initialization (should be release or seq_cst)
  deriving Repr, Inhabited

/-- Check if DCL barriers are sufficient -/
def DCLBarriers.areSufficient (b : DCLBarriers) : Bool :=
  b.loadOrder.hasAcquire && b.storeOrder.hasRelease

/-- Recommended DCL barriers (seq_cst is always safe) -/
def DCLBarriers.seqCst : DCLBarriers :=
  { loadOrder := .seq_cst, storeOrder := .seq_cst }

/-- Minimal correct DCL barriers -/
def DCLBarriers.minimal : DCLBarriers :=
  { loadOrder := .acquire, storeOrder := .release }

/-- Seq_cst barriers are sufficient -/
theorem seqcst_barriers_sufficient :
    DCLBarriers.seqCst.areSufficient = true := by
  simp [DCLBarriers.seqCst, DCLBarriers.areSufficient, MemoryOrder.hasAcquire, MemoryOrder.hasRelease]

/-- Minimal barriers are sufficient -/
theorem minimal_barriers_sufficient :
    DCLBarriers.minimal.areSufficient = true := by
  simp [DCLBarriers.minimal, DCLBarriers.areSufficient, MemoryOrder.hasAcquire, MemoryOrder.hasRelease]

end MPSVerify.Tactics.DCL
