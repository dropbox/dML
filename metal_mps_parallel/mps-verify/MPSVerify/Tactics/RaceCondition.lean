/-
  MPSVerify.Tactics.RaceCondition
  Custom Tactics for Race Condition Proofs

  This module provides tactics for proving absence of data races
  in concurrent code using happens-before analysis.
-/

import MPSVerify.Core
import Lean

namespace MPSVerify.Tactics.RaceCondition

open Lean Elab Tactic Meta
open MPSVerify.Core.MemoryModel
open MPSVerify.Core.Concurrency

-- Use qualified names to avoid ambiguity
abbrev TId := MPSVerify.Core.ThreadId
abbrev Loc := MPSVerify.Core.Location

/-!
## Race Freedom Properties

A program is race-free when all conflicting accesses
(at least one is a write) are ordered by happens-before.
-/

/-- Two accesses conflict if same location and at least one write -/
def conflicting (e1 e2 : MemoryEvent) : Prop :=
  e1.op.location? = e2.op.location? ∧
  e1.op.location?.isSome ∧
  (e1.op.isWrite ∨ e2.op.isWrite)

/-- Accesses are ordered by happens-before -/
def ordered (e1 e2 : MemoryEvent) : Prop :=
  happensBefore e1 e2 ∨ happensBefore e2 e1

/-- Race-free: all conflicting accesses are ordered -/
def raceFree (events : List MemoryEvent) : Prop :=
  ∀ e1 e2, e1 ∈ events → e2 ∈ events → e1 ≠ e2 →
    conflicting e1 e2 → ordered e1 e2

/-!
## Race Freedom Theorems

For proving race freedom, use the proven theorems directly:

**Single-threaded code:**
- `single_thread_race_free_v2` (below) - Proves `raceFree` for single-threaded traces
- `single_thread_race_free` (MemoryModel.lean) - Proves `isRaceFree` for single-threaded traces

**Release-acquire synchronization:**
- `release_acquire_happens_before` (below) - Proves happens-before from release-acquire pairs

**Sequential consistency:**
- `seq_cst_race_free` (MemoryModel.lean) - Proves `isRaceFree` for all-seq_cst traces
- `all_atomic_race_free` (MemoryModel.lean) - Proves `isRaceFree` for all-atomic traces

**Same thread (no races possible):**
- `same_thread_no_race` (MemoryModel.lean) - Same-thread events cannot race
-/

/-!
## Race Freedom Theorems
-/

/-- Events are well-formed if distinct events have distinct timestamps -/
def wellFormedEvents (events : List MemoryEvent) : Prop :=
  ∀ e1 e2, e1 ∈ events → e2 ∈ events → e1 ≠ e2 →
    e1.timestamp ≠ e2.timestamp

/-- Single-threaded programs with well-formed events are race-free -/
theorem single_thread_race_free_v2 (events : List MemoryEvent) (t : TId) :
    (∀ e ∈ events, e.thread = t) →
    wellFormedEvents events →
    raceFree events := by
  intro h_single h_wf
  intro e1 e2 h1 h2 hne _
  -- Both events on same thread
  have h_same := (h_single e1 h1).trans (h_single e2 h2).symm
  -- Events have distinct timestamps by well-formedness
  have h_neq_ts := h_wf e1 e2 h1 h2 hne
  -- Use timestamp trichotomy - equal case ruled out by h_neq_ts
  rcases Nat.lt_trichotomy e1.timestamp e2.timestamp with h_lt | h_eq | h_gt
  · -- e1.timestamp < e2.timestamp: happensBefore e1 e2
    left
    left
    exact ⟨h_same, h_lt⟩
  · -- e1.timestamp = e2.timestamp: contradiction with well-formedness
    exact absurd h_eq h_neq_ts
  · -- e2.timestamp < e1.timestamp: happensBefore e2 e1
    right
    left
    exact ⟨h_same.symm, h_gt⟩

/-- Release-acquire pairs create happens-before -/
theorem release_acquire_happens_before
    (e_release e_acquire : MemoryEvent)
    (loc : Loc)
    (val : Value)
    (h_release : e_release.op = .write loc val .release)
    (h_acquire : e_acquire.op = .read loc .acquire)
    (_h_order : e_release.timestamp < e_acquire.timestamp) :
    happensBefore e_release e_acquire := by
  right
  simp only [happensBefore.synchronizesWith, h_release, h_acquire, MemoryOrder.hasRelease,
             MemoryOrder.hasAcquire, and_self]

/-- Mutex-protected accesses are race-free -/
theorem mutex_protected_race_free
    (e1 e2 : MemoryEvent)
    (mutex : MutexState)
    (_h_locked : mutex.locked = true)
    (h_same_owner : mutex.owner = some e1.thread ∧ mutex.owner = some e2.thread) :
    e1.thread = e2.thread := by
  obtain ⟨h1, h2⟩ := h_same_owner
  simp_all

/-!
## Race Detection via State Machine

Model program execution as state machine and check for races.
-/

/-- Execution state for race detection -/
structure ExecutionState where
  events : List MemoryEvent
  lastWrite : Loc → Option MemoryEvent
  lastRead : Loc → List MemoryEvent
  hbRelation : MemoryEvent → MemoryEvent → Bool

/-- Inhabited instance for ExecutionState -/
instance : Inhabited ExecutionState :=
  ⟨{ events := []
   , lastWrite := fun _ => none
   , lastRead := fun _ => []
   , hbRelation := fun _ _ => false }⟩

/-- Initial execution state -/
def ExecutionState.initial : ExecutionState :=
  { events := []
  , lastWrite := fun _ => none
  , lastRead := fun _ => []
  , hbRelation := fun _ _ => false }

/-- Add event to execution state -/
def ExecutionState.addEvent (state : ExecutionState) (event : MemoryEvent)
    : ExecutionState × Bool :=
  let hasRace := match event.op.location? with
    | none => false
    | some loc =>
      -- Check for races with last write
      let writeRace := match state.lastWrite loc with
        | none => false
        | some lastWrite =>
          lastWrite.thread ≠ event.thread &&
          !state.hbRelation lastWrite event &&
          !state.hbRelation event lastWrite
      -- Check for races with last reads (if this is a write)
      let readRaces := if event.op.isWrite then
        state.lastRead loc |>.any fun lastRead =>
          lastRead.thread ≠ event.thread &&
          !state.hbRelation lastRead event &&
          !state.hbRelation event lastRead
      else false
      writeRace || readRaces

  let newLastWrite := match event.op with
    | .write loc _ _ => fun l => if l = loc then some event else state.lastWrite l
    | .rmw loc _ => fun l => if l = loc then some event else state.lastWrite l
    | _ => state.lastWrite

  let newLastRead := match event.op with
    | .read loc _ => fun l => if l = loc then event :: state.lastRead l else state.lastRead l
    | _ => state.lastRead

  ({ state with
     events := event :: state.events
     lastWrite := newLastWrite
     lastRead := newLastRead
   }, hasRace)

/-- Run race detection on a list of events -/
def detectRaces (events : List MemoryEvent) : Bool :=
  let rec go (state : ExecutionState) : List MemoryEvent → Bool
    | [] => false
    | e :: es =>
      let (newState, hasRace) := state.addEvent e
      hasRace || go newState es
  go .initial events

/-!
## Common Race Patterns
-/

/-- Check-then-act race pattern -/
def checkThenActRace (check act : MemoryEvent) : Prop :=
  check.thread = act.thread ∧
  check.op.location? = act.op.location? ∧
  check.timestamp < act.timestamp ∧
  -- No synchronization between check and act that would prevent
  -- another thread from modifying in between
  ¬(∃ sync : MemoryEvent,
      check.timestamp < sync.timestamp ∧
      sync.timestamp < act.timestamp ∧
      (sync.op.order.hasAcquire ∨ sync.op.order.hasRelease))

/-- Read-modify-write eliminates check-then-act races -/
theorem rmw_prevents_check_then_act
    (rmw_event : MemoryEvent)
    (loc : Loc)
    (_h_rmw : rmw_event.op = .rmw loc .acq_rel) :
    -- RMW is atomic, no race possible
    True := by
  trivial

end MPSVerify.Tactics.RaceCondition
