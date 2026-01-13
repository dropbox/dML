/-
  AGX Driver Race Condition - Buggy Model

  Machine-checked Lean 4 proof that the AGX driver design (as inferred from
  reverse engineering) contains a race condition that causes NULL pointer
  dereferences.

  This corresponds to TLA+ spec: mps-verify/specs/AGXContextRace.tla

  KEY THEOREM: race_condition_exists
  - Proves that starting from a valid initial state, there exists a sequence
    of legal transitions that leads to a NULL pointer dereference.
-/

import MPSVerify.AGX.Types

namespace MPSVerify.AGX.Race

open MPSVerify.AGX

/-- Action result type for state transitions -/
inductive ActionResult (cfg : Config) where
  | success : BuggyState cfg → ActionResult cfg
  | blocked : ActionResult cfg

/-- Action: Thread starts creating a context (allocates a slot) -/
def startCreateContext (cfg : Config) (s : BuggyState cfg) (t : Fin cfg.numThreads)
    (slot : Fin cfg.numContextSlots) : ActionResult cfg :=
  let ti := s.threads t
  let ci := s.contexts slot
  -- Preconditions: thread is idle, has no context, slot is invalid
  if ti.state == .idle && ti.context.isNone && ci.state == .invalid then
    .success {
      threads := fun t' =>
        if t' == t then { state := .creating, context := some slot.val }
        else s.threads t'
      contexts := fun c' =>
        if c' == slot then { state := .invalid, owner := some t.val }
        else s.contexts c'
      nullDerefCount := s.nullDerefCount
      raceWitnessed := s.raceWitnessed
    }
  else .blocked

/-- Action: Thread finishes creating context (marks it valid) -/
def finishCreateContext (cfg : Config) (s : BuggyState cfg) (t : Fin cfg.numThreads)
    : ActionResult cfg :=
  let ti := s.threads t
  -- Precondition: thread is creating with a valid slot reference
  match ti.state, ti.context with
  | .creating, some cid =>
    if h : cid < cfg.numContextSlots then
      let slot : Fin cfg.numContextSlots := ⟨cid, h⟩
      .success {
        threads := fun t' =>
          if t' == t then { ti with state := .encoding }
          else s.threads t'
        contexts := fun c' =>
          if c' == slot then { (s.contexts slot) with state := .valid }
          else s.contexts c'
        nullDerefCount := s.nullDerefCount
        raceWitnessed := s.raceWitnessed
      }
    else .blocked
  | _, _ => .blocked

/-- Action: Thread uses its context (THE CRASH POINT if context is invalid) -/
def useContext (cfg : Config) (s : BuggyState cfg) (t : Fin cfg.numThreads)
    : ActionResult cfg :=
  let ti := s.threads t
  -- Precondition: thread is encoding
  if ti.state != .encoding then .blocked
  else
    match ti.context with
    | some cid =>
      if h : cid < cfg.numContextSlots then
        let slot : Fin cfg.numContextSlots := ⟨cid, h⟩
        let ci := s.contexts slot
        -- Check if context is still valid
        if ci.state == .valid then
          -- Normal operation: no crash
          .success {
            threads := fun t' =>
              if t' == t then { ti with state := .destroying }
              else s.threads t'
            contexts := s.contexts
            nullDerefCount := s.nullDerefCount
            raceWitnessed := s.raceWitnessed
          }
        else
          -- BUG: Context was invalidated by another thread!
          -- This is the NULL pointer dereference
          .success {
            threads := fun t' =>
              if t' == t then { ti with state := .destroying }
              else s.threads t'
            contexts := s.contexts
            nullDerefCount := s.nullDerefCount + 1
            raceWitnessed := true
          }
      else .blocked
    | none => .blocked

/-- Action: Thread destroys its own context -/
def destroyContext (cfg : Config) (s : BuggyState cfg) (t : Fin cfg.numThreads)
    : ActionResult cfg :=
  let ti := s.threads t
  -- Precondition: thread is destroying with a valid slot reference
  if ti.state != .destroying then .blocked
  else
    match ti.context with
    | some cid =>
      if h : cid < cfg.numContextSlots then
        let slot : Fin cfg.numContextSlots := ⟨cid, h⟩
        .success {
          threads := fun t' =>
            if t' == t then { state := .idle, context := none }
            else s.threads t'
          contexts := fun c' =>
            if c' == slot then { state := .invalid, owner := none }
            else s.contexts c'
          nullDerefCount := s.nullDerefCount
          raceWitnessed := s.raceWitnessed
        }
      else .blocked
    | none => .blocked

/-- Action: Thread A invalidates Thread B's context (THE BUG) -/
def destroyOtherContext (cfg : Config) (s : BuggyState cfg)
    (attacker : Fin cfg.numThreads) (victim_slot : Fin cfg.numContextSlots)
    : ActionResult cfg :=
  let ci := s.contexts victim_slot
  let attackerInfo := s.threads attacker
  if attackerInfo.state != .idle then .blocked
  else if ci.state != .valid then .blocked
  else
    match ci.owner with
    | some owner_id =>
      if owner_id != attacker.val then
        -- THE BUG: Invalidate without checking if owner is using it
        .success {
          threads := s.threads
          contexts := fun c' =>
            if c' == victim_slot then { ci with state := .invalid }
            else s.contexts c'
          nullDerefCount := s.nullDerefCount
          raceWitnessed := s.raceWitnessed
        }
      else .blocked
    | none => .blocked

/-- Example configuration: 2 threads, 2 context slots -/
def exampleConfig : Config := {
  numThreads := 2
  numContextSlots := 2
  threads_pos := by decide
  contexts_pos := by decide
}

/-- Initial state for the example -/
def exampleInit : BuggyState exampleConfig := BuggyState.init exampleConfig

/-
  We construct the race scenario step by step, proving properties along the way.
-/

/-- Helper: extract success state or panic -/
def ActionResult.get! (r : ActionResult cfg) (h : ∃ s, r = .success s) : BuggyState cfg :=
  match r with
  | .success s => s
  | .blocked => by simp at h

/-- Step 1: Thread 0 starts creating context in slot 0 -/
def step1_result : ActionResult exampleConfig :=
  startCreateContext exampleConfig exampleInit ⟨0, by decide⟩ ⟨0, by decide⟩

theorem step1_is_success : ∃ s, step1_result = .success s := by
  simp only [step1_result, startCreateContext, exampleInit, BuggyState.init]
  exact ⟨_, rfl⟩

def step1 : BuggyState exampleConfig := step1_result.get! step1_is_success

/-- Step 1 verification: Thread 0 is now creating -/
theorem step1_thread0_creating : (step1.threads ⟨0, by decide⟩).state = .creating := by
  simp only [step1, ActionResult.get!, step1_result, startCreateContext, exampleInit, BuggyState.init]
  rfl

/-- Step 2: Thread 0 finishes creating context -/
def step2_result : ActionResult exampleConfig := finishCreateContext exampleConfig step1 ⟨0, by decide⟩

theorem step2_is_success : ∃ s, step2_result = .success s := by
  simp only [step2_result, finishCreateContext, step1, ActionResult.get!, step1_result,
             startCreateContext, exampleInit, BuggyState.init]
  exact ⟨_, rfl⟩

def step2 : BuggyState exampleConfig := step2_result.get! step2_is_success

/-- Step 2 verification: Thread 0 is now encoding -/
theorem step2_thread0_encoding : (step2.threads ⟨0, by decide⟩).state = .encoding := by
  simp only [step2, ActionResult.get!, step2_result, finishCreateContext, step1,
             step1_result, startCreateContext, exampleInit, BuggyState.init]
  rfl

/-- Step 2 verification: Context 0 is valid -/
theorem step2_context0_valid : (step2.contexts ⟨0, by decide⟩).state = .valid := by
  simp only [step2, ActionResult.get!, step2_result, finishCreateContext, step1,
             step1_result, startCreateContext, exampleInit, BuggyState.init]
  rfl

/-- Step 3: Thread 1 (idle) destroys Thread 0's context - THE BUG -/
def step3_result : ActionResult exampleConfig :=
  destroyOtherContext exampleConfig step2 ⟨1, by decide⟩ ⟨0, by decide⟩

theorem step3_is_success : ∃ s, step3_result = .success s := by
  simp only [step3_result, destroyOtherContext, step2, ActionResult.get!, step2_result,
             finishCreateContext, step1, step1_result, startCreateContext, exampleInit, BuggyState.init]
  exact ⟨_, rfl⟩

def step3 : BuggyState exampleConfig := step3_result.get! step3_is_success

/-- Step 3 verification: Context 0 is now INVALID (destroyed by Thread 1) -/
theorem step3_context0_invalid : (step3.contexts ⟨0, by decide⟩).state = .invalid := by
  simp only [step3, ActionResult.get!, step3_result, destroyOtherContext, step2,
             step2_result, finishCreateContext, step1, step1_result, startCreateContext,
             exampleInit, BuggyState.init]
  rfl

/-- Step 3 verification: Thread 0 is STILL encoding (doesn't know context is invalid) -/
theorem step3_thread0_still_encoding : (step3.threads ⟨0, by decide⟩).state = .encoding := by
  simp only [step3, ActionResult.get!, step3_result, destroyOtherContext, step2,
             step2_result, finishCreateContext, step1, step1_result, startCreateContext,
             exampleInit, BuggyState.init]
  rfl

/-- Step 4: Thread 0 tries to use context → CRASH -/
def step4_result : ActionResult exampleConfig := useContext exampleConfig step3 ⟨0, by decide⟩

theorem step4_is_success : ∃ s, step4_result = .success s := by
  simp only [step4_result, useContext, step3, ActionResult.get!, step3_result,
             destroyOtherContext, step2, step2_result, finishCreateContext, step1,
             step1_result, startCreateContext, exampleInit, BuggyState.init]
  exact ⟨_, rfl⟩

def step4 : BuggyState exampleConfig := step4_result.get! step4_is_success

/-- THE CRASH: NULL dereference detected -/
theorem step4_null_deref : step4.nullDerefCount = 1 := by
  simp only [step4, ActionResult.get!, step4_result, useContext, step3, step3_result,
             destroyOtherContext, step2, step2_result, finishCreateContext, step1,
             step1_result, startCreateContext, exampleInit, BuggyState.init]
  rfl

/-- Race condition was witnessed -/
theorem step4_race_witnessed : step4.raceWitnessed = true := by
  simp only [step4, ActionResult.get!, step4_result, useContext, step3, step3_result,
             destroyOtherContext, step2, step2_result, finishCreateContext, step1,
             step1_result, startCreateContext, exampleInit, BuggyState.init]
  rfl

/-
  ██████╗  █████╗  ██████╗███████╗     ██████╗ ██████╗  ██████╗ ██╗   ██╗███████╗███╗   ██╗
  ██╔══██╗██╔══██╗██╔════╝██╔════╝     ██╔══██╗██╔══██╗██╔═══██╗██║   ██║██╔════╝████╗  ██║
  ██████╔╝███████║██║     █████╗       ██████╔╝██████╔╝██║   ██║██║   ██║█████╗  ██╔██╗ ██║
  ██╔══██╗██╔══██║██║     ██╔══╝       ██╔═══╝ ██╔══██╗██║   ██║╚██╗ ██╔╝██╔══╝  ██║╚██╗██║
  ██║  ██║██║  ██║╚██████╗███████╗     ██║     ██║  ██║╚██████╔╝ ╚████╔╝ ███████╗██║ ╚████║
  ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚══════╝     ╚═╝     ╚═╝  ╚═╝ ╚═════╝   ╚═══╝  ╚══════╝╚═╝  ╚═══╝

  MAIN THEOREM: The AGX driver race condition EXISTS.

  This is machine-checked proof that:
  1. Starting from a valid initial state
  2. Following only legal transitions (as modeled)
  3. We can reach a state with nullDerefCount > 0

  This matches the empirically observed crash behavior at 55% crash rate.
-/

/-- The race condition exists: there is a reachable state with a NULL dereference -/
theorem race_condition_exists :
    step4.raceWitnessed = true ∧ step4.nullDerefCount > 0 := by
  constructor
  · exact step4_race_witnessed
  · simp only [step4_null_deref]
    decide

/-- Corollary: The buggy design CAN produce NULL pointer dereferences -/
theorem buggy_design_can_crash :
    ∃ (s : BuggyState exampleConfig), s.nullDerefCount > 0 :=
  ⟨step4, by rw [step4_null_deref]; exact Nat.zero_lt_one⟩

end MPSVerify.AGX.Race
