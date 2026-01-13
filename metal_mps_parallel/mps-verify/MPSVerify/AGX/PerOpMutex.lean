/-
  AGX Per-Operation Mutex Model - PROVES INSUFFICIENT

  Machine-checked Lean 4 proof that per-operation mutexes do NOT prevent
  the AGX driver race condition.

  Corresponds to TLA+ spec: mps-verify/specs/AGXPerOpMutex.tla

  KEY INSIGHT: Per-operation mutexes (create_mutex, encode_mutex, destroy_mutex)
  fail because different mutexes don't provide mutual exclusion. Thread A holding
  encode_mutex doesn't prevent Thread B from acquiring destroy_mutex.

  Worker: N=1476
-/

import MPSVerify.AGX.Types

namespace MPSVerify.AGX.PerOpMutex

open MPSVerify.AGX

/-- Configuration for the model -/
structure OpMutexConfig where
  numThreads : Nat
  numContextSlots : Nat
  threads_pos : numThreads > 0 := by decide
  contexts_pos : numContextSlots > 0 := by decide

/-- Per-operation mutex identifiers -/
inductive MutexType where
  | create  : MutexType  -- Mutex for context creation
  | encode  : MutexType  -- Mutex for encoding operations
  | destroy : MutexType  -- Mutex for context destruction
  deriving DecidableEq, Repr

/-- State with per-operation mutexes -/
structure PerOpMutexState (cfg : OpMutexConfig) where
  threads : Fin cfg.numThreads → ThreadInfo
  contexts : Fin cfg.numContextSlots → ContextInfo
  createMutexHolder : Option (Fin cfg.numThreads)   -- Who holds create_mutex
  encodeMutexHolder : Option (Fin cfg.numThreads)   -- Who holds encode_mutex
  destroyMutexHolder : Option (Fin cfg.numThreads)  -- Who holds destroy_mutex
  nullDerefCount : Nat
  raceWitnessed : Bool

/-- Initial state -/
def PerOpMutexState.init (cfg : OpMutexConfig) : PerOpMutexState cfg := {
  threads := fun _ => { state := .idle, context := none }
  contexts := fun _ => { state := .invalid, owner := none }
  createMutexHolder := none
  encodeMutexHolder := none
  destroyMutexHolder := none
  nullDerefCount := 0
  raceWitnessed := false
}

/-- Action result type -/
inductive ActionResult (cfg : OpMutexConfig) where
  | success : PerOpMutexState cfg → ActionResult cfg
  | blocked : ActionResult cfg

/-- Thread acquires create_mutex -/
def acquireCreateMutex (cfg : OpMutexConfig) (s : PerOpMutexState cfg)
    (t : Fin cfg.numThreads) : ActionResult cfg :=
  let ti := s.threads t
  if ti.state != .idle then .blocked
  else if s.createMutexHolder.isSome then .blocked
  else
    .success {
      threads := fun t' =>
        if t' == t then { ti with state := .creating }
        else s.threads t'
      contexts := s.contexts
      createMutexHolder := some t
      encodeMutexHolder := s.encodeMutexHolder
      destroyMutexHolder := s.destroyMutexHolder
      nullDerefCount := s.nullDerefCount
      raceWitnessed := s.raceWitnessed
    }

/-- Thread creates context and transitions to encoding -/
def createAndStartEncoding (cfg : OpMutexConfig) (s : PerOpMutexState cfg)
    (t : Fin cfg.numThreads) (slot : Fin cfg.numContextSlots) : ActionResult cfg :=
  let ti := s.threads t
  let ci := s.contexts slot
  if ti.state != .creating then .blocked
  else if s.createMutexHolder != some t then .blocked
  else if ci.state != .invalid then .blocked
  else if s.encodeMutexHolder.isSome then .blocked  -- Need encode mutex too
  else
    .success {
      threads := fun t' =>
        if t' == t then { state := .encoding, context := some slot.val }
        else s.threads t'
      contexts := fun c' =>
        if c' == slot then { state := .valid, owner := some t.val }
        else s.contexts c'
      createMutexHolder := none  -- Release create mutex
      encodeMutexHolder := some t  -- Acquire encode mutex
      destroyMutexHolder := s.destroyMutexHolder
      nullDerefCount := s.nullDerefCount
      raceWitnessed := s.raceWitnessed
    }

/-- Thread acquires destroy_mutex (while another thread holds encode_mutex!) -/
def acquireDestroyMutex (cfg : OpMutexConfig) (s : PerOpMutexState cfg)
    (t : Fin cfg.numThreads) : ActionResult cfg :=
  let ti := s.threads t
  -- THE BUG: Can acquire destroy mutex even while another thread is encoding!
  if ti.state != .idle then .blocked
  else if s.destroyMutexHolder.isSome then .blocked
  else
    .success {
      threads := fun t' =>
        if t' == t then { ti with state := .destroying }
        else s.threads t'
      contexts := s.contexts
      createMutexHolder := s.createMutexHolder
      encodeMutexHolder := s.encodeMutexHolder  -- Still held by other thread!
      destroyMutexHolder := some t
      nullDerefCount := s.nullDerefCount
      raceWitnessed := s.raceWitnessed
    }

/-- Thread destroys a context (while holding destroy_mutex, NOT encode_mutex) -/
def destroyOtherContext (cfg : OpMutexConfig) (s : PerOpMutexState cfg)
    (t : Fin cfg.numThreads) (slot : Fin cfg.numContextSlots) : ActionResult cfg :=
  let ti := s.threads t
  if ti.state != .destroying then .blocked
  else if s.destroyMutexHolder != some t then .blocked
  else
    .success {
      threads := fun t' =>
        if t' == t then { state := .idle, context := none }
        else s.threads t'
      contexts := fun c' =>
        if c' == slot then { state := .invalid, owner := none }  -- INVALIDATE!
        else s.contexts c'
      createMutexHolder := s.createMutexHolder
      encodeMutexHolder := s.encodeMutexHolder
      destroyMutexHolder := none  -- Release destroy mutex
      nullDerefCount := s.nullDerefCount
      raceWitnessed := s.raceWitnessed
    }

/-- Thread uses context while encoding - may race with destroy! -/
def useContext (cfg : OpMutexConfig) (s : PerOpMutexState cfg)
    (t : Fin cfg.numThreads) : ActionResult cfg :=
  let ti := s.threads t
  if ti.state != .encoding then .blocked
  else if s.encodeMutexHolder != some t then .blocked
  else
    match ti.context with
    | some cid =>
      if h : cid < cfg.numContextSlots then
        let slot : Fin cfg.numContextSlots := ⟨cid, h⟩
        let ci := s.contexts slot
        if ci.state == .valid then
          .success {
            threads := fun t' =>
              if t' == t then { ti with state := .destroying }
              else s.threads t'
            contexts := s.contexts
            createMutexHolder := s.createMutexHolder
            encodeMutexHolder := none  -- Release encode mutex
            destroyMutexHolder := s.destroyMutexHolder
            nullDerefCount := s.nullDerefCount
            raceWitnessed := s.raceWitnessed
          }
        else
          -- RACE! Context was invalidated while we held encode_mutex!
          .success {
            threads := fun t' =>
              if t' == t then { ti with state := .destroying }
              else s.threads t'
            contexts := s.contexts
            createMutexHolder := s.createMutexHolder
            encodeMutexHolder := none
            destroyMutexHolder := s.destroyMutexHolder
            nullDerefCount := s.nullDerefCount + 1
            raceWitnessed := true
          }
      else .blocked
    | none => .blocked

/-
  ██████╗ ██████╗  ██████╗  ██████╗ ███████╗
  ██╔══██╗██╔══██╗██╔═══██╗██╔═══██╗██╔════╝
  ██████╔╝██████╔╝██║   ██║██║   ██║█████╗
  ██╔═══╝ ██╔══██╗██║   ██║██║   ██║██╔══╝
  ██║     ██║  ██║╚██████╔╝╚██████╔╝██║
  ╚═╝     ╚═╝  ╚═╝ ╚═════╝  ╚═════╝ ╚═╝

  PROOF: Per-Operation Mutex is INSUFFICIENT

  SCENARIO:
  - 2 threads, 1 context slot
  - Thread 0: acquires create_mutex, creates context, acquires encode_mutex
  - Thread 1: acquires destroy_mutex (DIFFERENT mutex - succeeds!)
  - Thread 1: destroys Thread 0's context
  - Thread 0: uses context → NULL DEREFERENCE!
-/

/-- Example configuration -/
def exampleConfig : OpMutexConfig := {
  numThreads := 2
  numContextSlots := 1
  threads_pos := by decide
  contexts_pos := by decide
}

/-- Helper -/
def ActionResult.get! (r : ActionResult cfg) (h : ∃ s, r = .success s) : PerOpMutexState cfg :=
  match r with
  | .success s => s
  | .blocked => by simp at h

def initialState : PerOpMutexState exampleConfig := PerOpMutexState.init exampleConfig

/-- Step 1: Thread 0 acquires create_mutex -/
def step1_result : ActionResult exampleConfig :=
  acquireCreateMutex exampleConfig initialState ⟨0, by decide⟩

theorem step1_is_success : ∃ s, step1_result = .success s := by
  simp only [step1_result, acquireCreateMutex, initialState, PerOpMutexState.init]
  exact ⟨_, rfl⟩

def step1 : PerOpMutexState exampleConfig := step1_result.get! step1_is_success

/-- Step 2: Thread 0 creates context and acquires encode_mutex -/
def step2_result : ActionResult exampleConfig :=
  createAndStartEncoding exampleConfig step1 ⟨0, by decide⟩ ⟨0, by decide⟩

theorem step2_is_success : ∃ s, step2_result = .success s := by
  simp only [step2_result, createAndStartEncoding, step1, ActionResult.get!, step1_result,
             acquireCreateMutex, initialState, PerOpMutexState.init]
  exact ⟨_, rfl⟩

def step2 : PerOpMutexState exampleConfig := step2_result.get! step2_is_success

/-- Thread 0 holds encode_mutex -/
theorem step2_encode_mutex_held : step2.encodeMutexHolder = some ⟨0, by decide⟩ := by
  simp only [step2, ActionResult.get!, step2_result, createAndStartEncoding, step1, step1_result,
             acquireCreateMutex, initialState, PerOpMutexState.init]
  rfl

/-- Step 3: Thread 1 acquires destroy_mutex - SUCCEEDS! (different mutex) -/
def step3_result : ActionResult exampleConfig :=
  acquireDestroyMutex exampleConfig step2 ⟨1, by decide⟩

theorem step3_is_success : ∃ s, step3_result = .success s := by
  simp only [step3_result, acquireDestroyMutex, step2, ActionResult.get!, step2_result,
             createAndStartEncoding, step1, step1_result, acquireCreateMutex, initialState,
             PerOpMutexState.init]
  exact ⟨_, rfl⟩

def step3 : PerOpMutexState exampleConfig := step3_result.get! step3_is_success

/-- Thread 0 STILL holds encode_mutex while Thread 1 holds destroy_mutex! -/
theorem step3_both_mutexes_held :
    step3.encodeMutexHolder = some ⟨0, by decide⟩ ∧
    step3.destroyMutexHolder = some ⟨1, by decide⟩ := by
  constructor
  · simp only [step3, ActionResult.get!, step3_result, acquireDestroyMutex, step2, step2_result,
               createAndStartEncoding, step1, step1_result, acquireCreateMutex, initialState,
               PerOpMutexState.init]
    rfl
  · simp only [step3, ActionResult.get!, step3_result, acquireDestroyMutex, step2, step2_result,
               createAndStartEncoding, step1, step1_result, acquireCreateMutex, initialState,
               PerOpMutexState.init]
    rfl

/-- Step 4: Thread 1 destroys Thread 0's context! -/
def step4_result : ActionResult exampleConfig :=
  destroyOtherContext exampleConfig step3 ⟨1, by decide⟩ ⟨0, by decide⟩

theorem step4_is_success : ∃ s, step4_result = .success s := by
  simp only [step4_result, destroyOtherContext, step3, ActionResult.get!, step3_result,
             acquireDestroyMutex, step2, step2_result, createAndStartEncoding, step1,
             step1_result, acquireCreateMutex, initialState, PerOpMutexState.init]
  exact ⟨_, rfl⟩

def step4 : PerOpMutexState exampleConfig := step4_result.get! step4_is_success

/-- Context is now invalid -/
theorem step4_context_invalid : (step4.contexts ⟨0, by decide⟩).state = .invalid := by
  simp only [step4, ActionResult.get!, step4_result, destroyOtherContext, step3, step3_result,
             acquireDestroyMutex, step2, step2_result, createAndStartEncoding, step1,
             step1_result, acquireCreateMutex, initialState, PerOpMutexState.init]
  rfl

/-- Thread 0 still holds encode_mutex but context is invalid! -/
theorem step4_encode_mutex_still_held : step4.encodeMutexHolder = some ⟨0, by decide⟩ := by
  simp only [step4, ActionResult.get!, step4_result, destroyOtherContext, step3, step3_result,
             acquireDestroyMutex, step2, step2_result, createAndStartEncoding, step1,
             step1_result, acquireCreateMutex, initialState, PerOpMutexState.init]
  rfl

/-- Step 5: Thread 0 tries to use context → NULL DEREFERENCE! -/
def step5_result : ActionResult exampleConfig :=
  useContext exampleConfig step4 ⟨0, by decide⟩

theorem step5_is_success : ∃ s, step5_result = .success s := by
  simp only [step5_result, useContext, step4, ActionResult.get!, step4_result, destroyOtherContext,
             step3, step3_result, acquireDestroyMutex, step2, step2_result, createAndStartEncoding,
             step1, step1_result, acquireCreateMutex, initialState, PerOpMutexState.init]
  exact ⟨_, rfl⟩

def step5 : PerOpMutexState exampleConfig := step5_result.get! step5_is_success

/-- RACE WITNESSED -/
theorem step5_null_deref : step5.nullDerefCount = 1 := by
  simp only [step5, ActionResult.get!, step5_result, useContext, step4, step4_result,
             destroyOtherContext, step3, step3_result, acquireDestroyMutex, step2, step2_result,
             createAndStartEncoding, step1, step1_result, acquireCreateMutex, initialState,
             PerOpMutexState.init]
  rfl

theorem step5_race_witnessed : step5.raceWitnessed = true := by
  simp only [step5, ActionResult.get!, step5_result, useContext, step4, step4_result,
             destroyOtherContext, step3, step3_result, acquireDestroyMutex, step2, step2_result,
             createAndStartEncoding, step1, step1_result, acquireCreateMutex, initialState,
             PerOpMutexState.init]
  rfl

/-
  ████████╗██╗  ██╗███████╗ ██████╗ ██████╗ ███████╗███╗   ███╗
  ╚══██╔══╝██║  ██║██╔════╝██╔═══██╗██╔══██╗██╔════╝████╗ ████║
     ██║   ███████║█████╗  ██║   ██║██████╔╝█████╗  ██╔████╔██║
     ██║   ██╔══██║██╔══╝  ██║   ██║██╔══██╗██╔══╝  ██║╚██╔╝██║
     ██║   ██║  ██║███████╗╚██████╔╝██║  ██║███████╗██║ ╚═╝ ██║
     ╚═╝   ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝

  MAIN THEOREM: Per-Operation Mutex is INSUFFICIENT
-/

/-- Per-operation mutexes fail to prevent race conditions -/
theorem per_op_mutex_insufficient :
    step5.raceWitnessed = true ∧ step5.nullDerefCount > 0 := by
  constructor
  · exact step5_race_witnessed
  · simp only [step5_null_deref]; decide

/-- Corollary: encode_mutex was held but race still occurred -/
theorem race_despite_encode_mutex :
    step4_encode_mutex_still_held = step4_encode_mutex_still_held ∧
    step5.raceWitnessed = true := by
  constructor
  · rfl
  · exact step5_race_witnessed

end MPSVerify.AGX.PerOpMutex
