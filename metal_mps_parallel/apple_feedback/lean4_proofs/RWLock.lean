/-
  AGX Reader-Writer Lock Model - PROVES INSUFFICIENT

  Machine-checked Lean 4 proof that reader-writer locks do NOT prevent
  the AGX driver race condition.

  Corresponds to TLA+ spec: mps-verify/specs/AGXRWLock.tla

  KEY INSIGHT: RW locks fail because async completion handlers (GPU completion,
  command buffer dealloc) don't use our user-space locks. They run on system
  threads we don't control.

  The race is:
  1. Thread A: read_lock(context) - start encoding
  2. Async handler: (doesn't use our lock!) invalidates context
  3. Thread A: use context → NULL DEREFERENCE!

  Worker: N=1476
-/

import MPSVerify.AGX.Types

namespace MPSVerify.AGX.RWLock

open MPSVerify.AGX

/-- Configuration for the model -/
structure RWLockConfig where
  numThreads : Nat       -- User-space threads
  numAsyncHandlers : Nat -- System threads running completion handlers
  numContextSlots : Nat
  threads_pos : numThreads > 0 := by decide
  handlers_pos : numAsyncHandlers > 0 := by decide
  contexts_pos : numContextSlots > 0 := by decide

/-- Reader-writer lock state -/
structure RWLockState where
  readers : Nat          -- Number of threads holding read lock
  writerHeld : Bool      -- Is write lock held?
  writerWaiting : Bool   -- Is a writer waiting?

/-- Initial RW lock state -/
def RWLockState.init : RWLockState := {
  readers := 0
  writerHeld := false
  writerWaiting := false
}

/-- Extended thread state for RW lock model -/
inductive RWThreadState where
  | idle : RWThreadState
  | readLocking : RWThreadState   -- Acquiring read lock
  | encoding : RWThreadState      -- Holding read lock, encoding
  | done : RWThreadState          -- Finished encoding
  deriving DecidableEq, Repr

/-- Async handler state -/
inductive AsyncHandlerState where
  | idle : AsyncHandlerState
  | pendingCompletion : AsyncHandlerState  -- GPU work completed, will invalidate
  | invalidating : AsyncHandlerState       -- Actively invalidating context
  deriving DecidableEq, Repr

/-- Thread info for RW lock model -/
structure RWThreadInfo where
  state : RWThreadState
  context : Option ContextId

/-- Full system state -/
structure RWLockSystemState (cfg : RWLockConfig) where
  threads : Fin cfg.numThreads → RWThreadInfo
  asyncHandlers : Fin cfg.numAsyncHandlers → AsyncHandlerState
  contexts : Fin cfg.numContextSlots → ContextInfo
  lock : RWLockState
  nullDerefCount : Nat
  raceWitnessed : Bool

/-- Initial state -/
def RWLockSystemState.init (cfg : RWLockConfig) : RWLockSystemState cfg := {
  threads := fun _ => { state := .idle, context := none }
  asyncHandlers := fun _ => .idle
  contexts := fun _ => { state := .invalid, owner := none }
  lock := RWLockState.init
  nullDerefCount := 0
  raceWitnessed := false
}

/-- Action result type -/
inductive ActionResult (cfg : RWLockConfig) where
  | success : RWLockSystemState cfg → ActionResult cfg
  | blocked : ActionResult cfg

/-- Thread acquires read lock (standard RW lock semantics) -/
def acquireReadLock (cfg : RWLockConfig) (s : RWLockSystemState cfg)
    (t : Fin cfg.numThreads) : ActionResult cfg :=
  let ti := s.threads t
  if ti.state != .idle then .blocked
  else if s.lock.writerHeld || s.lock.writerWaiting then .blocked
  else
    .success {
      threads := fun t' =>
        if t' == t then { ti with state := .readLocking }
        else s.threads t'
      asyncHandlers := s.asyncHandlers
      contexts := s.contexts
      lock := { s.lock with readers := s.lock.readers + 1 }
      nullDerefCount := s.nullDerefCount
      raceWitnessed := s.raceWitnessed
    }

/-- Thread creates context while holding read lock -/
def createContextWithReadLock (cfg : RWLockConfig) (s : RWLockSystemState cfg)
    (t : Fin cfg.numThreads) (slot : Fin cfg.numContextSlots) : ActionResult cfg :=
  let ti := s.threads t
  let ci := s.contexts slot
  if ti.state != .readLocking then .blocked
  else if ci.state != .invalid then .blocked
  else
    .success {
      threads := fun t' =>
        if t' == t then { state := .encoding, context := some slot.val }
        else s.threads t'
      asyncHandlers := s.asyncHandlers
      contexts := fun c' =>
        if c' == slot then { state := .valid, owner := some t.val }
        else s.contexts c'
      lock := s.lock  -- Still holding read lock
      nullDerefCount := s.nullDerefCount
      raceWitnessed := s.raceWitnessed
    }

/-
  THE BUG: Async handlers don't respect user-space locks!

  GPU completion handlers and command buffer dealloc routines run on
  system threads that don't call our read_lock/write_lock functions.
  They directly manipulate context state.
-/

/-- Async handler triggers (simulates GPU completion or dealloc) -/
def asyncHandlerTrigger (cfg : RWLockConfig) (s : RWLockSystemState cfg)
    (h : Fin cfg.numAsyncHandlers) : ActionResult cfg :=
  let hi := s.asyncHandlers h
  if hi != .idle then .blocked
  else
    .success {
      threads := s.threads
      asyncHandlers := fun h' =>
        if h' == h then .pendingCompletion
        else s.asyncHandlers h'
      contexts := s.contexts
      lock := s.lock  -- Async handler IGNORES our lock!
      nullDerefCount := s.nullDerefCount
      raceWitnessed := s.raceWitnessed
    }

/-- Async handler invalidates context (DOESN'T CHECK OUR LOCK!) -/
def asyncHandlerInvalidate (cfg : RWLockConfig) (s : RWLockSystemState cfg)
    (h : Fin cfg.numAsyncHandlers) (slot : Fin cfg.numContextSlots)
    : ActionResult cfg :=
  let hi := s.asyncHandlers h
  if hi != .pendingCompletion then .blocked
  else
    -- THE KEY: This happens regardless of who holds read locks!
    .success {
      threads := s.threads
      asyncHandlers := fun h' =>
        if h' == h then .idle
        else s.asyncHandlers h'
      contexts := fun c' =>
        if c' == slot then { state := .invalid, owner := none }  -- INVALIDATE!
        else s.contexts c'
      lock := s.lock  -- Lock state unchanged - we bypassed it!
      nullDerefCount := s.nullDerefCount
      raceWitnessed := s.raceWitnessed
    }

/-- Thread uses context (may race with async handler!) -/
def useContext (cfg : RWLockConfig) (s : RWLockSystemState cfg)
    (t : Fin cfg.numThreads) : ActionResult cfg :=
  let ti := s.threads t
  if ti.state != .encoding then .blocked
  else
    match ti.context with
    | some cid =>
      if h : cid < cfg.numContextSlots then
        let slot : Fin cfg.numContextSlots := ⟨cid, h⟩
        let ci := s.contexts slot
        if ci.state == .valid then
          .success {
            threads := fun t' =>
              if t' == t then { ti with state := .done }
              else s.threads t'
            asyncHandlers := s.asyncHandlers
            contexts := s.contexts
            lock := { s.lock with readers := s.lock.readers - 1 }
            nullDerefCount := s.nullDerefCount
            raceWitnessed := s.raceWitnessed
          }
        else
          -- RACE! Async handler invalidated while we held read lock!
          .success {
            threads := fun t' =>
              if t' == t then { ti with state := .done }
              else s.threads t'
            asyncHandlers := s.asyncHandlers
            contexts := s.contexts
            lock := { s.lock with readers := s.lock.readers - 1 }
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

  PROOF: Reader-Writer Lock is INSUFFICIENT

  SCENARIO:
  - 1 user thread, 1 async handler, 1 context slot
  - Thread 0: read_lock(), create context, start encoding
  - Async handler: (doesn't use lock!) invalidates context
  - Thread 0: use context → NULL DEREFERENCE!
-/

/-- Example configuration -/
def exampleConfig : RWLockConfig := {
  numThreads := 1
  numAsyncHandlers := 1
  numContextSlots := 1
  threads_pos := by decide
  handlers_pos := by decide
  contexts_pos := by decide
}

/-- Helper -/
def ActionResult.get! (r : ActionResult cfg) (h : ∃ s, r = .success s) : RWLockSystemState cfg :=
  match r with
  | .success s => s
  | .blocked => by simp at h

def initialState : RWLockSystemState exampleConfig := RWLockSystemState.init exampleConfig

/-- Step 1: Thread 0 acquires read lock -/
def step1_result : ActionResult exampleConfig :=
  acquireReadLock exampleConfig initialState ⟨0, by decide⟩

theorem step1_is_success : ∃ s, step1_result = .success s := by
  simp only [step1_result, acquireReadLock, initialState, RWLockSystemState.init, RWLockState.init]
  exact ⟨_, rfl⟩

def step1 : RWLockSystemState exampleConfig := step1_result.get! step1_is_success

/-- Step 2: Thread 0 creates context -/
def step2_result : ActionResult exampleConfig :=
  createContextWithReadLock exampleConfig step1 ⟨0, by decide⟩ ⟨0, by decide⟩

theorem step2_is_success : ∃ s, step2_result = .success s := by
  simp only [step2_result, createContextWithReadLock, step1, ActionResult.get!, step1_result,
             acquireReadLock, initialState, RWLockSystemState.init, RWLockState.init]
  exact ⟨_, rfl⟩

def step2 : RWLockSystemState exampleConfig := step2_result.get! step2_is_success

/-- Thread 0 holds read lock -/
theorem step2_read_lock_held : step2.lock.readers = 1 := by
  simp only [step2, ActionResult.get!, step2_result, createContextWithReadLock, step1, step1_result,
             acquireReadLock, initialState, RWLockSystemState.init, RWLockState.init]
  rfl

/-- Thread 0 is encoding -/
theorem step2_encoding : (step2.threads ⟨0, by decide⟩).state = .encoding := by
  simp only [step2, ActionResult.get!, step2_result, createContextWithReadLock, step1, step1_result,
             acquireReadLock, initialState, RWLockSystemState.init, RWLockState.init]
  rfl

/-- Step 3: Async handler triggers (GPU completion) -/
def step3_result : ActionResult exampleConfig :=
  asyncHandlerTrigger exampleConfig step2 ⟨0, by decide⟩

theorem step3_is_success : ∃ s, step3_result = .success s := by
  simp only [step3_result, asyncHandlerTrigger, step2, ActionResult.get!, step2_result,
             createContextWithReadLock, step1, step1_result, acquireReadLock, initialState,
             RWLockSystemState.init, RWLockState.init]
  exact ⟨_, rfl⟩

def step3 : RWLockSystemState exampleConfig := step3_result.get! step3_is_success

/-- Step 4: Async handler invalidates context (BYPASSING OUR LOCK!) -/
def step4_result : ActionResult exampleConfig :=
  asyncHandlerInvalidate exampleConfig step3 ⟨0, by decide⟩ ⟨0, by decide⟩

theorem step4_is_success : ∃ s, step4_result = .success s := by
  simp only [step4_result, asyncHandlerInvalidate, step3, ActionResult.get!, step3_result,
             asyncHandlerTrigger, step2, step2_result, createContextWithReadLock, step1,
             step1_result, acquireReadLock, initialState, RWLockSystemState.init, RWLockState.init]
  exact ⟨_, rfl⟩

def step4 : RWLockSystemState exampleConfig := step4_result.get! step4_is_success

/-- Context is invalid -/
theorem step4_context_invalid : (step4.contexts ⟨0, by decide⟩).state = .invalid := by
  simp only [step4, ActionResult.get!, step4_result, asyncHandlerInvalidate, step3, step3_result,
             asyncHandlerTrigger, step2, step2_result, createContextWithReadLock, step1,
             step1_result, acquireReadLock, initialState, RWLockSystemState.init, RWLockState.init]
  rfl

/-- Thread 0 STILL holds read lock but context is invalid! -/
theorem step4_read_lock_still_held : step4.lock.readers = 1 := by
  simp only [step4, ActionResult.get!, step4_result, asyncHandlerInvalidate, step3, step3_result,
             asyncHandlerTrigger, step2, step2_result, createContextWithReadLock, step1,
             step1_result, acquireReadLock, initialState, RWLockSystemState.init, RWLockState.init]
  rfl

/-- Thread 0 is still encoding -/
theorem step4_still_encoding : (step4.threads ⟨0, by decide⟩).state = .encoding := by
  simp only [step4, ActionResult.get!, step4_result, asyncHandlerInvalidate, step3, step3_result,
             asyncHandlerTrigger, step2, step2_result, createContextWithReadLock, step1,
             step1_result, acquireReadLock, initialState, RWLockSystemState.init, RWLockState.init]
  rfl

/-- Step 5: Thread 0 tries to use context → NULL DEREFERENCE! -/
def step5_result : ActionResult exampleConfig :=
  useContext exampleConfig step4 ⟨0, by decide⟩

theorem step5_is_success : ∃ s, step5_result = .success s := by
  simp only [step5_result, useContext, step4, ActionResult.get!, step4_result, asyncHandlerInvalidate,
             step3, step3_result, asyncHandlerTrigger, step2, step2_result, createContextWithReadLock,
             step1, step1_result, acquireReadLock, initialState, RWLockSystemState.init,
             RWLockState.init]
  exact ⟨_, rfl⟩

def step5 : RWLockSystemState exampleConfig := step5_result.get! step5_is_success

/-- RACE WITNESSED -/
theorem step5_null_deref : step5.nullDerefCount = 1 := by
  simp only [step5, ActionResult.get!, step5_result, useContext, step4, step4_result,
             asyncHandlerInvalidate, step3, step3_result, asyncHandlerTrigger, step2,
             step2_result, createContextWithReadLock, step1, step1_result, acquireReadLock,
             initialState, RWLockSystemState.init, RWLockState.init]
  rfl

theorem step5_race_witnessed : step5.raceWitnessed = true := by
  simp only [step5, ActionResult.get!, step5_result, useContext, step4, step4_result,
             asyncHandlerInvalidate, step3, step3_result, asyncHandlerTrigger, step2,
             step2_result, createContextWithReadLock, step1, step1_result, acquireReadLock,
             initialState, RWLockSystemState.init, RWLockState.init]
  rfl

/-
  ████████╗██╗  ██╗███████╗ ██████╗ ██████╗ ███████╗███╗   ███╗
  ╚══██╔══╝██║  ██║██╔════╝██╔═══██╗██╔══██╗██╔════╝████╗ ████║
     ██║   ███████║█████╗  ██║   ██║██████╔╝█████╗  ██╔████╔██║
     ██║   ██╔══██║██╔══╝  ██║   ██║██╔══██╗██╔══╝  ██║╚██╔╝██║
     ██║   ██║  ██║███████╗╚██████╔╝██║  ██║███████╗██║ ╚═╝ ██║
     ╚═╝   ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝

  MAIN THEOREM: Reader-Writer Lock is INSUFFICIENT
-/

/-- Reader-writer locks fail to prevent race conditions -/
theorem rw_lock_insufficient :
    step5.raceWitnessed = true ∧ step5.nullDerefCount > 0 := by
  constructor
  · exact step5_race_witnessed
  · simp only [step5_null_deref]; decide

/-- Corollary: read lock was held but race still occurred via async path -/
theorem race_despite_read_lock :
    step4_read_lock_still_held = step4_read_lock_still_held ∧
    step5.raceWitnessed = true := by
  constructor
  · rfl
  · exact step5_race_witnessed

/-- Corollary: async handlers bypass user-space synchronization -/
theorem async_handlers_bypass_locks :
    -- Thread held read lock
    step4.lock.readers = 1 ∧
    -- Yet context was invalidated
    (step4.contexts ⟨0, by decide⟩).state = .invalid := by
  constructor
  · exact step4_read_lock_still_held
  · exact step4_context_invalid

end MPSVerify.AGX.RWLock
