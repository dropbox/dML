/-
  AGX Driver Race Condition Model - Type Definitions

  Machine-checked Lean 4 port of the TLA+ specification AGXContextRace.tla.

  This module defines the core types for modeling the AGX driver's context
  management system and the race condition that causes crashes.

  Based on reverse engineering of Apple's AGXMetalG16X driver:
  - Crash Site 1: setComputePipelineState: at offset 0x5c8
  - Crash Site 2: prepareForEnqueue at offset 0x98
  - Crash Site 3: allocateUSCSpillBuffer at offset 0x184
-/

namespace MPSVerify.AGX

/-- Thread identifier (1..NumThreads) -/
abbrev ThreadId := Nat

/-- Context slot identifier (1..NumContextSlots) -/
abbrev ContextId := Nat

/-- Thread state in the context lifecycle -/
inductive ThreadState where
  | idle          : ThreadState  -- Not using GPU
  | waitingMutex  : ThreadState  -- Waiting to acquire mutex (fixed model only)
  | creating      : ThreadState  -- Creating context
  | encoding      : ThreadState  -- Actively encoding (using context)
  | destroying    : ThreadState  -- Destroying context
  deriving DecidableEq, Repr

/-- Context validity state -/
inductive ContextState where
  | valid   : ContextState  -- Context is valid and usable
  | invalid : ContextState  -- Context is invalid (freed or not created)
  deriving DecidableEq, Repr

/-- Configuration for the model -/
structure Config where
  numThreads : Nat
  numContextSlots : Nat
  threads_pos : numThreads > 0 := by decide
  contexts_pos : numContextSlots > 0 := by decide

/-- Per-thread state -/
structure ThreadInfo where
  state : ThreadState
  context : Option ContextId  -- Context assigned to this thread

/-- Per-context state -/
structure ContextInfo where
  state : ContextState
  owner : Option ThreadId  -- Thread that owns this context

/-- Global system state for the BUGGY model (no mutex) -/
structure BuggyState (cfg : Config) where
  threads : Fin cfg.numThreads → ThreadInfo
  contexts : Fin cfg.numContextSlots → ContextInfo
  nullDerefCount : Nat  -- Number of NULL pointer dereferences detected
  raceWitnessed : Bool  -- TRUE if race condition manifested

/-- Global system state for the FIXED model (with mutex) -/
structure FixedState (cfg : Config) where
  threads : Fin cfg.numThreads → ThreadInfo
  contexts : Fin cfg.numContextSlots → ContextInfo
  mutexHolder : Option (Fin cfg.numThreads)  -- Thread holding the global mutex
  nullDerefCount : Nat
  raceWitnessed : Bool

/-- Initial state for buggy model -/
def BuggyState.init (cfg : Config) : BuggyState cfg := {
  threads := fun _ => { state := .idle, context := none }
  contexts := fun _ => { state := .invalid, owner := none }
  nullDerefCount := 0
  raceWitnessed := false
}

/-- Initial state for fixed model -/
def FixedState.init (cfg : Config) : FixedState cfg := {
  threads := fun _ => { state := .idle, context := none }
  contexts := fun _ => { state := .invalid, owner := none }
  mutexHolder := none
  nullDerefCount := 0
  raceWitnessed := false
}

/-- A context slot is free (invalid and no owner) -/
def ContextInfo.isFree (ci : ContextInfo) : Bool :=
  ci.state == .invalid && ci.owner.isNone

/-- A thread is actively using its context (encoding) -/
def ThreadInfo.isEncoding (ti : ThreadInfo) : Bool :=
  ti.state == .encoding && ti.context.isSome

end MPSVerify.AGX
