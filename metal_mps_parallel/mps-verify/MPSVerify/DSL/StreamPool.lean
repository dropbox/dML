/-
  MPSVerify.DSL.StreamPool
  Domain-Specific Language for MPS Stream Pool Verification

  This module provides abstractions for modeling and verifying
  the MPS stream pool concurrent patterns:
  - TLS (thread-local storage) binding
  - Stream lifecycle management
  - Pool capacity constraints
  - Fork safety
-/

import MPSVerify.Core

namespace MPSVerify.DSL.StreamPool

open MPSVerify.Core

/-!
## Stream Pool State

Model of the MPS stream pool state machine.
Corresponds to MPSStream.mm implementation.
-/

/-- Stream states in the lifecycle -/
inductive StreamState where
  | free       : StreamState  -- Available in pool
  | bound      : StreamState  -- Bound to a thread via TLS
  | active     : StreamState  -- Currently executing operations
  | forked     : StreamState  -- Invalidated due to fork()
  deriving Repr, DecidableEq, Inhabited

/-- Stream pool configuration -/
structure PoolConfig where
  maxStreams : Nat := 32      -- kStreamsPerPool in CUDA compatibility
  enableForkDetection : Bool := true
  deriving Repr, Inhabited

/-- Individual stream entry -/
structure StreamEntry where
  id : StreamId
  state : StreamState
  boundThread : Option ThreadId
  useCount : Nat
  deriving Repr, Inhabited

/-- Stream pool state -/
structure PoolState where
  streams : List StreamEntry
  freeList : List StreamId
  tlsBindings : ThreadId → Option StreamId
  config : PoolConfig
  forkGeneration : Nat  -- Incremented on fork detection

/-- Inhabited instance for PoolState -/
instance : Inhabited PoolState :=
  ⟨{ streams := []
   , freeList := []
   , tlsBindings := fun _ => none
   , config := default
   , forkGeneration := 0 }⟩

/-- Initial pool state with all streams free -/
def PoolState.initial (config : PoolConfig := {}) : PoolState :=
  let makeStreamId (i : Nat) : StreamId :=
    ⟨⟨i % 32, by omega⟩⟩
  let streamIds := List.range config.maxStreams |>.map makeStreamId
  let streams := streamIds.map fun sid =>
    { id := sid
    , state := .free
    , boundThread := none
    , useCount := 0 }
  { streams := streams
  , freeList := streamIds
  , tlsBindings := fun _ => none
  , config := config
  , forkGeneration := 0 }

/-!
## Stream Pool Operations

Thread-safe operations on the stream pool.
-/

/-- Acquire a stream for a thread -/
def acquireStream (pool : PoolState) (thread : ThreadId) : Option (StreamId × PoolState) :=
  -- Check if thread already has a bound stream
  match pool.tlsBindings thread with
  | some streamId =>
    -- Already bound, return existing
    some (streamId, pool)
  | none =>
    -- Need to allocate from free list
    match pool.freeList with
    | [] => none  -- Pool exhausted
    | streamId :: rest =>
      let newStreams := pool.streams.map fun entry =>
        if entry.id = streamId then
          { entry with
            state := .bound
            boundThread := some thread
            useCount := entry.useCount + 1 }
        else entry
      let newBindings := fun t =>
        if t = thread then some streamId
        else pool.tlsBindings t
      some (streamId, { pool with
        streams := newStreams
        freeList := rest
        tlsBindings := newBindings })

/-- Release a stream back to the pool -/
def releaseStream (pool : PoolState) (thread : ThreadId) : Option PoolState :=
  match pool.tlsBindings thread with
  | none => none  -- Thread doesn't have a stream
  | some streamId =>
    let newStreams := pool.streams.map fun entry =>
      if entry.id = streamId then
        { entry with
          state := .free
          boundThread := none }
      else entry
    let newBindings := fun t =>
      if t = thread then none
      else pool.tlsBindings t
    some { pool with
      streams := newStreams
      freeList := streamId :: pool.freeList
      tlsBindings := newBindings }

/-- Handle fork event - invalidate all bound streams -/
def handleFork (pool : PoolState) : PoolState :=
  if pool.config.enableForkDetection then
    let newStreams := pool.streams.map fun entry =>
      if entry.state = .bound || entry.state = .active then
        { entry with state := .forked }
      else entry
    { pool with
      streams := newStreams
      forkGeneration := pool.forkGeneration + 1 }
  else pool

/-!
## Stream Pool Invariants

Key properties that must hold for the stream pool.
-/

/-- No two threads are bound to the same stream -/
def noDoubleBinding (pool : PoolState) : Prop :=
  ∀ t1 t2 : ThreadId, t1 ≠ t2 →
    pool.tlsBindings t1 = pool.tlsBindings t2 →
    pool.tlsBindings t1 = none

/-- All streams in freeList are actually free -/
def freeListConsistent (pool : PoolState) : Prop :=
  ∀ streamId ∈ pool.freeList,
    ∃ entry ∈ pool.streams, entry.id = streamId ∧ entry.state = .free

/-- Stream count doesn't exceed max -/
def poolCapacityRespected (pool : PoolState) : Prop :=
  pool.streams.length ≤ pool.config.maxStreams

/-- Bound streams have valid owner -/
def boundStreamsHaveOwner (pool : PoolState) : Prop :=
  ∀ entry ∈ pool.streams,
    entry.state = .bound → entry.boundThread.isSome

/-!
## Theorems

Proofs of stream pool correctness.
-/

/-- Release returns stream to free state (binding cleared) -/
theorem release_clears_binding
    (pool : PoolState) (thread : ThreadId)
    (streamId : StreamId)
    (h_bound : pool.tlsBindings thread = some streamId) :
    match releaseStream pool thread with
    | some newPool => newPool.tlsBindings thread = none
    | none => False := by
  simp only [releaseStream, h_bound]
  simp

/-- Release adds stream to free list -/
theorem release_adds_to_freelist
    (pool : PoolState) (thread : ThreadId)
    (streamId : StreamId)
    (h_bound : pool.tlsBindings thread = some streamId) :
    match releaseStream pool thread with
    | some newPool => streamId ∈ newPool.freeList
    | none => False := by
  simp only [releaseStream, h_bound]
  exact List.Mem.head _

/-- Fork increments generation counter -/
theorem fork_increments_generation (pool : PoolState)
    (h_enabled : pool.config.enableForkDetection = true) :
    (handleFork pool).forkGeneration = pool.forkGeneration + 1 := by
  simp [handleFork, h_enabled]

/-- Acquire from empty pool fails -/
theorem acquire_from_empty_fails (pool : PoolState) (thread : ThreadId)
    (h_unbound : pool.tlsBindings thread = none)
    (h_empty : pool.freeList = []) :
    acquireStream pool thread = none := by
  simp [acquireStream, h_unbound, h_empty]

/-- Acquire when already bound returns same stream -/
theorem acquire_when_bound_returns_same
    (pool : PoolState) (thread : ThreadId)
    (streamId : StreamId)
    (h_bound : pool.tlsBindings thread = some streamId) :
    acquireStream pool thread = some (streamId, pool) := by
  simp [acquireStream, h_bound]

end MPSVerify.DSL.StreamPool
