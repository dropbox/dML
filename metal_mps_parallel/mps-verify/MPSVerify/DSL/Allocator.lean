/-
  MPSVerify.DSL.Allocator
  Domain-Specific Language for MPS Allocator Verification

  This module provides abstractions for modeling and verifying
  the MPS buffer allocator concurrent patterns:
  - Double-check locking for lazy initialization
  - ABA detection with generation counters
  - Buffer lifecycle management
  - Memory pool operations
-/

import MPSVerify.Core

namespace MPSVerify.DSL.Allocator

open MPSVerify.Core

/-!
## ABA Problem and Detection

The ABA problem occurs when:
1. Thread 1 reads value A from shared location
2. Thread 2 changes A to B, then back to A
3. Thread 1's CAS succeeds (sees A) but misses the intermediate change

Solution: Pair each pointer with a generation counter that always increments.
-/

/-- ABA-safe pointer with generation counter -/
structure ABAPointer (α : Type) where
  ptr : Option α
  generation : Nat
  deriving Repr, DecidableEq, Inhabited

/-- Initial ABA pointer (null with generation 0) -/
def ABAPointer.null : ABAPointer α := ⟨none, 0⟩

/-- Create new ABA pointer from value -/
def ABAPointer.fromValue (v : α) (gen : Nat := 0) : ABAPointer α := ⟨some v, gen⟩

/-- ABA-safe compare-and-swap
    Only succeeds if BOTH pointer and generation match -/
def ABAPointer.compareAndSwap
    (current : ABAPointer α) [DecidableEq α]
    (expected : ABAPointer α)
    (newPtr : Option α) : Bool × ABAPointer α :=
  if current.ptr = expected.ptr && current.generation = expected.generation then
    (true, ⟨newPtr, current.generation + 1⟩)
  else
    (false, current)

/-!
## Buffer States

Buffer lifecycle in the MPS allocator.
-/

/-- Buffer allocation states -/
inductive BufferState where
  | unallocated : BufferState  -- Not yet created
  | allocated   : BufferState  -- Valid, in use
  | freed       : BufferState  -- Returned to pool
  | invalid     : BufferState  -- Corrupted or ABA victim
  deriving Repr, DecidableEq, Inhabited

/-- Buffer metadata -/
structure BufferMeta where
  id : BufferId
  size : Nat
  state : BufferState
  owner : Option ThreadId
  lastAccessTime : Timestamp
  deriving Repr, Inhabited

/-!
## Double-Check Locking Pattern

Safe lazy initialization pattern used in MPS allocator.
-/

/-- DCL state for lazy initialization -/
inductive DCLState (α : Type) where
  | uninitialized : DCLState α
  | initializing : ThreadId → DCLState α
  | initialized : α → DCLState α
  deriving Repr

/-- Check if DCL is initialized (first check, without lock) -/
def DCLState.isInitialized : DCLState α → Bool
  | .initialized _ => true
  | _ => false

/-- Get value if initialized -/
def DCLState.getValue? : DCLState α → Option α
  | .initialized v => some v
  | _ => none

/-- DCL operation result -/
inductive DCLResult (α : Type) where
  | gotExisting : α → DCLResult α      -- Got already-initialized value
  | wasInitialized : α → DCLResult α   -- We initialized it
  | waiting : DCLResult α              -- Another thread is initializing
  deriving Repr

/-- Double-check locking acquire
    Returns the value and new state -/
def dclAcquire [Inhabited α]
    (state : DCLState α)
    (thread : ThreadId)
    (initFn : Unit → α) : DCLResult α × DCLState α :=
  match state with
  | .initialized v =>
    -- Fast path: already initialized
    (.gotExisting v, state)
  | .initializing t =>
    if t = thread then
      -- We're the one initializing (recursive call?)
      let v := initFn ()
      (.wasInitialized v, .initialized v)
    else
      -- Another thread is initializing, must wait
      (.waiting, state)
  | .uninitialized =>
    -- We take responsibility to initialize
    let v := initFn ()
    (.wasInitialized v, .initialized v)

/-!
## Allocator State

Full allocator state model.
-/

/-- Memory pool containing buffers -/
structure MemoryPool where
  buffers : List BufferMeta
  freeList : List BufferId
  totalSize : Nat
  usedSize : Nat
  deriving Repr, Inhabited

/-- Allocator state with sharded caches -/
structure AllocatorState where
  mainPool : MemoryPool
  shardedCache : List (ThreadId × List BufferId)  -- Per-thread caches
  abaCounters : BufferId → Nat  -- Generation counters
  dclState : DCLState MemoryPool  -- For lazy pool initialization

/-- Inhabited instance for AllocatorState -/
instance : Inhabited AllocatorState :=
  ⟨{ mainPool := default
   , shardedCache := []
   , abaCounters := fun _ => 0
   , dclState := .uninitialized }⟩

/-- Initial allocator state -/
def AllocatorState.initial : AllocatorState :=
  { mainPool := ⟨[], [], 0, 0⟩
  , shardedCache := []
  , abaCounters := fun _ => 0
  , dclState := .uninitialized }

/-!
## Allocator Operations
-/

/-- Allocate a buffer from the pool -/
def allocBuffer (state : AllocatorState) (thread : ThreadId) (size : Nat)
    : Option (BufferId × AllocatorState) :=
  -- First check thread-local cache
  match state.shardedCache.find? (·.1 = thread) with
  | some (_, bufferId :: rest) =>
    -- Got from cache
    let newCache := state.shardedCache.map fun (t, bufs) =>
      if t = thread then (t, rest) else (t, bufs)
    some (bufferId, { state with shardedCache := newCache })
  | _ =>
    -- Fall back to main pool
    match state.mainPool.freeList with
    | [] => none  -- Pool exhausted
    | bufferId :: rest =>
      let newBuffers := state.mainPool.buffers.map fun bufMeta =>
        if bufMeta.id = bufferId then
          { bufMeta with state := .allocated, owner := some thread }
        else bufMeta
      let newPool := { state.mainPool with
        buffers := newBuffers
        freeList := rest
        usedSize := state.mainPool.usedSize + size }
      -- Increment ABA counter
      let newAbaCounters := fun id =>
        if id = bufferId then state.abaCounters bufferId + 1
        else state.abaCounters id
      some (bufferId, { state with
        mainPool := newPool
        abaCounters := newAbaCounters })

/-- Free a buffer back to the pool -/
def freeBuffer (state : AllocatorState) (thread : ThreadId) (bufferId : BufferId)
    : Option AllocatorState :=
  -- Verify ownership
  match state.mainPool.buffers.find? (·.id = bufferId) with
  | none => none
  | some bufMeta =>
    if bufMeta.owner ≠ some thread then none
    else
      -- Return to thread-local cache first
      let newCache := state.shardedCache.map fun (t, bufs) =>
        if t = thread then (t, bufferId :: bufs) else (t, bufs)
      let newBuffers := state.mainPool.buffers.map fun m =>
        if m.id = bufferId then
          { m with state := .freed, owner := none }
        else m
      let newPool := { state.mainPool with buffers := newBuffers }
      -- Increment ABA counter on free
      let newAbaCounters := fun id =>
        if id = bufferId then state.abaCounters bufferId + 1
        else state.abaCounters id
      some { state with
        mainPool := newPool
        shardedCache := newCache
        abaCounters := newAbaCounters }

/-!
## ABA Detection Theorems

Key theorems proving ABA detection correctness.
-/

/-- ABA CAS fails if generation changed -/
theorem aba_cas_fails_on_generation_change {α : Type} [DecidableEq α]
    (ptr : ABAPointer α) (expected : ABAPointer α)
    (newVal : Option α)
    (h : ptr.generation ≠ expected.generation) :
    (ABAPointer.compareAndSwap ptr expected newVal).1 = false := by
  simp only [ABAPointer.compareAndSwap]
  split
  · case isTrue h_match =>
    simp only [Bool.and_eq_true, decide_eq_true_eq] at h_match
    exact absurd h_match.2 h
  · rfl

/-- ABA CAS succeeds only when both pointer and generation match -/
theorem aba_cas_requires_both_match {α : Type} [DecidableEq α]
    (ptr : ABAPointer α) (expected : ABAPointer α)
    (newVal : Option α)
    (h_success : (ABAPointer.compareAndSwap ptr expected newVal).1 = true) :
    ptr.ptr = expected.ptr ∧ ptr.generation = expected.generation := by
  simp only [ABAPointer.compareAndSwap] at h_success
  split at h_success
  · case isTrue h_match =>
    simp only [Bool.and_eq_true, decide_eq_true_eq] at h_match
    exact h_match
  · contradiction

/-- After successful CAS, generation strictly increases -/
theorem aba_cas_increments_generation {α : Type} [DecidableEq α]
    (ptr : ABAPointer α) (expected : ABAPointer α)
    (newVal : Option α)
    (h_success : (ABAPointer.compareAndSwap ptr expected newVal).1 = true) :
    (ABAPointer.compareAndSwap ptr expected newVal).2.generation = ptr.generation + 1 := by
  simp only [ABAPointer.compareAndSwap]
  split
  · rfl
  · simp only [ABAPointer.compareAndSwap] at h_success
    split at h_success <;> contradiction

/-!
## DCL Safety Theorems
-/

/-- DCL gotExisting returns same state -/
theorem dcl_gotExisting_same_state [Inhabited α]
    (v : α) (state : DCLState α) (thread : ThreadId)
    (initFn : Unit → α)
    (h : state = .initialized v) :
    (dclAcquire state thread initFn).2 = state := by
  simp [dclAcquire, h]

/-- DCL initialized results in initialized state -/
theorem dcl_initialized_becomes_initialized [Inhabited α]
    (state : DCLState α) (thread : ThreadId)
    (initFn : Unit → α)
    (h : state = .uninitialized) :
    (dclAcquire state thread initFn).2.isInitialized = true := by
  simp [dclAcquire, h, DCLState.isInitialized]

/-- DCL waiting preserves state -/
theorem dcl_waiting_preserves_state [Inhabited α]
    (t : ThreadId) (state : DCLState α) (thread : ThreadId)
    (initFn : Unit → α)
    (h_state : state = .initializing t)
    (h_diff : t ≠ thread) :
    (dclAcquire state thread initFn).2 = state := by
  simp [dclAcquire, h_state, h_diff]

end MPSVerify.DSL.Allocator
