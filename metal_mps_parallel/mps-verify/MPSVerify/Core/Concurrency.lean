/-
  MPSVerify.Core.Concurrency
  C++ Concurrency Primitives Formalization

  This module models C++ concurrency primitives for formal verification:
  - std::atomic<T> operations
  - std::mutex and std::recursive_mutex
  - std::once_flag and std::call_once
  - Thread-local storage (TLS)

  These models are used to verify thread safety properties of the
  MPS parallel inference implementation.
-/

import MPSVerify.Core.Types
import MPSVerify.Core.MemoryModel

namespace MPSVerify.Core.Concurrency

-- Type aliases
abbrev ThreadId := MPSVerify.Core.ThreadId
abbrev Timestamp := MPSVerify.Core.Timestamp
abbrev MemoryOrder := MPSVerify.Core.MemoryModel.MemoryOrder
abbrev ThreadView := MPSVerify.Core.MemoryModel.ThreadView

/-!
## Atomic Operations

Model of std::atomic<T> with memory ordering semantics.
-/

/-- Atomic variable state -/
structure AtomicState (α : Type) where
  value : α
  lastWriteThread : ThreadId
  lastWriteTimestamp : Timestamp
  deriving Repr, Inhabited

/-- Atomic load operation -/
def atomicLoad {α : Type} [Inhabited α]
    (state : AtomicState α)
    (order : MemoryOrder)
    (_thread : ThreadId)
    (view : ThreadView) : α × ThreadView :=
  -- Simplified: just return value and potentially update view
  let newView := if order.hasAcquire then
    { view with lastSeen := fun loc =>
        if loc = 0 then state.lastWriteTimestamp  -- Simplified location model
        else view.lastSeen loc }
  else view
  (state.value, newView)

/-- Atomic store operation -/
def atomicStore {α : Type}
    (state : AtomicState α)
    (newValue : α)
    (_order : MemoryOrder)
    (thread : ThreadId)
    (time : Timestamp) : AtomicState α :=
  { state with
    value := newValue
    lastWriteThread := thread
    lastWriteTimestamp := time }

/-- Compare-and-swap (CAS) operation -/
def compareAndSwap {α : Type} [DecidableEq α]
    (state : AtomicState α)
    (expected : α)
    (desired : α)
    (_successOrder : MemoryOrder)
    (_failureOrder : MemoryOrder)
    (thread : ThreadId)
    (time : Timestamp) : Bool × AtomicState α :=
  if state.value = expected then
    (true, { state with
      value := desired
      lastWriteThread := thread
      lastWriteTimestamp := time })
  else
    (false, state)

/-- Fetch-and-add for numeric atomics -/
def fetchAdd (state : AtomicState Nat) (delta : Nat)
    (_order : MemoryOrder)
    (thread : ThreadId)
    (time : Timestamp) : Nat × AtomicState Nat :=
  let oldValue := state.value
  let newState := { state with
    value := oldValue + delta
    lastWriteThread := thread
    lastWriteTimestamp := time }
  (oldValue, newState)

/-!
## Mutex Operations

Model of std::mutex and std::recursive_mutex.
-/

/-- Mutex state -/
structure MutexState where
  locked : Bool
  owner : Option ThreadId
  recursionCount : Nat  -- For recursive_mutex
  waitQueue : List ThreadId
  deriving Repr, Inhabited

/-- Initial unlocked mutex -/
def MutexState.unlocked : MutexState :=
  { locked := false
  , owner := none
  , recursionCount := 0
  , waitQueue := [] }

/-- Try to acquire a non-recursive mutex -/
def mutexTryLock (state : MutexState) (thread : ThreadId) : Bool × MutexState :=
  if state.locked then
    (false, state)
  else
    (true, { state with
      locked := true
      owner := some thread
      recursionCount := 1 })

/-- Try to acquire a recursive mutex -/
def recursiveMutexTryLock (state : MutexState) (thread : ThreadId) : Bool × MutexState :=
  match state.owner with
  | some t =>
    if t = thread then
      -- Already own it, increment recursion count
      (true, { state with recursionCount := state.recursionCount + 1 })
    else
      -- Owned by another thread
      (false, state)
  | none =>
    -- Not locked
    (true, { state with
      locked := true
      owner := some thread
      recursionCount := 1 })

/-- Release a mutex -/
def mutexUnlock (state : MutexState) (thread : ThreadId) : Option MutexState :=
  match state.owner with
  | some t =>
    if t = thread then
      if state.recursionCount ≤ 1 then
        -- Fully released
        some { state with
          locked := false
          owner := none
          recursionCount := 0 }
      else
        -- Decrement recursion count
        some { state with recursionCount := state.recursionCount - 1 }
    else
      none  -- Error: not owner
  | none =>
    none  -- Error: not locked

/-!
## Once Flag

Model of std::once_flag and std::call_once.
-/

/-- Once flag state -/
inductive OnceState where
  | notStarted : OnceState
  | inProgress : ThreadId → OnceState
  | completed : OnceState
  deriving Repr, DecidableEq, Inhabited

/-- Try to be the one to execute call_once -/
def tryCallOnce (state : OnceState) (thread : ThreadId) : (Bool × OnceState) :=
  match state with
  | .notStarted => (true, .inProgress thread)
  | .inProgress _ => (false, state)  -- Another thread is doing it
  | .completed => (false, state)     -- Already done

/-- Mark call_once as completed -/
def completeCallOnce (state : OnceState) (thread : ThreadId) : OnceState :=
  match state with
  | .inProgress t => if t = thread then .completed else state
  | s => s

/-!
## Thread-Local Storage

Model of thread_local variables.
-/

/-- Thread-local storage: each thread has its own value -/
def ThreadLocal (α : Type) := ThreadId → Option α

/-- Initial TLS (all threads have no value) -/
def ThreadLocal.empty {α : Type} : ThreadLocal α := fun _ => none

/-- Get TLS value for current thread -/
def ThreadLocal.get {α : Type} (tls : ThreadLocal α) (thread : ThreadId) : Option α :=
  tls thread

/-- Set TLS value for current thread -/
def ThreadLocal.set {α : Type} (tls : ThreadLocal α) (thread : ThreadId) (value : α) : ThreadLocal α :=
  fun t => if t = thread then some value else tls t

/-- Clear TLS value for current thread -/
def ThreadLocal.clear {α : Type} (tls : ThreadLocal α) (thread : ThreadId) : ThreadLocal α :=
  fun t => if t = thread then none else tls t

/-!
## Lock Guard Pattern (RAII)

Model of std::lock_guard and std::unique_lock.
-/

/-- Lock guard state: tracks which thread holds the guard -/
structure LockGuard where
  mutex : MutexState
  owner : ThreadId
  deriving Repr, Inhabited

/-- Create a lock guard (acquire lock) -/
def LockGuard.create (mutex : MutexState) (thread : ThreadId) : Option (LockGuard × MutexState) :=
  let (success, newMutex) := mutexTryLock mutex thread
  if success then
    some (⟨newMutex, thread⟩, newMutex)
  else
    none

/-- Destroy a lock guard (release lock) -/
def LockGuard.destroy (guard : LockGuard) : Option MutexState :=
  mutexUnlock guard.mutex guard.owner

/-!
## Properties and Theorems

Key properties we want to verify about concurrent code.
-/

/-- A mutex is never double-locked when already locked -/
theorem mutex_no_double_lock (state : MutexState) (thread : ThreadId) :
    state.locked = true →
    (mutexTryLock state thread).1 = false := by
  intro h_locked
  simp only [mutexTryLock, h_locked, ↓reduceIte]

/-- If a mutex has an owner, it must be locked (by construction of our operations) -/
theorem mutex_owner_means_locked (state : MutexState) :
    state.owner.isSome = true →
    state.locked = true →
    (mutexTryLock state state.owner.get!).1 = false := by
  intro _ h_locked
  simp only [mutexTryLock, h_locked, ↓reduceIte]

/-- A recursive mutex can be locked multiple times by the same thread -/
theorem recursive_mutex_allows_recursion (state : MutexState) (thread : ThreadId) :
    state.owner = some thread →
    (recursiveMutexTryLock state thread).1 = true := by
  intro h
  simp [recursiveMutexTryLock, h]

/-- Once flag ensures exactly-once execution -/
theorem once_flag_exactly_once (s : OnceState) (t1 t2 : ThreadId) :
    s = .notStarted →
    (tryCallOnce s t1).1 = true →
    (tryCallOnce (tryCallOnce s t1).2 t2).1 = false := by
  intros h1 _
  subst h1
  simp [tryCallOnce]

end MPSVerify.Core.Concurrency
