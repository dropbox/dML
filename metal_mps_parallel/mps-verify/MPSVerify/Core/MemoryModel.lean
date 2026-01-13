/-
  MPSVerify.Core.MemoryModel
  C++ Memory Model Formalization

  This module formalizes the C++11 memory model for verification of
  concurrent atomic operations. It models:
  - Memory orderings (relaxed, acquire, release, acq_rel, seq_cst)
  - Happens-before relationships
  - Synchronizes-with relationships
  - Memory visibility

  Reference: C++11 Standard Section 1.10 and 29
-/

import MPSVerify.Core.Types

namespace MPSVerify.Core.MemoryModel

-- Use aliases for types from the Types module
abbrev ThreadId := MPSVerify.Core.ThreadId
abbrev Location := MPSVerify.Core.Location
abbrev Timestamp := MPSVerify.Core.Timestamp
abbrev Value := MPSVerify.Core.Value

/-- C++ memory order enumeration -/
inductive MemoryOrder where
  | relaxed    -- memory_order_relaxed
  | consume    -- memory_order_consume (treated as acquire)
  | acquire    -- memory_order_acquire
  | release    -- memory_order_release
  | acq_rel    -- memory_order_acq_rel
  | seq_cst    -- memory_order_seq_cst
  deriving Repr, DecidableEq, Inhabited

/-- Memory operation type -/
inductive MemoryOp where
  | read : Location → MemoryOrder → MemoryOp        -- Atomic read
  | write : Location → Value → MemoryOrder → MemoryOp  -- Atomic write
  | rmw : Location → MemoryOrder → MemoryOp         -- Read-modify-write (atomic)
  | fence : MemoryOrder → MemoryOp                   -- Memory fence
  | nonAtomicRead : Location → MemoryOp             -- Plain (non-atomic) read
  | nonAtomicWrite : Location → Value → MemoryOp    -- Plain (non-atomic) write
  deriving Repr, Inhabited

/-- Check if a memory order is valid for a load operation -/
def MemoryOrder.validForLoad : MemoryOrder → Bool
  | .relaxed => true
  | .consume => true
  | .acquire => true
  | .release => false
  | .acq_rel => false
  | .seq_cst => true

/-- Check if a memory order is valid for a store operation -/
def MemoryOrder.validForStore : MemoryOrder → Bool
  | .relaxed => true
  | .consume => false
  | .acquire => false
  | .release => true
  | .acq_rel => false
  | .seq_cst => true

/-- Check if a memory order has acquire semantics -/
def MemoryOrder.hasAcquire : MemoryOrder → Bool
  | .acquire => true
  | .acq_rel => true
  | .seq_cst => true
  | _ => false

/-- Check if a memory order has release semantics -/
def MemoryOrder.hasRelease : MemoryOrder → Bool
  | .release => true
  | .acq_rel => true
  | .seq_cst => true
  | _ => false

/-- Memory event with thread and timestamp -/
structure MemoryEvent where
  thread : ThreadId
  timestamp : Timestamp
  op : MemoryOp
  deriving Repr, Inhabited

/-- Memory state: mapping from locations to values -/
def Memory := Location → Value

/-- Initial memory state (all null) -/
def Memory.initial : Memory := fun _ => Value.null

/-- Thread view: per-thread view of memory with timestamps -/
structure ThreadView where
  lastSeen : Location → Timestamp
  deriving Inhabited

/-- Initial thread view -/
def ThreadView.initial : ThreadView := ⟨fun _ => 0⟩

/-- Global memory state for verification -/
structure MemoryState where
  memory : Memory
  views : ThreadId → ThreadView
  globalTime : Timestamp
  events : List MemoryEvent

instance : Inhabited MemoryState :=
  ⟨{ memory := Memory.initial
   , views := fun _ => ThreadView.initial
   , globalTime := 0
   , events := [] }⟩

/-- Initial memory state -/
def MemoryState.initial : MemoryState :=
  { memory := Memory.initial
  , views := fun _ => ThreadView.initial
  , globalTime := 0
  , events := [] }

/-- Happens-before relation -/
def happensBefore (e1 e2 : MemoryEvent) : Prop :=
  -- Same thread: program order
  (e1.thread = e2.thread ∧ e1.timestamp < e2.timestamp) ∨
  -- Synchronizes-with (simplified)
  (synchronizesWith e1 e2)
where
  synchronizesWith (e1 e2 : MemoryEvent) : Prop :=
    -- Release-acquire synchronization
    match e1.op, e2.op with
    | .write loc _ ord1, .read loc' ord2 =>
      loc = loc' ∧ ord1.hasRelease ∧ ord2.hasAcquire
    | _, _ => False

/-- Get location from a memory operation -/
def MemoryOp.location? : MemoryOp → Option Location
  | .read loc _ => some loc
  | .write loc _ _ => some loc
  | .rmw loc _ => some loc
  | .fence _ => none
  | .nonAtomicRead loc => some loc
  | .nonAtomicWrite loc _ => some loc

/-- Check if operation is a write -/
def MemoryOp.isWrite : MemoryOp → Bool
  | .write _ _ _ => true
  | .rmw _ _ => true
  | .nonAtomicWrite _ _ => true
  | _ => false

/-- Check if operation is atomic (has a memory order) -/
def MemoryOp.isAtomic : MemoryOp → Bool
  | .read _ _ => true
  | .write _ _ _ => true
  | .rmw _ _ => true
  | .fence _ => true
  | .nonAtomicRead _ => false
  | .nonAtomicWrite _ _ => false

/-- Get memory order from operation (only for atomic operations) -/
def MemoryOp.order? : MemoryOp → Option MemoryOrder
  | .read _ ord => some ord
  | .write _ _ ord => some ord
  | .rmw _ ord => some ord
  | .fence ord => some ord
  | .nonAtomicRead _ => none
  | .nonAtomicWrite _ _ => none

/-- Get memory order from operation (defaults to relaxed for non-atomic) -/
def MemoryOp.order : MemoryOp → MemoryOrder
  | .read _ ord => ord
  | .write _ _ ord => ord
  | .rmw _ ord => ord
  | .fence ord => ord
  | .nonAtomicRead _ => .relaxed  -- Non-atomic has no ordering guarantee
  | .nonAtomicWrite _ _ => .relaxed

/--
Data race: two conflicting accesses without happens-before.

Per C++11 §1.10/21: "The execution of a program contains a data race if it
contains two conflicting actions in different threads, at least one of which
is not atomic, and neither happens before the other."

All atomic operations (read, write, rmw, fence) have well-defined behavior
under concurrent access. Only non-atomic operations can participate in races.
-/
def hasDataRace (e1 e2 : MemoryEvent) : Prop :=
  e1.thread ≠ e2.thread ∧
  (e1.op.location? = e2.op.location? ∧ e1.op.location?.isSome) ∧
  (e1.op.isWrite ∨ e2.op.isWrite) ∧
  (¬e1.op.isAtomic ∨ ¬e2.op.isAtomic) ∧  -- At least one must be non-atomic
  ¬happensBefore e1 e2 ∧
  ¬happensBefore e2 e1

/-- A memory trace is race-free if no two events have a data race -/
def isRaceFree (events : List MemoryEvent) : Prop :=
  ∀ e1 e2, e1 ∈ events → e2 ∈ events → e1 ≠ e2 → ¬hasDataRace e1 e2

/--
Seq_cst operations synchronize: a seq_cst write followed by a seq_cst read
on the same location creates a synchronizes-with relationship.
-/
theorem seq_cst_write_read_synchronizes
    (e1 e2 : MemoryEvent)
    (loc : Location)
    (val : Value) :
    e1.op = .write loc val .seq_cst →
    e2.op = .read loc .seq_cst →
    e1.timestamp < e2.timestamp →
    happensBefore e1 e2 := by
  intro h_write h_read _
  -- Synchronizes-with via release-acquire semantics of seq_cst
  right  -- Use synchronizesWith branch
  simp only [happensBefore.synchronizesWith, h_write, h_read, MemoryOrder.hasRelease,
             MemoryOrder.hasAcquire, and_self]

/--
Single-threaded events always have happens-before (program order).
-/
theorem same_thread_happens_before
    (e1 e2 : MemoryEvent) :
    e1.thread = e2.thread →
    e1.timestamp < e2.timestamp →
    happensBefore e1 e2 := by
  intro h_thread h_time
  left  -- Program order branch
  exact ⟨h_thread, h_time⟩

/--
No data race between events on the same thread.
-/
theorem same_thread_no_race
    (e1 e2 : MemoryEvent) :
    e1.thread = e2.thread →
    ¬hasDataRace e1 e2 := by
  intro h_same
  unfold hasDataRace
  intro ⟨h_diff, _⟩
  exact h_diff h_same

/--
An empty trace is trivially race-free.
-/
theorem empty_trace_race_free : isRaceFree [] := by
  simp [isRaceFree]

/--
A single-threaded trace is race-free.
-/
theorem single_thread_race_free (events : List MemoryEvent) (t : ThreadId) :
    (∀ e ∈ events, e.thread = t) →
    isRaceFree events := by
  intro h_single
  simp [isRaceFree]
  intro e1 e2 h_e1 h_e2 _
  have h1 := h_single e1 h_e1
  have h2 := h_single e2 h_e2
  have h_same : e1.thread = e2.thread := by rw [h1, h2]
  exact same_thread_no_race e1 e2 h_same

/--
Key lemma: If an operation has seq_cst order, it must be atomic.
This is because only atomic operations can have memory orders.
Non-atomic operations (nonAtomicRead, nonAtomicWrite) don't have seq_cst order.
-/
theorem seq_cst_implies_atomic (op : MemoryOp) :
    op.order = .seq_cst → op.isAtomic := by
  intro h_order
  cases op with
  | read _ ord =>
    simp [MemoryOp.isAtomic]
  | write _ _ ord =>
    simp [MemoryOp.isAtomic]
  | rmw _ ord =>
    simp [MemoryOp.isAtomic]
  | fence ord =>
    simp [MemoryOp.isAtomic]
  | nonAtomicRead _ =>
    -- nonAtomicRead has order = relaxed, not seq_cst, so this case is impossible
    simp [MemoryOp.order] at h_order
  | nonAtomicWrite _ _ =>
    -- nonAtomicWrite has order = relaxed, not seq_cst, so this case is impossible
    simp [MemoryOp.order] at h_order

/--
All-atomic traces are race-free.
Since data races require at least one non-atomic access, a trace where
all operations are atomic cannot have data races.
-/
theorem all_atomic_race_free (events : List MemoryEvent) :
    (∀ e ∈ events, e.op.isAtomic) →
    isRaceFree events := by
  intro h_atomic
  simp [isRaceFree]
  intro e1 e2 h_e1 h_e2 _
  intro h_race
  -- hasDataRace requires at least one non-atomic operation
  unfold hasDataRace at h_race
  obtain ⟨_, _, _, h_nonatomic, _⟩ := h_race
  -- But all operations are atomic, contradiction
  have h1 := h_atomic e1 h_e1
  have h2 := h_atomic e2 h_e2
  simp [h1, h2] at h_nonatomic

/--
seq_cst operations are race-free.

This is a consequence of all seq_cst operations being atomic, and data races
only occurring when at least one operation is non-atomic (per C++11 §1.10/21).
-/
theorem seq_cst_race_free (events : List MemoryEvent) :
    (∀ e ∈ events, e.op.order = .seq_cst) →
    isRaceFree events := by
  intro h_seq_cst
  apply all_atomic_race_free
  intro e h_e
  exact seq_cst_implies_atomic e.op (h_seq_cst e h_e)

end MPSVerify.Core.MemoryModel
