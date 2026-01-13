--------------------------- MODULE MPSStreamPool ---------------------------
\* TLA+ Specification for PyTorch MPS Stream Pool
\* Based on: pytorch-mps-fork/aten/src/ATen/mps/MPSStream.h
\*
\* Models the lock-free stream pool allocation mechanism:
\* - 32 streams (stream 0 is default, streams 1-31 are worker streams)
\* - Atomic bitmask freelist (free_slots_mask_)
\* - TLS-based thread binding
\* - Slot recycling on thread exit
\*
\* Key invariants to verify:
\* 1. No two threads bound to same stream (mutual exclusion)
\* 2. Slot recycling preserves pool integrity
\* 3. No deadlock in acquire/release cycle
\* 4. Default stream (0) always available

EXTENDS Naturals, Sequences, FiniteSets, TLC

CONSTANTS
    \* @type: Int;
    NumStreams,         \* Total streams in pool (32 in implementation)
    \* @type: Int;
    NumThreads,         \* Number of concurrent threads for model checking
    \* @type: Int;
    MaxOperations       \* Bound for model checking

ASSUME NumStreams > 1   \* At least stream 0 (default) + 1 worker
ASSUME NumThreads >= 1
ASSUME MaxOperations >= 1

VARIABLES
    \* Stream states: "free", "bound", "in_use"
    \* @type: Int -> Str;
    streams,
    \* Bitmask of free worker slots (bits 0 to NumStreams-2 for slots 1 to NumStreams-1)
    \* In implementation: free_slots_mask_ with bit (slot-1) = 1 means free
    \* @type: Int;
    free_mask,
    \* Thread -> Stream binding (0 = no stream, 1..NumStreams-1 = worker stream)
    \* @type: Int -> Int;
    thread_bindings,
    \* Track which thread is currently in critical section
    \* Models the atomic CAS on free_mask
    \* @type: Int;
    cas_in_progress,
    \* Operation counter for bounded model checking
    \* @type: Int;
    op_count

vars == <<streams, free_mask, thread_bindings, cas_in_progress, op_count>>

-----------------------------------------------------------------------------
\* Helper: Convert slot (1..NumStreams-1) to bitmask position (0..NumStreams-2)
SlotToBit(slot) == slot - 1

\* Helper: Check if bit is set in mask
BitSet(mask, bit) == (mask \div (2^bit)) % 2 = 1

\* Helper: Set bit in mask
SetBit(mask, bit) == mask + 2^bit

\* Helper: Clear bit in mask
ClearBit(mask, bit) == mask - 2^bit

\* Helper: Find lowest set bit
\* Precondition: mask > 0 (at least one bit set)
\* In TLA+, we search linearly since we're modeling, not optimizing
FindLowestSetBit(mask) ==
    LET bits == {b \in 0..(NumStreams-2) : BitSet(mask, b)}
    IN CHOOSE b \in bits : \A b2 \in bits : b <= b2

\* All worker slots mask (bits 0 to NumStreams-2 all set)
AllWorkerSlotsMask == (2^(NumStreams-1)) - 1

-----------------------------------------------------------------------------
\* Type Invariant
TypeOK ==
    /\ streams \in [1..NumStreams -> {"free", "bound", "in_use"}]
    /\ free_mask \in 0..AllWorkerSlotsMask
    /\ thread_bindings \in [1..NumThreads -> 0..(NumStreams-1)]
    /\ cas_in_progress \in 0..NumThreads  \* 0 = no thread, >0 = thread ID
    /\ op_count \in 0..MaxOperations

\* Initial state
Init ==
    /\ streams = [s \in 1..NumStreams |-> "free"]
    /\ free_mask = AllWorkerSlotsMask     \* All worker slots free initially
    /\ thread_bindings = [t \in 1..NumThreads |-> 0]  \* No bindings
    /\ cas_in_progress = 0
    /\ op_count = 0

-----------------------------------------------------------------------------
\* AcquireStream: Thread t attempts to get a stream from the pool
\* Models: MPSStreamPool::acquireStream() -> acquireSlot()
\*
\* Implementation uses atomic CAS on free_slots_mask_:
\*   do {
\*     mask = free_slots_mask_.load();
\*     if (mask == 0) return -1;  // Pool exhausted
\*     slot = ffs(mask);  // Find first set bit
\*     new_mask = mask & ~(1 << (slot-1));
\*   } while (!free_slots_mask_.compare_exchange_weak(mask, new_mask));

AcquireStream(t) ==
    /\ op_count < MaxOperations
    /\ thread_bindings[t] = 0              \* Thread doesn't have a stream
    /\ cas_in_progress = 0                  \* No CAS in progress
    /\ free_mask > 0                        \* Pool not exhausted
    /\ LET bit == FindLowestSetBit(free_mask)
           slot == bit + 1
       IN /\ cas_in_progress' = t           \* Enter critical section (model CAS)
          /\ streams' = [streams EXCEPT ![slot] = "bound"]
          /\ free_mask' = ClearBit(free_mask, bit)
          /\ thread_bindings' = [thread_bindings EXCEPT ![t] = slot]
          /\ op_count' = op_count + 1

\* Complete CAS for acquire (atomic completion)
CompleteAcquire(t) ==
    /\ cas_in_progress = t
    /\ cas_in_progress' = 0
    /\ UNCHANGED <<streams, free_mask, thread_bindings, op_count>>

\* ReleaseStreamSlot: Thread t releases its stream back to pool
\* Models: MPSStreamPool::releaseStreamSlot(slot)
\*
\* Implementation:
\*   free_slots_mask_.fetch_or(1 << (slot-1));

ReleaseStream(t) ==
    /\ op_count < MaxOperations
    /\ thread_bindings[t] > 0              \* Thread has a stream
    /\ cas_in_progress = 0
    /\ LET slot == thread_bindings[t]
           bit == SlotToBit(slot)
       IN /\ streams' = [streams EXCEPT ![slot] = "free"]
          /\ free_mask' = SetBit(free_mask, bit)
          /\ thread_bindings' = [thread_bindings EXCEPT ![t] = 0]
          /\ cas_in_progress' = 0
          /\ op_count' = op_count + 1

\* UseStream: Thread t uses its bound stream (transitions to "in_use")
\* Models actual GPU work submission
UseStream(t) ==
    /\ op_count < MaxOperations
    /\ thread_bindings[t] > 0
    /\ cas_in_progress = 0
    /\ LET slot == thread_bindings[t]
       IN /\ streams[slot] = "bound"
          /\ streams' = [streams EXCEPT ![slot] = "in_use"]
          /\ UNCHANGED <<free_mask, thread_bindings, cas_in_progress>>
          /\ op_count' = op_count + 1

\* FinishUse: Thread t finishes using stream (back to "bound")
FinishUse(t) ==
    /\ op_count < MaxOperations
    /\ thread_bindings[t] > 0
    /\ cas_in_progress = 0
    /\ LET slot == thread_bindings[t]
       IN /\ streams[slot] = "in_use"
          /\ streams' = [streams EXCEPT ![slot] = "bound"]
          /\ UNCHANGED <<free_mask, thread_bindings, cas_in_progress>>
          /\ op_count' = op_count + 1

\* DefaultStreamAccess: Any thread can access default stream (slot 0)
\* Models: getDefaultMPSStream() which always returns stream 0
\* Note: Stream 0 is NOT in the pool, always available
AccessDefaultStream(t) ==
    /\ op_count < MaxOperations
    /\ cas_in_progress = 0
    /\ streams[1] \in {"free", "in_use"}  \* Stream 1 (index 0+1) models default
    /\ UNCHANGED vars  \* Default stream access doesn't change pool state

-----------------------------------------------------------------------------
\* Next state relation
Next ==
    \/ \E t \in 1..NumThreads :
        \/ AcquireStream(t)
        \/ CompleteAcquire(t)
        \/ ReleaseStream(t)
        \/ UseStream(t)
        \/ FinishUse(t)
    \/ UNCHANGED vars  \* Stuttering for liveness

\* Fairness: Every enabled action eventually happens
Fairness ==
    /\ \A t \in 1..NumThreads :
        /\ WF_vars(AcquireStream(t))
        /\ WF_vars(CompleteAcquire(t))
        /\ WF_vars(ReleaseStream(t))
        /\ WF_vars(UseStream(t))
        /\ WF_vars(FinishUse(t))

Spec == Init /\ [][Next]_vars /\ Fairness

\* Apalache-compatible spec (no fairness - bounded safety only)
SpecNoFairness == Init /\ [][Next]_vars

-----------------------------------------------------------------------------
\* SAFETY PROPERTIES

\* Mutual Exclusion: No two threads bound to same stream
MutualExclusion ==
    \A t1, t2 \in 1..NumThreads :
        (t1 /= t2 /\ thread_bindings[t1] > 0) =>
        thread_bindings[t1] /= thread_bindings[t2]

\* Pool Integrity: Free mask correctly reflects stream states
\* If bit is set in free_mask, corresponding stream must be free
PoolIntegrity ==
    \A slot \in 2..NumStreams :
        LET bit == SlotToBit(slot)
        IN BitSet(free_mask, bit) =>
           (streams[slot] = "free" /\ \A t \in 1..NumThreads : thread_bindings[t] /= slot)

\* Bound streams have an owner
BoundStreamsHaveOwner ==
    \A slot \in 2..NumStreams :
        streams[slot] \in {"bound", "in_use"} =>
        \E t \in 1..NumThreads : thread_bindings[t] = slot

\* No orphaned bindings
NoOrphanedBindings ==
    \A t \in 1..NumThreads :
        thread_bindings[t] > 0 =>
        streams[thread_bindings[t]] \in {"bound", "in_use"}

\* Combined safety invariant
SafetyInvariant ==
    /\ TypeOK
    /\ MutualExclusion
    /\ PoolIntegrity
    /\ BoundStreamsHaveOwner
    /\ NoOrphanedBindings

-----------------------------------------------------------------------------
\* LIVENESS PROPERTIES

\* Progress: If pool has free slots and a thread wants a stream, it eventually gets one
\* (under fairness assumptions)
\* Guard: op_count < MaxOperations to handle bounded model
Progress ==
    \A t \in 1..NumThreads :
        (thread_bindings[t] = 0 /\ free_mask > 0 /\ op_count < MaxOperations) ~>
            (thread_bindings[t] > 0 \/ op_count = MaxOperations)

\* No starvation: A thread trying to release always succeeds
\* Guard: op_count < MaxOperations to handle bounded model
ReleaseAlwaysSucceeds ==
    \A t \in 1..NumThreads :
        (thread_bindings[t] > 0 /\ op_count < MaxOperations) ~>
            (thread_bindings[t] = 0 \/ op_count = MaxOperations)

-----------------------------------------------------------------------------
\* DEADLOCK FREEDOM

\* The system can always make progress (no deadlock)
\* At least one action is always enabled (or we've reached max operations)
\* CAS operations complete atomically, so CAS-in-progress is a valid transition state
DeadlockFree ==
    op_count = MaxOperations \/
    cas_in_progress > 0 \/    \* CAS in progress will complete atomically
    \E t \in 1..NumThreads :
        \/ /\ thread_bindings[t] = 0
           /\ free_mask > 0
        \/ thread_bindings[t] > 0

=============================================================================
\* Modification History
\* Created: 2025-12-18 by AI Worker N=1251
\* Based on: MPSStream.h MPSStreamPool implementation
