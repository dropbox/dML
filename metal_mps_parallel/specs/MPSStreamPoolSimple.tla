------------------------ MODULE MPSStreamPoolSimple ------------------------
\* Simplified TLA+ Specification for PyTorch MPS Stream Pool
\* Apalache-compatible version using sets instead of bitmasks
\*
\* This is a simplified model that avoids non-linear arithmetic (2^n)
\* for SMT solver compatibility with Apalache symbolic checking.
\*
\* Key invariants verified:
\* 1. No two threads bound to same stream (mutual exclusion)
\* 2. Slot recycling preserves pool integrity
\* 3. No deadlock in acquire/release cycle

EXTENDS Naturals, FiniteSets

CONSTANTS
    \* @type: Int;
    NumStreams,         \* Total streams in pool (excluding stream 0)
    \* @type: Int;
    NumThreads,         \* Number of concurrent threads
    \* @type: Int;
    MaxOperations       \* Bound for model checking

ASSUME NumStreams >= 1
ASSUME NumThreads >= 1
ASSUME MaxOperations >= 1

VARIABLES
    \* Set of free stream slots (1..NumStreams)
    \* @type: Set(Int);
    free_slots,
    \* Thread -> Stream binding (0 = no stream, 1..NumStreams = has stream)
    \* @type: Int -> Int;
    thread_bindings,
    \* Operation counter for bounded model checking
    \* @type: Int;
    op_count

vars == <<free_slots, thread_bindings, op_count>>

-----------------------------------------------------------------------------
\* Type Invariant
TypeOK ==
    /\ free_slots \subseteq (1..NumStreams)
    /\ thread_bindings \in [1..NumThreads -> 0..NumStreams]
    /\ op_count \in 0..MaxOperations

\* Initial state
Init ==
    /\ free_slots = 1..NumStreams       \* All worker slots free initially
    /\ thread_bindings = [t \in 1..NumThreads |-> 0]  \* No bindings
    /\ op_count = 0

-----------------------------------------------------------------------------
\* AcquireStream: Thread t attempts to get a stream from the pool
AcquireStream(t) ==
    /\ op_count < MaxOperations
    /\ thread_bindings[t] = 0              \* Thread doesn't have a stream
    /\ free_slots /= {}                    \* Pool not exhausted
    /\ \E slot \in free_slots :
        /\ free_slots' = free_slots \ {slot}
        /\ thread_bindings' = [thread_bindings EXCEPT ![t] = slot]
        /\ op_count' = op_count + 1

\* ReleaseStream: Thread t releases its stream back to pool
ReleaseStream(t) ==
    /\ op_count < MaxOperations
    /\ thread_bindings[t] > 0              \* Thread has a stream
    /\ LET slot == thread_bindings[t]
       IN /\ free_slots' = free_slots \union {slot}
          /\ thread_bindings' = [thread_bindings EXCEPT ![t] = 0]
          /\ op_count' = op_count + 1

-----------------------------------------------------------------------------
\* Next state relation
Next ==
    \/ \E t \in 1..NumThreads :
        \/ AcquireStream(t)
        \/ ReleaseStream(t)
    \/ UNCHANGED vars  \* Stuttering

\* Apalache-compatible spec (no fairness)
SpecNoFairness == Init /\ [][Next]_vars

-----------------------------------------------------------------------------
\* SAFETY PROPERTIES

\* Mutual Exclusion: No two threads bound to same stream
MutualExclusion ==
    \A t1, t2 \in 1..NumThreads :
        (t1 /= t2 /\ thread_bindings[t1] > 0) =>
        thread_bindings[t1] /= thread_bindings[t2]

\* Pool Integrity: Free slots and bound slots are disjoint
PoolIntegrity ==
    \A slot \in 1..NumStreams :
        (slot \in free_slots) <=>
        (\A t \in 1..NumThreads : thread_bindings[t] /= slot)

\* No orphaned bindings: bound slots are valid
NoOrphanedBindings ==
    \A t \in 1..NumThreads :
        thread_bindings[t] \in {0} \union (1..NumStreams)

\* Combined safety invariant
SafetyInvariant ==
    /\ TypeOK
    /\ MutualExclusion
    /\ PoolIntegrity
    /\ NoOrphanedBindings

\* Deadlock Freedom: system can always make progress or has reached bound
DeadlockFree ==
    op_count = MaxOperations \/
    \E t \in 1..NumThreads :
        \/ (thread_bindings[t] = 0 /\ free_slots /= {})  \* Can acquire
        \/ thread_bindings[t] > 0                          \* Can release

=============================================================================
\* Modification History
\* Created: 2025-12-19 by AI Worker N=1315
\* Based on: MPSStreamPool.tla, simplified for Apalache SMT compatibility
