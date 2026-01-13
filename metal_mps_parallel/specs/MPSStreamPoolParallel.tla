--------------------------- MODULE MPSStreamPoolParallel ---------------------------
\* TLA+ Specification for MPS Stream Pool Parallel Progress
\* Extends MPSStreamPool to track and verify parallel execution paths
\*
\* Purpose: Prove that the design permits at least one execution path where
\* two or more threads make concurrent progress (neither serialized nor blocked).
\*
\* This is an "aspirational property" per the verification paragon checklist.
\* It verifies that parallelism is NOT accidentally eliminated by the design.

EXTENDS Naturals, Sequences, FiniteSets, TLC

CONSTANTS
    NumStreams,         \* Total streams in pool (4 in this config for tractability)
    NumThreads,         \* Number of concurrent threads (3 for this config)
    MaxOperations       \* Bound for model checking (15 for this config)

ASSUME NumStreams > 1
ASSUME NumThreads >= 2   \* Need at least 2 threads to verify parallelism
ASSUME MaxOperations >= 1

VARIABLES
    streams,            \* Stream states: "free", "bound", "in_use"
    free_mask,          \* Bitmask of free worker slots
    thread_bindings,    \* Thread -> Stream binding
    cas_in_progress,    \* CAS critical section guard
    op_count,           \* Operation counter
    \* Parallel progress tracking
    parallel_count,     \* Count of states with simultaneous in_use observed
    max_parallel,       \* Maximum parallelism observed during trace
    parallel_witnessed  \* Has true parallelism (2+ threads in_use) been witnessed?

vars == <<streams, free_mask, thread_bindings, cas_in_progress, op_count,
          parallel_count, max_parallel, parallel_witnessed>>

parallel_vars == <<parallel_count, max_parallel, parallel_witnessed>>

-----------------------------------------------------------------------------
\* Helper functions (same as MPSStreamPool)
SlotToBit(slot) == slot - 1

BitSet(mask, bit) == (mask \div (2^bit)) % 2 = 1

SetBit(mask, bit) == mask + 2^bit

ClearBit(mask, bit) == mask - 2^bit

FindLowestSetBit(mask) ==
    LET bits == {b \in 0..(NumStreams-2) : BitSet(mask, b)}
    IN CHOOSE b \in bits : \A b2 \in bits : b <= b2

AllWorkerSlotsMask == (2^(NumStreams-1)) - 1

\* Count threads in "in_use" state based on given streams and bindings
CountInUse(s, b) ==
    Cardinality({t \in 1..NumThreads : b[t] > 0 /\ s[b[t]] = "in_use"})

\* Current count using unprimed variables
CountInUseNow == CountInUse(streams, thread_bindings)

-----------------------------------------------------------------------------
\* Type Invariant
TypeOK ==
    /\ streams \in [1..NumStreams -> {"free", "bound", "in_use"}]
    /\ free_mask \in 0..AllWorkerSlotsMask
    /\ thread_bindings \in [1..NumThreads -> 0..(NumStreams-1)]
    /\ cas_in_progress \in 0..NumThreads
    /\ op_count \in 0..MaxOperations
    /\ parallel_count \in Nat
    /\ max_parallel \in 0..NumThreads
    /\ parallel_witnessed \in BOOLEAN

\* Initial state
Init ==
    /\ streams = [s \in 1..NumStreams |-> "free"]
    /\ free_mask = AllWorkerSlotsMask
    /\ thread_bindings = [t \in 1..NumThreads |-> 0]
    /\ cas_in_progress = 0
    /\ op_count = 0
    /\ parallel_count = 0
    /\ max_parallel = 0
    /\ parallel_witnessed = FALSE

-----------------------------------------------------------------------------
\* Actions with parallel tracking integrated

AcquireStream(t) ==
    /\ op_count < MaxOperations
    /\ thread_bindings[t] = 0
    /\ cas_in_progress = 0
    /\ free_mask > 0
    /\ LET bit == FindLowestSetBit(free_mask)
           slot == bit + 1
           new_streams == [streams EXCEPT ![slot] = "bound"]
           new_bindings == [thread_bindings EXCEPT ![t] = slot]
           current == CountInUse(new_streams, new_bindings)
       IN /\ cas_in_progress' = t
          /\ streams' = new_streams
          /\ free_mask' = ClearBit(free_mask, bit)
          /\ thread_bindings' = new_bindings
          /\ op_count' = op_count + 1
          /\ parallel_count' = parallel_count + (IF current >= 2 THEN 1 ELSE 0)
          /\ max_parallel' = IF current > max_parallel THEN current ELSE max_parallel
          /\ parallel_witnessed' = parallel_witnessed \/ (current >= 2)

CompleteAcquire(t) ==
    /\ cas_in_progress = t
    /\ cas_in_progress' = 0
    /\ UNCHANGED <<streams, free_mask, thread_bindings, op_count>>
    /\ LET current == CountInUseNow
       IN /\ parallel_count' = parallel_count + (IF current >= 2 THEN 1 ELSE 0)
          /\ max_parallel' = IF current > max_parallel THEN current ELSE max_parallel
          /\ parallel_witnessed' = parallel_witnessed \/ (current >= 2)

ReleaseStream(t) ==
    /\ op_count < MaxOperations
    /\ thread_bindings[t] > 0
    /\ cas_in_progress = 0
    /\ LET slot == thread_bindings[t]
           bit == SlotToBit(slot)
           new_streams == [streams EXCEPT ![slot] = "free"]
           new_bindings == [thread_bindings EXCEPT ![t] = 0]
           current == CountInUse(new_streams, new_bindings)
       IN /\ streams' = new_streams
          /\ free_mask' = SetBit(free_mask, bit)
          /\ thread_bindings' = new_bindings
          /\ cas_in_progress' = 0
          /\ op_count' = op_count + 1
          /\ parallel_count' = parallel_count + (IF current >= 2 THEN 1 ELSE 0)
          /\ max_parallel' = IF current > max_parallel THEN current ELSE max_parallel
          /\ parallel_witnessed' = parallel_witnessed \/ (current >= 2)

UseStream(t) ==
    /\ op_count < MaxOperations
    /\ thread_bindings[t] > 0
    /\ cas_in_progress = 0
    /\ LET slot == thread_bindings[t]
           new_streams == [streams EXCEPT ![slot] = "in_use"]
           current == CountInUse(new_streams, thread_bindings)
           is_parallel == current >= 2
       IN /\ streams[slot] = "bound"
          /\ streams' = new_streams
          /\ UNCHANGED <<free_mask, thread_bindings, cas_in_progress>>
          /\ op_count' = op_count + 1
          /\ parallel_count' = parallel_count + (IF is_parallel THEN 1 ELSE 0)
          /\ max_parallel' = IF current > max_parallel THEN current ELSE max_parallel
          /\ parallel_witnessed' = IF is_parallel THEN TRUE ELSE parallel_witnessed

FinishUse(t) ==
    /\ op_count < MaxOperations
    /\ thread_bindings[t] > 0
    /\ cas_in_progress = 0
    /\ LET slot == thread_bindings[t]
           new_streams == [streams EXCEPT ![slot] = "bound"]
           current == CountInUse(new_streams, thread_bindings)
           is_parallel == current >= 2
       IN /\ streams[slot] = "in_use"
          /\ streams' = new_streams
          /\ UNCHANGED <<free_mask, thread_bindings, cas_in_progress>>
          /\ op_count' = op_count + 1
          /\ parallel_count' = parallel_count + (IF is_parallel THEN 1 ELSE 0)
          /\ max_parallel' = IF current > max_parallel THEN current ELSE max_parallel
          /\ parallel_witnessed' = IF is_parallel THEN TRUE ELSE parallel_witnessed

-----------------------------------------------------------------------------
\* Next state relation
Next ==
    \/ \E t \in 1..NumThreads :
        \/ AcquireStream(t)
        \/ CompleteAcquire(t)
        \/ ReleaseStream(t)
        \/ UseStream(t)
        \/ FinishUse(t)
    \/ UNCHANGED vars

Spec == Init /\ [][Next]_vars

-----------------------------------------------------------------------------
\* SAFETY PROPERTIES

MutualExclusion ==
    \A t1, t2 \in 1..NumThreads :
        (t1 /= t2 /\ thread_bindings[t1] > 0) =>
        thread_bindings[t1] /= thread_bindings[t2]

PoolIntegrity ==
    \A slot \in 2..NumStreams :
        LET bit == SlotToBit(slot)
        IN BitSet(free_mask, bit) =>
           (streams[slot] = "free" /\ \A t \in 1..NumThreads : thread_bindings[t] /= slot)

SafetyInvariant ==
    /\ TypeOK
    /\ MutualExclusion
    /\ PoolIntegrity

-----------------------------------------------------------------------------
\* PARALLEL PROGRESS PROPERTIES

\* Witness: Parallelism has been observed
ParallelWitnessed == parallel_witnessed

\* Current state has parallel execution
CurrentlyParallel == CountInUseNow >= 2

\* Maximum observed parallelism is within bounds
MaxParallelBounded == max_parallel <= NumThreads

\* Combined parallel progress invariant
ParallelProgressInvariant ==
    /\ MaxParallelBounded

-----------------------------------------------------------------------------
\* EXISTENCE CHECKING
\*
\* To verify parallelism EXISTS, we check these invariants expecting them to FAIL.
\* A counterexample proves the property CAN be achieved.

\* This property says parallelism is NEVER achieved - we expect it to FAIL
\* (finding a counterexample proves parallelism is possible)
NoParallelEver == ~parallel_witnessed

\* Alternative: Check that max_parallel stays < 2
\* (if this invariant fails, we found parallel execution)
MaxParallelNeverReaches2 == max_parallel < 2

-----------------------------------------------------------------------------
\* MODEL CHECKING NOTES:
\*
\* To verify parallel progress exists:
\* 1. Check NoParallelEver as an INVARIANT
\* 2. If TLC finds a counterexample, parallel execution is possible
\* 3. The counterexample trace shows HOW to achieve parallelism
\*
\* Expected result with NumStreams=4, NumThreads=3:
\* - TLC should find a state where 2+ threads are simultaneously in_use
\* - This proves the design permits true parallelism
\*
\* If TLC does NOT find a counterexample:
\* - The design may accidentally serialize all work
\* - Review global mutex usage, CAS bottlenecks, etc.

=============================================================================
\* Created: 2025-12-19 by AI Worker N=1302
\* Purpose: Phase 3 "Parallel Critical Section Exists" verification
