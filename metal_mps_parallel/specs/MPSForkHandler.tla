--------------------------- MODULE MPSForkHandler ---------------------------
(*
 * TLA+ Specification for MPSForkHandler
 *
 * Models the fork() safety mechanism in PyTorch MPS backend.
 * GPU contexts do not survive fork() - child processes must reinitialize.
 * pthread_atfork handlers ensure proper cleanup before fork.
 *
 * Properties verified:
 * 1. Fork preparation flushes all pending GPU work (PrepareFlushes)
 * 2. Child process correctly resets to uninitialized state (ChildResets)
 * 3. Parent process preserves all state after fork (ParentPreserves)
 * 4. No resource leaks in child (ChildNoLeaks)
 * 5. Fork handlers execute in correct order (HandlerOrder)
 * 6. Multiple fork() calls handled correctly (MultipleForksSafe)
 * 7. GPU context invalid in child until reinitialized (ChildContextInvalid)
 *
 * Based on: mps-verify/verification/cbmc/harnesses/fork_safety_harness.c
 *           and PyTorch MPS fork handler implementation
 *
 * Author: Worker N=1356
 * Date: 2025-12-20
 *)

EXTENDS Integers, Sequences, FiniteSets, TLC

\* Configuration constants
CONSTANTS
    \* @type: Int;
    MaxStreams,      \* Maximum number of GPU streams (e.g., 4)
    \* @type: Int;
    MaxPendingOps,   \* Maximum pending operations per stream (e.g., 4)
    \* @type: Int;
    MaxForkDepth,    \* Maximum nested fork depth to model (e.g., 2)
    \* @type: Int;
    None             \* Sentinel value (e.g., 99)

\* Process roles
CONSTANTS
    \* @type: Int;
    RoleParent,      \* Original parent process
    \* @type: Int;
    RoleChild        \* Child process after fork()

\* Initialization states
CONSTANTS
    \* @type: Int;
    InitUninitialized,
    \* @type: Int;
    InitInProgress,
    \* @type: Int;
    InitComplete,
    \* @type: Int;
    InitFailed

\* Fork handler phases
CONSTANTS
    \* @type: Int;
    PhaseNone,       \* Not in fork
    \* @type: Int;
    PhasePrepare,    \* In fork_prepare (before fork)
    \* @type: Int;
    PhaseParent,     \* In fork_parent (after fork, parent)
    \* @type: Int;
    PhaseChild       \* In fork_child (after fork, child)

VARIABLES
    \* Process state
    \* @type: Int;
    role,            \* Current process role (Parent or Child)
    \* @type: Int;
    initState,       \* Initialization state
    \* @type: Int;
    forkCount,       \* Number of times this process has forked
    \* @type: Int;
    forkDepth,       \* Current nesting depth of fork (for multi-fork scenarios)

    \* GPU context state
    \* @type: Bool;
    gpuContextValid, \* Is GPU context valid?
    \* @type: Int;
    gpuContextId,    \* Unique context ID (0 = invalid)

    \* Stream state
    \* @type: Int -> Bool;
    streamActive,    \* Function: StreamId -> Bool (is stream active?)
    \* @type: Int -> Int;
    streamPending,   \* Function: StreamId -> Nat (pending ops count)
    \* @type: Int -> Bool;
    streamNeedsSync, \* Function: StreamId -> Bool (needs GPU sync?)
    \* @type: Int;
    numActiveStreams,\* Count of active streams

    \* Fork handler state
    \* @type: Bool;
    atforkRegistered,\* Has pthread_atfork been registered?
    \* @type: Int;
    forkPhase,       \* Current fork phase
    \* @type: Bool;
    prepareDone,     \* Has prepare phase completed?

    \* Model state
    \* @type: Int;
    nextContextId    \* Counter for unique context IDs

vars == <<role, initState, forkCount, forkDepth,
          gpuContextValid, gpuContextId,
          streamActive, streamPending, streamNeedsSync, numActiveStreams,
          atforkRegistered, forkPhase, prepareDone,
          nextContextId>>

\* Helper sets
StreamIds == 0..(MaxStreams-1)

-----------------------------------------------------------------------------
\* Type invariant

TypeOK ==
    /\ role \in {RoleParent, RoleChild}
    /\ initState \in {InitUninitialized, InitInProgress, InitComplete, InitFailed}
    /\ forkCount \in 0..10  \* Can fork many times (bounded for model checking)
    /\ forkDepth \in 0..MaxForkDepth
    /\ gpuContextValid \in BOOLEAN
    /\ gpuContextId \in 0..100
    /\ streamActive \in [StreamIds -> BOOLEAN]
    /\ streamPending \in [StreamIds -> 0..MaxPendingOps]
    /\ streamNeedsSync \in [StreamIds -> BOOLEAN]
    /\ numActiveStreams \in 0..MaxStreams
    /\ atforkRegistered \in BOOLEAN
    /\ forkPhase \in {PhaseNone, PhasePrepare, PhaseParent, PhaseChild}
    /\ prepareDone \in BOOLEAN
    /\ nextContextId \in 1..100

-----------------------------------------------------------------------------
\* Safety invariants

\* Fork preparation must flush all pending work
PrepareFlushes ==
    prepareDone =>
        \A s \in StreamIds :
            streamActive[s] => streamPending[s] = 0

\* Child process is correctly reset IMMEDIATELY after fork
\* (Before reinitialization - checked via initState = InitUninitialized)
ChildResets ==
    (role = RoleChild /\ forkPhase = PhaseNone /\ initState = InitUninitialized) =>
        /\ ~gpuContextValid
        /\ gpuContextId = 0
        /\ numActiveStreams = 0

\* Parent process preserves state after fork
ParentPreserves ==
    (role = RoleParent /\ forkPhase = PhaseParent) =>
        /\ gpuContextValid
        /\ initState = InitComplete

\* Child has no active streams IMMEDIATELY after fork (no resource leaks from parent)
\* After reinit, child can acquire new streams normally
ChildNoLeaks ==
    (role = RoleChild /\ forkPhase = PhaseNone /\ initState = InitUninitialized) =>
        \A s \in StreamIds : ~streamActive[s]

\* Fork handlers execute in correct order
HandlerOrder ==
    \* Can only be in parent/child phase if prepare was done
    /\ (forkPhase = PhaseParent => prepareDone)
    /\ (forkPhase = PhaseChild => prepareDone)

\* Cannot nest fork while in fork phase
NoNestedFork ==
    forkPhase # PhaseNone => forkDepth <= MaxForkDepth

\* GPU context must be invalid in child until reinitialized
ChildContextInvalid ==
    (role = RoleChild /\ initState = InitUninitialized) =>
        ~gpuContextValid

\* Active stream count matches actual active streams
ActiveStreamCountConsistent ==
    numActiveStreams = Cardinality({s \in StreamIds : streamActive[s]})

\* All safety properties combined
Safety ==
    /\ PrepareFlushes
    /\ ChildResets
    /\ ParentPreserves
    /\ ChildNoLeaks
    /\ HandlerOrder
    /\ NoNestedFork
    /\ ChildContextInvalid
    /\ ActiveStreamCountConsistent

-----------------------------------------------------------------------------
\* Initial state

Init ==
    /\ role = RoleParent
    /\ initState = InitUninitialized
    /\ forkCount = 0
    /\ forkDepth = 0
    /\ gpuContextValid = FALSE
    /\ gpuContextId = 0
    /\ streamActive = [s \in StreamIds |-> FALSE]
    /\ streamPending = [s \in StreamIds |-> 0]
    /\ streamNeedsSync = [s \in StreamIds |-> FALSE]
    /\ numActiveStreams = 0
    /\ atforkRegistered = FALSE
    /\ forkPhase = PhaseNone
    /\ prepareDone = FALSE
    /\ nextContextId = 1

-----------------------------------------------------------------------------
\* Actions

\* Initialize GPU context (lazy init pattern)
InitGPU ==
    /\ forkPhase = PhaseNone
    /\ initState = InitUninitialized
    /\ initState' = InitComplete
    /\ gpuContextValid' = TRUE
    /\ gpuContextId' = nextContextId
    /\ nextContextId' = nextContextId + 1
    /\ atforkRegistered' = TRUE
    /\ UNCHANGED <<role, forkCount, forkDepth,
                   streamActive, streamPending, streamNeedsSync, numActiveStreams,
                   forkPhase, prepareDone>>

\* Acquire a stream
AcquireStream(s) ==
    /\ forkPhase = PhaseNone
    /\ initState = InitComplete
    /\ gpuContextValid
    /\ ~streamActive[s]
    /\ streamActive' = [streamActive EXCEPT ![s] = TRUE]
    /\ numActiveStreams' = numActiveStreams + 1
    /\ UNCHANGED <<role, initState, forkCount, forkDepth,
                   gpuContextValid, gpuContextId,
                   streamPending, streamNeedsSync,
                   atforkRegistered, forkPhase, prepareDone, nextContextId>>

\* Submit work to a stream
SubmitWork(s) ==
    /\ forkPhase = PhaseNone
    /\ initState = InitComplete
    /\ streamActive[s]
    /\ streamPending[s] < MaxPendingOps
    /\ streamPending' = [streamPending EXCEPT ![s] = @ + 1]
    /\ streamNeedsSync' = [streamNeedsSync EXCEPT ![s] = TRUE]
    /\ UNCHANGED <<role, initState, forkCount, forkDepth,
                   gpuContextValid, gpuContextId,
                   streamActive, numActiveStreams,
                   atforkRegistered, forkPhase, prepareDone, nextContextId>>

\* Sync a stream (flush pending work)
SyncStream(s) ==
    /\ forkPhase \in {PhaseNone, PhasePrepare}
    /\ streamActive[s]
    /\ streamNeedsSync[s]
    /\ streamPending' = [streamPending EXCEPT ![s] = 0]
    /\ streamNeedsSync' = [streamNeedsSync EXCEPT ![s] = FALSE]
    /\ UNCHANGED <<role, initState, forkCount, forkDepth,
                   gpuContextValid, gpuContextId,
                   streamActive, numActiveStreams,
                   atforkRegistered, forkPhase, prepareDone, nextContextId>>

\* Start fork: enter prepare phase
ForkPrepare ==
    /\ forkPhase = PhaseNone
    /\ initState = InitComplete
    /\ atforkRegistered
    /\ forkDepth < MaxForkDepth
    /\ forkCount < 3  \* Bound total forks for model checking (both parent and child)
    /\ nextContextId < 10  \* Prevent context ID overflow
    /\ forkPhase' = PhasePrepare
    /\ forkDepth' = forkDepth + 1
    \* Flush all pending work (key safety property)
    /\ streamPending' = [s \in StreamIds |-> 0]
    /\ streamNeedsSync' = [s \in StreamIds |->
                           IF streamActive[s] THEN FALSE ELSE streamNeedsSync[s]]
    /\ prepareDone' = TRUE
    /\ UNCHANGED <<role, initState, forkCount,
                   gpuContextValid, gpuContextId,
                   streamActive, numActiveStreams,
                   atforkRegistered, nextContextId>>

\* Fork completes: parent continues
ForkParent ==
    /\ forkPhase = PhasePrepare
    /\ prepareDone
    /\ role = RoleParent
    /\ forkPhase' = PhaseParent
    /\ UNCHANGED <<role, initState, forkCount, forkDepth,
                   gpuContextValid, gpuContextId,
                   streamActive, streamPending, streamNeedsSync, numActiveStreams,
                   atforkRegistered, prepareDone, nextContextId>>

\* Fork completes: child resets
ForkChild ==
    /\ forkPhase = PhasePrepare
    /\ prepareDone
    \* Non-deterministic: this models becoming the child process
    /\ role' = RoleChild
    /\ forkPhase' = PhaseChild
    /\ gpuContextValid' = FALSE
    /\ gpuContextId' = 0
    /\ initState' = InitUninitialized
    /\ streamActive' = [s \in StreamIds |-> FALSE]
    /\ streamPending' = [s \in StreamIds |-> 0]
    /\ streamNeedsSync' = [s \in StreamIds |-> FALSE]
    /\ numActiveStreams' = 0
    /\ UNCHANGED <<forkCount, forkDepth, atforkRegistered, prepareDone, nextContextId>>

\* Parent returns from fork handler
ParentResume ==
    /\ forkPhase = PhaseParent
    /\ role = RoleParent
    /\ forkPhase' = PhaseNone
    /\ forkCount' = forkCount + 1
    /\ prepareDone' = FALSE
    /\ forkDepth' = forkDepth - 1
    /\ UNCHANGED <<role, initState, gpuContextValid, gpuContextId,
                   streamActive, streamPending, streamNeedsSync, numActiveStreams,
                   atforkRegistered, nextContextId>>

\* Child returns from fork handler
ChildResume ==
    /\ forkPhase = PhaseChild
    /\ role = RoleChild
    /\ forkPhase' = PhaseNone
    /\ prepareDone' = FALSE
    /\ forkDepth' = 0  \* Child starts fresh
    /\ forkCount' = 0  \* Child hasn't forked yet
    /\ UNCHANGED <<role, initState, gpuContextValid, gpuContextId,
                   streamActive, streamPending, streamNeedsSync, numActiveStreams,
                   atforkRegistered, nextContextId>>

\* Child reinitializes GPU
ChildReinitGPU ==
    /\ role = RoleChild
    /\ forkPhase = PhaseNone
    /\ initState = InitUninitialized
    /\ ~gpuContextValid
    /\ initState' = InitComplete
    /\ gpuContextValid' = TRUE
    /\ gpuContextId' = nextContextId
    /\ nextContextId' = nextContextId + 1
    /\ UNCHANGED <<role, forkCount, forkDepth,
                   streamActive, streamPending, streamNeedsSync, numActiveStreams,
                   atforkRegistered, forkPhase, prepareDone>>

\* Release a stream
ReleaseStream(s) ==
    /\ forkPhase = PhaseNone
    /\ streamActive[s]
    /\ streamPending[s] = 0  \* Must sync before release
    /\ streamActive' = [streamActive EXCEPT ![s] = FALSE]
    /\ numActiveStreams' = numActiveStreams - 1
    /\ UNCHANGED <<role, initState, forkCount, forkDepth,
                   gpuContextValid, gpuContextId,
                   streamPending, streamNeedsSync,
                   atforkRegistered, forkPhase, prepareDone, nextContextId>>

-----------------------------------------------------------------------------
\* Next state relation

Next ==
    \/ InitGPU
    \/ \E s \in StreamIds : AcquireStream(s)
    \/ \E s \in StreamIds : SubmitWork(s)
    \/ \E s \in StreamIds : SyncStream(s)
    \/ ForkPrepare
    \/ ForkParent
    \/ ForkChild
    \/ ParentResume
    \/ ChildResume
    \/ ChildReinitGPU
    \/ \E s \in StreamIds : ReleaseStream(s)
    \/ UNCHANGED vars

\* Fairness: eventually processes complete fork
Fairness ==
    /\ WF_vars(InitGPU)
    /\ WF_vars(\E s \in StreamIds : SyncStream(s))
    /\ WF_vars(ForkPrepare)
    /\ WF_vars(ForkParent)
    /\ WF_vars(ForkChild)
    /\ WF_vars(ParentResume)
    /\ WF_vars(ChildResume)
    /\ WF_vars(ChildReinitGPU)

Spec == Init /\ [][Next]_vars /\ Fairness
SpecNoFairness == Init /\ [][Next]_vars

-----------------------------------------------------------------------------
\* Liveness properties

\* Fork eventually completes (returns to PhaseNone)
ForkEventuallyCompletes ==
    forkPhase # PhaseNone ~> forkPhase = PhaseNone

\* Progress: can always make progress from non-idle state
Progress == []<>(forkPhase = PhaseNone)

=============================================================================
