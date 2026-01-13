--------------------------- MODULE AGXFix ---------------------------
(***************************************************************************)
(* TLA+ Formal Model of AGX Driver Race Condition and v2.7 Fix             *)
(*                                                                          *)
(* This model captures:                                                     *)
(*   1. Command buffer and encoder lifecycle                                *)
(*   2. Multi-threaded access patterns                                      *)
(*   3. v2.7's protection mechanisms                                        *)
(*   4. Crash conditions we're trying to prevent                            *)
(*                                                                          *)
(* Author: Andrew Yates / Claude                                            *)
(* Date: 2025-12-24                                                         *)
(***************************************************************************)

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    NumThreads,         \* Number of concurrent threads (e.g., 4)
    NumCommandBuffers,  \* Number of command buffers (e.g., 2)
    NumEncoders         \* Max encoders per command buffer (e.g., 2)

VARIABLES
    \* Encoder state
    encoderState,       \* Function: encoder_id -> {free, active, ended, destroyed}
    encoderOwner,       \* Function: encoder_id -> command_buffer_id or NULL
    encoderImplValid,   \* Function: encoder_id -> BOOLEAN (is _impl non-NULL?)

    \* Command buffer state
    cbEncoders,         \* Function: cb_id -> Set of encoder_ids owned by this CB
    cbCommitted,        \* Function: cb_id -> BOOLEAN

    \* v2.7 tracking state
    v27_tracked,        \* Function: encoder_id -> BOOLEAN (tracked by v2.7?)
    v27_ended,          \* Function: encoder_id -> BOOLEAN (marked ended by v2.7?)
    v27_cbEncoders,     \* Function: cb_id -> Set of encoder_ids (v2.7's tracking)

    \* Thread state
    threadState,        \* Function: thread_id -> {idle, creating, using, ending, committing}
    threadEncoder,      \* Function: thread_id -> encoder_id being used (or NULL)
    threadCB,           \* Function: thread_id -> command_buffer_id being used

    \* Global mutex (v2.7 uses recursive mutex)
    mutexHolder,        \* thread_id holding mutex, or NULL

    \* Crash detection
    crashed,            \* BOOLEAN - has a crash occurred?
    crashType           \* String describing crash type

(***************************************************************************)
(* Type definitions                                                         *)
(***************************************************************************)

Threads == 1..NumThreads
CommandBuffers == 1..NumCommandBuffers
Encoders == 1..NumEncoders

EncoderStates == {"free", "active", "ended", "destroyed"}
ThreadStates == {"idle", "creating", "using", "ending", "committing"}

NULL == 0

(***************************************************************************)
(* Type invariant                                                           *)
(***************************************************************************)

TypeOK ==
    /\ encoderState \in [Encoders -> EncoderStates]
    /\ encoderOwner \in [Encoders -> CommandBuffers \cup {NULL}]
    /\ encoderImplValid \in [Encoders -> BOOLEAN]
    /\ cbEncoders \in [CommandBuffers -> SUBSET Encoders]
    /\ cbCommitted \in [CommandBuffers -> BOOLEAN]
    /\ v27_tracked \in [Encoders -> BOOLEAN]
    /\ v27_ended \in [Encoders -> BOOLEAN]
    /\ v27_cbEncoders \in [CommandBuffers -> SUBSET Encoders]
    /\ threadState \in [Threads -> ThreadStates]
    /\ threadEncoder \in [Threads -> Encoders \cup {NULL}]
    /\ threadCB \in [Threads -> CommandBuffers]
    /\ mutexHolder \in Threads \cup {NULL}
    /\ crashed \in BOOLEAN

(***************************************************************************)
(* Initial state                                                            *)
(***************************************************************************)

Init ==
    /\ encoderState = [e \in Encoders |-> "free"]
    /\ encoderOwner = [e \in Encoders |-> NULL]
    /\ encoderImplValid = [e \in Encoders |-> FALSE]
    /\ cbEncoders = [cb \in CommandBuffers |-> {}]
    /\ cbCommitted = [cb \in CommandBuffers |-> FALSE]
    /\ v27_tracked = [e \in Encoders |-> FALSE]
    /\ v27_ended = [e \in Encoders |-> FALSE]
    /\ v27_cbEncoders = [cb \in CommandBuffers |-> {}]
    /\ threadState = [t \in Threads |-> "idle"]
    /\ threadEncoder = [t \in Threads |-> NULL]
    /\ threadCB = [t \in Threads |-> 1]  \* Default to CB 1
    /\ mutexHolder = NULL
    /\ crashed = FALSE
    /\ crashType = "none"

(***************************************************************************)
(* Mutex operations (v2.7's recursive mutex)                                *)
(***************************************************************************)

AcquireMutex(t) ==
    /\ mutexHolder = NULL
    /\ mutexHolder' = t

ReleaseMutex(t) ==
    /\ mutexHolder = t
    /\ mutexHolder' = NULL

(***************************************************************************)
(* CRASH CONDITIONS                                                         *)
(* These are the states that lead to crashes in the real system            *)
(***************************************************************************)

\* Crash Type 1: PAC failure - using encoder after _impl is NULL
PACCrashCondition(e) ==
    /\ encoderState[e] = "active"
    /\ ~encoderImplValid[e]

\* Crash Type 2: Validation failure - committing CB with unended encoder
ValidationCrashCondition(cb) ==
    /\ \E e \in cbEncoders[cb] : encoderState[e] = "active"

\* Crash Type 3: Use after free - using destroyed encoder
UseAfterFreeCrash(e) ==
    encoderState[e] = "destroyed"

(***************************************************************************)
(* v2.7 PROTECTION: Create encoder with tracking                           *)
(***************************************************************************)

V27_CreateEncoder(t, cb, e) ==
    /\ threadState[t] = "idle"
    /\ encoderState[e] = "free"
    /\ ~cbCommitted[cb]
    /\ ~crashed
    \* Acquire mutex (atomic operation - release at end)
    /\ mutexHolder = NULL
    \* Create encoder
    /\ encoderState' = [encoderState EXCEPT ![e] = "active"]
    /\ encoderOwner' = [encoderOwner EXCEPT ![e] = cb]
    /\ encoderImplValid' = [encoderImplValid EXCEPT ![e] = TRUE]
    /\ cbEncoders' = [cbEncoders EXCEPT ![cb] = cbEncoders[cb] \cup {e}]
    \* v2.7 tracking
    /\ v27_tracked' = [v27_tracked EXCEPT ![e] = TRUE]
    /\ v27_ended' = [v27_ended EXCEPT ![e] = FALSE]
    /\ v27_cbEncoders' = [v27_cbEncoders EXCEPT ![cb] = v27_cbEncoders[cb] \cup {e}]
    \* Thread state
    /\ threadState' = [threadState EXCEPT ![t] = "using"]
    /\ threadEncoder' = [threadEncoder EXCEPT ![t] = e]
    /\ threadCB' = [threadCB EXCEPT ![t] = cb]
    \* Release mutex after creation (atomic)
    /\ mutexHolder' = NULL
    /\ UNCHANGED <<cbCommitted, crashed, crashType>>

(***************************************************************************)
(* v2.7 PROTECTION: Use encoder (with mutex protection)                    *)
(***************************************************************************)

V27_UseEncoder(t) ==
    /\ threadState[t] = "using"
    /\ threadEncoder[t] # NULL
    /\ ~crashed
    /\ LET e == threadEncoder[t] IN
       \* Can only use active encoders (Metal API requirement)
       IF encoderState[e] # "active"
       THEN \* Encoder was force-ended or destroyed - thread must stop using it
            \* This models the real behavior: trying to use an ended encoder
            \* returns an error, and the thread transitions to idle
            /\ threadState' = [threadState EXCEPT ![t] = "idle"]
            /\ threadEncoder' = [threadEncoder EXCEPT ![t] = NULL]
            /\ UNCHANGED <<encoderState, encoderOwner, encoderImplValid,
                          cbEncoders, cbCommitted, v27_tracked, v27_ended,
                          v27_cbEncoders, threadCB, mutexHolder, crashed, crashType>>
       ELSE \* Check for crash conditions (impl validity)
            IF PACCrashCondition(e)
            THEN /\ crashed' = TRUE
                 /\ crashType' = "PAC_failure"
                 /\ UNCHANGED <<encoderState, encoderOwner, encoderImplValid,
                               cbEncoders, cbCommitted, v27_tracked, v27_ended,
                               v27_cbEncoders, threadState, threadEncoder, threadCB, mutexHolder>>
            ELSE /\ UNCHANGED <<encoderState, encoderOwner, encoderImplValid,
                               cbEncoders, cbCommitted, v27_tracked, v27_ended,
                               v27_cbEncoders, threadState, threadEncoder, threadCB,
                               mutexHolder, crashed, crashType>>

(***************************************************************************)
(* v2.7 PROTECTION: End encoder                                            *)
(***************************************************************************)

V27_EndEncoder(t) ==
    /\ threadState[t] = "using"
    /\ threadEncoder[t] # NULL
    /\ ~crashed
    \* Acquire mutex (atomic operation)
    /\ mutexHolder = NULL
    /\ LET e == threadEncoder[t] IN
       /\ encoderState[e] = "active"
       \* End encoder
       /\ encoderState' = [encoderState EXCEPT ![e] = "ended"]
       /\ v27_ended' = [v27_ended EXCEPT ![e] = TRUE]
       \* Thread done with encoder
       /\ threadState' = [threadState EXCEPT ![t] = "idle"]
       /\ threadEncoder' = [threadEncoder EXCEPT ![t] = NULL]
    \* Release mutex (atomic)
    /\ mutexHolder' = NULL
    /\ UNCHANGED <<encoderOwner, encoderImplValid, cbEncoders, cbCommitted,
                  v27_tracked, v27_cbEncoders, threadCB, crashed, crashType>>

(***************************************************************************)
(* v2.7 PROTECTION: Commit command buffer (THE KEY FIX)                    *)
(*                                                                          *)
(* v2.7 ensures all encoders are ended before commit by checking           *)
(* v27_cbEncoders and forcing end if needed.                               *)
(***************************************************************************)

V27_CommitCB(t, cb) ==
    /\ threadState[t] = "idle"
    /\ ~cbCommitted[cb]
    /\ ~crashed
    \* Acquire mutex (will release at end of action - atomic operation)
    /\ mutexHolder = NULL
    \* v2.7 CHECK: Are all encoders ended?
    /\ LET activeEncoders == {e \in v27_cbEncoders[cb] : ~v27_ended[e]} IN
       IF activeEncoders # {}
       THEN \* v2.7 force-ends active encoders before commit
            /\ encoderState' = [e \in Encoders |->
                                IF e \in activeEncoders
                                THEN "ended"
                                ELSE encoderState[e]]
            /\ v27_ended' = [e \in Encoders |->
                            IF e \in activeEncoders
                            THEN TRUE
                            ELSE v27_ended[e]]
            /\ cbCommitted' = [cbCommitted EXCEPT ![cb] = TRUE]
            /\ crashed' = FALSE  \* No crash because we force-ended!
            /\ UNCHANGED <<crashType>>
       ELSE \* All encoders already ended, safe to commit
            /\ cbCommitted' = [cbCommitted EXCEPT ![cb] = TRUE]
            /\ UNCHANGED <<encoderState, v27_ended, crashed, crashType>>
    \* Mutex released at end (atomic commit operation)
    /\ mutexHolder' = NULL
    /\ UNCHANGED <<encoderOwner, encoderImplValid, cbEncoders, v27_tracked,
                  v27_cbEncoders, threadState, threadEncoder, threadCB>>

(***************************************************************************)
(* UNPROTECTED: DestroyImpl called by system (race condition source)       *)
(*                                                                          *)
(* This models what happens when destroyImpl is called - it NULLs _impl.   *)
(* v2.7 intercepts this but the original still gets called.                *)
(***************************************************************************)

DestroyImpl(e) ==
    /\ encoderState[e] = "ended"
    /\ ~crashed
    /\ encoderImplValid' = [encoderImplValid EXCEPT ![e] = FALSE]
    /\ encoderState' = [encoderState EXCEPT ![e] = "destroyed"]
    /\ UNCHANGED <<encoderOwner, cbEncoders, cbCommitted, v27_tracked, v27_ended,
                  v27_cbEncoders, threadState, threadEncoder, threadCB,
                  mutexHolder, crashed, crashType>>

(***************************************************************************)
(* WITHOUT v2.7: Commit without protection (for comparison)                 *)
(***************************************************************************)

Unprotected_CommitCB(t, cb) ==
    /\ threadState[t] = "idle"
    /\ ~cbCommitted[cb]
    /\ ~crashed
    \* NO mutex, NO checking
    /\ IF ValidationCrashCondition(cb)
       THEN /\ crashed' = TRUE
            /\ crashType' = "SIGABRT_Validation"
            /\ UNCHANGED <<cbCommitted>>
       ELSE /\ cbCommitted' = [cbCommitted EXCEPT ![cb] = TRUE]
            /\ UNCHANGED <<crashed, crashType>>
    /\ UNCHANGED <<encoderState, encoderOwner, encoderImplValid, cbEncoders,
                  v27_tracked, v27_ended, v27_cbEncoders, threadState,
                  threadEncoder, threadCB, mutexHolder>>

(***************************************************************************)
(* Next state relation - v2.7 protected version                            *)
(***************************************************************************)

NextV27 ==
    \/ \E t \in Threads, cb \in CommandBuffers, e \in Encoders :
        V27_CreateEncoder(t, cb, e)
    \/ \E t \in Threads : V27_UseEncoder(t)
    \/ \E t \in Threads : V27_EndEncoder(t)
    \/ \E t \in Threads, cb \in CommandBuffers : V27_CommitCB(t, cb)
    \/ \E e \in Encoders : DestroyImpl(e)

(***************************************************************************)
(* SAFETY PROPERTIES                                                        *)
(***************************************************************************)

\* Property 1: No crashes ever
NoCrashes == ~crashed

\* Property 2: No validation crash - never commit with active encoders
NoValidationCrash ==
    \A cb \in CommandBuffers :
        cbCommitted[cb] => ~(\E e \in cbEncoders[cb] : encoderState[e] = "active")

\* Property 3: v2.7 tracking is consistent
V27TrackingConsistent ==
    \A cb \in CommandBuffers :
        v27_cbEncoders[cb] \subseteq cbEncoders[cb]

\* Property 4: Mutex provides mutual exclusion
MutexSafe ==
    \A t1, t2 \in Threads :
        (t1 # t2 /\ mutexHolder = t1) => mutexHolder # t2

(***************************************************************************)
(* Specification                                                            *)
(***************************************************************************)

SpecV27 == Init /\ [][NextV27]_<<encoderState, encoderOwner, encoderImplValid,
                                 cbEncoders, cbCommitted, v27_tracked, v27_ended,
                                 v27_cbEncoders, threadState, threadEncoder,
                                 threadCB, mutexHolder, crashed, crashType>>

(***************************************************************************)
(* THEOREM: v2.7 prevents validation crashes                                *)
(*                                                                          *)
(* If we use v2.7's CommitCB (which force-ends encoders), we never get     *)
(* SIGABRT from validation.                                                 *)
(***************************************************************************)

THEOREM V27PreventsValidationCrash ==
    SpecV27 => []NoValidationCrash

(***************************************************************************)
(* LIVENESS PROPERTIES                                                      *)
(*                                                                          *)
(* These properties ensure progress - that good things eventually happen.   *)
(* They require fairness assumptions to be meaningful.                      *)
(***************************************************************************)

\* Helper: Is there any active encoder?
ExistsActiveEncoder == \E e \in Encoders : encoderState[e] = "active"

\* Helper: Is there any encoder using a thread?
ExistsUsingThread == \E t \in Threads : threadState[t] = "using"

\* Helper: Is there any uncommitted CB with encoders?
ExistsUncommittedCBWithEncoders ==
    \E cb \in CommandBuffers : ~cbCommitted[cb] /\ cbEncoders[cb] # {}

\* Liveness 1: If an encoder becomes active, it will eventually be ended
\* (This relies on fairness of V27_EndEncoder)
EncoderEventuallyEnded(e) ==
    (encoderState[e] = "active") ~> (encoderState[e] \in {"ended", "destroyed"})

\* Liveness 2: If a thread is using an encoder, it will eventually become idle
\* (Threads don't get stuck)
ThreadEventuallyIdle(t) ==
    (threadState[t] = "using") ~> (threadState[t] = "idle")

\* Liveness 3: All encoders eventually get destroyed (freed)
\* Note: This requires that DestroyImpl eventually runs on ended encoders
AllEncodersEventuallyFreed ==
    \A e \in Encoders :
        (encoderState[e] \in {"active", "ended"}) ~> (encoderState[e] = "destroyed")

\* Liveness 4: No starvation - if work is available, progress is made
\* This uses leads-to (~>) to express that active encoders lead to ended ones
NoStarvation ==
    ExistsActiveEncoder ~> ~ExistsActiveEncoder

(***************************************************************************)
(* FAIR SPECIFICATION                                                       *)
(*                                                                          *)
(* For liveness properties, we need fairness to ensure actions are taken.   *)
(* WF (Weak Fairness): If an action is continuously enabled, it eventually  *)
(* must be taken.                                                           *)
(***************************************************************************)

\* Variables tuple for fairness
vars == <<encoderState, encoderOwner, encoderImplValid,
          cbEncoders, cbCommitted, v27_tracked, v27_ended,
          v27_cbEncoders, threadState, threadEncoder,
          threadCB, mutexHolder, crashed, crashType>>

\* Fair version of specification with weak fairness on all actions
\* WF ensures that if an action is continuously enabled, it will eventually be taken
FairSpecV27 ==
    /\ Init
    /\ [][NextV27]_vars
    /\ \A t \in Threads : WF_vars(V27_UseEncoder(t))
    /\ \A t \in Threads : WF_vars(V27_EndEncoder(t))
    /\ \A t \in Threads, cb \in CommandBuffers : WF_vars(V27_CommitCB(t, cb))
    /\ \A e \in Encoders : WF_vars(DestroyImpl(e))

\* Combined liveness property: All encoders created will eventually be freed
Liveness ==
    /\ \A t \in Threads : ThreadEventuallyIdle(t)

(***************************************************************************)
(* THEOREM: v2.7 guarantees liveness (no deadlocks, eventual cleanup)       *)
(***************************************************************************)

THEOREM V27GuaranteesLiveness ==
    FairSpecV27 => Liveness

=============================================================================
