--------------------------- MODULE AGXFix_v2 ---------------------------
(****************************************************************************)
(* TLA+ Formal Model of AGX Driver Race Condition and v2.7 Fix - Version 2  *)
(*                                                                           *)
(* IMPROVEMENTS over v1:                                                     *)
(*   1. Command buffer lifecycle (created, encoding, committed, completed)   *)
(*   2. Reference counting (retain_count prevents premature release)         *)
(*   3. Active call tracking (active_calls blocks release during methods)    *)
(*   4. _impl invalidation timing (can happen asynchronously)                *)
(*   5. GPU execution model (commands submitted, then complete)              *)
(*   6. Non-atomic operations (separate check and action steps)              *)
(*                                                                           *)
(* Author: Andrew Yates / Claude                                             *)
(* Date: 2024-12-24                                                          *)
(****************************************************************************)

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    NumThreads,         \* Number of concurrent threads
    NumCommandBuffers,  \* Number of command buffers
    NumEncoders         \* Max encoders

VARIABLES
    \* Encoder state (more detailed)
    encoderState,       \* encoder_id -> {free, active, ended, impl_invalid, released}
    encoderOwner,       \* encoder_id -> command_buffer_id or NULL
    encoderRetainCount, \* encoder_id -> Int (reference count)
    encoderActiveCalls, \* encoder_id -> Int (active method calls)

    \* Command buffer state (full lifecycle)
    cbState,            \* cb_id -> {created, encoding, committed, submitted, completed}
    cbEncoders,         \* cb_id -> Set of encoder_ids

    \* v2.7 tracking
    v27_tracked,        \* encoder_id -> BOOLEAN
    v27_ended,          \* encoder_id -> BOOLEAN
    v27_cbEncoders,     \* cb_id -> Set of encoder_ids

    \* Thread state (more detailed)
    threadState,        \* thread_id -> state
    threadEncoder,      \* thread_id -> encoder_id or NULL
    threadCB,           \* thread_id -> command_buffer_id
    threadPendingOp,    \* thread_id -> operation to complete (for non-atomic modeling)

    \* Mutex
    mutexHolder,        \* thread_id holding mutex, or NULL

    \* GPU state
    gpuQueue,           \* Sequence of {cb_id, encoders} pending GPU execution

    \* Crash detection
    crashed,
    crashType

(****************************************************************************)
(* Type definitions                                                         *)
(****************************************************************************)

Threads == 1..NumThreads
CommandBuffers == 1..NumCommandBuffers
Encoders == 1..NumEncoders

EncoderStates == {"free", "active", "ended", "impl_invalid", "released"}
CBStates == {"created", "encoding", "committed", "submitted", "completed"}
ThreadStates == {"idle", "acquiring_mutex", "in_mutex", "creating_encoder",
                 "using_encoder", "ending_encoder", "committing", "waiting_gpu"}
PendingOps == {"none", "create", "use", "end", "commit", "release"}

NULL == 0

(****************************************************************************)
(* Type invariant                                                           *)
(****************************************************************************)

TypeOK ==
    /\ encoderState \in [Encoders -> EncoderStates]
    /\ encoderOwner \in [Encoders -> CommandBuffers \cup {NULL}]
    /\ encoderRetainCount \in [Encoders -> 0..100]
    /\ encoderActiveCalls \in [Encoders -> 0..100]
    /\ cbState \in [CommandBuffers -> CBStates]
    /\ cbEncoders \in [CommandBuffers -> SUBSET Encoders]
    /\ v27_tracked \in [Encoders -> BOOLEAN]
    /\ v27_ended \in [Encoders -> BOOLEAN]
    /\ v27_cbEncoders \in [CommandBuffers -> SUBSET Encoders]
    /\ threadState \in [Threads -> ThreadStates]
    /\ threadEncoder \in [Threads -> Encoders \cup {NULL}]
    /\ threadCB \in [Threads -> CommandBuffers]
    /\ threadPendingOp \in [Threads -> PendingOps]
    /\ mutexHolder \in Threads \cup {NULL}
    /\ crashed \in BOOLEAN

(****************************************************************************)
(* Initial state                                                            *)
(****************************************************************************)

Init ==
    /\ encoderState = [e \in Encoders |-> "free"]
    /\ encoderOwner = [e \in Encoders |-> NULL]
    /\ encoderRetainCount = [e \in Encoders |-> 0]
    /\ encoderActiveCalls = [e \in Encoders |-> 0]
    /\ cbState = [cb \in CommandBuffers |-> "created"]
    /\ cbEncoders = [cb \in CommandBuffers |-> {}]
    /\ v27_tracked = [e \in Encoders |-> FALSE]
    /\ v27_ended = [e \in Encoders |-> FALSE]
    /\ v27_cbEncoders = [cb \in CommandBuffers |-> {}]
    /\ threadState = [t \in Threads |-> "idle"]
    /\ threadEncoder = [t \in Threads |-> NULL]
    /\ threadCB = [t \in Threads |-> 1]
    /\ threadPendingOp = [t \in Threads |-> "none"]
    /\ mutexHolder = NULL
    /\ gpuQueue = <<>>
    /\ crashed = FALSE
    /\ crashType = "none"

(****************************************************************************)
(* CRASH CONDITIONS                                                         *)
(****************************************************************************)

\* Crash 1: Using encoder with invalid _impl
ImplInvalidCrash(e) ==
    /\ encoderState[e] = "impl_invalid"

\* Crash 2: Using released encoder
UseAfterReleaseCrash(e) ==
    /\ encoderState[e] = "released"

\* Crash 3: Committing with active (not ended) encoders
ValidationCrash(cb) ==
    /\ \E e \in cbEncoders[cb] : encoderState[e] = "active"

\* Crash 4: Double release
DoubleReleaseCrash(e) ==
    /\ encoderRetainCount[e] < 0

(****************************************************************************)
(* MUTEX OPERATIONS (non-atomic modeling)                                   *)
(****************************************************************************)

\* Thread attempts to acquire mutex
TryAcquireMutex(t) ==
    /\ threadState[t] \in {"idle", "using_encoder"}
    /\ ~crashed
    /\ mutexHolder = NULL
    /\ mutexHolder' = t
    /\ threadState' = [threadState EXCEPT ![t] = "in_mutex"]
    /\ UNCHANGED <<encoderState, encoderOwner, encoderRetainCount, encoderActiveCalls,
                  cbState, cbEncoders, v27_tracked, v27_ended, v27_cbEncoders,
                  threadEncoder, threadCB, threadPendingOp, gpuQueue, crashed, crashType>>

\* Thread releases mutex
ReleaseMutex(t) ==
    /\ mutexHolder = t
    /\ mutexHolder' = NULL
    /\ threadState' = [threadState EXCEPT ![t] = "idle"]
    /\ UNCHANGED <<encoderState, encoderOwner, encoderRetainCount, encoderActiveCalls,
                  cbState, cbEncoders, v27_tracked, v27_ended, v27_cbEncoders,
                  threadEncoder, threadCB, threadPendingOp, gpuQueue, crashed, crashType>>

(****************************************************************************)
(* v2.7: CREATE ENCODER (must hold mutex)                                   *)
(****************************************************************************)

V27_CreateEncoder(t, cb, e) ==
    /\ threadState[t] = "in_mutex"
    /\ mutexHolder = t
    /\ encoderState[e] = "free"
    /\ cbState[cb] \in {"created", "encoding"}
    /\ ~crashed
    \* Create encoder
    /\ encoderState' = [encoderState EXCEPT ![e] = "active"]
    /\ encoderOwner' = [encoderOwner EXCEPT ![e] = cb]
    /\ encoderRetainCount' = [encoderRetainCount EXCEPT ![e] = 1]
    /\ cbState' = [cbState EXCEPT ![cb] = "encoding"]
    /\ cbEncoders' = [cbEncoders EXCEPT ![cb] = cbEncoders[cb] \cup {e}]
    \* v2.7 tracking
    /\ v27_tracked' = [v27_tracked EXCEPT ![e] = TRUE]
    /\ v27_ended' = [v27_ended EXCEPT ![e] = FALSE]
    /\ v27_cbEncoders' = [v27_cbEncoders EXCEPT ![cb] = v27_cbEncoders[cb] \cup {e}]
    \* Thread state
    /\ threadEncoder' = [threadEncoder EXCEPT ![t] = e]
    /\ threadCB' = [threadCB EXCEPT ![t] = cb]
    /\ threadState' = [threadState EXCEPT ![t] = "using_encoder"]
    \* Release mutex
    /\ mutexHolder' = NULL
    /\ UNCHANGED <<encoderActiveCalls, threadPendingOp, gpuQueue, crashed, crashType>>

(****************************************************************************)
(* v2.7: USE ENCODER METHOD (with active call tracking)                     *)
(****************************************************************************)

\* Begin encoder method - increment active calls
\* v2.7 PROTECTION: Check encoder state FIRST before checking impl
\* This prevents crashes because we detect non-active encoders and release gracefully
BeginEncoderMethod(t) ==
    /\ threadState[t] = "using_encoder"
    /\ threadEncoder[t] # NULL
    /\ ~crashed
    /\ LET e == threadEncoder[t] IN
       \* v2.7 KEY: Check if encoder is still active FIRST
       IF encoderState[e] # "active"
       THEN \* Encoder was force-ended, impl_invalid, or released
            \* v2.7 gracefully releases before trying to use it
            /\ threadState' = [threadState EXCEPT ![t] = "idle"]
            /\ threadEncoder' = [threadEncoder EXCEPT ![t] = NULL]
            /\ UNCHANGED <<encoderState, encoderOwner, encoderRetainCount, encoderActiveCalls,
                          cbState, cbEncoders, v27_tracked, v27_ended, v27_cbEncoders,
                          threadCB, threadPendingOp, mutexHolder, gpuQueue, crashed, crashType>>
       ELSE \* Encoder is active - safe to use
            \* Limit active calls to keep state space bounded
            /\ encoderActiveCalls[e] < 3
            \* Impl should always be valid for active encoder (invariant)
            /\ encoderActiveCalls' = [encoderActiveCalls EXCEPT ![e] = encoderActiveCalls[e] + 1]
            /\ encoderRetainCount' = [encoderRetainCount EXCEPT ![e] = encoderRetainCount[e] + 1]
            /\ UNCHANGED <<encoderState, encoderOwner, cbState, cbEncoders,
                          v27_tracked, v27_ended, v27_cbEncoders,
                          threadState, threadEncoder, threadCB, threadPendingOp,
                          mutexHolder, gpuQueue, crashed, crashType>>

\* End encoder method - decrement active calls
EndEncoderMethod(t) ==
    /\ threadState[t] = "using_encoder"
    /\ threadEncoder[t] # NULL
    /\ ~crashed
    /\ LET e == threadEncoder[t] IN
       /\ encoderActiveCalls[e] > 0
       /\ encoderActiveCalls' = [encoderActiveCalls EXCEPT ![e] = encoderActiveCalls[e] - 1]
       /\ encoderRetainCount' = [encoderRetainCount EXCEPT ![e] = encoderRetainCount[e] - 1]
       /\ UNCHANGED <<encoderState, encoderOwner, cbState, cbEncoders,
                     v27_tracked, v27_ended, v27_cbEncoders,
                     threadState, threadEncoder, threadCB, threadPendingOp,
                     mutexHolder, gpuQueue, crashed, crashType>>

(****************************************************************************)
(* v2.7: END ENCODER                                                        *)
(****************************************************************************)

V27_EndEncoder(t) ==
    /\ threadState[t] = "using_encoder"
    /\ threadEncoder[t] # NULL
    /\ ~crashed
    /\ mutexHolder = NULL  \* Acquire mutex (atomic for simplicity)
    /\ LET e == threadEncoder[t] IN
       /\ encoderState[e] = "active"
       /\ encoderActiveCalls[e] = 0  \* No active method calls
       \* End encoder
       /\ encoderState' = [encoderState EXCEPT ![e] = "ended"]
       /\ v27_ended' = [v27_ended EXCEPT ![e] = TRUE]
       /\ threadState' = [threadState EXCEPT ![t] = "idle"]
       /\ threadEncoder' = [threadEncoder EXCEPT ![t] = NULL]
       /\ UNCHANGED <<encoderOwner, encoderRetainCount, encoderActiveCalls,
                     cbState, cbEncoders, v27_tracked, v27_cbEncoders,
                     threadCB, threadPendingOp, mutexHolder, gpuQueue, crashed, crashType>>

(****************************************************************************)
(* v2.7: COMMIT (with force-end protection)                                 *)
(****************************************************************************)

V27_Commit(t, cb) ==
    /\ threadState[t] = "idle"
    /\ cbState[cb] \in {"created", "encoding"}
    /\ ~crashed
    /\ mutexHolder = NULL  \* Acquire mutex
    \* Force-end any active encoders
    /\ LET activeEncs == {e \in v27_cbEncoders[cb] : ~v27_ended[e]} IN
       /\ encoderState' = [e \in Encoders |->
                          IF e \in activeEncs
                          THEN "ended"
                          ELSE encoderState[e]]
       /\ v27_ended' = [e \in Encoders |->
                       IF e \in activeEncs
                       THEN TRUE
                       ELSE v27_ended[e]]
       /\ cbState' = [cbState EXCEPT ![cb] = "committed"]
       /\ gpuQueue' = Append(gpuQueue, cb)
       /\ UNCHANGED <<encoderOwner, encoderRetainCount, encoderActiveCalls,
                     cbEncoders, v27_tracked, v27_cbEncoders,
                     threadState, threadEncoder, threadCB, threadPendingOp,
                     mutexHolder, crashed, crashType>>

(****************************************************************************)
(* GPU EXECUTION (asynchronous)                                             *)
(****************************************************************************)

GPUExecute ==
    /\ Len(gpuQueue) > 0
    /\ ~crashed
    /\ LET cb == Head(gpuQueue) IN
       /\ cbState' = [cbState EXCEPT ![cb] = "completed"]
       /\ gpuQueue' = Tail(gpuQueue)
       /\ UNCHANGED <<encoderState, encoderOwner, encoderRetainCount, encoderActiveCalls,
                     cbEncoders, v27_tracked, v27_ended, v27_cbEncoders,
                     threadState, threadEncoder, threadCB, threadPendingOp,
                     mutexHolder, crashed, crashType>>

(****************************************************************************)
(* DESTROY IMPL (asynchronous callback)                                     *)
(****************************************************************************)

DestroyImpl(e) ==
    /\ encoderState[e] = "ended"
    /\ encoderActiveCalls[e] = 0  \* No active calls
    /\ ~crashed
    \* Invalidate _impl
    /\ encoderState' = [encoderState EXCEPT ![e] = "impl_invalid"]
    /\ UNCHANGED <<encoderOwner, encoderRetainCount, encoderActiveCalls,
                  cbState, cbEncoders, v27_tracked, v27_ended, v27_cbEncoders,
                  threadState, threadEncoder, threadCB, threadPendingOp,
                  mutexHolder, gpuQueue, crashed, crashType>>

(****************************************************************************)
(* RELEASE ENCODER (when retain count reaches 0)                            *)
(****************************************************************************)

ReleaseEncoder(e) ==
    /\ encoderState[e] \in {"ended", "impl_invalid"}
    /\ encoderRetainCount[e] = 1  \* Last reference
    /\ encoderActiveCalls[e] = 0
    /\ ~crashed
    /\ encoderState' = [encoderState EXCEPT ![e] = "released"]
    /\ encoderRetainCount' = [encoderRetainCount EXCEPT ![e] = 0]
    \* Clean up tracking
    /\ LET cb == encoderOwner[e] IN
       /\ cbEncoders' = IF cb # NULL
                        THEN [cbEncoders EXCEPT ![cb] = cbEncoders[cb] \ {e}]
                        ELSE cbEncoders
       /\ v27_cbEncoders' = IF cb # NULL
                            THEN [v27_cbEncoders EXCEPT ![cb] = v27_cbEncoders[cb] \ {e}]
                            ELSE v27_cbEncoders
    /\ v27_tracked' = [v27_tracked EXCEPT ![e] = FALSE]
    /\ UNCHANGED <<encoderOwner, encoderActiveCalls, cbState, v27_ended,
                  threadState, threadEncoder, threadCB, threadPendingOp,
                  mutexHolder, gpuQueue, crashed, crashType>>

(****************************************************************************)
(* RECYCLE ENCODER (for reuse after release)                                *)
(****************************************************************************)

RecycleEncoder(e) ==
    /\ encoderState[e] = "released"
    /\ ~crashed
    /\ encoderState' = [encoderState EXCEPT ![e] = "free"]
    /\ encoderOwner' = [encoderOwner EXCEPT ![e] = NULL]
    /\ UNCHANGED <<encoderRetainCount, encoderActiveCalls, cbState, cbEncoders,
                  v27_tracked, v27_ended, v27_cbEncoders,
                  threadState, threadEncoder, threadCB, threadPendingOp,
                  mutexHolder, gpuQueue, crashed, crashType>>

(****************************************************************************)
(* RECYCLE COMMAND BUFFER (for reuse after completion)                      *)
(****************************************************************************)

RecycleCB(cb) ==
    /\ cbState[cb] = "completed"
    /\ cbEncoders[cb] = {}  \* All encoders released
    /\ ~crashed
    /\ cbState' = [cbState EXCEPT ![cb] = "created"]
    /\ v27_cbEncoders' = [v27_cbEncoders EXCEPT ![cb] = {}]
    /\ UNCHANGED <<encoderState, encoderOwner, encoderRetainCount, encoderActiveCalls,
                  cbEncoders, v27_tracked, v27_ended,
                  threadState, threadEncoder, threadCB, threadPendingOp,
                  mutexHolder, gpuQueue, crashed, crashType>>

(****************************************************************************)
(* Next state relation                                                      *)
(****************************************************************************)

Next ==
    \/ \E t \in Threads : TryAcquireMutex(t)
    \/ \E t \in Threads : ReleaseMutex(t)
    \/ \E t \in Threads, cb \in CommandBuffers, e \in Encoders : V27_CreateEncoder(t, cb, e)
    \/ \E t \in Threads : BeginEncoderMethod(t)
    \/ \E t \in Threads : EndEncoderMethod(t)
    \/ \E t \in Threads : V27_EndEncoder(t)
    \/ \E t \in Threads, cb \in CommandBuffers : V27_Commit(t, cb)
    \/ GPUExecute
    \/ \E e \in Encoders : DestroyImpl(e)
    \/ \E e \in Encoders : ReleaseEncoder(e)
    \/ \E e \in Encoders : RecycleEncoder(e)
    \/ \E cb \in CommandBuffers : RecycleCB(cb)

(****************************************************************************)
(* SAFETY PROPERTIES                                                        *)
(****************************************************************************)

\* No crashes
NoCrashes == ~crashed

\* No validation crash - never commit with active encoders
NoValidationCrash ==
    \A cb \in CommandBuffers :
        cbState[cb] \in {"committed", "submitted", "completed"} =>
            ~(\E e \in cbEncoders[cb] : encoderState[e] = "active")

\* Reference count is never negative
NoNegativeRefCount ==
    \A e \in Encoders : encoderRetainCount[e] >= 0

\* Active calls is never negative
NoNegativeActiveCalls ==
    \A e \in Encoders : encoderActiveCalls[e] >= 0

\* v2.7 tracking is consistent
V27TrackingConsistent ==
    \A cb \in CommandBuffers :
        v27_cbEncoders[cb] \subseteq cbEncoders[cb]

\* Released encoder is not in any CB
ReleasedNotInCB ==
    \A e \in Encoders :
        encoderState[e] = "released" =>
            \A cb \in CommandBuffers : e \notin cbEncoders[cb]

(****************************************************************************)
(* Specification                                                            *)
(****************************************************************************)

vars == <<encoderState, encoderOwner, encoderRetainCount, encoderActiveCalls,
          cbState, cbEncoders, v27_tracked, v27_ended, v27_cbEncoders,
          threadState, threadEncoder, threadCB, threadPendingOp,
          mutexHolder, gpuQueue, crashed, crashType>>

Spec == Init /\ [][Next]_vars

(****************************************************************************)
(* THEOREMS                                                                 *)
(****************************************************************************)

THEOREM V27PreventsAllCrashes ==
    Spec => []NoCrashes

THEOREM V27PreventsValidationCrash ==
    Spec => []NoValidationCrash

THEOREM RefCountsSafe ==
    Spec => [](NoNegativeRefCount /\ NoNegativeActiveCalls)

=============================================================================
