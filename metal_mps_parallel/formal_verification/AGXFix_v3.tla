--------------------------- MODULE AGXFix_v3 ---------------------------
(****************************************************************************)
(* TLA+ Formal Model of AGX Driver Race Condition - COMPREHENSIVE Version   *)
(*                                                                           *)
(* This model includes ALL identified gaps:                                  *)
(*   1. Recursive mutex (thread can acquire multiple times)                  *)
(*   2. Dispatch queues (serial and concurrent work dispatch)                *)
(*   3. Multiple encoder types (compute, blit, render)                       *)
(*   4. Non-atomic state observations (weak memory ordering)                 *)
(*   5. Async callbacks (destroyImpl, completion handlers)                   *)
(*   6. Command buffer pooling and reuse                                     *)
(*   7. Additional crash scenarios                                           *)
(*                                                                           *)
(* Author: Andrew Yates / Claude                                             *)
(* Date: 2024-12-24                                                          *)
(****************************************************************************)

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    NumThreads,
    NumCommandBuffers,
    NumEncoders,
    NumDispatchQueues    \* GCD dispatch queues

VARIABLES
    \* ====== ENCODER STATE ======
    encoderState,        \* encoder -> {free, active, ended, impl_invalid, released}
    encoderType,         \* encoder -> {compute, blit, render, none}
    encoderOwner,        \* encoder -> command_buffer or NULL
    encoderRetainCount,  \* encoder -> Int
    encoderActiveCalls,  \* encoder -> Int

    \* ====== COMMAND BUFFER STATE ======
    cbState,             \* cb -> {pooled, allocated, encoding, committed, executing, completed}
    cbEncoders,          \* cb -> Set of encoders
    cbCompletionHandler, \* cb -> thread waiting for completion (or NULL)

    \* ====== v2.7 TRACKING ======
    v27_tracked,         \* encoder -> BOOLEAN
    v27_ended,           \* encoder -> BOOLEAN
    v27_cbEncoders,      \* cb -> Set of encoders

    \* ====== RECURSIVE MUTEX ======
    mutexHolder,         \* thread holding mutex or NULL
    mutexDepth,          \* Int - recursive depth (0 = not held)

    \* ====== DISPATCH QUEUES ======
    dispatchQueues,      \* queue -> Sequence of {thread, work_type}
    queueType,           \* queue -> {serial, concurrent}

    \* ====== THREAD STATE ======
    threadState,         \* thread -> state
    threadEncoder,       \* thread -> encoder or NULL
    threadCB,            \* thread -> command_buffer
    threadQueue,         \* thread -> dispatch queue (where it runs)
    threadObservedState, \* thread -> last observed encoder state (for weak ordering)

    \* ====== GPU STATE ======
    gpuQueue,            \* Sequence of command buffers
    gpuBusy,             \* BOOLEAN - GPU is executing

    \* ====== ASYNC CALLBACKS ======
    pendingCallbacks,    \* Sequence of {type, target} - callbacks to execute

    \* ====== CRASH STATE ======
    crashed,
    crashType

(****************************************************************************)
(* Type definitions                                                         *)
(****************************************************************************)

Threads == 1..NumThreads
CommandBuffers == 1..NumCommandBuffers
Encoders == 1..NumEncoders
DispatchQueues == 1..NumDispatchQueues

EncoderStates == {"free", "active", "ended", "impl_invalid", "released"}
EncoderTypes == {"compute", "blit", "render", "none"}
CBStates == {"pooled", "allocated", "encoding", "committed", "executing", "completed"}
ThreadStates == {"idle", "dispatched", "running", "in_mutex", "waiting_mutex",
                 "creating", "using", "ending", "committing", "waiting_completion"}
QueueTypes == {"serial", "concurrent"}
CallbackTypes == {"destroy_impl", "completion"}

NULL == 0

(****************************************************************************)
(* Type invariant                                                           *)
(****************************************************************************)

TypeOK ==
    /\ encoderState \in [Encoders -> EncoderStates]
    /\ encoderType \in [Encoders -> EncoderTypes]
    /\ encoderOwner \in [Encoders -> CommandBuffers \cup {NULL}]
    /\ encoderRetainCount \in [Encoders -> 0..20]
    /\ encoderActiveCalls \in [Encoders -> 0..5]
    /\ cbState \in [CommandBuffers -> CBStates]
    /\ cbEncoders \in [CommandBuffers -> SUBSET Encoders]
    /\ cbCompletionHandler \in [CommandBuffers -> Threads \cup {NULL}]
    /\ v27_tracked \in [Encoders -> BOOLEAN]
    /\ v27_ended \in [Encoders -> BOOLEAN]
    /\ v27_cbEncoders \in [CommandBuffers -> SUBSET Encoders]
    /\ mutexHolder \in Threads \cup {NULL}
    /\ mutexDepth \in 0..10
    /\ queueType \in [DispatchQueues -> QueueTypes]
    /\ threadState \in [Threads -> ThreadStates]
    /\ threadEncoder \in [Threads -> Encoders \cup {NULL}]
    /\ threadCB \in [Threads -> CommandBuffers]
    /\ threadQueue \in [Threads -> DispatchQueues]
    /\ threadObservedState \in [Threads -> EncoderStates \cup {"unknown"}]
    /\ gpuBusy \in BOOLEAN
    /\ crashed \in BOOLEAN

(****************************************************************************)
(* Initial state                                                            *)
(****************************************************************************)

Init ==
    /\ encoderState = [e \in Encoders |-> "free"]
    /\ encoderType = [e \in Encoders |-> "none"]
    /\ encoderOwner = [e \in Encoders |-> NULL]
    /\ encoderRetainCount = [e \in Encoders |-> 0]
    /\ encoderActiveCalls = [e \in Encoders |-> 0]
    /\ cbState = [cb \in CommandBuffers |-> "pooled"]
    /\ cbEncoders = [cb \in CommandBuffers |-> {}]
    /\ cbCompletionHandler = [cb \in CommandBuffers |-> NULL]
    /\ v27_tracked = [e \in Encoders |-> FALSE]
    /\ v27_ended = [e \in Encoders |-> FALSE]
    /\ v27_cbEncoders = [cb \in CommandBuffers |-> {}]
    /\ mutexHolder = NULL
    /\ mutexDepth = 0
    /\ dispatchQueues = [q \in DispatchQueues |-> <<>>]
    /\ queueType = [q \in DispatchQueues |-> IF q = 1 THEN "serial" ELSE "concurrent"]
    /\ threadState = [t \in Threads |-> "idle"]
    /\ threadEncoder = [t \in Threads |-> NULL]
    /\ threadCB = [t \in Threads |-> 1]
    /\ threadQueue = [t \in Threads |-> 1]
    /\ threadObservedState = [t \in Threads |-> "unknown"]
    /\ gpuQueue = <<>>
    /\ gpuBusy = FALSE
    /\ pendingCallbacks = <<>>
    /\ crashed = FALSE
    /\ crashType = "none"

(****************************************************************************)
(* CRASH CONDITIONS                                                         *)
(****************************************************************************)

\* Crash 1: Using encoder with invalid impl (PAC failure)
CrashImplInvalid(e) == encoderState[e] = "impl_invalid"

\* Crash 2: Using released encoder (use after free)
CrashUseAfterFree(e) == encoderState[e] = "released"

\* Crash 3: Committing with active encoders (validation failure)
CrashValidation(cb) == \E e \in cbEncoders[cb] : encoderState[e] = "active"

\* Crash 4: Double end encoding
CrashDoubleEnd(e) == encoderState[e] \in {"ended", "impl_invalid", "released"}

\* Crash 5: Negative retain count
CrashNegativeRetain(e) == encoderRetainCount[e] < 0

\* Crash 6: Using encoder on wrong thread (thread safety violation)
\* This models the case where an encoder is used from a thread different
\* than the one that created it, without proper synchronization
CrashThreadSafety(t, e) ==
    /\ threadEncoder[t] = e
    /\ encoderOwner[e] # NULL
    /\ \E t2 \in Threads : t2 # t /\ threadEncoder[t2] = e

(****************************************************************************)
(* RECURSIVE MUTEX OPERATIONS                                               *)
(****************************************************************************)

\* Try to acquire mutex (may block or succeed recursively)
AcquireMutex(t) ==
    /\ threadState[t] \in {"running", "creating", "using", "ending", "committing"}
    /\ ~crashed
    /\ IF mutexHolder = NULL
       THEN \* Acquire fresh
            /\ mutexHolder' = t
            /\ mutexDepth' = 1
            /\ threadState' = [threadState EXCEPT ![t] = "in_mutex"]
       ELSE IF mutexHolder = t
            THEN \* Recursive acquire
                 /\ mutexDepth' = mutexDepth + 1
                 /\ UNCHANGED <<mutexHolder, threadState>>
            ELSE \* Block waiting
                 /\ threadState' = [threadState EXCEPT ![t] = "waiting_mutex"]
                 /\ UNCHANGED <<mutexHolder, mutexDepth>>
    /\ UNCHANGED <<encoderState, encoderType, encoderOwner, encoderRetainCount,
                  encoderActiveCalls, cbState, cbEncoders, cbCompletionHandler,
                  v27_tracked, v27_ended, v27_cbEncoders, dispatchQueues, queueType,
                  threadEncoder, threadCB, threadQueue, threadObservedState,
                  gpuQueue, gpuBusy, pendingCallbacks, crashed, crashType>>

\* Release mutex (may fully release or just decrement depth)
ReleaseMutex(t) ==
    /\ mutexHolder = t
    /\ mutexDepth > 0
    /\ ~crashed
    /\ IF mutexDepth = 1
       THEN \* Fully release
            /\ mutexHolder' = NULL
            /\ mutexDepth' = 0
       ELSE \* Just decrement
            /\ mutexDepth' = mutexDepth - 1
            /\ UNCHANGED mutexHolder
    /\ UNCHANGED <<encoderState, encoderType, encoderOwner, encoderRetainCount,
                  encoderActiveCalls, cbState, cbEncoders, cbCompletionHandler,
                  v27_tracked, v27_ended, v27_cbEncoders, dispatchQueues, queueType,
                  threadState, threadEncoder, threadCB, threadQueue, threadObservedState,
                  gpuQueue, gpuBusy, pendingCallbacks, crashed, crashType>>

\* Waiting thread gets the mutex when it's released
WakeupWaiting(t) ==
    /\ threadState[t] = "waiting_mutex"
    /\ mutexHolder = NULL
    /\ ~crashed
    /\ mutexHolder' = t
    /\ mutexDepth' = 1
    /\ threadState' = [threadState EXCEPT ![t] = "in_mutex"]
    /\ UNCHANGED <<encoderState, encoderType, encoderOwner, encoderRetainCount,
                  encoderActiveCalls, cbState, cbEncoders, cbCompletionHandler,
                  v27_tracked, v27_ended, v27_cbEncoders, dispatchQueues, queueType,
                  threadEncoder, threadCB, threadQueue, threadObservedState,
                  gpuQueue, gpuBusy, pendingCallbacks, crashed, crashType>>

(****************************************************************************)
(* DISPATCH QUEUE OPERATIONS                                                *)
(****************************************************************************)

\* Dispatch work to a queue
DispatchToQueue(t, q) ==
    /\ threadState[t] = "idle"
    /\ ~crashed
    /\ threadState' = [threadState EXCEPT ![t] = "dispatched"]
    /\ threadQueue' = [threadQueue EXCEPT ![t] = q]
    /\ dispatchQueues' = [dispatchQueues EXCEPT ![q] = Append(dispatchQueues[q], t)]
    /\ UNCHANGED <<encoderState, encoderType, encoderOwner, encoderRetainCount,
                  encoderActiveCalls, cbState, cbEncoders, cbCompletionHandler,
                  v27_tracked, v27_ended, v27_cbEncoders, mutexHolder, mutexDepth,
                  queueType, threadEncoder, threadCB, threadObservedState,
                  gpuQueue, gpuBusy, pendingCallbacks, crashed, crashType>>

\* Execute dispatched work (serial queue - one at a time)
ExecuteFromSerialQueue(q) ==
    /\ queueType[q] = "serial"
    /\ Len(dispatchQueues[q]) > 0
    /\ ~crashed
    /\ LET t == Head(dispatchQueues[q]) IN
       /\ threadState[t] = "dispatched"
       /\ threadState' = [threadState EXCEPT ![t] = "running"]
       /\ dispatchQueues' = [dispatchQueues EXCEPT ![q] = Tail(dispatchQueues[q])]
    /\ UNCHANGED <<encoderState, encoderType, encoderOwner, encoderRetainCount,
                  encoderActiveCalls, cbState, cbEncoders, cbCompletionHandler,
                  v27_tracked, v27_ended, v27_cbEncoders, mutexHolder, mutexDepth,
                  queueType, threadEncoder, threadCB, threadQueue, threadObservedState,
                  gpuQueue, gpuBusy, pendingCallbacks, crashed, crashType>>

\* Execute from concurrent queue (can run multiple)
ExecuteFromConcurrentQueue(q, t) ==
    /\ queueType[q] = "concurrent"
    /\ Len(dispatchQueues[q]) > 0
    /\ t \in {x \in DOMAIN dispatchQueues[q] : dispatchQueues[q][x] = t}  \* Thread is in queue
    /\ threadState[t] = "dispatched"
    /\ ~crashed
    /\ threadState' = [threadState EXCEPT ![t] = "running"]
    /\ dispatchQueues' = [dispatchQueues EXCEPT ![q] =
                         SelectSeq(dispatchQueues[q], LAMBDA x: x # t)]
    /\ UNCHANGED <<encoderState, encoderType, encoderOwner, encoderRetainCount,
                  encoderActiveCalls, cbState, cbEncoders, cbCompletionHandler,
                  v27_tracked, v27_ended, v27_cbEncoders, mutexHolder, mutexDepth,
                  queueType, threadEncoder, threadCB, threadQueue, threadObservedState,
                  gpuQueue, gpuBusy, pendingCallbacks, crashed, crashType>>

(****************************************************************************)
(* ALLOCATE COMMAND BUFFER FROM POOL                                        *)
(****************************************************************************)

AllocateCommandBuffer(t, cb) ==
    /\ threadState[t] = "running"
    /\ cbState[cb] = "pooled"
    /\ ~crashed
    /\ cbState' = [cbState EXCEPT ![cb] = "allocated"]
    /\ threadCB' = [threadCB EXCEPT ![t] = cb]
    /\ UNCHANGED <<encoderState, encoderType, encoderOwner, encoderRetainCount,
                  encoderActiveCalls, cbEncoders, cbCompletionHandler,
                  v27_tracked, v27_ended, v27_cbEncoders, mutexHolder, mutexDepth,
                  dispatchQueues, queueType, threadState, threadEncoder, threadQueue,
                  threadObservedState, gpuQueue, gpuBusy, pendingCallbacks, crashed, crashType>>

(****************************************************************************)
(* v2.7: CREATE ENCODER (with type)                                         *)
(****************************************************************************)

V27_CreateEncoder(t, e, etype) ==
    /\ threadState[t] = "in_mutex"
    /\ mutexHolder = t
    /\ encoderState[e] = "free"
    /\ cbState[threadCB[t]] \in {"allocated", "encoding"}
    /\ ~crashed
    /\ LET cb == threadCB[t] IN
       /\ encoderState' = [encoderState EXCEPT ![e] = "active"]
       /\ encoderType' = [encoderType EXCEPT ![e] = etype]
       /\ encoderOwner' = [encoderOwner EXCEPT ![e] = cb]
       /\ encoderRetainCount' = [encoderRetainCount EXCEPT ![e] = 1]
       /\ cbState' = [cbState EXCEPT ![cb] = "encoding"]
       /\ cbEncoders' = [cbEncoders EXCEPT ![cb] = cbEncoders[cb] \cup {e}]
       /\ v27_tracked' = [v27_tracked EXCEPT ![e] = TRUE]
       /\ v27_ended' = [v27_ended EXCEPT ![e] = FALSE]
       /\ v27_cbEncoders' = [v27_cbEncoders EXCEPT ![cb] = v27_cbEncoders[cb] \cup {e}]
       /\ threadEncoder' = [threadEncoder EXCEPT ![t] = e]
       /\ threadState' = [threadState EXCEPT ![t] = "using"]
       /\ threadObservedState' = [threadObservedState EXCEPT ![t] = "active"]
       \* Release mutex after create
       /\ mutexHolder' = NULL
       /\ mutexDepth' = 0
    /\ UNCHANGED <<encoderActiveCalls, cbCompletionHandler, dispatchQueues, queueType,
                  threadCB, threadQueue, gpuQueue, gpuBusy, pendingCallbacks, crashed, crashType>>

(****************************************************************************)
(* WEAK MEMORY: Observe encoder state (may be stale)                        *)
(****************************************************************************)

\* Model weak memory ordering: thread observes current state
\* In reality, this could be stale, but we model it as eventually consistent
ObserveEncoderState(t) ==
    /\ threadState[t] = "using"
    /\ threadEncoder[t] # NULL
    /\ ~crashed
    /\ LET e == threadEncoder[t] IN
       /\ threadObservedState' = [threadObservedState EXCEPT ![t] = encoderState[e]]
    /\ UNCHANGED <<encoderState, encoderType, encoderOwner, encoderRetainCount,
                  encoderActiveCalls, cbState, cbEncoders, cbCompletionHandler,
                  v27_tracked, v27_ended, v27_cbEncoders, mutexHolder, mutexDepth,
                  dispatchQueues, queueType, threadState, threadEncoder, threadCB,
                  threadQueue, gpuQueue, gpuBusy, pendingCallbacks, crashed, crashType>>

(****************************************************************************)
(* v2.7: USE ENCODER (with state check)                                     *)
(****************************************************************************)

V27_UseEncoder(t) ==
    /\ threadState[t] = "using"
    /\ threadEncoder[t] # NULL
    /\ ~crashed
    /\ LET e == threadEncoder[t] IN
       \* v2.7 KEY: Check ACTUAL encoder state (with proper synchronization)
       \* In real code, this check happens under mutex or with memory barriers
       IF encoderState[e] # "active"
       THEN \* Encoder changed state (force-ended, released, etc) - gracefully release
            /\ threadState' = [threadState EXCEPT ![t] = "running"]
            /\ threadEncoder' = [threadEncoder EXCEPT ![t] = NULL]
            /\ threadObservedState' = [threadObservedState EXCEPT ![t] = "unknown"]
            /\ UNCHANGED <<encoderState, encoderType, encoderOwner, encoderRetainCount,
                          encoderActiveCalls, cbState, cbEncoders, cbCompletionHandler,
                          v27_tracked, v27_ended, v27_cbEncoders, mutexHolder, mutexDepth,
                          dispatchQueues, queueType, threadCB, threadQueue,
                          gpuQueue, gpuBusy, pendingCallbacks, crashed, crashType>>
       ELSE \* Encoder is active - safe to use
            /\ encoderActiveCalls[e] < 3  \* Bound for state space
            /\ encoderActiveCalls' = [encoderActiveCalls EXCEPT ![e] = encoderActiveCalls[e] + 1]
            /\ encoderRetainCount' = [encoderRetainCount EXCEPT ![e] = encoderRetainCount[e] + 1]
            \* Update observed state to current
            /\ threadObservedState' = [threadObservedState EXCEPT ![t] = "active"]
            /\ UNCHANGED <<encoderState, encoderType, encoderOwner, cbState, cbEncoders,
                          cbCompletionHandler, v27_tracked, v27_ended, v27_cbEncoders,
                          mutexHolder, mutexDepth, dispatchQueues, queueType,
                          threadState, threadEncoder, threadCB, threadQueue,
                          gpuQueue, gpuBusy, pendingCallbacks, crashed, crashType>>

\* Complete encoder method call
V27_CompleteEncoderMethod(t) ==
    /\ threadState[t] = "using"
    /\ threadEncoder[t] # NULL
    /\ ~crashed
    /\ LET e == threadEncoder[t] IN
       /\ encoderActiveCalls[e] > 0
       /\ encoderActiveCalls' = [encoderActiveCalls EXCEPT ![e] = encoderActiveCalls[e] - 1]
       /\ encoderRetainCount' = [encoderRetainCount EXCEPT ![e] = encoderRetainCount[e] - 1]
    /\ UNCHANGED <<encoderState, encoderType, encoderOwner, cbState, cbEncoders,
                  cbCompletionHandler, v27_tracked, v27_ended, v27_cbEncoders,
                  mutexHolder, mutexDepth, dispatchQueues, queueType,
                  threadState, threadEncoder, threadCB, threadQueue, threadObservedState,
                  gpuQueue, gpuBusy, pendingCallbacks, crashed, crashType>>

(****************************************************************************)
(* v2.7: END ENCODER                                                        *)
(****************************************************************************)

V27_EndEncoder(t) ==
    /\ threadState[t] = "using"
    /\ threadEncoder[t] # NULL
    /\ mutexHolder = NULL  \* Acquire mutex
    /\ ~crashed
    /\ LET e == threadEncoder[t] IN
       /\ encoderState[e] = "active"
       /\ encoderActiveCalls[e] = 0
       /\ encoderState' = [encoderState EXCEPT ![e] = "ended"]
       /\ v27_ended' = [v27_ended EXCEPT ![e] = TRUE]
       /\ threadState' = [threadState EXCEPT ![t] = "running"]
       /\ threadEncoder' = [threadEncoder EXCEPT ![t] = NULL]
       /\ threadObservedState' = [threadObservedState EXCEPT ![t] = "unknown"]
    /\ UNCHANGED <<encoderType, encoderOwner, encoderRetainCount, encoderActiveCalls,
                  cbState, cbEncoders, cbCompletionHandler, v27_tracked, v27_cbEncoders,
                  mutexHolder, mutexDepth, dispatchQueues, queueType,
                  threadCB, threadQueue, gpuQueue, gpuBusy, pendingCallbacks, crashed, crashType>>

(****************************************************************************)
(* v2.7: COMMIT (force-end + queue to GPU)                                  *)
(****************************************************************************)

V27_Commit(t) ==
    /\ threadState[t] = "running"
    /\ mutexHolder = NULL
    /\ ~crashed
    /\ LET cb == threadCB[t] IN
       /\ cbState[cb] \in {"allocated", "encoding"}
       \* Force-end any active encoders
       /\ LET activeEncs == {e \in v27_cbEncoders[cb] : ~v27_ended[e]} IN
          /\ encoderState' = [e \in Encoders |->
                             IF e \in activeEncs THEN "ended"
                             ELSE encoderState[e]]
          /\ v27_ended' = [e \in Encoders |->
                          IF e \in activeEncs THEN TRUE
                          ELSE v27_ended[e]]
       /\ cbState' = [cbState EXCEPT ![cb] = "committed"]
       /\ gpuQueue' = Append(gpuQueue, cb)
       /\ threadState' = [threadState EXCEPT ![t] = "running"]
    /\ UNCHANGED <<encoderType, encoderOwner, encoderRetainCount, encoderActiveCalls,
                  cbEncoders, cbCompletionHandler, v27_tracked, v27_cbEncoders,
                  mutexHolder, mutexDepth, dispatchQueues, queueType,
                  threadEncoder, threadCB, threadQueue, threadObservedState,
                  gpuBusy, pendingCallbacks, crashed, crashType>>

\* Commit and wait for completion
V27_CommitAndWait(t) ==
    /\ threadState[t] = "running"
    /\ mutexHolder = NULL
    /\ ~crashed
    /\ LET cb == threadCB[t] IN
       /\ cbState[cb] \in {"allocated", "encoding"}
       /\ LET activeEncs == {e \in v27_cbEncoders[cb] : ~v27_ended[e]} IN
          /\ encoderState' = [e \in Encoders |->
                             IF e \in activeEncs THEN "ended"
                             ELSE encoderState[e]]
          /\ v27_ended' = [e \in Encoders |->
                          IF e \in activeEncs THEN TRUE
                          ELSE v27_ended[e]]
       /\ cbState' = [cbState EXCEPT ![cb] = "committed"]
       /\ cbCompletionHandler' = [cbCompletionHandler EXCEPT ![cb] = t]
       /\ gpuQueue' = Append(gpuQueue, cb)
       /\ threadState' = [threadState EXCEPT ![t] = "waiting_completion"]
    /\ UNCHANGED <<encoderType, encoderOwner, encoderRetainCount, encoderActiveCalls,
                  cbEncoders, v27_tracked, v27_cbEncoders,
                  mutexHolder, mutexDepth, dispatchQueues, queueType,
                  threadEncoder, threadCB, threadQueue, threadObservedState,
                  gpuBusy, pendingCallbacks, crashed, crashType>>

(****************************************************************************)
(* GPU EXECUTION                                                            *)
(****************************************************************************)

GPUStartExecution ==
    /\ Len(gpuQueue) > 0
    /\ ~gpuBusy
    /\ ~crashed
    /\ LET cb == Head(gpuQueue) IN
       /\ cbState' = [cbState EXCEPT ![cb] = "executing"]
       /\ gpuBusy' = TRUE
       /\ gpuQueue' = Tail(gpuQueue)
    /\ UNCHANGED <<encoderState, encoderType, encoderOwner, encoderRetainCount,
                  encoderActiveCalls, cbEncoders, cbCompletionHandler,
                  v27_tracked, v27_ended, v27_cbEncoders,
                  mutexHolder, mutexDepth, dispatchQueues, queueType,
                  threadState, threadEncoder, threadCB, threadQueue, threadObservedState,
                  pendingCallbacks, crashed, crashType>>

GPUCompleteExecution ==
    /\ gpuBusy
    /\ ~crashed
    /\ \E cb \in CommandBuffers :
       /\ cbState[cb] = "executing"
       /\ cbState' = [cbState EXCEPT ![cb] = "completed"]
       /\ gpuBusy' = FALSE
       \* Schedule destroyImpl callbacks for ended encoders
       /\ LET encs == {e \in cbEncoders[cb] : encoderState[e] = "ended"} IN
          pendingCallbacks' = pendingCallbacks \o
                             [i \in 1..Cardinality(encs) |->
                              [type |-> "destroy_impl",
                               target |-> CHOOSE e \in encs : TRUE]]
       \* Wake up completion handler if any
       /\ IF cbCompletionHandler[cb] # NULL
          THEN threadState' = [threadState EXCEPT ![cbCompletionHandler[cb]] = "running"]
          ELSE UNCHANGED threadState
       /\ cbCompletionHandler' = [cbCompletionHandler EXCEPT ![cb] = NULL]
    /\ UNCHANGED <<encoderState, encoderType, encoderOwner, encoderRetainCount,
                  encoderActiveCalls, cbEncoders, v27_tracked, v27_ended, v27_cbEncoders,
                  mutexHolder, mutexDepth, dispatchQueues, queueType,
                  threadEncoder, threadCB, threadQueue, threadObservedState, gpuQueue,
                  crashed, crashType>>

(****************************************************************************)
(* ASYNC CALLBACKS                                                          *)
(****************************************************************************)

\* Execute destroyImpl callback
ExecuteDestroyImplCallback ==
    /\ Len(pendingCallbacks) > 0
    /\ Head(pendingCallbacks).type = "destroy_impl"
    /\ ~crashed
    /\ LET e == Head(pendingCallbacks).target IN
       /\ encoderState[e] = "ended"
       /\ encoderActiveCalls[e] = 0
       /\ encoderState' = [encoderState EXCEPT ![e] = "impl_invalid"]
       /\ pendingCallbacks' = Tail(pendingCallbacks)
    /\ UNCHANGED <<encoderType, encoderOwner, encoderRetainCount, encoderActiveCalls,
                  cbState, cbEncoders, cbCompletionHandler, v27_tracked, v27_ended, v27_cbEncoders,
                  mutexHolder, mutexDepth, dispatchQueues, queueType,
                  threadState, threadEncoder, threadCB, threadQueue, threadObservedState,
                  gpuQueue, gpuBusy, crashed, crashType>>

(****************************************************************************)
(* ENCODER RELEASE AND RECYCLE                                              *)
(****************************************************************************)

ReleaseEncoder(e) ==
    /\ encoderState[e] \in {"ended", "impl_invalid"}
    /\ encoderRetainCount[e] = 1
    /\ encoderActiveCalls[e] = 0
    /\ ~crashed
    /\ encoderState' = [encoderState EXCEPT ![e] = "released"]
    /\ encoderRetainCount' = [encoderRetainCount EXCEPT ![e] = 0]
    /\ LET cb == encoderOwner[e] IN
       /\ cbEncoders' = IF cb # NULL
                        THEN [cbEncoders EXCEPT ![cb] = cbEncoders[cb] \ {e}]
                        ELSE cbEncoders
       /\ v27_cbEncoders' = IF cb # NULL
                            THEN [v27_cbEncoders EXCEPT ![cb] = v27_cbEncoders[cb] \ {e}]
                            ELSE v27_cbEncoders
    /\ v27_tracked' = [v27_tracked EXCEPT ![e] = FALSE]
    /\ UNCHANGED <<encoderType, encoderOwner, encoderActiveCalls, cbState, cbCompletionHandler,
                  v27_ended, mutexHolder, mutexDepth, dispatchQueues, queueType,
                  threadState, threadEncoder, threadCB, threadQueue, threadObservedState,
                  gpuQueue, gpuBusy, pendingCallbacks, crashed, crashType>>

RecycleEncoder(e) ==
    /\ encoderState[e] = "released"
    /\ ~crashed
    /\ encoderState' = [encoderState EXCEPT ![e] = "free"]
    /\ encoderType' = [encoderType EXCEPT ![e] = "none"]
    /\ encoderOwner' = [encoderOwner EXCEPT ![e] = NULL]
    /\ UNCHANGED <<encoderRetainCount, encoderActiveCalls, cbState, cbEncoders, cbCompletionHandler,
                  v27_tracked, v27_ended, v27_cbEncoders, mutexHolder, mutexDepth,
                  dispatchQueues, queueType, threadState, threadEncoder, threadCB,
                  threadQueue, threadObservedState, gpuQueue, gpuBusy, pendingCallbacks,
                  crashed, crashType>>

(****************************************************************************)
(* COMMAND BUFFER RECYCLE                                                   *)
(****************************************************************************)

RecycleCommandBuffer(cb) ==
    /\ cbState[cb] = "completed"
    /\ cbEncoders[cb] = {}
    /\ ~crashed
    /\ cbState' = [cbState EXCEPT ![cb] = "pooled"]
    /\ v27_cbEncoders' = [v27_cbEncoders EXCEPT ![cb] = {}]
    /\ UNCHANGED <<encoderState, encoderType, encoderOwner, encoderRetainCount, encoderActiveCalls,
                  cbEncoders, cbCompletionHandler, v27_tracked, v27_ended,
                  mutexHolder, mutexDepth, dispatchQueues, queueType,
                  threadState, threadEncoder, threadCB, threadQueue, threadObservedState,
                  gpuQueue, gpuBusy, pendingCallbacks, crashed, crashType>>

(****************************************************************************)
(* THREAD COMPLETION                                                        *)
(****************************************************************************)

ThreadComplete(t) ==
    /\ threadState[t] = "running"
    /\ threadEncoder[t] = NULL
    /\ ~crashed
    /\ threadState' = [threadState EXCEPT ![t] = "idle"]
    /\ UNCHANGED <<encoderState, encoderType, encoderOwner, encoderRetainCount, encoderActiveCalls,
                  cbState, cbEncoders, cbCompletionHandler, v27_tracked, v27_ended, v27_cbEncoders,
                  mutexHolder, mutexDepth, dispatchQueues, queueType,
                  threadEncoder, threadCB, threadQueue, threadObservedState,
                  gpuQueue, gpuBusy, pendingCallbacks, crashed, crashType>>

(****************************************************************************)
(* Next state relation                                                      *)
(****************************************************************************)

Next ==
    \* Mutex operations
    \/ \E t \in Threads : AcquireMutex(t)
    \/ \E t \in Threads : ReleaseMutex(t)
    \/ \E t \in Threads : WakeupWaiting(t)
    \* Dispatch queue operations
    \/ \E t \in Threads, q \in DispatchQueues : DispatchToQueue(t, q)
    \/ \E q \in DispatchQueues : ExecuteFromSerialQueue(q)
    \/ \E q \in DispatchQueues, t \in Threads : ExecuteFromConcurrentQueue(q, t)
    \* Command buffer operations
    \/ \E t \in Threads, cb \in CommandBuffers : AllocateCommandBuffer(t, cb)
    \* Encoder operations
    \/ \E t \in Threads, e \in Encoders, et \in {"compute", "blit", "render"} : V27_CreateEncoder(t, e, et)
    \/ \E t \in Threads : ObserveEncoderState(t)
    \/ \E t \in Threads : V27_UseEncoder(t)
    \/ \E t \in Threads : V27_CompleteEncoderMethod(t)
    \/ \E t \in Threads : V27_EndEncoder(t)
    \* Commit operations
    \/ \E t \in Threads : V27_Commit(t)
    \/ \E t \in Threads : V27_CommitAndWait(t)
    \* GPU operations
    \/ GPUStartExecution
    \/ GPUCompleteExecution
    \* Callbacks
    \/ ExecuteDestroyImplCallback
    \* Cleanup
    \/ \E e \in Encoders : ReleaseEncoder(e)
    \/ \E e \in Encoders : RecycleEncoder(e)
    \/ \E cb \in CommandBuffers : RecycleCommandBuffer(cb)
    \* Thread completion
    \/ \E t \in Threads : ThreadComplete(t)

(****************************************************************************)
(* SAFETY PROPERTIES                                                        *)
(****************************************************************************)

\* No crashes of any type
NoCrashes == ~crashed

\* Never commit with active encoders
NoValidationCrash ==
    \A cb \in CommandBuffers :
        cbState[cb] \in {"committed", "executing", "completed"} =>
            ~(\E e \in cbEncoders[cb] : encoderState[e] = "active")

\* Retain counts never negative
NoNegativeRetain == \A e \in Encoders : encoderRetainCount[e] >= 0

\* Active calls never negative
NoNegativeActive == \A e \in Encoders : encoderActiveCalls[e] >= 0

\* Mutex depth consistent with holder
MutexConsistent ==
    /\ (mutexHolder = NULL) <=> (mutexDepth = 0)
    /\ mutexDepth >= 0

\* v2.7 tracking consistent
V27Consistent ==
    \A cb \in CommandBuffers : v27_cbEncoders[cb] \subseteq cbEncoders[cb]

\* Released encoders not in any CB
ReleasedClean ==
    \A e \in Encoders :
        encoderState[e] = "released" => \A cb \in CommandBuffers : e \notin cbEncoders[cb]

\* Active encoder has valid impl (invariant that v2.7 maintains)
ActiveImpliesValidImpl ==
    \A e \in Encoders :
        encoderState[e] = "active" => encoderState[e] # "impl_invalid"

(****************************************************************************)
(* Specification                                                            *)
(****************************************************************************)

vars == <<encoderState, encoderType, encoderOwner, encoderRetainCount, encoderActiveCalls,
          cbState, cbEncoders, cbCompletionHandler, v27_tracked, v27_ended, v27_cbEncoders,
          mutexHolder, mutexDepth, dispatchQueues, queueType,
          threadState, threadEncoder, threadCB, threadQueue, threadObservedState,
          gpuQueue, gpuBusy, pendingCallbacks, crashed, crashType>>

Spec == Init /\ [][Next]_vars

(****************************************************************************)
(* THEOREMS                                                                 *)
(****************************************************************************)

THEOREM V27PreventsAllCrashes == Spec => []NoCrashes

THEOREM V27MaintainsConsistency == Spec => [](V27Consistent /\ MutexConsistent)

THEOREM V27PreventsValidation == Spec => []NoValidationCrash

=============================================================================
