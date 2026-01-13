---------------------------- MODULE MPSStreamPool ----------------------------
(*
 * MPS Stream Pool State Machine (Abstract Model)
 *
 * This TLA+ specification models the MPS stream-pool lifecycle and TLS stream
 * assignment at a conservative level to check safety properties like:
 *   1. NoUseAfterFree: No thread uses a stream when pool is dead
 *   2. TLS binding safety: No dangling pointers
 *   3. Stream assignment correctness
 *
 * NOTE: The current fork implementation uses a TLS-bound stream slot with a
 * lock-free freelist (see `pytorch-mps-fork/aten/src/ATen/mps/MPSStream.mm`).
 * This spec is not intended to be a line-by-line control-flow mirror of the
 * current implementation; it encodes the required liveness/safety obligations
 * around pool lifetime and stream assignment.
 *)

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    NumThreads,      \* Number of threads (e.g., 4 for model checking)
    NumStreams       \* Number of streams in pool (e.g., 4 for model checking)

ASSUME NumThreads \in Nat /\ NumThreads > 0
ASSUME NumStreams \in Nat /\ NumStreams > 1  \* Need at least stream 0 + 1 worker

VARIABLES
    pool_alive,           \* Is the pool currently alive?
    pool_ever_created,    \* Has the pool ever been created?
    in_forked_child,      \* Are we in a forked child process?
    stream_counter,       \* Round-robin counter for stream selection
    tls_stream,           \* Per-thread TLS stream binding: Thread -> StreamId | NULL
    is_main_thread,       \* Thread -> Bool, TRUE for exactly one thread (pthread_main_np)
    pc                    \* Program counter per thread (state machine)

vars == <<pool_alive, pool_ever_created, in_forked_child, stream_counter,
          tls_stream, is_main_thread, pc>>

Threads == 1..NumThreads
WorkerStreams == 1..(NumStreams - 1)  \* Streams 1 to NumStreams-1
DefaultStream == 0                     \* Stream 0 is the default (main thread)
NullStream == -1                       \* -1 represents NULL (no TLS binding)

(*
 * Program counter states model the getCurrentStream() control flow:
 *
 * idle -> check1_alive -> [if tls!=null] -> check2_toctou -> using_stream
 *                       -> [if tls==null] -> assigning -> check3_toctou -> using_stream
 *                       -> [pool dead/never created] -> creating_pool / return_null
 *)
PCStates == {
    "idle",                \* Not in getCurrentStream
    "check1_alive",        \* Line 722: if (!g_pool_alive)
    "check_tls",           \* Line 729: if (tls_current_stream != nullptr)
    "check2_toctou",       \* Line 736: TOCTOU fix 32.99 - re-check after TLS read
    "assigning",           \* Lines 745-751: Assigning stream (main vs worker)
    "check3_toctou",       \* Line 762: TOCTOU fix 32.104 - re-check after assignment
    "using_stream",        \* Successfully using the stream
    "return_null",         \* Returning nullptr (safe)
    "done"                 \* Completed getCurrentStream call
}

(* Type invariant *)
TypeOK ==
    /\ pool_alive \in BOOLEAN
    /\ pool_ever_created \in BOOLEAN
    /\ in_forked_child \in BOOLEAN
    /\ stream_counter \in Nat
    /\ tls_stream \in [Threads -> {NullStream} \cup (0..(NumStreams-1))]
    /\ is_main_thread \in [Threads -> BOOLEAN]
    /\ pc \in [Threads -> PCStates]

(* Initial state *)
Init ==
    /\ pool_alive = FALSE
    /\ pool_ever_created = FALSE
    /\ in_forked_child = FALSE
    /\ stream_counter = 0
    /\ tls_stream = [t \in Threads |-> NullStream]
    /\ is_main_thread = [t \in Threads |-> t = 1]  \* Thread 1 is "main thread"
    /\ pc = [t \in Threads |-> "idle"]

-----------------------------------------------------------------------------
(* THREAD ACTIONS - Model getCurrentStream() *)

(* Start getCurrentStream - enters check1_alive state *)
StartGetStream(t) ==
    /\ pc[t] = "idle"
    /\ ~in_forked_child  \* Line 713: TORCH_CHECK(!g_in_forked_child)
    /\ pc' = [pc EXCEPT ![t] = "check1_alive"]
    /\ UNCHANGED <<pool_alive, pool_ever_created, in_forked_child,
                   stream_counter, tls_stream, is_main_thread>>

(* CHECK 1 (Line 722): if (!g_pool_alive.load(acquire)) *)
Check1Alive(t) ==
    /\ pc[t] = "check1_alive"
    /\ \/ /\ ~pool_alive /\ pool_ever_created
          \* Pool was created and destroyed - return nullptr
          /\ pc' = [pc EXCEPT ![t] = "return_null"]
          /\ UNCHANGED <<pool_alive, pool_ever_created, in_forked_child,
                         stream_counter, tls_stream, is_main_thread>>
       \/ /\ ~pool_alive /\ ~pool_ever_created
          \* Pool never created - create it (lazy init)
          /\ pool_alive' = TRUE
          /\ pool_ever_created' = TRUE
          /\ pc' = [pc EXCEPT ![t] = "check_tls"]
          /\ UNCHANGED <<in_forked_child, stream_counter, tls_stream, is_main_thread>>
       \/ /\ pool_alive
          \* Pool is alive - proceed to TLS check
          /\ pc' = [pc EXCEPT ![t] = "check_tls"]
          /\ UNCHANGED <<pool_alive, pool_ever_created, in_forked_child,
                         stream_counter, tls_stream, is_main_thread>>

(* Line 729: if (tls_current_stream != nullptr) *)
CheckTLS(t) ==
    /\ pc[t] = "check_tls"
    /\ \/ /\ tls_stream[t] # NullStream
          \* Have cached TLS binding - need TOCTOU check 2
          /\ pc' = [pc EXCEPT ![t] = "check2_toctou"]
          /\ UNCHANGED <<pool_alive, pool_ever_created, in_forked_child,
                         stream_counter, tls_stream, is_main_thread>>
       \/ /\ tls_stream[t] = NullStream
          \* No TLS binding - need to assign stream
          /\ pc' = [pc EXCEPT ![t] = "assigning"]
          /\ UNCHANGED <<pool_alive, pool_ever_created, in_forked_child,
                         stream_counter, tls_stream, is_main_thread>>

(* CHECK 2 (Line 736): TOCTOU fix 32.99 - Re-check pool_alive after TLS read *)
Check2TOCTOU(t) ==
    /\ pc[t] = "check2_toctou"
    /\ \/ /\ ~pool_alive
          \* Pool died between check1 and here - return nullptr
          /\ pc' = [pc EXCEPT ![t] = "return_null"]
          /\ UNCHANGED <<pool_alive, pool_ever_created, in_forked_child,
                         stream_counter, tls_stream, is_main_thread>>
       \/ /\ pool_alive
          \* Pool still alive - safe to use cached TLS stream
          /\ pc' = [pc EXCEPT ![t] = "using_stream"]
          /\ UNCHANGED <<pool_alive, pool_ever_created, in_forked_child,
                         stream_counter, tls_stream, is_main_thread>>

(* Lines 745-751: Assign stream based on main thread vs worker *)
AssignStream(t) ==
    /\ pc[t] = "assigning"
    /\ \/ /\ is_main_thread[t]
          \* Main thread gets default stream (id 0)
          /\ tls_stream' = [tls_stream EXCEPT ![t] = DefaultStream]
          /\ UNCHANGED stream_counter
       \/ /\ ~is_main_thread[t]
          \* Worker thread: assign via round-robin
          /\ LET idx == (stream_counter % (NumStreams - 1)) + 1
             IN /\ stream_counter' = stream_counter + 1
                /\ tls_stream' = [tls_stream EXCEPT ![t] = idx]
    /\ pc' = [pc EXCEPT ![t] = "check3_toctou"]
    /\ UNCHANGED <<pool_alive, pool_ever_created, in_forked_child, is_main_thread>>

(* CHECK 3 (Line 762): TOCTOU fix 32.104 - Re-check pool_alive after assignment *)
Check3TOCTOU(t) ==
    /\ pc[t] = "check3_toctou"
    /\ \/ /\ ~pool_alive
          \* Pool died during assignment - return nullptr (UAF prevented!)
          /\ pc' = [pc EXCEPT ![t] = "return_null"]
          /\ UNCHANGED <<pool_alive, pool_ever_created, in_forked_child,
                         stream_counter, tls_stream, is_main_thread>>
       \/ /\ pool_alive
          \* Pool still alive - safe to use newly assigned stream
          /\ pc' = [pc EXCEPT ![t] = "using_stream"]
          /\ UNCHANGED <<pool_alive, pool_ever_created, in_forked_child,
                         stream_counter, tls_stream, is_main_thread>>

(* Finish using stream - return to idle *)
FinishUsingStream(t) ==
    /\ pc[t] = "using_stream"
    /\ pc' = [pc EXCEPT ![t] = "done"]
    /\ UNCHANGED <<pool_alive, pool_ever_created, in_forked_child,
                   stream_counter, tls_stream, is_main_thread>>

(* Return from return_null state *)
FinishReturnNull(t) ==
    /\ pc[t] = "return_null"
    /\ pc' = [pc EXCEPT ![t] = "done"]
    /\ UNCHANGED <<pool_alive, pool_ever_created, in_forked_child,
                   stream_counter, tls_stream, is_main_thread>>

(* Reset from done to idle (for repeated calls) *)
Reset(t) ==
    /\ pc[t] = "done"
    /\ pc' = [pc EXCEPT ![t] = "idle"]
    /\ UNCHANGED <<pool_alive, pool_ever_created, in_forked_child,
                   stream_counter, tls_stream, is_main_thread>>

-----------------------------------------------------------------------------
(* POOL LIFECYCLE ACTIONS *)

(* Pool destruction - happens at program exit *)
(* Only allowed when no threads are in "using_stream" state *)
DestroyPool ==
    /\ pool_alive
    /\ \A t \in Threads: pc[t] # "using_stream"  \* No active users
    /\ pool_alive' = FALSE
    \* TLS values become stale (dangling) but TOCTOU checks protect us
    /\ UNCHANGED <<pool_ever_created, in_forked_child, stream_counter,
                   tls_stream, is_main_thread, pc>>

(* Fork handler - called in child after fork() *)
(* In the forked child, no threads should continue from their current position.
 * The child process is a new process; any thread state is reset.
 * The TORCH_CHECK at the start of getCurrentStream prevents MPS use in child. *)
Fork ==
    /\ pool_ever_created
    /\ ~in_forked_child  \* Can only fork once
    /\ in_forked_child' = TRUE
    /\ pool_alive' = FALSE
    \* TLS is inherited but now invalid - cleared by fork handler
    /\ tls_stream' = [t \in Threads |-> NullStream]
    \* All threads reset to idle - child process starts fresh
    /\ pc' = [t \in Threads |-> "idle"]
    /\ UNCHANGED <<pool_ever_created, stream_counter, is_main_thread>>

-----------------------------------------------------------------------------
(* TERMINAL STATE *)

(* In a forked child, MPS is permanently disabled - this is a terminal state.
 * We allow stuttering (no change) to avoid TLC reporting it as a deadlock. *)
ForkedChildTerminated ==
    /\ in_forked_child
    /\ UNCHANGED vars

-----------------------------------------------------------------------------
(* NEXT STATE RELATION *)

Next ==
    \/ DestroyPool
    \/ Fork
    \/ ForkedChildTerminated
    \/ \E t \in Threads:
        \/ StartGetStream(t)
        \/ Check1Alive(t)
        \/ CheckTLS(t)
        \/ Check2TOCTOU(t)
        \/ AssignStream(t)
        \/ Check3TOCTOU(t)
        \/ FinishUsingStream(t)
        \/ FinishReturnNull(t)
        \/ Reset(t)

(* Fairness: all threads eventually make progress *)
Fairness == \A t \in Threads: WF_vars(StartGetStream(t))

Spec == Init /\ [][Next]_vars /\ Fairness

-----------------------------------------------------------------------------
(* SAFETY PROPERTIES *)

(* CRITICAL: No use-after-free - if a thread is "using_stream", pool must be alive *)
(* This is what the TOCTOU fixes 32.99 and 32.104 prevent *)
NoUseAfterFree ==
    \A t \in Threads:
        pc[t] = "using_stream" => pool_alive

(* TLS binding is always valid (within bounds) when pool is alive *)
TLSBindingValid ==
    \A t \in Threads:
        (tls_stream[t] # NullStream /\ pool_alive) =>
        tls_stream[t] \in 0..(NumStreams-1)

(* Stream assignment is within bounds *)
StreamBoundsValid ==
    \A t \in Threads:
        tls_stream[t] # NullStream => tls_stream[t] \in 0..(NumStreams-1)

(* Main thread gets default stream (0), workers get 1..NumStreams-1 *)
MainThreadGetsDefaultStream ==
    \A t \in Threads:
        (is_main_thread[t] /\ tls_stream[t] # NullStream) =>
        tls_stream[t] = DefaultStream

WorkerThreadsAvoidDefaultStream ==
    \A t \in Threads:
        (~is_main_thread[t] /\ tls_stream[t] # NullStream) =>
        tls_stream[t] \in WorkerStreams

(* Fork invalidates all TLS bindings *)
ForkInvalidatesTLS ==
    in_forked_child => \A t \in Threads: tls_stream[t] = NullStream

(* Pool alive implies ever created *)
PoolAliveImpliesCreated ==
    pool_alive => pool_ever_created

(* Exactly one main thread *)
ExactlyOneMainThread ==
    Cardinality({t \in Threads : is_main_thread[t]}) = 1

(* Combined safety invariant *)
Safety ==
    /\ TypeOK
    /\ NoUseAfterFree
    /\ TLSBindingValid
    /\ StreamBoundsValid
    /\ MainThreadGetsDefaultStream
    /\ WorkerThreadsAvoidDefaultStream
    /\ ForkInvalidatesTLS
    /\ PoolAliveImpliesCreated
    /\ ExactlyOneMainThread

-----------------------------------------------------------------------------
(* LIVENESS PROPERTIES *)

(* Eventually a thread finishes getCurrentStream call *)
EventuallyCompletes ==
    \A t \in Threads:
        pc[t] # "idle" ~> pc[t] = "done"

(* No deadlock: some action is always enabled *)
NoDeadlock ==
    [][ENABLED(Next)]_vars

=============================================================================
\* Modification History
\* Last modified: 2025-12-16
\* Fixed to accurately model TOCTOU fixes 32.99 (line 736) and 32.104 (line 762)
\* Created for MPS Parallel Inference Verification Platform
