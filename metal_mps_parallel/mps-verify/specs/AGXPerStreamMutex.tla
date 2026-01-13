---------------------------- MODULE AGXPerStreamMutex ----------------------------
(*
 * AGX Per-Stream Mutex Model - Proves Insufficient
 *
 * Worker: N=1474
 * Purpose: Phase 4.1 - Prove that per-stream mutexes do NOT prevent the race
 *
 * This model shows that even if we have one mutex per command queue/stream,
 * the race condition STILL occurs because:
 * 1. Different streams can share the same context registry
 * 2. Context invalidation on stream A affects threads using stream B
 *
 * EXPECTED RESULT: TLC finds a violation (race still occurs)
 *)

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    NumThreads,         \* Number of concurrent threads
    NumStreams,         \* Number of MPS streams (command queues)
    NumContextSlots     \* Number of slots in context registry

ASSUME NumThreads \in Nat /\ NumThreads > 0
ASSUME NumStreams \in Nat /\ NumStreams > 1  \* Need >1 to show race
ASSUME NumContextSlots \in Nat /\ NumContextSlots > 0

VARIABLES
    (* Per-stream mutex (THE "FIX" WE'RE TESTING) *)
    stream_mutex,       \* Stream -> Thread | NULL (who holds the mutex)

    (* Per-thread state *)
    thread_stream,      \* Thread -> StreamId (assigned stream via round-robin)
    thread_context,     \* Thread -> ContextId | NULL
    thread_state,       \* Thread -> ThreadState

    (* Shared context registry (STILL SHARED ACROSS STREAMS) *)
    context_registry,   \* ContextId -> {valid, invalid}
    context_stream,     \* ContextId -> StreamId (which stream created it)

    (* Bug detection *)
    null_deref_count,
    race_witnessed

vars == <<stream_mutex, thread_stream, thread_context, thread_state,
          context_registry, context_stream, null_deref_count, race_witnessed>>

Threads == 1..NumThreads
Streams == 1..NumStreams
ContextIds == 1..NumContextSlots
NULL == 0

ThreadStates == {"idle", "acquiring", "creating", "encoding", "destroying", "releasing"}

(* -------------------------------------------------------------------------- *)
(* Initial State                                                              *)
(* -------------------------------------------------------------------------- *)

Init ==
    /\ stream_mutex = [s \in Streams |-> NULL]
    \* Round-robin stream assignment
    /\ thread_stream = [t \in Threads |-> ((t - 1) % NumStreams) + 1]
    /\ thread_context = [t \in Threads |-> NULL]
    /\ thread_state = [t \in Threads |-> "idle"]
    /\ context_registry = [c \in ContextIds |-> "invalid"]
    /\ context_stream = [c \in ContextIds |-> NULL]
    /\ null_deref_count = 0
    /\ race_witnessed = FALSE

(* -------------------------------------------------------------------------- *)
(* Per-Stream Mutex Operations                                                *)
(* -------------------------------------------------------------------------- *)

\* Thread acquires its stream's mutex
AcquireStreamMutex(t) ==
    LET s == thread_stream[t] IN
    /\ thread_state[t] = "idle"
    /\ stream_mutex[s] = NULL
    /\ stream_mutex' = [stream_mutex EXCEPT ![s] = t]
    /\ thread_state' = [thread_state EXCEPT ![t] = "acquiring"]
    /\ UNCHANGED <<thread_stream, thread_context, context_registry,
                   context_stream, null_deref_count, race_witnessed>>

\* Thread releases its stream's mutex
ReleaseStreamMutex(t) ==
    LET s == thread_stream[t] IN
    /\ thread_state[t] = "releasing"
    /\ stream_mutex[s] = t
    /\ stream_mutex' = [stream_mutex EXCEPT ![s] = NULL]
    /\ thread_state' = [thread_state EXCEPT ![t] = "idle"]
    /\ UNCHANGED <<thread_stream, thread_context, context_registry,
                   context_stream, null_deref_count, race_witnessed>>

(* -------------------------------------------------------------------------- *)
(* Context Operations (Within Per-Stream Lock)                                *)
(* -------------------------------------------------------------------------- *)

\* Create context (holding stream mutex)
CreateContext(t) ==
    LET s == thread_stream[t] IN
    /\ thread_state[t] = "acquiring"
    /\ stream_mutex[s] = t
    /\ thread_context[t] = NULL
    /\ \E c \in ContextIds:
        /\ context_registry[c] = "invalid"
        /\ context_stream' = [context_stream EXCEPT ![c] = s]
        /\ thread_context' = [thread_context EXCEPT ![t] = c]
    /\ thread_state' = [thread_state EXCEPT ![t] = "creating"]
    /\ UNCHANGED <<stream_mutex, thread_stream, context_registry, null_deref_count, race_witnessed>>

\* Start encoding (publish context as valid)
StartEncoding(t) ==
    LET c == thread_context[t] IN
    /\ thread_state[t] = "creating"
    /\ c /= NULL
    /\ context_registry' = [context_registry EXCEPT ![c] = "valid"]
    /\ thread_state' = [thread_state EXCEPT ![t] = "encoding"]
    /\ UNCHANGED <<stream_mutex, thread_stream, thread_context, context_stream,
                   null_deref_count, race_witnessed>>

\* Continue encoding - check for NULL deref
\* THE BUG: Context might be invalid due to ANOTHER stream's thread
ContinueEncoding(t) ==
    LET c == thread_context[t] IN
    /\ thread_state[t] = "encoding"
    /\ c /= NULL
    /\ IF context_registry[c] = "invalid"
       THEN \* NULL pointer dereference!
            /\ null_deref_count' = null_deref_count + 1
            /\ race_witnessed' = TRUE
            /\ UNCHANGED <<stream_mutex, thread_stream, thread_context,
                          thread_state, context_registry, context_stream>>
       ELSE \* Normal encoding continues
            /\ UNCHANGED vars

\* Finish encoding, start destroy
FinishEncoding(t) ==
    /\ thread_state[t] = "encoding"
    /\ thread_context[t] /= NULL
    /\ thread_state' = [thread_state EXCEPT ![t] = "destroying"]
    /\ UNCHANGED <<stream_mutex, thread_stream, thread_context, context_registry,
                   context_stream, null_deref_count, race_witnessed>>

\* Destroy context (THE KEY: this affects shared registry)
\* Even though we hold our stream's mutex, we modify the GLOBAL registry
DestroyContext(t) ==
    LET c == thread_context[t] IN
    /\ thread_state[t] = "destroying"
    /\ c /= NULL
    /\ context_registry' = [context_registry EXCEPT ![c] = "invalid"]
    /\ context_stream' = [context_stream EXCEPT ![c] = NULL]
    /\ thread_context' = [thread_context EXCEPT ![t] = NULL]
    /\ thread_state' = [thread_state EXCEPT ![t] = "releasing"]
    /\ UNCHANGED <<stream_mutex, thread_stream, null_deref_count, race_witnessed>>

(* -------------------------------------------------------------------------- *)
(* Next State Relation                                                        *)
(* -------------------------------------------------------------------------- *)

Next ==
    \E t \in Threads:
        \/ AcquireStreamMutex(t)
        \/ CreateContext(t)
        \/ StartEncoding(t)
        \/ ContinueEncoding(t)
        \/ FinishEncoding(t)
        \/ DestroyContext(t)
        \/ ReleaseStreamMutex(t)

(* -------------------------------------------------------------------------- *)
(* Fairness (Ensure Progress)                                                 *)
(* -------------------------------------------------------------------------- *)

Fairness ==
    /\ \A t \in Threads: WF_vars(AcquireStreamMutex(t))
    /\ \A t \in Threads: WF_vars(CreateContext(t))
    /\ \A t \in Threads: WF_vars(StartEncoding(t))
    /\ \A t \in Threads: WF_vars(FinishEncoding(t))
    /\ \A t \in Threads: WF_vars(DestroyContext(t))
    /\ \A t \in Threads: WF_vars(ReleaseStreamMutex(t))

Spec == Init /\ [][Next]_vars /\ Fairness

(* -------------------------------------------------------------------------- *)
(* Safety Properties (THESE SHOULD FAIL)                                      *)
(* -------------------------------------------------------------------------- *)

\* This property SHOULD be violated, proving per-stream mutex insufficient
NoNullDereference ==
    null_deref_count = 0

\* Alternative formulation
NoRaceCondition ==
    ~race_witnessed

\* The mutex only protects within a stream, not across streams
PerStreamMutexIsInsufficient ==
    \* If thread t is encoding, another thread on DIFFERENT stream can still
    \* invalidate t's context
    \A t1, t2 \in Threads:
        (t1 /= t2 /\ thread_stream[t1] /= thread_stream[t2]) =>
            \* Even with mutex held, cross-stream race is possible
            TRUE

(* -------------------------------------------------------------------------- *)
(* Comments                                                                   *)
(* -------------------------------------------------------------------------- *)
(*
 * WHY PER-STREAM MUTEX FAILS:
 *
 * Thread 1 on Stream A:          Thread 2 on Stream B:
 * ----------------------         ----------------------
 * acquire(mutex_A)
 * context = registry[0]
 * registry[0] = VALID
 * start encoding...
 *                                acquire(mutex_B)
 *                                (gets SAME context slot!)
 *                                registry[0] = INVALID  <-- RACE!
 *                                release(mutex_B)
 * use context -> NULL DEREF!
 *
 * The context registry is GLOBAL, not per-stream. Per-stream mutexes
 * protect stream operations but not the shared context state.
 *
 * To run TLC:
 *   java -jar tla2tools.jar -config AGXPerStreamMutex.cfg AGXPerStreamMutex.tla
 *
 * Expected output: Invariant NoNullDereference is violated.
 *)
=============================================================================
