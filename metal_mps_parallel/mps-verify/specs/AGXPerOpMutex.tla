---------------------------- MODULE AGXPerOpMutex ----------------------------
(*
 * AGX Per-Operation Mutex Model - Proves Insufficient
 *
 * Worker: N=1474
 * Purpose: Phase 4.1 - Prove that per-operation mutexes do NOT prevent the race
 *
 * This model shows that having separate mutexes for different operation types
 * (create, encode, destroy) does NOT prevent the race because:
 * 1. A thread can hold the encode mutex while another holds the destroy mutex
 * 2. Context invalidation is not atomic with context usage
 *
 * EXPECTED RESULT: TLC finds a violation (race still occurs)
 *)

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    NumThreads,
    NumContextSlots

ASSUME NumThreads \in Nat /\ NumThreads > 1
ASSUME NumContextSlots \in Nat /\ NumContextSlots > 0

VARIABLES
    (* Per-operation mutexes (THE "FIX" WE'RE TESTING) *)
    create_mutex,       \* Thread | NULL
    encode_mutex,       \* Thread | NULL
    destroy_mutex,      \* Thread | NULL

    (* Per-thread state *)
    thread_context,     \* Thread -> ContextId | NULL
    thread_state,       \* Thread -> ThreadState

    (* Shared context registry *)
    context_registry,   \* ContextId -> {valid, invalid}

    (* Bug detection *)
    null_deref_count,
    race_witnessed

vars == <<create_mutex, encode_mutex, destroy_mutex, thread_context,
          thread_state, context_registry, null_deref_count, race_witnessed>>

Threads == 1..NumThreads
ContextIds == 1..NumContextSlots
NULL == 0

ThreadStates == {
    "idle",
    "want_create", "creating",
    "want_encode", "encoding",
    "want_destroy", "destroying"
}

(* -------------------------------------------------------------------------- *)
(* Initial State                                                              *)
(* -------------------------------------------------------------------------- *)

Init ==
    /\ create_mutex = NULL
    /\ encode_mutex = NULL
    /\ destroy_mutex = NULL
    /\ thread_context = [t \in Threads |-> NULL]
    /\ thread_state = [t \in Threads |-> "idle"]
    /\ context_registry = [c \in ContextIds |-> "invalid"]
    /\ null_deref_count = 0
    /\ race_witnessed = FALSE

(* -------------------------------------------------------------------------- *)
(* Create Operations (Protected by create_mutex)                              *)
(* -------------------------------------------------------------------------- *)

WantCreate(t) ==
    /\ thread_state[t] = "idle"
    /\ thread_context[t] = NULL
    /\ thread_state' = [thread_state EXCEPT ![t] = "want_create"]
    /\ UNCHANGED <<create_mutex, encode_mutex, destroy_mutex, thread_context,
                   context_registry, null_deref_count, race_witnessed>>

AcquireCreateMutex(t) ==
    /\ thread_state[t] = "want_create"
    /\ create_mutex = NULL
    /\ create_mutex' = t
    /\ UNCHANGED <<encode_mutex, destroy_mutex, thread_context, thread_state,
                   context_registry, null_deref_count, race_witnessed>>

CreateContext(t) ==
    /\ thread_state[t] = "want_create"
    /\ create_mutex = t
    /\ \E c \in ContextIds:
        /\ context_registry[c] = "invalid"
        /\ context_registry' = [context_registry EXCEPT ![c] = "valid"]
        /\ thread_context' = [thread_context EXCEPT ![t] = c]
    /\ create_mutex' = NULL  \* Release immediately after create
    /\ thread_state' = [thread_state EXCEPT ![t] = "creating"]
    /\ UNCHANGED <<encode_mutex, destroy_mutex, null_deref_count, race_witnessed>>

(* -------------------------------------------------------------------------- *)
(* Encode Operations (Protected by encode_mutex)                              *)
(* -------------------------------------------------------------------------- *)

WantEncode(t) ==
    /\ thread_state[t] = "creating"
    /\ thread_context[t] /= NULL
    /\ thread_state' = [thread_state EXCEPT ![t] = "want_encode"]
    /\ UNCHANGED <<create_mutex, encode_mutex, destroy_mutex, thread_context,
                   context_registry, null_deref_count, race_witnessed>>

AcquireEncodeMutex(t) ==
    /\ thread_state[t] = "want_encode"
    /\ encode_mutex = NULL
    /\ encode_mutex' = t
    /\ thread_state' = [thread_state EXCEPT ![t] = "encoding"]
    /\ UNCHANGED <<create_mutex, destroy_mutex, thread_context,
                   context_registry, null_deref_count, race_witnessed>>

\* Context aliasing/migration: another thread starts using the SAME context id.
\* Models driver-level context reuse/handoff that bypasses per-operation locks.
AliasContext(from, to) ==
    LET c == thread_context[from] IN
    /\ from \in Threads /\ to \in Threads /\ from /= to
    /\ thread_state[from] = "encoding"
    /\ encode_mutex = from
    /\ c /= NULL
    /\ thread_state[to] = "idle"
    /\ thread_context[to] = NULL
    /\ thread_context' = [thread_context EXCEPT ![to] = c]
    /\ thread_state' = [thread_state EXCEPT ![to] = "want_destroy"]
    /\ UNCHANGED <<create_mutex, encode_mutex, destroy_mutex, context_registry,
                   null_deref_count, race_witnessed>>

\* THE BUG: While holding encode_mutex, context can still be invalidated
\* by another thread holding destroy_mutex
ContinueEncoding(t) ==
    LET c == thread_context[t] IN
    /\ thread_state[t] = "encoding"
    /\ encode_mutex = t
    /\ c /= NULL
    /\ IF context_registry[c] = "invalid"
       THEN \* NULL pointer dereference!
            /\ null_deref_count' = null_deref_count + 1
            /\ race_witnessed' = TRUE
            /\ UNCHANGED <<create_mutex, encode_mutex, destroy_mutex,
                          thread_context, thread_state, context_registry>>
       ELSE
            /\ UNCHANGED vars

FinishEncoding(t) ==
    /\ thread_state[t] = "encoding"
    /\ encode_mutex = t
    /\ encode_mutex' = NULL  \* Release encode mutex
    /\ thread_state' = [thread_state EXCEPT ![t] = "want_destroy"]
    /\ UNCHANGED <<create_mutex, destroy_mutex, thread_context,
                   context_registry, null_deref_count, race_witnessed>>

(* -------------------------------------------------------------------------- *)
(* Destroy Operations (Protected by destroy_mutex)                            *)
(* -------------------------------------------------------------------------- *)

AcquireDestroyMutex(t) ==
    /\ thread_state[t] = "want_destroy"
    /\ destroy_mutex = NULL
    /\ destroy_mutex' = t
    /\ thread_state' = [thread_state EXCEPT ![t] = "destroying"]
    /\ UNCHANGED <<create_mutex, encode_mutex, thread_context,
                   context_registry, null_deref_count, race_witnessed>>

\* THE KEY: This invalidates context while another thread might be encoding
DestroyContext(t) ==
    LET c == thread_context[t] IN
    /\ thread_state[t] = "destroying"
    /\ destroy_mutex = t
    /\ c /= NULL
    /\ context_registry' = [context_registry EXCEPT ![c] = "invalid"]
    /\ thread_context' = [thread_context EXCEPT ![t] = NULL]
    /\ destroy_mutex' = NULL
    /\ thread_state' = [thread_state EXCEPT ![t] = "idle"]
    /\ UNCHANGED <<create_mutex, encode_mutex, null_deref_count, race_witnessed>>

(* -------------------------------------------------------------------------- *)
(* Next State Relation                                                        *)
(* -------------------------------------------------------------------------- *)

Next ==
    \E t \in Threads:
        \/ WantCreate(t)
        \/ AcquireCreateMutex(t)
        \/ CreateContext(t)
        \/ WantEncode(t)
        \/ AcquireEncodeMutex(t)
        \/ ContinueEncoding(t)
        \/ FinishEncoding(t)
        \/ AcquireDestroyMutex(t)
        \/ DestroyContext(t)
    \/ \E t1 \in Threads, t2 \in Threads: AliasContext(t1, t2)

Fairness ==
    /\ \A t \in Threads: WF_vars(WantCreate(t))
    /\ \A t \in Threads: WF_vars(CreateContext(t))
    /\ \A t \in Threads: WF_vars(WantEncode(t))
    /\ \A t \in Threads: WF_vars(AcquireEncodeMutex(t))
    /\ \A t \in Threads: WF_vars(FinishEncoding(t))
    /\ \A t \in Threads: WF_vars(AcquireDestroyMutex(t))
    /\ \A t \in Threads: WF_vars(DestroyContext(t))

Spec == Init /\ [][Next]_vars /\ Fairness

(* -------------------------------------------------------------------------- *)
(* Safety Properties (THESE SHOULD FAIL)                                      *)
(* -------------------------------------------------------------------------- *)

NoNullDereference ==
    null_deref_count = 0

NoRaceCondition ==
    ~race_witnessed

(* -------------------------------------------------------------------------- *)
(* Comments                                                                   *)
(* -------------------------------------------------------------------------- *)
(*
 * WHY PER-OPERATION MUTEX FAILS:
 *
 * Thread 1:                      Thread 2:
 * ---------                      ---------
 * create_mutex.acquire()
 * context = allocate()
 * create_mutex.release()
 *
 * encode_mutex.acquire()
 * start encoding with context
 *                                destroy_mutex.acquire()
 *                                (holds different mutex!)
 *                                context.invalidate()  <-- RACE!
 *                                destroy_mutex.release()
 * use context -> NULL DEREF!
 * encode_mutex.release()
 *
 * The problem: encode_mutex and destroy_mutex are DIFFERENT locks.
 * Holding one doesn't prevent the other from proceeding.
 *
 * To run TLC:
 *   java -jar tla2tools.jar -config AGXPerOpMutex.cfg AGXPerOpMutex.tla
 *
 * Expected output: Invariant NoNullDereference is violated.
 *)
=============================================================================
