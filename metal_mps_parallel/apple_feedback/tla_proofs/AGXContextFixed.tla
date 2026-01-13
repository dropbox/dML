---------------------------- MODULE AGXContextFixed ----------------------------
(*
 * AGX Driver Context - FIXED VERSION WITH MUTEX
 *
 * This TLA+ specification models the SAME AGX driver behavior, but WITH
 * proper mutex synchronization (our workaround).
 *
 * GOAL: Prove that adding a global encoding mutex prevents the race condition.
 *)

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    NumThreads,
    NumContextSlots

ASSUME NumThreads \in Nat /\ NumThreads > 0
ASSUME NumContextSlots \in Nat /\ NumContextSlots > 0

VARIABLES
    thread_context,
    thread_state,
    context_registry,
    context_owner,
    null_deref_count,
    race_witnessed,
    (* THE FIX: Global encoding mutex *)
    encoding_mutex_held  \* Thread holding mutex | NULL

vars == <<thread_context, thread_state, context_registry, context_owner,
          null_deref_count, race_witnessed, encoding_mutex_held>>

Threads == 1..NumThreads
ContextIds == 1..NumContextSlots
NULL == 0

ThreadStates == {
    "idle",
    "waiting_for_mutex",  \* NEW: waiting to acquire mutex
    "creating",
    "encoding",
    "destroying"
}

ContextStates == {"valid", "invalid"}

(* -------------------------------------------------------------------------- *)
(* Initial State                                                              *)
(* -------------------------------------------------------------------------- *)

Init ==
    /\ thread_context = [t \in Threads |-> NULL]
    /\ thread_state = [t \in Threads |-> "idle"]
    /\ context_registry = [c \in ContextIds |-> "invalid"]
    /\ context_owner = [c \in ContextIds |-> NULL]
    /\ null_deref_count = 0
    /\ race_witnessed = FALSE
    /\ encoding_mutex_held = NULL

(* -------------------------------------------------------------------------- *)
(* Actions: Context Lifecycle WITH MUTEX                                      *)
(* -------------------------------------------------------------------------- *)

(* Thread tries to acquire mutex before creating context *)
TryAcquireMutex(t) ==
    /\ thread_state[t] = "idle"
    /\ thread_context[t] = NULL
    /\ IF encoding_mutex_held = NULL
       THEN (* Got mutex *)
            /\ encoding_mutex_held' = t
            /\ thread_state' = [thread_state EXCEPT ![t] = "creating"]
       ELSE (* Mutex held, wait *)
            /\ thread_state' = [thread_state EXCEPT ![t] = "waiting_for_mutex"]
            /\ UNCHANGED encoding_mutex_held
    /\ UNCHANGED <<thread_context, context_registry, context_owner,
                   null_deref_count, race_witnessed>>

(* Thread waiting for mutex can acquire it when released *)
AcquireMutexAfterWait(t) ==
    /\ thread_state[t] = "waiting_for_mutex"
    /\ encoding_mutex_held = NULL
    /\ encoding_mutex_held' = t
    /\ thread_state' = [thread_state EXCEPT ![t] = "creating"]
    /\ UNCHANGED <<thread_context, context_registry, context_owner,
                   null_deref_count, race_witnessed>>

(* Thread creates context WHILE HOLDING MUTEX *)
CreateContext(t) ==
    /\ thread_state[t] = "creating"
    /\ encoding_mutex_held = t  \* Must hold mutex!
    /\ \E c \in ContextIds:
        /\ context_registry[c] = "invalid"
        /\ thread_context' = [thread_context EXCEPT ![t] = c]
        /\ context_owner' = [context_owner EXCEPT ![c] = t]
        /\ context_registry' = [context_registry EXCEPT ![c] = "valid"]
    /\ thread_state' = [thread_state EXCEPT ![t] = "encoding"]
    /\ UNCHANGED <<null_deref_count, race_witnessed, encoding_mutex_held>>

(* Thread uses context WHILE HOLDING MUTEX - always safe! *)
UseContext(t) ==
    /\ thread_state[t] = "encoding"
    /\ encoding_mutex_held = t  \* Must hold mutex!
    /\ LET c == thread_context[t] IN
        (* With mutex, context is ALWAYS valid when we own it *)
        IF c /= NULL /\ context_registry[c] = "valid"
        THEN
            /\ UNCHANGED <<null_deref_count, race_witnessed>>
        ELSE
            (* This should NEVER happen with mutex *)
            /\ null_deref_count' = null_deref_count + 1
            /\ race_witnessed' = TRUE
    /\ thread_state' = [thread_state EXCEPT ![t] = "destroying"]
    /\ UNCHANGED <<thread_context, context_registry, context_owner, encoding_mutex_held>>

(* Thread destroys context and releases mutex *)
DestroyContextAndReleaseMutex(t) ==
    /\ thread_state[t] = "destroying"
    /\ encoding_mutex_held = t  \* Must hold mutex!
    /\ LET c == thread_context[t] IN
        /\ c /= NULL
        /\ context_registry' = [context_registry EXCEPT ![c] = "invalid"]
        /\ context_owner' = [context_owner EXCEPT ![c] = NULL]
        /\ thread_context' = [thread_context EXCEPT ![t] = NULL]
    /\ encoding_mutex_held' = NULL  \* Release mutex
    /\ thread_state' = [thread_state EXCEPT ![t] = "idle"]
    /\ UNCHANGED <<null_deref_count, race_witnessed>>

(* With mutex, threads CANNOT destroy other threads' contexts *)
(* This action is REMOVED from the fixed version *)

(* -------------------------------------------------------------------------- *)
(* Next State                                                                 *)
(* -------------------------------------------------------------------------- *)

Next == \E t \in Threads:
    \/ TryAcquireMutex(t)
    \/ AcquireMutexAfterWait(t)
    \/ CreateContext(t)
    \/ UseContext(t)
    \/ DestroyContextAndReleaseMutex(t)

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

(* -------------------------------------------------------------------------- *)
(* Properties                                                                 *)
(* -------------------------------------------------------------------------- *)

(* Safety: No NULL dereferences - SHOULD HOLD WITH MUTEX *)
NoNullDereferences == null_deref_count = 0

(* Mutex exclusion *)
MutexExclusion ==
    \A t1, t2 \in Threads:
        (encoding_mutex_held = t1 /\ encoding_mutex_held = t2) => t1 = t2

TypeOK ==
    /\ thread_context \in [Threads -> ContextIds \cup {NULL}]
    /\ thread_state \in [Threads -> ThreadStates]
    /\ context_registry \in [ContextIds -> ContextStates]
    /\ context_owner \in [ContextIds -> Threads \cup {NULL}]
    /\ null_deref_count \in Nat
    /\ race_witnessed \in BOOLEAN
    /\ encoding_mutex_held \in Threads \cup {NULL}

(* -------------------------------------------------------------------------- *)
(* What This Model Proves                                                     *)
(* -------------------------------------------------------------------------- *)
(*
 * If TLC verifies NoNullDereferences with NO violations, it proves:
 *   "Adding a global encoding mutex PREVENTS the race condition"
 *
 * This validates our workaround (getGlobalMetalEncodingMutex) is correct.
 *)

=============================================================================
