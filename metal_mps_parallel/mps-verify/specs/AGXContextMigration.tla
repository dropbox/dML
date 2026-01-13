------------------------- MODULE AGXContextMigration -------------------------
(*
 * AGX Context Migration / Aliasing Race Model
 *
 * This model explores an edge-case scenario: the driver (or higher-level Metal
 * stack) allows a compute context pointer to be "migrated" or aliased across
 * threads (e.g., via caching, internal handoff, or unexpected cross-thread use).
 *
 * Key point: even if threads only destroy contexts they "own", migration can
 * create aliasing such that Thread A invalidates a context still in use by
 * Thread B, producing the same NULL-deref signature observed in the wild.
 *
 * EXPECTED RESULT: TLC finds a violation (NULL dereference) with small bounds.
 *)

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    NumThreads,         \* Number of concurrent user threads
    NumContextSlots     \* Number of slots in the shared context registry

ASSUME NumThreads \in Nat /\ NumThreads > 0
ASSUME NumContextSlots \in Nat /\ NumContextSlots > 0

VARIABLES
    (* Per-thread state *)
    thread_context,     \* Thread -> ContextId | NULL
    thread_state,       \* Thread -> ThreadState

    (* Shared registry *)
    context_registry,   \* ContextId -> {valid, invalid}

    (* Bug detection *)
    null_deref_count,
    race_witnessed

vars == <<thread_context, thread_state, context_registry,
          null_deref_count, race_witnessed>>

Threads == 1..NumThreads
ContextIds == 1..NumContextSlots
NULL == 0

ThreadStates == {"idle", "encoding", "destroying"}
ContextStates == {"valid", "invalid"}

(* -------------------------------------------------------------------------- *)
(* Initial State                                                              *)
(* -------------------------------------------------------------------------- *)

Init ==
    /\ thread_context = [t \in Threads |-> NULL]
    /\ thread_state = [t \in Threads |-> "idle"]
    /\ context_registry = [c \in ContextIds |-> "invalid"]
    /\ null_deref_count = 0
    /\ race_witnessed = FALSE

(* -------------------------------------------------------------------------- *)
(* Actions                                                                    *)
(* -------------------------------------------------------------------------- *)

(* Thread allocates a context slot and begins encoding. *)
CreateContext(t) ==
    /\ thread_state[t] = "idle"
    /\ thread_context[t] = NULL
    /\ \E c \in ContextIds:
        /\ context_registry[c] = "invalid"
        /\ thread_context' = [thread_context EXCEPT ![t] = c]
        /\ context_registry' = [context_registry EXCEPT ![c] = "valid"]
    /\ thread_state' = [thread_state EXCEPT ![t] = "encoding"]
    /\ UNCHANGED <<null_deref_count, race_witnessed>>

(* Context migration/aliasing: another thread starts using the SAME context id. *)
(* Models "unexpected cross-thread use" without requiring a DestroyOtherContext action. *)
MigrateContext(from, to) ==
    /\ from \in Threads /\ to \in Threads /\ from /= to
    /\ thread_state[from] = "encoding"
    /\ thread_context[from] /= NULL
    /\ thread_state[to] = "idle"
    /\ thread_context[to] = NULL
    /\ LET c == thread_context[from] IN
        /\ thread_context' = [thread_context EXCEPT ![to] = c]
    /\ thread_state' = [thread_state EXCEPT ![to] = "encoding"]
    /\ UNCHANGED <<context_registry, null_deref_count, race_witnessed>>

(* Thread performs an encoding step using its context. *)
UseContext(t) ==
    /\ thread_state[t] = "encoding"
    /\ LET c == thread_context[t] IN
        IF c /= NULL /\ context_registry[c] = "valid"
        THEN
            /\ UNCHANGED <<null_deref_count, race_witnessed>>
        ELSE
            /\ null_deref_count' = null_deref_count + 1
            /\ race_witnessed' = TRUE
    /\ UNCHANGED <<thread_context, thread_state, context_registry>>

(* Thread decides to destroy its context (e.g., deferredEndEncoding / dealloc). *)
StartDestroy(t) ==
    /\ thread_state[t] = "encoding"
    /\ thread_context[t] /= NULL
    /\ thread_state' = [thread_state EXCEPT ![t] = "destroying"]
    /\ UNCHANGED <<thread_context, context_registry, null_deref_count, race_witnessed>>

(* Thread destroys its context (invalidates shared registry). *)
DestroyContext(t) ==
    /\ thread_state[t] = "destroying"
    /\ LET c == thread_context[t] IN
        /\ c /= NULL
        /\ context_registry' = [context_registry EXCEPT ![c] = "invalid"]
        /\ thread_context' = [thread_context EXCEPT ![t] = NULL]
    /\ thread_state' = [thread_state EXCEPT ![t] = "idle"]
    /\ UNCHANGED <<null_deref_count, race_witnessed>>

(* -------------------------------------------------------------------------- *)
(* Next State                                                                 *)
(* -------------------------------------------------------------------------- *)

Next ==
    \/ \E t \in Threads: CreateContext(t)
    \/ \E from \in Threads, to \in Threads: MigrateContext(from, to)
    \/ \E t \in Threads: UseContext(t)
    \/ \E t \in Threads: StartDestroy(t)
    \/ \E t \in Threads: DestroyContext(t)

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

(* -------------------------------------------------------------------------- *)
(* Properties                                                                 *)
(* -------------------------------------------------------------------------- *)

NoNullDereference == null_deref_count = 0
RaceCanOccur == <>(race_witnessed = TRUE)

TypeOK ==
    /\ thread_context \in [Threads -> ContextIds \cup {NULL}]
    /\ thread_state \in [Threads -> ThreadStates]
    /\ context_registry \in [ContextIds -> ContextStates]
    /\ null_deref_count \in Nat
    /\ race_witnessed \in BOOLEAN

(*
 * To run TLC:
 *   java -jar tla2tools.jar -deadlock -config AGXContextMigration.cfg AGXContextMigration.tla
 *
 * Expected: Invariant NoNullDereference is violated after migration + destroy.
 *)
=============================================================================

