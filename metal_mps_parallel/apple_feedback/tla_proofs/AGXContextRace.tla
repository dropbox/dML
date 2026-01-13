---------------------------- MODULE AGXContextRace ----------------------------
(*
 * AGX Driver Context Race Condition Model
 *
 * This TLA+ specification models the HYPOTHESIZED behavior of Apple's AGX
 * driver based on reverse engineering analysis of crash reports.
 *
 * GOAL: Formally prove that the AGX driver design (as we infer it) has a
 * race condition that causes the observed NULL pointer dereferences.
 *
 * Based on crash analysis:
 * - Crash Site 1: useResourceCommon at offset 0x5c8 (NULL context pointer)
 * - Crash Site 2: allocateUSCSpillBuffer at offset 0x184 (NULL pointer write)
 * - Crash Site 3: prepareForEnqueue at offset 0x98 (NULL pointer read)
 *
 * HYPOTHESIS: The driver maintains a shared context registry without proper
 * synchronization, allowing Thread A's context to be invalidated while
 * Thread B is still using it.
 *)

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    NumThreads,         \* Number of concurrent threads (e.g., 4)
    NumContextSlots     \* Number of slots in context registry (e.g., 4)

ASSUME NumThreads \in Nat /\ NumThreads > 0
ASSUME NumContextSlots \in Nat /\ NumContextSlots > 0

VARIABLES
    (* Per-thread state *)
    thread_context,     \* Thread -> ContextId | NULL (context assigned to thread)
    thread_state,       \* Thread -> ThreadState

    (* Shared context registry (THE BUG: no synchronization) *)
    context_registry,   \* ContextId -> {valid, invalid, NULL}
    context_owner,      \* ContextId -> Thread | NULL (who created it)

    (* Bug detection *)
    null_deref_count,   \* Number of NULL pointer dereferences detected
    race_witnessed      \* TRUE if race condition manifested

vars == <<thread_context, thread_state, context_registry, context_owner,
          null_deref_count, race_witnessed>>

Threads == 1..NumThreads
ContextIds == 1..NumContextSlots
NULL == 0

ThreadStates == {
    "idle",             \* Not using GPU
    "creating",         \* Creating context
    "encoding",         \* Actively encoding (using context)
    "destroying"        \* Destroying context
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

(* -------------------------------------------------------------------------- *)
(* Actions: Context Lifecycle (THE BUGGY DESIGN)                              *)
(* -------------------------------------------------------------------------- *)

(* Thread starts creating a context *)
StartCreateContext(t) ==
    /\ thread_state[t] = "idle"
    /\ thread_context[t] = NULL
    \* Find an invalid slot (no lock - this is the bug!)
    /\ \E c \in ContextIds:
        /\ context_registry[c] = "invalid"
        /\ thread_context' = [thread_context EXCEPT ![t] = c]
        /\ context_owner' = [context_owner EXCEPT ![c] = t]
    /\ thread_state' = [thread_state EXCEPT ![t] = "creating"]
    /\ UNCHANGED <<context_registry, null_deref_count, race_witnessed>>

(* Thread finishes creating context (marks valid) *)
FinishCreateContext(t) ==
    /\ thread_state[t] = "creating"
    /\ LET c == thread_context[t] IN
        /\ c /= NULL
        /\ context_registry' = [context_registry EXCEPT ![c] = "valid"]
    /\ thread_state' = [thread_state EXCEPT ![t] = "encoding"]
    /\ UNCHANGED <<thread_context, context_owner, null_deref_count, race_witnessed>>

(* Thread uses context for encoding (THE CRASH POINT) *)
(* If context became invalid, this is a NULL deref *)
UseContext(t) ==
    /\ thread_state[t] = "encoding"
    /\ LET c == thread_context[t] IN
        IF c /= NULL /\ context_registry[c] = "valid"
        THEN \* Normal operation
            /\ UNCHANGED <<null_deref_count, race_witnessed>>
        ELSE \* BUG: Context was invalidated by another thread!
            /\ null_deref_count' = null_deref_count + 1
            /\ race_witnessed' = TRUE
    /\ thread_state' = [thread_state EXCEPT ![t] = "destroying"]
    /\ UNCHANGED <<thread_context, context_registry, context_owner>>

(* Thread destroys its context *)
DestroyContext(t) ==
    /\ thread_state[t] = "destroying"
    /\ LET c == thread_context[t] IN
        /\ c /= NULL
        \* THE BUG: No check if another thread is using this slot!
        /\ context_registry' = [context_registry EXCEPT ![c] = "invalid"]
        /\ context_owner' = [context_owner EXCEPT ![c] = NULL]
        /\ thread_context' = [thread_context EXCEPT ![t] = NULL]
    /\ thread_state' = [thread_state EXCEPT ![t] = "idle"]
    /\ UNCHANGED <<null_deref_count, race_witnessed>>

(* Thread can also destroy someone else's context (the race!) *)
(* This models the scenario where Thread B's destroy invalidates Thread A's context *)
DestroyOtherContext(t) ==
    /\ thread_state[t] = "idle"
    \* Find a valid context owned by someone else and destroy it
    \* (Models shared registry corruption)
    /\ \E c \in ContextIds:
        /\ context_registry[c] = "valid"
        /\ context_owner[c] /= t
        /\ context_owner[c] /= NULL
        /\ context_registry' = [context_registry EXCEPT ![c] = "invalid"]
    /\ UNCHANGED <<thread_context, thread_state, context_owner,
                   null_deref_count, race_witnessed>>

(* -------------------------------------------------------------------------- *)
(* Next State                                                                 *)
(* -------------------------------------------------------------------------- *)

Next == \E t \in Threads:
    \/ StartCreateContext(t)
    \/ FinishCreateContext(t)
    \/ UseContext(t)
    \/ DestroyContext(t)
    \/ DestroyOtherContext(t)

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

(* -------------------------------------------------------------------------- *)
(* Properties: What We're Proving                                             *)
(* -------------------------------------------------------------------------- *)

(* Safety: No NULL dereferences should occur in correct design *)
NoNullDereferences == null_deref_count = 0

(* The race condition can manifest *)
RaceCanOccur == <>(race_witnessed = TRUE)

(* Type invariant *)
TypeOK ==
    /\ thread_context \in [Threads -> ContextIds \cup {NULL}]
    /\ thread_state \in [Threads -> ThreadStates]
    /\ context_registry \in [ContextIds -> ContextStates]
    /\ context_owner \in [ContextIds -> Threads \cup {NULL}]
    /\ null_deref_count \in Nat
    /\ race_witnessed \in BOOLEAN

(* -------------------------------------------------------------------------- *)
(* What This Model Proves                                                     *)
(* -------------------------------------------------------------------------- *)
(*
 * If TLC finds a violation of NoNullDereferences, it proves:
 *   "The AGX driver design (as modeled) CAN produce NULL pointer dereferences"
 *
 * The counterexample trace will show exactly HOW the race occurs:
 *   1. Thread A creates context, starts encoding
 *   2. Thread B (or registry corruption) invalidates Thread A's context
 *   3. Thread A calls UseContext with invalid context → NULL deref
 *
 * This is exactly what we observed in the crash reports!
 *)

=============================================================================
