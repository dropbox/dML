------------------------- MODULE AGXNestedEncoders ---------------------------
(*
 * AGX Nested Encoder Creation / Re-Entrant Swizzle Model
 *
 * In the swizzle-based AGX fix, each intercepted driver method takes a lock.
 * Some driver methods call other swizzled methods internally (nested calls).
 *
 * If the lock is NOT recursive, a single thread can self-deadlock:
 *   outer_method() acquires lock
 *     -> calls nested_method() which tries to acquire the same lock again
 *        -> blocks forever (lock held by the same thread)
 *
 * This model demonstrates that nested encoder creation (or any nested swizzled
 * call path) REQUIRES a recursive mutex to avoid self-deadlock.
 *
 * EXPECTED RESULT (LockIsRecursive=FALSE): TLC reaches thread_state="blocked".
 *)

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    NumThreads,         \* Number of threads (set to 1 to isolate self-deadlock)
    LockIsRecursive     \* BOOLEAN: TRUE for recursive mutex, FALSE for non-recursive

ASSUME NumThreads \in Nat /\ NumThreads > 0
ASSUME LockIsRecursive \in BOOLEAN

VARIABLES
    mutex_owner,        \* Thread holding mutex | NULL
    mutex_depth,        \* Recursion depth (0 when unlocked)
    thread_state        \* Thread -> ThreadState

vars == <<mutex_owner, mutex_depth, thread_state>>

Threads == 1..NumThreads
NULL == 0

ThreadStates == {"idle", "outer", "nested", "blocked"}

(* -------------------------------------------------------------------------- *)
(* Initial State                                                              *)
(* -------------------------------------------------------------------------- *)

Init ==
    /\ mutex_owner = NULL
    /\ mutex_depth = 0
    /\ thread_state = [t \in Threads |-> "idle"]

(* -------------------------------------------------------------------------- *)
(* Actions                                                                    *)
(* -------------------------------------------------------------------------- *)

(* Enter outer swizzled method: acquire mutex once. *)
EnterOuter(t) ==
    /\ thread_state[t] = "idle"
    /\ mutex_owner = NULL
    /\ mutex_owner' = t
    /\ mutex_depth' = 1
    /\ thread_state' = [thread_state EXCEPT ![t] = "outer"]

(* Nested swizzled call: attempt to acquire the same mutex again. *)
EnterNested(t) ==
    /\ thread_state[t] = "outer"
    /\ mutex_owner = t
    /\ mutex_depth = 1
    /\ IF LockIsRecursive
       THEN
          /\ mutex_owner' = t
          /\ mutex_depth' = 2
          /\ thread_state' = [thread_state EXCEPT ![t] = "nested"]
       ELSE
          /\ thread_state' = [thread_state EXCEPT ![t] = "blocked"]
          /\ UNCHANGED <<mutex_owner, mutex_depth>>

(* Return from nested swizzled method: release one recursion level. *)
ExitNested(t) ==
    /\ thread_state[t] = "nested"
    /\ mutex_owner = t
    /\ mutex_depth = 2
    /\ mutex_owner' = t
    /\ mutex_depth' = 1
    /\ thread_state' = [thread_state EXCEPT ![t] = "outer"]

(* Return from outer method: release mutex fully. *)
ExitOuter(t) ==
    /\ thread_state[t] = "outer"
    /\ mutex_owner = t
    /\ mutex_depth = 1
    /\ mutex_owner' = NULL
    /\ mutex_depth' = 0
    /\ thread_state' = [thread_state EXCEPT ![t] = "idle"]

(* -------------------------------------------------------------------------- *)
(* Next State                                                                 *)
(* -------------------------------------------------------------------------- *)

Next ==
    \/ \E t \in Threads: EnterOuter(t)
    \/ \E t \in Threads: EnterNested(t)
    \/ \E t \in Threads: ExitNested(t)
    \/ \E t \in Threads: ExitOuter(t)

Spec == Init /\ [][Next]_vars

(* -------------------------------------------------------------------------- *)
(* Properties                                                                 *)
(* -------------------------------------------------------------------------- *)

MutexOwnerDepthConsistent ==
    /\ (mutex_owner = NULL) <=> (mutex_depth = 0)
    /\ mutex_depth \in 0..2
    /\ mutex_owner \in Threads \cup {NULL}

NoSelfDeadlock ==
    \A t \in Threads: thread_state[t] /= "blocked"

TypeOK ==
    /\ mutex_owner \in Threads \cup {NULL}
    /\ mutex_depth \in Nat
    /\ thread_state \in [Threads -> ThreadStates]

(*
 * To run TLC:
 *   java -jar tla2tools.jar -deadlock -config AGXNestedEncoders.cfg AGXNestedEncoders.tla
 *
 * Expected (LockIsRecursive=FALSE): invariant NoSelfDeadlock is violated
 * via EnterNested() setting thread_state="blocked".
 *)
=============================================================================
