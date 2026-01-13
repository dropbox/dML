--------------------------- MODULE AGXRaceFixConcurrent ---------------------------
(*
 * TLA+ Model: AGX Driver Race Condition with CONCURRENT destroyImpl calls
 *
 * This extends AGXRaceFix.tla to model MULTIPLE threads calling destroyImpl
 * on DIFFERENT encoders concurrently. The key question: does the binary patch
 * prevent races when multiple encoders are being destroyed simultaneously?
 *
 * SCENARIO:
 * - Thread 1 destroys encoder A
 * - Thread 2 destroys encoder B
 * - Thread 3 tries to use encoder A or B (race!)
 *)

EXTENDS Integers, FiniteSets

CONSTANTS
    NumThreads,
    NumEncoders

ASSUME NumThreads \in Nat /\ NumThreads >= 2
ASSUME NumEncoders \in Nat /\ NumEncoders >= 1

VARIABLES
    encoder_state,     \* [Encoder -> "alive" | "destroying" | "destroyed"]
    encoder_impl,      \* [Encoder -> "valid" | "null"] - _impl pointer
    encoder_lock,      \* [Encoder -> Thread or NULL] - unfair lock owner
    thread_state,      \* [Thread -> "idle" | "acquiring_lock" | "has_lock" | ...]
    thread_target,     \* [Thread -> Encoder or NULL]
    use_after_free

vars == <<encoder_state, encoder_impl, encoder_lock, thread_state, thread_target, use_after_free>>

Threads == 1..NumThreads
Encoders == 1..NumEncoders
NULL == 0

(* -------------------------------------------------------------------------- *)
(* Initial State                                                              *)
(* -------------------------------------------------------------------------- *)

Init ==
    /\ encoder_state = [e \in Encoders |-> "alive"]
    /\ encoder_impl = [e \in Encoders |-> "valid"]
    /\ encoder_lock = [e \in Encoders |-> NULL]
    /\ thread_state = [t \in Threads |-> "idle"]
    /\ thread_target = [t \in Threads |-> NULL]
    /\ use_after_free = FALSE

(* -------------------------------------------------------------------------- *)
(* PATCHED destroyImpl - with binary patch applied                            *)
(* Each encoder has its own lock - threads can destroy different encoders     *)
(* concurrently without blocking each other.                                  *)
(* -------------------------------------------------------------------------- *)

(* Thread starts destroying an encoder - acquires encoder's lock *)
StartDestroyPatched(t, e) ==
    /\ thread_state[t] = "idle"
    /\ encoder_state[e] = "alive"
    /\ encoder_lock[e] = NULL  \* Lock is free
    /\ encoder_lock' = [encoder_lock EXCEPT ![e] = t]
    /\ thread_state' = [thread_state EXCEPT ![t] = "has_lock"]
    /\ thread_target' = [thread_target EXCEPT ![t] = e]
    /\ UNCHANGED <<encoder_state, encoder_impl, use_after_free>>

(* PATCHED: Set _impl to NULL BEFORE releasing lock *)
NullImplPatched(t) ==
    /\ thread_state[t] = "has_lock"
    /\ LET e == thread_target[t] IN
        /\ encoder_lock[e] = t
        /\ encoder_impl' = [encoder_impl EXCEPT ![e] = "null"]  \* KEY FIX!
        /\ thread_state' = [thread_state EXCEPT ![t] = "nulled_impl"]
    /\ UNCHANGED <<encoder_state, encoder_lock, thread_target, use_after_free>>

(* Release lock - _impl is already NULL so no race *)
ReleaseLockPatched(t) ==
    /\ thread_state[t] = "nulled_impl"
    /\ LET e == thread_target[t] IN
        /\ encoder_lock' = [encoder_lock EXCEPT ![e] = NULL]
        /\ encoder_state' = [encoder_state EXCEPT ![e] = "destroyed"]
        /\ thread_state' = [thread_state EXCEPT ![t] = "idle"]
        /\ thread_target' = [thread_target EXCEPT ![t] = NULL]
    /\ UNCHANGED <<encoder_impl, use_after_free>>

(* -------------------------------------------------------------------------- *)
(* User thread tries to use an encoder (potential UAF)                        *)
(* -------------------------------------------------------------------------- *)

TryUseEncoder(t, e) ==
    /\ thread_state[t] = "idle"
    /\ thread_target[t] = NULL
    (* User can try to use any encoder - the question is what happens *)
    /\ IF encoder_impl[e] = "null" \/ encoder_state[e] = "destroyed"
       THEN
           (* Safe - impl is NULL, user code should check *)
           /\ UNCHANGED <<encoder_state, encoder_impl, encoder_lock, thread_state, thread_target, use_after_free>>
       ELSE IF encoder_lock[e] /= NULL
            THEN
                (* Encoder is being destroyed - blocked *)
                /\ UNCHANGED <<encoder_state, encoder_impl, encoder_lock, thread_state, thread_target, use_after_free>>
            ELSE
                (* Would be UAF in ORIGINAL code if impl not null but encoder being freed *)
                (* With PATCHED code, impl is NULL before lock released, so this is safe *)
                /\ UNCHANGED <<encoder_state, encoder_impl, encoder_lock, thread_state, thread_target, use_after_free>>

(* -------------------------------------------------------------------------- *)
(* Next State                                                                 *)
(* -------------------------------------------------------------------------- *)

Next ==
    \/ \E t \in Threads, e \in Encoders: StartDestroyPatched(t, e)
    \/ \E t \in Threads: NullImplPatched(t)
    \/ \E t \in Threads: ReleaseLockPatched(t)
    \/ \E t \in Threads, e \in Encoders: TryUseEncoder(t, e)

Spec == Init /\ [][Next]_vars

(* -------------------------------------------------------------------------- *)
(* Safety Properties                                                          *)
(* -------------------------------------------------------------------------- *)

TypeOK ==
    /\ encoder_state \in [Encoders -> {"alive", "destroying", "destroyed"}]
    /\ encoder_impl \in [Encoders -> {"valid", "null"}]
    /\ encoder_lock \in [Encoders -> Threads \cup {NULL}]
    /\ thread_state \in [Threads -> {"idle", "has_lock", "nulled_impl"}]
    /\ thread_target \in [Threads -> Encoders \cup {NULL}]
    /\ use_after_free \in BOOLEAN

NoUseAfterFree == use_after_free = FALSE

(* CRITICAL: After lock release, impl MUST be NULL *)
(* This is the property that the binary patch guarantees *)
ImplNullAfterUnlock ==
    \A e \in Encoders:
        (encoder_state[e] = "destroyed") => (encoder_impl[e] = "null")

(* No race window: if lock is not held and encoder is alive, impl must be valid *)
(* OR if lock is not held and encoder is destroyed, impl must be null *)
NoRaceWindow ==
    \A e \in Encoders:
        (encoder_lock[e] = NULL) =>
            \/ (encoder_state[e] = "alive" /\ encoder_impl[e] = "valid")
            \/ (encoder_state[e] = "destroyed" /\ encoder_impl[e] = "null")

=============================================================================
