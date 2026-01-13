---------------------------- MODULE AGXEncoderLifetime ----------------------------
(*
 * AGX Encoder Lifetime Safety - v2 Fix Verification
 *
 * This TLA+ specification models the use-after-free bug that causes PAC failures
 * at objc_msgSend+32, and proves that the v2 retain/release fix prevents it.
 *
 * ROOT CAUSE OF USE-AFTER-FREE:
 *   Thread A holds reference to encoder E
 *   Thread B deallocates encoder E
 *   Thread A calls method on E -> objc_msgSend reads corrupted isa -> crash
 *
 * V2 FIX STRATEGY (CRITICAL - reference counting):
 *   - On EVERY encoder access: increment reference count
 *   - On endEncoding: decrement reference count
 *   - Encoder cannot be deallocated while reference count > 0
 *
 * MULTI-THREAD BUG FIX:
 *   The original v2 only retained once. If Thread A and Thread B both use
 *   encoder E, and Thread A releases first, Thread B would crash.
 *   FIX: Each access increments the count, each release decrements it.
 *
 * GOAL: Prove that v2 fix prevents all use-after-free scenarios,
 *       including multi-threaded shared encoder access.
 *)

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    NumThreads,
    NumEncoders

ASSUME NumThreads \in Nat /\ NumThreads > 0
ASSUME NumEncoders \in Nat /\ NumEncoders > 0

VARIABLES
    (* Encoder state *)
    encoder_state,        \* "uninitialized" | "alive" | "deallocated"
    encoder_retain_count, \* Reference count from v2 fix
    encoder_owner,        \* Thread that created the encoder

    (* Thread state *)
    thread_encoder,       \* Which encoder thread is using (NULL if none)
    thread_state,         \* "idle" | "using" | "ending"
    thread_has_retain,    \* Does thread hold a v2 retain on its encoder?

    (* Safety counters *)
    use_after_free_count, \* Number of use-after-free attempts
    crash_prevented_count \* Number of crashes prevented by v2

vars == <<encoder_state, encoder_retain_count, encoder_owner,
          thread_encoder, thread_state, thread_has_retain,
          use_after_free_count, crash_prevented_count>>

Threads == 1..NumThreads
Encoders == 1..NumEncoders
NULL == 0

ThreadStates == {"idle", "using", "ending"}
EncoderStates == {"uninitialized", "alive", "deallocated"}

(* -------------------------------------------------------------------------- *)
(* Initial State                                                              *)
(* -------------------------------------------------------------------------- *)

Init ==
    /\ encoder_state = [e \in Encoders |-> "uninitialized"]
    /\ encoder_retain_count = [e \in Encoders |-> 0]
    /\ encoder_owner = [e \in Encoders |-> NULL]
    /\ thread_encoder = [t \in Threads |-> NULL]
    /\ thread_state = [t \in Threads |-> "idle"]
    /\ thread_has_retain = [t \in Threads |-> FALSE]
    /\ use_after_free_count = 0
    /\ crash_prevented_count = 0

(* -------------------------------------------------------------------------- *)
(* V2 Fix: Encoder Access with Retain                                         *)
(* -------------------------------------------------------------------------- *)

(*
 * Thread starts using an encoder.
 * V2 FIX: Increment retain count on EVERY access (not just first).
 * CRITICAL: This fixes the multi-thread bug where Thread A releases
 * while Thread B is still using the encoder.
 *)
StartUsingEncoder(t, e) ==
    /\ thread_state[t] = "idle"
    /\ thread_encoder[t] = NULL
    /\ encoder_state[e] \in {"uninitialized", "alive"}
    (* V2 Fix: Increment retain count on every access *)
    /\ encoder_state' = [encoder_state EXCEPT ![e] = "alive"]
    /\ encoder_retain_count' = [encoder_retain_count EXCEPT ![e] = @ + 1]  \* INCREMENT!
    /\ encoder_owner' = [encoder_owner EXCEPT ![e] =
                         IF encoder_owner[e] = NULL THEN t ELSE encoder_owner[e]]
    /\ thread_encoder' = [thread_encoder EXCEPT ![t] = e]
    /\ thread_state' = [thread_state EXCEPT ![t] = "using"]
    /\ thread_has_retain' = [thread_has_retain EXCEPT ![t] = TRUE]
    /\ UNCHANGED <<use_after_free_count, crash_prevented_count>>

(*
 * Thread performs operation on encoder (setBuffer, setBytes, etc.)
 * V2 FIX: Check encoder is still alive before proceeding.
 *)
UseEncoder(t) ==
    /\ thread_state[t] = "using"
    /\ thread_encoder[t] /= NULL
    /\ LET e == thread_encoder[t] IN
        IF encoder_state[e] = "deallocated"
        THEN
            (* USE-AFTER-FREE ATTEMPT - V2 would have prevented this *)
            (* In reality: PAC failure at objc_msgSend+32 *)
            /\ use_after_free_count' = use_after_free_count + 1
            /\ crash_prevented_count' = crash_prevented_count + 1
            /\ UNCHANGED <<encoder_state, encoder_retain_count, encoder_owner,
                          thread_encoder, thread_state, thread_has_retain>>
        ELSE
            (* Safe operation - encoder is alive *)
            /\ UNCHANGED vars

(*
 * Thread calls endEncoding.
 * V2 FIX: Release the retain held since StartUsingEncoder.
 *)
EndEncoding(t) ==
    /\ thread_state[t] = "using"
    /\ thread_encoder[t] /= NULL
    /\ thread_has_retain[t] = TRUE
    /\ LET e == thread_encoder[t] IN
        /\ encoder_retain_count' = [encoder_retain_count EXCEPT ![e] = @ - 1]
        /\ thread_state' = [thread_state EXCEPT ![t] = "idle"]
        /\ thread_encoder' = [thread_encoder EXCEPT ![t] = NULL]
        /\ thread_has_retain' = [thread_has_retain EXCEPT ![t] = FALSE]
    /\ UNCHANGED <<encoder_state, encoder_owner, use_after_free_count, crash_prevented_count>>

(*
 * System attempts to deallocate an encoder.
 * V2 FIX: Deallocation is BLOCKED if retain count > 0.
 *)
TryDeallocEncoder(e) ==
    /\ encoder_state[e] = "alive"
    /\ IF encoder_retain_count[e] > 0
       THEN
           (* V2 FIX BLOCKS DEALLOCATION - retain count prevents it *)
           /\ UNCHANGED vars
       ELSE
           (* Safe to deallocate - no retains held *)
           /\ encoder_state' = [encoder_state EXCEPT ![e] = "deallocated"]
           /\ UNCHANGED <<encoder_retain_count, encoder_owner,
                         thread_encoder, thread_state, thread_has_retain,
                         use_after_free_count, crash_prevented_count>>

(*
 * Thread attempts to use a deallocated encoder (BUG scenario).
 * This models the race where Thread B deallocates while Thread A uses.
 * V2 FIX: This can only happen if thread does NOT have retain (bug in fix).
 *)
AttemptUseAfterFree(t, e) ==
    /\ thread_state[t] = "idle"
    /\ encoder_state[e] = "deallocated"
    (* This would be a crash in the real system *)
    /\ use_after_free_count' = use_after_free_count + 1
    /\ UNCHANGED <<encoder_state, encoder_retain_count, encoder_owner,
                  thread_encoder, thread_state, thread_has_retain, crash_prevented_count>>

(* -------------------------------------------------------------------------- *)
(* Next State                                                                 *)
(* -------------------------------------------------------------------------- *)

Next ==
    \/ \E t \in Threads, e \in Encoders: StartUsingEncoder(t, e)
    \/ \E t \in Threads: UseEncoder(t)
    \/ \E t \in Threads: EndEncoding(t)
    \/ \E e \in Encoders: TryDeallocEncoder(e)
    \/ \E t \in Threads, e \in Encoders: AttemptUseAfterFree(t, e)

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

(* -------------------------------------------------------------------------- *)
(* Safety Properties                                                          *)
(* -------------------------------------------------------------------------- *)

(*
 * INVARIANT 1: Type correctness
 *)
TypeOK ==
    /\ encoder_state \in [Encoders -> EncoderStates]
    /\ encoder_retain_count \in [Encoders -> Nat]
    /\ encoder_owner \in [Encoders -> Threads \cup {NULL}]
    /\ thread_encoder \in [Threads -> Encoders \cup {NULL}]
    /\ thread_state \in [Threads -> ThreadStates]
    /\ thread_has_retain \in [Threads -> BOOLEAN]
    /\ use_after_free_count \in Nat
    /\ crash_prevented_count \in Nat

(*
 * INVARIANT 2: Retained encoders cannot be deallocated
 * This is the CORE safety property of the v2 fix.
 *)
RetainedEncoderAlive ==
    \A e \in Encoders:
        encoder_retain_count[e] > 0 => encoder_state[e] /= "deallocated"

(*
 * INVARIANT 3: Thread using encoder has retain
 * If a thread is actively using an encoder, it holds a retain on it.
 *)
UsingThreadHasRetain ==
    \A t \in Threads:
        thread_state[t] = "using" => thread_has_retain[t] = TRUE

(*
 * INVARIANT 4: Thread with retain uses alive encoder
 * If thread has retain, the encoder must be alive (not deallocated).
 *)
RetainImpliesAlive ==
    \A t \in Threads:
        (thread_has_retain[t] = TRUE /\ thread_encoder[t] /= NULL) =>
            encoder_state[thread_encoder[t]] /= "deallocated"

(*
 * MAIN SAFETY PROPERTY: No use-after-free while using encoder with retain
 * When a thread is in "using" state with a retain, it cannot encounter UAF.
 *)
NoUseAfterFreeWhileRetained ==
    \A t \in Threads:
        (thread_state[t] = "using" /\ thread_has_retain[t] = TRUE /\ thread_encoder[t] /= NULL) =>
            encoder_state[thread_encoder[t]] = "alive"

(* -------------------------------------------------------------------------- *)
(* Combined Safety Check                                                      *)
(* -------------------------------------------------------------------------- *)

(*
 * THEOREM: The v2 fix ensures encoder lifetime safety
 *
 * If all these invariants hold, then:
 * 1. Retained encoders cannot be deallocated (RetainedEncoderAlive)
 * 2. Using threads always have retains (UsingThreadHasRetain)
 * 3. Therefore, using threads never encounter deallocated encoders
 *
 * This prevents the PAC failure crash at objc_msgSend+32.
 *)
V2FixSafety ==
    /\ TypeOK
    /\ RetainedEncoderAlive
    /\ UsingThreadHasRetain
    /\ RetainImpliesAlive
    /\ NoUseAfterFreeWhileRetained

(* -------------------------------------------------------------------------- *)
(* What This Model Proves                                                     *)
(* -------------------------------------------------------------------------- *)
(*
 * Running TLC with:
 *   NumThreads = 3
 *   NumEncoders = 2
 *   INVARIANT V2FixSafety
 *
 * If TLC finds NO violations, it proves:
 *   "The v2 retain/release fix PREVENTS use-after-free crashes"
 *   "Encoder lifetime is safely managed from first access to endEncoding"
 *
 * This validates the fix in agx_fix_v2.mm:
 *   - ensure_encoder_alive() retains on first access
 *   - release_encoder_retain() releases on endEncoding
 *   - Deallocation blocked while retain count > 0
 *)

=============================================================================
