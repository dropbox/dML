---------------------------- MODULE AGXObjCRuntime ----------------------------
(*
 * AGX ObjC Runtime Model - PAC Failure at objc_msgSend
 *
 * This TLA+ specification models the ACTUAL crash mechanism:
 * 1. objc_msgSend reads the 'isa' pointer from the object
 * 2. If object is deallocated, isa is garbage/corrupted
 * 3. ARM64 PAC (Pointer Authentication Code) verification fails
 * 4. CRASH at objc_msgSend+32
 *
 * CRITICAL INSIGHT: The crash happens BEFORE the swizzled code runs!
 * objc_msgSend must successfully dispatch the message for our code to run.
 *
 * This models whether v2.1 fix prevents the crash.
 *
 * Toggle:
 *   - RetainOnCreation = FALSE : models v2.1 behavior (retain inside swizzled method) and should exhibit PAC failures
 *   - RetainOnCreation = TRUE  : models v2.3 behavior (retain on creation) and should eliminate PAC failures
 *)

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    NumThreads,
    NumEncoders,
    RetainOnCreation

ASSUME NumThreads \in Nat /\ NumThreads > 0
ASSUME NumEncoders \in Nat /\ NumEncoders > 0
ASSUME RetainOnCreation \in BOOLEAN

VARIABLES
    (* Encoder memory state *)
    encoder_allocated,      \* Is memory allocated for this encoder?
    encoder_isa_valid,      \* Is the isa pointer valid (not corrupted)?
    encoder_creation_retained, \* Was the encoder retained on creation (v2.3)?
    encoder_ended,          \* Has endEncoding been called? (post-end use is invalid)

    (* v2.1 fix state *)
    encoder_refcount,       \* Reference count from v2.1 fix
    thread_using_encoder,   \* Per-thread: set of encoders this thread is using

    (* Thread state *)
    thread_state,           \* "idle" | "dispatching" | "in_method" | "releasing"
    thread_target_encoder,  \* Which encoder thread is trying to access

    (* Global lock (v2.1 and v2.3 serialize encoder method calls) *)
    mutex_owner,            \* NULL or thread id

    (* Crash counters *)
    pac_failures,           \* Number of PAC failures (would-be crashes)
    crashes_prevented       \* Crashes prevented by v2.1

vars == <<encoder_allocated, encoder_isa_valid, encoder_creation_retained, encoder_ended, encoder_refcount,
          thread_using_encoder, thread_state, thread_target_encoder,
          mutex_owner, pac_failures, crashes_prevented>>

Threads == 1..NumThreads
Encoders == 1..NumEncoders
NULL == 0

ThreadStates == {"idle", "dispatching", "in_method", "releasing"}

(* -------------------------------------------------------------------------- *)
(* Initial State                                                              *)
(* -------------------------------------------------------------------------- *)

Init ==
    /\ encoder_allocated = [e \in Encoders |-> FALSE]
    /\ encoder_isa_valid = [e \in Encoders |-> FALSE]
    /\ encoder_creation_retained = [e \in Encoders |-> FALSE]
    /\ encoder_ended = [e \in Encoders |-> FALSE]
    /\ encoder_refcount = [e \in Encoders |-> 0]
    /\ thread_using_encoder = [t \in Threads |-> {}]
    /\ thread_state = [t \in Threads |-> "idle"]
    /\ thread_target_encoder = [t \in Threads |-> NULL]
    /\ mutex_owner = NULL
    /\ pac_failures = 0
    /\ crashes_prevented = 0

(* -------------------------------------------------------------------------- *)
(* Encoder Allocation/Deallocation                                            *)
(* -------------------------------------------------------------------------- *)

(* System allocates an encoder *)
AllocateEncoder(e) ==
    /\ encoder_allocated[e] = FALSE
    /\ encoder_allocated' = [encoder_allocated EXCEPT ![e] = TRUE]
    /\ encoder_isa_valid' = [encoder_isa_valid EXCEPT ![e] = TRUE]
    /\ encoder_creation_retained' = [encoder_creation_retained EXCEPT ![e] = RetainOnCreation]
    /\ encoder_ended' = [encoder_ended EXCEPT ![e] = FALSE]
    /\ UNCHANGED <<encoder_refcount, thread_using_encoder, thread_state,
                   thread_target_encoder, mutex_owner, pac_failures, crashes_prevented>>

(* System attempts to deallocate an encoder *)
TryDeallocateEncoder(e) ==
    /\ encoder_allocated[e] = TRUE
    /\ IF encoder_refcount[e] > 0 \/ encoder_creation_retained[e] = TRUE
       THEN
           (* v2.1 BLOCKS deallocation - refcount prevents it *)
           /\ UNCHANGED vars
       ELSE
           (* Deallocation proceeds - isa becomes INVALID *)
           /\ encoder_allocated' = [encoder_allocated EXCEPT ![e] = FALSE]
           /\ encoder_isa_valid' = [encoder_isa_valid EXCEPT ![e] = FALSE]
           /\ encoder_creation_retained' = [encoder_creation_retained EXCEPT ![e] = FALSE]
           /\ encoder_ended' = [encoder_ended EXCEPT ![e] = FALSE]
           /\ UNCHANGED <<encoder_refcount, thread_using_encoder, thread_state,
                         thread_target_encoder, mutex_owner, pac_failures, crashes_prevented>>

(* -------------------------------------------------------------------------- *)
(* Thread Method Call - Models objc_msgSend behavior                          *)
(* -------------------------------------------------------------------------- *)

(*
 * Thread starts a method call on an encoder.
 * This models: [encoder setBuffer:...]
 *
 * CRITICAL: objc_msgSend runs FIRST, before any user code!
 * If isa is invalid, PAC check fails here.
 *)
StartMethodCall(t, e) ==
    /\ mutex_owner = NULL
    /\ thread_state[t] = "idle"
    /\ encoder_allocated[e] = TRUE  \* Encoder exists in our view
    /\ encoder_ended[e] = FALSE
    /\ mutex_owner' = t
    /\ thread_state' = [thread_state EXCEPT ![t] = "dispatching"]
    /\ thread_target_encoder' = [thread_target_encoder EXCEPT ![t] = e]
    /\ UNCHANGED <<encoder_allocated, encoder_isa_valid, encoder_creation_retained, encoder_ended, encoder_refcount,
                   thread_using_encoder, pac_failures, crashes_prevented>>

(*
 * objc_msgSend dispatch - reads isa pointer
 * THIS IS WHERE PAC FAILURE HAPPENS if isa is corrupted
 *)
ObjcMsgSendDispatch(t) ==
    /\ mutex_owner = t
    /\ thread_state[t] = "dispatching"
    /\ LET e == thread_target_encoder[t] IN
        IF encoder_isa_valid[e] = FALSE
        THEN
            (* ISA IS INVALID - PAC FAILURE CRASH! *)
            /\ pac_failures' = pac_failures + 1
            /\ mutex_owner' = NULL
            /\ thread_state' = [thread_state EXCEPT ![t] = "idle"]
            /\ thread_target_encoder' = [thread_target_encoder EXCEPT ![t] = NULL]
            /\ UNCHANGED <<encoder_allocated, encoder_isa_valid, encoder_creation_retained, encoder_refcount,
                          encoder_ended, thread_using_encoder, crashes_prevented>>
        ELSE
            (* ISA valid - dispatch succeeds, enter swizzled method *)
            /\ thread_state' = [thread_state EXCEPT ![t] = "in_method"]
            /\ UNCHANGED <<encoder_allocated, encoder_isa_valid, encoder_creation_retained, encoder_refcount,
                          encoder_ended, thread_using_encoder, thread_target_encoder,
                          mutex_owner, pac_failures, crashes_prevented>>

(*
 * v2.1 swizzled method - ensure_encoder_alive
 * This runs AFTER objc_msgSend successfully dispatches
 *)
SwizzledMethodEntry(t) ==
    /\ mutex_owner = t
    /\ thread_state[t] = "in_method"
    /\ LET e == thread_target_encoder[t] IN
        (* v2.1: Only increment if this thread not already using this encoder *)
        IF e \in thread_using_encoder[t]
        THEN
            (* Already using - no refcount change (prevents memory leak) *)
            /\ UNCHANGED vars
        ELSE
            (* First access by this thread - increment refcount *)
            /\ thread_using_encoder' = [thread_using_encoder EXCEPT ![t] = @ \cup {e}]
            /\ encoder_refcount' = [encoder_refcount EXCEPT ![e] = @ + 1]
            /\ UNCHANGED <<encoder_allocated, encoder_isa_valid, encoder_creation_retained, thread_state,
                          encoder_ended, thread_target_encoder, mutex_owner, pac_failures, crashes_prevented>>

(*
 * Thread finishes method (not endEncoding, just a regular method)
 *)
FinishMethod(t) ==
    /\ mutex_owner = t
    /\ thread_state[t] = "in_method"
    /\ mutex_owner' = NULL
    /\ thread_state' = [thread_state EXCEPT ![t] = "idle"]
    /\ thread_target_encoder' = [thread_target_encoder EXCEPT ![t] = NULL]
    /\ UNCHANGED <<encoder_allocated, encoder_isa_valid, encoder_creation_retained, encoder_ended, encoder_refcount,
                  thread_using_encoder, pac_failures, crashes_prevented>>

(*
 * Thread calls endEncoding - releases its hold
 *)
CallEndEncoding(t, e) ==
    /\ mutex_owner = NULL
    /\ thread_state[t] = "idle"
    /\ e \in thread_using_encoder[t]
    /\ thread_using_encoder' = [thread_using_encoder EXCEPT ![t] = @ \ {e}]
    /\ encoder_refcount' = [encoder_refcount EXCEPT ![e] = @ - 1]
    /\ encoder_creation_retained' = [encoder_creation_retained EXCEPT ![e] = FALSE]
    /\ encoder_ended' = [encoder_ended EXCEPT ![e] = TRUE]
    /\ UNCHANGED <<encoder_allocated, encoder_isa_valid, thread_state,
                  thread_target_encoder, mutex_owner, pac_failures, crashes_prevented>>

(* -------------------------------------------------------------------------- *)
(* Next State                                                                 *)
(* -------------------------------------------------------------------------- *)

Next ==
    \/ \E e \in Encoders: AllocateEncoder(e)
    \/ \E e \in Encoders: TryDeallocateEncoder(e)
    \/ \E t \in Threads, e \in Encoders: StartMethodCall(t, e)
    \/ \E t \in Threads: ObjcMsgSendDispatch(t)
    \/ \E t \in Threads: SwizzledMethodEntry(t)
    \/ \E t \in Threads: FinishMethod(t)
    \/ \E t \in Threads, e \in Encoders: CallEndEncoding(t, e)

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

(* -------------------------------------------------------------------------- *)
(* Safety Properties                                                          *)
(* -------------------------------------------------------------------------- *)

TypeOK ==
    /\ encoder_allocated \in [Encoders -> BOOLEAN]
    /\ encoder_isa_valid \in [Encoders -> BOOLEAN]
    /\ encoder_creation_retained \in [Encoders -> BOOLEAN]
    /\ encoder_ended \in [Encoders -> BOOLEAN]
    /\ encoder_refcount \in [Encoders -> Nat]
    /\ thread_using_encoder \in [Threads -> SUBSET Encoders]
    /\ thread_state \in [Threads -> ThreadStates]
    /\ thread_target_encoder \in [Threads -> Encoders \cup {NULL}]
    /\ mutex_owner \in Threads \cup {NULL}
    /\ pac_failures \in Nat
    /\ crashes_prevented \in Nat

(*
 * CRITICAL INVARIANT: If refcount > 0, isa must be valid
 * This is what v2.1 guarantees
 *)
RefcountImpliesValidIsa ==
    \A e \in Encoders:
        encoder_refcount[e] > 0 => encoder_isa_valid[e] = TRUE

(*
 * CRITICAL INVARIANT: A thread in a method has valid target
 *)
InMethodImpliesValidTarget ==
    \A t \in Threads:
        thread_state[t] = "in_method" =>
            /\ thread_target_encoder[t] /= NULL
            /\ encoder_isa_valid[thread_target_encoder[t]] = TRUE

(*
 * MAIN SAFETY: No PAC failures when v2.1 is working correctly
 * NOTE: This may FAIL if there's a race window!
 *)
NoPacFailures == pac_failures = 0

(* -------------------------------------------------------------------------- *)
(* What This Model Reveals                                                    *)
(* -------------------------------------------------------------------------- *)
(*
 * If NoPacFailures is VIOLATED, it means there's a race window where:
 * 1. Thread A starts method call (enters "dispatching")
 * 2. Thread B deallocates encoder (isa becomes invalid)
 * 3. Thread A's objc_msgSend reads invalid isa -> CRASH
 *
 * This race exists BEFORE the v2.1 code runs! The fix must prevent
 * deallocation while ANY thread might be in the "dispatching" state.
 *)

=============================================================================
