---------------------------- MODULE AGXObjCRuntime_Fixed ----------------------------
(*
 * AGX ObjC Runtime Model - FIXED VERSION
 *
 * Gap 1 Fix: Correct mutex acquisition timing
 *
 * ORIGINAL BUG: The original spec acquired mutex BEFORE objc_msgSend.
 * REALITY: Mutex is acquired INSIDE the swizzled method, AFTER objc_msgSend.
 *
 * This creates a race window:
 * 1. Thread A calls [encoder setBuffer:...]
 * 2. Thread A's objc_msgSend reads isa pointer (NO MUTEX!)
 * 3. Thread B deallocates encoder (isa becomes invalid)
 * 4. Thread A crashes OR succeeds (race)
 *
 * v2.3 FIX: RetainOnCreation keeps encoder alive so deallocation is blocked.
 * But the SPEC must model the correct timing to verify this properly.
 *)

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    NumThreads,
    NumEncoders,
    RetainOnCreation   \* TRUE = v2.3 behavior

ASSUME NumThreads \in Nat /\ NumThreads > 0
ASSUME NumEncoders \in Nat /\ NumEncoders > 0
ASSUME RetainOnCreation \in BOOLEAN

VARIABLES
    (* Encoder memory state *)
    encoder_allocated,      \* Is memory allocated for this encoder?
    encoder_isa_valid,      \* Is the isa pointer valid (not corrupted)?
    encoder_creation_retained, \* Was the encoder retained on creation (v2.3)?
    encoder_ended,          \* Has endEncoding been called?

    (* v2.1/v2.3 fix state *)
    encoder_refcount,       \* Reference count from v2.1 fix

    (* Thread state - FIXED: More states to model correct timing *)
    thread_state,           \* "idle" | "dispatching" | "acquiring_mutex" | "in_method" | "ending"
    thread_target_encoder,  \* Which encoder thread is trying to access

    (* Global mutex - acquired INSIDE swizzled method *)
    mutex_owner,

    (* Crash counters *)
    pac_failures,           \* PAC failures (crashes) at objc_msgSend
    safe_dispatches         \* Successful dispatches

vars == <<encoder_allocated, encoder_isa_valid, encoder_creation_retained, encoder_ended,
          encoder_refcount, thread_state, thread_target_encoder,
          mutex_owner, pac_failures, safe_dispatches>>

Threads == 1..NumThreads
Encoders == 1..NumEncoders
NULL == 0

(* FIXED: Added "acquiring_mutex" state between dispatch and in_method *)
ThreadStates == {"idle", "dispatching", "acquiring_mutex", "in_method", "ending"}

(* -------------------------------------------------------------------------- *)
(* Initial State                                                              *)
(* -------------------------------------------------------------------------- *)

Init ==
    /\ encoder_allocated = [e \in Encoders |-> FALSE]
    /\ encoder_isa_valid = [e \in Encoders |-> FALSE]
    /\ encoder_creation_retained = [e \in Encoders |-> FALSE]
    /\ encoder_ended = [e \in Encoders |-> FALSE]
    /\ encoder_refcount = [e \in Encoders |-> 0]
    /\ thread_state = [t \in Threads |-> "idle"]
    /\ thread_target_encoder = [t \in Threads |-> NULL]
    /\ mutex_owner = NULL
    /\ pac_failures = 0
    /\ safe_dispatches = 0

(* -------------------------------------------------------------------------- *)
(* Encoder Allocation/Deallocation                                            *)
(* -------------------------------------------------------------------------- *)

AllocateEncoder(e) ==
    /\ encoder_allocated[e] = FALSE
    /\ encoder_allocated' = [encoder_allocated EXCEPT ![e] = TRUE]
    /\ encoder_isa_valid' = [encoder_isa_valid EXCEPT ![e] = TRUE]
    /\ encoder_creation_retained' = [encoder_creation_retained EXCEPT ![e] = RetainOnCreation]
    /\ encoder_ended' = [encoder_ended EXCEPT ![e] = FALSE]
    /\ encoder_refcount' = [encoder_refcount EXCEPT ![e] = IF RetainOnCreation THEN 1 ELSE 0]
    /\ UNCHANGED <<thread_state, thread_target_encoder, mutex_owner, pac_failures, safe_dispatches>>

(* System attempts to deallocate an encoder *)
(* v2.3 (RetainOnCreation=TRUE) blocks this if creation_retained or refcount > 0 *)
TryDeallocateEncoder(e) ==
    /\ encoder_allocated[e] = TRUE
    /\ encoder_refcount[e] = 0              \* No refs
    /\ encoder_creation_retained[e] = FALSE \* Not retained from creation
    (* Deallocation proceeds - isa becomes INVALID *)
    /\ encoder_allocated' = [encoder_allocated EXCEPT ![e] = FALSE]
    /\ encoder_isa_valid' = [encoder_isa_valid EXCEPT ![e] = FALSE]
    /\ encoder_ended' = [encoder_ended EXCEPT ![e] = FALSE]
    /\ UNCHANGED <<encoder_creation_retained, encoder_refcount, thread_state,
                  thread_target_encoder, mutex_owner, pac_failures, safe_dispatches>>

(* -------------------------------------------------------------------------- *)
(* Thread Method Call - FIXED: Correct timing                                 *)
(* -------------------------------------------------------------------------- *)

(*
 * STEP 1: Thread starts a method call [encoder setBuffer:...]
 * CRITICAL FIX: NO MUTEX ACQUIRED HERE!
 * Thread just records intent and enters "dispatching" state.
 *)
StartMethodCall(t, e) ==
    /\ thread_state[t] = "idle"
    /\ encoder_allocated[e] = TRUE
    /\ encoder_ended[e] = FALSE
    (* FIXED: NO mutex acquisition - just record target *)
    /\ thread_state' = [thread_state EXCEPT ![t] = "dispatching"]
    /\ thread_target_encoder' = [thread_target_encoder EXCEPT ![t] = e]
    /\ UNCHANGED <<encoder_allocated, encoder_isa_valid, encoder_creation_retained,
                   encoder_ended, encoder_refcount, mutex_owner, pac_failures, safe_dispatches>>

(*
 * STEP 2: objc_msgSend dispatch - reads isa pointer
 * CRITICAL: This runs WITHOUT any mutex protection!
 * If isa is invalid -> PAC failure CRASH
 * If isa is valid -> proceed to acquire mutex
 *)
ObjcMsgSendDispatch(t) ==
    /\ thread_state[t] = "dispatching"
    /\ LET e == thread_target_encoder[t] IN
        IF encoder_isa_valid[e] = FALSE
        THEN
            (* ISA IS INVALID - PAC FAILURE CRASH! *)
            /\ pac_failures' = pac_failures + 1
            /\ thread_state' = [thread_state EXCEPT ![t] = "idle"]
            /\ thread_target_encoder' = [thread_target_encoder EXCEPT ![t] = NULL]
            /\ UNCHANGED <<encoder_allocated, encoder_isa_valid, encoder_creation_retained,
                          encoder_ended, encoder_refcount, mutex_owner, safe_dispatches>>
        ELSE
            (* ISA valid - dispatch succeeds, now need to acquire mutex *)
            /\ safe_dispatches' = safe_dispatches + 1
            /\ thread_state' = [thread_state EXCEPT ![t] = "acquiring_mutex"]
            /\ UNCHANGED <<encoder_allocated, encoder_isa_valid, encoder_creation_retained,
                          encoder_ended, encoder_refcount, thread_target_encoder,
                          mutex_owner, pac_failures>>

(*
 * STEP 3: Acquire mutex INSIDE swizzled method
 * FIXED: This happens AFTER objc_msgSend succeeds
 *)
AcquireMutex(t) ==
    /\ thread_state[t] = "acquiring_mutex"
    /\ mutex_owner = NULL
    /\ mutex_owner' = t
    /\ thread_state' = [thread_state EXCEPT ![t] = "in_method"]
    /\ UNCHANGED <<encoder_allocated, encoder_isa_valid, encoder_creation_retained,
                   encoder_ended, encoder_refcount, thread_target_encoder,
                   pac_failures, safe_dispatches>>

(*
 * STEP 4: Thread finishes method, releases mutex
 *)
FinishMethod(t) ==
    /\ thread_state[t] = "in_method"
    /\ mutex_owner = t
    /\ mutex_owner' = NULL
    /\ thread_state' = [thread_state EXCEPT ![t] = "idle"]
    /\ thread_target_encoder' = [thread_target_encoder EXCEPT ![t] = NULL]
    /\ UNCHANGED <<encoder_allocated, encoder_isa_valid, encoder_creation_retained,
                   encoder_ended, encoder_refcount, pac_failures, safe_dispatches>>

(*
 * EndEncoding: Thread calls endEncoding to finish using encoder
 * This releases the creation-time retain and marks encoder as ended.
 *
 * IMPORTANT: endEncoding ALSO goes through objc_msgSend dispatch!
 * So it must acquire mutex (which serializes with other encoder methods).
 *
 * CRITICAL SAFETY CONSTRAINT: endEncoding can only be called when NO other
 * thread is in the "dispatching" state for this encoder. This is because
 * endEncoding releases the creation retain, which could allow deallocation
 * while another thread is mid-dispatch.
 *
 * In the real implementation, this is enforced by:
 * 1. PyTorch's thread ownership model - only the creating thread calls endEncoding
 * 2. The encoder is typically used sequentially within a single command buffer
 *)
NoOtherThreadDispatching(e) ==
    \A t2 \in Threads:
        ~(thread_state[t2] = "dispatching" /\ thread_target_encoder[t2] = e)

CallEndEncoding(t, e) ==
    /\ thread_state[t] = "idle"
    /\ mutex_owner = NULL
    /\ encoder_allocated[e] = TRUE
    /\ encoder_creation_retained[e] = TRUE
    /\ encoder_ended[e] = FALSE
    /\ NoOtherThreadDispatching(e)  \* CRITICAL: No other thread dispatching to this encoder
    /\ encoder_creation_retained' = [encoder_creation_retained EXCEPT ![e] = FALSE]
    /\ encoder_refcount' = [encoder_refcount EXCEPT ![e] = @ - 1]
    /\ encoder_ended' = [encoder_ended EXCEPT ![e] = TRUE]
    /\ UNCHANGED <<encoder_allocated, encoder_isa_valid, thread_state,
                  thread_target_encoder, mutex_owner, pac_failures, safe_dispatches>>

(* -------------------------------------------------------------------------- *)
(* Next State                                                                 *)
(* -------------------------------------------------------------------------- *)

Next ==
    \/ \E e \in Encoders: AllocateEncoder(e)
    \/ \E e \in Encoders: TryDeallocateEncoder(e)
    \/ \E t \in Threads, e \in Encoders: StartMethodCall(t, e)
    \/ \E t \in Threads: ObjcMsgSendDispatch(t)
    \/ \E t \in Threads: AcquireMutex(t)
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
    /\ thread_state \in [Threads -> ThreadStates]
    /\ thread_target_encoder \in [Threads -> Encoders \cup {NULL}]
    /\ mutex_owner \in Threads \cup {NULL}
    /\ pac_failures \in Nat
    /\ safe_dispatches \in Nat

(*
 * CRITICAL INVARIANT: No PAC failures
 *
 * With RetainOnCreation = FALSE: EXPECT VIOLATION (race exists)
 * With RetainOnCreation = TRUE: EXPECT TO HOLD (v2.3 prevents deallocation)
 *)
NoPacFailures == pac_failures = 0

(*
 * v2.3 SAFETY: Creation-retained encoders cannot be deallocated.
 * This is what prevents the race - TryDeallocateEncoder is blocked.
 *)
CreationRetainedImpliesValid ==
    \A e \in Encoders:
        encoder_creation_retained[e] = TRUE => encoder_isa_valid[e] = TRUE

(*
 * Thread in dispatching state has a target.
 *)
DispatchingHasTarget ==
    \A t \in Threads:
        thread_state[t] = "dispatching" => thread_target_encoder[t] /= NULL

(* -------------------------------------------------------------------------- *)
(* Key Insight from Fixed Model                                               *)
(* -------------------------------------------------------------------------- *)
(*
 * THE RACE WINDOW (with RetainOnCreation = FALSE):
 *
 * Time  Thread A                     Thread B
 * ----  --------------------------   --------------------------
 * T1    StartMethodCall(A, E1)       (idle)
 *       state[A] = "dispatching"
 *       target[A] = E1
 *
 * T2    (blocked on dispatch)        TryDeallocateEncoder(E1)
 *                                    encoder_isa_valid[E1] = FALSE
 *
 * T3    ObjcMsgSendDispatch(A)       (done)
 *       -> CRASH! isa_valid = FALSE
 *
 * THE FIX (with RetainOnCreation = TRUE):
 *
 * T1    StartMethodCall(A, E1)
 *       (same as before)
 *
 * T2    (blocked on dispatch)        TryDeallocateEncoder(E1)
 *                                    -> BLOCKED! creation_retained = TRUE
 *
 * T3    ObjcMsgSendDispatch(A)
 *       -> SUCCESS! isa_valid = TRUE
 *
 * The v2.3 fix works by BLOCKING deallocation, not by protecting the dispatch.
 *)

=============================================================================
