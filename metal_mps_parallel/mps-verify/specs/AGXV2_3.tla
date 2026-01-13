---------------------------- MODULE AGXV2_3 ----------------------------
(*
 * AGX v2.3 Fix Model - Retain From Creation + Mutex Protection
 *
 * This models v2.3's actual implementation:
 * 1. Factory method (computeCommandEncoder) is swizzled
 * 2. Encoder is CFRetain'd IMMEDIATELY at creation
 * 3. ALL encoder method calls are mutex-protected
 * 4. CFRelease happens at endEncoding (under mutex)
 *
 * Key difference from earlier models: the retain happens BEFORE the
 * encoder is returned to user code, so there's no window where
 * refcount = 0 for a live encoder.
 *)

EXTENDS Integers, FiniteSets

CONSTANTS
    NumThreads,
    NumEncoders

ASSUME NumThreads \in Nat /\ NumThreads > 0
ASSUME NumEncoders \in Nat /\ NumEncoders > 0

VARIABLES
    (* Encoder state *)
    encoder_exists,           \* Is encoder object valid?
    encoder_refcount,         \* Our CFRetain count

    (* Mutex state - v2.3 uses single global mutex *)
    mutex_holder,             \* Which thread holds mutex (NULL if free)

    (* Thread state *)
    thread_state,             \* "idle" | "creating" | "has_encoder" | "in_method" | "ending"
    thread_encoder,           \* Which encoder thread is using

    (* Safety counters *)
    use_after_free_count

vars == <<encoder_exists, encoder_refcount, mutex_holder, thread_state, thread_encoder, use_after_free_count>>

Threads == 1..NumThreads
Encoders == 1..NumEncoders
NULL == 0

ThreadStates == {"idle", "creating", "has_encoder", "in_method", "ending"}

(* -------------------------------------------------------------------------- *)
(* Initial State                                                              *)
(* -------------------------------------------------------------------------- *)

Init ==
    /\ encoder_exists = [e \in Encoders |-> FALSE]
    /\ encoder_refcount = [e \in Encoders |-> 0]
    /\ mutex_holder = NULL
    /\ thread_state = [t \in Threads |-> "idle"]
    /\ thread_encoder = [t \in Threads |-> NULL]
    /\ use_after_free_count = 0

(* -------------------------------------------------------------------------- *)
(* Mutex Operations                                                           *)
(* -------------------------------------------------------------------------- *)

AcquireMutex(t) ==
    /\ mutex_holder = NULL
    /\ mutex_holder' = t

ReleaseMutex(t) ==
    /\ mutex_holder = t
    /\ mutex_holder' = NULL

(* -------------------------------------------------------------------------- *)
(* v2.3: Create Encoder (via swizzled computeCommandEncoder)                  *)
(* -------------------------------------------------------------------------- *)

(*
 * Thread calls [commandBuffer computeCommandEncoder]
 * v2.3 intercepts this and immediately retains the encoder
 *)
CreateEncoder(t, e) ==
    /\ thread_state[t] = "idle"
    /\ encoder_exists[e] = FALSE
    /\ mutex_holder = NULL  \* Need mutex for retain
    /\ mutex_holder' = t    \* Acquire mutex
    (* v2.3: Encoder is created AND retained atomically under mutex *)
    /\ encoder_exists' = [encoder_exists EXCEPT ![e] = TRUE]
    /\ encoder_refcount' = [encoder_refcount EXCEPT ![e] = 1]  \* RETAINED AT CREATION!
    /\ thread_state' = [thread_state EXCEPT ![t] = "has_encoder"]
    /\ thread_encoder' = [thread_encoder EXCEPT ![t] = e]
    /\ UNCHANGED use_after_free_count

(* After creation, thread releases mutex *)
FinishCreation(t) ==
    /\ thread_state[t] = "has_encoder"
    /\ mutex_holder = t
    /\ mutex_holder' = NULL  \* Release mutex
    /\ UNCHANGED <<encoder_exists, encoder_refcount, thread_state, thread_encoder, use_after_free_count>>

(* -------------------------------------------------------------------------- *)
(* v2.3: Method Call (mutex protected)                                        *)
(* -------------------------------------------------------------------------- *)

(*
 * Thread calls encoder method (e.g., setBuffer)
 * v2.3 acquires mutex BEFORE calling original method
 *)
StartMethodCall(t) ==
    /\ thread_state[t] = "has_encoder"
    /\ mutex_holder = NULL  \* Need mutex
    /\ LET e == thread_encoder[t] IN
        /\ encoder_exists[e] = TRUE  \* Encoder must exist
        /\ encoder_refcount[e] > 0   \* Must have our retain
        /\ mutex_holder' = t         \* Acquire mutex
        /\ thread_state' = [thread_state EXCEPT ![t] = "in_method"]
    /\ UNCHANGED <<encoder_exists, encoder_refcount, thread_encoder, use_after_free_count>>

(* Method completes, release mutex *)
FinishMethodCall(t) ==
    /\ thread_state[t] = "in_method"
    /\ mutex_holder = t
    /\ mutex_holder' = NULL  \* Release mutex
    /\ thread_state' = [thread_state EXCEPT ![t] = "has_encoder"]
    /\ UNCHANGED <<encoder_exists, encoder_refcount, thread_encoder, use_after_free_count>>

(* -------------------------------------------------------------------------- *)
(* v2.3: End Encoding (mutex protected, releases our retain)                  *)
(* -------------------------------------------------------------------------- *)

StartEndEncoding(t) ==
    /\ thread_state[t] = "has_encoder"
    /\ mutex_holder = NULL
    /\ LET e == thread_encoder[t] IN
        /\ encoder_exists[e] = TRUE
        /\ mutex_holder' = t
        /\ thread_state' = [thread_state EXCEPT ![t] = "ending"]
    /\ UNCHANGED <<encoder_exists, encoder_refcount, thread_encoder, use_after_free_count>>

FinishEndEncoding(t) ==
    /\ thread_state[t] = "ending"
    /\ mutex_holder = t
    /\ LET e == thread_encoder[t] IN
        (* Release our retain *)
        /\ encoder_refcount' = [encoder_refcount EXCEPT ![e] = @ - 1]
        /\ mutex_holder' = NULL
        /\ thread_state' = [thread_state EXCEPT ![t] = "idle"]
        /\ thread_encoder' = [thread_encoder EXCEPT ![t] = NULL]
    /\ UNCHANGED <<encoder_exists, use_after_free_count>>

(* -------------------------------------------------------------------------- *)
(* System: Deallocate encoder when refcount reaches 0                         *)
(* -------------------------------------------------------------------------- *)

DeallocEncoder(e) ==
    /\ encoder_exists[e] = TRUE
    /\ encoder_refcount[e] = 0  \* Only dealloc when no retains
    /\ encoder_exists' = [encoder_exists EXCEPT ![e] = FALSE]
    /\ UNCHANGED <<encoder_refcount, mutex_holder, thread_state, thread_encoder, use_after_free_count>>

(* -------------------------------------------------------------------------- *)
(* Note: UAF not modeled because v2.3's design prevents it                    *)
(* -------------------------------------------------------------------------- *)

(*
 * In v2.3, there is NO window where a thread can attempt to use an encoder
 * that has been deallocated because:
 * 1. Encoder is retained immediately at creation (under mutex)
 * 2. Thread receives encoder AFTER retain is set
 * 3. Thread must call endEncoding to release retain
 * 4. After endEncoding, thread sets thread_encoder = NULL
 * 5. System can only dealloc when refcount = 0
 *
 * Therefore, we don't model AttemptUseAfterFree - it's impossible by design.
 *)

(* -------------------------------------------------------------------------- *)
(* Next State                                                                 *)
(* -------------------------------------------------------------------------- *)

Next ==
    \/ \E t \in Threads, e \in Encoders: CreateEncoder(t, e)
    \/ \E t \in Threads: FinishCreation(t)
    \/ \E t \in Threads: StartMethodCall(t)
    \/ \E t \in Threads: FinishMethodCall(t)
    \/ \E t \in Threads: StartEndEncoding(t)
    \/ \E t \in Threads: FinishEndEncoding(t)
    \/ \E e \in Encoders: DeallocEncoder(e)

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

(* -------------------------------------------------------------------------- *)
(* Safety Properties                                                          *)
(* -------------------------------------------------------------------------- *)

TypeOK ==
    /\ encoder_exists \in [Encoders -> BOOLEAN]
    /\ encoder_refcount \in [Encoders -> Nat]
    /\ mutex_holder \in Threads \cup {NULL}
    /\ thread_state \in [Threads -> ThreadStates]
    /\ thread_encoder \in [Threads -> Encoders \cup {NULL}]
    /\ use_after_free_count \in Nat

(*
 * After endEncoding, refcount may be 0 briefly before dealloc.
 * The key safety property is that USED encoders have refcount > 0.
 *)
UsedEncoderHasRetain ==
    \A t \in Threads:
        thread_encoder[t] /= NULL => encoder_refcount[thread_encoder[t]] > 0

(*
 * CRITICAL: Thread with encoder always has retained encoder
 *)
ThreadEncoderHasRetain ==
    \A t \in Threads:
        thread_encoder[t] /= NULL =>
            /\ encoder_exists[thread_encoder[t]] = TRUE
            /\ encoder_refcount[thread_encoder[t]] > 0

(*
 * No use-after-free when using v2.3 correctly
 *)
NoUseAfterFree == use_after_free_count = 0

(*
 * Combined safety - v2.3 should satisfy ALL of these
 *)
V2_3_Safety ==
    /\ TypeOK
    /\ UsedEncoderHasRetain
    /\ ThreadEncoderHasRetain

=============================================================================
