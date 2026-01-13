---------------------------- MODULE AGXV2_9_Expanded ----------------------------
(*
 * AGX v2.9 Fix Model - Expanded for Billion-State Verification
 *
 * This model expands on AGXV2_3 by adding:
 * 1. Multiple encoder method types (simulating 77 swizzled methods)
 * 2. Multiple method calls per encoder lifetime
 * 3. More fine-grained state machine
 *
 * The goal is to explore 1+ billion states to exhaustively verify
 * that the mutex-based fix prevents all race conditions.
 *)

EXTENDS Integers, FiniteSets

CONSTANTS
    NumThreads,
    NumEncoders,
    MaxMethodCalls  \* Max method calls per encoder

ASSUME NumThreads \in Nat /\ NumThreads > 0
ASSUME NumEncoders \in Nat /\ NumEncoders > 0
ASSUME MaxMethodCalls \in Nat /\ MaxMethodCalls > 0

VARIABLES
    (* Encoder state *)
    encoder_exists,           \* Is encoder object valid?
    encoder_refcount,         \* Our CFRetain count
    encoder_ended,            \* Has endEncoding been called?

    (* Mutex state - v2.9 uses single global mutex *)
    mutex_holder,             \* Which thread holds mutex (NULL if free)

    (* Thread state - more fine-grained *)
    thread_state,
    thread_encoder,           \* Which encoder thread is using
    thread_method_count,      \* How many method calls so far
    thread_current_method,    \* Which method type (1-4)

    (* Safety counters *)
    use_after_free_count,
    ended_encoder_use_count

vars == <<encoder_exists, encoder_refcount, encoder_ended, mutex_holder,
          thread_state, thread_encoder, thread_method_count, thread_current_method,
          use_after_free_count, ended_encoder_use_count>>

Threads == 1..NumThreads
Encoders == 1..NumEncoders
MethodTypes == 1..4  \* Simulates different method categories
NULL == 0

ThreadStates == {
    "idle",
    "acquiring_create",   \* Waiting for mutex to create
    "creating",           \* Creating encoder
    "releasing_create",   \* Releasing mutex after create
    "has_encoder",        \* Has encoder, ready for ops
    "acquiring_method",   \* Waiting for mutex for method call
    "in_method_1",        \* In method type 1 (setComputePipelineState-like)
    "in_method_2",        \* In method type 2 (setBuffer-like)
    "in_method_3",        \* In method type 3 (setBytes-like)
    "in_method_4",        \* In method type 4 (dispatchThreadgroups-like)
    "releasing_method",   \* Releasing mutex after method
    "acquiring_end",      \* Waiting for mutex to end
    "ending",             \* In endEncoding
    "releasing_end"       \* Releasing mutex after end
}

(* -------------------------------------------------------------------------- *)
(* Initial State                                                              *)
(* -------------------------------------------------------------------------- *)

Init ==
    /\ encoder_exists = [e \in Encoders |-> FALSE]
    /\ encoder_refcount = [e \in Encoders |-> 0]
    /\ encoder_ended = [e \in Encoders |-> FALSE]
    /\ mutex_holder = NULL
    /\ thread_state = [t \in Threads |-> "idle"]
    /\ thread_encoder = [t \in Threads |-> NULL]
    /\ thread_method_count = [t \in Threads |-> 0]
    /\ thread_current_method = [t \in Threads |-> 0]
    /\ use_after_free_count = 0
    /\ ended_encoder_use_count = 0

(* -------------------------------------------------------------------------- *)
(* Encoder Creation                                                           *)
(* -------------------------------------------------------------------------- *)

TryAcquireForCreate(t) ==
    /\ thread_state[t] = "idle"
    /\ IF mutex_holder = NULL
       THEN /\ mutex_holder' = t
            /\ thread_state' = [thread_state EXCEPT ![t] = "creating"]
       ELSE /\ thread_state' = [thread_state EXCEPT ![t] = "acquiring_create"]
            /\ UNCHANGED mutex_holder
    /\ UNCHANGED <<encoder_exists, encoder_refcount, encoder_ended,
                   thread_encoder, thread_method_count, thread_current_method,
                   use_after_free_count, ended_encoder_use_count>>

AcquireForCreate(t) ==
    /\ thread_state[t] = "acquiring_create"
    /\ mutex_holder = NULL
    /\ mutex_holder' = t
    /\ thread_state' = [thread_state EXCEPT ![t] = "creating"]
    /\ UNCHANGED <<encoder_exists, encoder_refcount, encoder_ended,
                   thread_encoder, thread_method_count, thread_current_method,
                   use_after_free_count, ended_encoder_use_count>>

CreateEncoder(t, e) ==
    /\ thread_state[t] = "creating"
    /\ mutex_holder = t
    /\ encoder_exists[e] = FALSE
    /\ encoder_exists' = [encoder_exists EXCEPT ![e] = TRUE]
    /\ encoder_refcount' = [encoder_refcount EXCEPT ![e] = 1]
    /\ encoder_ended' = [encoder_ended EXCEPT ![e] = FALSE]
    /\ thread_state' = [thread_state EXCEPT ![t] = "releasing_create"]
    /\ thread_encoder' = [thread_encoder EXCEPT ![t] = e]
    /\ thread_method_count' = [thread_method_count EXCEPT ![t] = 0]
    /\ UNCHANGED <<mutex_holder, thread_current_method,
                   use_after_free_count, ended_encoder_use_count>>

ReleaseAfterCreate(t) ==
    /\ thread_state[t] = "releasing_create"
    /\ mutex_holder = t
    /\ mutex_holder' = NULL
    /\ thread_state' = [thread_state EXCEPT ![t] = "has_encoder"]
    /\ UNCHANGED <<encoder_exists, encoder_refcount, encoder_ended,
                   thread_encoder, thread_method_count, thread_current_method,
                   use_after_free_count, ended_encoder_use_count>>

(* -------------------------------------------------------------------------- *)
(* Method Calls (4 types to increase state space)                             *)
(* -------------------------------------------------------------------------- *)

TryAcquireForMethod(t, m) ==
    /\ thread_state[t] = "has_encoder"
    /\ thread_encoder[t] /= NULL
    /\ thread_method_count[t] < MaxMethodCalls
    /\ m \in MethodTypes
    /\ IF mutex_holder = NULL
       THEN /\ mutex_holder' = t
            /\ thread_state' = [thread_state EXCEPT ![t] =
                IF m = 1 THEN "in_method_1"
                ELSE IF m = 2 THEN "in_method_2"
                ELSE IF m = 3 THEN "in_method_3"
                ELSE "in_method_4"]
       ELSE /\ thread_state' = [thread_state EXCEPT ![t] = "acquiring_method"]
            /\ UNCHANGED mutex_holder
    /\ thread_current_method' = [thread_current_method EXCEPT ![t] = m]
    /\ UNCHANGED <<encoder_exists, encoder_refcount, encoder_ended,
                   thread_encoder, thread_method_count,
                   use_after_free_count, ended_encoder_use_count>>

AcquireForMethod(t) ==
    /\ thread_state[t] = "acquiring_method"
    /\ mutex_holder = NULL
    /\ mutex_holder' = t
    /\ LET m == thread_current_method[t] IN
        thread_state' = [thread_state EXCEPT ![t] =
            IF m = 1 THEN "in_method_1"
            ELSE IF m = 2 THEN "in_method_2"
            ELSE IF m = 3 THEN "in_method_3"
            ELSE "in_method_4"]
    /\ UNCHANGED <<encoder_exists, encoder_refcount, encoder_ended,
                   thread_encoder, thread_method_count, thread_current_method,
                   use_after_free_count, ended_encoder_use_count>>

(* Check safety in method - encoder must exist and not be ended *)
ExecuteMethod(t) ==
    /\ thread_state[t] \in {"in_method_1", "in_method_2", "in_method_3", "in_method_4"}
    /\ mutex_holder = t
    /\ LET e == thread_encoder[t] IN
        IF e /= NULL /\ encoder_exists[e] = TRUE /\ encoder_ended[e] = FALSE
        THEN (* Safe - method executes normally *)
            /\ UNCHANGED <<use_after_free_count, ended_encoder_use_count>>
        ELSE IF e /= NULL /\ encoder_exists[e] = TRUE /\ encoder_ended[e] = TRUE
        THEN (* Bug: using ended encoder *)
            /\ ended_encoder_use_count' = ended_encoder_use_count + 1
            /\ UNCHANGED use_after_free_count
        ELSE (* Bug: use after free *)
            /\ use_after_free_count' = use_after_free_count + 1
            /\ UNCHANGED ended_encoder_use_count
    /\ thread_state' = [thread_state EXCEPT ![t] = "releasing_method"]
    /\ thread_method_count' = [thread_method_count EXCEPT ![t] = @ + 1]
    /\ UNCHANGED <<encoder_exists, encoder_refcount, encoder_ended, mutex_holder,
                   thread_encoder, thread_current_method>>

ReleaseAfterMethod(t) ==
    /\ thread_state[t] = "releasing_method"
    /\ mutex_holder = t
    /\ mutex_holder' = NULL
    /\ thread_state' = [thread_state EXCEPT ![t] = "has_encoder"]
    /\ UNCHANGED <<encoder_exists, encoder_refcount, encoder_ended,
                   thread_encoder, thread_method_count, thread_current_method,
                   use_after_free_count, ended_encoder_use_count>>

(* -------------------------------------------------------------------------- *)
(* End Encoding                                                               *)
(* -------------------------------------------------------------------------- *)

TryAcquireForEnd(t) ==
    /\ thread_state[t] = "has_encoder"
    /\ thread_encoder[t] /= NULL
    /\ IF mutex_holder = NULL
       THEN /\ mutex_holder' = t
            /\ thread_state' = [thread_state EXCEPT ![t] = "ending"]
       ELSE /\ thread_state' = [thread_state EXCEPT ![t] = "acquiring_end"]
            /\ UNCHANGED mutex_holder
    /\ UNCHANGED <<encoder_exists, encoder_refcount, encoder_ended,
                   thread_encoder, thread_method_count, thread_current_method,
                   use_after_free_count, ended_encoder_use_count>>

AcquireForEnd(t) ==
    /\ thread_state[t] = "acquiring_end"
    /\ mutex_holder = NULL
    /\ mutex_holder' = t
    /\ thread_state' = [thread_state EXCEPT ![t] = "ending"]
    /\ UNCHANGED <<encoder_exists, encoder_refcount, encoder_ended,
                   thread_encoder, thread_method_count, thread_current_method,
                   use_after_free_count, ended_encoder_use_count>>

EndEncoding(t) ==
    /\ thread_state[t] = "ending"
    /\ mutex_holder = t
    /\ LET e == thread_encoder[t] IN
        /\ e /= NULL
        /\ encoder_ended' = [encoder_ended EXCEPT ![e] = TRUE]
        /\ encoder_refcount' = [encoder_refcount EXCEPT ![e] = @ - 1]
    /\ thread_state' = [thread_state EXCEPT ![t] = "releasing_end"]
    /\ UNCHANGED <<encoder_exists, mutex_holder, thread_encoder,
                   thread_method_count, thread_current_method,
                   use_after_free_count, ended_encoder_use_count>>

ReleaseAfterEnd(t) ==
    /\ thread_state[t] = "releasing_end"
    /\ mutex_holder = t
    /\ mutex_holder' = NULL
    /\ thread_state' = [thread_state EXCEPT ![t] = "idle"]
    /\ thread_encoder' = [thread_encoder EXCEPT ![t] = NULL]
    /\ thread_method_count' = [thread_method_count EXCEPT ![t] = 0]
    /\ thread_current_method' = [thread_current_method EXCEPT ![t] = 0]
    /\ UNCHANGED <<encoder_exists, encoder_refcount, encoder_ended,
                   use_after_free_count, ended_encoder_use_count>>

(* -------------------------------------------------------------------------- *)
(* System: Deallocate encoder when refcount reaches 0                         *)
(* -------------------------------------------------------------------------- *)

DeallocEncoder(e) ==
    /\ encoder_exists[e] = TRUE
    /\ encoder_refcount[e] = 0
    /\ encoder_exists' = [encoder_exists EXCEPT ![e] = FALSE]
    /\ encoder_ended' = [encoder_ended EXCEPT ![e] = FALSE]
    /\ UNCHANGED <<encoder_refcount, mutex_holder, thread_state, thread_encoder,
                   thread_method_count, thread_current_method,
                   use_after_free_count, ended_encoder_use_count>>

(* -------------------------------------------------------------------------- *)
(* Next State                                                                 *)
(* -------------------------------------------------------------------------- *)

Next ==
    \/ \E t \in Threads: TryAcquireForCreate(t)
    \/ \E t \in Threads: AcquireForCreate(t)
    \/ \E t \in Threads, e \in Encoders: CreateEncoder(t, e)
    \/ \E t \in Threads: ReleaseAfterCreate(t)
    \/ \E t \in Threads, m \in MethodTypes: TryAcquireForMethod(t, m)
    \/ \E t \in Threads: AcquireForMethod(t)
    \/ \E t \in Threads: ExecuteMethod(t)
    \/ \E t \in Threads: ReleaseAfterMethod(t)
    \/ \E t \in Threads: TryAcquireForEnd(t)
    \/ \E t \in Threads: AcquireForEnd(t)
    \/ \E t \in Threads: EndEncoding(t)
    \/ \E t \in Threads: ReleaseAfterEnd(t)
    \/ \E e \in Encoders: DeallocEncoder(e)

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

(* -------------------------------------------------------------------------- *)
(* Safety Properties                                                          *)
(* -------------------------------------------------------------------------- *)

TypeOK ==
    /\ encoder_exists \in [Encoders -> BOOLEAN]
    /\ encoder_refcount \in [Encoders -> Nat]
    /\ encoder_ended \in [Encoders -> BOOLEAN]
    /\ mutex_holder \in Threads \cup {NULL}
    /\ thread_state \in [Threads -> ThreadStates]
    /\ thread_encoder \in [Threads -> Encoders \cup {NULL}]
    /\ thread_method_count \in [Threads -> 0..MaxMethodCalls]
    /\ thread_current_method \in [Threads -> MethodTypes \cup {NULL}]
    /\ use_after_free_count \in Nat
    /\ ended_encoder_use_count \in Nat

(* No use-after-free *)
NoUseAfterFree == use_after_free_count = 0

(* No use of ended encoder *)
NoEndedEncoderUse == ended_encoder_use_count = 0

(* Combined safety *)
V2_9_Safety ==
    /\ NoUseAfterFree
    /\ NoEndedEncoderUse

(* Thread with encoder always has valid encoder (only when in states that use it) *)
ThreadEncoderValid ==
    \A t \in Threads:
        /\ thread_encoder[t] /= NULL
        /\ thread_state[t] \in {"has_encoder", "acquiring_method", "in_method_1",
                                 "in_method_2", "in_method_3", "in_method_4",
                                 "releasing_method", "acquiring_end", "ending"}
        =>
            /\ encoder_exists[thread_encoder[t]] = TRUE
            /\ encoder_refcount[thread_encoder[t]] > 0

=============================================================================
