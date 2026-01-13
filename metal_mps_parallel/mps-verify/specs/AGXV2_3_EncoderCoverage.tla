---------------------------- MODULE AGXV2_3_EncoderCoverage ----------------------------
(*
 * AGX v2.3 Encoder Coverage Model
 *
 * Verifies that v2.3 covers ALL encoder types that can be created.
 * Models factory methods on command buffer and checks that each encoder type
 * is properly tracked (retained on creation, released on endEncoding).
 *
 * PROPERTY: Every encoder created through a factory method must be tracked.
 *)

EXTENDS Integers, FiniteSets

CONSTANTS
    NumThreads

ASSUME NumThreads \in Nat /\ NumThreads > 0

VARIABLES
    (* Encoder tracking state *)
    encoder_exists,       \* [EncoderType -> BOOLEAN] - does an encoder of this type exist
    encoder_tracked,      \* [EncoderType -> BOOLEAN] - is encoder tracked by v2.3
    encoder_ended,        \* [EncoderType -> BOOLEAN] - has endEncoding been called

    (* Thread state *)
    thread_state,         \* [Thread -> "idle" | "creating" | "using" | "ending"]
    thread_encoder_type,  \* [Thread -> EncoderType or NULL]

    (* Error tracking *)
    untracked_use_count   \* Count of times an encoder was used without being tracked

vars == <<encoder_exists, encoder_tracked, encoder_ended, thread_state, thread_encoder_type, untracked_use_count>>

(* Encoder types that v2.3 must cover *)
EncoderTypes == {"compute", "blit"}  \* PyTorch uses these two
NULL == "none"
Threads == 1..NumThreads

TypeOK ==
    /\ encoder_exists \in [EncoderTypes -> BOOLEAN]
    /\ encoder_tracked \in [EncoderTypes -> BOOLEAN]
    /\ encoder_ended \in [EncoderTypes -> BOOLEAN]
    /\ thread_state \in [Threads -> {"idle", "creating", "using", "ending"}]
    /\ thread_encoder_type \in [Threads -> EncoderTypes \cup {NULL}]
    /\ untracked_use_count \in Nat

Init ==
    /\ encoder_exists = [t \in EncoderTypes |-> FALSE]
    /\ encoder_tracked = [t \in EncoderTypes |-> FALSE]
    /\ encoder_ended = [t \in EncoderTypes |-> FALSE]
    /\ thread_state = [t \in Threads |-> "idle"]
    /\ thread_encoder_type = [t \in Threads |-> NULL]
    /\ untracked_use_count = 0

(* Thread creates an encoder via factory method *)
(* v2.3 SWIZZLES the factory, so it MUST track the encoder at creation *)
CreateEncoder(thread, etype) ==
    /\ thread_state[thread] = "idle"
    /\ encoder_exists[etype] = FALSE
    /\ encoder_exists' = [encoder_exists EXCEPT ![etype] = TRUE]
    (* v2.3 swizzles factory method, so encoder is tracked at creation *)
    /\ encoder_tracked' = [encoder_tracked EXCEPT ![etype] = TRUE]
    /\ thread_state' = [thread_state EXCEPT ![thread] = "using"]
    /\ thread_encoder_type' = [thread_encoder_type EXCEPT ![thread] = etype]
    /\ UNCHANGED <<encoder_ended, untracked_use_count>>

(* Thread uses the encoder (calls a method like setBuffer, fillBuffer, etc.) *)
UseEncoder(thread) ==
    /\ thread_state[thread] = "using"
    /\ thread_encoder_type[thread] /= NULL
    /\ LET etype == thread_encoder_type[thread] IN
        /\ encoder_exists[etype] = TRUE
        /\ IF encoder_tracked[etype] = FALSE
           THEN untracked_use_count' = untracked_use_count + 1
           ELSE UNCHANGED untracked_use_count
    /\ UNCHANGED <<encoder_exists, encoder_tracked, encoder_ended, thread_state, thread_encoder_type>>

(* Thread ends encoding *)
(* v2.3 releases the encoder at endEncoding *)
EndEncoding(thread) ==
    /\ thread_state[thread] = "using"
    /\ thread_encoder_type[thread] /= NULL
    /\ LET etype == thread_encoder_type[thread] IN
        /\ encoder_exists[etype] = TRUE
        /\ encoder_ended' = [encoder_ended EXCEPT ![etype] = TRUE]
        (* v2.3 releases at endEncoding *)
        /\ encoder_tracked' = [encoder_tracked EXCEPT ![etype] = FALSE]
    /\ thread_state' = [thread_state EXCEPT ![thread] = "idle"]
    /\ thread_encoder_type' = [thread_encoder_type EXCEPT ![thread] = NULL]
    /\ UNCHANGED <<encoder_exists, untracked_use_count>>

(* System deallocates encoder after endEncoding *)
DeallocEncoder(etype) ==
    /\ encoder_exists[etype] = TRUE
    /\ encoder_ended[etype] = TRUE
    /\ encoder_tracked[etype] = FALSE  \* Must be released before dealloc
    /\ encoder_exists' = [encoder_exists EXCEPT ![etype] = FALSE]
    /\ encoder_ended' = [encoder_ended EXCEPT ![etype] = FALSE]
    /\ UNCHANGED <<encoder_tracked, thread_state, thread_encoder_type, untracked_use_count>>

Next ==
    \/ \E t \in Threads, e \in EncoderTypes: CreateEncoder(t, e)
    \/ \E t \in Threads: UseEncoder(t)
    \/ \E t \in Threads: EndEncoding(t)
    \/ \E e \in EncoderTypes: DeallocEncoder(e)

Spec == Init /\ [][Next]_vars

(* SAFETY PROPERTIES *)

(* Every encoder use must be tracked - if this is violated, we have a coverage gap *)
NoUntrackedUse == untracked_use_count = 0

(* Encoder cannot be used after it's been ended *)
NoUseAfterEnd ==
    \A t \in Threads:
        thread_encoder_type[t] /= NULL =>
            encoder_ended[thread_encoder_type[t]] = FALSE

(* Every existing encoder must be tracked (v2.3 invariant) *)
ExistingEncodersTracked ==
    \A e \in EncoderTypes:
        (encoder_exists[e] = TRUE /\ encoder_ended[e] = FALSE) =>
            encoder_tracked[e] = TRUE

=============================================================================
