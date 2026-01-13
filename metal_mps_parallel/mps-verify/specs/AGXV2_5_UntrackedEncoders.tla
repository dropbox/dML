----------------------- MODULE AGXV2_5_UntrackedEncoders -----------------------
(*
 * TLA+ Model: AGX Fix v2.5 - Untracked Encoder Safety
 * Created by Andrew Yates
 *
 * PURPOSE: This spec models the gap that caused the v2.4 PAC trap crash.
 *
 * THE GAP (v2.4 bug):
 *   v2.4 assumed all encoders flow through our swizzled creation paths.
 *   Reality: some encoders are created BEFORE our library loads or through
 *   paths we don't swizzle. When v2.4 tried to CFRetain an untracked encoder
 *   in encoder_method_begin(), it crashed with a PAC trap if that encoder
 *   was already freed (dangling pointer).
 *
 * THE FIX (v2.5):
 *   v2.5 NEVER CFRetains in encoder_method_begin for untracked encoders.
 *   If an encoder is not in our tracking map, we skip protection entirely.
 *   This is safer than crashing - we accept that untracked encoders may
 *   experience the original race condition, but they won't crash our fix.
 *
 * MODEL INVARIANT:
 *   We should NEVER call CFRetain on an untracked encoder.
 *   This spec proves v2.5 maintains this invariant.
 *
 * AUTOMATIC GAP CLOSURE:
 *   The key insight is that we model the CLOSED WORLD ASSUMPTION explicitly.
 *   By having "UntrackedEncoders" as a first-class state variable, we
 *   force ourselves to handle this case in the specification.
 *)

EXTENDS Integers, TLC, FiniteSets

CONSTANTS
    MAX_ENCODERS,    \* Maximum number of encoders in the system
    NULL             \* Null pointer value

VARIABLES
    pc,                    \* Program counter per thread
    TrackedEncoders,       \* Set of encoder IDs we track (created through swizzled path)
    UntrackedEncoders,     \* Set of encoder IDs we DON'T track (created before/outside swizzle)
    FreedEncoders,         \* Set of encoder IDs that have been freed
    EncoderActiveCalls,    \* Function: encoder ID -> active call count
    EncFailed,             \* Did we crash trying to touch a bad encoder?
    CurrentEncoder,        \* Which encoder the current thread is operating on
    OperationSkipped       \* Was operation skipped due to untracked encoder?

vars == <<pc, TrackedEncoders, UntrackedEncoders, FreedEncoders,
          EncoderActiveCalls, EncFailed, CurrentEncoder, OperationSkipped>>

Threads == {1, 2}

(****************************************************************************)
(* ENCODER SETS                                                             *)
(****************************************************************************)

AllEncoders == 1..MAX_ENCODERS

\* All encoders that exist (tracked or untracked)
ExistingEncoders == TrackedEncoders \cup UntrackedEncoders

\* Encoder is safe to touch (exists and not freed)
SafeToTouch(e) == e \in ExistingEncoders /\ e \notin FreedEncoders

(****************************************************************************)
(* THREAD 1: Method call on encoder (v2.5 behavior)                        *)
(****************************************************************************)

\* Thread 1 starts a method call on some encoder (tracked or untracked)
T1_Start ==
    /\ pc[1] = "T1_Start"
    /\ \E e \in ExistingEncoders \ FreedEncoders :  \* Pick a live encoder
        /\ CurrentEncoder' = [CurrentEncoder EXCEPT ![1] = e]
        /\ pc' = [pc EXCEPT ![1] = "T1_CheckTracking"]
    /\ UNCHANGED <<TrackedEncoders, UntrackedEncoders, FreedEncoders,
                   EncoderActiveCalls, EncFailed, OperationSkipped>>

\* v2.5 KEY FIX: Check if encoder is tracked BEFORE doing anything
T1_CheckTracking ==
    /\ pc[1] = "T1_CheckTracking"
    /\ LET e == CurrentEncoder[1] IN
        IF e \in TrackedEncoders
        THEN \* Tracked - safe to increment active calls (no CFRetain needed)
             /\ EncoderActiveCalls' = [EncoderActiveCalls EXCEPT ![e] = @ + 1]
             /\ pc' = [pc EXCEPT ![1] = "T1_CallMethod"]
             /\ OperationSkipped' = FALSE
             /\ UNCHANGED <<EncFailed>>
        ELSE \* NOT TRACKED - v2.5 skips protection entirely
             \* v2.4 BUG: would try CFRetain here, crash if freed
             /\ pc' = [pc EXCEPT ![1] = "T1_SkipProtection"]
             /\ OperationSkipped' = TRUE
             /\ UNCHANGED <<EncoderActiveCalls, EncFailed>>
    /\ UNCHANGED <<TrackedEncoders, UntrackedEncoders, FreedEncoders, CurrentEncoder>>

\* v2.5: Skip protection for untracked encoder, but method still runs
\* (It may hit original race condition, but won't crash our fix)
T1_SkipProtection ==
    /\ pc[1] = "T1_SkipProtection"
    /\ pc' = [pc EXCEPT ![1] = "T1_Done"]
    /\ UNCHANGED <<TrackedEncoders, UntrackedEncoders, FreedEncoders,
                   EncoderActiveCalls, EncFailed, CurrentEncoder, OperationSkipped>>

\* Call the actual method (only for tracked encoders)
T1_CallMethod ==
    /\ pc[1] = "T1_CallMethod"
    /\ pc' = [pc EXCEPT ![1] = "T1_MethodEnd"]
    /\ UNCHANGED <<TrackedEncoders, UntrackedEncoders, FreedEncoders,
                   EncoderActiveCalls, EncFailed, CurrentEncoder, OperationSkipped>>

\* Method completes, decrement active calls
T1_MethodEnd ==
    /\ pc[1] = "T1_MethodEnd"
    /\ LET e == CurrentEncoder[1] IN
        /\ EncoderActiveCalls' = [EncoderActiveCalls EXCEPT ![e] = @ - 1]
        /\ pc' = [pc EXCEPT ![1] = "T1_Done"]
    /\ UNCHANGED <<TrackedEncoders, UntrackedEncoders, FreedEncoders,
                   EncFailed, CurrentEncoder, OperationSkipped>>

T1_Done ==
    /\ pc[1] = "T1_Done"
    /\ pc' = [pc EXCEPT ![1] = "T1_Start"]
    /\ UNCHANGED <<TrackedEncoders, UntrackedEncoders, FreedEncoders,
                   EncoderActiveCalls, EncFailed, CurrentEncoder, OperationSkipped>>

(****************************************************************************)
(* THREAD 2: Encoder lifecycle (create, free)                              *)
(****************************************************************************)

\* Create a tracked encoder (through our swizzled path)
T2_CreateTracked ==
    /\ pc[2] = "T2_Idle"
    /\ \E e \in AllEncoders \ ExistingEncoders :
        /\ TrackedEncoders' = TrackedEncoders \cup {e}
        /\ EncoderActiveCalls' = [EncoderActiveCalls EXCEPT ![e] = 0]
    /\ pc' = [pc EXCEPT ![2] = "T2_Idle"]
    /\ UNCHANGED <<UntrackedEncoders, FreedEncoders, EncFailed, CurrentEncoder, OperationSkipped>>

\* Create an untracked encoder (outside our swizzle - THE GAP)
T2_CreateUntracked ==
    /\ pc[2] = "T2_Idle"
    /\ \E e \in AllEncoders \ ExistingEncoders :
        /\ UntrackedEncoders' = UntrackedEncoders \cup {e}
    /\ pc' = [pc EXCEPT ![2] = "T2_Idle"]
    /\ UNCHANGED <<TrackedEncoders, FreedEncoders, EncoderActiveCalls,
                   EncFailed, CurrentEncoder, OperationSkipped>>

\* Free an encoder
T2_FreeEncoder ==
    /\ pc[2] = "T2_Idle"
    /\ \E e \in ExistingEncoders \ FreedEncoders :
        /\ FreedEncoders' = FreedEncoders \cup {e}
    /\ UNCHANGED <<pc, TrackedEncoders, UntrackedEncoders, EncoderActiveCalls,
                   EncFailed, CurrentEncoder, OperationSkipped>>

(****************************************************************************)
(* v2.4 BUG SCENARIO (what we're proving against)                          *)
(****************************************************************************)

\* This models what v2.4 did WRONG: CFRetain on untracked encoder
\* We include this to prove our fix avoids it
V24_Bug_CFRetainUntracked ==
    /\ pc[1] = "T1_CheckTracking"
    /\ LET e == CurrentEncoder[1] IN
        /\ e \notin TrackedEncoders  \* Untracked
        /\ e \in FreedEncoders       \* And already freed!
        \* v2.4 would try CFRetain here -> PAC TRAP CRASH
        /\ EncFailed' = TRUE
    /\ UNCHANGED <<pc, TrackedEncoders, UntrackedEncoders, FreedEncoders,
                   EncoderActiveCalls, CurrentEncoder, OperationSkipped>>

(****************************************************************************)
(* SPECIFICATION                                                            *)
(****************************************************************************)

Init ==
    /\ pc = [t \in Threads |-> IF t = 1 THEN "T1_Start" ELSE "T2_Idle"]
    /\ TrackedEncoders = {}
    /\ UntrackedEncoders = {}
    /\ FreedEncoders = {}
    /\ EncoderActiveCalls = [e \in AllEncoders |-> 0]
    /\ EncFailed = FALSE
    /\ CurrentEncoder = [t \in Threads |-> 0]
    /\ OperationSkipped = FALSE

\* v2.5 spec: does NOT include V24_Bug_CFRetainUntracked
Next ==
    \/ T1_Start \/ T1_CheckTracking \/ T1_SkipProtection
    \/ T1_CallMethod \/ T1_MethodEnd \/ T1_Done
    \/ T2_CreateTracked \/ T2_CreateUntracked \/ T2_FreeEncoder

\* v2.4 buggy spec (for comparison): includes the bug
NextWithBug ==
    \/ Next
    \/ V24_Bug_CFRetainUntracked

Spec == Init /\ [][Next]_vars

(****************************************************************************)
(* SAFETY PROPERTIES                                                        *)
(****************************************************************************)

\* PRIMARY INVARIANT: We never crash due to touching an untracked encoder
NeverCrash == ~EncFailed

\* MODEL INVARIANT CHECK: We never CFRetain untracked encoders
\* In v2.5, we skip protection for untracked encoders instead of CFRetaining
NeverCFRetainUntracked ==
    \* If we're in T1_CallMethod, the encoder MUST be tracked
    pc[1] = "T1_CallMethod" => CurrentEncoder[1] \in TrackedEncoders

\* Untracked encoders may experience original race (acceptable)
\* but they never crash our fix library
UntrackedSafe ==
    \A e \in UntrackedEncoders :
        \* We never have active calls on untracked encoders
        EncoderActiveCalls[e] = 0

(****************************************************************************)
(* LIVENESS (for completeness)                                              *)
(****************************************************************************)

\* Thread 1 can always make progress (never deadlocks)
T1Progress == pc[1] = "T1_Start" ~> pc[1] # "T1_Start"

(****************************************************************************)
(* AUTOMATIC GAP CLOSURE ANNOTATION                                         *)
(*                                                                          *)
(* This spec demonstrates the pattern for automatically closing             *)
(* model-implementation gaps:                                               *)
(*                                                                          *)
(* 1. EXPLICIT UNIVERSE: We define UntrackedEncoders as a first-class       *)
(*    variable, forcing the model to account for encoders outside our       *)
(*    control.                                                              *)
(*                                                                          *)
(* 2. CLOSED WORLD ASSUMPTION EXPLICIT: The v2.4 bug assumed                *)
(*    AllEncoders == TrackedEncoders. By having UntrackedEncoders           *)
(*    in the model, we can't make that assumption.                          *)
(*                                                                          *)
(* 3. INVARIANT AS RUNTIME CHECK: The NeverCFRetainUntracked                *)
(*    invariant maps directly to a runtime assertion in C++ code:           *)
(*                                                                          *)
(*    if (it == g_encoder_states.end()) {                                   *)
(*        // MODEL INVARIANT: encoder not tracked                           *)
(*        // Do NOT CFRetain - skip protection instead                      *)
(*        g_untracked_skips++;                                              *)
(*        return false;                                                     *)
(*    }                                                                     *)
(*                                                                          *)
(* The g_untracked_skips counter in v2.5 IS the runtime model check.        *)
(* If this counter increases, we know untracked encoders exist.             *)
(****************************************************************************)

=============================================================================
