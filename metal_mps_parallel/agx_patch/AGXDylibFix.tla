--------------------------- MODULE AGXDylibFix ---------------------------
(*
 * TLA+ Model: AGX Driver Dylib Fix
 * Created by Andrew Yates
 *
 * This model proves that the dylib method swizzling fix with direct
 * _impl NULL checking prevents crashes from NULL pointer dereferences.
 *
 * THE PROBLEM: The binary patch approach reorders instructions inside
 * destroyImpl, but the dylib approach cannot change instruction order.
 * Instead, it:
 *   1. Uses a global mutex to serialize encoder operations with destroyImpl
 *   2. Directly checks _impl before calling original methods
 *
 * THE FIX: Before any encoder operation, we read _impl directly.
 * If _impl is NULL, we skip the operation and return early.
 * This prevents crashes regardless of when _impl was NULLed.
 *
 * SCOPE: This model abstracts ALL encoder methods that access _impl:
 *   - setComputePipelineState:
 *   - dispatchThreads:threadsPerThreadgroup:
 *   - dispatchThreadgroups:threadsPerThreadgroup:
 *   - setBuffer:offset:atIndex:       (crash site fixed!)
 *   - setBytes:length:atIndex:
 *   - setTexture:atIndex:
 *   - setBuffers:offsets:withRange:
 *   - setTextures:withRange:
 *   - setSamplerState:atIndex:
 *   - setSamplerStates:withRange:
 *   - setThreadgroupMemoryLength:atIndex:
 *   - useResource:usage:
 *   - useResources:count:usage:
 *   - useHeap:
 *   - useHeaps:count:
 *   - memoryBarrierWithScope:
 *   - memoryBarrierWithResources:count:
 *   - executeCommandsInBuffer:withRange:
 *   - setStageInRegion:
 *   - setImageblockWidth:height:
 *   - endEncoding
 *   - destroyImpl                     (lifecycle method)
 *   - initWithQueue:                  (lifecycle method)
 *
 * All these methods follow the same pattern (mutex + _impl check), so
 * T1 in this model represents ANY of them. The proof covers all 23.
 *)

EXTENDS Integers, TLC

CONSTANTS NULL, VALID_PTR

VARIABLES
    pc,           \* Program counter per thread
    MutexHolder,  \* Which thread holds our mutex (0 = none)
    ImplPtr,      \* Value of _impl pointer
    ThreadCrashed \* Did any thread crash?

vars == <<pc, MutexHolder, ImplPtr, ThreadCrashed>>

Threads == {1, 2}  \* Two threads for the model

(****************************************************************************)
(* HELPER DEFINITIONS                                                       *)
(****************************************************************************)

\* A thread can acquire the mutex if it's free
CanAcquireMutex(t) == MutexHolder = 0

\* A thread holds the mutex
HoldsMutex(t) == MutexHolder = t

(****************************************************************************)
(* THREAD 1: Encoder Operation (e.g., dispatchThreadgroups)                *)
(****************************************************************************)

T1_Start ==
    /\ pc[1] = "T1_Start"
    /\ CanAcquireMutex(1)
    /\ pc' = [pc EXCEPT ![1] = "T1_HoldMutex"]
    /\ MutexHolder' = 1
    /\ UNCHANGED <<ImplPtr, ThreadCrashed>>

T1_CheckImpl ==
    /\ pc[1] = "T1_HoldMutex"
    /\ IF ImplPtr = NULL
       THEN /\ pc' = [pc EXCEPT ![1] = "T1_Skip"]  \* NULL check catches it!
            /\ UNCHANGED <<MutexHolder, ImplPtr, ThreadCrashed>>
       ELSE /\ pc' = [pc EXCEPT ![1] = "T1_CallOriginal"]
            /\ UNCHANGED <<MutexHolder, ImplPtr, ThreadCrashed>>

T1_CallOriginal ==
    /\ pc[1] = "T1_CallOriginal"
    /\ IF ImplPtr = NULL
       THEN \* This would be a crash, but we already checked!
            /\ ThreadCrashed' = TRUE
            /\ pc' = [pc EXCEPT ![1] = "T1_Done"]
            /\ UNCHANGED <<MutexHolder, ImplPtr>>
       ELSE /\ pc' = [pc EXCEPT ![1] = "T1_Done"]
            /\ UNCHANGED <<MutexHolder, ImplPtr, ThreadCrashed>>

T1_Skip ==
    /\ pc[1] = "T1_Skip"
    /\ pc' = [pc EXCEPT ![1] = "T1_Done"]
    /\ UNCHANGED <<MutexHolder, ImplPtr, ThreadCrashed>>

T1_Done ==
    /\ pc[1] = "T1_Done"
    /\ MutexHolder' = 0  \* Release mutex
    /\ pc' = [pc EXCEPT ![1] = "T1_Start"]  \* Can start again
    /\ UNCHANGED <<ImplPtr, ThreadCrashed>>

(****************************************************************************)
(* THREAD 2: destroyImpl                                                    *)
(****************************************************************************)

T2_Start ==
    /\ pc[2] = "T2_Start"
    /\ CanAcquireMutex(2)
    /\ pc' = [pc EXCEPT ![2] = "T2_HoldMutex"]
    /\ MutexHolder' = 2
    /\ UNCHANGED <<ImplPtr, ThreadCrashed>>

T2_CallOriginalDestroy ==
    /\ pc[2] = "T2_HoldMutex"
    /\ pc' = [pc EXCEPT ![2] = "T2_InsideDestroy"]
    /\ UNCHANGED <<MutexHolder, ImplPtr, ThreadCrashed>>

\* Inside original destroyImpl, _impl gets NULLed
\* (This happens inside the original method, we can't control order)
T2_NullImpl ==
    /\ pc[2] = "T2_InsideDestroy"
    /\ ImplPtr' = NULL
    /\ pc' = [pc EXCEPT ![2] = "T2_Done"]
    /\ UNCHANGED <<MutexHolder, ThreadCrashed>>

T2_Done ==
    /\ pc[2] = "T2_Done"
    /\ MutexHolder' = 0  \* Release mutex
    /\ pc' = [pc EXCEPT ![2] = "T2_Finished"]  \* Only destroy once
    /\ UNCHANGED <<ImplPtr, ThreadCrashed>>

(****************************************************************************)
(* SPECIFICATION                                                            *)
(****************************************************************************)

Init ==
    /\ pc = [t \in Threads |-> IF t = 1 THEN "T1_Start" ELSE "T2_Start"]
    /\ MutexHolder = 0
    /\ ImplPtr = VALID_PTR
    /\ ThreadCrashed = FALSE

Next ==
    \/ T1_Start \/ T1_CheckImpl \/ T1_CallOriginal \/ T1_Skip \/ T1_Done
    \/ T2_Start \/ T2_CallOriginalDestroy \/ T2_NullImpl \/ T2_Done

Spec == Init /\ [][Next]_vars

(****************************************************************************)
(* SAFETY PROPERTIES                                                        *)
(****************************************************************************)

\* No thread ever crashes
NoCrash == ~ThreadCrashed

\* The key invariant: If we reach T1_CallOriginal, _impl MUST be non-NULL
\* because we checked it in T1_CheckImpl while holding the mutex
ImplValidWhenCalled ==
    pc[1] = "T1_CallOriginal" => ImplPtr # NULL

(*
 * PROOF SKETCH:
 *
 * 1. T1 acquires mutex in T1_Start
 * 2. T1 checks ImplPtr in T1_CheckImpl (while holding mutex)
 * 3. If ImplPtr = NULL, T1 goes to T1_Skip (no crash)
 * 4. If ImplPtr # NULL, T1 goes to T1_CallOriginal
 * 5. T2 cannot modify ImplPtr while T1 holds mutex
 * 6. Therefore when T1 is at T1_CallOriginal, ImplPtr # NULL
 * 7. T1_CallOriginal only "crashes" if ImplPtr = NULL
 * 8. But we proved ImplPtr # NULL at T1_CallOriginal
 * 9. Therefore no crash occurs: NoCrash is always TRUE
 *
 * The direct _impl check is the key: we verify state at the moment
 * of use, not at some earlier registration time.
 *)

=============================================================================
