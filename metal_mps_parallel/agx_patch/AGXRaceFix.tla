--------------------------- MODULE AGXRaceFix ---------------------------
(*
 * TLA+ Model: AGX Driver Race Condition Fix
 * Created by Andrew Yates
 *
 * This model proves that the binary patch closes the race window in
 * AGXG16XFamilyComputeContext::destroyImpl.
 *
 * THE BUG: Original code has a state where:
 *   - Lock is released (LockHeld = FALSE)
 *   - _impl pointer is still non-NULL (ImplPtr # NULL)
 *   This allows another thread to see a dangling pointer.
 *
 * THE FIX: Patched code ensures:
 *   - _impl is set to NULL BEFORE lock release
 *   - No state exists where LockHeld=FALSE AND ImplPtr#NULL
 *)

EXTENDS Integers, Sequences, TLC

CONSTANTS NULL, VALID_PTR

VARIABLES
    pc,          \* Program counter (current instruction)
    LockHeld,    \* Is the unfair lock held?
    ImplPtr,     \* Value of self->_impl pointer
    FreelistFull \* Is freelist count > 7?

vars == <<pc, LockHeld, ImplPtr, FreelistFull>>

(***************************************************************************)
(* ORIGINAL (BUGGY) CODE MODEL                                            *)
(* ====================================================================== *)
(* This models the original instruction sequence where str xzr comes      *)
(* AFTER bl unlock, creating a race window.                               *)
(***************************************************************************)

OrigInit ==
    /\ pc = "Start"
    /\ LockHeld = FALSE
    /\ ImplPtr = VALID_PTR  \* Context has valid impl
    /\ FreelistFull \in {TRUE, FALSE}

OrigAcquireLock ==
    /\ pc = "Start"
    /\ pc' = "LockAcquired"
    /\ LockHeld' = TRUE
    /\ UNCHANGED <<ImplPtr, FreelistFull>>

OrigPath1Store ==
    /\ pc = "LockAcquired"
    /\ ~FreelistFull
    /\ pc' = "Path1Stored"
    /\ UNCHANGED <<LockHeld, ImplPtr, FreelistFull>>

OrigPath1Unlock ==
    /\ pc = "Path1Stored"
    /\ pc' = "Path1Unlocked"
    /\ LockHeld' = FALSE
    /\ UNCHANGED <<ImplPtr, FreelistFull>>  \* BUG: ImplPtr still VALID_PTR!

OrigPath1NullImpl ==
    /\ pc = "Path1Unlocked"
    /\ pc' = "Done"
    /\ ImplPtr' = NULL
    /\ UNCHANGED <<LockHeld, FreelistFull>>

OrigPath2Unlock ==
    /\ pc = "LockAcquired"
    /\ FreelistFull
    /\ pc' = "Path2Unlocked"
    /\ LockHeld' = FALSE
    /\ UNCHANGED <<ImplPtr, FreelistFull>>  \* BUG: ImplPtr still VALID_PTR!

OrigPath2Free ==
    /\ pc = "Path2Unlocked"
    /\ pc' = "Path2Freed"
    /\ UNCHANGED <<LockHeld, ImplPtr, FreelistFull>>

OrigPath2NullImpl ==
    /\ pc = "Path2Freed"
    /\ pc' = "Done"
    /\ ImplPtr' = NULL
    /\ UNCHANGED <<LockHeld, FreelistFull>>

OrigNext ==
    \/ OrigAcquireLock
    \/ OrigPath1Store
    \/ OrigPath1Unlock
    \/ OrigPath1NullImpl
    \/ OrigPath2Unlock
    \/ OrigPath2Free
    \/ OrigPath2NullImpl

OrigSpec == OrigInit /\ [][OrigNext]_vars

(***************************************************************************)
(* PATCHED (FIXED) CODE MODEL                                             *)
(* ====================================================================== *)
(* This models the patched instruction sequence where str xzr comes       *)
(* BEFORE bl unlock, closing the race window.                             *)
(***************************************************************************)

FixedInit ==
    /\ pc = "Start"
    /\ LockHeld = FALSE
    /\ ImplPtr = VALID_PTR
    /\ FreelistFull \in {TRUE, FALSE}

FixedAcquireLock ==
    /\ pc = "Start"
    /\ pc' = "LockAcquired"
    /\ LockHeld' = TRUE
    /\ UNCHANGED <<ImplPtr, FreelistFull>>

FixedPath1Store ==
    /\ pc = "LockAcquired"
    /\ ~FreelistFull
    /\ pc' = "Path1Stored"
    /\ UNCHANGED <<LockHeld, ImplPtr, FreelistFull>>

FixedPath1NullImpl ==
    /\ pc = "Path1Stored"
    /\ pc' = "Path1Nulled"
    /\ ImplPtr' = NULL              \* FIX: NULL _impl FIRST!
    /\ UNCHANGED <<LockHeld, FreelistFull>>

FixedPath1Unlock ==
    /\ pc = "Path1Nulled"
    /\ pc' = "Done"
    /\ LockHeld' = FALSE            \* Unlock AFTER _impl is NULL
    /\ UNCHANGED <<ImplPtr, FreelistFull>>

FixedPath2NullImpl ==
    /\ pc = "LockAcquired"
    /\ FreelistFull
    /\ pc' = "Path2Nulled"
    /\ ImplPtr' = NULL              \* FIX: NULL _impl FIRST!
    /\ UNCHANGED <<LockHeld, FreelistFull>>

FixedPath2Unlock ==
    /\ pc = "Path2Nulled"
    /\ pc' = "Path2Unlocked"
    /\ LockHeld' = FALSE            \* Unlock AFTER _impl is NULL
    /\ UNCHANGED <<ImplPtr, FreelistFull>>

FixedPath2Skip ==
    /\ pc = "Path2Unlocked"
    /\ pc' = "Done"
    /\ UNCHANGED <<LockHeld, ImplPtr, FreelistFull>>
    (* Note: free() is skipped in binary patch due to space constraints *)

FixedNext ==
    \/ FixedAcquireLock
    \/ FixedPath1Store
    \/ FixedPath1NullImpl
    \/ FixedPath1Unlock
    \/ FixedPath2NullImpl
    \/ FixedPath2Unlock
    \/ FixedPath2Skip

FixedSpec == FixedInit /\ [][FixedNext]_vars

(***************************************************************************)
(* SAFETY PROPERTY: NO RACE WINDOW                                        *)
(* ====================================================================== *)
(* The race window exists when:                                           *)
(*   - Lock is NOT held (another thread can acquire it)                   *)
(*   - ImplPtr is NOT NULL (another thread sees valid-looking pointer)    *)
(*                                                                        *)
(* This is the BAD state that causes use-after-free crashes.              *)
(***************************************************************************)

NoRaceWindow ==
    \* If lock was acquired and then released, _impl MUST be NULL
    \* (Only applies after we've entered the critical section)
    (pc \notin {"Start", "LockAcquired"} /\ ~LockHeld) => (ImplPtr = NULL)

(*
 * VERIFICATION RESULTS:
 *
 * OrigSpec violates NoRaceWindow:
 *   - At state "Path1Unlocked": LockHeld=FALSE, ImplPtr=VALID_PTR
 *   - At state "Path2Unlocked": LockHeld=FALSE, ImplPtr=VALID_PTR
 *   These are the race windows where crashes occur.
 *
 * FixedSpec satisfies NoRaceWindow:
 *   - At "Path1Nulled": LockHeld=TRUE, ImplPtr=NULL (safe)
 *   - At "Path2Nulled": LockHeld=TRUE, ImplPtr=NULL (safe)
 *   - At "Path2Unlocked": LockHeld=FALSE, ImplPtr=NULL (safe)
 *   - At "Done": LockHeld=FALSE, ImplPtr=NULL (safe)
 *   No state has LockHeld=FALSE with ImplPtr=VALID_PTR.
 *)

(***************************************************************************)
(* ADDITIONAL INVARIANTS                                                  *)
(***************************************************************************)

\* Lock is held from acquisition until explicit release
LockInvariant ==
    pc \in {"LockAcquired", "Path1Stored", "Path1Nulled", "Path2Nulled"} => LockHeld

\* ImplPtr is either VALID_PTR or NULL (no garbage)
ImplPtrValid ==
    ImplPtr \in {VALID_PTR, NULL}

=============================================================================
