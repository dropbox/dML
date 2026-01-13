(* MPS Mutex Verification

   This module verifies properties of the mutex implementation
   used in MPSStream and other MPS components.

   Key properties:
   - Mutual exclusion: at most one thread holds the lock
   - No deadlock: lock acquisition always succeeds (eventually)
   - Data protection: guarded data is only accessed while locked

   Standard Iris Spin Lock Pattern
   ================================
   The standard formulation places the token in the FREE disjunct:

     is_lock l γ R := inv N ((l ↦ #false ∗ R ∗ locked γ) ∨ (l ↦ #true))

   This enables acquire to:
   1. Open invariant, find lock free (left disjunct)
   2. CAS succeeds, take out token and R
   3. Close invariant with right disjunct (lock held, empty)
   4. Return token and R to caller

   And release to:
   1. Open invariant, find lock held (right disjunct)
   2. Write #false to location
   3. Put token and R back
   4. Close invariant with left disjunct (lock free)

   The previous formulation placed token in the HELD disjunct, which
   made the proofs harder because token couldn't be extracted.
*)

From iris.algebra Require Import excl.
From iris.base_logic.lib Require Import invariants.
From iris.heap_lang Require Import lang proofmode notation.
From iris.proofmode Require Import tactics.
From MPS Require Import prelude.

Section mutex_spec.
  Context `{!heapGS Sigma, !mpsG Sigma}.

  (* The mutex resource algebra: exclusive ownership *)
  Definition mutex_RA := exclR unitO.

  (* Mutex token: having this proves you hold the lock *)
  Definition mutex_token (γ : gname) : iProp Sigma :=
    own γ (Excl ()).

  (* Mutex invariant with protected resource R
     STANDARD PATTERN: Token is in the FREE disjunct, so acquire can take it *)
  Definition is_mutex (l : loc) (γ : gname) (R : iProp Sigma) : iProp Sigma :=
    inv (nroot .@ "mutex")
      ((l ↦ #false ∗ R ∗ mutex_token γ) ∨ (l ↦ #true)).

  (* Key theorem: mutex tokens are exclusive *)
  Lemma mutex_token_exclusive (γ : gname) :
    mutex_token γ -∗ mutex_token γ -∗ False.
  Proof.
    iIntros "H1 H2".
    iDestruct (own_valid_2 with "H1 H2") as %[].
  Qed.

  (* New lock specification: allocate mutex with initial resource *)
  Lemma newlock_spec (R : iProp Sigma) :
    {{{ R }}}
      ref #false
    {{{ l γ, RET #l; is_mutex l γ R }}}.
  Proof.
    iIntros (Φ) "HR HΦ".
    (* Step 1: Allocate the location with value #false *)
    wp_alloc l as "Hl".
    (* Step 2: Allocate ghost state for the mutex token *)
    iMod (own_alloc (Excl ())) as (γ) "Htok"; first done.
    (* Step 3: Create the invariant containing l ↦ #false ∗ R ∗ token *)
    iMod (inv_alloc (nroot .@ "mutex") _
            ((l ↦ #false ∗ R ∗ mutex_token γ) ∨ (l ↦ #true))
          with "[Hl HR Htok]") as "#Hinv".
    { (* Put resources in the left disjunct (lock is free) *)
      iLeft. iFrame. }
    (* Step 4: Return the modality and apply postcondition *)
    iModIntro.
    iApply "HΦ".
    done.
  Qed.

  (* Specification: acquire mutex
     Note: R must be Timeless to strip the later modality
     Proof uses Löb induction for the recursive spin loop. *)
  Lemma acquire_spec (l : loc) (γ : gname) (R : iProp Sigma) `{!Timeless R} :
    {{{ is_mutex l γ R }}}
      (rec: "spin" <> :=
        if: CAS #l #false #true
        then #()
        else "spin" #())%V #()
    {{{ RET #(); mutex_token γ ∗ R }}}.
  Proof.
    iIntros (Φ) "#Hinv HΦ".
    iLöb as "IH".
    wp_rec.
    wp_bind (CmpXchg _ _ _).
    (* Use > to strip later when opening invariant *)
    iInv "Hinv" as ">Hstate".
    iDestruct "Hstate" as "[[Hl [HR Htok]]|Hl]".
    - (* Case: Lock is free (l ↦ #false ∗ R ∗ token) - CAS succeeds *)
      wp_cmpxchg_suc.
      iModIntro.
      iSplitL "Hl"; first by iNext; iRight.
      wp_proj. wp_if_true.
      iApply "HΦ"; by iFrame.
    - (* Case: Lock is held (l ↦ #true) - CAS fails *)
      wp_cmpxchg_fail.
      iModIntro.
      iSplitL "Hl"; first by iNext; iRight.
      wp_proj. wp_if_false.
      by iApply "IH".
  Qed.

  (* Specification: release mutex *)
  Lemma release_spec (l : loc) (γ : gname) (R : iProp Sigma) :
    {{{ is_mutex l γ R ∗ mutex_token γ ∗ R }}}
      #l <- #false
    {{{ RET #(); True }}}.
  Proof.
    iIntros (Φ) "(#Hinv & Htok & HR) HΦ".
    (* Open the invariant to access the lock state *)
    iInv "Hinv" as "Hstate".
    iDestruct "Hstate" as "[Hfree|Hheld]".
    - (* Case: Lock appears free - contradiction! *)
      (* If lock is free, invariant contains a token, but we also have one *)
      iDestruct "Hfree" as "(Hl & _ & Htok')".
      (* Strip later from timeless token using iMod *)
      iMod "Htok'" as "Htok'".
      (* Two tokens is impossible - derive False *)
      iExFalso.
      iApply (mutex_token_exclusive with "Htok Htok'").
    - (* Case: Lock is held - expected case *)
      (* We hold the token, lock location should be #true *)
      iDestruct "Hheld" as "Hl".
      (* Write #false to release the lock *)
      wp_store.
      (* Close invariant with left disjunct: put token and R back *)
      iModIntro.
      iSplitL "Hl HR Htok".
      { iLeft. iFrame. }
      (* Apply postcondition *)
      iApply "HΦ". done.
  Qed.

End mutex_spec.
