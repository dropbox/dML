(* MPS Parallel Inference - Iris Separation Logic Proofs

   This module provides the foundational imports and definitions
   for verifying the MPS (Metal Performance Shaders) parallel inference
   implementation using Iris separation logic.

   Key properties to verify:
   1. Memory safety - no use-after-free
   2. Race freedom - no data races
   3. Callback lifetime - callbacks don't outlive objects
*)

From iris.algebra Require Import excl auth gmap agree.
From iris.base_logic.lib Require Import invariants.
From iris.heap_lang Require Import lang proofmode notation.
From iris.proofmode Require Import tactics.

(* Standard library imports *)
From stdpp Require Import gmap.

(* Define the ghost state for MPS verification *)
Class mpsG Sigma := MpsG {
  (* Mutex ownership tracking *)
  mps_mutex_inG : inG Sigma (exclR unitO);
  (* Buffer generation counters for ABA detection *)
  mps_gen_inG : inG Sigma (authR (optionUR (exclR natO)));
  (* TLS binding: exclusive ownership of stream slots *)
  mps_tls_inG : inG Sigma (exclR natO);
  (* Callback pending counter tracking *)
  mps_callback_inG : inG Sigma (authR natUR);
}.

#[export] Existing Instances mps_mutex_inG mps_gen_inG mps_tls_inG mps_callback_inG.

(* Thread ID type - represents worker threads *)
Definition tid := nat.

(* Stream ID type - represents MPS streams in the pool *)
Definition stream_id := nat.

(* Buffer ID type - represents allocated buffers *)
Definition buffer_id := nat.

(* Generation counter for ABA detection *)
Definition generation := nat.

(* Constants matching PyTorch MPS implementation *)
Definition MAX_STREAMS : nat := 32.
Definition MAX_THREADS : nat := 64.

(* A stream slot can be free or bound to a thread *)
Inductive slot_state :=
  | Free
  | Bound (owner : tid).

(* Buffer state for the allocator *)
Inductive buffer_state :=
  | BufAllocated (gen : generation)
  | BufFreed (gen : generation)
  | BufInvalid.

Section mps_definitions.
  Context `{!heapGS Sigma, !mpsG Sigma}.

  (* Mutex invariant: either held by someone or available *)
  Definition mutex_inv (l : loc) (γ : gname) : iProp Sigma :=
    (l ↦ #false) ∨ (l ↦ #true ∗ own γ (Excl ())).

  (* Locked predicate: asserts we hold the mutex *)
  Definition locked (γ : gname) : iProp Sigma :=
    own γ (Excl ()).

End mps_definitions.
