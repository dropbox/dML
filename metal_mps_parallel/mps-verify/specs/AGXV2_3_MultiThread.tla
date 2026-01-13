---------------------------- MODULE AGXV2_3_MultiThread ----------------------------
(*
 * AGX v2.3 Multi-Thread Model - Can threads share encoders?
 *
 * Scenario:
 * 1. Thread A creates encoder (v2.3 retains it)
 * 2. Thread A passes pointer to Thread B (out of band)
 * 3. Thread A calls endEncoding (v2.3 releases)
 * 4. Thread B tries to use encoder -> ???
 *
 * This models whether v2.3 handles encoder sharing between threads.
 *)

EXTENDS Integers, FiniteSets

CONSTANTS
    NumThreads

ASSUME NumThreads \in Nat /\ NumThreads > 0

VARIABLES
    encoder_exists,
    encoder_refcount,
    encoder_owner,        \* Which thread created the encoder
    mutex_holder,
    thread_state,         \* "idle" | "has_encoder" | "ending"
    thread_encoder,       \* Which encoder thread is using (may be borrowed!)
    use_after_free_count

vars == <<encoder_exists, encoder_refcount, encoder_owner, mutex_holder, thread_state, thread_encoder, use_after_free_count>>

Threads == 1..NumThreads
NULL == 0

ThreadStates == {"idle", "has_encoder", "ending", "using"}

Init ==
    /\ encoder_exists = FALSE
    /\ encoder_refcount = 0
    /\ encoder_owner = NULL
    /\ mutex_holder = NULL
    /\ thread_state = [t \in Threads |-> "idle"]
    /\ thread_encoder = [t \in Threads |-> NULL]
    /\ use_after_free_count = 0

(* Thread creates encoder - only one encoder for simplicity *)
CreateEncoder(t) ==
    /\ thread_state[t] = "idle"
    /\ encoder_exists = FALSE
    /\ mutex_holder = NULL
    /\ mutex_holder' = t
    /\ encoder_exists' = TRUE
    /\ encoder_refcount' = 1
    /\ encoder_owner' = t
    /\ thread_state' = [thread_state EXCEPT ![t] = "has_encoder"]
    /\ thread_encoder' = [thread_encoder EXCEPT ![t] = 1]
    /\ UNCHANGED use_after_free_count

FinishCreation(t) ==
    /\ thread_state[t] = "has_encoder"
    /\ mutex_holder = t
    /\ mutex_holder' = NULL
    /\ UNCHANGED <<encoder_exists, encoder_refcount, encoder_owner, thread_state, thread_encoder, use_after_free_count>>

(* Thread B borrows encoder from Thread A *)
BorrowEncoder(borrower, owner) ==
    /\ borrower /= owner
    /\ thread_state[borrower] = "idle"
    /\ thread_encoder[borrower] = NULL
    /\ thread_state[owner] = "has_encoder"
    /\ thread_encoder[owner] = 1
    /\ encoder_exists = TRUE
    (* No mutex needed - just pointer passing (out of band) *)
    (* v2.3 does NOT add extra retain for borrower! *)
    /\ thread_state' = [thread_state EXCEPT ![borrower] = "has_encoder"]
    /\ thread_encoder' = [thread_encoder EXCEPT ![borrower] = 1]
    /\ UNCHANGED <<encoder_exists, encoder_refcount, encoder_owner, mutex_holder, use_after_free_count>>

(* Owner calls endEncoding - can only call once *)
OwnerEndEncoding(t) ==
    /\ thread_state[t] = "has_encoder"
    /\ encoder_owner = t
    /\ thread_encoder[t] = 1  \* Still has encoder
    /\ encoder_refcount > 0   \* Can only end if refcount > 0
    /\ mutex_holder = NULL
    /\ mutex_holder' = t
    /\ encoder_refcount' = encoder_refcount - 1
    /\ thread_state' = [thread_state EXCEPT ![t] = "ending"]
    /\ thread_encoder' = [thread_encoder EXCEPT ![t] = NULL]
    /\ UNCHANGED <<encoder_exists, encoder_owner, use_after_free_count>>

FinishOwnerEnd(t) ==
    /\ mutex_holder = t
    /\ thread_state[t] = "ending"
    /\ mutex_holder' = NULL
    /\ thread_state' = [thread_state EXCEPT ![t] = "idle"]
    /\ UNCHANGED <<encoder_exists, encoder_refcount, encoder_owner, thread_encoder, use_after_free_count>>

(* System deallocates when refcount = 0 *)
DeallocEncoder ==
    /\ encoder_exists = TRUE
    /\ encoder_refcount = 0
    /\ encoder_exists' = FALSE
    /\ UNCHANGED <<encoder_refcount, encoder_owner, mutex_holder, thread_state, thread_encoder, use_after_free_count>>

(* Borrower tries to use encoder - MIGHT CRASH! *)
BorrowerUseEncoder(t) ==
    /\ thread_state[t] = "has_encoder"
    /\ encoder_owner /= t  \* Not the owner - is a borrower
    /\ thread_encoder[t] = 1
    /\ mutex_holder = NULL
    /\ mutex_holder' = t
    /\ IF encoder_exists = FALSE
       THEN
           (* USE-AFTER-FREE! *)
           /\ use_after_free_count' = use_after_free_count + 1
           /\ UNCHANGED <<encoder_exists, encoder_refcount, encoder_owner, thread_state, thread_encoder>>
       ELSE
           (* Safe - encoder still exists *)
           /\ UNCHANGED <<encoder_exists, encoder_refcount, encoder_owner, thread_state, thread_encoder, use_after_free_count>>

FinishBorrowerUse(t) ==
    /\ thread_state[t] = "has_encoder"
    /\ encoder_owner /= t
    /\ mutex_holder = t
    /\ mutex_holder' = NULL
    /\ UNCHANGED <<encoder_exists, encoder_refcount, encoder_owner, thread_state, thread_encoder, use_after_free_count>>

Next ==
    \/ \E t \in Threads: CreateEncoder(t)
    \/ \E t \in Threads: FinishCreation(t)
    \/ \E t1, t2 \in Threads: BorrowEncoder(t1, t2)
    \/ \E t \in Threads: OwnerEndEncoding(t)
    \/ \E t \in Threads: FinishOwnerEnd(t)
    \/ DeallocEncoder
    \/ \E t \in Threads: BorrowerUseEncoder(t)
    \/ \E t \in Threads: FinishBorrowerUse(t)

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ encoder_exists \in BOOLEAN
    /\ encoder_refcount \in Nat
    /\ encoder_owner \in Threads \cup {NULL}
    /\ mutex_holder \in Threads \cup {NULL}
    /\ thread_state \in [Threads -> ThreadStates]
    /\ use_after_free_count \in Nat

NoUAF == use_after_free_count = 0

=============================================================================
