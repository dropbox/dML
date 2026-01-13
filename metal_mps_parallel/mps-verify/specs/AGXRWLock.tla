---------------------------- MODULE AGXRWLock ----------------------------
(*
 * AGX Reader-Writer Lock Model - Proves Insufficient
 *
 * Worker: N=1475
 * Purpose: Phase 4.1 - Prove that reader-writer lock does NOT prevent the race
 *
 * This model shows that a reader-writer lock approach fails because:
 * 1. If RW lock is per-context: async completion handlers bypass the lock
 * 2. If RW lock is global: it's equivalent to a global mutex (no improvement)
 *
 * We model the first case: per-context RW locks fail because destruction
 * can occur through async completion paths that don't acquire the lock.
 *
 * EXPECTED RESULT: TLC finds a violation (race still occurs)
 *)

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    NumThreads,
    NumContextSlots

ASSUME NumThreads \in Nat /\ NumThreads > 1
ASSUME NumContextSlots \in Nat /\ NumContextSlots > 0

VARIABLES
    (* Per-context reader-writer lock (THE "FIX" WE'RE TESTING) *)
    context_readers,    \* ContextId -> Set of Threads holding read lock
    context_writer,     \* ContextId -> Thread | NULL (who holds write lock)

    (* Per-thread state *)
    thread_context,     \* Thread -> ContextId | NULL
    thread_state,       \* Thread -> ThreadState

    (* Shared context registry *)
    context_registry,   \* ContextId -> {valid, invalid}

    (* Async completion simulation *)
    pending_destroy,    \* Set of ContextIds scheduled for async destruction

    (* Bug detection *)
    null_deref_count,
    race_witnessed

vars == <<context_readers, context_writer, thread_context, thread_state,
          context_registry, pending_destroy, null_deref_count, race_witnessed>>

Threads == 1..NumThreads
ContextIds == 1..NumContextSlots
NULL == 0

ThreadStates == {
    "idle",
    "creating",
    "acquiring_read",   \* Waiting for read lock
    "encoding",         \* Holding read lock, encoding
    "releasing_read",   \* Done encoding, releasing read
    "committing"        \* Committed command buffer
}

(* -------------------------------------------------------------------------- *)
(* Initial State                                                              *)
(* -------------------------------------------------------------------------- *)

Init ==
    /\ context_readers = [c \in ContextIds |-> {}]
    /\ context_writer = [c \in ContextIds |-> NULL]
    /\ thread_context = [t \in Threads |-> NULL]
    /\ thread_state = [t \in Threads |-> "idle"]
    /\ context_registry = [c \in ContextIds |-> "invalid"]
    /\ pending_destroy = {}
    /\ null_deref_count = 0
    /\ race_witnessed = FALSE

(* -------------------------------------------------------------------------- *)
(* RW Lock Helpers                                                            *)
(* -------------------------------------------------------------------------- *)

\* Can acquire read lock if no writer holds the lock
CanAcquireRead(c) ==
    context_writer[c] = NULL

\* Can acquire write lock if no readers and no writer
CanAcquireWrite(c) ==
    /\ context_readers[c] = {}
    /\ context_writer[c] = NULL

(* -------------------------------------------------------------------------- *)
(* Context Creation (No Lock Needed - Thread Creates Own Context)             *)
(* -------------------------------------------------------------------------- *)

CreateContext(t) ==
    /\ thread_state[t] = "idle"
    /\ thread_context[t] = NULL
    /\ \E c \in ContextIds:
        /\ context_registry[c] = "invalid"
        /\ context_registry' = [context_registry EXCEPT ![c] = "valid"]
        /\ thread_context' = [thread_context EXCEPT ![t] = c]
    /\ thread_state' = [thread_state EXCEPT ![t] = "creating"]
    /\ UNCHANGED <<context_readers, context_writer, pending_destroy,
                   null_deref_count, race_witnessed>>

(* Context aliasing/migration: another thread starts using the SAME context id. *)
AliasContext(from, to) ==
    LET c == thread_context[from] IN
    /\ from \in Threads /\ to \in Threads /\ from /= to
    /\ thread_state[from] \in {"creating", "encoding", "acquiring_read"}
    /\ c /= NULL
    /\ context_registry[c] = "valid"
    /\ thread_state[to] = "idle"
    /\ thread_context[to] = NULL
    /\ thread_context' = [thread_context EXCEPT ![to] = c]
    /\ thread_state' = [thread_state EXCEPT ![to] = "creating"]
    /\ UNCHANGED <<context_readers, context_writer, context_registry, pending_destroy,
                   null_deref_count, race_witnessed>>

(* -------------------------------------------------------------------------- *)
(* Encoding Operations (Protected by Read Lock)                               *)
(* -------------------------------------------------------------------------- *)

\* Request read lock on context
WantEncode(t) ==
    /\ thread_state[t] = "creating"
    /\ thread_context[t] /= NULL
    /\ thread_state' = [thread_state EXCEPT ![t] = "acquiring_read"]
    /\ UNCHANGED <<context_readers, context_writer, thread_context,
                   context_registry, pending_destroy, null_deref_count, race_witnessed>>

\* Acquire read lock
AcquireReadLock(t) ==
    LET c == thread_context[t] IN
    /\ thread_state[t] = "acquiring_read"
    /\ c /= NULL
    /\ CanAcquireRead(c)
    /\ context_readers' = [context_readers EXCEPT ![c] = @ \cup {t}]
    /\ thread_state' = [thread_state EXCEPT ![t] = "encoding"]
    /\ UNCHANGED <<context_writer, thread_context, context_registry,
                   pending_destroy, null_deref_count, race_witnessed>>

\* Continue encoding - check for NULL deref (THE BUG)
ContinueEncoding(t) ==
    LET c == thread_context[t] IN
    /\ thread_state[t] = "encoding"
    /\ c /= NULL
    /\ t \in context_readers[c]  \* Holding read lock
    /\ IF context_registry[c] = "invalid"
       THEN \* NULL pointer dereference despite holding read lock!
            /\ null_deref_count' = null_deref_count + 1
            /\ race_witnessed' = TRUE
            /\ UNCHANGED <<context_readers, context_writer, thread_context,
                          thread_state, context_registry, pending_destroy>>
       ELSE
            /\ UNCHANGED vars

\* Finish encoding, release read lock
FinishEncoding(t) ==
    LET c == thread_context[t] IN
    /\ thread_state[t] = "encoding"
    /\ c /= NULL
    /\ t \in context_readers[c]
    /\ context_readers' = [context_readers EXCEPT ![c] = @ \ {t}]
    /\ thread_state' = [thread_state EXCEPT ![t] = "releasing_read"]
    /\ UNCHANGED <<context_writer, thread_context, context_registry,
                   pending_destroy, null_deref_count, race_witnessed>>

(* -------------------------------------------------------------------------- *)
(* Command Buffer Commit (Schedules Async Destruction)                        *)
(* -------------------------------------------------------------------------- *)

\* Commit command buffer - schedules async destruction of context
CommitBuffer(t) ==
    LET c == thread_context[t] IN
    /\ thread_state[t] = "releasing_read"
    /\ c /= NULL
    \* THE KEY: Commit schedules async destruction, doesn't wait for locks
    /\ pending_destroy' = pending_destroy \cup {c}
    /\ thread_context' = [thread_context EXCEPT ![t] = NULL]
    /\ thread_state' = [thread_state EXCEPT ![t] = "idle"]
    /\ UNCHANGED <<context_readers, context_writer, context_registry,
                   null_deref_count, race_witnessed>>

(* -------------------------------------------------------------------------- *)
(* Async Completion Handler (THE RACE - BYPASSES RW LOCK)                     *)
(* -------------------------------------------------------------------------- *)

(*
 * THE KEY INSIGHT:
 * Async completion handlers in Metal/AGX don't acquire our RW lock.
 * They can destroy contexts while other threads are encoding.
 * This is because:
 * 1. The completion handler runs on a system thread we don't control
 * 2. It's triggered by GPU completion, not by our code
 * 3. It invalidates the context regardless of who's using it
 *)
AsyncDestruction ==
    /\ pending_destroy /= {}
    /\ \E c \in pending_destroy:
        \* NO LOCK CHECK - async handler doesn't use our RW lock
        /\ context_registry' = [context_registry EXCEPT ![c] = "invalid"]
        /\ pending_destroy' = pending_destroy \ {c}
    /\ UNCHANGED <<context_readers, context_writer, thread_context, thread_state,
                   null_deref_count, race_witnessed>>

(* -------------------------------------------------------------------------- *)
(* Alternative: Thread B Reuses Slot (Different Race Vector)                  *)
(* -------------------------------------------------------------------------- *)

(*
 * Another race vector: Thread B creates a new context in the same slot
 * while Thread A is still encoding with a pointer to that slot.
 * The slot reuse invalidates Thread A's context.
 *)
ReuseSlot(t) ==
    LET c == thread_context[t] IN
    /\ thread_state[t] = "idle"
    /\ thread_context[t] = NULL
    /\ \E c2 \in ContextIds:
        \* Slot was invalidated by async destruction
        /\ context_registry[c2] = "invalid"
        \* But another thread might still be encoding with old pointer
        /\ context_registry' = [context_registry EXCEPT ![c2] = "valid"]
        /\ thread_context' = [thread_context EXCEPT ![t] = c2]
    /\ thread_state' = [thread_state EXCEPT ![t] = "creating"]
    /\ UNCHANGED <<context_readers, context_writer, pending_destroy,
                   null_deref_count, race_witnessed>>

(* -------------------------------------------------------------------------- *)
(* Next State Relation                                                        *)
(* -------------------------------------------------------------------------- *)

Next ==
    \/ \E t \in Threads:
        \/ CreateContext(t)
        \/ \E t2 \in Threads: AliasContext(t, t2)
        \/ WantEncode(t)
        \/ AcquireReadLock(t)
        \/ ContinueEncoding(t)
        \/ FinishEncoding(t)
        \/ CommitBuffer(t)
        \/ ReuseSlot(t)
    \/ AsyncDestruction

Fairness ==
    /\ \A t \in Threads: WF_vars(CreateContext(t))
    /\ \A t \in Threads: WF_vars(WantEncode(t))
    /\ \A t \in Threads: WF_vars(AcquireReadLock(t))
    /\ \A t \in Threads: WF_vars(FinishEncoding(t))
    /\ \A t \in Threads: WF_vars(CommitBuffer(t))
    /\ WF_vars(AsyncDestruction)

Spec == Init /\ [][Next]_vars /\ Fairness

(* -------------------------------------------------------------------------- *)
(* Safety Properties (THESE SHOULD FAIL)                                      *)
(* -------------------------------------------------------------------------- *)

NoNullDereference ==
    null_deref_count = 0

NoRaceCondition ==
    ~race_witnessed

(* -------------------------------------------------------------------------- *)
(* Comments                                                                   *)
(* -------------------------------------------------------------------------- *)
(*
 * WHY READER-WRITER LOCK FAILS:
 *
 * Thread 1:                      Async Completion:
 * ---------                      -----------------
 * context = create()
 * rw_lock.read_lock(context)
 * start encoding...
 *                                [GPU finishes command buffer]
 *                                (doesn't know about our RW lock!)
 *                                context.invalidate()  <-- RACE!
 * use context -> NULL DEREF!
 * rw_lock.read_unlock(context)
 *
 * The fundamental problem: we can't add RW locks to Apple's driver.
 * Any user-space RW lock is bypassed by:
 * 1. Async completion handlers (run on system threads)
 * 2. Command buffer deallocation
 * 3. Device loss/reset
 *
 * Even if we could add locks to the driver, the async nature of GPU
 * completion means destruction events don't synchronize with our locks.
 *
 * CONCLUSION:
 * - Per-context RW lock: Bypassed by async destruction
 * - Global RW lock: Equivalent to global mutex (no benefit)
 * - Global mutex: The ONLY correct solution at user-space level
 *
 * To run TLC:
 *   java -jar tla2tools.jar -config AGXRWLock.cfg AGXRWLock.tla
 *
 * Expected output: Invariant NoNullDereference is violated.
 *)
=============================================================================
