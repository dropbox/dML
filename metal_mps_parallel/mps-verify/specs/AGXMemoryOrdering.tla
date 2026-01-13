---------------------------- MODULE AGXMemoryOrdering ----------------------------
(*
 * AGX Memory Ordering Model - ARM64 Weak Memory
 *
 * This TLA+ specification models potential memory ordering bugs in v2.1 fix.
 *
 * ARM64 has weak memory ordering:
 * - Stores can be reordered with other stores
 * - Loads can be reordered with other loads
 * - Stores can be reordered with subsequent loads
 * - Only acquire/release semantics provide ordering guarantees
 *
 * CRITICAL: The v2.1 fix uses:
 * - std::unordered_map (NOT thread-safe)
 * - std::unordered_set (NOT thread-safe)
 * - No memory barriers between check and increment
 *
 * RACE CONDITIONS TO MODEL:
 * 1. Thread A reads g_retained_encoders[ptr] while Thread B writes
 * 2. Thread A checks g_destroyed_encoders while Thread B inserts
 * 3. Store to thread_local visible before store to global map
 *)

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    NumThreads

ASSUME NumThreads \in Nat /\ NumThreads > 0

VARIABLES
    (* Global state (shared, need synchronization) *)
    global_retain_count,     \* Maps encoder -> count
    global_destroyed_set,    \* Set of destroyed encoders

    (* Per-thread local state (thread_local, no sync needed) *)
    thread_local_using,      \* Per thread: set of encoders

    (* Store buffers model ARM64 weak ordering *)
    store_buffer,            \* Per thread: pending writes not yet visible

    (* Thread state *)
    thread_pc,               \* Program counter: what step thread is at
    thread_encoder,          \* Which encoder thread is operating on

    (* Bug counters *)
    data_races,              \* Number of data races detected
    torn_reads,              \* Number of torn reads detected
    stale_reads              \* Number of stale reads detected

vars == <<global_retain_count, global_destroyed_set, thread_local_using,
          store_buffer, thread_pc, thread_encoder, data_races, torn_reads, stale_reads>>

Threads == 1..NumThreads
Encoder == 1  \* Single encoder for simplicity - races happen regardless

NULL == 0

(* Program counter states - models ensure_encoder_alive *)
PCStates == {
    "idle",
    "check_destroyed",       \* if (g_destroyed_encoders.count(ptr) > 0)
    "check_local",           \* if (t_thread_using_encoders.count(ptr) > 0)
    "insert_local",          \* t_thread_using_encoders.insert(ptr)
    "read_global_count",     \* it = g_retained_encoders.find(ptr)
    "write_global_count",    \* g_retained_encoders[ptr]++ or CFRetain
    "done"
}

(* -------------------------------------------------------------------------- *)
(* Store Buffer Model                                                          *)
(* -------------------------------------------------------------------------- *)
(*
 * On ARM64, stores go to a store buffer first.
 * Other threads don't see them until they're flushed.
 * This models the reordering.
 *)

(* A write entry in store buffer *)
WriteEntry == [var: {"retain_count", "destroyed", "local"}, value: Int]

(* -------------------------------------------------------------------------- *)
(* Initial State                                                              *)
(* -------------------------------------------------------------------------- *)

Init ==
    /\ global_retain_count = 0
    /\ global_destroyed_set = FALSE
    /\ thread_local_using = [t \in Threads |-> FALSE]
    /\ store_buffer = [t \in Threads |-> <<>>]
    /\ thread_pc = [t \in Threads |-> "idle"]
    /\ thread_encoder = [t \in Threads |-> NULL]
    /\ data_races = 0
    /\ torn_reads = 0
    /\ stale_reads = 0

(* -------------------------------------------------------------------------- *)
(* Thread Actions - Model ensure_encoder_alive step by step                    *)
(* -------------------------------------------------------------------------- *)

(* Thread starts accessing encoder *)
StartAccess(t) ==
    /\ thread_pc[t] = "idle"
    /\ thread_pc' = [thread_pc EXCEPT ![t] = "check_destroyed"]
    /\ thread_encoder' = [thread_encoder EXCEPT ![t] = Encoder]
    /\ UNCHANGED <<global_retain_count, global_destroyed_set, thread_local_using,
                   store_buffer, data_races, torn_reads, stale_reads>>

(* Check if encoder is destroyed *)
CheckDestroyed(t) ==
    /\ thread_pc[t] = "check_destroyed"
    /\ IF global_destroyed_set = TRUE
       THEN
           (* Encoder destroyed - abort *)
           /\ thread_pc' = [thread_pc EXCEPT ![t] = "idle"]
           /\ thread_encoder' = [thread_encoder EXCEPT ![t] = NULL]
       ELSE
           (* Continue to check local *)
           /\ thread_pc' = [thread_pc EXCEPT ![t] = "check_local"]
           /\ UNCHANGED thread_encoder
    /\ UNCHANGED <<global_retain_count, global_destroyed_set, thread_local_using,
                   store_buffer, data_races, torn_reads, stale_reads>>

(* Check thread_local set *)
CheckLocal(t) ==
    /\ thread_pc[t] = "check_local"
    /\ IF thread_local_using[t] = TRUE
       THEN
           (* Already using - skip increment *)
           /\ thread_pc' = [thread_pc EXCEPT ![t] = "done"]
       ELSE
           (* Need to insert and increment *)
           /\ thread_pc' = [thread_pc EXCEPT ![t] = "insert_local"]
    /\ UNCHANGED <<global_retain_count, global_destroyed_set, thread_local_using,
                   store_buffer, thread_encoder, data_races, torn_reads, stale_reads>>

(* Insert into thread_local set *)
InsertLocal(t) ==
    /\ thread_pc[t] = "insert_local"
    /\ thread_local_using' = [thread_local_using EXCEPT ![t] = TRUE]
    /\ thread_pc' = [thread_pc EXCEPT ![t] = "read_global_count"]
    /\ UNCHANGED <<global_retain_count, global_destroyed_set,
                   store_buffer, thread_encoder, data_races, torn_reads, stale_reads>>

(*
 * CRITICAL RACE: Read global count
 *
 * BUG: If another thread is writing at the same time, we have a data race!
 * std::unordered_map is NOT thread-safe for concurrent read/write.
 *)
ReadGlobalCount(t) ==
    /\ thread_pc[t] = "read_global_count"
    (* Check if another thread is writing - DATA RACE! *)
    /\ LET writing_threads == {t2 \in Threads :
            t2 /= t /\ thread_pc[t2] = "write_global_count"}
       IN
        IF writing_threads /= {}
        THEN
            (* DATA RACE DETECTED! *)
            /\ data_races' = data_races + 1
            /\ thread_pc' = [thread_pc EXCEPT ![t] = "write_global_count"]
            /\ UNCHANGED <<global_retain_count, global_destroyed_set, thread_local_using,
                          store_buffer, thread_encoder, torn_reads, stale_reads>>
        ELSE
            (* Safe read *)
            /\ thread_pc' = [thread_pc EXCEPT ![t] = "write_global_count"]
            /\ UNCHANGED <<global_retain_count, global_destroyed_set, thread_local_using,
                          store_buffer, thread_encoder, data_races, torn_reads, stale_reads>>

(*
 * CRITICAL RACE: Write global count
 *
 * BUG: Lost update if two threads increment simultaneously!
 * Thread A reads count=0, Thread B reads count=0
 * Thread A writes count=1, Thread B writes count=1
 * Result: count=1 (should be 2!)
 *)
WriteGlobalCount(t) ==
    /\ thread_pc[t] = "write_global_count"
    (* Check for concurrent writers - lost update! *)
    /\ LET concurrent_writers == {t2 \in Threads :
            t2 /= t /\ thread_pc[t2] = "write_global_count"}
       IN
        IF concurrent_writers /= {}
        THEN
            (* LOST UPDATE - both threads see old value! *)
            /\ torn_reads' = torn_reads + 1
            (* Both threads write same incremented value *)
            /\ global_retain_count' = global_retain_count + 1
        ELSE
            (* Safe write *)
            /\ global_retain_count' = global_retain_count + 1
            /\ UNCHANGED torn_reads
    /\ thread_pc' = [thread_pc EXCEPT ![t] = "done"]
    /\ UNCHANGED <<global_destroyed_set, thread_local_using, store_buffer,
                   thread_encoder, data_races, stale_reads>>

(* Thread finishes *)
FinishAccess(t) ==
    /\ thread_pc[t] = "done"
    /\ thread_pc' = [thread_pc EXCEPT ![t] = "idle"]
    /\ thread_encoder' = [thread_encoder EXCEPT ![t] = NULL]
    /\ UNCHANGED <<global_retain_count, global_destroyed_set, thread_local_using,
                   store_buffer, data_races, torn_reads, stale_reads>>

(* -------------------------------------------------------------------------- *)
(* Destruction - happens from command buffer completion                        *)
(* -------------------------------------------------------------------------- *)

DestroyEncoder ==
    /\ global_destroyed_set = FALSE
    /\ global_retain_count = 0
    /\ global_destroyed_set' = TRUE
    /\ UNCHANGED <<global_retain_count, thread_local_using, store_buffer,
                   thread_pc, thread_encoder, data_races, torn_reads, stale_reads>>

(*
 * TOCTOU RACE: Encoder destroyed between check and increment
 *
 * Thread A: check destroyed -> FALSE
 * System: destroy encoder (count was 0)
 * Thread A: increment count (on destroyed encoder!)
 *)
TocTouRace(t) ==
    /\ thread_pc[t] \in {"check_local", "insert_local", "read_global_count"}
    /\ global_destroyed_set = FALSE
    /\ global_retain_count = 0
    (* System destroys while thread is between check and increment *)
    /\ global_destroyed_set' = TRUE
    /\ stale_reads' = stale_reads + 1
    /\ UNCHANGED <<global_retain_count, thread_local_using, store_buffer,
                   thread_pc, thread_encoder, data_races, torn_reads>>

(* -------------------------------------------------------------------------- *)
(* Next State                                                                 *)
(* -------------------------------------------------------------------------- *)

Next ==
    \/ \E t \in Threads: StartAccess(t)
    \/ \E t \in Threads: CheckDestroyed(t)
    \/ \E t \in Threads: CheckLocal(t)
    \/ \E t \in Threads: InsertLocal(t)
    \/ \E t \in Threads: ReadGlobalCount(t)
    \/ \E t \in Threads: WriteGlobalCount(t)
    \/ \E t \in Threads: FinishAccess(t)
    \/ DestroyEncoder
    \/ \E t \in Threads: TocTouRace(t)

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

(* -------------------------------------------------------------------------- *)
(* Safety Properties                                                          *)
(* -------------------------------------------------------------------------- *)

TypeOK ==
    /\ global_retain_count \in Nat
    /\ global_destroyed_set \in BOOLEAN
    /\ thread_local_using \in [Threads -> BOOLEAN]
    /\ thread_pc \in [Threads -> PCStates]
    /\ data_races \in Nat
    /\ torn_reads \in Nat
    /\ stale_reads \in Nat

(*
 * MAIN SAFETY: No data races
 * This WILL BE VIOLATED - proving the v2.1 fix has race conditions!
 *)
NoDataRaces == data_races = 0

(*
 * MAIN SAFETY: No torn reads (lost updates)
 * This WILL BE VIOLATED - proving lost updates are possible!
 *)
NoTornReads == torn_reads = 0

(*
 * MAIN SAFETY: No TOCTOU races
 * This WILL BE VIOLATED - proving check-then-act is not atomic!
 *)
NoTocTouRaces == stale_reads = 0

(*
 * Combined safety - expect ALL to be violated!
 *)
AllSafe == NoDataRaces /\ NoTornReads /\ NoTocTouRaces

(* -------------------------------------------------------------------------- *)
(* What This Model Reveals                                                    *)
(* -------------------------------------------------------------------------- *)
(*
 * EXPECTED VIOLATIONS:
 *
 * 1. NoDataRaces VIOLATED:
 *    std::unordered_map is not thread-safe. Concurrent read/write = UB.
 *    FIX NEEDED: Use std::mutex or std::atomic operations.
 *
 * 2. NoTornReads VIOLATED:
 *    g_retained_encoders[ptr]++ is read-modify-write, not atomic.
 *    Two threads can both read 0 and both write 1 (lost update).
 *    FIX NEEDED: Use std::atomic<int> or mutex protection.
 *
 * 3. NoTocTouRaces VIOLATED:
 *    Between checking g_destroyed_encoders and incrementing the count,
 *    the encoder can be destroyed!
 *    FIX NEEDED: Hold mutex during entire ensure_encoder_alive.
 *
 * REQUIRED FIX for v2.2:
 *    Replace std::unordered_map with synchronized access:
 *    - Option A: std::mutex protecting all operations
 *    - Option B: std::shared_mutex for read-heavy workloads
 *    - Option C: Lock-free concurrent hash map
 *)

=============================================================================
