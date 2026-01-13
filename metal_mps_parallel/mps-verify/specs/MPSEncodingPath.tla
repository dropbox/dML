---------------------------- MODULE MPSEncodingPath ----------------------------
(*
 * MPS Encoding Path Serialization Analysis
 *
 * This TLA+ specification models the question:
 * "Is the global encoding mutex NECESSARY for correctness?"
 *
 * We model:
 * 1. Multiple threads, each with their own stream/command queue
 * 2. Shared state: allocator, buffer pool
 * 3. The encoding path: acquire stream -> get buffer -> create encoder -> encode -> commit
 *
 * The key question: Can concurrent encoding cause data races on shared state?
 *
 * If GlobalSerializationRequired is FALSE in all reachable states, the mutex is NOT necessary.
 *)

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    NumThreads,       \* Number of concurrent threads (e.g., 4)
    NumStreams,       \* Number of streams in pool (e.g., 4)
    NumBuffers        \* Number of buffers in shared pool (e.g., 8)

ASSUME NumThreads \in Nat /\ NumThreads > 0
ASSUME NumStreams \in Nat /\ NumStreams >= NumThreads
ASSUME NumBuffers \in Nat /\ NumBuffers >= NumThreads

VARIABLES
    (* Per-thread state *)
    thread_stream,        \* Thread -> StreamId (which stream thread owns)
    thread_encoder,       \* Thread -> EncoderId | NULL (active encoder)
    thread_buffer,        \* Thread -> BufferId | NULL (buffer being used)
    pc,                   \* Thread -> PCState

    (* Shared state - THE KEY QUESTION: Do these need global mutex? *)
    allocator_free_list,  \* Set of free buffer IDs
    allocator_mutex_held, \* Thread holding allocator mutex | NULL
    buffer_refcount,      \* BufferId -> refcount (how many threads using it)

    (* Encoder state - per stream, not global *)
    stream_encoder_active,\* StreamId -> Bool (is encoder active on this stream?)

    (* Metrics for analysis *)
    max_concurrent_encoding,  \* Max threads encoding simultaneously
    global_serialization_witnessed  \* Did we ever need global serialization?

vars == <<thread_stream, thread_encoder, thread_buffer, pc,
          allocator_free_list, allocator_mutex_held, buffer_refcount,
          stream_encoder_active, max_concurrent_encoding, global_serialization_witnessed>>

Threads == 1..NumThreads
Streams == 1..NumStreams
Buffers == 1..NumBuffers
NULL == 0

PCStates == {
    "idle",
    "acquiring_stream",
    "allocating_buffer",     \* Needs allocator mutex
    "waiting_for_allocator", \* Blocked on allocator mutex
    "creating_encoder",      \* Creates command encoder on stream
    "encoding",              \* Actively encoding GPU commands
    "committing",            \* Committing command buffer
    "releasing_buffer",      \* Releasing buffer back to pool
    "done"
}

(* -------------------------------------------------------------------------- *)
(* Initial State                                                              *)
(* -------------------------------------------------------------------------- *)

Init ==
    /\ thread_stream = [t \in Threads |-> t]  \* Each thread gets unique stream
    /\ thread_encoder = [t \in Threads |-> NULL]
    /\ thread_buffer = [t \in Threads |-> NULL]
    /\ pc = [t \in Threads |-> "idle"]
    /\ allocator_free_list = Buffers
    /\ allocator_mutex_held = NULL
    /\ buffer_refcount = [b \in Buffers |-> 0]
    /\ stream_encoder_active = [s \in Streams |-> FALSE]
    /\ max_concurrent_encoding = 0
    /\ global_serialization_witnessed = FALSE

(* -------------------------------------------------------------------------- *)
(* Helper: Count threads in encoding state                                    *)
(* -------------------------------------------------------------------------- *)

ThreadsEncoding == {t \in Threads : pc[t] = "encoding"}
NumThreadsEncoding == Cardinality(ThreadsEncoding)

(* -------------------------------------------------------------------------- *)
(* Actions                                                                    *)
(* -------------------------------------------------------------------------- *)

(* Thread starts operation *)
StartOperation(t) ==
    /\ pc[t] = "idle"
    /\ pc' = [pc EXCEPT ![t] = "acquiring_stream"]
    /\ UNCHANGED <<thread_stream, thread_encoder, thread_buffer,
                   allocator_free_list, allocator_mutex_held, buffer_refcount,
                   stream_encoder_active, max_concurrent_encoding, global_serialization_witnessed>>

(* Thread acquires its stream (already assigned, just transition) *)
AcquireStream(t) ==
    /\ pc[t] = "acquiring_stream"
    /\ pc' = [pc EXCEPT ![t] = "allocating_buffer"]
    /\ UNCHANGED <<thread_stream, thread_encoder, thread_buffer,
                   allocator_free_list, allocator_mutex_held, buffer_refcount,
                   stream_encoder_active, max_concurrent_encoding, global_serialization_witnessed>>

(* Thread tries to allocate buffer - needs allocator mutex *)
TryAllocateBuffer(t) ==
    /\ pc[t] = "allocating_buffer"
    /\ IF allocator_mutex_held = NULL
       THEN (* Got mutex, allocate buffer *)
            /\ allocator_free_list /= {}
            /\ \E b \in allocator_free_list:
                /\ thread_buffer' = [thread_buffer EXCEPT ![t] = b]
                /\ allocator_free_list' = allocator_free_list \ {b}
                /\ buffer_refcount' = [buffer_refcount EXCEPT ![b] = 1]
            /\ allocator_mutex_held' = NULL  \* Release immediately
            /\ pc' = [pc EXCEPT ![t] = "creating_encoder"]
       ELSE (* Mutex held by another thread - must wait *)
            /\ pc' = [pc EXCEPT ![t] = "waiting_for_allocator"]
            /\ UNCHANGED <<thread_buffer, allocator_free_list, allocator_mutex_held, buffer_refcount>>
    /\ UNCHANGED <<thread_stream, thread_encoder, stream_encoder_active,
                   max_concurrent_encoding, global_serialization_witnessed>>

(* Thread waiting for allocator mutex *)
WaitForAllocator(t) ==
    /\ pc[t] = "waiting_for_allocator"
    /\ allocator_mutex_held = NULL
    /\ pc' = [pc EXCEPT ![t] = "allocating_buffer"]
    /\ UNCHANGED <<thread_stream, thread_encoder, thread_buffer,
                   allocator_free_list, allocator_mutex_held, buffer_refcount,
                   stream_encoder_active, max_concurrent_encoding, global_serialization_witnessed>>

(* Thread creates command encoder on its own stream *)
(* KEY: This uses per-stream state, NOT global state *)
CreateEncoder(t) ==
    /\ pc[t] = "creating_encoder"
    /\ LET s == thread_stream[t] IN
        /\ ~stream_encoder_active[s]  \* Per-stream check, not global
        /\ stream_encoder_active' = [stream_encoder_active EXCEPT ![s] = TRUE]
        /\ thread_encoder' = [thread_encoder EXCEPT ![t] = s]
    /\ pc' = [pc EXCEPT ![t] = "encoding"]
    /\ UNCHANGED <<thread_stream, thread_buffer, allocator_free_list,
                   allocator_mutex_held, buffer_refcount,
                   max_concurrent_encoding, global_serialization_witnessed>>

(* Thread is encoding - update max concurrent *)
Encoding(t) ==
    /\ pc[t] = "encoding"
    /\ LET concurrent == NumThreadsEncoding IN
        /\ max_concurrent_encoding' = IF concurrent > max_concurrent_encoding
                                      THEN concurrent
                                      ELSE max_concurrent_encoding
    /\ pc' = [pc EXCEPT ![t] = "committing"]
    /\ UNCHANGED <<thread_stream, thread_encoder, thread_buffer,
                   allocator_free_list, allocator_mutex_held, buffer_refcount,
                   stream_encoder_active, global_serialization_witnessed>>

(* Thread commits and ends encoder *)
Commit(t) ==
    /\ pc[t] = "committing"
    /\ LET s == thread_stream[t] IN
        /\ stream_encoder_active' = [stream_encoder_active EXCEPT ![s] = FALSE]
        /\ thread_encoder' = [thread_encoder EXCEPT ![t] = NULL]
    /\ pc' = [pc EXCEPT ![t] = "releasing_buffer"]
    /\ UNCHANGED <<thread_stream, thread_buffer, allocator_free_list,
                   allocator_mutex_held, buffer_refcount,
                   max_concurrent_encoding, global_serialization_witnessed>>

(* Thread releases buffer back to pool *)
ReleaseBuffer(t) ==
    /\ pc[t] = "releasing_buffer"
    /\ LET b == thread_buffer[t] IN
        /\ b /= NULL
        /\ allocator_free_list' = allocator_free_list \cup {b}
        /\ buffer_refcount' = [buffer_refcount EXCEPT ![b] = 0]
        /\ thread_buffer' = [thread_buffer EXCEPT ![t] = NULL]
    /\ pc' = [pc EXCEPT ![t] = "done"]
    /\ UNCHANGED <<thread_stream, thread_encoder, allocator_mutex_held,
                   stream_encoder_active, max_concurrent_encoding, global_serialization_witnessed>>

(* Thread completes, can start new operation *)
Complete(t) ==
    /\ pc[t] = "done"
    /\ pc' = [pc EXCEPT ![t] = "idle"]
    /\ UNCHANGED <<thread_stream, thread_encoder, thread_buffer,
                   allocator_free_list, allocator_mutex_held, buffer_refcount,
                   stream_encoder_active, max_concurrent_encoding, global_serialization_witnessed>>

(* -------------------------------------------------------------------------- *)
(* Next State                                                                 *)
(* -------------------------------------------------------------------------- *)

Next == \E t \in Threads:
    \/ StartOperation(t)
    \/ AcquireStream(t)
    \/ TryAllocateBuffer(t)
    \/ WaitForAllocator(t)
    \/ CreateEncoder(t)
    \/ Encoding(t)
    \/ Commit(t)
    \/ ReleaseBuffer(t)
    \/ Complete(t)

Spec == Init /\ [][Next]_vars

(* -------------------------------------------------------------------------- *)
(* Invariants                                                                 *)
(* -------------------------------------------------------------------------- *)

(* Each buffer is used by at most one thread *)
NoBufferSharing ==
    \A b \in Buffers: buffer_refcount[b] <= 1

(* Each stream's encoder is used by at most one thread *)
NoEncoderSharing ==
    \A s \in Streams:
        Cardinality({t \in Threads : thread_encoder[t] = s}) <= 1

(* THE KEY INVARIANT: Is global serialization ever required? *)
(* If multiple threads can be in "encoding" simultaneously, global mutex is NOT needed *)
ParallelEncodingWitnessed ==
    max_concurrent_encoding >= 2

(* -------------------------------------------------------------------------- *)
(* What We're Proving                                                         *)
(* -------------------------------------------------------------------------- *)

(*
 * If we can reach a state where:
 *   - max_concurrent_encoding >= 2
 *   - NoBufferSharing holds
 *   - NoEncoderSharing holds
 *
 * Then the global encoding mutex is NOT NECESSARY for correctness,
 * because parallel encoding is safe.
 *
 * Run TLC with:
 *   Invariant: NoBufferSharing /\ NoEncoderSharing
 *   Property: <>ParallelEncodingWitnessed
 *
 * If property holds and invariant is not violated, mutex is unnecessary.
 *)

TypeOK ==
    /\ thread_stream \in [Threads -> Streams]
    /\ thread_encoder \in [Threads -> Streams \cup {NULL}]
    /\ thread_buffer \in [Threads -> Buffers \cup {NULL}]
    /\ pc \in [Threads -> PCStates]
    /\ allocator_free_list \subseteq Buffers
    /\ allocator_mutex_held \in Threads \cup {NULL}
    /\ buffer_refcount \in [Buffers -> 0..NumThreads]
    /\ stream_encoder_active \in [Streams -> BOOLEAN]
    /\ max_concurrent_encoding \in 0..NumThreads
    /\ global_serialization_witnessed \in BOOLEAN

=============================================================================
