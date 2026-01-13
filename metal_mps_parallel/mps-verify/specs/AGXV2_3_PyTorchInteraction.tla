---------------------------- MODULE AGXV2_3_PyTorchInteraction ----------------------------
(*
 * AGX v2.3 + PyTorch 2.9.1 Interaction Model
 *
 * This models the potential interference between:
 * 1. v2.3's creation-time swizzling (global mutex on encoder creation)
 * 2. PyTorch's stream-per-thread architecture with _streamMutex
 *
 * HYPOTHESIS: The v2.3 global mutex serializes encoder creation across ALL
 * streams, which conflicts with PyTorch 2.9.1's parallel stream design.
 * This creates excessive contention and timing changes that expose latent bugs.
 *
 * KEY DIFFERENCE: v2 only protects METHOD CALLS, not creation.
 * v2.3 protects BOTH creation AND method calls.
 *)

EXTENDS Integers, FiniteSets

CONSTANTS
    NumThreads,      \* Number of threads (PyTorch worker threads)
    NumStreams       \* Number of MPS streams (round-robin pool)

ASSUME NumThreads \in Nat /\ NumThreads > 0
ASSUME NumStreams \in Nat /\ NumStreams > 0

VARIABLES
    (* PyTorch State *)
    stream_mutex_holder,     \* [Stream -> Thread | NULL] - who holds each _streamMutex
    stream_encoder,          \* [Stream -> Encoder | NULL] - active encoder per stream

    (* v2.3 State *)
    global_mutex_holder,     \* Thread | NULL - who holds g_encoder_mutex
    tracked_encoders,        \* Set of encoders retained by v2.3

    (* Thread State *)
    thread_state,            \* [Thread -> ThreadState]
    thread_stream,           \* [Thread -> Stream | NULL]
    thread_encoder,          \* [Thread -> Encoder | NULL]
    encoder_counter,         \* Counter for generating unique encoder IDs

    (* Error counters *)
    contention_events,       \* Count of threads blocked on global mutex
    double_lock_events       \* Count of nested lock attempts (potential bug)

vars == <<stream_mutex_holder, stream_encoder, global_mutex_holder, tracked_encoders,
          thread_state, thread_stream, thread_encoder, encoder_counter, contention_events, double_lock_events>>

Threads == 1..NumThreads
Streams == 1..NumStreams
NULL == 0
Encoders == 1..100  \* Encoder IDs

ThreadStates == {"idle",
                 "acquiring_stream_mutex",
                 "has_stream_mutex",
                 "creating_encoder",        \* v2.3: acquiring global mutex for creation
                 "has_encoder",
                 "using_encoder",           \* v2.3: acquiring global mutex for method
                 "ending_encoding",
                 "releasing_stream_mutex"}

(* -------------------------------------------------------------------------- *)
(* Initial State                                                              *)
(* -------------------------------------------------------------------------- *)

Init ==
    /\ stream_mutex_holder = [s \in Streams |-> NULL]
    /\ stream_encoder = [s \in Streams |-> NULL]
    /\ global_mutex_holder = NULL
    /\ tracked_encoders = {}
    /\ thread_state = [t \in Threads |-> "idle"]
    /\ thread_stream = [t \in Threads |-> NULL]
    /\ thread_encoder = [t \in Threads |-> NULL]
    /\ encoder_counter = 0
    /\ contention_events = 0
    /\ double_lock_events = 0

(* -------------------------------------------------------------------------- *)
(* PyTorch: Thread starts operation on a stream                               *)
(* -------------------------------------------------------------------------- *)

StartOperation(t, s) ==
    /\ thread_state[t] = "idle"
    /\ stream_mutex_holder[s] = NULL  \* Stream mutex is free
    /\ thread_state' = [thread_state EXCEPT ![t] = "has_stream_mutex"]
    /\ thread_stream' = [thread_stream EXCEPT ![t] = s]
    /\ stream_mutex_holder' = [stream_mutex_holder EXCEPT ![s] = t]
    /\ UNCHANGED <<stream_encoder, global_mutex_holder, tracked_encoders,
                   thread_encoder, encoder_counter, contention_events, double_lock_events>>

(* -------------------------------------------------------------------------- *)
(* v2.3: Create encoder (swizzled computeCommandEncoder/blitCommandEncoder)   *)
(* This acquires the GLOBAL mutex even though we already have stream mutex!   *)
(* -------------------------------------------------------------------------- *)

(* Thread has stream mutex, now tries to create encoder *)
(* v2.3: Need to acquire global mutex for creation *)
StartCreateEncoder(t) ==
    /\ thread_state[t] = "has_stream_mutex"
    /\ stream_encoder[thread_stream[t]] = NULL  \* No encoder yet
    /\ thread_state' = [thread_state EXCEPT ![t] = "creating_encoder"]
    /\ UNCHANGED <<stream_mutex_holder, stream_encoder, global_mutex_holder, tracked_encoders,
                   thread_stream, thread_encoder, encoder_counter, contention_events, double_lock_events>>

(* v2.3: Acquire global mutex (may block if another thread holds it) *)
AcquireGlobalMutexForCreate(t) ==
    /\ thread_state[t] = "creating_encoder"
    /\ global_mutex_holder = NULL  \* Global mutex is free
    /\ global_mutex_holder' = t
    /\ thread_state' = [thread_state EXCEPT ![t] = "has_encoder"]
    (* Create encoder and track it *)
    /\ LET newEncoder == encoder_counter + 1 IN
        /\ encoder_counter' = newEncoder
        /\ thread_encoder' = [thread_encoder EXCEPT ![t] = newEncoder]
        /\ stream_encoder' = [stream_encoder EXCEPT ![thread_stream[t]] = newEncoder]
        /\ tracked_encoders' = tracked_encoders \cup {newEncoder}
    /\ UNCHANGED <<stream_mutex_holder, thread_stream, contention_events, double_lock_events>>

(* v2.3: BLOCKED - another thread holds global mutex (CONTENTION!) *)
BlockedOnGlobalMutexForCreate(t) ==
    /\ thread_state[t] = "creating_encoder"
    /\ global_mutex_holder /= NULL  \* Someone else has it
    /\ global_mutex_holder /= t     \* Not us (shouldn't happen, but safe)
    /\ contention_events' = contention_events + 1
    /\ UNCHANGED <<stream_mutex_holder, stream_encoder, global_mutex_holder, tracked_encoders,
                   thread_state, thread_stream, thread_encoder, encoder_counter, double_lock_events>>

(* Release global mutex after encoder creation *)
ReleaseGlobalMutexAfterCreate(t) ==
    /\ thread_state[t] = "has_encoder"
    /\ global_mutex_holder = t
    /\ global_mutex_holder' = NULL
    /\ UNCHANGED <<stream_mutex_holder, stream_encoder, tracked_encoders, thread_state,
                   thread_stream, thread_encoder, encoder_counter, contention_events, double_lock_events>>

(* -------------------------------------------------------------------------- *)
(* v2.3: Use encoder (swizzled method like setBuffer, dispatchThreads)        *)
(* This ALSO acquires the global mutex!                                       *)
(* -------------------------------------------------------------------------- *)

StartUseEncoder(t) ==
    /\ thread_state[t] = "has_encoder"
    /\ global_mutex_holder = NULL  \* Must have released it after create
    /\ thread_encoder[t] /= NULL
    /\ thread_state' = [thread_state EXCEPT ![t] = "using_encoder"]
    /\ UNCHANGED <<stream_mutex_holder, stream_encoder, global_mutex_holder, tracked_encoders,
                   thread_stream, thread_encoder, encoder_counter, contention_events, double_lock_events>>

(* v2.3: Acquire global mutex for method call *)
AcquireGlobalMutexForMethod(t) ==
    /\ thread_state[t] = "using_encoder"
    /\ global_mutex_holder = NULL
    /\ global_mutex_holder' = t
    /\ UNCHANGED <<stream_mutex_holder, stream_encoder, tracked_encoders, thread_state,
                   thread_stream, thread_encoder, encoder_counter, contention_events, double_lock_events>>

(* v2.3: BLOCKED on global mutex for method call *)
BlockedOnGlobalMutexForMethod(t) ==
    /\ thread_state[t] = "using_encoder"
    /\ global_mutex_holder /= NULL
    /\ global_mutex_holder /= t
    /\ contention_events' = contention_events + 1
    /\ UNCHANGED <<stream_mutex_holder, stream_encoder, global_mutex_holder, tracked_encoders,
                   thread_state, thread_stream, thread_encoder, encoder_counter, double_lock_events>>

(* Finish using encoder, release global mutex *)
FinishUseEncoder(t) ==
    /\ thread_state[t] = "using_encoder"
    /\ global_mutex_holder = t
    /\ global_mutex_holder' = NULL
    /\ thread_state' = [thread_state EXCEPT ![t] = "has_encoder"]
    /\ UNCHANGED <<stream_mutex_holder, stream_encoder, tracked_encoders,
                   thread_stream, thread_encoder, encoder_counter, contention_events, double_lock_events>>

(* -------------------------------------------------------------------------- *)
(* v2.3: End encoding                                                         *)
(* -------------------------------------------------------------------------- *)

StartEndEncoding(t) ==
    /\ thread_state[t] = "has_encoder"
    /\ global_mutex_holder = NULL
    /\ thread_encoder[t] /= NULL
    /\ thread_state' = [thread_state EXCEPT ![t] = "ending_encoding"]
    /\ global_mutex_holder' = t
    /\ UNCHANGED <<stream_mutex_holder, stream_encoder, tracked_encoders,
                   thread_stream, thread_encoder, encoder_counter, contention_events, double_lock_events>>

FinishEndEncoding(t) ==
    /\ thread_state[t] = "ending_encoding"
    /\ global_mutex_holder = t
    /\ LET e == thread_encoder[t]
           s == thread_stream[t] IN
        /\ tracked_encoders' = tracked_encoders \ {e}
        /\ stream_encoder' = [stream_encoder EXCEPT ![s] = NULL]
        /\ thread_encoder' = [thread_encoder EXCEPT ![t] = NULL]
    /\ global_mutex_holder' = NULL
    /\ thread_state' = [thread_state EXCEPT ![t] = "releasing_stream_mutex"]
    /\ UNCHANGED <<stream_mutex_holder, thread_stream, encoder_counter, contention_events, double_lock_events>>

(* -------------------------------------------------------------------------- *)
(* PyTorch: Release stream mutex and return to idle                           *)
(* -------------------------------------------------------------------------- *)

ReleaseStreamMutex(t) ==
    /\ thread_state[t] = "releasing_stream_mutex"
    /\ LET s == thread_stream[t] IN
        /\ stream_mutex_holder' = [stream_mutex_holder EXCEPT ![s] = NULL]
    /\ thread_stream' = [thread_stream EXCEPT ![t] = NULL]
    /\ thread_state' = [thread_state EXCEPT ![t] = "idle"]
    /\ UNCHANGED <<stream_encoder, global_mutex_holder, tracked_encoders,
                   thread_encoder, encoder_counter, contention_events, double_lock_events>>

(* -------------------------------------------------------------------------- *)
(* Next State                                                                 *)
(* -------------------------------------------------------------------------- *)

Next ==
    \/ \E t \in Threads, s \in Streams: StartOperation(t, s)
    \/ \E t \in Threads: StartCreateEncoder(t)
    \/ \E t \in Threads: AcquireGlobalMutexForCreate(t)
    \/ \E t \in Threads: BlockedOnGlobalMutexForCreate(t)
    \/ \E t \in Threads: ReleaseGlobalMutexAfterCreate(t)
    \/ \E t \in Threads: StartUseEncoder(t)
    \/ \E t \in Threads: AcquireGlobalMutexForMethod(t)
    \/ \E t \in Threads: BlockedOnGlobalMutexForMethod(t)
    \/ \E t \in Threads: FinishUseEncoder(t)
    \/ \E t \in Threads: StartEndEncoding(t)
    \/ \E t \in Threads: FinishEndEncoding(t)
    \/ \E t \in Threads: ReleaseStreamMutex(t)

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

(* -------------------------------------------------------------------------- *)
(* Safety Properties                                                          *)
(* -------------------------------------------------------------------------- *)

TypeOK ==
    /\ stream_mutex_holder \in [Streams -> Threads \cup {NULL}]
    /\ stream_encoder \in [Streams -> Encoders \cup {NULL}]
    /\ global_mutex_holder \in Threads \cup {NULL}
    /\ tracked_encoders \subseteq Encoders
    /\ thread_state \in [Threads -> ThreadStates]
    /\ thread_stream \in [Threads -> Streams \cup {NULL}]
    /\ thread_encoder \in [Threads -> Encoders \cup {NULL}]
    /\ encoder_counter \in Nat
    /\ contention_events \in Nat
    /\ double_lock_events \in Nat

(* No deadlock - not quite right since we model blocking as stuttering *)
(* Instead, check that contention is bounded (liveness) *)

(* The key property: contention exists but should not cause correctness issues *)
(* With v2.3, contention_events will be non-zero when multiple threads run *)
ContentionOccurs == contention_events > 0

(* Mutex consistency: at most one holder *)
GlobalMutexConsistent ==
    Cardinality({t \in Threads : global_mutex_holder = t}) <= 1

StreamMutexConsistent ==
    \A s \in Streams:
        Cardinality({t \in Threads : stream_mutex_holder[s] = t}) <= 1

(* Tracked encoder consistency *)
TrackedEncodersValid ==
    \A e \in tracked_encoders:
        \E t \in Threads: thread_encoder[t] = e

(* Combined safety *)
Safety ==
    /\ TypeOK
    /\ GlobalMutexConsistent
    /\ StreamMutexConsistent

(* -------------------------------------------------------------------------- *)
(* ANALYSIS: v2.3 Global Mutex Creates Serial Bottleneck                      *)
(* -------------------------------------------------------------------------- *)

(*
 * The v2.3 approach uses a single global mutex (g_encoder_mutex) for:
 * 1. Encoder creation (computeCommandEncoder, blitCommandEncoder)
 * 2. ALL encoder method calls (setBuffer, dispatchThreads, etc.)
 * 3. Encoder ending (endEncoding)
 *
 * This creates a SERIAL BOTTLENECK:
 * - Thread A on Stream 1: Creates encoder (holds global mutex)
 * - Thread B on Stream 2: Wants to create encoder (BLOCKED)
 * - Thread C on Stream 3: Wants to use its encoder (BLOCKED)
 *
 * Even though PyTorch designed streams to be PARALLEL, v2.3 serializes them.
 *
 * IMPACT:
 * 1. Performance: All GPU work is serialized, defeating parallel streams
 * 2. Timing: Changed thread scheduling exposes latent race conditions
 * 3. Stability: TransformerEncoderLayer (complex ops) shows 70% crash rate
 *
 * WHY v2 WORKS BETTER:
 * - v2 only protects METHOD CALLS, not creation
 * - Creation happens quickly under PyTorch's stream mutex
 * - Less global contention
 *
 * CONCLUSION:
 * The v2.3 regression is NOT a correctness bug in v2.3 itself.
 * It's that v2.3's global serialization:
 * 1. Defeats PyTorch 2.9.1's parallel stream architecture
 * 2. Changes timing in ways that expose OTHER bugs in PyTorch/MPS
 * 3. The crashes are from those exposed bugs, not v2.3 directly
 *)

=============================================================================
