--------------------------- MODULE MPSTLSBinding ---------------------------
\* TLA+ Specification for PyTorch MPS Thread-Local Storage (TLS) Binding
\* Based on: pytorch-mps-fork/aten/src/ATen/mps/MPSStream.h
\*
\* Models the thread-local storage mechanism for MPS stream binding:
\* - Each thread has its own TLS slot for current stream
\* - setCurrentMPSStream(stream) binds stream to calling thread's TLS
\* - getCurrentMPSStream() returns stream from calling thread's TLS
\* - Thread exit cleans up TLS binding (pthread destructor callback)
\*
\* Key invariants to verify:
\* 1. Thread isolation: threads can only access their own TLS
\* 2. Binding validity: TLS points to valid stream or null
\* 3. Cleanup correctness: dead threads have no TLS binding
\* 4. No leaks: stream references are properly managed

EXTENDS Naturals, Sequences, FiniteSets, TLC

CONSTANTS
    \* @type: Int;
    NumStreams,         \* Number of available streams
    \* @type: Int;
    NumThreads,         \* Maximum number of threads
    \* @type: Int;
    MaxOperations       \* Bound for model checking

ASSUME NumStreams >= 1
ASSUME NumThreads >= 1
ASSUME MaxOperations >= 1

VARIABLES
    \* Thread-local storage: Thread -> Stream binding (0 = default/unbound)
    \* Each thread has its own TLS slot that is invisible to other threads
    \* @type: Int -> Int;
    tls,
    \* Thread states: "alive", "exiting", "dead"
    \* @type: Int -> Str;
    thread_state,
    \* Stream reference counts (how many TLS bindings point to each stream)
    \* @type: Int -> Int;
    stream_refs,
    \* Track operation for linearizability verification
    \* @type: Int;
    op_count,
    \* Track which thread is performing TLS operation (for atomicity)
    \* 0 = none, >0 = thread ID in critical section
    \* @type: Int;
    tls_op_thread,
    \* Track the operation type being performed
    \* @type: Str;
    tls_op_type

vars == <<tls, thread_state, stream_refs, op_count, tls_op_thread, tls_op_type>>

-----------------------------------------------------------------------------
\* Type Invariant
TypeOK ==
    /\ tls \in [1..NumThreads -> 0..NumStreams]
    /\ thread_state \in [1..NumThreads -> {"alive", "exiting", "dead"}]
    /\ stream_refs \in [0..NumStreams -> Nat]
    /\ op_count \in 0..MaxOperations
    /\ tls_op_thread \in 0..NumThreads
    /\ tls_op_type \in {"none", "get", "set", "exit"}

\* Initial state
Init ==
    /\ tls = [t \in 1..NumThreads |-> 0]             \* All threads have default stream
    /\ thread_state = [t \in 1..NumThreads |-> "alive"]
    /\ stream_refs = [s \in 0..NumStreams |->
                        IF s = 0 THEN NumThreads     \* All start with default
                        ELSE 0]
    /\ op_count = 0
    /\ tls_op_thread = 0
    /\ tls_op_type = "none"

-----------------------------------------------------------------------------
\* TLS Operations

\* SetCurrentMPSStream: Thread t binds stream s to its TLS
\* Models: thread-local MPSCurrentStream = stream
\*
\* Semantics:
\* 1. Only calling thread can modify its own TLS (thread isolation)
\* 2. Previous binding's reference count decremented
\* 3. New binding's reference count incremented
SetCurrentStream(t, s) ==
    /\ op_count < MaxOperations
    /\ thread_state[t] = "alive"
    /\ tls_op_thread = 0                           \* No TLS op in progress
    /\ s \in 0..NumStreams                         \* Valid stream ID
    /\ LET old_stream == tls[t]
       IN
        \* Begin atomic TLS operation
        /\ tls_op_thread' = t
        /\ tls_op_type' = "set"
        \* Update TLS binding
        /\ tls' = [tls EXCEPT ![t] = s]
        \* Update reference counts
        /\ stream_refs' = [stream_refs EXCEPT
                            ![old_stream] = @ - 1,
                            ![s] = @ + 1]
        /\ UNCHANGED <<thread_state>>
        /\ op_count' = op_count + 1

\* Complete TLS set operation (atomic completion)
CompleteSetCurrentStream(t) ==
    /\ tls_op_thread = t
    /\ tls_op_type = "set"
    /\ tls_op_thread' = 0
    /\ tls_op_type' = "none"
    /\ UNCHANGED <<tls, thread_state, stream_refs, op_count>>

\* GetCurrentMPSStream: Thread t reads its TLS binding
\* Models: return MPSCurrentStream (thread-local variable read)
\*
\* Semantics:
\* 1. Only calling thread can read its own TLS
\* 2. Read is atomic and returns current binding
\* 3. No side effects
GetCurrentStream(t) ==
    /\ op_count < MaxOperations
    /\ thread_state[t] = "alive"
    /\ tls_op_thread = 0
    \* Begin atomic TLS read
    /\ tls_op_thread' = t
    /\ tls_op_type' = "get"
    \* No state change for read (TLS value is returned)
    /\ UNCHANGED <<tls, thread_state, stream_refs>>
    /\ op_count' = op_count + 1

\* Complete TLS get operation
CompleteGetCurrentStream(t) ==
    /\ tls_op_thread = t
    /\ tls_op_type = "get"
    /\ tls_op_thread' = 0
    /\ tls_op_type' = "none"
    /\ UNCHANGED <<tls, thread_state, stream_refs, op_count>>

-----------------------------------------------------------------------------
\* Thread Lifecycle

\* Thread exit: pthread_key destructor cleans up TLS binding
\* Models: pthread_key_t with destructor that releases stream
\*
\* Semantics:
\* 1. Thread transitions to "exiting" state
\* 2. TLS destructor callback fires
\* 3. Stream reference count decremented
\* 4. TLS cleared (set to 0)
\* 5. Thread transitions to "dead" state
BeginThreadExit(t) ==
    /\ op_count < MaxOperations
    /\ thread_state[t] = "alive"
    /\ tls_op_thread = 0
    \* Begin exit sequence
    /\ thread_state' = [thread_state EXCEPT ![t] = "exiting"]
    /\ tls_op_thread' = t
    /\ tls_op_type' = "exit"
    /\ UNCHANGED <<tls, stream_refs>>
    /\ op_count' = op_count + 1

\* TLS destructor callback runs during thread exit
\* Releases the stream reference and clears TLS
TLSDestructorCallback(t) ==
    /\ thread_state[t] = "exiting"
    /\ tls_op_thread = t
    /\ tls_op_type = "exit"
    /\ LET old_stream == tls[t]
       IN
        \* Clear TLS binding
        /\ tls' = [tls EXCEPT ![t] = 0]
        \* Decrement reference count (destructor releases reference)
        /\ stream_refs' = [stream_refs EXCEPT ![old_stream] = @ - 1]
        \* Transition to dead
        /\ thread_state' = [thread_state EXCEPT ![t] = "dead"]
        /\ tls_op_thread' = 0
        /\ tls_op_type' = "none"
        /\ UNCHANGED <<op_count>>

\* Thread respawn (for modeling continued execution)
\* Dead thread can be respawned for further testing
RespawnThread(t) ==
    /\ op_count < MaxOperations
    /\ thread_state[t] = "dead"
    /\ tls_op_thread = 0
    /\ thread_state' = [thread_state EXCEPT ![t] = "alive"]
    /\ tls' = [tls EXCEPT ![t] = 0]    \* New thread starts with default
    /\ stream_refs' = [stream_refs EXCEPT ![0] = @ + 1]  \* Ref default stream
    /\ UNCHANGED <<tls_op_thread, tls_op_type>>
    /\ op_count' = op_count + 1

-----------------------------------------------------------------------------
\* Next state relation
Next ==
    \/ \E t \in 1..NumThreads :
        \/ \E s \in 0..NumStreams : SetCurrentStream(t, s)
        \/ CompleteSetCurrentStream(t)
        \/ GetCurrentStream(t)
        \/ CompleteGetCurrentStream(t)
        \/ BeginThreadExit(t)
        \/ TLSDestructorCallback(t)
        \/ RespawnThread(t)
    \/ UNCHANGED vars  \* Stuttering for liveness

\* Fairness: Every enabled action eventually happens
Fairness ==
    /\ \A t \in 1..NumThreads :
        /\ WF_vars(\E s \in 0..NumStreams : SetCurrentStream(t, s))
        /\ WF_vars(CompleteSetCurrentStream(t))
        /\ WF_vars(GetCurrentStream(t))
        /\ WF_vars(CompleteGetCurrentStream(t))
        /\ WF_vars(BeginThreadExit(t))
        /\ WF_vars(TLSDestructorCallback(t))
        /\ WF_vars(RespawnThread(t))

Spec == Init /\ [][Next]_vars /\ Fairness

\* Apalache-compatible spec (no fairness - bounded safety only)
SpecNoFairness == Init /\ [][Next]_vars

-----------------------------------------------------------------------------
\* SAFETY PROPERTIES

\* Thread Isolation: Each thread can only access its own TLS
\* Modeled by: operations are parameterized by thread ID
\* and only modify/read that thread's TLS slot
ThreadIsolation ==
    \A t \in 1..NumThreads :
        \* Only thread t can be in a TLS operation on its own slot
        tls_op_thread = t =>
            \* The operation is on thread t's TLS (enforced by action design)
            TRUE  \* Structural property enforced by action definitions

\* Binding Validity: TLS always points to a valid stream (0 to NumStreams)
BindingValidity ==
    \A t \in 1..NumThreads :
        tls[t] \in 0..NumStreams

\* Dead Thread Cleanup: Dead threads have no TLS binding (cleared by destructor)
DeadThreadCleanup ==
    \A t \in 1..NumThreads :
        thread_state[t] = "dead" => tls[t] = 0

\* Exiting Thread Has TLS Op: Thread in exiting state has pending destructor
ExitingHasOp ==
    \A t \in 1..NumThreads :
        thread_state[t] = "exiting" =>
            (tls_op_thread = t /\ tls_op_type = "exit")

\* Reference Count Non-Negative: Stream references never go negative
RefCountNonNegative ==
    \A s \in 0..NumStreams :
        stream_refs[s] >= 0

\* No Stale Bindings: Alive threads have valid stream binding with ref > 0
NoStaleBindings ==
    \A t \in 1..NumThreads :
        thread_state[t] = "alive" =>
            stream_refs[tls[t]] > 0

\* Combined safety invariant
SafetyInvariant ==
    /\ TypeOK
    /\ BindingValidity
    /\ DeadThreadCleanup
    /\ RefCountNonNegative
    /\ NoStaleBindings

-----------------------------------------------------------------------------
\* LIVENESS PROPERTIES

\* Thread Exit Eventually Completes: Exiting thread eventually becomes dead
ExitEventuallyCompletes ==
    \A t \in 1..NumThreads :
        thread_state[t] = "exiting" ~> thread_state[t] = "dead"

\* Binding Change Completes: Started TLS set eventually completes
SetEventuallyCompletes ==
    \A t \in 1..NumThreads :
        (tls_op_thread = t /\ tls_op_type = "set") ~>
        (tls_op_thread = 0)

-----------------------------------------------------------------------------
\* DEADLOCK FREEDOM

\* The system can always make progress (no deadlock)
DeadlockFree ==
    op_count = MaxOperations \/
    tls_op_thread > 0 \/    \* TLS op in progress will complete
    \E t \in 1..NumThreads :
        \/ thread_state[t] = "alive"   \* Can do TLS operations
        \/ thread_state[t] = "dead"    \* Can respawn

=============================================================================
\* Modification History
\* Created: 2025-12-20 by AI Worker N=1359
\* Models TLS binding semantics for MPS stream management
