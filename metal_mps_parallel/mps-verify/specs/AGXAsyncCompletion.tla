---------------------------- MODULE AGXAsyncCompletion ----------------------------
(*
 * AGX Async Completion Handler Race Model
 *
 * This TLA+ specification models the interaction between Metal command buffer
 * completion handlers and user-space operations.
 *
 * GOAL: Formally model how async completion handlers can race with:
 *   1. Encoder operations on the same buffer
 *   2. Resource cleanup (dealloc/release)
 *   3. Context invalidation
 *
 * MOTIVATION: The MPSGraph race conditions that persist even with our encoder
 * mutex may be caused by completion handlers running asynchronously on
 * Apple's internal dispatch queues, bypassing user-space synchronization.
 *
 * Based on Metal documentation and observed behavior:
 * - Completion handlers run on an internal serial queue
 * - Multiple command buffers can complete out-of-order (if not using events)
 * - Completion handlers can access resources that user code thinks are safe
 *)

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    NumThreads,         \* Number of user threads (e.g., 4)
    NumBuffers,         \* Number of command buffers in flight (e.g., 4)
    NumResources        \* Number of GPU resources (e.g., 4)

ASSUME NumThreads \in Nat /\ NumThreads > 0
ASSUME NumBuffers \in Nat /\ NumBuffers > 0
ASSUME NumResources \in Nat /\ NumResources > 0

VARIABLES
    (* Command buffer state *)
    buffer_state,       \* BufferId -> BufferState
    buffer_owner,       \* BufferId -> Thread | NULL
    buffer_resources,   \* BufferId -> Set of ResourceIds used

    (* Resource state *)
    resource_state,     \* ResourceId -> {allocated, deallocating, freed}
    resource_refcount,  \* ResourceId -> Nat (reference count)

    (* Completion handler queue (serial, FIFO) *)
    completion_queue,   \* Sequence of BufferIds waiting for completion

    (* Bug detection *)
    use_after_free,     \* TRUE if completion handler accesses freed resource
    double_complete,    \* TRUE if buffer completed twice
    race_count          \* Number of race conditions detected

vars == <<buffer_state, buffer_owner, buffer_resources,
          resource_state, resource_refcount,
          completion_queue, use_after_free, double_complete, race_count>>

Threads == 1..NumThreads
BufferIds == 1..NumBuffers
ResourceIds == 1..NumResources
NULL == 0

BufferStates == {
    "idle",             \* Not in use
    "encoding",         \* Being encoded by user thread
    "committed",        \* Committed to GPU, waiting execution
    "executing",        \* GPU is executing
    "completing",       \* Completion handler is running
    "completed"         \* Done, ready for reuse
}

ResourceStates == {"allocated", "deallocating", "freed"}

(* -------------------------------------------------------------------------- *)
(* Helper Functions                                                           *)
(* -------------------------------------------------------------------------- *)

(* Check if any resource in set is freed or deallocating *)
HasFreedResource(rset) ==
    \E r \in rset: resource_state[r] \in {"deallocating", "freed"}

(* -------------------------------------------------------------------------- *)
(* Initial State                                                              *)
(* -------------------------------------------------------------------------- *)

Init ==
    /\ buffer_state = [b \in BufferIds |-> "idle"]
    /\ buffer_owner = [b \in BufferIds |-> NULL]
    /\ buffer_resources = [b \in BufferIds |-> {}]
    /\ resource_state = [r \in ResourceIds |-> "allocated"]
    /\ resource_refcount = [r \in ResourceIds |-> 1]  \* User holds initial ref
    /\ completion_queue = <<>>
    /\ use_after_free = FALSE
    /\ double_complete = FALSE
    /\ race_count = 0

(* -------------------------------------------------------------------------- *)
(* User Thread Actions                                                        *)
(* -------------------------------------------------------------------------- *)

(* Thread starts encoding a command buffer *)
StartEncoding(t, b) ==
    /\ buffer_state[b] = "idle"
    /\ buffer_owner' = [buffer_owner EXCEPT ![b] = t]
    /\ buffer_state' = [buffer_state EXCEPT ![b] = "encoding"]
    /\ UNCHANGED <<buffer_resources, resource_state, resource_refcount,
                   completion_queue, use_after_free, double_complete, race_count>>

(* Thread encodes a resource reference into the buffer *)
EncodeResource(t, b, r) ==
    /\ buffer_state[b] = "encoding"
    /\ buffer_owner[b] = t
    /\ resource_state[r] = "allocated"  \* Only encode allocated resources
    /\ buffer_resources' = [buffer_resources EXCEPT ![b] = @ \cup {r}]
    (* GPU gets a reference to the resource *)
    /\ resource_refcount' = [resource_refcount EXCEPT ![r] = @ + 1]
    /\ UNCHANGED <<buffer_state, buffer_owner, resource_state,
                   completion_queue, use_after_free, double_complete, race_count>>

(* Thread commits buffer to GPU *)
CommitBuffer(t, b) ==
    /\ buffer_state[b] = "encoding"
    /\ buffer_owner[b] = t
    /\ buffer_state' = [buffer_state EXCEPT ![b] = "committed"]
    /\ UNCHANGED <<buffer_owner, buffer_resources, resource_state, resource_refcount,
                   completion_queue, use_after_free, double_complete, race_count>>

(* User thread releases a resource (starts deallocation) *)
(* THE BUG: User might release resource while GPU still using it *)
UserReleaseResource(r) ==
    /\ resource_state[r] = "allocated"
    /\ resource_refcount[r] >= 1
    /\ resource_refcount' = [resource_refcount EXCEPT ![r] = @ - 1]
    /\ IF resource_refcount[r] = 1  \* This release will free it
       THEN resource_state' = [resource_state EXCEPT ![r] = "deallocating"]
       ELSE UNCHANGED resource_state
    /\ UNCHANGED <<buffer_state, buffer_owner, buffer_resources,
                   completion_queue, use_after_free, double_complete, race_count>>

(* Deallocation completes *)
FinishDealloc(r) ==
    /\ resource_state[r] = "deallocating"
    /\ resource_refcount[r] = 0
    /\ resource_state' = [resource_state EXCEPT ![r] = "freed"]
    /\ UNCHANGED <<buffer_state, buffer_owner, buffer_resources, resource_refcount,
                   completion_queue, use_after_free, double_complete, race_count>>

(* -------------------------------------------------------------------------- *)
(* GPU Actions                                                                *)
(* -------------------------------------------------------------------------- *)

(* GPU starts executing a committed buffer *)
GPUStartExecution(b) ==
    /\ buffer_state[b] = "committed"
    /\ buffer_state' = [buffer_state EXCEPT ![b] = "executing"]
    /\ UNCHANGED <<buffer_owner, buffer_resources, resource_state, resource_refcount,
                   completion_queue, use_after_free, double_complete, race_count>>

(* GPU finishes execution, queues completion handler *)
GPUFinishExecution(b) ==
    /\ buffer_state[b] = "executing"
    /\ buffer_state' = [buffer_state EXCEPT ![b] = "completing"]
    /\ completion_queue' = Append(completion_queue, b)
    /\ UNCHANGED <<buffer_owner, buffer_resources, resource_state, resource_refcount,
                   use_after_free, double_complete, race_count>>

(* -------------------------------------------------------------------------- *)
(* Completion Handler Actions (runs on internal dispatch queue)               *)
(* -------------------------------------------------------------------------- *)

(* Completion handler runs - THE POTENTIAL RACE *)
RunCompletionHandler ==
    /\ Len(completion_queue) > 0
    /\ LET b == Head(completion_queue) IN
        /\ buffer_state[b] = "completing"
        (* Check if any resource was freed while we were executing *)
        /\ IF HasFreedResource(buffer_resources[b])
           THEN /\ use_after_free' = TRUE
                /\ race_count' = race_count + 1
           ELSE /\ UNCHANGED use_after_free
                /\ UNCHANGED race_count
        (* Release GPU's references to resources *)
        /\ resource_refcount' = [r \in ResourceIds |->
            IF r \in buffer_resources[b]
            THEN IF resource_refcount[r] > 0 THEN resource_refcount[r] - 1 ELSE 0
            ELSE resource_refcount[r]]
        (* Clear buffer state *)
        /\ buffer_state' = [buffer_state EXCEPT ![b] = "completed"]
        /\ buffer_owner' = [buffer_owner EXCEPT ![b] = NULL]
        /\ buffer_resources' = [buffer_resources EXCEPT ![b] = {}]
        /\ completion_queue' = Tail(completion_queue)
        /\ UNCHANGED <<resource_state, double_complete>>

(* Buffer can be reused after completion *)
ReuseBuffer(b) ==
    /\ buffer_state[b] = "completed"
    /\ buffer_state' = [buffer_state EXCEPT ![b] = "idle"]
    /\ UNCHANGED <<buffer_owner, buffer_resources, resource_state, resource_refcount,
                   completion_queue, use_after_free, double_complete, race_count>>

(* -------------------------------------------------------------------------- *)
(* Anomalous Action: Completion handler tries to run on already-completed buffer *)
(* -------------------------------------------------------------------------- *)

DoubleFire(b) ==
    /\ buffer_state[b] = "completed"
    /\ double_complete' = TRUE
    /\ race_count' = race_count + 1
    /\ UNCHANGED <<buffer_state, buffer_owner, buffer_resources,
                   resource_state, resource_refcount, completion_queue, use_after_free>>

(* -------------------------------------------------------------------------- *)
(* Next State                                                                 *)
(* -------------------------------------------------------------------------- *)

Next ==
    \/ \E t \in Threads, b \in BufferIds: StartEncoding(t, b)
    \/ \E t \in Threads, b \in BufferIds, r \in ResourceIds: EncodeResource(t, b, r)
    \/ \E t \in Threads, b \in BufferIds: CommitBuffer(t, b)
    \/ \E r \in ResourceIds: UserReleaseResource(r)
    \/ \E r \in ResourceIds: FinishDealloc(r)
    \/ \E b \in BufferIds: GPUStartExecution(b)
    \/ \E b \in BufferIds: GPUFinishExecution(b)
    \/ RunCompletionHandler
    \/ \E b \in BufferIds: ReuseBuffer(b)
    \/ \E b \in BufferIds: DoubleFire(b)

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

(* -------------------------------------------------------------------------- *)
(* Properties                                                                 *)
(* -------------------------------------------------------------------------- *)

(* Safety: No use-after-free in completion handlers *)
NoUseAfterFree == use_after_free = FALSE

(* Safety: No double completion *)
NoDoubleComplete == double_complete = FALSE

(* Combined safety property *)
SafeCompletion == NoUseAfterFree /\ NoDoubleComplete

(* Liveness: Completion handlers eventually run *)
HandlersEventuallyRun ==
    \A b \in BufferIds:
        buffer_state[b] = "completing" ~> buffer_state[b] = "completed"

(* Type invariant *)
TypeOK ==
    /\ buffer_state \in [BufferIds -> BufferStates]
    /\ buffer_owner \in [BufferIds -> Threads \cup {NULL}]
    /\ buffer_resources \in [BufferIds -> SUBSET ResourceIds]
    /\ resource_state \in [ResourceIds -> ResourceStates]
    /\ resource_refcount \in [ResourceIds -> Nat]
    /\ completion_queue \in Seq(BufferIds)
    /\ use_after_free \in BOOLEAN
    /\ double_complete \in BOOLEAN
    /\ race_count \in Nat

(* -------------------------------------------------------------------------- *)
(* What This Model Demonstrates                                               *)
(* -------------------------------------------------------------------------- *)
(*
 * This model shows that async completion handlers can race with user-space
 * resource cleanup, even when user-space encoding is properly synchronized.
 *
 * The race window:
 *   1. User encodes resource R into buffer B
 *   2. User commits B to GPU
 *   3. GPU executes B
 *   4. MEANWHILE: User releases R (thinks B is done)
 *   5. R is freed
 *   6. Completion handler for B runs, accesses freed R
 *
 * This explains why MPSGraph operations can crash even with encoder mutex:
 * - Our mutex protects encoding operations
 * - But completion handlers run on Apple's internal queue
 * - User-space cannot synchronize with kernel-scheduled completions
 *
 * FIX (in Metal design): Metal should retain resources until completion
 * handler finishes. If it doesn't, this is an Apple bug.
 *)

=============================================================================
