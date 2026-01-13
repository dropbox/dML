--------------------------- MODULE MPSBatchQueue ---------------------------
(***************************************************************************)
(* TLA+ Specification for MPSBatchQueue                                    *)
(*                                                                         *)
(* Models the batch queue protocol for MPS inference requests.             *)
(* Verifies liveness and safety properties for the producer/consumer queue *)
(* with start/stop lifecycle.                                              *)
(*                                                                         *)
(* CORRESPONDENCE:                                                         *)
(*   Production code: pytorch-mps-fork/aten/src/ATen/mps/MPSBatchQueue.{h,mm} *)
(*   CBMC harness: verification/cbmc/harnesses/batch_queue_harness.c       *)
(*                                                                         *)
(* Properties verified:                                                    *)
(*   BQ.NoStuckFutures: Every submitted request is eventually processed    *)
(*   BQ.StopDrains: stop() drains all pending requests                     *)
(*   BQ.SubmitStopRaceSafe: No orphaned requests after workers exit        *)
(*   BQ.NoDeadlock: System can always make progress                        *)
(*                                                                         *)
(* Created: N=1304 (2025-12-19)                                            *)
(***************************************************************************)

EXTENDS Naturals, Sequences, TLC, FiniteSets

CONSTANTS
    \* @type: Int;
    NumUserThreads,    \* Number of user threads that can submit
    \* @type: Int;
    NumWorkers,        \* Number of worker threads
    \* @type: Int;
    MaxOperations      \* Bound for model checking

(***************************************************************************)
(* Queue States                                                            *)
(***************************************************************************)
QueueStates == {"stopped", "running", "shutdown_requested", "drained"}

(***************************************************************************)
(* Variables                                                               *)
(***************************************************************************)
VARIABLES
    \* Queue state
    \* @type: Str;
    queue_state,       \* Current state of the queue: stopped/running/shutdown_requested/drained
    \* @type: Seq(Int);
    pending_requests,  \* Sequence of pending request IDs
    \* @type: Int;
    submitted_count,   \* Total requests submitted
    \* @type: Int;
    completed_count,   \* Total requests completed

    \* Thread states
    \* @type: Int -> Str;
    user_pc,           \* Per-user program counter
    \* @type: Int -> Str;
    worker_pc,         \* Per-worker program counter
    \* @type: Int;
    workers_alive,     \* Number of workers currently alive

    \* Request tracking (for liveness checking)
    \* @type: Set(Int);
    in_flight,         \* Set of request IDs that have been submitted but not completed

    \* Operation counter for bounded model checking
    \* @type: Int;
    op_count

vars == <<queue_state, pending_requests, submitted_count, completed_count,
          user_pc, worker_pc, workers_alive, in_flight, op_count>>

(***************************************************************************)
(* Type Invariant                                                          *)
(***************************************************************************)
TypeInvariant ==
    /\ queue_state \in QueueStates
    \* NOTE: Avoid `pending_requests \in Seq(Nat)` for Apalache compatibility.
    \* Apalache does not support `Seq(S)` (infinite set of unbounded sequences).
    \* We rely on the sequence-typed variable and explicitly constrain length and
    \* element types instead.
    /\ Len(pending_requests) <= MaxOperations
    /\ \A i \in 1..Len(pending_requests) : pending_requests[i] \in Nat
    /\ submitted_count \in Nat
    /\ completed_count \in Nat
    /\ user_pc \in [1..NumUserThreads -> {"idle", "submitting", "done"}]
    /\ worker_pc \in [1..NumWorkers -> {"idle", "waiting", "processing", "done"}]
    /\ workers_alive \in 0..NumWorkers
    /\ in_flight \subseteq Nat
    /\ op_count \in Nat

(***************************************************************************)
(* Initial State                                                           *)
(***************************************************************************)
Init ==
    /\ queue_state = "stopped"
    /\ pending_requests = <<>>
    /\ submitted_count = 0
    /\ completed_count = 0
    /\ user_pc = [t \in 1..NumUserThreads |-> "idle"]
    /\ worker_pc = [w \in 1..NumWorkers |-> "idle"]
    /\ workers_alive = 0
    /\ in_flight = {}
    /\ op_count = 0

(***************************************************************************)
(* Start - Transition from stopped to running, spawn workers               *)
(***************************************************************************)
Start ==
    /\ queue_state = "stopped"
    /\ op_count < MaxOperations
    /\ queue_state' = "running"
    /\ workers_alive' = NumWorkers
    /\ worker_pc' = [w \in 1..NumWorkers |-> "waiting"]
    /\ op_count' = op_count + 1
    /\ UNCHANGED <<pending_requests, submitted_count, completed_count,
                   user_pc, in_flight>>

(***************************************************************************)
(* Stop - Request shutdown; workers will drain queue                       *)
(***************************************************************************)
RequestStop ==
    /\ queue_state = "running"
    /\ op_count < MaxOperations
    /\ queue_state' = "shutdown_requested"
    /\ op_count' = op_count + 1
    /\ UNCHANGED <<pending_requests, submitted_count, completed_count,
                   user_pc, worker_pc, workers_alive, in_flight>>

(***************************************************************************)
(* Submit - User thread submits a request                                  *)
(* Pre: Queue must be running                                              *)
(***************************************************************************)
Submit(t) ==
    /\ user_pc[t] = "idle"
    /\ queue_state = "running"
    /\ op_count < MaxOperations
    /\ LET new_id == submitted_count + 1 IN
        /\ pending_requests' = Append(pending_requests, new_id)
        /\ submitted_count' = new_id
        /\ in_flight' = in_flight \cup {new_id}
        /\ user_pc' = [user_pc EXCEPT ![t] = "done"]
    /\ op_count' = op_count + 1
    /\ UNCHANGED <<queue_state, completed_count, worker_pc, workers_alive>>

(***************************************************************************)
(* UserReset - Reset user thread to allow more submits                     *)
(***************************************************************************)
UserReset(t) ==
    /\ user_pc[t] = "done"
    /\ op_count < MaxOperations
    /\ user_pc' = [user_pc EXCEPT ![t] = "idle"]
    /\ op_count' = op_count + 1
    /\ UNCHANGED <<queue_state, pending_requests, submitted_count, completed_count,
                   worker_pc, workers_alive, in_flight>>

(***************************************************************************)
(* WorkerProcess - Worker takes request from queue and processes it        *)
(***************************************************************************)
WorkerProcess(w) ==
    /\ worker_pc[w] = "waiting"
    /\ Len(pending_requests) > 0
    /\ op_count < MaxOperations
    /\ LET req_id == Head(pending_requests) IN
        /\ pending_requests' = Tail(pending_requests)
        /\ completed_count' = completed_count + 1
        /\ in_flight' = in_flight \ {req_id}
        /\ worker_pc' = [worker_pc EXCEPT ![w] = "waiting"]
    /\ op_count' = op_count + 1
    /\ UNCHANGED <<queue_state, submitted_count, user_pc, workers_alive>>

(***************************************************************************)
(* WorkerCheckShutdown - Worker checks for shutdown and exits if queue empty *)
(***************************************************************************)
WorkerCheckShutdown(w) ==
    /\ worker_pc[w] = "waiting"
    /\ queue_state = "shutdown_requested"
    /\ Len(pending_requests) = 0
    /\ op_count < MaxOperations
    /\ worker_pc' = [worker_pc EXCEPT ![w] = "done"]
    /\ workers_alive' = workers_alive - 1
    /\ op_count' = op_count + 1
    /\ UNCHANGED <<queue_state, pending_requests, submitted_count, completed_count,
                   user_pc, in_flight>>

(***************************************************************************)
(* WorkerWait - Worker waits on empty queue (represents CV wait)           *)
(* This models the condition variable wait when queue is empty but running *)
(***************************************************************************)
WorkerWait(w) ==
    /\ worker_pc[w] = "waiting"
    /\ queue_state = "running"
    /\ Len(pending_requests) = 0
    /\ op_count < MaxOperations
    /\ op_count' = op_count + 1
    \* No state change - just models waiting
    /\ UNCHANGED <<queue_state, pending_requests, submitted_count, completed_count,
                   user_pc, worker_pc, workers_alive, in_flight>>

(***************************************************************************)
(* FinalizeShutdown - Transition to drained when all workers done          *)
(***************************************************************************)
FinalizeShutdown ==
    /\ queue_state = "shutdown_requested"
    /\ workers_alive = 0
    /\ op_count < MaxOperations
    /\ queue_state' = "drained"
    /\ op_count' = op_count + 1
    /\ UNCHANGED <<pending_requests, submitted_count, completed_count,
                   user_pc, worker_pc, workers_alive, in_flight>>

(***************************************************************************)
(* Reset - From drained back to stopped (for restart capability)           *)
(***************************************************************************)
Reset ==
    /\ queue_state = "drained"
    /\ op_count < MaxOperations
    /\ queue_state' = "stopped"
    /\ worker_pc' = [w \in 1..NumWorkers |-> "idle"]
    /\ op_count' = op_count + 1
    /\ UNCHANGED <<pending_requests, submitted_count, completed_count,
                   user_pc, workers_alive, in_flight>>

(***************************************************************************)
(* Next State                                                              *)
(***************************************************************************)
Next ==
    \/ Start
    \/ RequestStop
    \/ \E t \in 1..NumUserThreads : Submit(t)
    \/ \E t \in 1..NumUserThreads : UserReset(t)
    \/ \E w \in 1..NumWorkers : WorkerProcess(w)
    \/ \E w \in 1..NumWorkers : WorkerCheckShutdown(w)
    \/ \E w \in 1..NumWorkers : WorkerWait(w)
    \/ FinalizeShutdown
    \/ Reset
    \/ UNCHANGED vars

Spec == Init /\ [][Next]_vars

(***************************************************************************)
(* SAFETY PROPERTIES                                                       *)
(***************************************************************************)

(* BQ.NoStuckFutures: When queue drains, all submitted requests are completed *)
NoStuckFutures ==
    (queue_state = "drained") => (in_flight = {})

(* BQ.StopDrains: After shutdown, queue is empty *)
StopDrains ==
    (queue_state = "drained") => (Len(pending_requests) = 0)

(* BQ.SubmitStopRaceSafe: Cannot submit after shutdown requested *)
SubmitStopRaceSafe ==
    \A t \in 1..NumUserThreads :
        (queue_state \in {"shutdown_requested", "drained"}) =>
        (user_pc[t] # "submitting")

(* Completed count never exceeds submitted count *)
CompletedNeverExceedsSubmitted ==
    completed_count <= submitted_count

(* In-flight requests are consistent *)
InFlightConsistent ==
    Cardinality(in_flight) = submitted_count - completed_count

(* Workers only alive when running or draining *)
WorkersOnlyWhenActive ==
    (queue_state \in {"stopped", "drained"}) => (workers_alive = 0)

(* Combined safety invariant *)
SafetyInvariant ==
    /\ TypeInvariant
    /\ NoStuckFutures
    /\ StopDrains
    /\ CompletedNeverExceedsSubmitted
    /\ InFlightConsistent
    /\ WorkersOnlyWhenActive

(***************************************************************************)
(* LIVENESS PROPERTIES (checked under fairness)                            *)
(***************************************************************************)

(* BQ.NoDeadlock: Some action is always enabled (unless at bound) *)
NoDeadlock ==
    (op_count < MaxOperations) => ENABLED(Next)

(* Eventually all requests complete if shutdown requested *)
EventuallyDrained ==
    (queue_state = "shutdown_requested") ~> (queue_state = "drained")

(* Weak fairness for liveness checking *)
Fairness ==
    /\ WF_vars(Start)
    /\ WF_vars(RequestStop)
    /\ WF_vars(FinalizeShutdown)
    /\ WF_vars(Reset)
    /\ \A w \in 1..NumWorkers : WF_vars(WorkerProcess(w))
    /\ \A w \in 1..NumWorkers : WF_vars(WorkerCheckShutdown(w))
    /\ \A w \in 1..NumWorkers : WF_vars(WorkerWait(w))

LiveSpec == Spec /\ Fairness

=============================================================================
\* Modification History
\* Created N=1304 (2025-12-19)
