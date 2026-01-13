--------------------------- MODULE MPSCommandBuffer ---------------------------
(*
 * TLA+ Specification for MPSCommandBuffer Lifecycle Safety (Opportunity Map B1.4)
 *
 * This spec models the minimal command buffer lifecycle guarantees relied on by
 * the PyTorch MPS backend `MPSStream` implementation.
 *
 * PRIMARY GOAL:
 *   Encode the rule that `addCompletedHandler` MUST be registered on an
 *   uncommitted command buffer (Metal forbids attaching handlers after commit).
 *   This captures the real crash fixed in N=1305 where `_prevCommandBuffer`
 *   (already committed) was mistakenly used.
 *
 * MODELS:
 * - `MPSStream.mm`: `_commandBuffer` (uncommitted) and `_prevCommandBuffer`
 *   (committed but not yet waited/released).
 * - `flush()`: commits `_commandBuffer` and moves it to `_prevCommandBuffer`
 * - `commitAndWait()`: waits for `_prevCommandBuffer` and commits+waits current
 * - `addCompletedHandler()`: attaches handler to current (or newly created)
 *
 * KEY SAFETY INVARIANTS:
 * - CB.CommandBufferUncommitted: `_commandBuffer` is never committed
 * - CB.PrevBufferCommitted: `_prevCommandBuffer` is never uncommitted
 * - CB.NoHandlerAfterCommit: handler registration after commit is impossible
 * - CB.NoRefsToFree: stream never references a freed/reclaimed buffer
 *
 * OPTIONAL UNSAFE MODE:
 * Set `AllowUnsafePatterns = TRUE` in the .cfg to enable an action that models
 * the historical bug (handler attached to `_prevCommandBuffer`). TLC should
 * then produce a counterexample violating `NoIllegalHandler`.
 *
 * CODE ANCHORS:
 * - aten/src/ATen/mps/MPSStream.mm:153-170 (commandBufferLocked)
 * - aten/src/ATen/mps/MPSStream.mm:272-309 (flush, commitAndWait)
 * - aten/src/ATen/mps/MPSStream.mm:311-351 (addCompletedHandler)
 *
 * Created: 2025-12-20 (Iteration N=1355)
 *)

EXTENDS Naturals, FiniteSets, TLC

CONSTANTS
    \* @type: Int;
    NumBuffers,              \* Finite pool of buffer IDs for bounded checking
    \* @type: Bool;
    EnableCommitAndContinue, \* BOOLEAN: models _enableCommitAndContinue (not used by worker streams)
    \* @type: Bool;
    AllowUnsafePatterns      \* BOOLEAN: enables the historical bug action

\* Sentinel values (TLC doesn't support negative ints by default)
NoBuf == NumBuffers

BufIds == 0..(NumBuffers - 1)

BufStates == {"free", "uncommitted", "committed", "completed"}

VARIABLES
    \* @type: Int;
    command_buf,          \* Buffer currently being encoded (_commandBuffer), or NoBuf
    \* @type: Int;
    prev_buf,             \* Previous committed buffer retained for wait (_prevCommandBuffer), or NoBuf
    \* @type: Int -> Str;
    buf_state,            \* buf_id -> BufStates
    \* @type: Int -> Bool;
    handler_registered,   \* buf_id -> BOOLEAN (a handler was registered before commit)
    \* @type: Bool;
    illegal_handler_seen  \* BOOLEAN (TRUE if a post-commit handler registration was attempted)

vars == <<command_buf, prev_buf, buf_state, handler_registered, illegal_handler_seen>>

TypeOK ==
    /\ command_buf \in 0..NumBuffers
    /\ prev_buf \in 0..NumBuffers
    /\ buf_state \in [BufIds -> BufStates]
    /\ handler_registered \in [BufIds -> BOOLEAN]
    /\ illegal_handler_seen \in BOOLEAN

Init ==
    /\ command_buf = NoBuf
    /\ prev_buf = NoBuf
    /\ buf_state = [b \in BufIds |-> "free"]
    /\ handler_registered = [b \in BufIds |-> FALSE]
    /\ illegal_handler_seen = FALSE

\* Allocate a new uncommitted command buffer when `_commandBuffer` is nil.
AcquireCommandBuffer ==
    /\ command_buf = NoBuf
    /\ \E b \in BufIds :
        /\ buf_state[b] = "free"
        /\ command_buf' = b
        /\ buf_state' = [buf_state EXCEPT ![b] = "uncommitted"]
        /\ handler_registered' = [handler_registered EXCEPT ![b] = FALSE]
        /\ UNCHANGED <<prev_buf, illegal_handler_seen>>

\* Register a completion handler on the current (uncommitted) command buffer.
\* If `_commandBuffer` is nil, create it first (matches addCompletedHandler()).
AddCompletedHandlerSafe ==
    \/ /\ command_buf # NoBuf
       /\ buf_state[command_buf] = "uncommitted"
       /\ handler_registered' = [handler_registered EXCEPT ![command_buf] = TRUE]
       /\ UNCHANGED <<command_buf, prev_buf, buf_state, illegal_handler_seen>>
    \/ /\ command_buf = NoBuf
       /\ \E b \in BufIds :
          /\ buf_state[b] = "free"
          /\ command_buf' = b
          /\ buf_state' = [buf_state EXCEPT ![b] = "uncommitted"]
          /\ handler_registered' = [handler_registered EXCEPT ![b] = TRUE]
          /\ UNCHANGED <<prev_buf, illegal_handler_seen>>

\* Unsafe action: historical bug pattern (N=1305) where addCompletedHandler
\* attaches to `_prevCommandBuffer` after it has already been committed.
AddCompletedHandlerUnsafePrev ==
    /\ AllowUnsafePatterns = TRUE
    /\ prev_buf # NoBuf
    /\ buf_state[prev_buf] \in {"committed", "completed"}
    /\ handler_registered' = [handler_registered EXCEPT ![prev_buf] = TRUE]
    /\ illegal_handler_seen' = TRUE
    /\ UNCHANGED <<command_buf, prev_buf, buf_state>>

\* Model `flush()` / `commit()` when commitAndContinue is disabled:
\* commit `_commandBuffer`, move it to `_prevCommandBuffer`, and clear `_commandBuffer`.
Flush ==
    /\ EnableCommitAndContinue = FALSE
    /\ command_buf # NoBuf
    /\ buf_state[command_buf] = "uncommitted"
    /\ buf_state' = [buf_state EXCEPT ![command_buf] = "committed"]
    /\ prev_buf' = command_buf
    /\ command_buf' = NoBuf
    /\ UNCHANGED <<handler_registered, illegal_handler_seen>>

\* Model completion of any committed buffer (GPU execution completes asynchronously).
GPUComplete ==
    /\ \E b \in BufIds :
        /\ buf_state[b] = "committed"
        /\ buf_state' = [buf_state EXCEPT ![b] = "completed"]
        /\ UNCHANGED <<command_buf, prev_buf, handler_registered, illegal_handler_seen>>

\* Model the queue reclaiming completed buffers once the stream no longer references them.
ReclaimCompleted ==
    /\ \E b \in BufIds :
        /\ buf_state[b] = "completed"
        /\ b # command_buf
        /\ b # prev_buf
        /\ buf_state' = [buf_state EXCEPT ![b] = "free"]
        /\ handler_registered' = [handler_registered EXCEPT ![b] = FALSE]
        /\ UNCHANGED <<command_buf, prev_buf, illegal_handler_seen>>

\* Model `commitAndWait()` on `_prevCommandBuffer`: wait forces completion and releases it.
WaitPrev ==
    /\ prev_buf # NoBuf
    /\ buf_state[prev_buf] \in {"committed", "completed"}
    /\ buf_state' = [buf_state EXCEPT ![prev_buf] = "completed"]
    /\ prev_buf' = NoBuf
    /\ UNCHANGED <<command_buf, handler_registered, illegal_handler_seen>>

\* Model `commitAndWait()` on `_commandBuffer`: commit+wait completes it and clears `_commandBuffer`.
WaitCommand ==
    /\ command_buf # NoBuf
    /\ buf_state[command_buf] = "uncommitted"
    /\ buf_state' = [buf_state EXCEPT ![command_buf] = "completed"]
    /\ command_buf' = NoBuf
    /\ UNCHANGED <<prev_buf, handler_registered, illegal_handler_seen>>

Next ==
    \/ AcquireCommandBuffer
    \/ AddCompletedHandlerSafe
    \/ AddCompletedHandlerUnsafePrev
    \/ Flush
    \/ WaitPrev
    \/ WaitCommand
    \/ GPUComplete
    \/ ReclaimCompleted

Spec ==
    Init /\ [][Next]_vars

\* Safety properties
CommandBufferUncommitted ==
    command_buf = NoBuf \/ buf_state[command_buf] = "uncommitted"

PrevBufferCommitted ==
    prev_buf = NoBuf \/ buf_state[prev_buf] \in {"committed", "completed"}

NoAliasing ==
    (command_buf = NoBuf) \/ (prev_buf = NoBuf) \/ (command_buf # prev_buf)

NoUncommittedOrphans ==
    \A b \in BufIds : buf_state[b] = "uncommitted" => command_buf = b

NoRefsToFree ==
    /\ command_buf = NoBuf \/ buf_state[command_buf] # "free"
    /\ prev_buf = NoBuf \/ buf_state[prev_buf] # "free"

HandlersResetWhenFree ==
    \A b \in BufIds : buf_state[b] = "free" => handler_registered[b] = FALSE

NoIllegalHandler ==
    illegal_handler_seen = FALSE

=============================================================================
