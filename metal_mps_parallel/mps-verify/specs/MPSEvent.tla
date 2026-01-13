---------------------------- MODULE MPSEvent ----------------------------
(*
 * MPS Event State Machine (Abstract Model)
 *
 * This TLA+ specification models MPSEvent lifecycle + notification behavior at
 * a conservative level to reason about callback lifetime safety and pool reuse.
 *
 * NOTE: The current fork implementation uses shared_ptr-backed in-use event
 * storage (see `pytorch-mps-fork/aten/src/ATen/mps/MPSEvent.h`) and an explicit
 * dispatch queue for Metal event notifications (see `MPSEvent.mm`). This spec
 * captures the intended safety obligations; it is not guaranteed to be a
 * line-by-line control-flow mirror of the current Objective-C++ code.
 *)

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    NumEvents,           \* Number of events in the pool (e.g., 3)
    NumThreads,          \* Number of threads (e.g., 3)
    NumCallbackStates    \* Callback state IDs (must be >= NumEvents * 2 for reuse)

ASSUME NumEvents \in Nat /\ NumEvents > 0
ASSUME NumThreads \in Nat /\ NumThreads > 0
ASSUME NumCallbackStates \in Nat /\ NumCallbackStates >= NumEvents * 2

VARIABLES
    \* Pool state
    pool_alive,              \* Is the event pool alive?
    pool_events,             \* Set of event IDs in the pool (available for reuse)
    in_use_events,           \* Set of event IDs currently acquired by threads

    \* Per-event state
    event_alive,             \* Event ID -> Bool (event object exists)
    event_callback_state,    \* Event ID -> callback state ID (current owner's state)
    event_listener,          \* Event ID -> Bool (has listener registered)
    event_signal_counter,    \* Event ID -> Nat (signal counter)
    event_mutex_holder,      \* Event ID -> Thread ID | 0 (who holds event mutex)

    \* Per-callback-state state (survives event destruction via shared_ptr)
    cb_state_alive,          \* CB State ID -> Bool (alive flag)
    cb_state_sync_completed, \* CB State ID -> Bool (sync_completed flag)
    cb_state_ref_count,      \* CB State ID -> Nat (number of shared_ptr refs)

    \* Callback execution state (models GPU completion callbacks)
    pending_callbacks,       \* Set of {callback_state_id, event_id} pairs

    \* Thread state
    pc,                      \* Thread -> program counter
    thread_event,            \* Thread -> event ID being operated on
    thread_op,               \* Thread -> operation type
    thread_cb_state,         \* Thread -> callback state captured for callback execution

    \* Tracking
    next_cb_state_id,        \* Counter for allocating new callback state IDs
    completed_ops            \* Count of completed operations

vars == <<pool_alive, pool_events, in_use_events, event_alive, event_callback_state,
          event_listener, event_signal_counter, event_mutex_holder, cb_state_alive,
          cb_state_sync_completed, cb_state_ref_count, pending_callbacks, pc,
          thread_event, thread_op, thread_cb_state, next_cb_state_id, completed_ops>>

Events == 1..NumEvents
Threads == 1..NumThreads
CallbackStates == 1..NumCallbackStates
NULL == 0

(* Type invariant *)
TypeOK ==
    /\ pool_alive \in BOOLEAN
    /\ pool_events \subseteq Events
    /\ in_use_events \subseteq Events
    /\ event_alive \in [Events -> BOOLEAN]
    /\ event_callback_state \in [Events -> (CallbackStates \cup {NULL})]
    /\ event_listener \in [Events -> BOOLEAN]
    /\ event_signal_counter \in [Events -> Nat]
    /\ event_mutex_holder \in [Events -> (Threads \cup {NULL})]
    /\ cb_state_alive \in [CallbackStates -> BOOLEAN]
    /\ cb_state_sync_completed \in [CallbackStates -> BOOLEAN]
    /\ cb_state_ref_count \in [CallbackStates -> Nat]
    /\ pending_callbacks \subseteq (CallbackStates \X Events)
    /\ pc \in [Threads -> {"idle", "acquire_event", "record_lock", "record_notify",
                           "record_done", "sync_lock", "sync_notify", "sync_wait",
                           "reset_lock", "reset_invalidate", "release_event",
                           "destroy_set_not_alive", "destroy_wait", "destroy_done",
                           "callback_start", "callback_check_alive", "callback_notify",
                           "done"}]
    /\ thread_event \in [Threads -> (Events \cup {NULL})]
    /\ thread_op \in [Threads -> {"none", "acquire", "record", "sync", "reset", "destroy", "callback"}]
    /\ thread_cb_state \in [Threads -> (CallbackStates \cup {NULL})]
    /\ next_cb_state_id \in 1..(NumCallbackStates + 1)
    /\ completed_ops \in Nat

(* Initial state *)
Init ==
    /\ pool_alive = TRUE
    /\ pool_events = Events  \* All events in pool initially
    /\ in_use_events = {}
    /\ event_alive = [e \in Events |-> TRUE]
    /\ event_callback_state = [e \in Events |-> e]  \* Initial state IDs match event IDs
    /\ event_listener = [e \in Events |-> FALSE]
    /\ event_signal_counter = [e \in Events |-> 0]
    /\ event_mutex_holder = [e \in Events |-> NULL]
    /\ cb_state_alive = [cs \in CallbackStates |-> cs <= NumEvents]  \* First NumEvents are alive
    /\ cb_state_sync_completed = [cs \in CallbackStates |-> TRUE]
    /\ cb_state_ref_count = [cs \in CallbackStates |-> IF cs <= NumEvents THEN 1 ELSE 0]
    /\ pending_callbacks = {}
    /\ pc = [t \in Threads |-> "idle"]
    /\ thread_event = [t \in Threads |-> NULL]
    /\ thread_op = [t \in Threads |-> "none"]
    /\ thread_cb_state = [t \in Threads |-> NULL]
    /\ next_cb_state_id = NumEvents + 1
    /\ completed_ops = 0

-----------------------------------------------------------------------------
(* HELPER DEFINITIONS *)

\* Check if thread holds event's mutex
HoldsEventMutex(t, e) == event_mutex_holder[e] = t

\* Can acquire event mutex (not held by anyone)
CanAcquireEventMutex(e) == event_mutex_holder[e] = NULL

\* Allocate a new callback state ID
AllocateCallbackState ==
    /\ next_cb_state_id <= NumCallbackStates
    /\ next_cb_state_id' = next_cb_state_id + 1

-----------------------------------------------------------------------------
(* ACQUIRE EVENT FROM POOL *)

StartAcquireEvent(t) ==
    /\ pc[t] = "idle"
    /\ pool_alive
    /\ pool_events # {}  \* Pool has available events
    /\ thread_op' = [thread_op EXCEPT ![t] = "acquire"]
    /\ pc' = [pc EXCEPT ![t] = "acquire_event"]
    /\ UNCHANGED <<pool_alive, pool_events, in_use_events, event_alive, event_callback_state,
                   event_listener, event_signal_counter, event_mutex_holder, cb_state_alive,
                   cb_state_sync_completed, cb_state_ref_count, pending_callbacks, thread_event,
                   thread_cb_state, next_cb_state_id, completed_ops>>

AcquireEvent(t) ==
    /\ pc[t] = "acquire_event"
    /\ pool_alive
    /\ pool_events # {}
    /\ LET e == CHOOSE x \in pool_events : TRUE
       IN /\ pool_events' = pool_events \ {e}
          /\ in_use_events' = in_use_events \cup {e}
          /\ thread_event' = [thread_event EXCEPT ![t] = e]
    /\ pc' = [pc EXCEPT ![t] = "done"]
    /\ completed_ops' = completed_ops + 1
    /\ UNCHANGED <<pool_alive, event_alive, event_callback_state, event_listener,
                   event_signal_counter, event_mutex_holder, cb_state_alive,
                   cb_state_sync_completed, cb_state_ref_count, pending_callbacks,
                   thread_op, thread_cb_state, next_cb_state_id>>

-----------------------------------------------------------------------------
(* RECORD EVENT (schedules callback) *)

StartRecord(t) ==
    /\ pc[t] = "idle"
    /\ pool_alive
    /\ thread_event[t] # NULL
    /\ thread_event[t] \in in_use_events
    /\ thread_op' = [thread_op EXCEPT ![t] = "record"]
    /\ pc' = [pc EXCEPT ![t] = "record_lock"]
    /\ UNCHANGED <<pool_alive, pool_events, in_use_events, event_alive, event_callback_state,
                   event_listener, event_signal_counter, event_mutex_holder, cb_state_alive,
                   cb_state_sync_completed, cb_state_ref_count, pending_callbacks, thread_event,
                   thread_cb_state, next_cb_state_id, completed_ops>>

RecordLock(t) ==
    /\ pc[t] = "record_lock"
    /\ thread_event[t] # NULL
    /\ LET e == thread_event[t]
       IN /\ CanAcquireEventMutex(e)
          /\ event_mutex_holder' = [event_mutex_holder EXCEPT ![e] = t]
    /\ pc' = [pc EXCEPT ![t] = "record_notify"]
    /\ UNCHANGED <<pool_alive, pool_events, in_use_events, event_alive, event_callback_state,
                   event_listener, event_signal_counter, cb_state_alive, cb_state_sync_completed,
                   cb_state_ref_count, pending_callbacks, thread_event, thread_op, thread_cb_state,
                   next_cb_state_id, completed_ops>>

RecordNotify(t) ==
    /\ pc[t] = "record_notify"
    /\ thread_event[t] # NULL
    /\ LET e == thread_event[t]
           cs == event_callback_state[e]
       IN /\ HoldsEventMutex(t, e)
          /\ cs # NULL
          \* Capture callback state and schedule callback
          /\ cb_state_sync_completed' = [cb_state_sync_completed EXCEPT ![cs] = FALSE]
          /\ cb_state_ref_count' = [cb_state_ref_count EXCEPT ![cs] = @ + 1]  \* Callback holds ref
          /\ pending_callbacks' = pending_callbacks \cup {<<cs, e>>}
          /\ event_signal_counter' = [event_signal_counter EXCEPT ![e] = @ + 1]
          /\ event_listener' = [event_listener EXCEPT ![e] = TRUE]
          /\ event_mutex_holder' = [event_mutex_holder EXCEPT ![e] = NULL]  \* Release mutex
    /\ pc' = [pc EXCEPT ![t] = "record_done"]
    /\ UNCHANGED <<pool_alive, pool_events, in_use_events, event_alive, event_callback_state,
                   cb_state_alive, thread_event, thread_op, thread_cb_state, next_cb_state_id,
                   completed_ops>>

RecordDone(t) ==
    /\ pc[t] = "record_done"
    /\ pc' = [pc EXCEPT ![t] = "done"]
    /\ completed_ops' = completed_ops + 1
    /\ UNCHANGED <<pool_alive, pool_events, in_use_events, event_alive, event_callback_state,
                   event_listener, event_signal_counter, event_mutex_holder, cb_state_alive,
                   cb_state_sync_completed, cb_state_ref_count, pending_callbacks, thread_event,
                   thread_op, thread_cb_state, next_cb_state_id>>

-----------------------------------------------------------------------------
(* SYNCHRONIZE (wait for callback) *)

StartSync(t) ==
    /\ pc[t] = "idle"
    /\ pool_alive
    /\ thread_event[t] # NULL
    /\ thread_event[t] \in in_use_events
    /\ thread_op' = [thread_op EXCEPT ![t] = "sync"]
    /\ pc' = [pc EXCEPT ![t] = "sync_lock"]
    /\ UNCHANGED <<pool_alive, pool_events, in_use_events, event_alive, event_callback_state,
                   event_listener, event_signal_counter, event_mutex_holder, cb_state_alive,
                   cb_state_sync_completed, cb_state_ref_count, pending_callbacks, thread_event,
                   thread_cb_state, next_cb_state_id, completed_ops>>

SyncLock(t) ==
    /\ pc[t] = "sync_lock"
    /\ thread_event[t] # NULL
    /\ LET e == thread_event[t]
       IN /\ CanAcquireEventMutex(e)
          /\ event_mutex_holder' = [event_mutex_holder EXCEPT ![e] = t]
    /\ pc' = [pc EXCEPT ![t] = "sync_notify"]
    /\ UNCHANGED <<pool_alive, pool_events, in_use_events, event_alive, event_callback_state,
                   event_listener, event_signal_counter, cb_state_alive, cb_state_sync_completed,
                   cb_state_ref_count, pending_callbacks, thread_event, thread_op, thread_cb_state,
                   next_cb_state_id, completed_ops>>

SyncNotify(t) ==
    /\ pc[t] = "sync_notify"
    /\ thread_event[t] # NULL
    /\ LET e == thread_event[t]
           cs == event_callback_state[e]
       IN /\ HoldsEventMutex(t, e)
          /\ cs # NULL
          /\ \/ \* Already completed - skip notify
                /\ cb_state_sync_completed[cs]
                /\ event_mutex_holder' = [event_mutex_holder EXCEPT ![e] = NULL]
                /\ pc' = [pc EXCEPT ![t] = "done"]
                /\ completed_ops' = completed_ops + 1
                /\ UNCHANGED <<cb_state_sync_completed, cb_state_ref_count, pending_callbacks, thread_cb_state>>
             \/ \* Not completed - schedule notify callback and wait
                /\ ~cb_state_sync_completed[cs]
                \* Avoid unbounded ref_count growth: scheduling a callback that is already
                \* pending must be a no-op (it does not create a new asynchronous callback).
                /\ LET cb == <<cs, e>> IN
                     IF cb \in pending_callbacks
                     THEN /\ cb_state_ref_count' = cb_state_ref_count
                          /\ pending_callbacks' = pending_callbacks
                     ELSE /\ cb_state_ref_count' = [cb_state_ref_count EXCEPT ![cs] = @ + 1]
                          /\ pending_callbacks' = pending_callbacks \cup {cb}
                /\ event_mutex_holder' = [event_mutex_holder EXCEPT ![e] = NULL]
                /\ thread_cb_state' = [thread_cb_state EXCEPT ![t] = cs]
                /\ pc' = [pc EXCEPT ![t] = "sync_wait"]
                /\ UNCHANGED <<completed_ops, cb_state_sync_completed>>
    /\ UNCHANGED <<pool_alive, pool_events, in_use_events, event_alive, event_callback_state,
                   event_listener, event_signal_counter, cb_state_alive, thread_event, thread_op,
                   next_cb_state_id>>

SyncWait(t) ==
    /\ pc[t] = "sync_wait"
    /\ thread_cb_state[t] # NULL
    /\ LET cs == thread_cb_state[t]
       IN \* Wait until sync_completed (or model timeout by allowing escape)
          \/ /\ cb_state_sync_completed[cs]
             /\ pc' = [pc EXCEPT ![t] = "done"]
             /\ completed_ops' = completed_ops + 1
          \/ \* Model timeout (destructor can also timeout)
             /\ ~cb_state_sync_completed[cs]
             /\ pc' = [pc EXCEPT ![t] = "done"]  \* Timeout escape
             /\ UNCHANGED completed_ops
    /\ UNCHANGED <<pool_alive, pool_events, in_use_events, event_alive, event_callback_state,
                   event_listener, event_signal_counter, event_mutex_holder, cb_state_alive,
                   cb_state_sync_completed, cb_state_ref_count, pending_callbacks, thread_event,
                   thread_op, thread_cb_state, next_cb_state_id>>

-----------------------------------------------------------------------------
(* RESET EVENT (for pool reuse - 32.89/32.107 fix) *)

StartReset(t) ==
    /\ pc[t] = "idle"
    /\ pool_alive
    /\ thread_event[t] # NULL
    /\ thread_event[t] \in in_use_events
    /\ thread_op' = [thread_op EXCEPT ![t] = "reset"]
    /\ pc' = [pc EXCEPT ![t] = "reset_lock"]
    /\ UNCHANGED <<pool_alive, pool_events, in_use_events, event_alive, event_callback_state,
                   event_listener, event_signal_counter, event_mutex_holder, cb_state_alive,
                   cb_state_sync_completed, cb_state_ref_count, pending_callbacks, thread_event,
                   thread_cb_state, next_cb_state_id, completed_ops>>

ResetLock(t) ==
    /\ pc[t] = "reset_lock"
    /\ thread_event[t] # NULL
    /\ LET e == thread_event[t]
       IN /\ CanAcquireEventMutex(e)
          /\ event_mutex_holder' = [event_mutex_holder EXCEPT ![e] = t]
    /\ pc' = [pc EXCEPT ![t] = "reset_invalidate"]
    /\ UNCHANGED <<pool_alive, pool_events, in_use_events, event_alive, event_callback_state,
                   event_listener, event_signal_counter, cb_state_alive, cb_state_sync_completed,
                   cb_state_ref_count, pending_callbacks, thread_event, thread_op, thread_cb_state,
                   next_cb_state_id, completed_ops>>

ResetInvalidate(t) ==
    /\ pc[t] = "reset_invalidate"
    /\ thread_event[t] # NULL
    /\ next_cb_state_id <= NumCallbackStates  \* Can allocate new state
    /\ LET e == thread_event[t]
           old_cs == event_callback_state[e]
           new_cs == next_cb_state_id
       IN /\ HoldsEventMutex(t, e)
          /\ \/ \* Case 1: old_cs exists - invalidate it and create new
                /\ old_cs # NULL
                \* 32.89/32.107 fix: Set old state to not-alive, create new state
                /\ cb_state_alive' = [cb_state_alive EXCEPT ![old_cs] = FALSE, ![new_cs] = TRUE]
                /\ cb_state_ref_count' = [cb_state_ref_count EXCEPT ![old_cs] = @ - 1, ![new_cs] = 1]
             \/ \* Case 2: old_cs is NULL (after destroy) - just allocate new
                /\ old_cs = NULL
                /\ cb_state_alive' = [cb_state_alive EXCEPT ![new_cs] = TRUE]
                /\ cb_state_ref_count' = [cb_state_ref_count EXCEPT ![new_cs] = 1]
          /\ cb_state_sync_completed' = [cb_state_sync_completed EXCEPT ![new_cs] = TRUE]
          /\ event_callback_state' = [event_callback_state EXCEPT ![e] = new_cs]
          /\ event_listener' = [event_listener EXCEPT ![e] = FALSE]  \* Release listener
          /\ event_mutex_holder' = [event_mutex_holder EXCEPT ![e] = NULL]
          /\ next_cb_state_id' = next_cb_state_id + 1
    /\ pc' = [pc EXCEPT ![t] = "done"]
    /\ completed_ops' = completed_ops + 1
    /\ UNCHANGED <<pool_alive, pool_events, in_use_events, event_alive, event_signal_counter,
                   pending_callbacks, thread_event, thread_op, thread_cb_state>>

-----------------------------------------------------------------------------
(* RELEASE EVENT BACK TO POOL *)

StartReleaseEvent(t) ==
    /\ pc[t] = "idle"
    /\ pool_alive
    /\ thread_event[t] # NULL
    /\ thread_event[t] \in in_use_events
    /\ thread_op' = [thread_op EXCEPT ![t] = "destroy"]
    /\ pc' = [pc EXCEPT ![t] = "release_event"]
    /\ UNCHANGED <<pool_alive, pool_events, in_use_events, event_alive, event_callback_state,
                   event_listener, event_signal_counter, event_mutex_holder, cb_state_alive,
                   cb_state_sync_completed, cb_state_ref_count, pending_callbacks, thread_event,
                   thread_cb_state, next_cb_state_id, completed_ops>>

ReleaseEvent(t) ==
    /\ pc[t] = "release_event"
    /\ thread_event[t] # NULL
    /\ LET e == thread_event[t]
       IN /\ in_use_events' = in_use_events \ {e}
          /\ pool_events' = pool_events \cup {e}
    /\ pc' = [pc EXCEPT ![t] = "done"]
    /\ thread_event' = [thread_event EXCEPT ![t] = NULL]
    /\ completed_ops' = completed_ops + 1
    /\ UNCHANGED <<pool_alive, event_alive, event_callback_state, event_listener,
                   event_signal_counter, event_mutex_holder, cb_state_alive,
                   cb_state_sync_completed, cb_state_ref_count, pending_callbacks,
                   thread_op, thread_cb_state, next_cb_state_id>>

-----------------------------------------------------------------------------
(* DESTRUCTOR (32.107 fix - waits for callbacks) *)

StartDestroy(t) ==
    /\ pc[t] = "idle"
    /\ pool_alive
    /\ thread_event[t] # NULL
    /\ thread_event[t] \in in_use_events
    /\ thread_op' = [thread_op EXCEPT ![t] = "destroy"]
    /\ pc' = [pc EXCEPT ![t] = "destroy_set_not_alive"]
    /\ UNCHANGED <<pool_alive, pool_events, in_use_events, event_alive, event_callback_state,
                   event_listener, event_signal_counter, event_mutex_holder, cb_state_alive,
                   cb_state_sync_completed, cb_state_ref_count, pending_callbacks, thread_event,
                   thread_cb_state, next_cb_state_id, completed_ops>>

DestroySetNotAlive(t) ==
    /\ pc[t] = "destroy_set_not_alive"
    /\ thread_event[t] # NULL
    /\ LET e == thread_event[t]
           cs == event_callback_state[e]
       IN /\ cs # NULL
          \* 32.107 fix: Set alive=false FIRST before any wait
          /\ cb_state_alive' = [cb_state_alive EXCEPT ![cs] = FALSE]
          /\ thread_cb_state' = [thread_cb_state EXCEPT ![t] = cs]
    /\ pc' = [pc EXCEPT ![t] = "destroy_wait"]
    /\ UNCHANGED <<pool_alive, pool_events, in_use_events, event_alive, event_callback_state,
                   event_listener, event_signal_counter, event_mutex_holder,
                   cb_state_sync_completed, cb_state_ref_count, pending_callbacks,
                   thread_event, thread_op, next_cb_state_id, completed_ops>>

DestroyWait(t) ==
    /\ pc[t] = "destroy_wait"
    /\ thread_cb_state[t] # NULL
    /\ LET cs == thread_cb_state[t]
           e == thread_event[t]
       IN \/ \* Callback completed - proceed with destruction
             /\ cb_state_sync_completed[cs]
             /\ event_alive' = [event_alive EXCEPT ![e] = FALSE]
             /\ cb_state_ref_count' = [cb_state_ref_count EXCEPT ![cs] = @ - 1]
             /\ pc' = [pc EXCEPT ![t] = "destroy_done"]
          \/ \* Timeout - destructor proceeds anyway (32.106/32.107 fix)
             /\ ~cb_state_sync_completed[cs]
             /\ event_alive' = [event_alive EXCEPT ![e] = FALSE]
             /\ cb_state_ref_count' = [cb_state_ref_count EXCEPT ![cs] = @ - 1]
             /\ pc' = [pc EXCEPT ![t] = "destroy_done"]
    /\ UNCHANGED <<pool_alive, pool_events, in_use_events, event_callback_state,
                   event_listener, event_signal_counter, event_mutex_holder, cb_state_alive,
                   cb_state_sync_completed, pending_callbacks, thread_event, thread_op,
                   thread_cb_state, next_cb_state_id, completed_ops>>

DestroyDone(t) ==
    /\ pc[t] = "destroy_done"
    /\ thread_event[t] # NULL
    /\ LET e == thread_event[t]
       IN /\ in_use_events' = in_use_events \ {e}
          /\ pool_events' = pool_events \cup {e}  \* Return to pool for simplicity
          \* IMPORTANT: Clear callback_state so next acquirer knows to allocate fresh
          /\ event_callback_state' = [event_callback_state EXCEPT ![e] = NULL]
    /\ pc' = [pc EXCEPT ![t] = "done"]
    /\ thread_event' = [thread_event EXCEPT ![t] = NULL]
    /\ completed_ops' = completed_ops + 1
    /\ UNCHANGED <<pool_alive, event_alive, event_listener,
                   event_signal_counter, event_mutex_holder, cb_state_alive,
                   cb_state_sync_completed, cb_state_ref_count, pending_callbacks,
                   thread_op, thread_cb_state, next_cb_state_id>>

-----------------------------------------------------------------------------
(* CALLBACK EXECUTION (GPU completion callback) *)

\* A pending callback starts execution
StartCallback(t) ==
    /\ pc[t] = "idle"
    /\ pending_callbacks # {}
    /\ LET cb == CHOOSE x \in pending_callbacks : TRUE
           cs == cb[1]
           e == cb[2]
       IN /\ pending_callbacks' = pending_callbacks \ {cb}
          /\ thread_cb_state' = [thread_cb_state EXCEPT ![t] = cs]
          /\ thread_event' = [thread_event EXCEPT ![t] = e]
          /\ thread_op' = [thread_op EXCEPT ![t] = "callback"]
    /\ pc' = [pc EXCEPT ![t] = "callback_check_alive"]
    /\ UNCHANGED <<pool_alive, pool_events, in_use_events, event_alive, event_callback_state,
                   event_listener, event_signal_counter, event_mutex_holder, cb_state_alive,
                   cb_state_sync_completed, cb_state_ref_count, next_cb_state_id, completed_ops>>

\* Callback checks alive flag before accessing state
CallbackCheckAlive(t) ==
    /\ pc[t] = "callback_check_alive"
    /\ thread_cb_state[t] # NULL
    /\ LET cs == thread_cb_state[t]
       IN \/ \* alive=true - proceed with normal callback
             /\ cb_state_alive[cs]
             /\ pc' = [pc EXCEPT ![t] = "callback_notify"]
             /\ UNCHANGED <<cb_state_sync_completed, cb_state_ref_count, event_alive>>
          \/ \* alive=false (32.107 fix) - still notify completion but skip other work
             /\ ~cb_state_alive[cs]
             /\ cb_state_sync_completed' = [cb_state_sync_completed EXCEPT ![cs] = TRUE]
             /\ cb_state_ref_count' = [cb_state_ref_count EXCEPT ![cs] = @ - 1]  \* Release ref
             /\ pc' = [pc EXCEPT ![t] = "done"]
             /\ UNCHANGED <<event_alive>>
    /\ UNCHANGED <<pool_alive, pool_events, in_use_events, event_callback_state,
                   event_listener, event_signal_counter, event_mutex_holder, cb_state_alive,
                   pending_callbacks, thread_event, thread_op, thread_cb_state,
                   next_cb_state_id, completed_ops>>

\* Callback notifies completion (sync primitives are in CallbackState, safe to access)
CallbackNotify(t) ==
    /\ pc[t] = "callback_notify"
    /\ thread_cb_state[t] # NULL
    /\ LET cs == thread_cb_state[t]
       IN /\ cb_state_sync_completed' = [cb_state_sync_completed EXCEPT ![cs] = TRUE]
          /\ cb_state_ref_count' = [cb_state_ref_count EXCEPT ![cs] = @ - 1]  \* Release ref
    /\ pc' = [pc EXCEPT ![t] = "done"]
    /\ completed_ops' = completed_ops + 1
    /\ UNCHANGED <<pool_alive, pool_events, in_use_events, event_alive, event_callback_state,
                   event_listener, event_signal_counter, event_mutex_holder, cb_state_alive,
                   pending_callbacks, thread_event, thread_op, thread_cb_state, next_cb_state_id>>

-----------------------------------------------------------------------------
(* THREAD RESET *)

Reset(t) ==
    /\ pc[t] = "done"
    /\ pc' = [pc EXCEPT ![t] = "idle"]
    /\ thread_cb_state' = [thread_cb_state EXCEPT ![t] = NULL]
    \* Callbacks don't own events - clear thread_event after callback completes
    \* Other ops (acquire, record, sync, reset) keep their event ownership
    /\ IF thread_op[t] = "callback"
       THEN thread_event' = [thread_event EXCEPT ![t] = NULL]
       ELSE UNCHANGED <<thread_event>>
    /\ thread_op' = [thread_op EXCEPT ![t] = "none"]
    /\ UNCHANGED <<pool_alive, pool_events, in_use_events, event_alive, event_callback_state,
                   event_listener, event_signal_counter, event_mutex_holder, cb_state_alive,
                   cb_state_sync_completed, cb_state_ref_count, pending_callbacks,
                   next_cb_state_id, completed_ops>>

-----------------------------------------------------------------------------
(* POOL SHUTDOWN *)

PoolShutdown ==
    /\ pool_alive
    /\ in_use_events = {}  \* All events returned
    /\ pending_callbacks = {}  \* No pending callbacks
    /\ pool_alive' = FALSE
    /\ UNCHANGED <<pool_events, in_use_events, event_alive, event_callback_state,
                   event_listener, event_signal_counter, event_mutex_holder, cb_state_alive,
                   cb_state_sync_completed, cb_state_ref_count, pending_callbacks, pc,
                   thread_event, thread_op, thread_cb_state, next_cb_state_id, completed_ops>>

-----------------------------------------------------------------------------
(* NEXT STATE RELATION *)

Next ==
    \/ PoolShutdown
    \/ \E t \in Threads:
        \/ StartAcquireEvent(t)
        \/ AcquireEvent(t)
        \/ StartRecord(t)
        \/ RecordLock(t)
        \/ RecordNotify(t)
        \/ RecordDone(t)
        \/ StartSync(t)
        \/ SyncLock(t)
        \/ SyncNotify(t)
        \/ SyncWait(t)
        \/ StartReset(t)
        \/ ResetLock(t)
        \/ ResetInvalidate(t)
        \/ StartReleaseEvent(t)
        \/ ReleaseEvent(t)
        \/ StartDestroy(t)
        \/ DestroySetNotAlive(t)
        \/ DestroyWait(t)
        \/ DestroyDone(t)
        \/ StartCallback(t)
        \/ CallbackCheckAlive(t)
        \/ CallbackNotify(t)
        \/ Reset(t)

(* Fairness *)
Fairness == \A t \in Threads:
    WF_vars(Reset(t) \/ StartCallback(t) \/ CallbackCheckAlive(t) \/ CallbackNotify(t))

Spec == Init /\ [][Next]_vars /\ Fairness

-----------------------------------------------------------------------------
(* SAFETY PROPERTIES *)

(* Callback State Lifetime (CRITICAL): If a callback is accessing a callback state,
   the ref count must be > 0 (shared_ptr keeps CallbackState alive).
   This is the REAL safety property that prevents UAF. *)
CallbackStateRefCountPositive ==
    \A t \in Threads:
        (pc[t] \in {"callback_check_alive", "callback_notify"} /\ thread_cb_state[t] # NULL) =>
        cb_state_ref_count[thread_cb_state[t]] > 0

(* No Double Free: Events in pool are not also in use *)
PoolInUseDisjoint ==
    pool_events \cap in_use_events = {}

(* Mutex Exclusivity: Only one thread holds an event's mutex *)
MutexExclusivity ==
    \A e \in Events:
        (event_mutex_holder[e] # NULL) =>
        (Cardinality({t \in Threads : event_mutex_holder[e] = t}) = 1)

(* Pool Reuse Safety (32.89): After reset, new callbacks use new callback state.
   Old callbacks (if pending) will see alive=false for the old state. *)
ResetCreatesNewState ==
    \A e \in Events:
        \A cb \in pending_callbacks:
            \* If there's a pending callback for an event, and the event has a different
            \* callback state now, the pending callback's state must have alive=false
            (cb[2] = e /\ cb[1] # event_callback_state[e]) =>
            ~cb_state_alive[cb[1]]

(* State constraint to bound state space *)
StateConstraint ==
    /\ completed_ops <= 3
    /\ next_cb_state_id <= NumCallbackStates + 1
    /\ \A e \in Events : event_signal_counter[e] <= 1

(* Combined Safety Invariant
   NOTE: CallbackNeverAccessesDeadState was REMOVED because the 32.107 fix
   specifically allows callbacks to run when alive=false. The callback only
   accesses sync primitives in CallbackState (protected by shared_ptr refcount).
   The real safety property is CallbackStateRefCountPositive. *)
Safety ==
    /\ TypeOK
    /\ CallbackStateRefCountPositive
    /\ PoolInUseDisjoint
    /\ MutexExclusivity

-----------------------------------------------------------------------------
(* LIVENESS PROPERTIES *)

(* Eventually callbacks complete (unless pool shuts down) *)
EventuallyCallbacksComplete ==
    (pool_alive /\ pending_callbacks # {}) ~> (pending_callbacks = {})

(* No deadlock: some action is always enabled *)
NoDeadlock ==
    [](\E t \in Threads:
        ENABLED(Reset(t) \/ StartAcquireEvent(t) \/ StartRecord(t) \/
                StartSync(t) \/ StartReset(t) \/ StartCallback(t)))

=============================================================================
\* Modification History
\* Last modified: 2025-12-16
\* Created for MPS Parallel Inference Verification Platform
\* Models the callback lifetime safety pattern (32.107, 32.89 fixes)
