---------------------------- MODULE TensorLifetime ----------------------------
\* TensorLifetime - Use-After-Free Race Condition in MPS Kernel Encoding
\*
\* This TLA+ specification models the CRASH_FIX_ANALYSIS_2025-12-22.md bug:
\* A use-after-free race condition where Python GC frees a tensor's MTLBuffer
\* while a C++ worker thread is inside dispatch_sync_with_rethrow.
\*
\* CRASH MECHANISM:
\* 1. Thread A calls layer_norm_mps(input)
\* 2. Thread A gets X = input.expect_contiguous() (MaybeOwned<Tensor>)
\* 3. Thread A enters dispatch_sync_with_rethrow()
\* 4. Thread B (Python GC) sees tensor refcount -> 0, frees MTLBuffer
\* 5. Thread A calls getMTLBufferStorage(*X) - returns DANGLING pointer
\* 6. [encoder setBuffer:dangling_ptr ...] -> CRASH in AGX::bindResource
\*
\* FIX (Option 1 from analysis):
\* - Create owned copy: Tensor X_owned = X->contiguous()
\* - Use __block storage: __block Tensor X_block = X_owned
\* - Block captures X_block by value, preventing GC deallocation
\*
\* TOGGLE:
\* - CaptureByValue = FALSE : models vulnerable code (MaybeOwned borrow)
\* - CaptureByValue = TRUE  : models fixed code (__block owned capture)

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    \* @type: Int;
    NumThreads,       \* Number of worker threads (e.g., 8)
    \* @type: Int;
    NumTensors,       \* Number of tensors in the system
    \* @type: Bool;
    CaptureByValue    \* TRUE = fixed code, FALSE = vulnerable code

ASSUME NumThreads \in Nat /\ NumThreads > 0
ASSUME NumTensors \in Nat /\ NumTensors > 0
ASSUME CaptureByValue \in BOOLEAN

VARIABLES
    \* Tensor memory state
    \* @type: Int -> Bool;
    tensor_allocated,      \* Is memory allocated for this tensor?
    \* @type: Int -> Int;
    tensor_refcount,       \* Reference count (Python + C++ refs)
    \* @type: Int -> Bool;
    tensor_mtlbuffer_valid,\* Is the MTLBuffer valid (not freed)?

    \* Per-thread state
    \* @type: Int -> Str;
    thread_state,          \* "idle" | "got_maybeowned" | "in_dispatch" | "encoding" | "done"
    \* @type: Int -> Int;
    thread_target_tensor,  \* Which tensor thread is operating on
    \* @type: Int -> Bool;
    thread_owns_tensor,    \* Does this thread have an owned copy? (via __block)

    \* Global encoding mutex (layer_norm_mps has s_layer_norm_mutex)
    \* @type: Int;
    encoding_mutex_owner,

    \* Crash counters
    \* @type: Int;
    use_after_free_crashes,  \* Crashes from accessing freed MTLBuffer
    \* @type: Int;
    crashes_prevented        \* Potential crashes prevented by owning tensor

vars == <<tensor_allocated, tensor_refcount, tensor_mtlbuffer_valid,
          thread_state, thread_target_tensor, thread_owns_tensor,
          encoding_mutex_owner, use_after_free_crashes, crashes_prevented>>

Threads == 1..NumThreads
Tensors == 1..NumTensors
NULL == 0

ThreadStates == {"idle", "got_maybeowned", "in_dispatch", "encoding", "done"}

\* --------------------------------------------------------------------------
\* Initial State
\* --------------------------------------------------------------------------

Init ==
    /\ tensor_allocated = [t \in Tensors |-> FALSE]
    /\ tensor_refcount = [t \in Tensors |-> 0]
    /\ tensor_mtlbuffer_valid = [t \in Tensors |-> FALSE]
    /\ thread_state = [t \in Threads |-> "idle"]
    /\ thread_target_tensor = [t \in Threads |-> NULL]
    /\ thread_owns_tensor = [t \in Threads |-> FALSE]
    /\ encoding_mutex_owner = NULL
    /\ use_after_free_crashes = 0
    /\ crashes_prevented = 0

\* --------------------------------------------------------------------------
\* Tensor Lifecycle (Python side)
\* --------------------------------------------------------------------------

\* Python creates a tensor
CreateTensor(tensor) ==
    /\ tensor_allocated[tensor] = FALSE
    /\ tensor_allocated' = [tensor_allocated EXCEPT ![tensor] = TRUE]
    /\ tensor_refcount' = [tensor_refcount EXCEPT ![tensor] = 1]  \* Python holds ref
    /\ tensor_mtlbuffer_valid' = [tensor_mtlbuffer_valid EXCEPT ![tensor] = TRUE]
    /\ UNCHANGED <<thread_state, thread_target_tensor, thread_owns_tensor,
                   encoding_mutex_owner, use_after_free_crashes, crashes_prevented>>

\* Python GC attempts to free a tensor.
\* This can race with C++ code accessing the tensor.
PythonGCFreeTensor(tensor) ==
    /\ tensor_allocated[tensor] = TRUE
    /\ tensor_refcount[tensor] <= 1  \* Only Python's ref remains
    /\ IF tensor_refcount[tensor] = 0
       THEN
           \* No refs - definitely free
           /\ tensor_allocated' = [tensor_allocated EXCEPT ![tensor] = FALSE]
           /\ tensor_mtlbuffer_valid' = [tensor_mtlbuffer_valid EXCEPT ![tensor] = FALSE]
           /\ UNCHANGED <<tensor_refcount, thread_state, thread_target_tensor,
                         thread_owns_tensor, encoding_mutex_owner,
                         use_after_free_crashes, crashes_prevented>>
       ELSE IF tensor_refcount[tensor] = 1
       THEN
           \* Python's ref going to 0 - free if no C++ owns it
           /\ tensor_refcount' = [tensor_refcount EXCEPT ![tensor] = 0]
           /\ tensor_allocated' = [tensor_allocated EXCEPT ![tensor] = FALSE]
           /\ tensor_mtlbuffer_valid' = [tensor_mtlbuffer_valid EXCEPT ![tensor] = FALSE]
           /\ UNCHANGED <<thread_state, thread_target_tensor, thread_owns_tensor,
                         encoding_mutex_owner, use_after_free_crashes, crashes_prevented>>
       ELSE
           UNCHANGED vars

\* --------------------------------------------------------------------------
\* Worker Thread: layer_norm_mps flow
\* --------------------------------------------------------------------------

\* Thread starts layer_norm_mps(input).
\* Gets X = input.expect_contiguous() which returns MaybeOwned<Tensor>.
\* MaybeOwned may BORROW without incrementing refcount!
StartLayerNorm(thread, tensor) ==
    /\ thread_state[thread] = "idle"
    /\ tensor_allocated[tensor] = TRUE
    /\ thread_state' = [thread_state EXCEPT ![thread] = "got_maybeowned"]
    /\ thread_target_tensor' = [thread_target_tensor EXCEPT ![thread] = tensor]
    /\ IF CaptureByValue
       THEN
           \* FIX: Create owned copy, increment refcount
           /\ tensor_refcount' = [tensor_refcount EXCEPT ![tensor] = @ + 1]
           /\ thread_owns_tensor' = [thread_owns_tensor EXCEPT ![thread] = TRUE]
       ELSE
           \* BUG: MaybeOwned borrows, doesn't increment refcount
           /\ UNCHANGED tensor_refcount
           /\ thread_owns_tensor' = [thread_owns_tensor EXCEPT ![thread] = FALSE]
    /\ UNCHANGED <<tensor_allocated, tensor_mtlbuffer_valid, encoding_mutex_owner,
                   use_after_free_crashes, crashes_prevented>>

\* Thread acquires encoding mutex and enters dispatch_sync_with_rethrow.
\* This is BEFORE getMTLBufferStorage is called.
AcquireMutexEnterDispatch(thread) ==
    /\ thread_state[thread] = "got_maybeowned"
    /\ encoding_mutex_owner = NULL
    /\ encoding_mutex_owner' = thread
    /\ thread_state' = [thread_state EXCEPT ![thread] = "in_dispatch"]
    /\ UNCHANGED <<tensor_allocated, tensor_refcount, tensor_mtlbuffer_valid,
                   thread_target_tensor, thread_owns_tensor,
                   use_after_free_crashes, crashes_prevented>>

\* Thread calls getMTLBufferStorage(*X) and [encoder setBuffer:...].
\* THIS IS WHERE THE CRASH HAPPENS if MTLBuffer was freed!
EncodeWithBuffer(thread) ==
    /\ thread_state[thread] = "in_dispatch"
    /\ encoding_mutex_owner = thread
    /\ LET tensor == thread_target_tensor[thread] IN
        IF tensor_mtlbuffer_valid[tensor] = FALSE
        THEN
            \* CRASH! MTLBuffer was freed by GC
            /\ use_after_free_crashes' = use_after_free_crashes + 1
            \* Thread "crashes" - resets to idle
            /\ thread_state' = [thread_state EXCEPT ![thread] = "idle"]
            /\ thread_target_tensor' = [thread_target_tensor EXCEPT ![thread] = NULL]
            /\ thread_owns_tensor' = [thread_owns_tensor EXCEPT ![thread] = FALSE]
            /\ encoding_mutex_owner' = NULL
            /\ UNCHANGED <<tensor_allocated, tensor_refcount, tensor_mtlbuffer_valid,
                          crashes_prevented>>
        ELSE
            \* MTLBuffer valid - encoding succeeds
            /\ thread_state' = [thread_state EXCEPT ![thread] = "encoding"]
            /\ UNCHANGED <<tensor_allocated, tensor_refcount, tensor_mtlbuffer_valid,
                          thread_target_tensor, thread_owns_tensor, encoding_mutex_owner,
                          use_after_free_crashes, crashes_prevented>>

\* Thread finishes encoding, releases mutex.
FinishEncoding(thread) ==
    /\ thread_state[thread] = "encoding"
    /\ encoding_mutex_owner = thread
    /\ thread_state' = [thread_state EXCEPT ![thread] = "done"]
    /\ encoding_mutex_owner' = NULL
    /\ UNCHANGED <<tensor_allocated, tensor_refcount, tensor_mtlbuffer_valid,
                   thread_target_tensor, thread_owns_tensor,
                   use_after_free_crashes, crashes_prevented>>

\* Thread completes, releases owned tensor ref if applicable.
CompleteOperation(thread) ==
    /\ thread_state[thread] = "done"
    /\ LET tensor == thread_target_tensor[thread] IN
        /\ IF thread_owns_tensor[thread] /\ tensor /= NULL
           THEN
               \* FIX: Release owned ref
               tensor_refcount' = [tensor_refcount EXCEPT ![tensor] = @ - 1]
           ELSE
               UNCHANGED tensor_refcount
    /\ thread_state' = [thread_state EXCEPT ![thread] = "idle"]
    /\ thread_target_tensor' = [thread_target_tensor EXCEPT ![thread] = NULL]
    /\ thread_owns_tensor' = [thread_owns_tensor EXCEPT ![thread] = FALSE]
    /\ UNCHANGED <<tensor_allocated, tensor_mtlbuffer_valid, encoding_mutex_owner,
                   use_after_free_crashes, crashes_prevented>>

\* --------------------------------------------------------------------------
\* Next State
\* --------------------------------------------------------------------------

Next ==
    \/ \E tensor \in Tensors: CreateTensor(tensor)
    \/ \E tensor \in Tensors: PythonGCFreeTensor(tensor)
    \/ \E thread \in Threads, tensor \in Tensors: StartLayerNorm(thread, tensor)
    \/ \E thread \in Threads: AcquireMutexEnterDispatch(thread)
    \/ \E thread \in Threads: EncodeWithBuffer(thread)
    \/ \E thread \in Threads: FinishEncoding(thread)
    \/ \E thread \in Threads: CompleteOperation(thread)

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

\* --------------------------------------------------------------------------
\* Type Invariant
\* --------------------------------------------------------------------------

TypeOK ==
    /\ tensor_allocated \in [Tensors -> BOOLEAN]
    /\ tensor_refcount \in [Tensors -> Nat]
    /\ tensor_mtlbuffer_valid \in [Tensors -> BOOLEAN]
    /\ thread_state \in [Threads -> ThreadStates]
    /\ thread_target_tensor \in [Threads -> Tensors \cup {NULL}]
    /\ thread_owns_tensor \in [Threads -> BOOLEAN]
    /\ encoding_mutex_owner \in Threads \cup {NULL}
    /\ use_after_free_crashes \in Nat
    /\ crashes_prevented \in Nat

\* --------------------------------------------------------------------------
\* Safety Properties
\* --------------------------------------------------------------------------

\* CRITICAL INVARIANT: No use-after-free crashes
\* This SHOULD FAIL when CaptureByValue = FALSE (vulnerable code)
\* This SHOULD PASS when CaptureByValue = TRUE (fixed code)
NoUseAfterFreeCrashes == use_after_free_crashes = 0

\* If a thread owns a tensor (CaptureByValue=TRUE), the MTLBuffer must be valid.
\* The ownership prevents GC from freeing it.
OwnedTensorIsValid ==
    \A thread \in Threads:
        thread_owns_tensor[thread] /\ thread_target_tensor[thread] /= NULL =>
            tensor_mtlbuffer_valid[thread_target_tensor[thread]] = TRUE

\* A thread in encoding state must have a valid MTLBuffer.
\* This is the core safety property we want to guarantee.
EncodingImpliesValidBuffer ==
    \A thread \in Threads:
        thread_state[thread] \in {"in_dispatch", "encoding"} =>
            LET tensor == thread_target_tensor[thread] IN
                tensor /= NULL /\ tensor_mtlbuffer_valid[tensor] = TRUE

\* Refcount consistency: allocated tensors have refcount >= 0,
\* deallocated tensors have refcount = 0.
RefcountConsistent ==
    \A tensor \in Tensors:
        (~tensor_allocated[tensor] => tensor_refcount[tensor] = 0)

\* --------------------------------------------------------------------------
\* What This Model Reveals
\* --------------------------------------------------------------------------
\* Run TLC with:
\*   NumThreads = 2, NumTensors = 2
\*   CaptureByValue = FALSE  (vulnerable)
\*   Invariant: NoUseAfterFreeCrashes
\*
\* EXPECTED: Invariant VIOLATED - counterexample shows race:
\* 1. Thread 1 calls StartLayerNorm (gets MaybeOwned, no refcount inc)
\* 2. Thread 1 calls AcquireMutexEnterDispatch (in_dispatch)
\* 3. Python GC runs PythonGCFreeTensor (MTLBuffer freed!)
\* 4. Thread 1 calls EncodeWithBuffer (CRASH!)
\*
\* Run TLC with:
\*   CaptureByValue = TRUE  (fixed)
\*   Invariant: NoUseAfterFreeCrashes
\*
\* EXPECTED: Invariant HOLDS - owned tensor prevents GC deallocation.
\*
\* This formally proves the fix in patches/040-layer-norm-tensor-lifetime-fix.patch
\* prevents the use-after-free race condition.

=============================================================================
