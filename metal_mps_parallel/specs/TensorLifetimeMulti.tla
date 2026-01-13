---------------------------- MODULE TensorLifetimeMulti ----------------------------
\* TensorLifetimeMulti - Models MULTIPLE tensor lifetime in MPS kernel encoding
\*
\* This spec extends TensorLifetime.tla to model the ACTUAL layer_norm_mps code
\* which has multiple tensors: X, gamma, bias, out, mean, rstd
\*
\* GAPS FOUND in current fix (patches/040-layer-norm-tensor-lifetime-fix.patch):
\* 1. X_owned is created - GOOD
\* 2. gamma_owned is created - GOOD
\* 3. bias is NOT converted to owned - BUG! Uses `__block Tensor bias_block = bias;`
\*
\* This spec proves that partial fix (X + gamma owned, but bias borrowed) is UNSAFE.
\*
\* TOGGLE:
\* - BiasOwned = FALSE : models current buggy fix (bias not owned)
\* - BiasOwned = TRUE  : models correct fix (all tensors owned)

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    \* @type: Int;
    NumThreads,
    \* @type: Bool;
    XOwned,       \* Is X captured by value? (TRUE in current fix)
    \* @type: Bool;
    GammaOwned,   \* Is gamma captured by value? (TRUE in current fix)
    \* @type: Bool;
    BiasOwned     \* Is bias captured by value? (FALSE in current fix - BUG!)

ASSUME NumThreads \in Nat /\ NumThreads > 0
ASSUME XOwned \in BOOLEAN
ASSUME GammaOwned \in BOOLEAN
ASSUME BiasOwned \in BOOLEAN

VARIABLES
    \* Tensor states: X, gamma, bias
    \* @type: Bool;
    X_allocated,
    \* @type: Int;
    X_refcount,
    \* @type: Bool;
    X_valid,

    \* @type: Bool;
    gamma_allocated,
    \* @type: Int;
    gamma_refcount,
    \* @type: Bool;
    gamma_valid,

    \* @type: Bool;
    bias_allocated,
    \* @type: Int;
    bias_refcount,
    \* @type: Bool;
    bias_valid,

    \* Per-thread state
    \* @type: Int -> Str;
    thread_state,
    \* @type: Int -> Bool;
    thread_owns_X,
    \* @type: Int -> Bool;
    thread_owns_gamma,
    \* @type: Int -> Bool;
    thread_owns_bias,

    \* Mutex
    \* @type: Int;
    mutex_owner,

    \* Crash counter
    \* @type: Int;
    crashes

vars == <<X_allocated, X_refcount, X_valid,
          gamma_allocated, gamma_refcount, gamma_valid,
          bias_allocated, bias_refcount, bias_valid,
          thread_state, thread_owns_X, thread_owns_gamma, thread_owns_bias,
          mutex_owner, crashes>>

Threads == 1..NumThreads
NULL == 0

ThreadStates == {"idle", "preparing", "in_dispatch", "encoding", "done"}

\* --------------------------------------------------------------------------
\* Initial State
\* --------------------------------------------------------------------------

Init ==
    /\ X_allocated = FALSE
    /\ X_refcount = 0
    /\ X_valid = FALSE
    /\ gamma_allocated = FALSE
    /\ gamma_refcount = 0
    /\ gamma_valid = FALSE
    /\ bias_allocated = FALSE
    /\ bias_refcount = 0
    /\ bias_valid = FALSE
    /\ thread_state = [t \in Threads |-> "idle"]
    /\ thread_owns_X = [t \in Threads |-> FALSE]
    /\ thread_owns_gamma = [t \in Threads |-> FALSE]
    /\ thread_owns_bias = [t \in Threads |-> FALSE]
    /\ mutex_owner = NULL
    /\ crashes = 0

\* --------------------------------------------------------------------------
\* Tensor Creation (Python creates input tensors before calling layer_norm)
\* --------------------------------------------------------------------------

CreateTensors ==
    /\ X_allocated = FALSE
    /\ X_allocated' = TRUE
    /\ X_refcount' = 1
    /\ X_valid' = TRUE
    /\ gamma_allocated' = TRUE
    /\ gamma_refcount' = 1
    /\ gamma_valid' = TRUE
    /\ bias_allocated' = TRUE
    /\ bias_refcount' = 1
    /\ bias_valid' = TRUE
    /\ UNCHANGED <<thread_state, thread_owns_X, thread_owns_gamma, thread_owns_bias,
                   mutex_owner, crashes>>

\* --------------------------------------------------------------------------
\* Python GC can free any tensor with refcount <= 1
\* --------------------------------------------------------------------------

GCFreeX ==
    /\ X_allocated = TRUE
    /\ X_refcount <= 1
    /\ X_refcount' = 0
    /\ X_allocated' = FALSE
    /\ X_valid' = FALSE
    /\ UNCHANGED <<gamma_allocated, gamma_refcount, gamma_valid,
                   bias_allocated, bias_refcount, bias_valid,
                   thread_state, thread_owns_X, thread_owns_gamma, thread_owns_bias,
                   mutex_owner, crashes>>

GCFreeGamma ==
    /\ gamma_allocated = TRUE
    /\ gamma_refcount <= 1
    /\ gamma_refcount' = 0
    /\ gamma_allocated' = FALSE
    /\ gamma_valid' = FALSE
    /\ UNCHANGED <<X_allocated, X_refcount, X_valid,
                   bias_allocated, bias_refcount, bias_valid,
                   thread_state, thread_owns_X, thread_owns_gamma, thread_owns_bias,
                   mutex_owner, crashes>>

GCFreeBias ==
    /\ bias_allocated = TRUE
    /\ bias_refcount <= 1
    /\ bias_refcount' = 0
    /\ bias_allocated' = FALSE
    /\ bias_valid' = FALSE
    /\ UNCHANGED <<X_allocated, X_refcount, X_valid,
                   gamma_allocated, gamma_refcount, gamma_valid,
                   thread_state, thread_owns_X, thread_owns_gamma, thread_owns_bias,
                   mutex_owner, crashes>>

\* --------------------------------------------------------------------------
\* Thread starts layer_norm_mps
\* --------------------------------------------------------------------------

StartLayerNorm(thread) ==
    /\ thread_state[thread] = "idle"
    /\ X_allocated = TRUE
    /\ gamma_allocated = TRUE
    /\ bias_allocated = TRUE
    /\ thread_state' = [thread_state EXCEPT ![thread] = "preparing"]
    \* Increment refcount if owned
    /\ IF XOwned
       THEN /\ X_refcount' = X_refcount + 1
            /\ thread_owns_X' = [thread_owns_X EXCEPT ![thread] = TRUE]
       ELSE /\ UNCHANGED X_refcount
            /\ thread_owns_X' = [thread_owns_X EXCEPT ![thread] = FALSE]
    /\ IF GammaOwned
       THEN /\ gamma_refcount' = gamma_refcount + 1
            /\ thread_owns_gamma' = [thread_owns_gamma EXCEPT ![thread] = TRUE]
       ELSE /\ UNCHANGED gamma_refcount
            /\ thread_owns_gamma' = [thread_owns_gamma EXCEPT ![thread] = FALSE]
    /\ IF BiasOwned
       THEN /\ bias_refcount' = bias_refcount + 1
            /\ thread_owns_bias' = [thread_owns_bias EXCEPT ![thread] = TRUE]
       ELSE /\ UNCHANGED bias_refcount
            /\ thread_owns_bias' = [thread_owns_bias EXCEPT ![thread] = FALSE]
    /\ UNCHANGED <<X_allocated, X_valid, gamma_allocated, gamma_valid,
                   bias_allocated, bias_valid, mutex_owner, crashes>>

\* --------------------------------------------------------------------------
\* Thread acquires mutex and enters dispatch
\* --------------------------------------------------------------------------

AcquireMutex(thread) ==
    /\ thread_state[thread] = "preparing"
    /\ mutex_owner = NULL
    /\ mutex_owner' = thread
    /\ thread_state' = [thread_state EXCEPT ![thread] = "in_dispatch"]
    /\ UNCHANGED <<X_allocated, X_refcount, X_valid,
                   gamma_allocated, gamma_refcount, gamma_valid,
                   bias_allocated, bias_refcount, bias_valid,
                   thread_owns_X, thread_owns_gamma, thread_owns_bias, crashes>>

\* --------------------------------------------------------------------------
\* Thread encodes kernel - accesses ALL tensors
\* --------------------------------------------------------------------------

EncodeKernel(thread) ==
    /\ thread_state[thread] = "in_dispatch"
    /\ mutex_owner = thread
    \* Check if ANY tensor is invalid - CRASH
    /\ IF X_valid = FALSE \/ gamma_valid = FALSE \/ bias_valid = FALSE
       THEN
           /\ crashes' = crashes + 1
           /\ thread_state' = [thread_state EXCEPT ![thread] = "idle"]
           /\ mutex_owner' = NULL
           /\ thread_owns_X' = [thread_owns_X EXCEPT ![thread] = FALSE]
           /\ thread_owns_gamma' = [thread_owns_gamma EXCEPT ![thread] = FALSE]
           /\ thread_owns_bias' = [thread_owns_bias EXCEPT ![thread] = FALSE]
           /\ UNCHANGED <<X_allocated, X_refcount, X_valid,
                         gamma_allocated, gamma_refcount, gamma_valid,
                         bias_allocated, bias_refcount, bias_valid>>
       ELSE
           /\ thread_state' = [thread_state EXCEPT ![thread] = "encoding"]
           /\ UNCHANGED <<X_allocated, X_refcount, X_valid,
                         gamma_allocated, gamma_refcount, gamma_valid,
                         bias_allocated, bias_refcount, bias_valid,
                         thread_owns_X, thread_owns_gamma, thread_owns_bias,
                         mutex_owner, crashes>>

\* --------------------------------------------------------------------------
\* Thread finishes and releases resources
\* --------------------------------------------------------------------------

FinishEncoding(thread) ==
    /\ thread_state[thread] = "encoding"
    /\ mutex_owner = thread
    /\ thread_state' = [thread_state EXCEPT ![thread] = "done"]
    /\ mutex_owner' = NULL
    /\ UNCHANGED <<X_allocated, X_refcount, X_valid,
                   gamma_allocated, gamma_refcount, gamma_valid,
                   bias_allocated, bias_refcount, bias_valid,
                   thread_owns_X, thread_owns_gamma, thread_owns_bias, crashes>>

ReleaseOwnership(thread) ==
    /\ thread_state[thread] = "done"
    /\ IF thread_owns_X[thread]
       THEN X_refcount' = X_refcount - 1
       ELSE UNCHANGED X_refcount
    /\ IF thread_owns_gamma[thread]
       THEN gamma_refcount' = gamma_refcount - 1
       ELSE UNCHANGED gamma_refcount
    /\ IF thread_owns_bias[thread]
       THEN bias_refcount' = bias_refcount - 1
       ELSE UNCHANGED bias_refcount
    /\ thread_state' = [thread_state EXCEPT ![thread] = "idle"]
    /\ thread_owns_X' = [thread_owns_X EXCEPT ![thread] = FALSE]
    /\ thread_owns_gamma' = [thread_owns_gamma EXCEPT ![thread] = FALSE]
    /\ thread_owns_bias' = [thread_owns_bias EXCEPT ![thread] = FALSE]
    /\ UNCHANGED <<X_allocated, X_valid, gamma_allocated, gamma_valid,
                   bias_allocated, bias_valid, mutex_owner, crashes>>

\* --------------------------------------------------------------------------
\* Next State
\* --------------------------------------------------------------------------

Next ==
    \/ CreateTensors
    \/ GCFreeX
    \/ GCFreeGamma
    \/ GCFreeBias
    \/ \E t \in Threads: StartLayerNorm(t)
    \/ \E t \in Threads: AcquireMutex(t)
    \/ \E t \in Threads: EncodeKernel(t)
    \/ \E t \in Threads: FinishEncoding(t)
    \/ \E t \in Threads: ReleaseOwnership(t)

Spec == Init /\ [][Next]_vars

\* --------------------------------------------------------------------------
\* Invariants
\* --------------------------------------------------------------------------

TypeOK ==
    /\ X_allocated \in BOOLEAN
    /\ X_refcount \in Nat
    /\ X_valid \in BOOLEAN
    /\ gamma_allocated \in BOOLEAN
    /\ gamma_refcount \in Nat
    /\ gamma_valid \in BOOLEAN
    /\ bias_allocated \in BOOLEAN
    /\ bias_refcount \in Nat
    /\ bias_valid \in BOOLEAN
    /\ thread_state \in [Threads -> ThreadStates]
    /\ thread_owns_X \in [Threads -> BOOLEAN]
    /\ thread_owns_gamma \in [Threads -> BOOLEAN]
    /\ thread_owns_bias \in [Threads -> BOOLEAN]
    /\ mutex_owner \in Threads \cup {NULL}
    /\ crashes \in Nat

\* CRITICAL: No crashes
NoCrashes == crashes = 0

\* If thread owns a tensor, it must be valid
OwnedTensorsValid ==
    \A t \in Threads:
        /\ (thread_owns_X[t] => X_valid = TRUE)
        /\ (thread_owns_gamma[t] => gamma_valid = TRUE)
        /\ (thread_owns_bias[t] => bias_valid = TRUE)

\* --------------------------------------------------------------------------
\* Expected Results:
\*
\* Config 1: XOwned=TRUE, GammaOwned=TRUE, BiasOwned=FALSE (current buggy fix)
\*   EXPECTED: NoCrashes VIOLATED - bias can be freed during encoding
\*
\* Config 2: XOwned=TRUE, GammaOwned=TRUE, BiasOwned=TRUE (correct fix)
\*   EXPECTED: NoCrashes HOLDS - all tensors protected
\* --------------------------------------------------------------------------

=============================================================================
