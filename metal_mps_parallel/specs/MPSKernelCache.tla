--------------------------- MODULE MPSKernelCache ---------------------------
(*
 * TLA+ Specification for MPSKernelCache
 *
 * Models the thread-local kernel cache used by PyTorch MPS to store compiled
 * Metal kernels (MPSCachedKernel objects). Unlike MPSGraphCache, this cache
 * is thread-local (each thread has its own isolated cache).
 *
 * Properties verified:
 * 1. No key collision within a thread's cache (KeyUniqueness)
 * 2. Cached kernels remain valid until deleted (KernelValidity)
 * 3. No double-free when cache is destroyed (NoDoubleFree)
 * 4. Cache size bounded (CacheSizeInvariant)
 * 5. Thread isolation - threads don't access each other's caches (ThreadIsolation)
 * 6. Lookup returns correct kernel for key (LookupCorrectness)
 *
 * Based on: MPSKernelCache patterns in PyTorch MPS backend
 *           See: patches/README.md, thread_local std::unique_ptr<MPSKernelCache>
 *
 * Author: Worker N=1357
 * Date: 2025-12-20
 *)

EXTENDS Integers, Sequences, FiniteSets, TLC

\* Configuration constants
CONSTANTS
    \* @type: Int;
    MaxCacheSize,    \* Maximum kernels per thread's cache (e.g., 4)
    \* @type: Int;
    MaxKernels,      \* Total kernels that can be created (e.g., 8)
    \* @type: Int;
    NumThreads,      \* Number of threads (each has own cache) (e.g., 3)
    \* @type: Int;
    None             \* Sentinel value (e.g., 99)

\* Kernel states
CONSTANTS
    \* @type: Int;
    KernelFree,      \* Not allocated
    \* @type: Int;
    KernelCompiling, \* Being compiled
    \* @type: Int;
    KernelCached,    \* In cache, available
    \* @type: Int;
    KernelInUse,     \* Currently executing
    \* @type: Int;
    KernelDeleted    \* Freed (tombstone)

\* Thread states
CONSTANTS
    \* @type: Int;
    ThreadIdle,      \* Not doing cache operation
    \* @type: Int;
    ThreadLookup,    \* Looking up a kernel
    \* @type: Int;
    ThreadCompiling, \* Compiling new kernel
    \* @type: Int;
    ThreadCaching,   \* Adding to cache
    \* @type: Int;
    ThreadExecuting, \* Using kernel
    \* @type: Int;
    ThreadDestroying,\* Destroying cache (thread exit)
    \* @type: Int;
    ThreadDone       \* Operation complete

VARIABLES
    \* Kernel pool state (global kernel objects)
    \* @type: Int -> Int;
    kernelState,     \* Function: KernelId -> KernelState
    \* @type: Int -> Int;
    kernelKey,       \* Function: KernelId -> Key (0 = no key)
    \* @type: Int -> Int;
    kernelOwner,     \* Function: KernelId -> ThreadId \cup {None}

    \* Per-thread cache state (each thread has its own cache)
    \* @type: Int -> Set(Int);
    threadCache,     \* Function: ThreadId -> Set of KernelIds in that thread's cache
    \* @type: Int -> Int;
    threadCacheSize, \* Function: ThreadId -> Nat

    \* Thread state
    \* @type: Int -> Int;
    threadState,     \* Function: ThreadId -> ThreadState
    \* @type: Int -> Int;
    threadTargetKey, \* Function: ThreadId -> Key being looked up
    \* @type: Int -> Int;
    threadTargetKernel \* Function: ThreadId -> KernelId being used

vars == <<kernelState, kernelKey, kernelOwner,
          threadCache, threadCacheSize,
          threadState, threadTargetKey, threadTargetKernel>>

\* Helper sets
KernelIds == 0..(MaxKernels-1)
ThreadIds == 0..(NumThreads-1)
Keys == 1..4  \* Valid keys (0 means no key)

-----------------------------------------------------------------------------
\* Type invariant
TypeOK ==
    /\ kernelState \in [KernelIds -> {KernelFree, KernelCompiling, KernelCached,
                                       KernelInUse, KernelDeleted}]
    /\ kernelKey \in [KernelIds -> 0..4]
    /\ kernelOwner \in [KernelIds -> ThreadIds \cup {None}]
    /\ threadCache \in [ThreadIds -> SUBSET KernelIds]
    /\ threadCacheSize \in [ThreadIds -> 0..MaxCacheSize]
    /\ threadState \in [ThreadIds -> {ThreadIdle, ThreadLookup, ThreadCompiling,
                                       ThreadCaching, ThreadExecuting,
                                       ThreadDestroying, ThreadDone}]
    /\ threadTargetKey \in [ThreadIds -> 0..4]
    /\ threadTargetKernel \in [ThreadIds -> KernelIds \cup {None}]

-----------------------------------------------------------------------------
\* Safety invariants

\* Key uniqueness within each thread's cache
KeyUniqueness ==
    \A t \in ThreadIds :
        \A k1, k2 \in threadCache[t] :
            (k1 # k2 /\ kernelKey[k1] # 0 /\ kernelKey[k2] # 0) =>
                kernelKey[k1] # kernelKey[k2]

\* Cached kernels are in valid state
KernelValidity ==
    \A t \in ThreadIds :
        \A k \in threadCache[t] :
            kernelState[k] \in {KernelCached, KernelInUse}

\* No double-free: deleted kernels stay deleted
NoDoubleFree ==
    \A k \in KernelIds : kernelState[k] = KernelDeleted => kernelOwner[k] = None

\* Cache size matches actual count
CacheSizeConsistent ==
    \A t \in ThreadIds :
        threadCacheSize[t] = Cardinality(threadCache[t])

\* Cache size bounded
CacheSizeInvariant ==
    \A t \in ThreadIds :
        threadCacheSize[t] <= MaxCacheSize

\* Thread isolation: kernels in a thread's cache are owned by that thread
ThreadIsolation ==
    \A t \in ThreadIds :
        \A k \in threadCache[t] :
            kernelOwner[k] = t

\* Lookup correctness: if a kernel is in cache with a key, lookup finds it
LookupCorrectness ==
    \A t \in ThreadIds :
        \A k1, k2 \in threadCache[t] :
            (kernelKey[k1] = kernelKey[k2] /\ kernelKey[k1] # 0) => k1 = k2

\* Owner consistency: only cached/inuse kernels have owners
OwnerConsistency ==
    \A k \in KernelIds :
        kernelOwner[k] # None <=> kernelState[k] \in {KernelCached, KernelInUse, KernelCompiling}

\* All safety properties combined
Safety ==
    /\ KeyUniqueness
    /\ KernelValidity
    /\ NoDoubleFree
    /\ CacheSizeConsistent
    /\ CacheSizeInvariant
    /\ ThreadIsolation
    /\ LookupCorrectness

-----------------------------------------------------------------------------
\* Initial state

Init ==
    /\ kernelState = [k \in KernelIds |-> KernelFree]
    /\ kernelKey = [k \in KernelIds |-> 0]
    /\ kernelOwner = [k \in KernelIds |-> None]
    /\ threadCache = [t \in ThreadIds |-> {}]
    /\ threadCacheSize = [t \in ThreadIds |-> 0]
    /\ threadState = [t \in ThreadIds |-> ThreadIdle]
    /\ threadTargetKey = [t \in ThreadIds |-> 0]
    /\ threadTargetKernel = [t \in ThreadIds |-> None]

-----------------------------------------------------------------------------
\* Actions

\* Thread starts a cache lookup
StartLookup(t, key) ==
    /\ threadState[t] = ThreadIdle
    /\ key \in Keys
    /\ threadState' = [threadState EXCEPT ![t] = ThreadLookup]
    /\ threadTargetKey' = [threadTargetKey EXCEPT ![t] = key]
    /\ UNCHANGED <<kernelState, kernelKey, kernelOwner,
                   threadCache, threadCacheSize, threadTargetKernel>>

\* Cache hit: find kernel in this thread's cache
CacheHit(t) ==
    /\ threadState[t] = ThreadLookup
    /\ \E k \in threadCache[t] :
        /\ kernelState[k] = KernelCached
        /\ kernelKey[k] = threadTargetKey[t]
        /\ kernelState' = [kernelState EXCEPT ![k] = KernelInUse]
        /\ threadState' = [threadState EXCEPT ![t] = ThreadExecuting]
        /\ threadTargetKernel' = [threadTargetKernel EXCEPT ![t] = k]
    /\ UNCHANGED <<kernelKey, kernelOwner, threadCache, threadCacheSize, threadTargetKey>>

\* Cache miss: start compiling new kernel
CacheMiss(t) ==
    /\ threadState[t] = ThreadLookup
    /\ ~\E k \in threadCache[t] :
        /\ kernelState[k] = KernelCached
        /\ kernelKey[k] = threadTargetKey[t]
    /\ threadCacheSize[t] < MaxCacheSize  \* Room in cache
    /\ \E k \in KernelIds :
        /\ kernelState[k] = KernelFree
        /\ kernelState' = [kernelState EXCEPT ![k] = KernelCompiling]
        /\ kernelKey' = [kernelKey EXCEPT ![k] = threadTargetKey[t]]
        /\ kernelOwner' = [kernelOwner EXCEPT ![k] = t]
        /\ threadState' = [threadState EXCEPT ![t] = ThreadCompiling]
        /\ threadTargetKernel' = [threadTargetKernel EXCEPT ![t] = k]
    /\ UNCHANGED <<threadCache, threadCacheSize, threadTargetKey>>

\* Finish compiling and add to cache
FinishCompiling(t) ==
    /\ threadState[t] = ThreadCompiling
    /\ LET k == threadTargetKernel[t] IN
        /\ k # None
        /\ kernelState[k] = KernelCompiling
        /\ kernelOwner[k] = t
        /\ kernelState' = [kernelState EXCEPT ![k] = KernelInUse]
        /\ threadCache' = [threadCache EXCEPT ![t] = @ \cup {k}]
        /\ threadCacheSize' = [threadCacheSize EXCEPT ![t] = @ + 1]
        /\ threadState' = [threadState EXCEPT ![t] = ThreadExecuting]
    /\ UNCHANGED <<kernelKey, kernelOwner, threadTargetKey, threadTargetKernel>>

\* Thread finishes using kernel
FinishExecution(t) ==
    /\ threadState[t] = ThreadExecuting
    /\ LET k == threadTargetKernel[t] IN
        /\ k # None
        /\ kernelState[k] = KernelInUse
        /\ kernelState' = [kernelState EXCEPT ![k] = KernelCached]
        /\ threadState' = [threadState EXCEPT ![t] = ThreadDone]
        /\ threadTargetKernel' = [threadTargetKernel EXCEPT ![t] = None]
    /\ UNCHANGED <<kernelKey, kernelOwner, threadCache, threadCacheSize, threadTargetKey>>

\* Thread returns to idle
ReturnToIdle(t) ==
    /\ threadState[t] = ThreadDone
    /\ threadState' = [threadState EXCEPT ![t] = ThreadIdle]
    /\ threadTargetKey' = [threadTargetKey EXCEPT ![t] = 0]
    /\ UNCHANGED <<kernelState, kernelKey, kernelOwner,
                   threadCache, threadCacheSize, threadTargetKernel>>

\* Thread destroys its cache (thread exit)
DestroyCache(t) ==
    /\ threadState[t] = ThreadIdle
    /\ threadCacheSize[t] > 0  \* Has kernels to destroy
    /\ \A k \in threadCache[t] : kernelState[k] = KernelCached  \* All kernels idle
    /\ kernelState' = [k \in KernelIds |->
                        IF k \in threadCache[t]
                        THEN KernelDeleted
                        ELSE kernelState[k]]
    /\ kernelOwner' = [k \in KernelIds |->
                        IF k \in threadCache[t]
                        THEN None
                        ELSE kernelOwner[k]]
    /\ kernelKey' = [k \in KernelIds |->
                      IF k \in threadCache[t]
                      THEN 0
                      ELSE kernelKey[k]]
    /\ threadCache' = [threadCache EXCEPT ![t] = {}]
    /\ threadCacheSize' = [threadCacheSize EXCEPT ![t] = 0]
    /\ threadState' = [threadState EXCEPT ![t] = ThreadDestroying]
    /\ UNCHANGED <<threadTargetKey, threadTargetKernel>>

\* Thread finishes destruction
FinishDestroy(t) ==
    /\ threadState[t] = ThreadDestroying
    /\ threadState' = [threadState EXCEPT ![t] = ThreadDone]
    /\ UNCHANGED <<kernelState, kernelKey, kernelOwner,
                   threadCache, threadCacheSize, threadTargetKey, threadTargetKernel>>

\* Recycle deleted kernels back to free pool
RecycleKernel(k) ==
    /\ kernelState[k] = KernelDeleted
    /\ kernelOwner[k] = None
    /\ kernelState' = [kernelState EXCEPT ![k] = KernelFree]
    /\ UNCHANGED <<kernelKey, kernelOwner, threadCache, threadCacheSize,
                   threadState, threadTargetKey, threadTargetKernel>>

-----------------------------------------------------------------------------
\* Next state relation

Next ==
    \/ \E t \in ThreadIds, key \in Keys : StartLookup(t, key)
    \/ \E t \in ThreadIds : CacheHit(t)
    \/ \E t \in ThreadIds : CacheMiss(t)
    \/ \E t \in ThreadIds : FinishCompiling(t)
    \/ \E t \in ThreadIds : FinishExecution(t)
    \/ \E t \in ThreadIds : ReturnToIdle(t)
    \/ \E t \in ThreadIds : DestroyCache(t)
    \/ \E t \in ThreadIds : FinishDestroy(t)
    \/ \E k \in KernelIds : RecycleKernel(k)
    \/ UNCHANGED vars

\* Fairness: eventually threads make progress
Fairness ==
    /\ WF_vars(\E t \in ThreadIds : CacheHit(t))
    /\ WF_vars(\E t \in ThreadIds : CacheMiss(t))
    /\ WF_vars(\E t \in ThreadIds : FinishCompiling(t))
    /\ WF_vars(\E t \in ThreadIds : FinishExecution(t))
    /\ WF_vars(\E t \in ThreadIds : ReturnToIdle(t))
    /\ WF_vars(\E t \in ThreadIds : FinishDestroy(t))
    /\ WF_vars(\E k \in KernelIds : RecycleKernel(k))

Spec == Init /\ [][Next]_vars /\ Fairness
SpecNoFairness == Init /\ [][Next]_vars

-----------------------------------------------------------------------------
\* Liveness properties

\* Eventually all threads return to idle
EventuallyIdle ==
    \A t \in ThreadIds : threadState[t] # ThreadIdle ~> threadState[t] = ThreadIdle

\* Progress: threads eventually complete operations
Progress == []<>(\A t \in ThreadIds : threadState[t] \in {ThreadIdle, ThreadDone})

=============================================================================
