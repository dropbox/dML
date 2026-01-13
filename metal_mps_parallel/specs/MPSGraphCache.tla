--------------------------- MODULE MPSGraphCache ---------------------------
(*
 * TLA+ Specification for MPSGraphCache
 *
 * Models the graph cache used by PyTorch MPS to store compiled MPSGraph objects.
 * Thread-safe access is critical for parallel inference.
 *
 * Properties verified:
 * 1. Cache size never exceeds maximum (CacheSizeInvariant)
 * 2. Cache size never goes negative (CacheSizeNonNegative)
 * 3. No double-free of graph objects (NoDoubleFree)
 * 4. Graphs are properly reference-counted (RefCountInvariant)
 * 5. Evicted graphs have zero references (EvictionSafety)
 * 6. Mutex exclusion for cache operations (MutexExclusion)
 *
 * Based on: MPSGraphCache patterns in PyTorch MPS backend
 *           and mps-verify/verification/cbmc/harnesses/graph_cache_harness.c
 *
 * Author: Worker N=1353
 * Date: 2025-12-20
 *)

EXTENDS Integers, Sequences, FiniteSets, TLC

\* Configuration constants
CONSTANTS
    \* @type: Int;
    MaxCacheSize,    \* Maximum number of graphs in cache (e.g., 3)
    \* @type: Int;
    MaxGraphs,       \* Total graphs that can be created (e.g., 6)
    \* @type: Int;
    NumThreads,      \* Number of concurrent threads (e.g., 2)
    \* @type: Int;
    None             \* Sentinel value for "no thread/graph" (e.g., 99)

\* Graph states
CONSTANTS
    \* @type: Int;
    GraphFree,       \* Not allocated
    \* @type: Int;
    GraphCreating,   \* Being compiled
    \* @type: Int;
    GraphCached,     \* In cache, available for use
    \* @type: Int;
    GraphInUse,      \* Currently being executed
    \* @type: Int;
    GraphEvicting,   \* Being removed from cache
    \* @type: Int;
    GraphDeleted     \* Freed (tombstone)

\* Thread states for cache operations
CONSTANTS
    \* @type: Int;
    ThreadIdle,           \* Not doing cache operation
    \* @type: Int;
    ThreadLookup,         \* Looking up a graph
    \* @type: Int;
    ThreadCompiling,      \* Compiling a new graph
    \* @type: Int;
    ThreadInserting,      \* Inserting into cache
    \* @type: Int;
    ThreadExecuting,      \* Using a cached graph
    \* @type: Int;
    ThreadEvicting,       \* Evicting a graph
    \* @type: Int;
    ThreadDone            \* Operation complete

VARIABLES
    \* Graph pool state
    \* @type: Int -> Int;
    graphState,      \* Function: GraphId -> GraphState
    \* @type: Int -> Int;
    graphRefCount,   \* Function: GraphId -> Nat (reference count)
    \* @type: Int -> Int;
    graphKey,        \* Function: GraphId -> Nat (hash key, 0 = no key)
    \* @type: Int -> Int;
    graphCacheSlot,  \* Function: GraphId -> CacheSlot \cup {None}

    \* Cache state
    \* @type: Int -> Int;
    cacheSlots,      \* Function: CacheSlot -> GraphId \cup {None}
    \* @type: Int;
    cacheSize,       \* Current number of cached graphs

    \* Mutex state
    \* @type: Int;
    mutexHolder,     \* ThreadId holding mutex, or None if free

    \* Thread state
    \* @type: Int -> Int;
    threadState,     \* Function: ThreadId -> ThreadState
    \* @type: Int -> Int;
    threadTargetKey, \* Function: ThreadId -> Key being looked up
    \* @type: Int -> Int;
    threadTargetGraph\* Function: ThreadId -> GraphId being used

vars == <<graphState, graphRefCount, graphKey, graphCacheSlot,
          cacheSlots, cacheSize, mutexHolder,
          threadState, threadTargetKey, threadTargetGraph>>

\* Helper sets
GraphIds == 0..(MaxGraphs-1)
CacheSlots == 0..(MaxCacheSize-1)
ThreadIds == 0..(NumThreads-1)
Keys == 1..3  \* Valid keys (0 means no key)

-----------------------------------------------------------------------------
\* Type invariant
TypeOK ==
    /\ graphState \in [GraphIds -> {GraphFree, GraphCreating, GraphCached,
                                     GraphInUse, GraphEvicting, GraphDeleted}]
    /\ graphRefCount \in [GraphIds -> 0..NumThreads]
    /\ graphKey \in [GraphIds -> 0..3]
    /\ graphCacheSlot \in [GraphIds -> CacheSlots \cup {None}]
    /\ cacheSlots \in [CacheSlots -> GraphIds \cup {None}]
    /\ cacheSize \in 0..MaxCacheSize
    /\ mutexHolder \in ThreadIds \cup {None}
    /\ threadState \in [ThreadIds -> {ThreadIdle, ThreadLookup, ThreadCompiling,
                                       ThreadInserting, ThreadExecuting,
                                       ThreadEvicting, ThreadDone}]
    /\ threadTargetKey \in [ThreadIds -> 0..3]
    /\ threadTargetGraph \in [ThreadIds -> GraphIds \cup {None}]

-----------------------------------------------------------------------------
\* Safety invariants

\* Cache size never exceeds maximum
CacheSizeInvariant == cacheSize <= MaxCacheSize

\* Cache size never goes negative
CacheSizeNonNegative == cacheSize >= 0

\* Cache size matches actual cached graphs (GraphCached + GraphInUse are both in cache)
CacheSizeConsistent ==
    cacheSize = Cardinality({g \in GraphIds : graphState[g] \in {GraphCached, GraphInUse}})

\* No double-free: deleted graphs stay deleted
NoDoubleFree ==
    \A g \in GraphIds : graphState[g] = GraphDeleted => graphRefCount[g] = 0

\* Reference count invariant: in-use graphs have positive refcount
RefCountInvariant ==
    \A g \in GraphIds : graphState[g] = GraphInUse => graphRefCount[g] > 0

\* Eviction safety: only evict graphs with zero references
EvictionSafety ==
    \A g \in GraphIds : graphState[g] = GraphEvicting => graphRefCount[g] = 0

\* Mutex exclusion
MutexExclusion ==
    Cardinality({t \in ThreadIds : threadState[t] \in
                 {ThreadInserting, ThreadEvicting}}) <= 1

\* Cache slot consistency (both GraphCached and GraphInUse have slots)
CacheSlotConsistent ==
    \A g \in GraphIds :
        graphState[g] \in {GraphCached, GraphInUse} =>
            /\ graphCacheSlot[g] \in CacheSlots
            /\ cacheSlots[graphCacheSlot[g]] = g

\* All safety properties combined
Safety ==
    /\ CacheSizeInvariant
    /\ CacheSizeNonNegative
    /\ CacheSizeConsistent
    /\ NoDoubleFree
    /\ RefCountInvariant
    /\ EvictionSafety
    /\ MutexExclusion
    /\ CacheSlotConsistent

-----------------------------------------------------------------------------
\* Initial state

Init ==
    /\ graphState = [g \in GraphIds |-> GraphFree]
    /\ graphRefCount = [g \in GraphIds |-> 0]
    /\ graphKey = [g \in GraphIds |-> 0]
    /\ graphCacheSlot = [g \in GraphIds |-> None]
    /\ cacheSlots = [s \in CacheSlots |-> None]
    /\ cacheSize = 0
    /\ mutexHolder = None
    /\ threadState = [t \in ThreadIds |-> ThreadIdle]
    /\ threadTargetKey = [t \in ThreadIds |-> 0]
    /\ threadTargetGraph = [t \in ThreadIds |-> None]

-----------------------------------------------------------------------------
\* Actions

\* Thread starts a cache lookup
StartLookup(t, key) ==
    /\ threadState[t] = ThreadIdle
    /\ key \in Keys
    /\ threadState' = [threadState EXCEPT ![t] = ThreadLookup]
    /\ threadTargetKey' = [threadTargetKey EXCEPT ![t] = key]
    /\ UNCHANGED <<graphState, graphRefCount, graphKey, graphCacheSlot,
                   cacheSlots, cacheSize, mutexHolder, threadTargetGraph>>

\* Cache hit: find and use an existing graph
CacheHit(t) ==
    /\ threadState[t] = ThreadLookup
    /\ \E g \in GraphIds :
        /\ graphState[g] = GraphCached
        /\ graphKey[g] = threadTargetKey[t]
        /\ graphState' = [graphState EXCEPT ![g] = GraphInUse]
        /\ graphRefCount' = [graphRefCount EXCEPT ![g] = @ + 1]
        /\ threadState' = [threadState EXCEPT ![t] = ThreadExecuting]
        /\ threadTargetGraph' = [threadTargetGraph EXCEPT ![t] = g]
    /\ UNCHANGED <<graphKey, graphCacheSlot, cacheSlots, cacheSize,
                   mutexHolder, threadTargetKey>>

\* Cache miss: start compiling a new graph
CacheMiss(t) ==
    /\ threadState[t] = ThreadLookup
    /\ ~\E g \in GraphIds :
        /\ graphState[g] = GraphCached
        /\ graphKey[g] = threadTargetKey[t]
    /\ \E g \in GraphIds :
        /\ graphState[g] = GraphFree
        /\ graphState' = [graphState EXCEPT ![g] = GraphCreating]
        /\ graphKey' = [graphKey EXCEPT ![g] = threadTargetKey[t]]
        /\ threadState' = [threadState EXCEPT ![t] = ThreadCompiling]
        /\ threadTargetGraph' = [threadTargetGraph EXCEPT ![t] = g]
    /\ UNCHANGED <<graphRefCount, graphCacheSlot, cacheSlots, cacheSize,
                   mutexHolder, threadTargetKey>>

\* Acquire mutex for cache insertion
AcquireMutex(t) ==
    /\ threadState[t] = ThreadCompiling
    /\ mutexHolder = None
    /\ mutexHolder' = t
    /\ threadState' = [threadState EXCEPT ![t] = ThreadInserting]
    /\ UNCHANGED <<graphState, graphRefCount, graphKey, graphCacheSlot,
                   cacheSlots, cacheSize, threadTargetKey, threadTargetGraph>>

\* Insert graph into cache (with possible eviction)
InsertIntoCache(t) ==
    /\ threadState[t] = ThreadInserting
    /\ mutexHolder = t
    /\ LET g == threadTargetGraph[t] IN
        /\ g # None
        /\ graphState[g] = GraphCreating
        \* Graph goes to GraphInUse since thread is executing it (refcount=1)
        \* If cache is full, evict a slot (simplified: pick any with refcount 0)
        /\ IF cacheSize >= MaxCacheSize
           THEN \E evictSlot \in CacheSlots :
                LET evictGraph == cacheSlots[evictSlot] IN
                /\ evictGraph # None
                /\ graphRefCount[evictGraph] = 0  \* Eviction safety
                /\ graphState' = [graphState EXCEPT
                                  ![evictGraph] = GraphDeleted,
                                  ![g] = GraphInUse]
                /\ graphCacheSlot' = [graphCacheSlot EXCEPT
                                      ![evictGraph] = None,
                                      ![g] = evictSlot]
                /\ cacheSlots' = [cacheSlots EXCEPT ![evictSlot] = g]
                /\ cacheSize' = cacheSize  \* Size unchanged (evict+insert)
           ELSE \E freeSlot \in CacheSlots :
                /\ cacheSlots[freeSlot] = None
                /\ graphState' = [graphState EXCEPT ![g] = GraphInUse]
                /\ graphCacheSlot' = [graphCacheSlot EXCEPT ![g] = freeSlot]
                /\ cacheSlots' = [cacheSlots EXCEPT ![freeSlot] = g]
                /\ cacheSize' = cacheSize + 1
        /\ mutexHolder' = None  \* Release mutex
        /\ threadState' = [threadState EXCEPT ![t] = ThreadExecuting]
        /\ graphRefCount' = [graphRefCount EXCEPT ![g] = 1]
    /\ UNCHANGED <<graphKey, threadTargetKey, threadTargetGraph>>

\* Thread finishes using a graph
FinishExecution(t) ==
    /\ threadState[t] = ThreadExecuting
    /\ LET g == threadTargetGraph[t] IN
        /\ g # None
        /\ graphState[g] = GraphInUse
        /\ graphRefCount[g] > 0
        /\ IF graphRefCount[g] = 1
           THEN graphState' = [graphState EXCEPT ![g] = GraphCached]
           ELSE graphState' = graphState
        /\ graphRefCount' = [graphRefCount EXCEPT ![g] = @ - 1]
        /\ threadState' = [threadState EXCEPT ![t] = ThreadDone]
        /\ threadTargetGraph' = [threadTargetGraph EXCEPT ![t] = None]
    /\ UNCHANGED <<graphKey, graphCacheSlot, cacheSlots, cacheSize,
                   mutexHolder, threadTargetKey>>

\* Thread returns to idle
ReturnToIdle(t) ==
    /\ threadState[t] = ThreadDone
    /\ threadState' = [threadState EXCEPT ![t] = ThreadIdle]
    /\ threadTargetKey' = [threadTargetKey EXCEPT ![t] = 0]
    /\ UNCHANGED <<graphState, graphRefCount, graphKey, graphCacheSlot,
                   cacheSlots, cacheSize, mutexHolder, threadTargetGraph>>

\* Garbage collection: recycle deleted graphs back to free pool
RecycleGraph(g) ==
    /\ graphState[g] = GraphDeleted
    /\ graphRefCount[g] = 0
    /\ graphState' = [graphState EXCEPT ![g] = GraphFree]
    /\ graphKey' = [graphKey EXCEPT ![g] = 0]
    /\ UNCHANGED <<graphRefCount, graphCacheSlot, cacheSlots, cacheSize,
                   mutexHolder, threadState, threadTargetKey, threadTargetGraph>>

-----------------------------------------------------------------------------
\* Next state relation

Next ==
    \/ \E t \in ThreadIds, k \in Keys : StartLookup(t, k)
    \/ \E t \in ThreadIds : CacheHit(t)
    \/ \E t \in ThreadIds : CacheMiss(t)
    \/ \E t \in ThreadIds : AcquireMutex(t)
    \/ \E t \in ThreadIds : InsertIntoCache(t)
    \/ \E t \in ThreadIds : FinishExecution(t)
    \/ \E t \in ThreadIds : ReturnToIdle(t)
    \/ \E g \in GraphIds : RecycleGraph(g)
    \/ UNCHANGED vars

\* Fairness: eventually threads make progress
Fairness ==
    /\ WF_vars(\E t \in ThreadIds : CacheHit(t))
    /\ WF_vars(\E t \in ThreadIds : CacheMiss(t))
    /\ WF_vars(\E t \in ThreadIds : AcquireMutex(t))
    /\ WF_vars(\E t \in ThreadIds : InsertIntoCache(t))
    /\ WF_vars(\E t \in ThreadIds : FinishExecution(t))
    /\ WF_vars(\E t \in ThreadIds : ReturnToIdle(t))
    /\ WF_vars(\E g \in GraphIds : RecycleGraph(g))

Spec == Init /\ [][Next]_vars /\ Fairness
SpecNoFairness == Init /\ [][Next]_vars

-----------------------------------------------------------------------------
\* Liveness properties

\* Eventually all threads return to idle
EventuallyIdle ==
    \A t \in ThreadIds : threadState[t] # ThreadIdle ~> threadState[t] = ThreadIdle

\* Progress: threads eventually complete operations
Progress == []<>(\A t \in ThreadIds : threadState[t] = ThreadIdle)

=============================================================================
