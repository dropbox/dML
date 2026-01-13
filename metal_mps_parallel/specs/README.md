# TLA+ Formal Specifications for MPS Parallel Inference

This directory contains TLA+ formal specifications for the key concurrent data structures in the PyTorch MPS backend fork.

## Specifications

### 1. MPSStreamPool.tla

**Models**: `aten/src/ATen/mps/MPSStream.h` - MPSStreamPool class

**Key mechanisms verified**:
- Lock-free freelist using atomic bitmask (`free_slots_mask_`)
- TLS-based thread-to-stream binding
- Stream 0 (default) always available
- Slot recycling on thread exit

**Verified properties**:
- `MutualExclusion`: No two threads bound to same stream
- `PoolIntegrity`: Freelist correctly reflects stream states
- `BoundStreamsHaveOwner`: Every bound stream has owning thread
- `NoOrphanedBindings`: Thread bindings always point to valid streams
- `DeadlockFree`: System can always make progress

**TLC Results** (NumStreams=4, NumThreads=3, MaxOperations=12):
```
7,981 states generated, 1,992 distinct states found
No errors found
Search depth: 18
```

### 2. MPSAllocator.tla

**Models**: `aten/src/ATen/mps/MPSAllocator.h` - MPSHeapAllocatorImpl class

**Key mechanisms verified**:
- Buffer pools with per-pool mutex
- ABA counters for buffer IDs (`buffer_counter`)
- Stream-aware allocation (`alloc_stream`, `stream_uses`)
- Pending buffer states for async completion

**Verified properties**:
- `ABAMonotonicity`: Buffer counter only increases (prevents ABA bugs)
- `NoIDReuse`: Each allocated buffer has unique ID
- `FreebufferConsistency`: Free buffers are in available list
- `PendingBufferConsistency`: Pending buffers tracked correctly
- `NoUseAfterFree`: Cannot use a freed buffer

**TLC Results** (NumBuffers=3, NumStreams=2, NumThreads=2, MaxOperations=10):
```
2,821,612 states generated, 396,567 distinct states found
No errors found
Search depth: 16
```

### 3. MPSEvent.tla

**Models**: `aten/src/ATen/mps/MPSEvent.h` - MPSEvent and MPSEventPool classes

**Key mechanisms verified**:
- Event pool with freelist (`m_pool`) and in-use map (`m_in_use_events`)
- MTLSharedEvent signal counter semantics
- Reference counting for callback survival (shared_ptr pattern)
- Event lifecycle: pooled -> acquired -> recorded -> signaled -> released

**Verified properties**:
- `EventIDUniqueness`: No two in-use events share ID
- `CallbackSurvival`: Events with pending callbacks can't be released (KEY)
- `SignalCounterMonotonicity`: Signal counter only increases
- `PoolInUsePartition`: Events in pool XOR in use (never both)
- `NoUseAfterRelease`: Released events properly pooled

**TLC Results** (NumEvents=3, NumStreams=2, NumThreads=2, MaxOperations=10):
```
11,914,912 states generated, 1,389,555 distinct states found
No errors found
Search depth: 25
```

### 4. MPSBatchQueue.tla

**Models**: `pytorch-mps-fork/aten/src/ATen/mps/MPSBatchQueue.{h,mm}` - producer/consumer batching queue

**Key mechanisms verified**:
- Queue lifecycle: start/stop/drain semantics
- Producer submission vs worker processing interleavings
- No stuck futures (submitted requests eventually complete)

**Verified properties**:
- `SafetyInvariant`: No orphaned requests after stop, no negative counters

**TLC Results** (NumUserThreads=3, NumWorkers=2, MaxOperations=20):
```
24,419 states generated, 6,635 distinct states found
No errors found
Search depth: 21
```

### 5. MPSGraphCache.tla

**Models**: `aten/src/ATen/native/mps/MPSGraphCache.*` patterns - compiled graph cache coherency under concurrency

**Key mechanisms verified**:
- Reference counting for cached graph objects
- Eviction safety (no eviction while in use)
- Mutex exclusion for cache operations

**Verified properties**:
- `CacheSizeInvariant`, `NoDoubleFree`, `RefCountInvariant`, `EvictionSafety`, `MutexExclusion`

**TLC Results** (MaxCacheSize=2, MaxGraphs=4, NumThreads=2):
```
776,185 states generated, 218,704 distinct states found
No errors found
Search depth: 38
```

### 6. MPSCommandBuffer.tla

**Models**: `aten/src/ATen/mps/MPSStream.mm` - `_commandBuffer`/`_prevCommandBuffer` lifecycle

**Key mechanisms verified**:
- `_commandBuffer` remains uncommitted while referenced as the active buffer
- `_prevCommandBuffer` is never uncommitted (commit-before-wait discipline)
- `addCompletedHandler` only targets uncommitted buffers (post-commit registration forbidden)

**Verified properties**:
- `CommandBufferUncommitted`, `PrevBufferCommitted`, `NoIllegalHandler`, `NoRefsToFree`

**TLC Results** (NumBuffers=3):
```
3,559 states generated, 807 distinct states found
No errors found
Search depth: 12
```

### 7. MPSForkHandler.tla

**Models**: `pthread_atfork` fork safety handlers for GPU context management

**Key mechanisms verified**:
- Fork preparation flushes all pending GPU work
- Child process correctly resets to uninitialized state (GPU context invalid)
- Parent process preserves GPU context and state after fork
- Fork handlers execute in correct order (prepare -> parent/child)
- Multiple sequential forks handled correctly

**Verified properties**:
- `PrepareFlushes`: All pending work flushed before fork
- `ChildResets`: Child GPU context invalid, uninitialized state
- `ParentPreserves`: Parent state intact after fork
- `ChildNoLeaks`: No resource leaks in child from parent's streams
- `HandlerOrder`: Fork phases execute in correct sequence
- `ChildContextInvalid`: Child must reinitialize before using GPU

**TLC Results** (MaxStreams=3, MaxPendingOps=3, MaxForkDepth=2):
```
8,606 states generated, 1,623 distinct states found
No errors found
Search depth: 46
```

### 8. MPSKernelCache.tla

**Models**: `aten/src/ATen/native/mps/OperationUtils.h` - MPSKernelCache class (thread-local cache for compiled Metal kernels)

**Key mechanisms verified**:
- Thread-local singleton pattern (`thread_local std::unique_ptr<MPSKernelCache>`)
- Key-based lookup with collision detection
- Thread isolation (each thread has its own independent cache)
- Kernel lifecycle: free -> compiling -> cached -> in-use -> cached/deleted

**Verified properties**:
- `KeyUniqueness`: No key collisions within a thread's cache
- `KernelValidity`: Cached kernels are in valid state
- `NoDoubleFree`: Deleted kernels stay deleted
- `CacheSizeConsistent`: Cache size matches actual count
- `CacheSizeInvariant`: Cache size bounded
- `ThreadIsolation`: Kernels in cache are owned by that thread
- `LookupCorrectness`: Lookup returns correct kernel for key

**TLC Results** (MaxCacheSize=3, MaxKernels=6, NumThreads=2):
```
107,884,993 states generated, 30,167,872 distinct states found
No errors found
Search depth: 39
```

### 9. MPSTLSBinding.tla

**Models**: `aten/src/ATen/mps/MPSStream.h` - Thread-local storage (TLS) binding mechanism

**Key mechanisms verified**:
- Thread-local storage semantics (each thread has its own TLS slot)
- TLS set operation (setCurrentMPSStream)
- TLS get operation (getCurrentMPSStream)
- Thread exit cleanup (pthread destructor callback)
- Stream reference counting via TLS bindings

**Verified properties**:
- `ThreadIsolation`: Threads can only access their own TLS slot
- `BindingValidity`: TLS always points to valid stream (0 to NumStreams)
- `DeadThreadCleanup`: Dead threads have no TLS binding (cleared by destructor)
- `RefCountNonNegative`: Stream references never go negative
- `NoStaleBindings`: Alive threads have valid stream binding with ref > 0
- `DeadlockFree`: System can always make progress

**TLC Results** (NumStreams=3, NumThreads=3, MaxOperations=15):
```
48,621 states generated, 13,310 distinct states found
No errors found
Search depth: 27
```

### 10. MPSFullSystem.tla

**Models**: Composed system integrating StreamPool, Allocator, and Event subsystems

**Key mechanisms verified**:
- Cross-component resource management
- Stream-to-thread binding with resource affinity
- Buffer and event lifecycle across component boundaries
- Mutex coordination between subsystems
- System-wide resource cleanup patterns

**Verified properties**:
- `StreamMutualExclusion`: No two threads bound to same stream
- `StreamAffinity`: Allocated resources bound to valid streams
- `CallbackSurvival`: Events with pending callbacks cannot be released
- `PoolConsistency`: Pooled resources have zero associations
- `ResourceOwnership`: Acquired resources have proper reference counts
- `MutexExclusivity`: Only one thread holds each mutex
- `DeadlockFree`: System can always make progress
- `NoOrphanedBuffers`: All allocated buffers have stream owner
- `NoOrphanedEvents`: All acquired events have reference count

**TLC Results** (NumStreams=3, NumThreads=2, NumBuffers=2, NumEvents=2, MaxOperations=20):
```
8,036,503 states generated, 961,821 distinct states found
No errors found
Search depth: 28
```

## Running the Model Checker

### Prerequisites

- Java 21+ (OpenJDK recommended)
- TLA+ tools (downloaded to `specs/tools/tla2tools.jar`)

### Setup

```bash
# Install Java (macOS)
brew install openjdk@21

# Configure Java
export JAVA_HOME=$(/usr/libexec/java_home -v 21)
export PATH="$JAVA_HOME/bin:$PATH"

# Tools already downloaded
ls specs/tools/tla2tools.jar
```

### Run Verification

```bash
# Run all specifications
./specs/run_tlc.sh MPSStreamPool
./specs/run_tlc.sh MPSAllocator
./specs/run_tlc.sh MPSEvent
./specs/run_tlc.sh MPSBatchQueue
./specs/run_tlc.sh MPSGraphCache
./specs/run_tlc.sh MPSCommandBuffer
./specs/run_tlc.sh MPSForkHandler
./specs/run_tlc.sh MPSKernelCache
./specs/run_tlc.sh MPSTLSBinding
./specs/run_tlc.sh MPSFullSystem

# Or manually
cd specs
java -XX:+UseParallelGC -Xmx4g -cp tools/tla2tools.jar tlc2.TLC \
    -workers auto -cleanup MPSStreamPool
```

## Configuration Files

Each specification has a `.cfg` file that defines:
- **CONSTANT** values for bounded model checking
- **SPECIFICATION** entry point
- **INVARIANT** safety properties to check
- **PROPERTY** temporal properties to check

The constants are set small for tractable state spaces:
- NumStreams/Events/Buffers: 2-4 (vs 32 in production)
- NumThreads: 2-3 (vs potentially many more)
- MaxOperations: 10-12 (limits state explosion)

## Verified Properties Summary

| Specification | States | Distinct | Depth | Key Properties |
|--------------|--------|----------|-------|----------------|
| MPSStreamPool | 7,981 | 1,992 | 18 | Mutual exclusion, pool integrity |
| MPSAllocator | 2.8M | 397K | 16 | ABA monotonicity, no use-after-free |
| MPSEvent | 11.9M | 1.4M | 25 | Callback survival, ID uniqueness |
| MPSBatchQueue | 24,419 | 6,635 | 21 | Stop drains, no stuck futures |
| MPSGraphCache | 776,185 | 218,704 | 38 | No double-free, eviction safety |
| MPSCommandBuffer | 3,559 | 807 | 12 | No handler-after-commit, pointer safety |
| MPSForkHandler | 8,606 | 1,623 | 46 | Fork safety, child reset, parent preserves |
| MPSKernelCache | 107.9M | 30.2M | 39 | Key uniqueness, thread isolation |
| MPSTLSBinding | 48,621 | 13,310 | 27 | Thread isolation, binding validity, cleanup |
| MPSFullSystem | 8.0M | 962K | 28 | Cross-component safety, deadlock freedom |

All specifications above pass TLC model checking with no errors.

**Note:** Additional focused specs (e.g., encoding lock, dispatch queue context,
bounded-wait, and parallel-progress) live alongside these core models in this
directory; see `FORMAL_VERIFICATION_ROADMAP.md` for the full inventory.

## Mapping to Implementation

| TLA+ Concept | C++ Implementation |
|--------------|-------------------|
| `free_mask` bitmask | `MPSStreamPool::free_slots_mask_` atomic |
| `thread_bindings` | Thread-local storage via pthread |
| `buffer_counter` | `BufferBlock::buffer_counter` atomic |
| `ref_counts` | `shared_ptr` in `m_in_use_events` |
| `signal_counters` | `MTLSharedEvent` signalValue |

## Limitations

These specifications model the high-level concurrent behavior, not:
- Objective-C memory management
- Metal command buffer semantics
- Actual GPU execution timing
- Performance characteristics

The models prove absence of certain classes of bugs (deadlock, race conditions on pool state) but don't verify the correctness of Metal API usage.

---

## Apalache Symbolic Model Checking

In addition to TLC's explicit-state model checking, several specifications support Apalache's symbolic model checking using SMT solvers (Z3).

### Specifications with Apalache Support

| Specification | Apalache Config | Type Annotations | Status |
|--------------|-----------------|------------------|--------|
| MPSStreamPool | `*_Apalache.cfg` | ✅ Complete | Verified |
| MPSAllocator | `*_Apalache.cfg` | ✅ Complete | Verified |
| MPSEvent | `*_Apalache.cfg` | ✅ Complete | Verified |
| MPSFullSystem | `*_Apalache.cfg` | ✅ Complete | Verified |
| MPSTLSBinding | `*_Apalache.cfg` | ✅ Complete | Verified |
| MPSCommandBuffer | `*_Apalache.cfg` | ✅ Complete | Verified (N=1360) |
| MPSBatchQueue | `*_Apalache.cfg` | ✅ Complete | Annotations added (N=1361) |
| MPSForkHandler | `*_Apalache.cfg` | ✅ Complete | Annotations added (N=1361) |
| MPSGraphCache | `*_Apalache.cfg` | ✅ Complete | Annotations added (N=1361) |
| MPSKernelCache | `*_Apalache.cfg` | ✅ Complete | Annotations added (N=1361) |

### Running Apalache

```bash
# Setup (requires Java 17+)
export JAVA_HOME=/opt/homebrew/opt/openjdk@21
export PATH=$JAVA_HOME/bin:$PATH

# Run Apalache on a spec
cd specs
../tools/apalache/bin/apalache-mc check \
  --config=MPSCommandBuffer_Apalache.cfg MPSCommandBuffer.tla
```

### Apalache vs TLC

| Aspect | TLC | Apalache |
|--------|-----|----------|
| Approach | Explicit state enumeration | Symbolic (SMT solver) |
| State space | Exhaustive within bounds | Bounded symbolic |
| Strengths | Complete coverage | Unbounded types, induction |
| Requirement | Small constants | Type annotations |
| Speed | Scales with state count | Scales with constraint complexity |

### Adding Apalache Support to New Specs

Specs require type annotations for Apalache. Add `\* @type: <type>;` comments before CONSTANTS and VARIABLES:

```tla
CONSTANTS
    \* @type: Int;
    NumBuffers,
    \* @type: Bool;
    EnableFeature

VARIABLES
    \* @type: Int -> Str;
    buf_state,
    \* @type: Int;
    counter
```

---

## Clang Thread Safety Analysis (TSA)

In addition to TLA+ model checking, the MPS backend now includes Clang TSA annotations for compile-time race detection.

### Annotated Headers

| Header | Annotations | Description |
|--------|-------------|-------------|
| `MPSThreadSafety.h` | 22 macros | TSA macro definitions |
| `MPSStream.h` | 6 | Stream pool state protection |
| `MPSEvent.h` | 13 | Event pool and event state protection |
| `MPSAllocator.h` | 15 | Buffer pool and allocator state protection |

### TSA Macro Reference

```cpp
// Field annotations
MPS_GUARDED_BY(mutex)       // Field protected by mutex
MPS_PT_GUARDED_BY(mutex)    // Pointed-to data protected by mutex

// Function annotations
MPS_REQUIRES(mutex)         // Function requires mutex to be held
MPS_EXCLUDES(mutex)         // Function must NOT hold mutex
MPS_ACQUIRE(mutex)          // Function acquires mutex
MPS_RELEASE(mutex)          // Function releases mutex

// Escape hatch (use sparingly!)
MPS_NO_THREAD_SAFETY_ANALYSIS
```

### Lock Hierarchy

To prevent deadlocks, locks must be acquired in this order:

| Level | Mutex | Purpose |
|-------|-------|---------|
| 1 | `MPSStreamPool::stream_creation_mutex_` | Stream pool creation |
| 2 | `BufferPool::pool_mutex` | Per-pool allocation |
| 3 | `MPSHeapAllocatorImpl::m_mutex` / `MPSEventPool::m_mutex` | Global state |
| 4 | `MPSStream::_streamMutex` / `MPSEvent::m_mutex` | Per-object state |
| 5 | `getGlobalMetalEncodingMutex()` | Metal encoding (always last) |

### Running TSA Analysis

```bash
# Quick validation of annotations
./scripts/run_tsa.sh

# Full analysis during PyTorch build
CMAKE_CXX_FLAGS="-Wthread-safety -Wthread-safety-negative" python setup.py build
```

### TSA vs TLA+

| Aspect | TLA+ | Clang TSA |
|--------|------|-----------|
| Analysis type | Model checking | Static analysis |
| Finds | Protocol bugs, deadlocks | Missing locks, lock order |
| Coverage | High-level design | Implementation code |
| When | Design phase | Build time |
| Cost | Slow (state explosion) | Fast (compile time) |

Both tools complement each other: TLA+ verifies the concurrent protocol design, while TSA enforces correct lock discipline in implementation.

---

## References

- [TLA+ Video Course](https://lamport.azurewebsites.net/video/videos.html)
- [TLA+ Hyperbook](https://lamport.azurewebsites.net/tla/hyperbook.html)
- [Learn TLA+](https://learntla.com/)
- [Clang TSA Documentation](https://clang.llvm.org/docs/ThreadSafetyAnalysis.html)
