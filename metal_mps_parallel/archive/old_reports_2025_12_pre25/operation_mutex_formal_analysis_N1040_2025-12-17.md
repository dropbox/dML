# Operation-Level Mutex Formal Analysis N=1040

**Date**: 2025-12-17 10:30 PST
**Critical Finding**: Apple MPS has internal thread-safety issues requiring global serialization

## Executive Summary

The MPS backend has **two execution paths** for nn.Linear and other operations:
1. **No-graph path**: Fast per-operation but requires global mutex (serializes all threads)
2. **Graph path**: Uses MPSGraph, thread-safe but has compilation overhead

The code attempts to auto-select based on `parallel_streams_active`, but this may not be working optimally.

## The Two Paths

### Path 1: No-Graph (MPSNDArrayMatrixMultiplication)

```cpp
// Linear.mm:19
static std::mutex s_linear_nograph_mutex;  // GLOBAL SERIALIZER

static void _mps_linear_nograph(...) {
    auto kernel = cachedKernel->kernel<MPSNDArrayMatrixMultiplication>();

    dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
        // ... setup ...

        std::lock_guard<std::mutex> lock(s_linear_nograph_mutex);  // ALL THREADS SERIALIZE HERE
        [kernel encodeToCommandEncoder:computeEncoder ...];
    });
}
```

**Why the mutex exists:**
```cpp
// THREAD-SAFETY: Global mutex for MPSNDArrayMatrixMultiplication encoding.
// Apple's MPS framework has internal shared state that makes concurrent encoding
// of MPSNDArrayMatrixMultiplication kernels unsafe, even with per-thread instances.
```

### Path 2: Graph (MPSGraph)

```cpp
// Linear.mm:161+
MPSStream* stream = getCurrentMPSStream();
auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto* mpsGraph, auto* newCachedGraph) {
    // Build graph using MPSGraph operations
    auto outputTensor = [mpsGraph matrixMultiplicationWithPrimaryTensor:...];
});
// No global mutex - MPSGraph handles thread safety internally
```

## Path Selection Logic

```cpp
// Linear.mm:140-160
const bool parallel_streams_active =
    MPSStreamPool::isPoolAlive() && MPSStreamPool::instance().getActiveStreamCount() > 1;
const bool force_graph_path = force_graph_path_env || parallel_streams_active;

if (!force_graph_path && is_macos_13_or_newer(MACOS_VER_15_0_PLUS) && is_contiguous) {
    _mps_linear_nograph(...);  // Mutex path
} else {
    // MPSGraph path - no mutex
}
```

## The `getActiveStreamCount()` Implementation

```cpp
// MPSStream.mm:565
static std::atomic<bool> g_worker_stream_used{false};

// MPSStream.mm:785-787
size_t MPSStreamPool::getActiveStreamCount() const {
    return g_worker_stream_used.load(std::memory_order_acquire) ? 2 : 1;
}
```

`g_worker_stream_used` is set to `true` when:
1. A worker thread gets a stream (line 751)
2. `setCurrentStream()` is called with a non-default stream (line 781)
3. `getStream()` is called with `use_default=false` (line 845)

## Potential Issues

### Issue 1: Main Thread Always Uses No-Graph Path First

```cpp
if (pthread_main_np() == 1) {
    // Main thread uses default stream (id 0)
    tls_current_stream = getDefaultStream();
    // g_worker_stream_used stays FALSE
}
```

If the main thread calls linear BEFORE any worker thread gets a stream, it will use the no-graph path. The main thread's operations will ALWAYS use the no-graph path until a worker thread first gets a stream.

### Issue 2: Mutex Holds During Kernel Encoding Only

The mutex is held INSIDE `dispatch_sync_with_rethrow`, meaning:
1. Thread enters dispatch_sync (GPU work queued)
2. Thread acquires mutex
3. Thread encodes kernel
4. Thread releases mutex
5. dispatch_sync returns

The actual GPU work happens asynchronously. The mutex only serializes CPU-side encoding, not GPU execution.

### Issue 3: Seven Global Operation Mutexes

```cpp
s_linear_nograph_mutex      // Linear.mm:19
s_layer_norm_mutex          // Normalization.mm:31
s_bmm_tiled_mutex           // LinearAlgebra.mm:49
s_lu_decomposition_mutex    // LinearAlgebra.mm:50
s_lu_solve_mutex            // LinearAlgebra.mm:51
s_solve_triangular_mutex    // LinearAlgebra.mm:52
s_ndarray_identity_mutex    // OperationUtils.mm:484
```

All these have the same pattern - Apple's MPS internals are not thread-safe for these operations.

## Formal Verification Extensions

### TLA+ Model: Operation-Level Serialization

Extend MPSAllocator.tla to model operation mutexes:

```tla+
VARIABLES
    op_mutex_holder,        \* Thread holding operation mutex (0 = unlocked)
    path_selection,         \* Thread -> {"graph", "nograph", "none"}
    g_worker_stream_used    \* Global flag for worker stream usage

\* Path selection based on g_worker_stream_used
SelectPath(t) ==
    /\ pc[t] = "select_path"
    /\ IF g_worker_stream_used THEN
           path_selection' = [path_selection EXCEPT ![t] = "graph"]
       ELSE
           path_selection' = [path_selection EXCEPT ![t] = "nograph"]
    /\ pc' = [pc EXCEPT ![t] = "execute_op"]

\* No-graph path requires mutex
ExecuteNoGraphOp(t) ==
    /\ pc[t] = "execute_op"
    /\ path_selection[t] = "nograph"
    /\ op_mutex_holder = 0  \* Must acquire mutex
    /\ op_mutex_holder' = t
    /\ pc' = [pc EXCEPT ![t] = "encode_kernel"]
```

### Scalability Property: Graph Path Parallelism

```tla+
\* When g_worker_stream_used is true, multiple threads can be in execute_op simultaneously
GraphPathParallel ==
    g_worker_stream_used =>
        \* All threads should use graph path and can execute in parallel
        \A t \in Threads: path_selection[t] = "graph"

\* No-graph path serializes
NoGraphSerializes ==
    ~g_worker_stream_used =>
        Cardinality({t \in Threads : pc[t] \in {"encode_kernel"}}) <= 1
```

## Recommended Experiments

### Experiment 1: Force Graph Path
```bash
MPS_FORCE_GRAPH_PATH=1 python benchmark_parallel_mps.py
```
Expected: Better 8-thread scaling (graph path is thread-safe)
Trade-off: Possible slowdown for small tensors (graph compilation overhead)

### Experiment 2: Verify Path Selection
Add instrumentation to Linear.mm:
```cpp
static std::atomic<size_t> graph_path_count{0};
static std::atomic<size_t> nograph_path_count{0};

// Inside _mps_linear:
if (!force_graph_path && ...) {
    nograph_path_count.fetch_add(1);
    _mps_linear_nograph(...);
} else {
    graph_path_count.fetch_add(1);
    // Graph path
}
```

### Experiment 3: Remove Mutex (Stress Test)
Comment out the mutex and run stress test to verify if Apple has fixed the underlying issue in recent macOS versions:
```cpp
// std::lock_guard<std::mutex> lock(s_linear_nograph_mutex);  // TEST ONLY
[kernel encodeToCommandEncoder:...];
```

**WARNING**: This may crash. Only for testing hypothesis that Apple fixed the issue.

## Hypotheses for 30% Efficiency

### Hypothesis A: Benchmark Uses No-Graph Path
The benchmark might be hitting the no-graph path due to timing:
1. Main thread calls linear before workers get streams
2. `g_worker_stream_used` is false
3. All operations use no-graph path with mutex

### Hypothesis B: Graph Path Has Overhead
Even if graph path is used, it has compilation overhead that reduces effective throughput for small workloads.

### Hypothesis C: Other Mutexes
The benchmark uses nn.Linear and TransformerEncoderLayer. TransformerEncoderLayer uses LayerNorm which hits `s_layer_norm_mutex`.

## Verification Priority

1. **Experiment 1**: Run with `MPS_FORCE_GRAPH_PATH=1` to see if efficiency improves
2. **Experiment 2**: Add instrumentation to verify which path is being taken
3. **TLA+ Extension**: Model operation-level serialization
4. **CBMC Harness**: Verify race conditions in path selection

## Files Referenced

- `pytorch-mps-fork/aten/src/ATen/native/mps/operations/Linear.mm` - Linear operation with mutex
- `pytorch-mps-fork/aten/src/ATen/mps/MPSStream.mm` - getActiveStreamCount() implementation
- `mps-verify/specs/MPSAllocator.tla` - TLA+ model (needs extension)
