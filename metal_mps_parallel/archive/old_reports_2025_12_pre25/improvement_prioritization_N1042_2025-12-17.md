# Improvement Prioritization N=1042

**Date**: 2025-12-17
**Current State**: 29.3% efficiency at 8 threads (Apple MPS ceiling)

## On Our Side: Prioritized Improvements

### Priority 1: Process-Based Parallelism Wrapper (LOW effort, HIGH gain)
**Expected Gain**: Near-linear scaling (70-90% efficiency possible)
**Effort**: 2-3 AI commits
**Risk**: Low

```python
# Example API
from torch_mps_parallel import ProcessPoolInference

with ProcessPoolInference(num_workers=8, model=model) as pool:
    results = pool.map(inputs)  # Each worker is a separate process
```

**Downsides of Process-Based Parallelism**:
1. **Memory overhead**: Each process loads separate model copy (~2-4x memory)
2. **IPC latency**: Data must be serialized/deserialized between processes
3. **Startup time**: Process spawning is slow (~100-500ms per process)
4. **Complexity**: Requires pickling models, managing process lifecycle
5. **No shared GPU memory**: Each process gets independent GPU allocations
6. **Error handling**: Process crashes harder to handle than thread exceptions

**When to use**: Batch inference, large models, when 8+ way parallelism needed

### Priority 2: Adaptive Path Selection (MEDIUM effort, MEDIUM gain)
**Expected Gain**: +5-10% efficiency
**Effort**: 3-4 AI commits
**Risk**: Medium (tuning required)

Select no-graph vs graph path based on tensor size:
- Small tensors (<4KB): Use no-graph (mutex overhead < graph compilation)
- Large tensors (>4KB): Use graph (compilation amortized)

```cpp
const size_t tensor_bytes = input.numel() * input.element_size();
const bool use_graph = parallel_streams_active || tensor_bytes > 4096;
```

### Priority 3: Per-Operation-Type Mutex Sharding (MEDIUM effort, LOW-MEDIUM gain)
**Expected Gain**: +5-10% efficiency
**Effort**: 4-5 AI commits
**Risk**: Medium (must verify correctness)

Currently all linear ops share one mutex. Sharding allows different operation types to run in parallel:

```cpp
// Current: All ops serialize
static std::mutex s_linear_nograph_mutex;

// Improved: Per-input-size sharding
static std::array<std::mutex, 8> s_linear_nograph_mutexes;
static size_t get_mutex_idx(const at::Tensor& input) {
    return std::hash<int64_t>{}(input.numel()) % 8;
}
```

### Priority 4: Operation Batching/Coalescing (HIGH effort, MEDIUM gain)
**Expected Gain**: +10-15% efficiency
**Effort**: 8-10 AI commits
**Risk**: High (complex implementation)

Batch multiple small operations into single mutex acquisition:
```cpp
struct OperationBatch {
    std::vector<LinearOp> ops;
    void execute() {
        std::lock_guard lock(s_linear_nograph_mutex);
        for (auto& op : ops) {
            encode_linear(op);  // All encoded under one lock
        }
    }
};
```

### Priority 5: Custom Metal Kernels (MLX approach) (VERY HIGH effort, HIGH gain)
**Expected Gain**: Near-linear scaling (bypass MPS entirely)
**Effort**: 50+ AI commits (essentially new backend)
**Risk**: Very high

Port MLX's "Steel GEMM" kernels to PyTorch. Would bypass MPS thread-safety issues entirely but requires massive engineering effort.

## On Apple's Side: What Could Be Done

### Option A: Submit Radar Bug Report (Our action)
**File**: `APPLE_RADAR_FB123456.md`
**URL**: https://feedbackassistant.apple.com
**Timeline**: Unknown (Apple doesn't commit to timelines)

The radar documents:
1. `MPSNDArrayMatrixMultiplication` internal shared state crash
2. Metal compute kernel crash at 4+ threads
3. Reproduction code
4. Impact on PyTorch/ML ecosystem

### Option B: Apple Fixes MPS Internals
**What Apple would need to do**:
1. Make `MPSNDArrayMatrixMultiplication` kernel instances thread-safe
2. Remove internal global state in MPS
3. Add thread-safety guarantees to MPS documentation

**Likelihood**: Unknown. Apple's MPS team may be aware (their MLX team avoided MPS).

### Option C: Apple Provides Thread-Safe Alternative
**Possibility**: Apple could add a thread-safe variant:
```objc
// Hypothetical thread-safe API
MPSNDArrayMatrixMultiplication *matmul =
    [[MPSNDArrayMatrixMultiplication alloc]
     initWithDevice:device
     threadSafe:YES];  // New option
```

### Option D: MPSGraph Improvements
Apple could improve MPSGraph to:
1. Reduce compilation overhead (currently hurts small tensors)
2. Add caching for common graph patterns
3. Provide explicit thread-safety guarantees

## MPS Source Code Availability

**Is MPS Open Source?** NO.

- MetalPerformanceShaders.framework is closed-source
- Only headers available in SDK: `/Library/Developer/CommandLineTools/SDKs/MacOSX15.4.sdk/System/Library/Frameworks/MetalPerformanceShaders.framework/`
- No implementation code on opensource.apple.com
- Apple's MLX avoided MPS by writing custom kernels

**What we CAN access**:
1. MPS headers (API definitions)
2. WWDC sessions on MPS
3. Apple developer documentation
4. MLX source code (Apple's alternative approach)

## Recommendation Matrix

| Improvement | Effort | Gain | Risk | Priority |
|-------------|--------|------|------|----------|
| Process parallelism wrapper | Low | High | Low | **1** |
| Adaptive path selection | Medium | Medium | Medium | **2** |
| Per-op-type mutex sharding | Medium | Low-Medium | Medium | **3** |
| Operation batching | High | Medium | High | 4 |
| Custom Metal kernels | Very High | High | Very High | 5 |
| Apple radar submission | None | Unknown | None | **Do now** |

## Immediate Actions

1. **Submit Apple Radar** - File APPLE_RADAR_FB123456.md at feedbackassistant.apple.com
2. **Implement process wrapper** - Provides immediate workaround for users needing >4 threads
3. **Document limitation** - Update PyTorch MPS docs about threading ceiling
