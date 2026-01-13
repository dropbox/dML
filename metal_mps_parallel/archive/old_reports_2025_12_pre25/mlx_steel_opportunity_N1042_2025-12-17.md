# MLX Steel Kernels: Opportunity for PyTorch MPS

**Date**: 2025-12-17
**License**: MIT (Apple Inc.) - **Fully compatible with PyTorch BSD-3**
**Source**: https://github.com/ml-explore/mlx

## Executive Summary

Apple's MLX framework contains **32,000+ lines of MIT-licensed Metal shader code** that bypasses MetalPerformanceShaders entirely. This code could be adapted for PyTorch MPS to eliminate the thread-safety limitations we discovered.

## What MLX Provides

### Steel GEMM (~3,000 lines)
High-performance matrix multiplication without MPS:

```
mlx/backend/metal/kernels/steel/gemm/
├── gemm.h         (294 lines) - Main GEMM kernel template
├── mma.h         (1146 lines) - Matrix multiply-accumulate
├── nax.h         (1084 lines) - NAX (Neural Accelerator) support
├── loader.h       (137 lines) - Memory loading utilities
├── transforms.h    (71 lines) - Output transforms
└── params.h        (64 lines) - Kernel parameters
```

**Key features**:
- Tiled GEMM with configurable block sizes (BM, BN, BK)
- Supports transpose variations (NN, NT, TN, TT)
- Uses threadgroup memory for fast access
- Templated for different data types (float16, float32, bfloat16)
- No global mutex - fully concurrent

### Steel Attention (~1,500 lines)
Fused attention kernels:
```
mlx/backend/metal/kernels/steel/attn/
├── attn.h
├── mma.h
├── loader.h
└── params.h
```

### Steel Convolution
Conv2D implementations:
```
mlx/backend/metal/kernels/steel/conv/
├── conv.h
├── loader.h
└── params.h
```

### Other Operations (~27,000 lines)
- `binary.h` - Element-wise binary ops
- `unary.h` - Element-wise unary ops
- `reduce.h` - Reduction operations
- `softmax.h` - Softmax kernels
- `scan.h` - Prefix scan
- `sort.h` - Sorting
- `quantized.h` - Quantization support

## License Compatibility

| Project | License | Compatible? |
|---------|---------|-------------|
| MLX | MIT | ✅ Yes |
| PyTorch | BSD-3-Clause | ✅ Yes |
| Our Patch | BSD-3-Clause | ✅ Yes |

MIT is permissive and allows:
- Commercial use
- Modification
- Distribution
- Sublicensing

Only requirement: Include copyright notice.

## Integration Approaches

### Approach 1: Direct Port (Recommended)
Port Steel GEMM directly into PyTorch MPS as alternative to `MPSNDArrayMatrixMultiplication`:

```cpp
// pytorch-mps-fork/aten/src/ATen/native/mps/operations/Linear.mm

#include "steel/gemm.h"  // Port from MLX

static void _mps_linear_steel(...) {
    // Use Steel GEMM - no mutex needed!
    auto kernel = getSteel_GEMM_Kernel<float>(BM, BN, BK, ...);
    dispatch_to_metal(kernel, input, weight, output);
}

Tensor _mps_linear(const Tensor& input, const Tensor& weight, ...) {
    const bool use_steel = parallel_streams_active || force_steel_env;

    if (use_steel) {
        return _mps_linear_steel(input, weight, ...);  // Thread-safe!
    } else {
        return _mps_linear_mps(input, weight, ...);    // Original MPS path
    }
}
```

**Effort**: ~15-20 AI commits
**Risk**: Medium (need to match MPS numerical precision)
**Gain**: Near-linear scaling for Linear ops

### Approach 2: MLX as Backend
Use MLX directly as an alternative PyTorch backend:

```python
import torch
torch.set_default_device("mlx")  # Use MLX instead of MPS
```

**Effort**: ~50+ AI commits (new backend)
**Risk**: High (different tensor semantics)
**Gain**: Full MLX performance

### Approach 3: Selective Kernel Replacement
Replace only the problematic MPS operations:

| Operation | MPS Issue | Steel Replacement |
|-----------|-----------|-------------------|
| `nn.Linear` | `s_linear_nograph_mutex` | `steel/gemm` |
| `nn.LayerNorm` | `s_layer_norm_mutex` | `steel/` + custom |
| `bmm` | `s_bmm_tiled_mutex` | `steel/gemm` (batched) |

**Effort**: ~10-15 AI commits
**Risk**: Low-Medium
**Gain**: Eliminates worst bottlenecks

## Technical Details

### Steel GEMM Architecture

```cpp
// Tiled GEMM with threadgroup memory
template <typename T, int BM, int BN, int BK, int WM, int WN, ...>
struct GEMMKernel {
    // Tile sizes for A and B matrices
    STEEL_CONST short tgp_mem_size_a = BM * (BK + padding);
    STEEL_CONST short tgp_mem_size_b = BK * (BN + padding);

    // Main GEMM loop
    static METAL_FUNC void gemm_loop(
        threadgroup T* As,      // Tile of A in threadgroup memory
        threadgroup T* Bs,      // Tile of B in threadgroup memory
        int gemm_k_iterations,  // Number of K-dimension tiles
        ...
    ) {
        for (int k = 0; k < gemm_k_iterations; k++) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // Load tile from global to threadgroup memory
            loader_a.load_safe(...);
            loader_b.load_safe(...);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // Compute partial results
            mma_op.mma(...);
        }
    }
};
```

### Why Steel Is Thread-Safe

1. **No shared state**: Each kernel invocation is independent
2. **Threadgroup memory only**: No global synchronization needed
3. **No MPS dependency**: Pure Metal compute shaders
4. **Per-invocation resources**: Command encoder creates fresh state

## Implementation Roadmap

### Phase 1: Proof of Concept (3-5 commits)
1. Port `steel/gemm/gemm.h` to PyTorch
2. Create minimal integration in `Linear.mm`
3. Verify numerical accuracy vs MPS
4. Benchmark single-threaded performance

### Phase 2: Integration (5-8 commits)
1. Add path selection logic (Steel vs MPS)
2. Handle edge cases (small tensors, non-contiguous)
3. Add data type support (float16, bfloat16)
4. Thread safety testing

### Phase 3: Optimization (5-7 commits)
1. Tune block sizes for Apple Silicon generations
2. Add batched GEMM support
3. Optimize memory access patterns
4. Profile and tune

### Phase 4: Additional Operations (5-10 commits)
1. Port LayerNorm using Steel primitives
2. Port BMM
3. Port attention if needed

**Total**: ~20-30 AI commits for significant improvement

## Expected Results

| Metric | Current (MPS) | With Steel |
|--------|---------------|------------|
| 2-thread efficiency | 77% | ~85-90% |
| 4-thread efficiency | 47% | ~70-80% |
| 8-thread efficiency | 29% | ~50-60% |
| Bottleneck | MPS mutex | GPU saturation |

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Numerical differences | Extensive accuracy testing vs MPS |
| Performance regression (small tensors) | Keep MPS path as fallback |
| Maintenance burden | Upstream to PyTorch for shared maintenance |
| Apple Silicon version differences | Test on M1/M2/M3/M4 |

## Conclusion

Apple's MLX provides a production-ready, MIT-licensed solution to the exact problem we face. The Steel GEMM kernels are designed by Apple engineers specifically to avoid MPS limitations. Porting these to PyTorch MPS is a viable path to dramatically improved parallel scaling.

## References

- MLX Repository: https://github.com/ml-explore/mlx
- Steel GEMM: `mlx/backend/metal/kernels/steel/gemm/`
- MLX License: MIT (Copyright © 2023 Apple Inc.)
- PyTorch License: BSD-3-Clause
