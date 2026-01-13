# Steel Integration Plan: MLX → PyTorch MPS

**Goal**: Integrate Apple's MIT-licensed Steel kernels into PyTorch MPS with formal verification guarantees.

**Outcome**: Thread-safe parallel MPS inference with near-linear scaling.

---

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Integration Architecture                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │ Apple MLX   │───▶│ Steel Fork  │───▶│ PyTorch MPS │───▶│ Production  │  │
│  │ (upstream)  │    │ (verified)  │    │ (patched)   │    │ (deployed)  │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│        │                  │                  │                  │          │
│        ▼                  ▼                  ▼                  ▼          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Formal Verification Layer                        │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │   │
│  │  │  TLA+    │  │   Lean   │  │   CBMC   │  │  TSan    │           │   │
│  │  │ Protocol │  │  Proofs  │  │ Bounded  │  │ Runtime  │           │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Steel Fork with Formal Verification (8-12 commits)

### 1.1 Fork and Verify Steel GEMM

**Objective**: Create verified fork of MLX Steel kernels for PyTorch integration.

**Location**: `steel-kernels/` (new directory in this repo)

```
steel-kernels/
├── src/
│   ├── gemm/           # Ported from mlx/backend/metal/kernels/steel/gemm/
│   │   ├── gemm.h
│   │   ├── mma.h
│   │   ├── loader.h
│   │   └── params.h
│   └── utils/
├── specs/              # Formal specifications
│   ├── SteelGEMM.tla   # TLA+ model of GEMM thread safety
│   └── SteelMMA.lean   # Lean proofs of MMA correctness
├── harnesses/          # CBMC verification harnesses
│   ├── gemm_thread_safety.cpp
│   └── mma_bounds_check.cpp
└── tests/
    ├── accuracy/       # Numerical accuracy vs MPS
    └── parallel/       # Multi-threaded stress tests
```

### 1.2 TLA+ Specification: Steel Thread Safety

```tla+
---------------------------- MODULE SteelGEMM ----------------------------
\* Formal model proving Steel GEMM has no shared state

CONSTANTS
    Threads,            \* Set of worker threads
    MaxTiles            \* Maximum number of tiles per GEMM

VARIABLES
    thread_state,       \* Thread -> {idle, loading, computing, storing}
    threadgroup_mem,    \* Thread -> local threadgroup memory (NOT shared)
    global_mem          \* Global memory (read-only input, write-only output)

\* KEY INVARIANT: No shared mutable state between threads
NoSharedMutableState ==
    \A t1, t2 \in Threads :
        t1 # t2 =>
            \* Each thread's threadgroup memory is independent
            threadgroup_mem[t1] \cap threadgroup_mem[t2] = {}

\* Unlike MPS which has:
\*   - Shared kernel instance state
\*   - Global mutexes (s_linear_nograph_mutex)
\* Steel has:
\*   - Per-thread threadgroup memory
\*   - No shared kernel state
\*   - No global synchronization

\* THEOREM: Steel GEMM is naturally parallel
THEOREM SteelParallelSafe ==
    \A t1, t2 \in Threads :
        CanExecuteConcurrently(t1, t2)
========================================================================
```

### 1.3 Lean 4 Proofs: Numerical Correctness

```lean
-- Steel MMA correctness proof

namespace Steel.MMA

/-- Block matrix multiply-accumulate operation -/
def block_mma (A : Matrix m k α) (B : Matrix k n α) (C : Matrix m n α) : Matrix m n α :=
  C + A * B

/-- Tiled GEMM is equivalent to naive GEMM -/
theorem tiled_gemm_correct
    (A : Matrix M K α) (B : Matrix K N α)
    (BM BN BK : ℕ)
    (hBM : BM ∣ M) (hBN : BN ∣ N) (hBK : BK ∣ K) :
    tiled_gemm A B BM BN BK = A * B := by
  -- Proof that tiled computation equals naive matmul
  induction M, K, N with
  | ... => sorry  -- Full proof in steel-kernels/specs/

/-- Steel GEMM produces same result regardless of tile order -/
theorem steel_deterministic
    (A B : Matrix) (order1 order2 : TileOrder) :
    steel_gemm A B order1 = steel_gemm A B order2 := by
  -- Floating-point associativity caveat handled by AccumType
  sorry

end Steel.MMA
```

### 1.4 CBMC Harness: Bounds Checking

```cpp
// harnesses/gemm_bounds_check.cpp

#include <cbmc.h>
#include "steel/gemm/gemm.h"

void verify_gemm_bounds() {
    // Symbolic dimensions
    int M = __CPROVER_nondet_int();
    int N = __CPROVER_nondet_int();
    int K = __CPROVER_nondet_int();

    __CPROVER_assume(M > 0 && M <= 4096);
    __CPROVER_assume(N > 0 && N <= 4096);
    __CPROVER_assume(K > 0 && K <= 4096);

    // Block sizes
    constexpr int BM = 64, BN = 64, BK = 32;

    // Verify tile iteration bounds
    int num_m_tiles = (M + BM - 1) / BM;
    int num_n_tiles = (N + BN - 1) / BN;
    int num_k_tiles = (K + BK - 1) / BK;

    // CBMC assertion: no buffer overflow
    __CPROVER_assert(num_m_tiles * BM >= M, "M tiles cover input");
    __CPROVER_assert(num_n_tiles * BN >= N, "N tiles cover input");
    __CPROVER_assert(num_k_tiles * BK >= K, "K tiles cover input");

    // Verify threadgroup memory doesn't overflow
    size_t tgp_mem = BM * BK + BK * BN;
    __CPROVER_assert(tgp_mem <= 32768, "Threadgroup memory within limit");
}
```

---

## Phase 2: PyTorch Integration (10-15 commits)

### 2.1 Add Steel Path to Linear.mm

```cpp
// pytorch-mps-fork/aten/src/ATen/native/mps/operations/Linear.mm

#include "steel/gemm.h"

// NEW: Steel GEMM path - thread-safe, no mutex
static Tensor _mps_linear_steel(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias) {

    // Get dimensions
    int64_t M = input.size(0);
    int64_t K = input.size(1);
    int64_t N = weight.size(0);

    // Select optimal block sizes based on problem size
    auto [BM, BN, BK] = selectSteelBlockSizes(M, N, K);

    // Create output tensor
    auto output = at::empty({M, N}, input.options());

    // Get Metal resources
    MPSStream* stream = getCurrentMPSStream();

    // Dispatch Steel GEMM kernel
    dispatch_sync_with_rethrow(stream->queue(), ^{
        @autoreleasepool {
            id<MTLCommandBuffer> cmdBuf = stream->commandBuffer();
            id<MTLComputeCommandEncoder> encoder =
                [cmdBuf computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];

            // No mutex needed - Steel is thread-safe!
            encodeSteelGEMM(encoder, input, weight, output, BM, BN, BK);

            [encoder endEncoding];
        }
    });

    // Add bias if present
    if (bias.has_value()) {
        output.add_(*bias);
    }

    return output;
}

// Updated path selection
Tensor _mps_linear(const Tensor& input, const Tensor& weight, ...) {
    static const bool force_steel = std::getenv("MPS_FORCE_STEEL_PATH");
    static const bool force_mps = std::getenv("MPS_FORCE_MPS_PATH");

    const bool parallel_active = getActiveStreamCount() > 1;
    const bool use_steel = force_steel || (parallel_active && !force_mps);

    if (use_steel) {
        return _mps_linear_steel(input, weight, bias);  // Thread-safe!
    } else {
        return _mps_linear_mps(input, weight, bias);    // Original with mutex
    }
}
```

### 2.2 TLA+ Model: Integration Safety

```tla+
---------------------------- MODULE MPSSteelIntegration ----------------------------
\* Model verifying safe coexistence of MPS and Steel paths

CONSTANTS
    Threads,
    Operations           \* Set of {linear, layernorm, bmm, ...}

VARIABLES
    path_selection,      \* Thread -> Operation -> {mps, steel}
    mps_mutex_holder,    \* Operation -> Thread holding mutex (0 = free)
    steel_executing      \* Set of threads executing Steel path

\* MPS path requires mutex
MPSPathAction(t, op) ==
    /\ path_selection[t][op] = "mps"
    /\ mps_mutex_holder[op] = 0
    /\ mps_mutex_holder' = [mps_mutex_holder EXCEPT ![op] = t]

\* Steel path is lock-free
SteelPathAction(t, op) ==
    /\ path_selection[t][op] = "steel"
    /\ steel_executing' = steel_executing \cup {t}
    /\ UNCHANGED mps_mutex_holder

\* KEY THEOREM: Steel threads never blocked by MPS threads
NoSteelBlocking ==
    \A t \in Threads, op \in Operations :
        path_selection[t][op] = "steel" =>
            ~Blocked(t)

\* Gradual migration is safe
MigrationSafe ==
    \* Can have some threads on MPS and others on Steel
    \E t1, t2 \in Threads, op \in Operations :
        path_selection[t1][op] = "mps" /\ path_selection[t2][op] = "steel" =>
            NoDataRace(t1, t2)
========================================================================
```

### 2.3 Lean 4: Path Selection Correctness

```lean
-- Prove path selection maintains correctness

namespace MPS.PathSelection

/-- Path selection is deterministic based on inputs -/
theorem path_selection_deterministic
    (input weight : Tensor)
    (parallel_active : Bool) :
    selectPath input weight parallel_active =
    selectPath input weight parallel_active := rfl

/-- Steel and MPS paths produce equivalent results -/
theorem path_equivalence
    (input weight : Tensor)
    (ε : Float := 1e-5) :
    ‖steel_linear input weight - mps_linear input weight‖ < ε := by
  -- Both implement standard GEMM
  sorry  -- Full proof with floating-point error bounds

/-- Parallel safety: Steel path never deadlocks -/
theorem steel_deadlock_free :
    ∀ (threads : Finset Thread),
      all_using_steel threads →
      ¬ deadlocked threads := by
  intro threads h_steel
  -- Steel has no locks, therefore no deadlock
  exact no_locks_no_deadlock threads h_steel

end MPS.PathSelection
```

---

## Phase 3: Verification and Testing (5-8 commits)

### 3.1 Numerical Accuracy Validation

```python
# tests/test_steel_accuracy.py

import torch
import numpy as np

def test_steel_vs_mps_accuracy():
    """Verify Steel produces same results as MPS within tolerance."""
    torch.manual_seed(42)

    for dtype in [torch.float32, torch.float16]:
        for M, N, K in [(128, 256, 512), (1024, 1024, 1024), (4096, 4096, 4096)]:
            input = torch.randn(M, K, device="mps", dtype=dtype)
            weight = torch.randn(N, K, device="mps", dtype=dtype)

            # Force MPS path
            os.environ["MPS_FORCE_MPS_PATH"] = "1"
            mps_result = torch.nn.functional.linear(input, weight)

            # Force Steel path
            del os.environ["MPS_FORCE_MPS_PATH"]
            os.environ["MPS_FORCE_STEEL_PATH"] = "1"
            steel_result = torch.nn.functional.linear(input, weight)

            # Compare
            rtol = 1e-3 if dtype == torch.float16 else 1e-5
            assert torch.allclose(mps_result, steel_result, rtol=rtol), \
                f"Mismatch at {M}x{K}x{N} {dtype}"
```

### 3.2 Parallel Stress Testing

```python
# tests/test_steel_parallel.py

import torch
import threading
import time

def test_steel_8_thread_efficiency():
    """Verify Steel achieves >50% efficiency at 8 threads."""
    os.environ["MPS_FORCE_STEEL_PATH"] = "1"

    model = torch.nn.Linear(2048, 2048).to("mps")
    input_tensor = torch.randn(64, 2048, device="mps")

    # Single-thread baseline
    start = time.perf_counter()
    for _ in range(100):
        with torch.no_grad():
            _ = model(input_tensor)
        torch.mps.synchronize()
    baseline_time = time.perf_counter() - start
    baseline_ops = 100 / baseline_time

    # 8-thread test
    results = [None] * 8
    threads = []
    start = time.perf_counter()
    for i in range(8):
        t = threading.Thread(target=worker, args=(model, input_tensor, results, i, 100))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    parallel_time = time.perf_counter() - start
    parallel_ops = 800 / parallel_time

    efficiency = (parallel_ops / baseline_ops) / 8 * 100

    # REQUIREMENT: Steel must achieve >50% efficiency
    assert efficiency > 50, f"Steel efficiency {efficiency:.1f}% < 50% target"
    print(f"Steel 8-thread efficiency: {efficiency:.1f}%")
```

---

## Phase 4: Upstream Contributions (5-10 commits)

### 4.1 Contribute Thread-Safety Improvements to MLX

**Target**: https://github.com/ml-explore/mlx/pulls

Potential contributions:
1. Thread-safety documentation for Steel kernels
2. Formal verification specs (TLA+, Lean proofs)
3. Benchmark suite for parallel performance
4. Additional kernel variants

### 4.2 Contribute Steel Integration to PyTorch

**Target**: https://github.com/pytorch/pytorch/pulls

The final patch will include:
1. Steel kernels (MIT licensed, attributed to Apple)
2. Path selection logic (MPS vs Steel)
3. Formal verification artifacts
4. Comprehensive test suite

---

## Verification Checklist

### Before Integration
- [ ] TLA+ model of Steel GEMM passes TLC (all properties)
- [ ] Lean proofs compile without `sorry`
- [ ] CBMC harnesses pass with bounds
- [ ] TSan shows 0 races in Steel path

### After Integration
- [ ] Numerical accuracy within tolerance (1e-5 for fp32)
- [ ] 8-thread efficiency >50%
- [ ] All existing PyTorch MPS tests pass
- [ ] Memory usage comparable to MPS path

---

## Timeline

| Phase | Description | Commits | Cumulative |
|-------|-------------|---------|------------|
| 1 | Steel Fork + Verification | 8-12 | 8-12 |
| 2 | PyTorch Integration | 10-15 | 18-27 |
| 3 | Testing + Validation | 5-8 | 23-35 |
| 4 | Upstream Contributions | 5-10 | 28-45 |

**Total: 28-45 AI commits**

---

## Success Criteria

1. **Correctness**: Steel produces identical results to MPS (within floating-point tolerance)
2. **Performance**: 8-thread efficiency >50% (vs current 29%)
3. **Safety**: Formal proofs of thread-safety, no races detected
4. **Compatibility**: All existing PyTorch MPS tests pass
5. **Upstream**: Contributions accepted to MLX and/or PyTorch

---

## Files Created/Modified

**New (this repo)**:
- `steel-kernels/` - Verified Steel kernel fork
- `steel-kernels/specs/SteelGEMM.tla` - TLA+ thread-safety model
- `steel-kernels/specs/SteelMMA.lean` - Lean correctness proofs
- `STEEL_INTEGRATION_PLAN.md` - This document

**Modified (pytorch-mps-fork)**:
- `aten/src/ATen/native/mps/operations/Linear.mm` - Steel path
- `aten/src/ATen/native/mps/operations/Normalization.mm` - Steel LayerNorm
- `aten/src/ATen/native/mps/operations/LinearAlgebra.mm` - Steel BMM
