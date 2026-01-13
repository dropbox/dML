# Sync Frequency Data Correction

**Worker N=1410**
**Date: 2025-12-20**

## Issue Found

N=1409 reported 6.8x sync frequency improvement (5,610 → 38,329 ops/s). This used a 1-layer model, not the standard 3-layer benchmark model.

## Verification Test Results

Testing with the standard 3-layer benchmark model (512→1024→1024→512):

| Sync Every | Ops/s | Speedup |
|------------|-------|---------|
| 1 | 4,346 | 1.00x |
| 2 | 5,770 | 1.33x |
| 4 | 6,585 | 1.52x |
| 8 | 7,045 | 1.62x |
| 16 | 7,306 | 1.68x |
| 32 | 7,844 | 1.80x |
| 64 | 9,444 | 2.17x |
| 128 | 10,260 | 2.36x |
| 256 | 10,461 | **2.41x** |

## Model Complexity Comparison

| Model | Sync Every 1 | Sync Every 256 | Speedup |
|-------|--------------|----------------|---------|
| 1-layer tiny (64x64) | 7,945 ops/s | 41,252 ops/s | 5.19x |
| 1-layer (512x512) | 8,109 ops/s | 41,338 ops/s | 5.10x |
| 2-layer (512x512) | 5,756 ops/s | 17,059 ops/s | 2.96x |
| **3-layer (benchmark)** | 4,451 ops/s | 10,488 ops/s | **2.36x** |
| Pure matmul 512x512 | 5,909 ops/s | 29,707 ops/s | 5.03x |

## Explanation

The 6.8x speedup is achievable with simpler operations (1-layer model: 5.1x, pure matmul: 5.0x). However, the standard benchmark model (3-layer) only sees 2.4x improvement.

**Why simpler operations benefit more:**
- More kernel launch overhead relative to compute
- Sync overhead dominates total time
- Reducing sync frequency hides more overhead

**Why complex models benefit less:**
- GPU compute dominates total time
- Sync overhead is smaller fraction
- Kernel fusion reduces launch overhead

## Corrections Applied

1. `docs/PYTORCH_UPSTREAM_ISSUE.md` - Updated sync frequency table with model-specific results
2. `docs/USER_GUIDE.md` - Updated sync frequency section with accurate 2.4x figure for benchmark model

## Conclusion

The sync frequency finding is **directionally correct** - reducing sync frequency helps. But the magnitude varies significantly by model complexity:
- 5.0-5.2x for 1-layer models
- 2.4x for 3-layer benchmark model

Documentation now reflects these model-specific results.

**Hardware:** Apple M4 Max (40 GPU cores), macOS 15.7.3, PyTorch from pytorch-mps-fork
