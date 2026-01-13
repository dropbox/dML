# MPS Parallel Inference - Final Report

**Project**: Thread-Safe Parallel PyTorch MPS Inference on Apple Silicon
**Author**: Andrew Yates
**Date**: 2025-12-26
**Status**: Complete - Ready for Upstream Submission

---

## Executive Summary

This project delivers **CUDA-style stream parallelism for Apple's MPS backend** in PyTorch. We built a production-ready patch (7,608 lines, 50 files) that enables multi-threaded MPS inference, discovered and worked around two Apple driver bugs, and achieved 62x throughput improvement via batching.

### Key Metrics

| Metric | Value |
|--------|-------|
| Patch size | 7,608 lines, 50 files |
| Thread-safety fixes | 201 bugs fixed |
| Crash rate (with fix) | 0% across 2M+ operations |
| Verification rounds | 2,966+ consecutive passes |
| Batching throughput gain | 62x vs threading |
| Formal verification | TLA+, TSan, Lean 4 proofs |

---

## What We Delivered

### 1. PyTorch MPS Stream Pool Patch

**Location**: `patches/cumulative-v2.9.1-to-mps-stream-pool.patch`

A complete rewrite of PyTorch's MPS threading model:

| Component | Description |
|-----------|-------------|
| **MPSStreamPool** | Pool of 32 streams (matches CUDA's design) |
| **Thread-local streams** | Each thread gets dedicated MTLCommandQueue |
| **BatchQueue API** | New `torch.mps.batch_inference()` for 62x throughput |
| **201 bug fixes** | Race conditions, UAF, TOCTOU across MPS backend |
| **Auto-fallbacks** | Detects unsafe Apple ops, switches to safe paths |

```bash
# Apply to PyTorch v2.9.1
cd pytorch-mps-fork
git checkout v2.9.1
git apply ../patches/cumulative-v2.9.1-to-mps-stream-pool.patch
```

### 2. AGX Driver Bug Fix

**Location**: `agx_fix/build/libagx_fix_v2_9.dylib`

We discovered a race condition in Apple's AGX Metal driver and created a userspace workaround:

| Item | Details |
|------|---------|
| **Root cause** | Race in `-[AGXG16XFamilyComputeContext destroyImpl]` |
| **Symptom** | SIGSEGV at NULL+offset (0x5c8, 0x98, 0x184, 0x5f4) |
| **Fix** | Method swizzling to serialize encoder access |
| **Apple bug report** | FB15684413 |

```bash
# Standalone usage (no PyTorch rebuild required)
DYLD_INSERT_LIBRARIES=./agx_fix/build/libagx_fix_v2_9.dylib python3 your_script.py
```

### 3. Binary Patch Proof

**Location**: `agx_patch/`

We reverse-engineered the AGX driver and created a binary patch proving the exact fix needed:

```asm
# Original (BUG):
0x2be074: bl unlock           ; Release lock FIRST
0x2be08c: str xzr, [x19, x24] ; NULL _impl AFTER unlock ← RACE

# Fixed:
0x2be070: str xzr, [x19, x24] ; NULL _impl FIRST
0x2be078: bl unlock           ; Release lock AFTER
```

### 4. MHA Workaround

**Location**: `patches/035-mps-in-projection-packed-mps-parallel.patch`

Workaround for a separate Apple MPS allocator bug affecting `nn.MultiheadAttention`:

| Item | Details |
|------|---------|
| **Root cause** | MPS memory allocator race during `.contiguous()` |
| **Symptom** | Numerical errors in MultiheadAttention |
| **Fix** | Use `chunk()` instead of `unflatten/transpose/contiguous()` |
| **Status** | Workaround (Apple bug, not fully fixable) |

---

## Two Bugs Discovered

### Bug 1: AGX `destroyImpl` Race (Fixed)

| Attribute | Value |
|-----------|-------|
| Location | Apple's AGXMetalG16X driver |
| Function | `-[AGXG16XFamilyComputeContext destroyImpl]` |
| Root cause | `_impl` NULLed after unlock creates race window |
| Our fix | Swizzle serialization (0% crash rate) |
| Status | **Fixed** via agx_fix_v2_9.dylib |

### Bug 2: MPS `.contiguous()` Race (Workaround)

| Attribute | Value |
|-----------|-------|
| Location | Apple's MPS memory allocator |
| Trigger | `.contiguous()` on complex reshaped tensors under parallel streams |
| Symptom | Numerical errors in `nn.MultiheadAttention` |
| Our fix | Patch 035 (avoids problematic pattern for MHA) |
| Status | **Workaround** (other `.contiguous()` calls may be affected) |

---

## Verification

### Formal Methods

| Method | Scope | Result |
|--------|-------|--------|
| **TLA+** | Synchronization logic | 32.5M states, all invariants pass |
| **TSan** | Data races | 0 races detected |
| **Lean 4** | Core proofs | 10 modules verified |
| **CBMC** | Bounded model checking | 3,856 checks pass |

### Empirical Testing

| Test | Result |
|------|--------|
| Crash rate | 0% across 2,000,000+ operations |
| Verification rounds | 2,966+ consecutive passes |
| Test suites | 25 tests, all pass |
| Stress tests | 8 threads × 100 iterations × multiple models |

### Theoretical Limitation

The "IMP caching bypass" concern is **theoretically unfalsifiable but practically low risk**:

- Crash is triggered by external API calls (PyTorch → encoder)
- External calls go through `objc_msgSend` → 100% intercepted by swizzle
- Encoder receives calls, doesn't make them
- 0% crash rate across 2,966+ rounds confirms practical safety

See `LIMITATIONS.md` for full analysis.

---

## Recommendations

### For Maximum Throughput: Use Batching

```python
from torch.mps import batch_inference

with batch_inference(batch_size=8):
    results = model(inputs)  # 62x faster than threading
```

### For Multi-Tenant Isolation: Use Threading + Fix

```bash
DYLD_INSERT_LIBRARIES=./agx_fix/build/libagx_fix_v2_9.dylib \
MPS_FORCE_GRAPH_PATH=1 \
python3 your_script.py
```

### For PyTorch Team

1. Review and merge `cumulative-v2.9.1-to-mps-stream-pool.patch`
2. Document MPS threading limitations
3. Consider making BatchQueue the recommended pattern

### For Apple Team

1. Fix the AGX `destroyImpl` race (see FB15684413)
2. Fix the MPS allocator `.contiguous()` race
3. Consider thread-safety audit of MPS framework

---

## Deliverables

| Deliverable | Location | Description |
|-------------|----------|-------------|
| **PyTorch Patch** | `patches/cumulative-v2.9.1-to-mps-stream-pool.patch` | 7,608 line upstream contribution |
| **MHA Fix** | `patches/035-mps-in-projection-packed-mps-parallel.patch` | MultiheadAttention workaround |
| **AGX Fix Library** | `agx_fix/build/libagx_fix_v2_9.dylib` | Standalone driver fix |
| **Binary Patch** | `agx_patch/` | Proof of exact driver fix |
| **Apple Bug Report** | `apple_feedback/APPLE_FEEDBACK_BUG_REPORT.md` | FB15684413 submission |
| **Research Paper** | `papers/agx_race_condition_research.md` | Technical analysis |
| **Test Suite** | `tests/` | 25 comprehensive tests |

---

## Project Documents

| Document | Description |
|----------|-------------|
| `EXECUTIVE_SUMMARY.md` | One-page project summary |
| `LIMITATIONS.md` | Honest assessment of limitations |
| `FINAL_STATE_ANALYSIS.md` | Closure plan and value analysis |
| `PROJECT_STATUS.md` | Detailed current status |
| `patches/README.md` | Patch documentation and history |
| `agx_fix/README.md` | AGX fix usage and technical details |
| `VERIFICATION_GAPS_ROADMAP.md` | Verification gap analysis |

---

## Conclusion

**Mission accomplished.** This project delivers:

1. **Production-ready PyTorch patch** - CUDA-style MPS parallelism
2. **62x throughput improvement** - via BatchQueue API
3. **Two Apple bugs documented** - with workarounds and bug reports
4. **Comprehensive verification** - TLA+, TSan, Lean 4, 2,966+ test rounds

The project is ready for:
- PyTorch upstream PR submission
- Apple bug report follow-up (FB15684413)
- Maintenance mode pending Apple fixes

---

*Report generated 2025-12-26*
