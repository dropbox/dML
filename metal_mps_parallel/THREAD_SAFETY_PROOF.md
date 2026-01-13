# Thread Safety Analysis for MPS Parallel Inference

**Date**: 2025-12-23 (Updated 2025-12-25)
**Status**: EMPIRICALLY VERIFIED - 0% observed crashes under test conditions

---

## CRITICAL: Read LIMITATIONS.md First

**This document describes OBSERVED behavior, not PROVEN safety.**

The AGX fix has one fundamental limitation that CANNOT be resolved:

1. **IMP Caching Bypass (UNFALSIFIABLE)**: Objective-C method swizzling cannot guarantee all calls go through our swizzled methods. Call-site IMP caches may bypass our protection entirely. We have NO way to detect or prevent this.

Previously identified limitations have been **CLOSED**:

- ~~ARM64 Memory Model Gap~~ - **CLOSED** (N=3690): Litmus tests pass on hardware, code audit confirms all shared state within mutex protection.
- ~~Missing Encoder Factories~~ - **CLOSED** (N=3690): `parallelRenderCommandEncoderWithDescriptor:` is already swizzled in v2.9.

**All observed stability (0% crash rate) may be coincidental if critical code paths bypass our swizzle.**

See **[LIMITATIONS.md](LIMITATIONS.md)** for complete details.

---

## Important Caveats

- **"0% observed" ≠ "guaranteed"**: Empirical testing cannot prove absence of all crashes
- **TLA+ verification is model-limited**: The model does not capture IMP caching. (ARM64 memory ordering was validated via litmus tests at N=3690)
- **Test conditions may not cover all scenarios**: Production workloads may differ
- **Swizzle bypass is undetectable**: We cannot prove calls actually use our swizzled methods

See `VERIFICATION_GAPS_ROADMAP.md` for detailed analysis of verification gaps.

---

## Summary

This document demonstrates that the Python-level `_mps_lock` serialization significantly reduces crash probability during multi-threaded MPS inference. Under test conditions, 0% crashes were observed across 50 rounds.

## The Fix

```python
# In tests/complete_story_test_suite.py

# Global throttle for serializing MPS command submissions
# NOTE: Semaphore(1) is equivalent to a Lock.
_mps_throttle = threading.Semaphore(1)

def worker(tid):
    for i in range(iterations):
        # CRITICAL: Lock prevents AGX driver race
        with _mps_throttle:
            x = torch.randn(4, 32, 256, device=DEVICE)
            with torch.no_grad():
                y = models[tid](x)
            _ = y.sum().cpu()  # Safe sync without MPS Events
            _ = (x, y)  # Keep tensors alive until after sync
        completed[tid] += 1
```

## Why This Works

### Race Conditions Prevented

1. **AGX Driver Race** (Gap 1 from FORMAL_PROOF_GAPS_ANALYSIS)
   - Without lock: Thread A's `objc_msgSend` reads encoder's `isa` pointer while Thread B deallocates it → PAC trap CRASH
   - With lock: Only one thread can access Metal encoders at a time → NO RACE

2. **Tensor Lifetime Race** (Gap 3)
   - Without lock: Tensor `x` can be garbage collected while GPU still using it → SIGSEGV
   - With lock: `_ = (x, y)` keeps references alive until after the sync point completes → NO USE-AFTER-FREE

3. **Memory Ordering Race** (Gap 4)
   - Without lock: ARM64 store reordering can make refcount increment invisible to other threads
   - With lock: Lock acquire/release provides memory barriers → ORDERED VISIBILITY

### Formal Specification

The fix is formally specified in TLA+ at:
- `mps-verify/specs/PythonMPSLock.tla`
- `mps-verify/specs/PythonMPSLock_Safe.cfg` (with lock - expect NO violations)
- `mps-verify/specs/PythonMPSLock_Unsafe.cfg` (no lock - expect violations)

Key invariants proven:
- `NoUseAfterFree`: No tensor is freed while GPU is using it
- `NoDriverRace`: No concurrent AGX driver access
- `LockMutualExclusion`: Only one thread in critical section
- `GPUExclusiveAccess`: GPU operations are serialized

### TLC Model Checking Results (2025-12-24)

```
=== SAFE CONFIG (UsePythonLock = TRUE) ===
Result: ALL INVARIANTS HOLD
States: 3841 generated, 3328 distinct
Final state: thread_completed = <<3,3,3,3>> (all iterations complete)
driver_race_count = 0 (ALWAYS)
use_after_free_count = 0 (ALWAYS)

=== UNSAFE CONFIG (UsePythonLock = FALSE) ===
Result: INVARIANT VIOLATED
Violation: NoCrashes at state 6
Counterexample:
  State 5: Thread 1 and Thread 3 both in "running_model" phase
  State 6: Thread 1 runs RunModel_NoLock -> driver_race_count = 1
           (Two threads accessing GPU concurrently = CRASH)
```

**Conclusion**: Within the model's assumptions, TLC verified that the Python lock prevents the modeled race conditions. This is high confidence, not certainty (see caveats above).

### Why Python-Level Lock is Stronger

| Protection Level | What It Protects | Gaps Addressed |
|------------------|------------------|----------------|
| ObjC mutex (v2.3) | Individual encoder method calls | Gap 1 (partial) |
| AGX fix v2.5 | Encoder lifecycle + PAC traps | Gap 1 |
| MPS_FORCE_GRAPH_PATH | Uses thread-safe graph API | N/A |
| **Python _mps_lock** | **Entire GPU operation: create → infer → sync → cleanup** | **ALL (1,3,4)** |

The Python lock is the most comprehensive because it serializes the ENTIRE operation, not just individual calls.

## Empirical Verification

### Test Configuration
- 8 threads × 20 iterations per thread = 160 GPU operations per round
- TransformerBlock model (2-layer encoder, d_model=256)
- Environment: `MPS_FORCE_GRAPH_PATH=1 DYLD_INSERT_LIBRARIES=agx_fix/build/libagx_fix_v2_5.dylib`

### Results

| Configuration | Pass Rate | Notes |
|---------------|-----------|-------|
| No fix | ~5% | Frequent SIGSEGV |
| AGX v2.3 only | ~40% | PAC trap crashes |
| AGX v2.5 only | ~50% | Improved but still races |
| v2.5 + MPS_FORCE_GRAPH_PATH | ~60% | Better but tensor lifetime issues |
| **v2.5 + GRAPH_PATH + _mps_lock** | **100%** | **50/50 rounds passed** |

## How to Run Tests

```bash
# Use the crash-check wrapper (recommended)
./scripts/run_test_with_crash_check.sh python3 tests/complete_story_test_suite.py

# Run verification rounds
./scripts/run_verification_rounds.sh 50
```

## Trade-off: Safety vs Parallelism

The `_mps_lock` serializes GPU operations, which reduces parallelism. However:

1. **GPU is already the bottleneck**: Threading efficiency was only ~13% at 8 threads (see Chapter 2 of test suite)
2. **Batching is superior**: The test suite proves batching achieves higher throughput than threading
3. **Safety is mandatory**: Crashes are unacceptable in production

**Recommendation**: Use batching for throughput, threading for isolation (separate models).

## Conclusion

The combination of:
1. AGX fix v2.5 (encoder lifecycle protection)
2. MPS_FORCE_GRAPH_PATH=1 (thread-safe graph API)
3. Python `_mps_lock` (full operation serialization)
4. Tensor retention (`_ = (x, y)`)

Achieved **0% observed crashes** in multi-threaded MPS inference testing. This is modeled in TLA+ (with limitations) and empirically verified with 50 consecutive passing rounds under test conditions.

**Important**: "0% observed" is an empirical result, not a formal guarantee. See `VERIFICATION_GAPS_ROADMAP.md` for remaining verification gaps.
