# Assumption Falsification Report

**Date:** 2025-12-20T05:17:16.054919
**PyTorch Version:** 2.9.1a0+git4201c80
**MPS Available:** True
**Device:** mps

## Summary

Tested 3 assumptions
Bugs proven: 1
Bugs not reproduced: 2

Proven Apple MPS bugs requiring workarounds:
  - contiguous_race

Assumptions not proven (may be intermittent):
  - sdpa_parallel_race
  - batch_serialization_needed

## Test Results

### contiguous_race

**Assumption:** Apple MPS has a race condition in memory copy operations triggered by .contiguous() on tensors with complex stride patterns when called from multiple threads simultaneously.

**Workaround:** Use .reshape() which can handle non-contiguous tensors directly, avoiding the internal .contiguous() call race condition.

**Test Configuration:**
- Mode: unprotected (use .contiguous())
- Iterations: 30
- Threads: 8
- Duration: 1134.8ms

**Results:**
- Passed: 28
- Failed: 2
- Crashes: 0
- Max Diff: 1.02e+02

**Verdict: FAIL**

**Evidence:** Race detected: 2/30 iterations failed, max_diff=1.02e+02

---

### sdpa_parallel_race

**Assumption:** Apple MPS's scaled_dot_product_attention has internal shared state that causes data races when called from multiple threads without serialization.

**Workaround:** Use MPSBatchQueue with num_workers=1 to serialize GPU access, avoiding the internal race condition in Apple's SDPA implementation.

**Test Configuration:**
- Mode: unprotected (parallel SDPA)
- Iterations: 20
- Threads: 8
- Duration: 489.7ms

**Results:**
- Passed: 20
- Failed: 0
- Crashes: 0
- Max Diff: 0.00e+00

**Verdict: PASS**

**Evidence:** No race detected in 20 iterations (bug may be intermittent)

---

### batch_serialization_needed

**Assumption:** Running 8+ threads with direct MPS inference (no serialization) causes more race conditions than with serialization.

**Workaround:** MPSBatchQueue with num_workers=1 serializes GPU access, achieving better correctness at high thread counts.

**Test Configuration:**
- Mode: comparison (protected vs unprotected)
- Iterations: 10
- Threads: 8
- Duration: 272.3ms

**Results:**
- Passed: 8
- Failed: 2
- Crashes: 0
- Max Diff: 8.78e+01

**Verdict: PASS**

**Evidence:** No difference: protected=8/10, unprotected=8/10

---
