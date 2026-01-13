# Phase 1 Progress Report: Apple Feedback Package

**Worker**: N=1468
**Date**: 2025-12-21
**Tasks Completed**: Task 1.1, Task 1.3

---

## Summary

Completed Phase 1 Tasks 1.1 and 1.3. Created minimal Metal reproductions and prepared the Apple Feedback package for submission.

---

## Task 1.1: Minimal Metal Reproduction

### Files Created

1. `tests/metal_race_repro.mm` - Pure Metal compute version (220 lines)
   - 8 threads, 100 ops/thread, 1000 iterations
   - Uses simple copy/add kernels
   - Compiled: `clang++ -std=c++17 -framework Metal -framework Foundation`

2. `tests/metal_race_repro_mps.mm` - MetalPerformanceShaders version
   - Uses MPSMatrixMultiplication
   - Shared command queue (mimics PyTorch pattern)
   - Includes synchronizer thread

### Test Results

**Pure Metal Version**:
- Completed 200+ iterations without crash
- "Context leak detected" warnings observed (~every 6 iterations)
- Resource exhaustion at ~250 iterations (command queue creation failures)
- Throughput: ~114K ops/s

**MPS Version**:
- Completed 100 iterations without crash
- "Context leak detected" warning at iteration 100
- Throughput: ~11K ops/s (matrix multiply bound)

### Key Finding

The crash requires PyTorch's specific dispatch pattern to reliably trigger. The minimal reproductions demonstrate driver instability (via "Context leak detected" warnings) but the race window is too narrow to consistently trigger crashes without PyTorch's exact timing.

The documented 55% crash rate was observed specifically with PyTorch MPS operations, not pure Metal.

---

## Task 1.3: Apple Feedback Package

### Package Structure

```
apple_feedback/
├── FEEDBACK_SUBMISSION.md     # Main bug report (comprehensive)
├── README.md                  # Build/run instructions
├── crash_reports/
│   ├── CRASH_ANALYSIS_2025-12-20_173618.md
│   └── CRASH_ANALYSIS_2025-12-20_174241.md
├── tla_proofs/
│   ├── AGXContextRace.tla     # Buggy driver model
│   └── AGXContextFixed.tla    # Fixed model with mutex
└── reproduction/
    ├── metal_race_repro.mm    # Pure Metal version
    └── metal_race_repro_mps.mm # MPS version
```

### Feedback Submission Content

The FEEDBACK_SUBMISSION.md includes:
- Executive summary of the bug
- All 3 crash sites with offsets and descriptions
- Register state showing NULL pointer (x20=0x0, far=0x5c8)
- Reproduction steps (PyTorch test + standalone Metal tests)
- Crash rate table (0% at 1 thread → 55% at 8 threads)
- Workaround description (global mutex)
- Formal verification summary (TLA+)
- System configuration details
- Suggested fix (per-context mutex or reference counting)

---

## Task 1.2: Mutex Overhead (Pending)

Not started. This task requires:
1. Running benchmark with mutex disabled (will crash)
2. Measuring ops/s just before crash
3. Comparing with mutex enabled throughput
4. Statistical analysis with confidence intervals

---

## Files Changed

| File | Status | Description |
|------|--------|-------------|
| tests/metal_race_repro.mm | Created | Pure Metal reproduction |
| tests/metal_race_repro_mps.mm | Created | MPS reproduction |
| apple_feedback/FEEDBACK_SUBMISSION.md | Created | Bug report |
| apple_feedback/README.md | Created | Package instructions |
| apple_feedback/reproduction/* | Copied | Reproduction code |
| apple_feedback/crash_reports/* | Copied | Crash analyses |
| apple_feedback/tla_proofs/* | Copied | TLA+ specs |
| AGX_RESEARCH_ROADMAP.md | Updated | Marked tasks complete |

---

## Recommendations for Next Worker

1. **Task 1.2**: Quantify mutex overhead with confidence intervals
   - May need to run PyTorch benchmark with `MPS_DISABLE_ENCODING_MUTEX=1`
   - Collect timing data before crashes occur
   - Compare with protected runs

2. **Consider**: The Apple Feedback package is ready for submission via Feedback Assistant

3. **Note**: The "Context leak detected" message in the minimal reproductions is a related driver issue that could be included as additional evidence
