# Crash Demonstration Tests

This directory contains tests that **intentionally demonstrate crashes** in
Apple's AGX Metal driver. These tests are for research and documentation
purposes only.

## DO NOT RUN IN NORMAL TEST SUITES

These tests are designed to crash. They should NEVER be run as part of
automated testing or CI/CD pipelines.

## Purpose

These tests document the AGX driver race condition described in:
- `AGX_RESEARCH_ROADMAP.md`
- `apple_feedback/FEEDBACK_SUBMISSION.md`
- `reports/crash_reports/`

## Tests

### test_shutdown_crash.py

Reproduces the shutdown crash that occurs when multi-threaded MPS code exits
without the AGX fix loaded. With `MPS_DISABLE_ENCODING_MUTEX=1`, this crashes
approximately 55% of the time.

**Expected crash:**
```
Exception Type: EXC_BAD_ACCESS (SIGSEGV)
Location: -[AGXG16XFamilyComputeContext setComputePipelineState:] + 32
Fault Address: 0x5c8
```

**To reproduce the crash:**
```bash
MPS_DISABLE_ENCODING_MUTEX=1 python3 tests/crash_demos/test_shutdown_crash.py
```

**To run safely (with AGX fix):**
```bash
./scripts/run_mps_test.sh tests/crash_demos/test_shutdown_crash.py
```

## Background

The AGX driver has a race condition where concurrent encoding operations can
access a context that has been invalidated by another thread. This causes NULL
pointer dereferences at three known locations in the driver:

1. `setComputePipelineState:` + 0x5c8
2. `prepareForEnqueue` + 0x98
3. `allocateUSCSpillBuffer` + 0x184

Our `libagx_fix.dylib` prevents these crashes by serializing access to the
encoding operations via Objective-C method swizzling.

## See Also

- `agx_fix/` - The fix implementation
- `scripts/run_mps_test.sh` - Wrapper script that loads the fix
- `apple_feedback/` - Apple Feedback Assistant submission package
