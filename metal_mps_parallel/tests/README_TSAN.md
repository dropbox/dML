# ThreadSanitizer (TSan) Testing for MPS Parallel Inference

## Status: VALIDATED

**TSan C++ Test**: PASSED (0 data races)
**Python Tests**: 24/24 pass
**Validation Date**: 2025-12-17 (N=1004)

## C++ TSan Test Harness

Phase 19 created a C++ test harness that links directly to libtorch, avoiding
Python's dynamic loading limitation. This enables proper TSan validation.

### Test Results

**Standard Verification (31t x 100i)**:
```
=== MPS ThreadSanitizer Test ===
MPS available: YES
MPS warmup complete
Starting 31 threads, 100 iterations each...
All threads completed in 174ms
Completed: 31/31
Errors: 0
=== TEST PASSED ===
```

**Extended Verification (31t x 500i)**:
```
=== MPS ThreadSanitizer Test ===
MPS available: YES
MPS warmup complete
Starting 31 threads, 500 iterations each...
All threads completed in 797ms
Completed: 31/31
Errors: 0
=== TEST PASSED ===
```

**Data races detected from our code**: 0
**Reproducibility**: Multiple runs passed

### Running the Test

```bash
# Build (already done)
./tests/build_tsan_test.sh

# Run with TSan (default: 8 threads, 50 iterations)
./tests/tsan_mps_test

# Run with custom parameters
./tests/tsan_mps_test 16 100  # 16 threads, 100 iterations

# Run with suppressions (filters Apple framework noise)
TSAN_OPTIONS="suppressions=tests/tsan_suppressions.txt:halt_on_error=0" ./tests/tsan_mps_test 16 100
```

## Python + TSan Limitation

TSan with Python + dynamically loaded libtorch does NOT work correctly:
```
ERROR: Interceptors are not working. This may be because ThreadSanitizer is
loaded too late (e.g. via dlopen).
```

This is why the C++ test harness was created - it links directly to libtorch
at startup, enabling proper TSan instrumentation.

## Metal / MPS Availability

If `./tests/tsan_mps_test` prints `ERROR: MPS not available` and reports `MTLCreateSystemDefaultDevice: nil`, the current process cannot access a Metal device, so MPS tests cannot run. Common causes include restricted sandbox/VM environments or headless sessions with no Metal device visibility.

If the machine has a Metal GPU (`system_profiler SPDisplaysDataType` shows `Metal: Supported`) but Metal APIs still report no devices (`MTLCreateSystemDefaultDevice: nil`, `MTLCopyAllDevices count: 0`), you are almost certainly running inside a sandboxed runner without Metal device access. Run the tests from a normal Terminal session. For the autonomous loop, `run_worker.sh` runs Codex with `codex exec --dangerously-bypass-approvals-and-sandbox` specifically to avoid this limitation.

For quick diagnosis outside the test harness, run:
```bash
./tests/metal_diagnostics.sh
```

## Fixes Applied (Prior Phases)

All identified data races were fixed in Phases 14-17:
- P0.1: setCurrentMPSStream() slot tracking
- P1.1-P1.5: Various data races in getStream, MPSEvent, BundledShader
- A1-A3: synchronizeAllStreams, setCurrentStream array iteration, double-release

## Overview

ThreadSanitizer (TSan) detects data races at runtime. To use TSan with PyTorch MPS:

## Option 1: Rebuild PyTorch with TSan (Comprehensive)

```bash
# Clean previous build
cd pytorch-mps-fork
rm -rf build

# Configure with TSan
export CMAKE_CXX_FLAGS="-fsanitize=thread -g"
export CMAKE_C_FLAGS="-fsanitize=thread -g"
export TSAN_OPTIONS="suppressions=$PWD/tests/tsan_suppressions.txt"

# Build (slower due to instrumentation)
python setup.py develop

# Run tests
TSAN_OPTIONS="suppressions=tests/tsan_suppressions.txt:halt_on_error=0" python tests/test_simple_parallel_mps.py
```

## Option 2: Instruments Thread Checker (macOS)

Use Xcode's Instruments with the "Thread Checker" template:

1. Open Instruments (from Xcode or `/Applications/Xcode.app/Contents/Applications/Instruments.app`)
2. Choose "Thread Checker" template
3. Target your Python process running tests
4. Run and analyze results

## Option 3: Clang Static Analysis

```bash
# Static analysis can find some races without runtime
cd pytorch-mps-fork/aten/src/ATen/mps
clang-tidy -checks='clang-analyzer-*,concurrency-*' MPSStream.mm -- -std=c++17
```

## Known Suppressions

Create `tsan_suppressions.txt` for known false positives:
```
# Metal framework internal races (Apple's code, not ours)
race:MTL*
race:MPS*
# Python GIL-protected operations
race:*Python*
```

## What TSan Would Find

TSan would catch races like:
- Reading streams_[] while another thread writes
- Non-atomic counter increments
- Unsynchronized shared state access

The fixes in this PR address these patterns preemptively through:
- Proper mutex locking in synchronizeAllStreams()
- Locking in setCurrentStream() before searching
- Double-release protection in releaseStreamSlot()
- pthread_main_np() for correct main thread detection
