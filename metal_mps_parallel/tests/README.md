# MPS Stream Pool - Test Suite

**Created by Andrew Yates**

## Overview

This directory contains tests for the MPS Stream Pool parallel inference implementation.

## Running Tests

```bash
# Quick check: verify Metal devices are visible to this process
./tests/metal_diagnostics.sh

# Scriptable check: exit 0 if Metal visible (1 = not visible, 2 = unknown)
./tests/metal_diagnostics.sh --check

# IMPORTANT: Always run MPS tests with the crash-check wrapper (fails if new
# crash logs appear in crash_logs/, even if Python exits 0).
./scripts/run_test_with_crash_check.sh python3 tests/test_stress_extended.py

# Bare Metal reproduction (no ML frameworks; may crash if Metal/AGX bug reproduces)
./tests/run_metal_bare_thread_race.sh --compile-only
./tests/run_metal_bare_thread_race.sh

# Run full test suite (25 tests; fails if new crash logs appear in crash_logs/)
./tests/run_all_tests.sh

# Run TSan thread-safety test (requires build)
./tests/build_tsan_test.sh
./tests/tsan_mps_test --threads=31 --iterations=100

# 8-thread correctness via batching (requires torch.mps.BatchQueue from the patch)
./scripts/run_test_with_crash_check.sh python3 tests/test_batch_inference.py
./scripts/run_test_with_crash_check.sh python3 tests/correctness_benchmark.py --parallel --use-batching --threads 8 --workers 1
./scripts/run_test_with_crash_check.sh python3 tests/transformer_direct_vs_batched.py --workers 1
```

NOTE: This directory contains many long-running standalone scripts named `test_*.py`. Avoid running `pytest tests/` unless you intend to run the full soak/stress set; prefer `./tests/run_all_tests.sh` or run a specific pytest file (e.g. `pytest tests/test_mps_parallel_story.py ...`).

If `metal_diagnostics.sh` reports `MTLCreateSystemDefaultDevice: nil`, Metal devices are not visible to this process (common under sandboxed/VM/headless runners).

## Test Suite (25 tests in run_all_tests.sh)

| Test | File | Description |
|------|------|-------------|
| 1. Fork Safety | `test_fork_safety.py` | Verifies forked child is blocked from using MPS |
| 2. Simple Parallel MPS | `test_parallel_mps_simple.py` | Basic parallel MPS operations (2-4 threads) |
| 3. Extended Stress Test | `test_stress_extended.py` | High iteration stress test |
| 4. Thread Boundary | `test_thread_boundary.py` | Stream assignment at thread boundaries |
| 5. Stream Assignment | `test_stream_assignment.py` | TLS stream assignment validation |
| 6. Benchmark (nn.Linear) | `benchmark_parallel_mps.py` | Performance benchmark with nn.Linear |
| 7. Real Models Parallel | `test_real_models_parallel.py` | nn.Sequential, MLP, Conv1D models |
| 8. Stream Pool Wraparound | `test_oversubscription.py` | Verifies >31 sequential threads do not fail |
| 9. Thread Churn | `test_thread_churn.py` | Stability under thread churn |
| 10. Cross-Stream Tensor | `test_cross_stream_tensor.py` | Cross-stream tensor operations |
| 11. Linalg Ops Parallel | `test_linalg_ops_parallel.py` | Phase 23 mutex protection validation |
| 12. Large Workload Efficiency | `test_efficiency_large_workload.py` | Validates 50%+ efficiency at 2 threads |
| 13. Max Streams Stress | `test_max_streams_stress.py` | 31 threads x 100 iterations stress test (Phase 39.1) |
| 14. OOM Recovery Parallel | `test_oom_recovery_parallel.py` | Memory pressure and OOM handling (Phase 39.2) |
| 15. Graph Compilation Stress | `test_graph_compilation_stress.py` | Concurrent graph compilation races (Phase 39.3) |
| 16. Stream Pool: Round Robin | `test_stream_pool_boundaries.py` | Round-robin wraparound validation |
| 17. Stream Pool: Reuse Churn | `test_stream_pool_boundaries.py` | Stream reuse under thread churn |
| 18. Stream Pool: Sync Active | `test_stream_pool_boundaries.py` | synchronize_all during active use |
| 19. Static: MPS Cleanup | `test_static_destruction.py` | MPS state during interpreter cleanup |
| 20. Static: Atexit | `test_static_destruction.py` | atexit cleanup order |
| 21. Static: Thread Cleanup | `test_static_destruction.py` | Thread pool cleanup order |
| 22. Fork: Active Threads | `test_fork_safety_stress.py` | Fork with active worker threads |
| 23. Fork: Parent Continues | `test_fork_safety_stress.py` | Parent process continues after fork |
| 24. Fork: MP Spawn | `test_fork_safety_stress.py` | multiprocessing.spawn compatibility |
| 25. LayerNorm Verification | `verify_layernorm_fix.py` | LayerNorm correctness + thread-consistency regression |

## TSan Tests

| File | Description |
|------|-------------|
| `tsan_mps_test.mm` | Objective-C++ TSan test for thread-safety |
| `build_tsan_test.sh` | Build script for TSan test |
| `README_TSAN.md` | TSan test documentation |
| `test_tsan_basic.py` | Basic Python TSan smoke test |

## record_stream Tests

| File | Description |
|------|-------------|
| `test_record_stream.mm` | Objective-C++ test for record_stream semantics (External Audit Gap #4) |
| `build_record_stream_test.sh` | Build script for record_stream test |

Run with: `./build_record_stream_test.sh && ./record_stream_test`

## Experimental/Investigation Files

These files are not in the main test suite but may be useful for debugging:

| File | Description |
|------|-------------|
| `test_parallel_mps.py` | Original parallel test (superseded by test_parallel_mps_simple.py) |
| `test_nnlinear_threads.py` | nn.Linear thread-safety investigation |
| `test_multiprocess_vs_multithread.py` | Process vs thread comparison |
| `multiprocess_inference_pool.py` | Multiprocess inference pool experiment |
| `test_thread_limit_investigation.py` | Thread limit debugging |
| `repro_transformer_block_race.py` | Repro: TransformerEncoderLayer output corruption under parallel MPS streams (includes a CPU-barrier mitigation flag) |
| `repro_sdpa_strided_race.py` | SDPA strided-layout harness + stress test (useful for inspecting MHA Q/K/V strides) |
| `repro_dispatch_sync_with_rethrow_reentrancy.mm` | Repro for nested dispatch_sync deadlock + queue-specific inline fix |

## Phase 3 Investigation Files (N=1270-1273, N=1333-1338)

Root cause analysis of MPS race condition in `nn.MultiheadAttention`. See `reports/main/apple_mps_bug_investigation_N1270.md` for initial findings and `reports/main/view_mm_fix_verification_N1337.md` + `reports/main/projection_sdpa_device_sync_barrier_N1338.md` for final conclusions.

| File | Description |
|------|-------------|
| `isolate_mps_operation.py` | Tests individual MPS operations in parallel (matmul, softmax, linear, etc.) |
| `isolate_sdpa_components.py` | SDPA sub-component isolation (Q@K.T, softmax, scale, etc.) |
| `isolate_mha_internals.py` | MHA internal operations (projections, reshapes, output) |
| `isolate_mha_functional.py` | PyTorch's `F.multi_head_attention_forward` vs manual implementation |
| `isolate_projection_pattern.py` | PyTorch's `_in_projection_packed` pattern analysis |
| `isolate_contiguous.py` | `.contiguous()` bug isolation and verification |
| `isolate_transformer_ffn.py` | TransformerBlock component isolation (FFN, LayerNorm, MHA) |
| `minimal_mps_contiguous_race.py` | Minimal standalone reproduction suitable for PyTorch bug report |

### Phase 3 Conclusions (N=1337-1338)

**Key Finding**: Pure `.contiguous()` is now thread-safe after the View.mm `MPSEncodingLock` fix (N=1336). The remaining race is **NOT in the individual operations** but in **inter-operation timing** when running SDPA immediately after projection-copy across multiple MPS command queues.

**Evidence**:
- Pure `.contiguous()` operations: 250/250 pass (100%)
- Pure F.linear: 50/50 pass (100%)
- Pure SDPA: 30/30 pass (100%)
- Combined projection + SDPA without barrier: ~70-90% fail
- Combined projection + SDPA **with device sync barrier**: 30/30 pass (100%)

**Root Cause**: This is an Apple MPS framework limitation in multi-stream command queue coordination, not a bug in PyTorch's tensor operations. When SDPA executes immediately after projection-copy on different command queues, timing races cause output corruption.

**Production Solution**: Use BatchQueue with `num_workers=1` to serialize all MPS GPU access for correctness. For throughput, prefer batching/dynamic batching (see `python3 tests/complete_story_test_suite.py`).

## Performance Benchmarks

| File | Description |
|------|-------------|
| `test_graph_path_benchmark.py` | MPS_FORCE_GRAPH_PATH=1 vs default benchmark (3-6% improvement, N=3683) |
| `test_memory_pool.py` | Memory pooling benchmark (35% improvement with inference, N=3682) |
| `test_production_metrics.py` | Memory growth and P99 latency measurement (N=3684) |

## Environment Variables

- `MPS_FORCE_GRAPH_PATH=1`: Force MPSGraph path - provides 3-6% performance improvement (thread-safe)
- `MPS_TESTS_ALLOW_TORCH_MISMATCH=1`: Allow tests with stale torch build
- `MPS_TEST_SYNC_AFTER_PROJECTION=1`: Debug-only; inserts a device-wide `torch.mps.synchronize()` between projection-copy (`.contiguous()`/`.clone()`) and SDPA in select repro scripts

## Known Issues

### Intermittent SIGSEGV on Static Thread Cleanup (Test 21)

Test 21 "Static: Thread Cleanup" may occasionally fail with SIGSEGV (exit code 139) during Python interpreter shutdown. This is a **known limitation**, not a bug in our code.

**Behavior:**
- Test logic itself passes (all threads complete, no errors)
- SIGSEGV occurs during Python module cleanup after test completion
- Happens ~20-40% of the time, intermittent

**Root Cause:**
- Race condition between Python's module cleanup and MPS static destruction
- Static destruction order between Python C extensions is undefined
- When Python unloads torch module, MPS cleanup may race with other cleanup

**Mitigation:**
- Test has retry logic (max_retries=2 in run_all_tests.sh)
- Usually passes on first or second attempt
- Does not indicate a problem with parallel inference functionality

**Impact:**
- None on production code - this only affects Python interpreter shutdown
- Tests using MPS in subprocess (like the fork tests) are unaffected

## Requirements

- macOS with Metal device access (not in sandbox/VM)
- PyTorch built from `pytorch-mps-fork/` with stream pool patch applied
- Python 3.x with torch installed
