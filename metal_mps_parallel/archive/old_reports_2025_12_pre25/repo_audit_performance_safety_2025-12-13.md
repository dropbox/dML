# Repo Audit: Performance & Safety Guidance (MPS Parallel Inference)

**Date**: 2025-12-13
**Updated**: 2025-12-16 (Phase 37 final verification + docs sync)
**Project goal**: Thread-safe, scalable parallel inference on PyTorch MPS via per-thread streams/queues.

This file consolidates audit findings and actionable guidance for workers.

---

## ✅ PROJECT STATUS: COMPLETE

**All goals achieved.** Correctness verified, performance goals met for production workloads.

| Goal | Target | Actual | Status |
|------|--------|--------|--------|
| Thread-safe inference | 8+ threads | Works | ✅ |
| Scaling efficiency | 50%+ | 53.4% at 4T (large workloads) | ✅ |
| Correctness | TSan clean | 0 races (31T x 100i) | ✅ |

**Note on efficiency metrics**: Small benchmark workloads (256x128, batch=1) show ~20% efficiency at 8T due to GPU saturation. Large workloads (2048+, batch=64+) achieve 53.4% efficiency at 4 threads (exceeds 50% target). This is correct behavior - the M4 GPU has fixed compute capacity.

See MPS_PARALLEL_INFERENCE_PLAN.md Phase 21 Part C for detailed scaling analysis.

---

## Verified Current State (Correctness Only)

- `./tests/run_all_tests.sh`: See `patches/README.md` for the last recorded local test run and verification notes.
- **Canonical cumulative patch**: `patches/cumulative-v2.9.1-to-mps-stream-pool.patch` (see `patches/README.md` for current MD5 / fork HEAD / file count)
- **Critical discovery (N=34)** (editable install importing baseline torch) is fixed and guarded against regression by the test runner preflight.
- **Known platform constraints (Apple framework)**: Some MPS/Metal kernels are not safe for concurrent encoding at higher thread counts (notably `nn.Linear` no-graph path and some compute-kernel ops). This patch mitigates these issues (auto graph path + targeted serialization), but for highest concurrency or LayerNorm-heavy models consider multi-process parallelism (see `tests/multiprocess_inference_pool.py`).

---

## Issue Audit (All Previously Tracked Items)

| Issue | Status | Evidence / Fix Location |
|---|---|---|
| Wrong torch import (baseline vs fork) | ✅ FIXED | `tests/run_all_tests.sh` preflight prints `torch.__version__`, `torch.__file__`, and fork HEAD, and fails if they differ. |
| Metal command-buffer assertions under concurrency | ✅ FIXED | Per-stream `std::recursive_mutex` around command buffer/encoder state + commitAndContinue enabled only for default stream (disabled for workers) + synchronize before recycling stream slots (`pytorch-mps-fork/aten/src/ATen/mps/MPSStream.mm`). |
| `torch.mps.synchronize()` must sync all streams | ✅ FIXED | `pytorch-mps-fork/aten/src/ATen/mps/MPSHooks.mm` uses `MPSStreamPool::instance().synchronizeAllStreams()`. |
| `MPSHooks` must use current stream (command buffer/queue) | ✅ FIXED | `pytorch-mps-fork/aten/src/ATen/mps/MPSHooks.mm` uses `getCurrentMPSStream()` for hooks. |
| `MetalShaderLibrary::compileLibrary()` race | ✅ FIXED | `pytorch-mps-fork/aten/src/ATen/native/mps/OperationUtils.mm` uses local `lib` (no shared member write), and `getLibrary()` uses `std::call_once`. |
| Bundled shader library init race | ✅ FIXED | `pytorch-mps-fork/aten/src/ATen/native/mps/OperationUtils.mm` uses `std::call_once` in `BundledShaderLibary::getLibrary()`. |
| `streams_` data race / DCL | ✅ FIXED | `pytorch-mps-fork/aten/src/ATen/mps/MPSStream.mm` locks `stream_creation_mutex_` around `streams_` access. |
| `getStream()` OOB fallback hides bugs | ✅ FIXED | `pytorch-mps-fork/aten/src/ATen/mps/MPSStream.mm` uses `TORCH_CHECK(index < kMPSStreamsPerPool, ...)`. |
| `MPSEventPool` `nullptr` stream fallback | ✅ FIXED | `pytorch-mps-fork/aten/src/ATen/mps/MPSEvent.mm` uses `getCurrentMPSStream()` when `stream == nullptr`. |
| “First thread is main” detection | ✅ FIXED | `pytorch-mps-fork/aten/src/ATen/mps/MPSStream.mm` uses `pthread_main_np() == 1`. |
| `setCurrentMPSStream()` slot tracking correctness | ✅ FIXED | `pytorch-mps-fork/aten/src/ATen/mps/MPSStream.mm` updates TLS slot index and releases old slots safely. |
| Profiler runtime thread safety | ✅ DOCUMENTED | `pytorch-mps-fork/aten/src/ATen/mps/MPSProfiler.h` contains a thread-safety warning and explicit guidance to disable profiling in multi-thread inference. |
| Cross-stream tensor correctness coverage | ✅ COVERED | `tests/test_cross_stream_tensor.py` documents safe usage (sync before cross-thread use) and validates the supported pattern. |
| `dispatch_sync` + TLS hazard | ✅ FIXED | Phase 20: MetalKernelFunction::startEncoding() now uses captured stream instead of TLS lookup inside dispatch block. All other usages already captured stream before dispatch_sync. |
| `synchronizeStream/queryStream` API support | ⚠️ DOCUMENTED LIMITATION | MPS streams are internal; use `torch.mps.synchronize()` for device-wide sync and events for ordering. (No public per-stream Python API.) |

---

## ⚠️ Phase 21: Pending Issues

### Safety Issues (Fixed)

| # | Severity | Issue | Status |
|---|----------|-------|--------|
| 21.1 | HIGH | Exception safety in `runCommandBlock()` | ✅ FIXED (N=110) |
| 21.2 | MEDIUM | Thread-safety docs for `MetalKernelFunction` | ✅ FIXED (N=110) |
| 21.3 | HIGH | MTLLibrary leak on compile race | ✅ FIXED (N=110) |
| 21.4 | HIGH | Command-buffer leak when commitAndContinue disabled | ✅ FIXED (N=110) |
| 21.5 | CRITICAL | Lock-order inversion (deadlock risk) | ✅ FIXED (N=110) |
| 21.11 | HIGH | `dispatch_sync_with_rethrow()` deadlock docs | ✅ FIXED (N=110) |
| 21.12 | HIGH | `getNewStream()` slot lifecycle docs | ✅ FIXED (N=110) |

### Performance Optimizations

| # | Impact | Issue | Status |
|---|--------|-------|--------|
| 21.6 | **CRITICAL** | `g_mpsgraph_encode_mutex` serializes ALL graph encoding | ✅ FIXED (N=109) |
| 21.8 | HIGH | Unnecessary dispatch_sync in thread-local caches | ✅ FIXED (N=111) |
| 21.7 | **CRITICAL** | `s_linear_nograph_mutex` serializes non-graph linear | ✅ FIXED (N=117) - auto-detect solution |
| 21.14 | HIGH | Singleton `MPSAllocator` mutex contention | DEFERRED (no perf impact observed) |
| 21.9 | MEDIUM | std::function overhead in hot paths | DEFERRED (minor optimization) |
| 21.10 | LOW | Mutex overhead in commandBuffer()/commandEncoder() | DEFERRED (minor optimization) |

**Performance after N=117**:
- Large workloads (2048+ hidden, batch 64+): 53.4% efficiency at 4 threads ✅ TARGET MET
- Small workloads: Lower efficiency expected due to GPU saturation (correct behavior)

### Correctness Issues from External Review (21.15-21.20)

| # | Severity | Issue | Status |
|---|----------|-------|--------|
| 21.15 | ~~CRITICAL~~ | `synchronizeDevice()` only syncs current stream, not all | ✅ FIXED (N=113) |
| 21.16 | HIGH | Guard `record()`/`block()` uses Stream argument | ✅ VERIFIED (N=117) |
| 21.17 | ~~CRITICAL~~ | `setCurrentMPSStream()` doesn't acquire freelist slot | ✅ FIXED (N=113) |
| 21.18 | HIGH | `elapsedTime()` checks timing before wait | ✅ VERIFIED (N=117) |
| 21.19 | ~~MEDIUM~~ | `~MPSStream()` leaks `_prevCommandBuffer` | ✅ FIXED (N=113) |
| 21.20 | LOW | `getStreamFromPool()` docs updated with lifecycle warning | ✅ VERIFIED (N=117) |

**All issues verified resolved.** See `WORKER_DIRECTIVE.md` for detailed status.

---

## Notes for Production Use

1. **Best throughput for models**: For highest concurrency (or LayerNorm-heavy models), prefer **multi-process** parallelism to avoid remaining Apple framework in-process concurrency limits and to bypass LayerNorm serialization.
2. **In-process scaling**: Raw tensor ops (e.g., `torch.mm`) scale to 8+ threads in-process (bounded by GPU saturation). `nn.Module` scaling depends on model ops and Apple framework behavior.
3. **Disable MPS profiling in multi-thread inference**: See the warning in `pytorch-mps-fork/aten/src/ATen/mps/MPSProfiler.h`.

---

## Test Coverage (Local)

**Primary suite (gated by preflight)**: `./tests/run_all_tests.sh` (12 tests; see `patches/README.md` for last recorded run).

**Additional checks**:
- `tests/test_cross_stream_tensor.py`: Cross-thread tensor usage rules + smoke test (included in primary suite as of N=119)
- `tests/test_tsan_basic.py`: Minimal concurrency test intended for TSAN-style validation
