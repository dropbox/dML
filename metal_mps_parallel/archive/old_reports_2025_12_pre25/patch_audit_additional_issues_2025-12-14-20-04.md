# Patch Audit — Additional Issues (Performance + Safety) — 2025-12-14 20:04

Point-in-time audit focused on **patch-specific** gaps in the current `pytorch-mps-fork` diff (parallel inference thread-safety work). This report only claims verifications that were actually run in this session.

## State Verified

- **Repo HEAD**: `6792497156a7b80aaaec3ddc9d9a1c873e8e0e05` (`# 311: Phase 32 cleanup - pointer safety, deprecated code removal`)
- **Fork HEAD** (`pytorch-mps-fork`): `06326711e2a0d5d369c72d6bd8b68e2170856113` (`Phase 32.13+32.16+32.17: Fix memory leak, RAII safety, env var validation`)
- **Cumulative patch**: `patches/cumulative-v2.9.1-to-mps-stream-pool.patch`
  - **MD5**: `b35aa9693a0df3ea0a7ed8af88bfb9fc`
  - Verified in-sync with fork diff: `./scripts/regenerate_cumulative_patch.sh --check` (PASS)

## Prior Items (Quick Re-Check)

- **Env var parsing validation is present** (warnings on invalid values): `pytorch-mps-fork/aten/src/ATen/mps/MPSStream.mm:38`.
- **MPSStreamGuard now checks pool-alive on destruction** (reduces teardown hazards): `pytorch-mps-fork/aten/src/ATen/mps/MPSStream.h:385`.
- **MetalShaderLibrary “sharded mutex” issue remains** (still a correctness/safety concern): `pytorch-mps-fork/aten/src/ATen/native/mps/MetalShaderLibrary.h:185`.

## Additional Patch-Specific Issues / Gaps

### 1) MetalShaderLibrary sharded locking is still data-racy (UB)

The code uses per-shard mutexes but protects a single shared `std::unordered_map` (`libMap` / `cplMap`). Concurrent access under different shard locks is still concurrent access to the same container, which is undefined behavior.

- Evidence: `pytorch-mps-fork/aten/src/ATen/native/mps/MetalShaderLibrary.h:185`
- Evidence (sharded locks around same map): `pytorch-mps-fork/aten/src/ATen/native/mps/OperationUtils.mm:851`

Fix direction: use **one mutex per map**, or actually **shard the maps** (e.g., `std::array<std::unordered_map<...>, kCacheShards>`), or move to a concurrent hashmap.

### 2) MPSEventPool still returns raw pointers for most APIs (UAF race on concurrent release)

`elapsedTime()` was fixed to take `shared_ptr` copies, but `recordEvent()`, `waitForEvent()`, `synchronizeEvent()`, and `queryEvent()` still call `getInUseEvent()` which returns a raw pointer and then use it outside the pool lock. Another thread calling `releaseEvent()` can erase the last `shared_ptr`, freeing the event while these calls are in flight.

- Evidence (raw pointer use): `pytorch-mps-fork/aten/src/ATen/mps/MPSEvent.mm:261`
- Evidence (erase/release): `pytorch-mps-fork/aten/src/ATen/mps/MPSEvent.mm:254`

Fix direction: mirror `elapsedTime()` and use `getInUseEventShared()` in all public MPSEventPool APIs.

### 3) MPSStream lock ordering / deadlock risk is inconsistent

`addCompletedHandler()` explicitly warns about deadlocks if `commandBuffer()` is called inside a `dispatch_sync` block while `_streamMutex` is held, but several other methods still:

1) lock `_streamMutex`, then
2) `dispatch_sync` to `_serialQueue`, and
3) call `commandBuffer()` / `synchronize()` inside the dispatched block (both take `_streamMutex`).

If `dispatch_sync` runs the block on a different thread (or the queue is busy), this can deadlock under contention, especially when other code paths `dispatch_sync` without first taking `_streamMutex`.

- Evidence (deadlock note + fixed pattern): `pytorch-mps-fork/aten/src/ATen/mps/MPSStream.mm:229`
- Evidence (still does it): `pytorch-mps-fork/aten/src/ATen/mps/MPSStream.mm:346`
- Evidence (commandBuffer locks `_streamMutex`): `pytorch-mps-fork/aten/src/ATen/mps/MPSStream.mm:123`

Fix direction: choose one consistent serialization primitive:
- either rely on `_serialQueue` and avoid taking `_streamMutex` across `dispatch_sync`, or
- capture needed objects under `_streamMutex` before `dispatch_sync` (like `addCompletedHandler()`), or
- move the `_streamMutex` lock *into* the dispatched block.

### 4) `endKernelCoalescing()` is not internally synchronized but is called without stream locks

`endKernelCoalescing()` manipulates `_commandEncoder` without locking. Multiple call sites invoke it without holding `_streamMutex` (e.g., MPSEvent record/wait, MPSHooks `getCommandBuffer()`).

- Evidence (no internal lock): `pytorch-mps-fork/aten/src/ATen/mps/MPSStream.mm:202`
- Evidence (called from MPSEvent): `pytorch-mps-fork/aten/src/ATen/mps/MPSEvent.mm:23`
- Evidence (called from MPSHooks): `pytorch-mps-fork/aten/src/ATen/mps/MPSHooks.mm:83`

Fix direction: add a `std::lock_guard<std::recursive_mutex>` inside `endKernelCoalescing()` (mutex is already recursive).

### 5) Allocator wait path can block while holding pool mutex (scalability + deadlock risk)

`MPSHeapAllocatorImpl::waitForEvents()` holds `pool.pool_mutex` while calling `event->synchronize()` (CPU blocking wait). This serializes allocator operations across threads and can amplify contention; it also risks deadlock if any callback or dependent path needs the pool lock while the wait is in progress.

- Evidence: `pytorch-mps-fork/aten/src/ATen/mps/MPSAllocator.mm:857`

Fix direction: avoid blocking under `pool_mutex` by holding a strong reference to the event outside the lock (requires event lifetime model changes), or restructure the ownership so the wait can safely occur after releasing the pool lock.

### 6) LayerNorm is globally serialized (scaling limitation versus project goals)

LayerNorm is guarded by a global mutex to prevent crashes, which will bottleneck multi-thread inference on LayerNorm-heavy models.

- Evidence (mutex): `pytorch-mps-fork/aten/src/ATen/native/mps/operations/Normalization.mm:27`
- Evidence (locking): `pytorch-mps-fork/aten/src/ATen/native/mps/operations/Normalization.mm:951`

Fix direction: find/implement a safe parallel LayerNorm path (e.g., MPSGraph fallback or safe Metal kernel path) rather than process-wide serialization.

### 7) Python API `release_current_thread_slot()` can block while holding the GIL

The Python binding calls into `releaseCurrentThreadSlot()` without releasing the GIL. The current C++ implementation synchronizes the stream with `COMMIT_AND_WAIT`, which can stall other Python threads and distort throughput measurements.

- Evidence (binding): `pytorch-mps-fork/torch/csrc/mps/Module.cpp:73`
- Evidence (sync in release): `pytorch-mps-fork/aten/src/ATen/mps/MPSStream.mm:684`

Fix direction: wrap the C++ call in `Py_BEGIN_ALLOW_THREADS` / `Py_END_ALLOW_THREADS` (or equivalent PyTorch helpers) around the potentially blocking synchronize.

