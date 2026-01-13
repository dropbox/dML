# Additional Patch Gaps / Risks (MPS Parallel Inference)

**Date**: 2025-12-13 20:19 (local)

This is an addendum audit focused on remaining correctness/performance risks in the current MPS stream-pool patch.

## State Verified (Local)

- Fork HEAD: `pytorch-mps-fork` @ `03cce0bc2e73ecf5a511ec52c5fc38ae99c8ae1e`
- Cumulative patch matches fork diff: `sha256(a64534d51a3cb31fea94ab8c41af5a5a8cdd81da209331deda6793fa85ef4a7a)`
  - Verified by hashing `git -C pytorch-mps-fork diff v2.9.1..HEAD` and `patches/cumulative-v2.9.1-to-mps-stream-pool.patch`.

## Findings (New / Not Previously Tracked)

### 1) `synchronizeDevice()` is not device-wide (semantic mismatch)

- **Location**: `pytorch-mps-fork/aten/src/ATen/mps/MPSGuardImpl.mm:65`
- **Problem**: `MPSGuardImpl::synchronizeDevice()` only synchronizes `getCurrentMPSStream()`, while `MPSHooks::deviceSynchronize()` correctly synchronizes *all* streams.
- **Impact**: Callers using the accelerator/device-guard path can observe “device synchronize” returning while work on other MPS streams is still running.
- **Suggested fix**: Use `at::mps::MPSStreamPool::instance().synchronizeAllStreams()` in `MPSGuardImpl::synchronizeDevice()` (matching `MPSHooks::deviceSynchronize()`).

### 2) `MPSGuardImpl::{record,block}()` ignore the passed `Stream` argument

- **Locations**:
  - `pytorch-mps-fork/aten/src/ATen/mps/MPSGuardImpl.mm:18`
  - `pytorch-mps-fork/aten/src/ATen/mps/MPSEvent.mm:161`
- **Problem**:
  - `MPSGuardImpl::record()`/`block()` receive a `const Stream& stream` but never use it to select where the event is recorded/waited.
  - Event acquisition defaults to `getCurrentMPSStream()` when `stream == nullptr` (`MPSEventPool::acquireEvent`), so events become implicitly tied to TLS “current stream”, not necessarily the `Stream` argument passed into the guard API.
  - The temporary `MPSStream mps_stream{stream};` currently has no effect on which stream is used.
- **Impact**: If the guard/event APIs are called with a non-current stream (or across threads), event ordering and synchronization semantics can be incorrect.
- **Suggested fix**:
  - Map the incoming `Stream` to the owning `MPSStream*` (via `MPSStreamPool`) and plumb it into `MPSEventPool` so events are bound to the intended stream.
  - Remove or repurpose the current unused `MPSStream mps_stream{stream};`.

### 3) `setCurrentMPSStream()` does not acquire the target freelist slot (pool ownership corruption risk)

- **Location**: `pytorch-mps-fork/aten/src/ATen/mps/MPSStream.mm:496`
- **Problem**: `MPSStreamPool::setCurrentStream()` updates TLS `slot_index` but does **not** ensure the new slot is removed from `free_slots_`, nor does it check whether another thread currently “owns” it.
- **Impact**:
  - A thread can set its current stream to a worker slot that is still present in the freelist, allowing another thread to acquire and reuse the same slot concurrently.
  - On thread-exit, TLS teardown can “release” a slot that was never acquired, potentially reintroducing a slot into the freelist while another thread is using it.
- **Suggested fix**:
  - Treat worker stream selection as an ownership transfer: acquire the new slot (or fail if not available) before setting TLS, or add explicit ownership tracking per slot.
  - Consider removing the raw `MPSStream*` setter from public API in favor of an RAII stream-handle that models slot ownership.

### 4) `elapsedTime()` can hang if events were not created with `enable_timing=True`

- **Location**: `pytorch-mps-fork/aten/src/ATen/mps/MPSEvent.mm:223`
- **Problem**: `MPSEventPool::elapsedTime()` calls `end_event->waitForCpuSync()` before checking `completion_time != 0`.
- **Impact**: If a user calls `elapsedTime()` on events that were not timing-enabled, this can block indefinitely instead of throwing a clear error.
- **Suggested fix**: Check timing-enabled / completion-time readiness before waiting, or gate the wait on `enable_timing`.

### 5) `MPSStream` destructor does not release `_prevCommandBuffer` (retained buffer lifetime)

- **Locations**:
  - `pytorch-mps-fork/aten/src/ATen/mps/MPSStream.mm:55`
  - `pytorch-mps-fork/aten/src/ATen/mps/MPSStream.mm:157`
- **Problem**: When commit-and-continue is disabled, `flush()` can retain `_prevCommandBuffer`; `~MPSStream()` asserts `_commandBuffer == nil` but does not release `_prevCommandBuffer`.
- **Impact**: Each stream can retain a command buffer indefinitely (and whatever resources it retains) until process exit.
- **Suggested fix**: Release `_prevCommandBuffer` (and ensure encoders are ended) in `~MPSStream()`.

### 6) `getStreamFromPool()` docs over-promise slot recycling

- **Locations**:
  - `pytorch-mps-fork/aten/src/ATen/mps/MPSStream.h:168`
  - `pytorch-mps-fork/aten/src/ATen/mps/MPSStream.mm:550`
- **Problem**: The docstring implies the returned stream slot will be recycled “when worker threads exit”, but `getStreamFromPool()` returns a raw `MPSStream*` without attaching it to TLS ownership, so the slot is not automatically released unless the caller also makes it current.
- **Impact**: Easy-to-misuse API that can silently exhaust the stream pool.
- **Suggested fix**: Update the doc to reflect actual lifecycle, or change the API to return an RAII handle that releases the slot.

