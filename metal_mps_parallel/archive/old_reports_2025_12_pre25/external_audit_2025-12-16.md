# External Audit (2025-12-16)

## Snapshot (Updated N=905)

- Top-level repo HEAD: `8e1fc83` (N=905)
- Fork HEAD (`pytorch-mps-fork/`): `05d47cb66cbd46669aa9ca24bcbca089463be8a7`
- Patch packaging check: `./scripts/regenerate_cumulative_patch.sh --check` **PASS**
  - Patch: `patches/cumulative-v2.9.1-to-mps-stream-pool.patch`
  - Alias: `patches/archive/015-cumulative-final.patch`
  - MD5: `2d5b6248b58b4e475c0b14de685a3e04`

## Latest Verification (N=905)

- `./tests/metal_diagnostics.sh`: Metal visible (`MTLCreateSystemDefaultDevice: Apple M4 Max`)
- tsan_mps_test (8t x 50i): no TSan reports, 31ms
- tsan_mps_test (31t x 100i): no TSan reports, 180ms (max unique streams)
- record_stream_test: 6/6 PASS (rebuilt + re-run)
- Patch check: **PASS**

## Previously raised items — verified resolved in current fork

- **MPSEventPool raw-pointer race**: in-use events are stored as `std::shared_ptr<MPSEvent>` and retrieved via `getInUseEventShared()` (prevents UAF when another thread calls `releaseEvent()`).
  - Evidence: `pytorch-mps-fork/aten/src/ATen/mps/MPSEvent.mm:393-415`, `pytorch-mps-fork/aten/src/ATen/mps/MPSEvent.mm:509-515`
- **MPSEvent::reset() cross-talk fix**: counter is based on current `signaledValue` and advanced to `+1`, avoiding “query true before record” and preventing old GPU signals from new owners.
  - Evidence: `pytorch-mps-fork/aten/src/ATen/mps/MPSEvent.mm:253-290`
- **MetalShaderLibrary cache correctness**: `getLibrary(params)` uses `find()`/`emplace()` under shard lock (no default-insert on miss) and destructor releases `libMaps_` entries.
  - Evidence: `pytorch-mps-fork/aten/src/ATen/native/mps/OperationUtils.mm:825-903`
- **MPSProfiler TLS/dispatch-context fix**: `getCurrentMPSStream()` prefers queue-specific stream via `dispatch_get_specific()`, so code running inside stream queues uses the owning stream, not stale TLS.
  - Evidence: `pytorch-mps-fork/aten/src/ATen/mps/MPSStream.mm:645-653`, `pytorch-mps-fork/aten/src/ATen/mps/MPSProfiler.mm:507-531`

## New high-severity gaps (not covered by "resolved" checklist)

### 1) Missing `recordDataPtrOnStream()` integration (correctness)

**Status: ✅ FIXED (N=477)**

Implemented `MPSGuardImpl::recordDataPtrOnStream(const c10::DataPtr&, const Stream&)` that routes to allocator's `recordStream()` via the new `IMPSAllocator::recordStream(ptr, stream_id)` interface method.

- Implementation: `pytorch-mps-fork/aten/src/ATen/mps/MPSGuardImpl.mm:111-120`
- Interface: `pytorch-mps-fork/aten/src/ATen/mps/MPSAllocatorInterface.h:43-45`

### 2) `recordStream()` semantics diverge from CUDA (correctness)

**Status: ⚠️ RE-EVALUATION (N=905): still a correctness risk**

The MPS allocator's `recordStream()` records an event only the *first* time a given stream ID is seen (`insert(...).second`) and never updates that event for later work on the same stream. CUDA's allocator only tracks the set of streams and records events at the appropriate time (at free) to cover the **latest** work.

- MPS records only on first insert: `pytorch-mps-fork/aten/src/ATen/mps/MPSAllocator.mm:1216-1228`
- Free path only queries/erases events (does not record a fresh "last-use" fence): `pytorch-mps-fork/aten/src/ATen/mps/MPSAllocator.mm:667-684`
- CUDA reference behavior: `pytorch-mps-fork/c10/cuda/CUDACachingAllocator.cpp:3529-3546` (`insert_events()` records at free-time)

**Semantic difference analysis:**
- **CUDA**: `recordStream()` only adds stream to `stream_uses` set; events are recorded at free-time via `insert_events()`
- **MPS**: `recordStream()` creates and records event immediately on first call per stream

**Why this is still risky**:
- Upstream call sites can record a tensor onto a consumer/accum stream *before* the consumer has performed its later work, and the tensor may be used across multiple command buffers before it is finally freed.
  - Evidence: `pytorch-mps-fork/torch/csrc/autograd/input_buffer.cpp:246-288` calls `recordDataPtrOnStream(..., consumer_stream)` before later accumulation/consumer work.
- With the current first-insert gating (`stream_uses_ids.insert(...).second`), subsequent `recordStream()` calls for the same stream become no-ops, so the allocator does not refresh the “last-use” fence when the tensor is used again on the same stream later in its lifetime.

**Fix options for upstream consideration:**
1. Track only stream IDs and record events at free (like CUDA) - cleaner but more invasive
2. Record a new event on every `recordStream()` call - simpler but creates more events

**Recommendation**: Treat this as an open correctness risk until either (a) the implementation matches CUDA (events recorded at free-time against the latest work), or (b) there is a targeted regression test that proves buffers cannot be recycled while still referenced by later command buffers on the same stream after an early `recordDataPtrOnStream()` call.

### 3) Per-stream `queryStream()` / `synchronizeStream()` are still unsupported (stability/API completeness)

**Status: ✅ FIXED (N=477)**

Implemented both methods in `MPSGuardImpl`:
- `queryStream()`: Returns `MPSStream::query()` which checks if command buffers are complete
- `synchronizeStream()`: Calls `MPSStream::synchronize(SyncType::COMMIT_AND_WAIT)`

Also added `MPSStream::query()` method that checks `_commandBuffer` and `_prevCommandBuffer` completion status.

- Implementation: `pytorch-mps-fork/aten/src/ATen/mps/MPSGuardImpl.mm:92-108`
- MPSStream::query(): `pytorch-mps-fork/aten/src/ATen/mps/MPSStream.mm:174-188`

### 4) No tests exercise record-stream semantics (regression risk)

**Status: ✅ FIXED (N=478)**

Added comprehensive test suite `tests/test_record_stream.mm` with 6 tests covering:
1. `test_basic_record_stream` - Basic recordStream API call
2. `test_multi_stream_record` - Recording same buffer on multiple streams
3. `test_cross_thread_record_stream` - Cross-thread buffer passing with recordStream
4. `test_null_pointer_safety` - Null pointer handling (safe no-op)
5. `test_freed_buffer_safety` - Freed buffer handling (safe no-op)
6. `test_concurrent_record_stream` - Stress test with 8 threads x 20 iterations

All 6 tests PASS with both regular build and ThreadSanitizer (TSan) build.

- Test file: `tests/test_record_stream.mm`
- Build script: `tests/build_record_stream_test.sh`
- Usage: `./record_stream_test` or `./build_record_stream_test.sh --tsan && ./record_stream_test_tsan`

### 5) Thread-local caches scale memory/compile cost linearly with thread count (scalability)

**Status: ✅ WILL_NOT_FIX - Thread-local is correct for safety; scalability is future enhancement**

Graph/kernel caches are per-thread (`thread_local`). This is the right correctness choice for thread safety, but it means N threads pay N× compilation + memory overhead, and thread churn amplifies it.

- Evidence: `pytorch-mps-fork/aten/src/ATen/native/mps/OperationUtils.h:298`, `pytorch-mps-fork/aten/src/ATen/native/mps/OperationUtils.h:397`

**Existing mitigation**: `torch.mps.release_current_thread_slot()` clears the TLS *stream binding*, allowing stream reuse after thread exit. However, the kernel/graph caches (`MPSKernelCache`, `MPSGraphCache`) are separate and not cleared by this function.

**Fix options for upstream consideration:**
1. Add `release_current_thread_caches()` to clear TLS kernel/graph caches
2. Add cache size limits with LRU eviction policy
3. Expose both: explicit clear API + bounded cache size

**Practical impact**: Low for typical usage (8-32 threads). Higher thread counts or long-running thread pools with varying workloads may accumulate significant cache memory. Thread churn is the main concern (new threads compile fresh caches).

**Decision**: WILL_NOT_FIX. Thread-local is the correct design for thread safety. Cache limits/eviction is a future enhancement, out of scope for threading correctness patch.

## Additional performance opportunities (lower severity)

- **Reduce lock contention in shader compilation**: `MetalShaderLibrary::getLibrary(params)` holds the shard lock across compilation (slow path).
  - Evidence: `pytorch-mps-fork/aten/src/ATen/native/mps/OperationUtils.mm:862-902`
  - Option: store “in-flight compile” futures per key to avoid holding a mutex across compilation while still preventing duplicate work.
- **Avoid fixed-duration spin loops on shutdown**: multiple destructors wait with fixed 5s timeouts (profiler/allocator/event), which can add exit latency and hide true pending work.
  - Evidence: `pytorch-mps-fork/aten/src/ATen/mps/MPSProfiler.mm:176-194`, `pytorch-mps-fork/aten/src/ATen/mps/MPSAllocator.mm:1483-1498`, `pytorch-mps-fork/aten/src/ATen/mps/MPSEvent.mm:40-48`

## Additional issues found (N=905)

### A) `getCurrentMPSStream()` returns queue-specific stream without pool-liveness guard (shutdown safety)

`getCurrentMPSStream()` returns `dispatch_get_specific(...)` result directly. If the stream pool (and its streams) are destroyed while GCD blocks still execute on those queues, this can return a dangling `MPSStream*`.

- Evidence: `pytorch-mps-fork/aten/src/ATen/mps/MPSStream.mm:698-706`
- Suggested hardening: check `MPSStreamPool::isPoolAlive()` before returning the queue-specific pointer (or clear queue-specific values during teardown).

### B) `MPSStream::query()` likely misreports when `commitAndContinue` is enabled (API correctness)

`MPSStream::query()` returns false whenever `_commandBuffer != nil`, which is always true after the first use if stream 0 keeps a live command buffer under `commitAndContinue`.

- Evidence: `pytorch-mps-fork/aten/src/ATen/mps/MPSStream.mm:65-76` (commitAndContinue enable), `pytorch-mps-fork/aten/src/ATen/mps/MPSStream.mm:174-188` (query implementation)
- Suggested fix: track “has pending encoded work” separately from “command buffer object exists”, or query completion status in a way that works with commit-and-continue semantics.

### C) `IMPSAllocator::recordStream(ptr, stream_id)` should defensively guard `stream_id` range (robustness)

The interface accepts `int64_t stream_id`, but the implementation casts to `size_t` and calls `getStream(index)` which throws on out-of-range IDs. If an unexpected stream id reaches this path (e.g., from a generic `c10::Stream`), this can become an avoidable hard failure.

- Evidence: `pytorch-mps-fork/aten/src/ATen/mps/MPSAllocator.mm:1622-1629`, `pytorch-mps-fork/aten/src/ATen/mps/MPSStream.mm:562-570`
- Suggested fix: early-return if `stream_id < 0 || stream_id >= MPSStreamPool::poolSize()`.

### D) `MetalShaderLibrary::getKernelFunction()` allocates a new wrapper each call (hot-path overhead)

`getKernelFunction(name)` returns a fresh `MetalKernelFunction` wrapper each time, even when the pipeline state/function are cached.

- Evidence: `pytorch-mps-fork/aten/src/ATen/native/mps/OperationUtils.mm:988-991`
- Potential improvement: add a per-thread cache of `MetalKernelFunction` wrappers keyed by kernel name (keeping the existing “do not share across threads” constraint).

### E) Potential ABA in allocator “double-check” pattern (rare correctness risk)

Several allocator APIs use a two-phase lookup (lock `m_mutex` → drop → lock `pool_mutex` → re-lock `m_mutex`) and only validate pointer equality (`it->second != buffer_block`). If a `BufferBlock*` is freed and a new `BufferBlock` is later allocated at the same address between the two checks, the validation can succeed while referring to a different logical block generation, corrupting the new owner’s metadata.

- Evidence (pattern): `pytorch-mps-fork/aten/src/ATen/mps/MPSAllocator.mm:1175-1228` (`recordStream()`), `pytorch-mps-fork/aten/src/ATen/mps/MPSAllocator.mm:1063-1113` (`recordEvents()`), `pytorch-mps-fork/aten/src/ATen/mps/MPSAllocator.mm:1116-1173` (`waitForEvents()`)
- Suggested hardening: capture a generation counter (e.g., `buffer_block->use_count`) during the first lookup and re-validate it under the second `m_mutex` lock alongside pointer equality.
