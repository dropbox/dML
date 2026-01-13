# More Scalability & Efficiency Improvements (MPS Parallel Inference)

**Date**: 2025-12-13 23:09 (local)

This is a follow-up list of additional scalability/efficiency improvements beyond those already discussed in earlier audits.

## Notes

- No benchmarks/tests were run for these items.

## Improvements

### 1) Reduce global allocator mutex contention (hot-path scalability limiter)

- **Locations**:
  - Global allocator lock: `pytorch-mps-fork/aten/src/ATen/mps/MPSAllocator.h:361`
  - Example hot-path entrypoints holding the lock:
    - `pytorch-mps-fork/aten/src/ATen/mps/MPSAllocator.mm:509` (`MPSHeapAllocatorImpl::malloc`)
    - `pytorch-mps-fork/aten/src/ATen/mps/MPSAllocator.mm:655` (`MPSHeapAllocatorImpl::free`)
    - `pytorch-mps-fork/aten/src/ATen/mps/MPSAllocator.mm:556` (`recordEvents`)
- **Observation**: Multi-threaded inference increases concurrent allocator calls; a single global `std::recursive_mutex` can become a scaling bottleneck even if streams/graphs are per-thread.
- **Improvement ideas**:
  - Shard allocator state/locks by pool (small/large/shared/private) or by size class.
  - Add per-thread caches for small allocations (fast-path w/o global lock), flushing to the global pools occasionally.

### 2) Gate `commitAndContinue` disabling to preserve single-thread throughput

- **Location**: `pytorch-mps-fork/aten/src/ATen/mps/MPSStream.mm:43`
- **Observation**: The patch disables `commitAndContinue` globally “at the cost of some pipelining efficiency”.
- **Improvement ideas**:
  - Re-enable commit-and-continue for the default stream (or when only one stream is active) to recover single-thread throughput while keeping the parallel safety behavior when multiple streams are in use.
  - Make this explicitly configurable via an env var for experimentation.

### 3) Eliminate global lock + scan in `MPSStreamPool::setCurrentStream` (stream switching overhead)

- **Location**: `pytorch-mps-fork/aten/src/ATen/mps/MPSStream.mm:503`
- **Observation**: `setCurrentStream()` scans `streams_` under `stream_creation_mutex_` to discover the slot index (`:509-517`). This adds lock contention and repeated work on every stream switch.
- **Improvement ideas**:
  - Store the pool index on the `MPSStream` itself (or maintain a pointer→index map built at creation time) so `setCurrentStream()` doesn’t need to scan.
  - Avoid sharing `stream_creation_mutex_` for both “read existing pointer” and “create new stream”.

### 4) Reduce lock overhead for `getStream(index)` / `createStream(index)` in guard/event paths

- **Location**: `pytorch-mps-fork/aten/src/ATen/mps/MPSStream.mm:402`
- **Observation**: `createStream()` takes `stream_creation_mutex_` even when `streams_[index]` is already initialized (`:408-414`). Guard/event paths call `MPSStreamPool::getStream(index)` frequently enough that this can become measurable on highly concurrent workloads.
- **Improvement ideas**:
  - Use per-index `std::once_flag` (or an atomic pointer fast-path + locked slow-path) to make the common “already created” case lock-free.

### 5) Reduce freelist contention for high thread churn workloads

- **Location**: `pytorch-mps-fork/aten/src/ATen/mps/MPSStream.mm:445`
- **Observation**: `acquireSlot()` and `releaseStreamSlot()` serialize on `slot_mutex_`. This is not in the per-op hot path for long-lived worker threads, but becomes expensive if the workload creates/destroys threads frequently.
- **Improvement ideas**:
  - Use a lock-free freelist/bitset for 31 worker slots or encourage the intended usage pattern (thread pool / long-lived workers) at the API boundary.

