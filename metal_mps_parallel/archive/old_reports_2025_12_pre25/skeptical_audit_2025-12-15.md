# Skeptical Repo Audit (Performance + Safety) — 2025-12-15

**Status**: EXPIRED (point-in-time audit; see `patches/README.md` for current patch + verification state)

This is a point-in-time verification of the repo’s current patch/correctness state, plus additional gaps and scalability suggestions.

## State Verified

- **Repo HEAD**: `40ede63` (`# 297: Verification - 12/12 tests pass, TSan clean, 59% efficiency`)
- **Fork HEAD** (`pytorch-mps-fork`): `5287a4a3` (“Phase 29.3: Fix static destruction order UB in MPSProfiler”)
- **Cumulative patch**: `patches/cumulative-v2.9.1-to-mps-stream-pool.patch`
  - **MD5**: `8ed6120ee8e4dd15d998135d4feca351`
  - **Lines**: `4037`
  - Verified in-sync with fork diff: `./scripts/regenerate_cumulative_patch.sh --check`

## Tests Actually Run (This Audit Session)

- `bash -lc './tests/run_all_tests.sh'`
  - Result: **12 passed, 0 failed**
  - Preflight confirmed torch import is from `pytorch-mps-fork` and matches fork HEAD (`git5287a4a`)

## Previously Raised Issues — Current Status

### “Outdated / already fixed” claims (confirmed fixed in fork + patch)

1. **MPSEventPool raw-pointer lifetime/race**
   - Fixed: in-use events stored as `std::shared_ptr<MPSEvent>` and can be retrieved as shared ownership.
   - Evidence: `pytorch-mps-fork/aten/src/ATen/mps/MPSEvent.mm:244`, `pytorch-mps-fork/aten/src/ATen/mps/MPSEvent.h:124`

2. **MPSEvent reset cross-talk (resetting counters to 0)**
   - Fixed: reset bases the next counter on the current `signaledValue` (monotonic).
   - Evidence: `pytorch-mps-fork/aten/src/ATen/mps/MPSEvent.mm:164`

3. **MetalShaderLibrary cache default-insert via `operator[]`**
   - Fixed: `find()` + `emplace()` pattern (no default insert on miss).
   - Evidence: `pytorch-mps-fork/aten/src/ATen/native/mps/OperationUtils.mm:851`

### “Was real” issues (confirmed fixed and re-verified)

4. **Backpressure hang/regression**
   - Fixed: explicit `torch.mps.release_current_thread_slot()` API + `notify_all()` + polling wait.
   - Evidence: `pytorch-mps-fork/aten/src/ATen/mps/MPSStream.mm:494`, `pytorch-mps-fork/torch/mps/__init__.py:37`, `tests/test_stream_backpressure.py:107`
   - Verified by test suite run: `tests/run_all_tests.sh` (Test 12).

5. **Static destruction order UB (MPSProfiler destructor calling stream pool after it’s destroyed)**
   - Fixed: `MPSProfiler::~MPSProfiler()` checks `MPSStreamPool::isPoolAlive()` before synchronizing.
   - Evidence: `pytorch-mps-fork/aten/src/ATen/mps/MPSProfiler.mm:126`, `pytorch-mps-fork/aten/src/ATen/mps/MPSStream.h:294`, `pytorch-mps-fork/aten/src/ATen/mps/MPSStream.mm:589`

## New Gaps / Improvements (Not Strictly Blockers)

1. **`getActiveStreamCount()` semantics don’t match “currently assigned” wording when using `release_current_thread_slot()`**
   - `MPSStreamPool::releaseCurrentThreadSlot()` clears `slot_index`/`stream` but does not decrement the “active stream users” counter.
   - Impact: heuristics that use `getActiveStreamCount()` (e.g., `Linear.mm` graph-path forcing, LayerNorm warning) can remain in “parallel mode” even after a thread releases its slot but keeps running.
   - Evidence: `pytorch-mps-fork/aten/src/ATen/mps/MPSStream.h:268`, `pytorch-mps-fork/aten/src/ATen/mps/MPSStream.mm:667`, `pytorch-mps-fork/aten/src/ATen/native/mps/operations/Linear.mm:149`
   - Options: (a) decrement `g_active_stream_users` + clear `counted` on explicit release; or (b) rename/re-document it as “threads that have used MPS in this process”.

2. **Static-destruction mitigation still leaves a “best-effort” shutdown path**
   - Current fix avoids UB, but if `pendingCompletionHandlers > 0` and the pool is already dead, the profiler destructor skips synchronization; this can leave completion handlers racing with teardown.
   - Evidence: `pytorch-mps-fork/aten/src/ATen/mps/MPSProfiler.mm:126`
   - Option: force pool construction before profiler (in `getMPSProfiler()`) so pool outlives profiler, making the “pool dead” case unreachable.

3. **Backpressure wait scalability (thundering herd + 50ms polling interval)**
   - `notify_all()` + periodic `wait_for(50ms)` scales poorly with many waiters and adds latency for small timeouts.
   - Evidence: `pytorch-mps-fork/aten/src/ATen/mps/MPSStream.mm:515`
   - Option: track waiter count and call `notify_one()` per released slot, or use a semaphore-based design (one permit per free worker slot).

## Scalability / Efficiency Opportunities (At Least 3)

1. **Semaphore-based slot acquisition**
   - Replace CV + bitmask polling with a semaphore (permits = free worker slots) plus a separate structure for choosing a specific slot.
   - Benefit: avoids `notify_all()` herd; improves scalability under heavy oversubscription.

2. **Reduce `s_layer_norm_mutex` impact**
   - Current LayerNorm serialization is safe but can bottleneck multi-thread inference on LayerNorm-heavy models.
   - Benefit: an MPSGraph-based LayerNorm path (or other safe kernel) would unlock additional parallelism.
   - Evidence: `pytorch-mps-fork/aten/src/ATen/native/mps/operations/Normalization.mm:950`

3. **Make backpressure polling interval adaptive**
   - Use smaller waits for small `MPS_STREAM_POOL_WAIT_TIMEOUT_MS` (e.g., 1–5ms), larger waits for long timeouts, or exponential backoff.
   - Benefit: reduces tail latency while keeping CPU overhead bounded.
   - Evidence: `pytorch-mps-fork/aten/src/ATen/mps/MPSStream.mm:515`
