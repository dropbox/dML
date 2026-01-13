# AI Guidance / Repo Audit: MPS Parallel Inference

**Date**: 2025-12-13 (UTC) / 2025-12-12 (local PST)  
**Branch**: `main`  
**HEAD**: `7e4cf36` (`# 16: Thread Concurrency Limit Analysis`)  
**Current Plan**: `MPS_PARALLEL_INFERENCE_PLAN.md`  
**Current Patch**: `patches/008-upstream-hardened-buildfix.patch` (cumulative)

## 1) Repo state (factual)

- The repo is in a **clean git state** at `7e4cf36` (last worker commit is **N=16**).
- Plan status: **Phase 8 “AWAITING HUMAN”** (PR prep done; requires CLA + GitHub PR workflow).
- Known concurrency limit is documented in `MPS_PARALLEL_INFERENCE_PLAN.md`:
  - **31 worker threads safe** (streams 1–31).
  - **32+ worker threads can crash** due to **stream reuse** (wraparound).
- Worker loop state:
  - `worker_status.json` is **stale** (reports iteration 9 “running”, but PID is not running).
  - Latest codex log shows `Error: stdout is not a terminal`, indicating the codex mode is not currently usable in the loop.

## 2) Verification performed (factual)

Executed from this repo after the N=16 commit:

- `bash tests/run_all_tests.sh`: **ALL TESTS PASSED** (5/5).
- `venv_mps_test/bin/python tests/test_real_models_parallel.py`: **ALL REAL MODEL TESTS PASSED**.
  - This includes an **`nn.Sequential` MLP** with multiple `nn.Linear` layers running concurrently.
  - PyTorch printed: `2.9.1a0+gitd38164a` and `MPS available: True`.

## 3) Status of reported issue: `nn.Sequential` crash (contiguous → no-graph Linear)

**Issue**: `nn.Sequential` could crash because the second `Linear` often receives a contiguous input, selecting the **no-graph** path (`_mps_linear_nograph`) which uses Apple’s `MPSNDArrayMatrixMultiplication` and was observed to be **thread-unsafe** under concurrent encode.

**Current status**: **Addressed** in `patches/008-upstream-hardened-buildfix.patch`.

- Fix is in `pytorch-mps-fork/aten/src/ATen/native/mps/operations/Linear.mm`:
  - A **global mutex** (`s_linear_nograph_mutex`) serializes `_mps_linear_nograph(...)` to avoid Apple framework crashes.
- Validation: `tests/test_real_models_parallel.py` (MLP built via `nn.Sequential`) ran successfully under concurrency (see section 2).

## 4) Direction to AIs: what remains to finish “project goal” (now Phase 8)

At this point the technical work is **documented and patchified**; the remaining “project goal” is to land this upstream safely and predictably:

- Apply `patches/008-upstream-hardened-buildfix.patch` to a clean `pytorch/pytorch` checkout at the intended baseline (v2.9.1).
- Run PyTorch’s own validation (e.g., `lintrunner`, `python test/test_mps.py`) and fix any style/test fallout.
- Ensure the PR clearly documents:
  - The **Linear no-graph mutex** and the Apple limitation it mitigates.
  - The **thread concurrency limit** (see Issue A below) and intended behavior.

## 5) Three new issues for the next AI worker (with proposed fixes)

### Issue A — Stream pool over-subscription can hard-crash (>31 worker threads)

**Symptom**: With more worker threads than unique worker streams, the pool wraps and **reuses a stream**, which can lead to Metal assertions (documented in `MPS_PARALLEL_INFERENCE_PLAN.md`).

**Root cause**: `MPSStreamPool::acquireStream()` maps an ever-increasing counter into `[1, kMPSStreamsPerPool-1]` via modulo. When wraparound happens, **multiple threads share a single `MPSStream`**, but `MPSStream` is **not designed to be concurrently accessed** (e.g., `synchronize()`/commit paths are not protected by the stream’s serial queue).

**Proposed fix options** (pick one; document tradeoffs):

1. **Preferred (robust)**: Make the pool **dynamically grow** (unique stream per thread) up to a configurable cap.
   - Replace fixed `std::array` storage with a mutex-protected `std::vector<std::unique_ptr<MPSStream>>`.
   - Allocate a unique stream index per thread (monotonic assignment, no modulo).
   - Add a cap via env var (e.g., `PYTORCH_MPS_STREAM_POOL_SIZE`) to prevent unbounded queue creation.
2. **Minimal**: Increase `kMPSStreamsPerPool` (e.g., 64/128) to reduce practical risk, while still finite.
3. **Safety stopgap**: Detect exhaustion and throw a controlled error (`TORCH_CHECK`) instead of allowing Metal to assert/crash.

**Test plan**:
- Add a deterministic boundary test that tries `main + 32 workers` and asserts **no Metal assertion crash**:
  - Either “passes” (if pool grows), or fails with a controlled exception + clear message (if capped).
- Keep the existing “<=31 worker” test passing.

### Issue B — Upstream-facing tests + local runner gap for `nn.Sequential`

**Problem**: The repo has good standalone tests, but the upstream PR will be stronger with a **PyTorch-native regression test**. Also, `tests/run_all_tests.sh` currently does not run `tests/test_real_models_parallel.py`, which is the easiest local regression coverage for the `nn.Sequential`/Linear no-graph path.

**Proposed fixes**:

1. **Upstream test** (in the PyTorch tree when preparing the PR):
   - Add a small, bounded test that:
     - Runs `nn.Linear` and a small `nn.Sequential` MLP concurrently with a barrier.
     - Uses `torch.mps.synchronize()` and asserts no exceptions for a small iteration count.
     - Skips when MPS is unavailable.
2. **Local test runner improvement (this repo)**:
   - Add `tests/test_real_models_parallel.py` to `tests/run_all_tests.sh` so “sequential + multiple Linear” is always exercised.

**Acceptance**:
- New upstream test is stable and non-flaky on at least one macOS 15 machine.
- Local `bash tests/run_all_tests.sh` covers real-model concurrency.

### Issue C — `run_worker.sh` codex mode is broken (TTY requirement)

**Symptom**: `worker_logs/worker_iter_9_codex_*.jsonl` contains `Error: stdout is not a terminal`; `worker_status.json` is stale and points at a codex iteration.

**Root cause**: `run_worker.sh` runs `codex` in **interactive** mode and pipes it to `tee`. The interactive UI requires a TTY.

**Proposed fix**:
- Switch to non-interactive mode: `codex exec ...` (optionally `--json`), which is designed for piping/logging.
  - Example shape: `codex exec --dangerously-bypass-approvals-and-sandbox --json "$PROMPT" | tee "$LOG_FILE" | ./json_to_text.py`
- Ensure `worker_status.json` updates to `completed` with correct exit codes.

**Acceptance**:
- Iteration N divisible by 9 no longer fails immediately due to missing TTY.

## 6) Additional issues to consider (triage)

- Update the PR template / submission guide to include the **31-worker thread limit** and the intended behavior for over-subscription.
- Consider whether “first thread to touch MPS becomes main thread” semantics should be documented more prominently (or replaced with a more explicit policy) to avoid surprises in non-main-thread-first workloads.
