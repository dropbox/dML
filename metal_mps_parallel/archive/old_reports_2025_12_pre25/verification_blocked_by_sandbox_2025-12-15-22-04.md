# Verification Blocked by Sandbox (Metal Not Visible)

## Summary (2025-12-15 22:04 local)

This Codex sandboxed environment cannot see Metal devices (`MTLCreateSystemDefaultDevice: nil`, `MTLCopyAllDevices count: 0`). As a result, `torch.backends.mps.is_available()` is `False` and the MPS test suite cannot run here. Patch packaging checks *do* work and were verified.

## Observations

- `./tests/metal_diagnostics.sh`
  - `MTLCreateSystemDefaultDevice: nil`
  - `MTLCopyAllDevices count: 0`
- `source venv_mps_test/bin/activate && python -c 'import torch; ...'`
  - `torch.backends.mps.is_built() == True`
  - `torch.backends.mps.is_available() == False`

## Verified (Non-Metal)

- `./scripts/regenerate_cumulative_patch.sh --check`
  - PASS
  - Fork HEAD: `05d47cb6`
  - Patch MD5: `2d5b6248b58b4e475c0b14de685a3e04`
  - Patch aliases in sync

## Code Changes Made (Uncommitted)

- `patches/README.md`
  - Added a short note under “Verification Status” that MPS tests require Metal device access and that sandboxed/headless runners may report no visible `MTLDevice`, suggesting running via `./run_worker.sh` (Codex `--dangerously-bypass-approvals-and-sandbox`) or a normal Terminal session.

## Git Commit Blocker

`git commit` fails in this environment because the sandbox blocks writes to `.git/`:

- Error: `fatal: Unable to create '.../.git/index.lock': Operation not permitted`

To persist this change, commit from an environment that allows `.git/` writes.

## Proposed Commit Message (N=506)

```
# 506: Document Metal sandbox limitation in patch verification docs
**Current Plan**: MPS_PARALLEL_INFERENCE_PLAN.md
**Checklist**: N=506 - Doc hardening for sandboxed verification

## Changes
- Added explicit note to patches/README.md that the MPS test suite requires Metal device access and that sandboxed/headless runners can report no visible MTLDevice.

## Verification (N=506)
- ./scripts/regenerate_cumulative_patch.sh --check: PASS (MD5 2d5b6248b58b4e475c0b14de685a3e04; aliases in sync; fork HEAD 05d47cb6)
- ./tests/metal_diagnostics.sh: MTLCreateSystemDefaultDevice=nil, MTLCopyAllDevices count=0 (Metal not visible under this sandbox; cannot run MPS tests here)

## New Lessons
- Some sandboxed runners hide Metal devices even on Apple Silicon; run MPS validation from a non-sandboxed Terminal session or via ./run_worker.sh.

## Information Expiration
- None

## Next AI: N=507
- If Metal is visible, continue verification iterations (./tests/run_all_tests.sh + ./tests/tsan_mps_test + ./tests/record_stream_test{,_tsan}) and update patches/README.md verification markers only when tests actually run.
```

