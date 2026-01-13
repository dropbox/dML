# Verification Report N=3884

**Date**: 2025-12-26 05:19 PST
**Worker**: N=3884
**Platform**: Apple Silicon (Metal visible)

## Summary

Hardened `scripts/run_mps_test.sh` to match the current worker guardrails:
- Skip cleanly when Metal devices are not visible (sandbox/headless runners).
- Capture crash logs from macOS DiagnosticReports even when `crash_monitor.sh` is not running.

## Changes

- `scripts/run_mps_test.sh`
  - Added Metal visibility preflight via `tests/metal_diagnostics.sh --check` (opt out with `MPS_TEST_METAL_PREFLIGHT=0`).
  - Added opportunistic crash report capture from:
    - `~/Library/Logs/DiagnosticReports`
    - `/Library/Logs/DiagnosticReports`
    into `crash_logs/`, filtered for `Python/AGX/Metal/mps/torch`.

## Sanity Check

- Command: `MPS_TEST_CRASH_WAIT_SECS=0 ./scripts/run_mps_test.sh -c 'print("run_mps_test ok")'`
- Command: `MPS_TEST_CRASH_WAIT_SECS=0 ./scripts/run_mps_test.sh -c 'import torch; print("mps_available", torch.backends.mps.is_available()); x=torch.zeros(1, device="mps"); torch.mps.synchronize(); print("ok")'`
- Result: both exit 0; output includes `mps_available True`; no new crash logs reported by the wrapper.
