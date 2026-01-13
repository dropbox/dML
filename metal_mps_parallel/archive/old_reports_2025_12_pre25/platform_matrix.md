# Platform Matrix (R3)

**Created**: 2025-12-20  
**Addressing**: Reviewer Objection #3 — No Multi-Platform Verification

This project is validated primarily on Apple M4 Max. This matrix tracks what has been tested and where the evidence lives, and provides a repeatable way for other machines (M1/M2/M3/Ultra, different RAM sizes, and different macOS versions) to contribute equivalent artifacts.

## How to Contribute a Platform Entry

1. Confirm Metal is visible:
   - `./tests/metal_diagnostics.sh`
   - If it prints `MTLCreateSystemDefaultDevice: nil` and `MTLCopyAllDevices count: 0`, the process cannot run MPS/Metal tests (sandbox/headless runner). Use a normal Terminal session with Metal device access.

2. Collect a platform fingerprint (text):
   - `./tools/collect_platform_fingerprint.sh > reports/main/platform_fingerprint_<chip>_<date>.txt`

3. Run platform assumption checks (JSON):
   - `./verification/run_platform_checks --json > reports/main/platform_report_<chip>.json`

4. Run the platform-specific test suite (JSON printed to stdout):
   - `python tests/test_platform_specific.py`

5. Run the full test suite:
   - `./tests/run_all_tests.sh`

Attach the produced artifacts to a PR and add/refresh the corresponding row below.

## Test Matrix

| Chip | RAM | macOS | Tested | Result | Evidence | Notes |
|------|-----|-------|--------|--------|----------|-------|
| M4 Max | 128GB | 15.7.3 | ✅ | PASS | `reports/main/platform_fingerprint_M4Max_N1317.txt`, `reports/main/platform_report_M4Max.json` | Primary dev machine; Dynamic Caching enabled; 40 GPU cores |
| M3 (any) | - | - | ❌ | - | - | Needed (Dynamic Caching generation) |
| M2 (any) | - | - | ❌ | - | - | Needed |
| M1 (any) | - | - | ❌ | - | - | Needed |
| Ultra (any) | - | - | ❌ | - | - | Needed (dual-die) |

## Notes

- Platform fingerprint script: `tools/collect_platform_fingerprint.sh`
- Platform assumption checks: `tests/test_assumption_falsification.py` (writes a machine-readable JSON report)
- Full suite runner prints platform info at start: `tests/run_all_tests.sh`
