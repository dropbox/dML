# Verification Report N=1004

**Date**: 2025-12-17
**Worker**: N=1004 (Codex)

## Summary

Re-verified patch packaging and MPS parallel inference tests in an environment with Metal device access.

## Environment

- macOS 15.7.2 (arm64)
- Metal device: Apple M4 Max (`MTLCreateSystemDefaultDevice` non-nil; `MTLCopyAllDevices` count: 1)
- torch: `2.9.1a0+git755017b` imported from `pytorch-mps-fork/`

## Results

| Check | Result |
|------|--------|
| `./scripts/regenerate_cumulative_patch.sh --check` | PASS (base `v2.9.1`, head `755017b6`, MD5 `142970ee0a7950e1fa16cbdd908381ee`) |
| `wc -l patches/cumulative-v2.9.1-to-mps-stream-pool.patch` | `7376` |
| `./tests/run_all_tests.sh` | PASS (24 passed, 0 failed) |
| `./tests/tsan_mps_test` | PASS (8t×50i: 30ms, 0 errors) |
| `./tests/tsan_mps_test 31 100` | PASS (31t×100i: 176ms, 0 errors) |
| `./tests/record_stream_test_tsan` | PASS (6/6) |
| `./venv_mps_test/bin/python tests/test_efficiency_large_workload.py` | PASS (2T efficiency: 74.4%) |
| `./venv_mps_test/bin/python tests/test_max_streams_stress.py` | PASS (31t×100i elapsed: 0.595s; sustained rounds PASS) |

## CBMC Bounded Model Checking

```
CBMC v6.8.0 - 4/4 harnesses PASS (0/1190 assertions failed)
  - ABA Detection: 0/384 failed
  - Alloc/Free: 0/239 failed
  - TLS Cache: 0/318 failed
  - Stream Pool: 0/249 failed
```

## Notes

- Documentation updated to reflect current patch MD5/size and verification N=1004.
- CBMC verification re-run confirms all 4 harnesses pass.
