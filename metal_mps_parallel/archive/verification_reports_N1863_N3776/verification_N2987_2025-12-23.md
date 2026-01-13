# Verification Report N=2987

**Date**: 2025-12-23 16:18 PST
**Worker**: N=2987
**Previous**: N=2986

## Test Results

| Test | Threads | Ops | Result | Throughput | New Crashes |
|------|---------|-----|--------|------------|-------------|
| Linear matmul | 4 | 80 | PASS | 2255 ops/s | 0 |
| Linear matmul | 8 | 400 | PASS | 3737 ops/s | 0 |
| LayerNorm | 4 | 100 | PASS | 2277 ops/s | 0 |
| complete_story | multi | multi | CRASH | exit 139 | 1 |

## Crash Details

**complete_story test CRASHED** (exit code 139 = SIGSEGV)

Crash type: PAC failure at `objc_msgSend + 32`
- Address: 0x00009ba36e2f2650 -> 0x00001ba36e2f2650
- Fault: KERN_INVALID_ADDRESS (possible pointer authentication failure)

This confirms N=2978's TLA+ analysis: userspace v2.5 fix cannot prevent 100% of crashes.
The race between `objc_msgSend` dispatch and encoder deallocation remains.

## Status

- Total crash logs: 248 (was 247, +1 from this test)
- SIP: enabled (binary patch blocked)
- v2.5 dylib: works for simple/moderate tests
- complete_story: intermittently crashes (probabilistic)

## Artifact Checksums

| Artifact | Status |
|----------|--------|
| AGX_DRIVER_BACKUP.bin | MD5 17a72eeac6b66418096952dc4d725c01 ✅ |
| AGXMetalG16X_universal_patched | 20MB ✅ |
| libagx_fix_v2_5.dylib | 100KB ✅ |
| deploy_patch.sh | exists ✅ |
| verify_patch.py | exists ✅ |

## Conclusion

v2.5 provides partial protection but complete_story crashes are still possible.
Binary patch requires user to disable SIP.
