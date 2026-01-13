# Static Analysis Summary

**Date:** 2025-12-17 00:59:02
**Repository:** /Users/ayates/metal_mps_parallel

## Tools Run

### Clang Thread Safety Analysis
- Status: Completed
- Warnings: [0;32m[INFO][0m Running Clang Thread Safety Analysis...
[0;32m[INFO][0m Output: /Users/ayates/metal_mps_parallel/mps-verify/verification/static/results/clang_tsa_20251217_005859.log
[0;32m[INFO][0m Thread Safety Analysis complete.
[0;32m[INFO][0m Warnings found: 78
[0;32m[INFO][0m Full output: /Users/ayates/metal_mps_parallel/mps-verify/verification/static/results/clang_tsa_20251217_005859.log
78

### Facebook Infer
- Status: Not installed

## Next Steps

1. Review warnings in the output files
2. Add TSA annotations to critical code paths
3. Fix any real concurrency bugs found
4. Re-run analysis to verify fixes

## Files Analyzed

      12 header files in:
- /Users/ayates/metal_mps_parallel/pytorch-mps-fork/aten/src/ATen/mps

