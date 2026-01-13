# Appendix Directory

This directory contains all supporting evidence for the AGX race condition research paper.

## Contents

| File | Description |
|------|-------------|
| `appendix_a_crash_reports.md` | Full crash reports with register dumps and stack traces |
| `appendix_b_tlaplus.md` | TLA+ specifications and TLC verification results |
| `appendix_c_lean4.md` | Lean 4 machine-checked proofs |
| `appendix_d_disassembly.md` | Disassembly and reverse engineering analysis |
| `appendix_e_benchmarks.md` | Complete benchmark data and statistics |

## Cross-Reference from Main Paper

The main research paper (`papers/agx_race_condition_research.md`) references these appendices as follows:

- **Section 6 (Formal Verification)** → Appendix B (TLA+), Appendix C (Lean 4)
- **Section 7 (Driver Analysis)** → Appendix A (Crash Reports), Appendix D (Disassembly)
- **Section 8 (Evaluation)** → Appendix E (Benchmarks)

## Evidence Summary

### Crash Reports (Appendix A)
- 3 distinct crash sites in AGXMetalG16X driver
- NULL pointer dereferences at offsets 0x5c8, 0x184, 0x98
- ~55% crash rate without mutex protection

### Formal Verification (Appendix B, C)
- TLA+: 32.5M states explored, race condition found
- TLA+ (Phase 4.1): 3 additional models proving global mutex is minimal solution
  - AGXPerStreamMutex.tla: Per-stream mutex insufficient
  - AGXPerOpMutex.tla: Per-operation mutex insufficient
  - AGXRWLock.tla: Reader-writer lock insufficient
- Lean 4: Machine-checked proofs of race existence and mutex fix
- Lean 4 (Phase 5.3): 3 additional proofs proving global mutex is minimal
  - PerStreamMutex.lean: Per-stream mutex insufficient
  - PerOpMutex.lean: Per-operation mutex insufficient
  - RWLock.lean: Reader-writer lock insufficient

### Reverse Engineering (Appendix D)
- ContextCommon structure layout inferred
- Three crash sites mapped to driver functions
- Root cause identified as missing synchronization

### Benchmarks (Appendix E)
- Threading plateaus at ~3,900 ops/s
- Batching achieves 1.4M samples/s (365x better)
- Mutex overhead: 0.34% (statistically insignificant)
