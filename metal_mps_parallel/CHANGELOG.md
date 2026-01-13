# Changelog

All notable changes to the MPS Parallel Inference patch are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- **v2.7 userspace fix** (`libagx_fix_v2_7.dylib`) - adds command buffer commit-safety for `-[IOGPUMetalCommandBuffer validate]` SIGABRT; pair with Semaphore(2) throttling for heavy multi-threaded workloads
- `tests/test_semaphore_recommended.py` - demonstrates Semaphore(2) throttling (0-crash heavy threading with partial parallelism)
- **v2.3 userspace fix** (`libagx_fix_v2_3.dylib`) - combines v2.2's retain-from-creation with v2.1's mutex protection
- TLA+ specification: `agx_patch/AGXDylibFix.tla` - proves v2.3 correctness (8.2B states explored)
- Binary patch deployment scripts: `deploy_patch.sh`, `revert_patch.sh`, `verify_patch.py`

### Fixed
- **v2.2 regression** (N=1968): v2.2 crashed at 8 threads due to missing mutex protection in encoder methods
- Test path issues (N=1970): `tests/complete_story_test_suite.py` and `tests/verify_layernorm_fix.py` now work from any directory

### Deprecated
- v2.2 (`libagx_fix_v2_2.dylib`) - crashes at 8 threads, replaced by v2.3
- v2.3 (`libagx_fix_v2_3.dylib`) - superseded by v2.7 (adds commit-safety for validate() SIGABRT)

### Verified
- Full verification pass (N=1820): Lean 4 proofs BUILD SUCCESS (60 jobs)
- v2.7 + Semaphore(2) stability (N=3110): `./scripts/run_test_with_crash_check.sh python3 tests/test_semaphore_recommended.py` PASS (0 new crashes), `./scripts/run_test_with_crash_check.sh python3 tests/complete_story_test_suite.py` PASS (0 new crashes)
- v2.3 thread safety (N=1971): 160/160 operations at 8 threads, no crashes
- v2.3 correctness: outputs match CPU reference (max diff < 1e-6)
- Multi-queue parallel (65k/10 kernel-iters): 9.91x scaling at 16T shared queue (53,681 ops/s), 8.78x per-thread at 16T (69,003 ops/s)
- Async pipeline: +1972% single-threaded (115,057 ops/s at depth=32), +28% multi-threaded at 8T (100,008 ops/s at depth=4)

### Documentation
- Removed a non-reproducible 3.64x throughput claim; current performance discussion is verified by `python3 tests/complete_story_test_suite.py`.

## [1.2.0] - 2025-12-18

### Added
- `torch.mps.BatchQueue` - Thread-safe batch queue for serialized GPU access
- `MPSBatchQueue.h/.mm` - C++ batch queue implementation with worker pool
- Python bindings: `torch.mps.batch_queue.py` - High-level batching API
- TLA+ specifications: `specs/MPSStreamPool.tla`, `MPSAllocator.tla`, `MPSEvent.tla`
- Lean 4 verification platform: `mps-verify/` - Multi-language formal verification
- CBMC harnesses: 4 bounded model checking harnesses (0 failures)

### Changed
- Default `num_workers=1` for `BatchQueue` (serializes GPU access for correctness)
- Updated documentation to reflect single-worker batching requirement

### Fixed
- **8-thread correctness**: 10/10 tests PASS via single-worker batching (N=1260)
- TransformerBlock/SDPA race conditions avoided by serialized GPU access

### Performance
- Threading throughput scaling is limited on current Apple MPS/Metal stacks; see `python3 tests/complete_story_test_suite.py`.
- For throughput, prefer batching/dynamic batching (see `docs/USER_GUIDE.md` and `BLOG_POST.md`).

### Verified
- TLA+ model checking: Deadlock freedom verified
- Lean 4 proofs: ABA detection correctness proven
- CBMC: 10 harnesses, 3,856 checks, 0 failures (N=1289)
- Clang TSA + Infer: Static analysis clean

## [1.1.0] - 2025-12-17

### Added
- `CONTRIBUTING.md` - Contribution guidelines for the patch
- `SUBMISSION_PROOF.md` - PyTorch contribution requirements proof
- `apple_feedback/` - Apple Feedback submission package (bug report, repro scripts)
- Complete Quick Start build instructions in README.md
- Documentation table of contents in README.md
- Separate test patch: `test-mps-parallel-inference.patch`
- Known Issues section in tests/README.md documenting intermittent SIGSEGV

### Changed
- Regenerated cumulative patch from fork diff (3800 lines, 27 files)
- Updated BLOG_POST.md with accurate statistics (201 issues, 1060+ iterations)

### Investigated
- Phase 0: Steel kernel hypothesis invalidated - Metal/AGX driver is the bottleneck
- MLX v0.30.0 benchmarks confirm same Metal limitation (crashes at 2+ threads)
- Our patches achieve 29% efficiency at 8 threads; MLX crashes

## [1.0.0] - 2025-12-16

### Added
- **MPSStreamPool**: Thread-safe stream pool with 32 streams (1 default + 31 pooled)
- **Thread-local stream assignment**: Each thread gets automatic stream from pool
- **TLS cleanup**: Worker stream slots returned to pool on thread exit
- **Auto-detection**: Parallel mode switches to thread-safe paths automatically
- **Test suite**: 24 tests covering parallel inference, TSan verification, fork safety

### Fixed
- 201 threading issues (32.110-32.310) resolved during development
- Thread-safe mutex protection for Apple MPS framework workarounds
- PSO deadlock in Metal shader compilation
- commitAndWait() blocking with mutex held
- Structured binding capture in Objective-C++ blocks
- LRU cache thread safety in allocator

### Security
- Thread sanitizer (TSan) verified: 0 data races (8 threads x 50 iterations)

### Known Limitations
- Apple MPS framework has internal thread-safety issues (documented in `apple_feedback/APPLE_FEEDBACK_BUG_REPORT.md`)
- `nn.Linear` no-graph path requires mutex serialization at 3+ threads
- `nn.LayerNorm` requires mutex serialization at 4+ threads
- Pool size limited to 32 streams (pool exhaustion behavior controlled by `MPS_STREAM_POOL_WAIT_TIMEOUT_MS`)

## Version History

| Version | Date | Patch Lines | Notes |
|---------|------|-------------|-------|
| 1.2.0 | 2025-12-18 | 3800 | Batching for 10/10 correctness, formal verification |
| 1.1.0 | 2025-12-17 | 3800 | Regenerated patch, Apple Feedback package, Steel investigation |
| 1.0.2 | 2025-12-16 | 7376 | 32.310 fix (fork commit 755017b6) |
| 1.0.1 | 2025-12-16 | 7210 | With tests (fork commit 10e734a0) |
| 1.0.0 | 2025-12-16 | 7082 | Initial release (fork commit 4002a2c0) |
| 0.9.0 | 2025-12-15 | 6951 | Pre-release (fork commit 5af829b7) |

## References

- [patches/README.md](patches/README.md) - Detailed patch evolution
- [WORKER_DIRECTIVE.md](WORKER_DIRECTIVE.md) - Active worker directive (AGX driver patch deployment/testing)
- [archive/WORKER_DIRECTIVE_HISTORICAL.md](archive/WORKER_DIRECTIVE_HISTORICAL.md) - Historical issue tracking (201 issues)
- [AI_TECHNICAL_SPEC.md](AI_TECHNICAL_SPEC.md) - Technical architecture
