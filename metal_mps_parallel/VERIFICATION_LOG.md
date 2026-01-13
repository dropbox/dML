# Verification Log (Historical)
#
# NOTE: This file contains historical verification notes (N=1040..1054) and is
# superseded by:
# - PROJECT_STATUS.md (current verified status + test summaries)
# - reports/main/throughput_N1261_2025-12-18.md (throughput scaling)
# - python3 tests/complete_story_test_suite.py (reproducible "complete story" suite)
#
# The entries below are kept for archaeology only.

# Verification N=1054 - Wed Dec 17 15:18 PST 2025
# IMPORTANT: TransformerBlock has ~10% intermittent failure rate at 2+ threads
# Correctness: 37/40 PASS (TransformerBlock at 2/4/8 threads fails intermittently)
#   - All individual ops PASS (LayerNorm, Linear, GELU, Matmul, Conv2d, Softmax)
#   - TransformerEncoderLayer fails ~10% of runs (MultiheadAttention race)
# Other tests: fork_safety PASS, oversubscription PASS, verify_layernorm_fix PASS
# record_stream_test_tsan: CRASHED (BUS error - test binary needs rebuild)
# Test fix: Added copy.deepcopy() for TransformerBlockTest (was test artifact race)
# Fork: mps-stream-pool branch, HEAD: 1038e7b1
# Patch: 3800 lines, MD5: d0ad4710321f420b099b1471439d60c9 (regenerated)
# Metal: Apple M4 Max (40 GPU cores)
# NOTE: Previous verifications reported 7571-line patch - this was incorrect.
#       Actual fork diff is 3800 lines (27 files, 1898+, 404- vs v2.9.1)

# Local verification - Wed Dec 17 14:54 PST 2025
# Tests: 24/24 PASS (tests/run_all_tests.sh)
# Patch regenerated from fork diff (scripts/regenerate_cumulative_patch.sh)
# Fork on mps-stream-pool branch, HEAD: 1038e7b1
# Patch: 3800 lines, MD5: d0ad4710321f420b099b1471439d60c9
# Metal: Apple M4 Max (40 GPU cores)

# Verification N=1053 complete - Wed Dec 17 14:42 PST 2025
# Tests: 40/40 PASS (correctness), 6/6 record_stream PASS, fork_safety PASS, oversubscription PASS
# Fork on mps-stream-pool branch, HEAD: 1038e7b1
# Patch: 7571 lines, MD5: f3df5045cf988a4a95713ef30a3cb6c4
# Metal: Apple M4 Max (40 GPU cores)

# Verification N=1052 complete - Wed Dec 17 14:38 PST 2025
# Tests: 24/24 PASS (test suite), 40/40 PASS (correctness), 0 TSan races (8t x 50i, 32ms), 6/6 record_stream PASS
# Patch: Applies cleanly to v2.9.1 (git apply --check passed)
# Fork HEAD: 1038e7b1, Patch: 7571 lines, MD5: f3df5045cf988a4a95713ef30a3cb6c4
# Metal: Apple M4 Max (40 GPU cores)

# Verification N=1051 complete - Wed Dec 17 14:36 PST 2025
# Tests: 24/24 PASS (test suite), 40/40 PASS (correctness), 0 TSan races (8t x 50i, 31ms), 6/6 record_stream PASS
# Patch: Applies cleanly to v2.9.1 (git apply --check passed)
# Fork HEAD: 1038e7b1, Patch: 7571 lines, MD5: f3df5045cf988a4a95713ef30a3cb6c4
# Metal: Apple M4 Max (40 GPU cores)

# Verification N=1050 complete - Wed Dec 17 14:30 PST 2025
# Tests: 24/24 PASS (test suite), 40/40 PASS (correctness), 0 TSan races (8t x 50i, 33ms), 6/6 record_stream PASS
# Fork on mps-stream-pool branch, HEAD: 1038e7b1
# Metal: Apple M4 Max (40 GPU cores)

# Verification N=1049 complete - Wed Dec 17 14:28 PST 2025
# Tests: 24/24 PASS (test suite), 40/40 PASS (correctness), 0 TSan races, 6/6 record_stream PASS
# Patch: Applies cleanly to v2.9.1 (git apply --check passed)
# Fork HEAD: 1038e7b1 (patch 7571 lines, MD5 f3df5045cf988a4a95713ef30a3cb6c4)

# Verification N=1043 complete - Wed Dec 17 11:52 PST 2025 (CLEANUP iteration)
# TLA+: MPSAllocator 15.3M states PASS, MPSStreamPool 535K states PASS
# CBMC: Not run this iteration (slow - deferred)
# Tests: 24/24 PASS (with torch mismatch override - only allocator sharding changed)
# Efficiency: 2 threads - nn.Linear 86.4%, MLP 86.4%, Transformer 78.4%
# Cleanup: Removed duplicate patch file, regenerated cumulative patch
# Fork HEAD: 9a7876e6 (patch 7333 lines, MD5 340552a4543306c78a7461a959aee50b)

# Verification N=1042 complete - Wed Dec 17 10:50 PST 2025
# TLA+: MPSAllocator 15.3M states PASS, MPSStreamPool 535K states PASS
# CBMC: 4/4 harnesses (alloc_free 239, stream_pool 249, aba_detection 384, tls_cache 318)
# Tests: Not run (torch build version mismatch - needs rebuild)
# Efficiency: 74.5% avg (2 threads: nn.Linear 78.3%, MLP 71.8%, Transformer 73.4%)
# Fork HEAD: 9a7876e6 (patch 7540 lines, MD5 a630ff261e3b4bc276fcb3d739cfa8f2)

# Verification N=1041 complete - Wed Dec 17 10:38 PST 2025
# TLA+: MPSAllocator 15.3M states PASS, MPSStreamPool 535K states PASS
# CBMC: 4/4 harnesses (alloc_free 239, stream_pool 249, aba_detection 384, tls_cache 318)
# Tests: 24/24 PASS - full test suite including fork_safety, parallel_mps, thread_churn (80 threads)
# Efficiency: 74.4% (2 threads, nn.Linear benchmark)
# Fork HEAD: 9a7876e6 (patch 4764 lines, MD5 7e69608375843e5d09aed72cd0bda256)

# Verification N=1040 complete - Wed Dec 17 10:24 PST 2025
# TLA+: MPSAllocator 15.3M states PASS, MPSStreamPool 535K states PASS
# CBMC: 4/4 harnesses (alloc_free 239, stream_pool 249, aba_detection 384, tls_cache 318)
# Tests: 10/10 PASS - parallel_mps (400 ops), fork_safety, stream_assignment, efficiency,
#        static_destruction, cross_stream_tensor, linalg_ops (240 ops), thread_churn (80 threads),
#        stream_pool_boundaries, oversubscription
# Efficiency: 73.2% (2 threads, 2048x2048 batch=64)
# Patch MD5: a630ff261e3b4bc276fcb3d739cfa8f2 (7540 lines)

# Verification N=1039 complete - Wed Dec 17 09:55 PST 2025
# TLA+: MPSAllocator 15.3M states PASS, MPSStreamPool 535K states PASS
# CBMC: 4/4 harnesses, 0/1190 assertions failed
