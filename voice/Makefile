# Voice Project Makefile
# Test runner shortcuts for pytest and C++ tests
#
# Usage:
#   make test-smoke       - Quick smoke tests (<10s)
#   make test-unit        - Unit tests (<60s)
#   make test-integration - Integration tests (<5min)
#   make test-quality     - LLM-as-judge quality tests
#   make test-stress      - Stress tests (~10min)
#   make test-all         - All tests except stress
#   make test-cpp         - C++ unit tests (queue tests)

.PHONY: test-smoke test-unit test-integration test-quality test-stress test-all test-cpp test-roundtrip test-multilingual help test-stress-concurrent test-stress-sequential test-stress-memory test-stress-latency test-stress-report test-pipeline test-daemon test-streaming test-errors validate validate-models verify-models-full sync-submodules test-latency-quick build build-debug build-lto benchmark-nllb perf-smoke perf-snapshot check-deps

# Directories
TESTS_DIR := tests
CPP_BUILD_DIR := stream-tts-cpp/build

# Python environment - prefer .venv (project standard), fall back to venv if present
VENV := $(if $(wildcard .venv/bin/activate),.venv,$(if $(wildcard venv/bin/activate),venv,.venv))
PYTEST := $(if $(wildcard $(VENV)/bin/pytest),$(VENV)/bin/pytest,pytest)
PYTHON := $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3)

# Default target
help:
	@echo "Voice Project Test Runner"
	@echo ""
	@echo "Test Targets:"
	@echo "  make test-smoke       Quick smoke tests (<10s)"
	@echo "  make test-unit        Unit tests (<60s)"
	@echo "  make test-integration Integration tests (<5min)"
	@echo "  make test-quality     LLM-as-judge quality tests"
	@echo "  make test-stress      Stress tests (~10min)"
	@echo "  make test-all         All tests except stress"
	@echo "  make test-cpp         C++ unit tests (queue tests)"
	@echo "  make test-roundtrip   TTS->STT roundtrip verification"
	@echo ""
	@echo "Multilingual Tests (Phase 6-7):"
	@echo "  make test-multilingual-smoke    Fast: 1 sentence/language (11 langs)"
	@echo "  make test-multilingual          Full multilingual (EN/JA/ZH)"
	@echo "  make test-multilingual-english  English only (5 sentences)"
	@echo "  make test-multilingual-japanese Japanese only (5 sentences)"
	@echo "  make test-multilingual-chinese  Chinese only (5 sentences)"
	@echo "  make test-wer                   WER/CER aggregate reports"
	@echo ""
	@echo "Performance Tests:"
	@echo "  make test-latency-quick         Quick latency benchmark (~2min, for CI)"
	@echo "  make benchmark-nllb             Run NLLB translation benchmark"
	@echo "  make perf-smoke                 Multi-stream perf benchmark + GPU util"
	@echo "  make perf-snapshot              GPU performance snapshot tool"
	@echo ""
	@echo "Stress Tests (Phase 6, Worker #117):"
	@echo "  make test-stress                All stress tests (~10min)"
	@echo "  make test-stress-concurrent     10 concurrent streams"
	@echo "  make test-stress-sequential     1000 sequential requests"
	@echo "  make test-stress-memory         Memory leak detection"
	@echo "  make test-stress-latency        Latency P99 measurement"
	@echo "  make test-stress-report         Summary report"
	@echo ""
	@echo "Pipeline Integration Tests (Phase 6, Worker #118):"
	@echo "  make test-pipeline              Full pipeline tests"
	@echo "  make test-daemon                Daemon mode tests"
	@echo "  make test-streaming             Streaming mode tests"
	@echo "  make test-errors                Error handling tests"
	@echo ""
	@echo "Build Targets:"
	@echo "  make check-deps       Check dependencies (PyTorch >= 2.9.1)"
	@echo "  make build            Build C++ binaries (Release mode, optimized)"
	@echo "  make build-debug      Build C++ binaries (Debug mode)"
	@echo "  make build-lto        Build C++ binaries (Release + Link-Time Optimization)"
	@echo "  make validate         Build + validate models + smoke tests"
	@echo "  make validate-models    Quick model check (existence + size)"
	@echo "  make verify-models-full Full model verification (with hashes)"
	@echo "  make clean            Clean build artifacts"
	@echo ""
	@echo "Git Hook Targets:"
	@echo "  make install-hooks    Install git hooks"
	@echo ""
	@echo "Submodule Management:"
	@echo "  make sync-submodules  Check/sync submodules to recorded SHAs"

# =============================================================================
# Test Targets
# =============================================================================

test-smoke:
	@echo "=== Smoke Tests (<10s) ==="
	$(PYTEST) $(TESTS_DIR)/smoke -v -m "unit or integration" 2>/dev/null || $(PYTEST) $(TESTS_DIR)/smoke -v

test-unit:
	@echo "=== Unit Tests (<60s) ==="
	$(PYTEST) $(TESTS_DIR) -v -m unit 2>/dev/null || $(PYTEST) $(TESTS_DIR) -v -m unit
	@$(MAKE) test-cpp

test-integration:
	@echo "=== Integration Tests (<5min) ==="
	$(PYTEST) $(TESTS_DIR) -v -m integration 2>/dev/null || $(PYTEST) $(TESTS_DIR) -v -m integration

test-quality:
	@echo "=== Quality Tests (LLM Judge) ==="
	$(PYTEST) $(TESTS_DIR) -v -m quality 2>/dev/null || $(PYTEST) $(TESTS_DIR)/quality -v

test-stress:
	@echo "=== Stress Tests (~10min) ==="
	$(PYTEST) $(TESTS_DIR)/stress -v -m stress

# Granular stress test targets (Phase 6, Worker #117)
test-stress-concurrent:
	@echo "=== Stress: Concurrent Streams (10 parallel) ==="
	$(PYTEST) $(TESTS_DIR)/stress/test_stress.py -v -k "TestConcurrentStreams"

test-stress-sequential:
	@echo "=== Stress: Sequential Throughput (1000 requests) ==="
	$(PYTEST) $(TESTS_DIR)/stress/test_stress.py -v -k "TestSequentialThroughput"

test-stress-memory:
	@echo "=== Stress: Memory Leak Detection ==="
	$(PYTEST) $(TESTS_DIR)/stress/test_stress.py -v -k "TestMemoryLeakDetection"

test-stress-latency:
	@echo "=== Stress: Latency P99 Measurement ==="
	$(PYTEST) $(TESTS_DIR)/stress/test_stress.py -v -k "TestLatencyP99"

test-stress-report:
	@echo "=== Stress: Summary Report ==="
	$(PYTEST) $(TESTS_DIR)/stress/test_stress.py -v -k "test_generate_stress_report"

test-all: test-unit test-integration
	@echo ""
	@echo "=== ALL TESTS COMPLETE ==="

test-cpp:
	@echo "=== C++ Unit Tests ==="
	@if [ -d "$(CPP_BUILD_DIR)" ]; then \
		echo "Running lock-free queue tests..."; \
		$(CPP_BUILD_DIR)/test_lock_free_queue && \
		echo "" && \
		echo "Running speech queue tests..."; \
		$(CPP_BUILD_DIR)/test_speech_queue && \
		echo "" && \
		echo "Running TTS queue tests..."; \
		$(CPP_BUILD_DIR)/test_tts_queue; \
	else \
		echo "Error: C++ build directory not found. Run 'make build' first."; \
		exit 1; \
	fi

# =============================================================================
# Roundtrip Test (TTS -> STT verification)
# =============================================================================

test-roundtrip:
	@echo "=== TTS->STT Roundtrip Tests ==="
	@if [ -f "stream-tts-cpp/scripts/tts_stt_roundtrip_test.sh" ]; then \
		./stream-tts-cpp/scripts/tts_stt_roundtrip_test.sh; \
	else \
		echo "Roundtrip test script not found"; \
		exit 1; \
	fi

# =============================================================================
# Multilingual Round-Trip Tests (Phase 6)
# =============================================================================

test-multilingual:
	@echo "=== Multilingual Round-Trip Tests (EN/JA/ZH) ==="
	$(PYTEST) $(TESTS_DIR)/integration/test_multilingual_roundtrip.py -v -m multilingual 2>/dev/null || \
	$(PYTEST) $(TESTS_DIR)/integration/test_multilingual_roundtrip.py -v

test-multilingual-english:
	@echo "=== English Round-Trip Tests ==="
	$(PYTEST) $(TESTS_DIR)/integration/test_multilingual_roundtrip.py -v -k "english"

test-multilingual-japanese:
	@echo "=== Japanese Round-Trip Tests ==="
	$(PYTEST) $(TESTS_DIR)/integration/test_multilingual_roundtrip.py -v -k "japanese"

test-multilingual-chinese:
	@echo "=== Chinese Round-Trip Tests ==="
	$(PYTEST) $(TESTS_DIR)/integration/test_multilingual_roundtrip.py -v -k "chinese"

test-wer:
	@echo "=== WER/CER Measurement Tests ==="
	$(PYTEST) $(TESTS_DIR)/integration/test_multilingual_roundtrip.py -v -k "aggregate"

test-multilingual-smoke:
	@echo "=== Fast Multilingual Smoke (1 sentence/language, 11 languages) ==="
	$(PYTEST) $(TESTS_DIR)/integration/test_multilingual_smoke.py -v

# =============================================================================
# Pipeline Integration Tests (Phase 6, Worker #118)
# =============================================================================

test-pipeline:
	@echo "=== Full Pipeline Tests ==="
	$(PYTEST) $(TESTS_DIR)/integration/test_pipeline_integration.py -v -k "TestFullPipeline"

test-daemon:
	@echo "=== Daemon Mode Tests ==="
	$(PYTEST) $(TESTS_DIR)/integration/test_pipeline_integration.py -v -k "TestDaemonMode"

test-streaming:
	@echo "=== Streaming Mode Tests ==="
	$(PYTEST) $(TESTS_DIR)/integration/test_pipeline_integration.py -v -k "TestStreamingMode"

test-errors:
	@echo "=== Error Handling Tests ==="
	$(PYTEST) $(TESTS_DIR)/integration/test_pipeline_integration.py -v -k "TestErrorHandling"

# =============================================================================
# Dependency Checking
# =============================================================================

# MF1: Check that PyTorch >= 2.9.1 is installed (required for native MPS support)
check-deps:
	@echo "=== Checking Dependencies ==="
	@python3 -c "import torch; v = torch.__version__.split('+')[0]; parts = list(map(int, v.split('.')[:3])); \
		assert (parts[0], parts[1], parts[2] if len(parts) > 2 else 0) >= (2, 9, 1), \
		f'PyTorch >= 2.9.1 required for native MPS support, found {v}'; \
		print(f'  PyTorch: {v} (OK)')" || (echo "ERROR: PyTorch >= 2.9.1 required. Install with: pip install torch>=2.9.1" && exit 1)
	@python3 -c "import torchaudio; print(f'  torchaudio: {torchaudio.__version__} (OK)')" 2>/dev/null || echo "  torchaudio: not installed (optional)"
	@echo "=== Dependencies OK ==="

# =============================================================================
# Build Targets
# =============================================================================

build: check-deps
	@echo "=== Building C++ binaries (Release mode) ==="
	@mkdir -p $(CPP_BUILD_DIR)
	cd $(CPP_BUILD_DIR) && cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_LIBTORCH=ON \
		-DCMAKE_PREFIX_PATH="$$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)' 2>/dev/null || echo '')" \
		&& make -j$$(sysctl -n hw.ncpu 2>/dev/null || echo 4)

build-debug: check-deps
	@echo "=== Building C++ binaries (Debug mode) ==="
	@mkdir -p $(CPP_BUILD_DIR)
	cd $(CPP_BUILD_DIR) && cmake .. -DCMAKE_BUILD_TYPE=Debug -DUSE_LIBTORCH=ON \
		-DCMAKE_PREFIX_PATH="$$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)' 2>/dev/null || echo '')" \
		&& make -j$$(sysctl -n hw.ncpu 2>/dev/null || echo 4)

build-lto: check-deps
	@echo "=== Building C++ binaries (Release + LTO) ==="
	@mkdir -p $(CPP_BUILD_DIR)
	cd $(CPP_BUILD_DIR) && cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_LIBTORCH=ON \
		-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON \
		-DCMAKE_PREFIX_PATH="$$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)' 2>/dev/null || echo '')" \
		&& make -j$$(sysctl -n hw.ncpu 2>/dev/null || echo 4)

clean:
	@echo "=== Cleaning build artifacts ==="
	rm -rf $(CPP_BUILD_DIR)
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# =============================================================================
# Git Hooks
# =============================================================================

install-hooks:
	@echo "=== Installing git hooks ==="
	git config core.hooksPath .githooks
	chmod +x .githooks/*
	@echo "Git hooks installed. Pre-commit will run smoke tests."

# =============================================================================
# CI/Pre-commit Targets
# =============================================================================

# Pre-commit: must be fast (<10s)
pre-commit: test-smoke

# Pre-push: thorough (<5min)
pre-push: test-unit test-integration
	@echo "All pre-push tests passed."

# =============================================================================
# Quick Validation (build + smoke + roundtrip)
# =============================================================================

# Build, validate models, and run multilingual roundtrip in one command
# Use this after pulling changes or modifying C++ code
validate: build validate-models test-smoke test-multilingual-smoke
	@echo ""
	@echo "=== VALIDATION COMPLETE ==="
	@echo "- Build: OK"
	@echo "- Models: OK"
	@echo "- Smoke tests: OK"
	@echo "- Multilingual TTS: OK"

# Quick model validation (check required files exist)
validate-models:
	@echo "=== Validating Model Assets ==="
	@test -f stream-tts-cpp/build/stream-tts-cpp || (echo "ERROR: stream-tts-cpp binary missing - run 'make build'" && exit 1)
	@$(PYTHON) scripts/verify_model_assets.py --skip-hash
	@echo "All required models and binaries verified."

# Full model verification with hash checks (slower, for pre-perf-test validation)
verify-models-full:
	@echo "=== Full Model Verification (with hashes) ==="
	@$(PYTHON) scripts/verify_model_assets.py

# =============================================================================
# Submodule Management
# =============================================================================

sync-submodules:
	@echo "=== Checking Submodule Status ==="
	@./scripts/sync_submodules.sh

# =============================================================================
# Performance Benchmarks (Repo Audit #12, #17, #20)
# =============================================================================

test-latency-quick:
	@echo "=== Quick Latency Benchmark (~2min) ==="
	$(PYTEST) $(TESTS_DIR)/integration/test_latency_benchmark.py -v

# Repo Audit #17: NLLB Translation Benchmark (CI latency tracking)
benchmark-nllb:
	@echo "=== NLLB Translation Benchmark ==="
	@if [ -f "$(CPP_BUILD_DIR)/benchmark_nllb" ]; then \
		$(CPP_BUILD_DIR)/benchmark_nllb; \
	else \
		echo "Error: benchmark_nllb not built. Run 'make build' first."; \
		exit 1; \
	fi

# Repo Audit #20: Performance Smoke Test (release build + multi-stream benchmark)
perf-smoke:
	@echo "=== Performance Smoke Test ==="
	@echo "Step 1: Verify Release build exists..."
	@if [ ! -f "$(CPP_BUILD_DIR)/stream-tts-cpp" ]; then \
		echo "Building in Release mode..."; \
		$(MAKE) build; \
	fi
	@echo ""
	@echo "Step 2: Running multi-stream performance benchmark..."
	$(PYTEST) $(TESTS_DIR)/performance/test_perf_smoke.py -v -s
	@echo ""
	@echo "=== Performance Smoke Test Complete ==="

# Run perf snapshot tool (GPU timing capture)
perf-snapshot:
	@echo "=== GPU Performance Snapshot ==="
	@if [ -f "$(CPP_BUILD_DIR)/perf_snapshot" ]; then \
		$(CPP_BUILD_DIR)/perf_snapshot; \
	else \
		echo "Error: perf_snapshot not built. Run 'make build' first."; \
		exit 1; \
	fi
