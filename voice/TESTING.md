# Testing Guide - Voice TTS System

This guide covers all available tests for the Voice TTS system.

---

## Quick Start

### Install Git Hooks (Recommended)
```bash
make install-hooks
```
This installs pre-commit and pre-push hooks that automatically run tests.

### Run Smoke Tests (<10s)
```bash
make test-smoke
```
Quick validation that the system is functional.

### Wrapper Scripts (delegate to pytest)
The legacy shell entry points now call the unified pytest/Makefile targets:
- `./smoke_test.sh` → `make test-smoke` + golden audio quality check + multilingual smoke
- `./test_integration.sh` → `make test-unit`, `make test-integration`, `make test-wer`, optional `make test-quality`
- `./test_tts_working.sh` → `make test-multilingual-smoke`

### Run All Unit Tests (<60s)
```bash
make test-unit
```
Python unit tests + C++ queue tests.

---

## Test Framework

### Pytest-Based Testing
Tests are organized using pytest with markers:

```bash
# Smoke tests only
make test-smoke

# Unit tests (fast, isolated)
make test-unit

# Integration tests (slower, full pipeline)
make test-integration

# Quality tests (LLM-as-judge, requires API)
make test-quality

# Stress tests (long-running)
make test-stress

# All tests except stress
make test-all
```

### Directory Structure
```
tests/
├── conftest.py          # Pytest fixtures
├── pytest.ini           # Pytest configuration
├── smoke/               # Quick validation tests
│   └── test_smoke.py    # Binary, model, config checks
├── unit/                # Fast unit tests
│   └── python/          # Python unit tests
├── integration/         # Full pipeline tests
├── quality/             # LLM-as-judge tests
└── stress/              # Load and stress tests
```

### Test Markers
```python
@pytest.mark.unit           # Fast unit tests
@pytest.mark.integration    # Integration tests
@pytest.mark.quality        # LLM quality tests
@pytest.mark.stress         # Stress tests
@pytest.mark.slow           # Tests taking >5s
@pytest.mark.requires_binary   # Needs compiled binary
@pytest.mark.requires_models   # Needs model files
```

---

## Git Hooks

### Pre-commit Hook
Runs automatically before every commit:
- Smoke tests (8 tests, <3s)
- C++ queue tests (if queue files changed)
- Integration tests (if CLAUDE.md claims WORKING)

Skip with: `git commit --no-verify`

### Pre-push Hook
Runs automatically before every push:
- Full unit test suite (Python + C++)
- Integration tests

Skip with: `git push --no-verify`

### Installing Hooks
```bash
make install-hooks
```

This sets `core.hooksPath` to `.githooks/`.

---

## C++ Tests

### Queue Tests
```bash
make test-cpp
```

Runs:
- Lock-free queue tests (15 tests)
- Speech queue tests (12 tests)
- TTS queue tests (8 tests)

### Test Binaries
Located in `stream-tts-cpp/build/`:
- `test_lock_free_queue`
- `test_speech_queue`
- `test_tts_queue`

---

## Makefile Targets

| Target | Description | Time |
|--------|-------------|------|
| `test-smoke` | Quick validation | <10s |
| `test-unit` | Unit tests (Python + C++) | <60s |
| `test-integration` | Integration tests | <5min |
| `test-quality` | LLM-as-judge tests | ~2min |
| `test-stress` | Stress tests | ~10min |
| `test-all` | Unit + Integration | <6min |
| `test-cpp` | C++ queue tests only | <5s |
| `test-roundtrip` | TTS->STT verification | ~30s |

### Build Targets
```bash
make build        # Build C++ binaries
make clean        # Clean build artifacts
make install-hooks # Install git hooks
```

---

## Python Virtual Environment

Tests use the `.venv` Python environment:
```bash
# The Makefile automatically uses .venv/bin/pytest if available
make test-smoke

# Or activate manually
source .venv/bin/activate
pytest tests/smoke -v
```

---

## Test Configuration

### pytest.ini
```ini
[pytest]
markers =
    unit: Fast unit tests
    integration: Integration tests
    quality: LLM-as-judge quality tests
    stress: Long-running stress tests
    slow: Tests that take >5s
    requires_binary: Tests requiring compiled binary
    requires_models: Tests requiring model files
```

### conftest.py Fixtures
- `cpp_binary` - Path to stream-tts-cpp binary
- `test_kokoro_binary` - Path to test_kokoro_torchscript
- `kokoro_model_dir` - Path to Kokoro models
- `default_config` - Path to default.yaml
- `english_config` - Path to kokoro-mps-en.yaml
- `temp_wav_file` - Temporary WAV file for tests

---

## Performance Targets

| Metric | Target | Blocker |
|--------|--------|---------|
| Smoke tests | <10s | Yes |
| Unit tests | <60s | Yes |
| Integration tests | <5min | Yes |
| TTS warm latency | <200ms | Yes |
| TTS cold latency | <2s | No |

---

## TTS->STT Roundtrip Tests

Verify TTS output matches expected text:
```bash
make test-roundtrip
```

Or run directly:
```bash
./stream-tts-cpp/scripts/tts_stt_roundtrip_test.sh
```

---

## Troubleshooting

### Tests Not Found
```bash
# Check pytest is in venv
.venv/bin/pytest --version

# Check test discovery
.venv/bin/pytest tests/ --collect-only
```

### C++ Tests Fail to Build
```bash
make build
# Or
cd stream-tts-cpp/build && cmake .. && make
```

### Hooks Not Running
```bash
# Verify hooks path
git config core.hooksPath

# Should show: .githooks
# If not, run: make install-hooks
```

### Model Files Missing
Smoke tests verify model files exist. If missing:
```bash
# Check model paths
ls models/kokoro/
ls models/kokoro/voices/
```

---

## Continuous Testing

### Before Committing
Run automatically via pre-commit hook, or manually:
```bash
make test-smoke
```

### Before Pushing
Run automatically via pre-push hook, or manually:
```bash
make test-unit && make test-integration
```

### Full Validation
```bash
make test-all
```

---

## Contributing Tests

When adding new tests:
1. Place in appropriate directory (`unit/`, `integration/`, etc.)
2. Add pytest markers (`@pytest.mark.unit`, etc.)
3. Use fixtures from `conftest.py`
4. Update this guide if user-facing

### Test Template
```python
import pytest

@pytest.mark.unit
def test_feature_works(cpp_binary):
    """Test that feature X works correctly."""
    assert cpp_binary.exists()
    # Test logic here
```

---

## License

Copyright 2025 Andrew Yates. All rights reserved.
