#!/usr/bin/env python3
"""
Model Path Validation Tests

Verifies that all required model files exist BEFORE starting the TTS daemon.
This prevents cryptic failures when model paths are misconfigured.

Run: pytest tests/unit/python/test_model_paths.py -v
Or:  python tests/unit/python/test_model_paths.py

These tests should be run:
1. Before starting the daemon
2. In CI/CD before deployment
3. After changing model paths in config

Classification: UNIT TEST (fast, no model loading)

Note: StyleTTS2 tests removed (2025-12-05) - project now uses Kokoro TTS.
      Kokoro model tests are in tests/integration/test_model_validation.py
"""

import os
import sys
import pytest
import yaml


# Required files for each model
NLLB_REQUIRED_FILES = [
    "nllb-encoder-mps.pt",
    "nllb-decoder-mps.pt",
    "nllb-decoder-first-mps.pt",
    "nllb-decoder-kvcache-mps.pt",
    "sentencepiece.bpe.model",
]


def get_nllb_path():
    """Get NLLB model path from environment or defaults."""
    nllb_path = os.environ.get("NLLB_MODEL_PATH", "")

    if not nllb_path or not os.path.exists(nllb_path):
        common_nllb_paths = [
            os.path.expanduser("~/voice/models/nllb"),
            "/Users/ayates/voice/models/nllb",
        ]
        for path in common_nllb_paths:
            if os.path.exists(path):
                nllb_path = path
                break

    return nllb_path


def check_nllb_models(path: str) -> tuple[bool, list[str]]:
    """Check if NLLB model files exist."""
    if not path:
        return False, ["NLLB_MODEL_PATH not set"]

    if not os.path.exists(path):
        return False, [f"NLLB model directory does not exist: {path}"]

    missing = []
    for filename in NLLB_REQUIRED_FILES:
        filepath = os.path.join(path, filename)
        if not os.path.exists(filepath):
            missing.append(f"Missing: {filepath}")

    return len(missing) == 0, missing


# ========== PYTEST TESTS ==========


@pytest.mark.unit
def test_nllb_model_path_exists():
    """NLLB model directory must exist (for translation)."""
    nllb_path = get_nllb_path()
    assert nllb_path, "NLLB_MODEL_PATH not set"
    assert os.path.exists(nllb_path), f"NLLB model directory does not exist: {nllb_path}"


@pytest.mark.unit
def test_nllb_required_files_exist():
    """All required NLLB model files must exist."""
    nllb_path = get_nllb_path()
    success, errors = check_nllb_models(nllb_path)
    assert success, "\n".join(errors)


@pytest.mark.unit
def test_default_config_exists():
    """Default config file must exist."""
    config_paths = [
        os.path.expanduser("~/voice/stream-tts-cpp/config/default.yaml"),
        "/Users/ayates/voice/stream-tts-cpp/config/default.yaml",
    ]
    found = False
    for path in config_paths:
        if os.path.exists(path):
            found = True
            break
    assert found, f"No config found in: {config_paths}"


# ========== CLI RUNNER ==========

def main():
    """Run validation checks from CLI."""
    print("=" * 60)
    print("Model Path Validation (NLLB Translation)")
    print("=" * 60)

    nllb_path = get_nllb_path()

    print(f"\nNLLB_MODEL_PATH: {nllb_path or '(not set)'}")

    # Check NLLB
    print("\n--- NLLB Model Files ---")
    success, errors = check_nllb_models(nllb_path)
    if success:
        print("  All files present: OK")
        print("\n" + "=" * 60)
        print("RESULT: ALL CHECKS PASSED")
        sys.exit(0)
    else:
        for err in errors:
            print(f"  ERROR: {err}")
        print("\n" + "=" * 60)
        print("RESULT: VALIDATION FAILED")
        print("\nTranslation will NOT work with these settings.")
        print("Fix the model paths and run this check again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
