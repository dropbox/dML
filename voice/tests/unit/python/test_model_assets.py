#!/usr/bin/env python3
"""
Model Asset Verification Tests

Run before performance tests to ensure model files are correct.
Prevents false regressions from corrupted/wrong model versions.

Usage:
    pytest tests/unit/python/test_model_assets.py -v
    pytest tests/unit/python/test_model_assets.py -v -k "not hash"  # Skip slow hash checks

Worker #222 - Audit Item #18: Model asset verifier integration
"""

import sys
import pytest
from pathlib import Path

# Add scripts to path for importing
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "scripts"))

from verify_model_assets import (
    verify_all_models,
    verify_category,
    CRITICAL_MODELS,
    MODELS_DIR,
)


class TestModelAssetVerification:
    """Test model files exist with correct sizes and hashes."""

    @pytest.mark.unit
    def test_kokoro_models_exist(self):
        """Kokoro TTS models must exist."""
        passed, messages = verify_category("kokoro", MODELS_DIR)
        # Check existence only (don't verify hash in fast test)
        for asset in CRITICAL_MODELS["kokoro"]:
            assert (MODELS_DIR / asset.path).exists(), f"Missing: {asset.path}"

    @pytest.mark.unit
    def test_nllb_models_exist(self):
        """NLLB translation models must exist."""
        passed, messages = verify_category("nllb", MODELS_DIR)
        for asset in CRITICAL_MODELS["nllb"]:
            assert (MODELS_DIR / asset.path).exists(), f"Missing: {asset.path}"

    @pytest.mark.unit
    def test_whisper_models_exist(self):
        """Whisper STT models must exist."""
        passed, messages = verify_category("whisper", MODELS_DIR)
        for asset in CRITICAL_MODELS["whisper"]:
            assert (MODELS_DIR / asset.path).exists(), f"Missing: {asset.path}"

    @pytest.mark.unit
    def test_kokoro_model_sizes(self):
        """Kokoro models should have expected sizes."""
        for asset in CRITICAL_MODELS["kokoro"]:
            filepath = MODELS_DIR / asset.path
            if filepath.exists() and asset.size_bytes > 0:
                actual_size = filepath.stat().st_size
                assert actual_size == asset.size_bytes, \
                    f"{asset.path}: expected {asset.size_bytes}, got {actual_size}"

    @pytest.mark.unit
    def test_nllb_model_sizes(self):
        """NLLB models should have expected sizes."""
        for asset in CRITICAL_MODELS["nllb"]:
            filepath = MODELS_DIR / asset.path
            if filepath.exists() and asset.size_bytes > 0:
                actual_size = filepath.stat().st_size
                assert actual_size == asset.size_bytes, \
                    f"{asset.path}: expected {asset.size_bytes}, got {actual_size}"

    @pytest.mark.slow
    @pytest.mark.unit
    def test_all_models_hash_verification(self):
        """
        Full hash verification for all models.

        NOTE: This test is slow (~30s) because it reads all model files.
        Skip with: pytest -m "not slow"
        """
        passed, results = verify_all_models(MODELS_DIR)

        failed = []
        for category, messages in results.items():
            for msg in messages:
                if not msg.startswith("OK"):
                    failed.append(f"[{category}] {msg}")

        assert passed, f"Model verification failed:\n" + "\n".join(failed)


@pytest.mark.unit
def test_verify_model_script_importable():
    """Verify the model asset script can be imported."""
    from verify_model_assets import verify_all_models, generate_manifest
    assert callable(verify_all_models)
    assert callable(generate_manifest)
