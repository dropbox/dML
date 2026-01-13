"""
Model Validation Tests - Pre-flight checks for TTS models

These tests catch configuration and model issues BEFORE quality tests run.
They are designed to fail fast with clear error messages.

Critical Issue Caught (2025-12-05):
- Voice files had wrong shape [256] instead of [510, 1, 256]
- This caused audio to fade off after first 2 words
- Issue was masked because tests only checked "audio generated", not quality

Worker #117 - Test Infrastructure Hardening
"""

import os
import pytest
import subprocess
import torch
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
KOKORO_DIR = MODELS_DIR / "kokoro"
STREAM_TTS_CPP = PROJECT_ROOT / "stream-tts-cpp"
BUILD_DIR = STREAM_TTS_CPP / "build"
CONFIG_DIR = STREAM_TTS_CPP / "config"


# =============================================================================
# Voice Pack Validation
# =============================================================================

class TestVoicePackValidation:
    """
    Validate voice pack files have correct format.

    Kokoro uses length-indexed voice embeddings: pack[len(phonemes)-1]
    If voice files are wrong shape, audio quality degrades for longer text.
    """

    EXPECTED_SHAPE = (510, 1, 256)  # [num_lengths, 1, embedding_dim]

    # All 54 Kokoro voice packs (complete set from HuggingFace hexgrad/Kokoro-82M)
    VOICE_FILES = [
        # American English (20 voices)
        "voice_af_alloy.pt", "voice_af_aoede.pt", "voice_af_bella.pt", "voice_af_heart.pt",
        "voice_af_jessica.pt", "voice_af_kore.pt", "voice_af_nicole.pt", "voice_af_nova.pt",
        "voice_af_river.pt", "voice_af_sarah.pt", "voice_af_sky.pt",
        "voice_am_adam.pt", "voice_am_echo.pt", "voice_am_eric.pt", "voice_am_fenrir.pt",
        "voice_am_liam.pt", "voice_am_michael.pt", "voice_am_onyx.pt", "voice_am_puck.pt",
        "voice_am_santa.pt",
        # British English (8 voices)
        "voice_bf_alice.pt", "voice_bf_emma.pt", "voice_bf_isabella.pt", "voice_bf_lily.pt",
        "voice_bm_daniel.pt", "voice_bm_fable.pt", "voice_bm_george.pt", "voice_bm_lewis.pt",
        # Spanish (3 voices)
        "voice_ef_dora.pt", "voice_em_alex.pt", "voice_em_santa.pt",
        # French (1 voice)
        "voice_ff_siwis.pt",
        # Hindi (4 voices)
        "voice_hf_alpha.pt", "voice_hf_beta.pt", "voice_hm_omega.pt", "voice_hm_psi.pt",
        # Italian (2 voices)
        "voice_if_sara.pt", "voice_im_nicola.pt",
        # Japanese (5 voices)
        "voice_jf_alpha.pt", "voice_jf_gongitsune.pt", "voice_jf_nezumi.pt",
        "voice_jf_tebukuro.pt", "voice_jm_kumo.pt",
        # Portuguese (3 voices)
        "voice_pf_dora.pt", "voice_pm_alex.pt", "voice_pm_santa.pt",
        # Chinese (8 voices)
        "voice_zf_xiaobei.pt", "voice_zf_xiaoni.pt", "voice_zf_xiaoxiao.pt", "voice_zf_xiaoyi.pt",
        "voice_zm_yunjian.pt", "voice_zm_yunxi.pt", "voice_zm_yunxia.pt", "voice_zm_yunyang.pt",
    ]

    @pytest.mark.parametrize("voice_file", VOICE_FILES)
    def test_voice_pack_shape(self, voice_file):
        """Each voice file must have shape [510, 1, 256] for length-indexed selection."""
        voice_path = KOKORO_DIR / voice_file

        if not voice_path.exists():
            pytest.skip(f"Voice file not found: {voice_path}")

        pack = torch.load(voice_path, weights_only=True, map_location="cpu")

        assert isinstance(pack, torch.Tensor), \
            f"{voice_file}: Expected torch.Tensor, got {type(pack)}"

        assert pack.shape == torch.Size(self.EXPECTED_SHAPE), \
            f"{voice_file}: Expected shape {self.EXPECTED_SHAPE}, got {tuple(pack.shape)}. " \
            f"Single [256] embedding causes audio fade-off for longer sentences. " \
            f"Fix: Download proper voice pack from HuggingFace hexgrad/Kokoro-82M"

    def test_all_voices_consistent(self):
        """All voice files should have the same shape."""
        shapes = {}
        for voice_file in self.VOICE_FILES:
            voice_path = KOKORO_DIR / voice_file
            if voice_path.exists():
                pack = torch.load(voice_path, weights_only=True, map_location="cpu")
                shapes[voice_file] = tuple(pack.shape)

        if len(shapes) < 2:
            pytest.skip("Need at least 2 voice files to check consistency")

        unique_shapes = set(shapes.values())
        assert len(unique_shapes) == 1, \
            f"Voice files have inconsistent shapes: {shapes}"


# =============================================================================
# TTS Stderr Warning Detection
# =============================================================================

class TestTTSWarningDetection:
    """
    Detect warnings in TTS stderr that indicate configuration issues.

    Critical warnings that should fail tests:
    - "Old voice format" - voice pack has wrong shape
    - "Failed to load" - model loading issues
    - "Truncating" - input too long (may indicate chunking issues)
    """

    CRITICAL_WARNINGS = [
        "Old voice format",      # Wrong voice pack shape
        "Failed to load",        # Model loading failure
        "Failed to initialize",  # Initialization failure
        "error",                 # Any error-level log
    ]

    # Warnings that are acceptable (informational only)
    ACCEPTABLE_WARNINGS = [
        "Truncating to 510",     # Expected for very long inputs
        "warmup",                # Warmup messages
    ]

    @pytest.fixture
    def tts_binary(self):
        binary = BUILD_DIR / "stream-tts-cpp"
        if not binary.exists():
            pytest.skip(f"TTS binary not found: {binary}")
        return binary

    @pytest.fixture
    def english_config(self):
        config = CONFIG_DIR / "kokoro-mps-en.yaml"
        if not config.exists():
            pytest.skip(f"Config not found: {config}")
        return config

    def test_no_critical_warnings_short_text(self, tts_binary, english_config):
        """Short text should produce no critical warnings."""
        result = subprocess.run(
            [str(tts_binary), "--speak", "Hello world", "--lang", "en"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(STREAM_TTS_CPP)
        )

        stderr_lower = result.stderr.lower()
        stdout_lower = result.stdout.lower()
        combined = stderr_lower + stdout_lower

        for warning in self.CRITICAL_WARNINGS:
            # Skip if it's in an acceptable context
            skip = any(acc.lower() in combined for acc in self.ACCEPTABLE_WARNINGS
                      if warning.lower() in combined)
            if skip:
                continue

            assert warning.lower() not in combined, \
                f"Critical warning '{warning}' found in TTS output. " \
                f"This indicates a configuration issue.\n" \
                f"STDERR: {result.stderr[:500]}\n" \
                f"STDOUT: {result.stdout[:500]}"

    def test_no_critical_warnings_long_text(self, tts_binary, english_config):
        """Long text (pangram) should produce no critical warnings."""
        result = subprocess.run(
            [str(tts_binary), "--speak",
             "The quick brown fox jumps over the lazy dog near the river bank",
             "--lang", "en"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(STREAM_TTS_CPP)
        )

        stderr_lower = result.stderr.lower()
        stdout_lower = result.stdout.lower()
        combined = stderr_lower + stdout_lower

        for warning in self.CRITICAL_WARNINGS:
            skip = any(acc.lower() in combined for acc in self.ACCEPTABLE_WARNINGS)
            if skip:
                continue

            assert warning.lower() not in combined, \
                f"Critical warning '{warning}' found in TTS output for long text. " \
                f"STDERR: {result.stderr[:500]}"

    def test_voice_pack_loaded_correctly(self, tts_binary, english_config):
        """Verify voice pack loads with correct shape (510 vectors)."""
        result = subprocess.run(
            [str(tts_binary), "--speak", "Test", "--lang", "en"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(STREAM_TTS_CPP)
        )

        combined = result.stdout + result.stderr

        # Should see "510 length-indexed vectors" NOT "Old voice format"
        assert "510 length-indexed vectors" in combined or "510, 1, 256" in combined, \
            f"Voice pack should load with 510 vectors. Got:\n{combined[:1000]}"

        assert "Old voice format" not in combined, \
            f"Voice pack loaded with wrong format (single [256] embedding). " \
            f"This causes audio fade-off. Fix voice files."


# =============================================================================
# Model File Existence
# =============================================================================

class TestModelFileExistence:
    """Verify required model files exist before running TTS tests."""

    REQUIRED_KOKORO_FILES = [
        "kokoro_mps.pt",        # Main model (MPS)
        "voice_af_heart.pt",    # Default English voice
    ]

    REQUIRED_LEXICON_FILES = [
        "lexicon/us_gold.json", # English lexicon
    ]

    def test_kokoro_model_exists(self):
        """Main Kokoro model must exist."""
        for filename in self.REQUIRED_KOKORO_FILES:
            path = KOKORO_DIR / filename
            assert path.exists(), \
                f"Required model file missing: {path}"

    def test_lexicon_exists(self):
        """Lexicon files must exist for proper G2P."""
        for filename in self.REQUIRED_LEXICON_FILES:
            path = KOKORO_DIR / filename
            assert path.exists(), \
                f"Required lexicon file missing: {path}"


# =============================================================================
# Configuration Validation
# =============================================================================

class TestConfigurationValidation:
    """Validate TTS configuration files."""

    def test_english_config_valid_yaml(self):
        """English config must be valid YAML."""
        import yaml

        config_path = CONFIG_DIR / "kokoro-mps-en.yaml"
        if not config_path.exists():
            pytest.skip(f"Config not found: {config_path}")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert "tts" in config, "Config missing 'tts' section"
        assert config["tts"].get("engine") == "kokoro-torchscript", \
            "Expected kokoro-torchscript engine"

    def test_config_voice_matches_available(self):
        """Config voice setting should match available voice files."""
        import yaml

        config_path = CONFIG_DIR / "kokoro-mps-en.yaml"
        if not config_path.exists():
            pytest.skip(f"Config not found: {config_path}")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        voice = config.get("tts", {}).get("voice", "en")

        # Map voice codes to files
        voice_map = {
            "en": "voice_af_heart.pt",
            "af_heart": "voice_af_heart.pt",
            "ja": "voice_jf_alpha.pt",
            "jf_alpha": "voice_jf_alpha.pt",
        }

        if voice in voice_map:
            voice_file = KOKORO_DIR / voice_map[voice]
            assert voice_file.exists(), \
                f"Config references voice '{voice}' but file missing: {voice_file}"
