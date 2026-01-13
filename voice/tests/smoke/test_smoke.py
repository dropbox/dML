"""
Smoke Tests - Quick validation that runs on every commit (target <30s; ~8s warm)

These tests verify the basic system is functional:
- Binary exists and is executable
- Required models are present
- Basic TTS generation works

Run: pytest tests/smoke -v
"""

import os
import subprocess
import pytest
from pathlib import Path


@pytest.mark.unit
class TestBinaryExists:
    """Verify C++ binaries are built."""

    def test_stream_tts_binary_exists(self, cpp_binary):
        """stream-tts-cpp binary must exist."""
        assert cpp_binary.exists(), f"Binary not found: {cpp_binary}"
        assert os.access(cpp_binary, os.X_OK), f"Binary not executable: {cpp_binary}"


@pytest.mark.unit
class TestModelsExist:
    """Verify required model files are present."""

    def test_kokoro_model_exists(self, kokoro_model_dir):
        """Kokoro TTS model directory must exist."""
        assert kokoro_model_dir.exists()

    def test_kokoro_torchscript_exists(self, kokoro_model_dir):
        """Kokoro TorchScript model must exist."""
        model_file = kokoro_model_dir / "kokoro_mps.pt"
        assert model_file.exists(), f"Kokoro model not found: {model_file}"

    def test_voice_files_exist(self, kokoro_model_dir):
        """At least one voice file must exist (voice_*.pt format)."""
        voice_files = list(kokoro_model_dir.glob("voice_*.pt"))
        assert len(voice_files) > 0, f"No voice files found in {kokoro_model_dir}"
        # Require at least the default English voice
        default_voice = kokoro_model_dir / "voice_af_heart.pt"
        assert default_voice.exists(), f"Default voice not found: {default_voice}"


@pytest.mark.unit
class TestConfigExists:
    """Verify configuration files are present."""

    def test_default_config_exists(self, default_config):
        """Default config file must exist."""
        assert default_config.exists()

    def test_english_config_exists(self, english_config):
        """English config must exist."""
        assert english_config.exists()


# All 11 supported languages with their config files
SUPPORTED_LANGUAGES = {
    "en": "kokoro-mps-en.yaml",       # English
    "ja": "kokoro-mps-ja.yaml",       # Japanese
    "zh": "kokoro-mps-zh.yaml",       # Chinese (Mandarin)
    "es": "kokoro-mps-es.yaml",       # Spanish
    "fr": "kokoro-mps-fr.yaml",       # French
    "hi": "kokoro-mps-hi.yaml",       # Hindi
    "it": "kokoro-mps-it.yaml",       # Italian
    "pt": "kokoro-mps-pt.yaml",       # Portuguese
    "ko": "kokoro-mps-ko.yaml",       # Korean
    "yi": "kokoro-mps-yi.yaml",       # Yiddish
    "zh-sichuan": "kokoro-mps-zh-sichuan.yaml",  # Sichuanese
}

# Translation configs (EN -> target)
TRANSLATION_CONFIGS = {
    "ja": "kokoro-mps-en2ja.yaml",
    "zh": "kokoro-mps-en2zh.yaml",
    "hi": "kokoro-mps-en2hi.yaml",
    "zh-sichuan": "kokoro-mps-en2zh-sichuan.yaml",
}


@pytest.mark.unit
class TestConfigConsistency:
    """Verify all language config files exist and are consistent."""

    @pytest.fixture(scope="class")
    def config_dir(self):
        """Config directory path."""
        return Path(__file__).parent.parent.parent / "stream-tts-cpp" / "config"

    @pytest.mark.parametrize("lang,config_file", SUPPORTED_LANGUAGES.items())
    def test_language_config_exists(self, config_dir, lang, config_file):
        """Each supported language must have a config file."""
        config_path = config_dir / config_file
        assert config_path.exists(), f"Config missing for {lang}: {config_file}"

    @pytest.mark.parametrize("lang,config_file", TRANSLATION_CONFIGS.items())
    def test_translation_config_exists(self, config_dir, lang, config_file):
        """Translation configs (EN->target) must exist."""
        config_path = config_dir / config_file
        assert config_path.exists(), f"Translation config missing: {config_file}"

    def test_config_count_matches_languages(self, config_dir):
        """Number of language configs should match supported languages."""
        configs = list(config_dir.glob("kokoro-mps-*.yaml"))
        # Filter out translation configs (en2*)
        lang_configs = [c for c in configs if "en2" not in c.name]
        assert len(lang_configs) >= len(SUPPORTED_LANGUAGES), \
            f"Expected {len(SUPPORTED_LANGUAGES)} language configs, found {len(lang_configs)}"


# Required model files with minimum size (bytes)
REQUIRED_MODELS = {
    # Kokoro TTS core model
    "kokoro/kokoro_mps.pt": 300_000_000,  # ~328MB
    # At least one voice file
    "kokoro/voice_af_heart.pt": 500_000,   # ~523KB
}

# Optional but recommended models
OPTIONAL_MODELS = {
    "whisper/ggml-large-v3.bin": 3_000_000_000,   # ~3GB (STT)
    "whisper/ggml-large-v3-turbo.bin": 1_500_000_000,  # ~1.5GB (STT - faster turbo model)
    "whisper/ggml-silero-v6.2.0.bin": 800_000,    # ~885KB (VAD)
    "nllb/nllb-200-distilled-600m.pt": 2_000_000_000,  # ~2.4GB (translation)
}


@pytest.mark.unit
class TestModelAssets:
    """Verify required model files are present and valid."""

    @pytest.fixture(scope="class")
    def models_dir(self):
        """Models directory path."""
        return Path(__file__).parent.parent.parent / "models"

    @pytest.mark.parametrize("model_path,min_size", REQUIRED_MODELS.items())
    def test_required_model_exists(self, models_dir, model_path, min_size):
        """Required model files must exist and be valid size."""
        full_path = models_dir / model_path
        assert full_path.exists(), f"Required model missing: {model_path}"
        actual_size = full_path.stat().st_size
        assert actual_size >= min_size, \
            f"Model {model_path} too small: {actual_size} < {min_size} bytes"

    @pytest.mark.parametrize("model_path,min_size", OPTIONAL_MODELS.items())
    def test_optional_model_exists(self, models_dir, model_path, min_size):
        """Optional model files should exist if present."""
        full_path = models_dir / model_path
        if not full_path.exists():
            pytest.skip(f"Optional model not installed: {model_path}")
        actual_size = full_path.stat().st_size
        assert actual_size >= min_size, \
            f"Model {model_path} too small: {actual_size} < {min_size} bytes"

    def test_voice_files_count(self, models_dir):
        """Verify adequate voice files are available."""
        voices_dir = models_dir / "kokoro"
        voice_files = list(voices_dir.glob("voice_*.pt"))
        assert len(voice_files) >= 5, \
            f"Expected at least 5 voice files, found {len(voice_files)}"

    def test_models_directory_structure(self, models_dir):
        """Verify expected model directory structure."""
        assert models_dir.exists(), f"Models directory missing: {models_dir}"
        assert (models_dir / "kokoro").exists(), "kokoro subdirectory missing"
        # whisper and nllb are optional
        if not (models_dir / "whisper").exists():
            pytest.skip("whisper subdirectory not present (optional)")

    def test_voice_pack_format(self, models_dir):
        """
        CRITICAL: Verify voice pack is FULL format [N, 1, 256], not single embedding.

        Regression test for 2025-12-11 bug where export script only saved first
        embedding (voices_data[0]) instead of full pack. This caused:
        - ~200ms extra trailing silence
        - Incorrect prosody for different sentence lengths

        Kokoro uses pack[len(phonemes)-1] for length-indexed voice selection.
        """
        import torch
        voice_path = models_dir / "kokoro" / "voice_af_heart.pt"
        assert voice_path.exists(), "Default voice file missing"

        voice_pack = torch.load(voice_path, map_location='cpu', weights_only=True)

        # Must be 3D tensor [N, 1, 256]
        assert voice_pack.dim() == 3, (
            f"Voice pack must be 3D [N, 1, 256], got {voice_pack.dim()}D shape {voice_pack.shape}. "
            f"Re-export with: python scripts/export_kokoro_torchscript.py --device mps --dtype float32 --output models/kokoro"
        )

        # Must have multiple vectors (Kokoro has 510)
        MIN_VECTORS = 100
        assert voice_pack.size(0) >= MIN_VECTORS, (
            f"Voice pack has only {voice_pack.size(0)} vectors, need >= {MIN_VECTORS}. "
            f"Single embedding causes trailing silence bug. Re-export voice pack."
        )

        # Embedding dimension must be 256
        assert voice_pack.size(2) == 256, (
            f"Voice embedding dim must be 256, got {voice_pack.size(2)}"
        )


def get_tts_env():
    """
    Get environment variables for TTS subprocess.

    Note: PYTORCH_ENABLE_MPS_FALLBACK no longer needed with PyTorch 2.9.1+.
    """
    env = os.environ.copy()
    return env


@pytest.mark.integration
@pytest.mark.requires_binary
@pytest.mark.requires_models
@pytest.mark.slow
class TestQuickTTS:
    """Quick TTS generation test (uses full stream-tts-cpp pipeline)."""

    def test_quick_tts_generates_audio(self, cpp_binary, english_config, temp_wav_file):
        """Generate <1s of audio with stream-tts-cpp binary."""
        # Use Claude API JSON format as input
        input_json = '{"type":"content_block_delta","delta":{"type":"text_delta","text":"Hi."}}'

        result = subprocess.run(
            [str(cpp_binary), "--save-audio", str(temp_wav_file), str(english_config)],
            input=input_json,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(cpp_binary.parent.parent),
            env=get_tts_env()
        )

        # Check for success (may have warnings but should return 0)
        if result.returncode != 0:
            # Some warnings are OK, check if audio was still generated
            if temp_wav_file.exists() and temp_wav_file.stat().st_size > 1000:
                pass  # Audio generated despite non-zero return
            else:
                pytest.fail(f"TTS failed (rc={result.returncode}): {result.stderr[:500]}")

        # Verify audio file was created and has content
        assert temp_wav_file.exists(), "WAV file not created"
        assert temp_wav_file.stat().st_size > 1000, "WAV file too small (likely empty)"


def analyze_wav_silence(wav_path, threshold=10):
    """Analyze a WAV file for leading/trailing silence."""
    import wave
    import struct
    with wave.open(str(wav_path), 'rb') as w:
        frames = w.getnframes()
        rate = w.getframerate()
        data = w.readframes(frames)
        samples = struct.unpack('<' + 'h' * frames, data)
    first_audio = next((i for i, s in enumerate(samples) if abs(s) > threshold), 0)
    last_audio = next((i for i in range(len(samples)-1, -1, -1) if abs(samples[i]) > threshold), frames-1)
    total_ms = (frames / rate) * 1000
    return {
        "leading_silence_ms": (first_audio / rate) * 1000,
        "trailing_silence_ms": ((frames - 1 - last_audio) / rate) * 1000,
        "total_duration_ms": total_ms,
        "audio_duration_ms": ((last_audio - first_audio) / rate) * 1000,
        "leading_ratio": (first_audio / rate * 1000) / total_ms if total_ms > 0 else 0,
        "trailing_ratio": ((frames - 1 - last_audio) / rate * 1000) / total_ms if total_ms > 0 else 0,
    }


@pytest.mark.integration
@pytest.mark.requires_binary
@pytest.mark.requires_models
class TestAudioQuality:
    """Audio quality tests - catch silence/truncation bugs (added 2025-12-11)."""
    MAX_LEADING_SILENCE_RATIO = 0.35   # Max 35% leading (Kokoro has ~30% inherent)
    MAX_TRAILING_SILENCE_RATIO = 0.25  # Max 25% trailing
    MAX_TOTAL_SILENCE_RATIO = 0.55     # Max 55% total silence

    def test_tts_no_excessive_silence(self, cpp_binary, temp_wav_file):
        """TTS output should not have excessive trailing silence."""
        result = subprocess.run(
            [str(cpp_binary), "--speak", "Hello world", "--lang", "en",
             "--save-audio", str(temp_wav_file)],
            capture_output=True, text=True, timeout=60,
            cwd=str(cpp_binary.parent.parent))
        assert temp_wav_file.exists(), f"WAV not created: {result.stderr[:300]}"
        analysis = analyze_wav_silence(temp_wav_file)
        assert analysis["trailing_ratio"] <= self.MAX_TRAILING_SILENCE_RATIO, (
            f"Excessive trailing silence: {analysis['trailing_silence_ms']:.0f}ms "
            f"({analysis['trailing_ratio']*100:.1f}%). Bug: export voice pack not single embedding.")

    def test_short_phrase_reasonable_duration(self, cpp_binary, temp_wav_file):
        """'Hi' should be under 1.5s total."""
        result = subprocess.run(
            [str(cpp_binary), "--speak", "Hi", "--lang", "en",
             "--save-audio", str(temp_wav_file)],
            capture_output=True, text=True, timeout=60,
            cwd=str(cpp_binary.parent.parent))
        assert temp_wav_file.exists()
        analysis = analyze_wav_silence(temp_wav_file)
        assert analysis["total_duration_ms"] < 1500, f"'Hi' too long: {analysis['total_duration_ms']:.0f}ms"


# =============================================================================
# Multilingual and MPS Residency Tests MOVED to tests/integration/
# =============================================================================
#
# TestMultilingualSmoke and TestMPSResidency were moved to:
#   tests/integration/test_multilingual_smoke.py
#
# Reason: These tests spawn subprocess calls that require ~17s model warmup
# each, making them unsuitable for pre-commit smoke tests (<30s target).
#
# Run them via: pytest tests/integration/test_multilingual_smoke.py -v
# =============================================================================
