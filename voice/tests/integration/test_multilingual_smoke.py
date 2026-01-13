"""
Multilingual Smoke Tests - Moved from tests/smoke/ to tests/integration/

These tests were moved because they spawn 11+ TTS subprocess calls, each requiring
~17s model warmup. This makes them unsuitable for pre-commit smoke tests which
should complete in <30s.

These are still "smoke" tests in spirit (quick verification that each language
works) but belong in integration tests due to their runtime.

Run: pytest tests/integration/test_multilingual_smoke.py -v
"""

import os
import subprocess
import pytest
from pathlib import Path
import json


def get_tts_env():
    """
    Get environment variables for TTS subprocess.

    Note: PYTORCH_ENABLE_MPS_FALLBACK no longer needed with PyTorch 2.9.1+.
    """
    env = os.environ.copy()
    # Fix OpenMP duplicate library crash when llama.cpp + libtorch both link OpenMP
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    return env


# Test sentences for all 14 languages - short phrases for fast execution
MULTILINGUAL_TEST_SENTENCES = {
    # Kokoro TTS languages (MPS GPU)
    "en": {"text": "Hello world.", "config": "kokoro-mps-en.yaml"},
    "ja": {"text": "こんにちは。", "config": "kokoro-mps-ja.yaml"},
    "zh": {"text": "你好。", "config": "kokoro-mps-zh.yaml"},
    "es": {"text": "Hola mundo.", "config": "kokoro-mps-es.yaml"},
    "fr": {"text": "Bonjour monde.", "config": "kokoro-mps-fr.yaml"},
    "hi": {"text": "नमस्ते।", "config": "kokoro-mps-hi.yaml"},
    "it": {"text": "Ciao mondo.", "config": "kokoro-mps-it.yaml"},
    "pt": {"text": "Ola mundo.", "config": "kokoro-mps-pt.yaml"},
    "ko": {"text": "안녕하세요.", "config": "kokoro-mps-ko.yaml"},
    "yi": {"text": "שלום וועלט", "config": "kokoro-mps-yi.yaml"},
    "zh-sichuan": {"text": "你好。", "config": "kokoro-mps-zh-sichuan.yaml"},
    # MMS-TTS languages (CPU, RTF ~0.14)
    "ar": {"text": "مرحبا بالعالم", "config": "mms-tts-ar.yaml"},
    "tr": {"text": "Merhaba dünya.", "config": "mms-tts-tr.yaml"},
    "fa": {"text": "سلام دنیا", "config": "mms-tts-fa.yaml"},
}

def get_capabilities(cpp_binary):
    """Fetch capabilities JSON from the C++ binary."""
    result = subprocess.run(
        [str(cpp_binary), "--capabilities-json"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, f"Failed to get capabilities: {result.stderr}"
    return json.loads(result.stdout)


@pytest.mark.integration
@pytest.mark.requires_binary
@pytest.mark.requires_models
@pytest.mark.multilingual
class TestMultilingualSmoke:
    """
    Multilingual smoke test - verifies TTS works for all 14 languages.

    Languages tested:
    - Kokoro TTS (11): en, ja, zh, es, fr, hi, it, pt, ko, yi, zh-sichuan
    - MMS-TTS (3): ar, tr, fa

    This is a quick verification (1 sentence per language) to catch
    configuration issues or broken language pipelines before running
    full integration tests.

    NOTE: Moved from tests/smoke/ because each language requires a separate
    process with ~17s model warmup. Total runtime: ~4 minutes.
    """

    @pytest.fixture(scope="class")
    def config_dir(self):
        """Config directory path."""
        return Path(__file__).parent.parent.parent / "stream-tts-cpp" / "config"

    @pytest.mark.parametrize("lang,test_data", MULTILINGUAL_TEST_SENTENCES.items())
    def test_language_generates_audio(self, cpp_binary, config_dir, lang, test_data, tmp_path):
        """
        Verify each language can generate audio.

        This is a smoke test - it only verifies:
        1. TTS completes without crashing
        2. Audio file is created
        3. Audio file has content (>1KB)

        It does NOT verify audio quality - use integration tests for that.
        """
        config_path = config_dir / test_data["config"]
        assert config_path.exists(), f"Config not found: {config_path}"

        text = test_data["text"]
        wav_file = tmp_path / f"smoke_{lang}.wav"

        # Create JSON input in Claude API format
        escaped_text = text.replace('"', '\\"')
        input_json = f'{{"type":"content_block_delta","delta":{{"type":"text_delta","text":"{escaped_text}"}}}}'

        result = subprocess.run(
            [str(cpp_binary), "--save-audio", str(wav_file), str(config_path)],
            input=input_json.encode('utf-8'),
            capture_output=True,
            timeout=30,  # 30s per language should be plenty
            cwd=str(cpp_binary.parent.parent),
            env=get_tts_env()
        )

        # Decode output with error tolerance (C++ may output non-UTF-8 debug info)
        stderr_text = result.stderr.decode('utf-8', errors='replace') if result.stderr else ''

        # Check for success
        audio_generated = wav_file.exists() and wav_file.stat().st_size > 1000

        if not audio_generated:
            # Provide detailed failure message
            pytest.fail(
                f"Language {lang} failed to generate audio:\n"
                f"  Text: {text}\n"
                f"  Return code: {result.returncode}\n"
                f"  Stderr: {stderr_text[:500] if stderr_text else 'empty'}\n"
                f"  WAV exists: {wav_file.exists()}\n"
                f"  WAV size: {wav_file.stat().st_size if wav_file.exists() else 0}"
            )

        # Clean assertion for pytest output
        assert audio_generated, f"TTS failed for language {lang}"

    def test_all_languages_have_configs(self, cpp_binary, config_dir):
        """Verify all test languages have matching config files and registry defaults."""
        missing = []
        # Registry should provide default_config for these languages
        caps = get_capabilities(cpp_binary)
        default_cfg = {l["code"]: l.get("default_config", "") for l in caps["languages"]}
        for lang, test_data in MULTILINGUAL_TEST_SENTENCES.items():
            config_path = config_dir / test_data["config"]
            if not config_path.exists():
                missing.append(f"{lang}: {test_data['config']}")
            # If registry has a default config, it should match the test config
            if default_cfg.get(lang):
                assert default_cfg[lang] == test_data["config"], \
                    f"Registry default_config mismatch for {lang}: {default_cfg[lang]} vs {test_data['config']}"

        assert not missing, f"Missing configs: {missing}"

    def test_multilingual_coverage(self, cpp_binary):
        """Verify smoke tests cover all languages with default configs."""
        caps = get_capabilities(cpp_binary)
        expected_langs = {
            l["code"] for l in caps["languages"] if l.get("default_config")
        }
        tested_langs = set(MULTILINGUAL_TEST_SENTENCES.keys())

        assert tested_langs == expected_langs, \
            f"Smoke language set mismatch. expected={expected_langs} tested={tested_langs}"


# =============================================================================
# MPS Device Residency Test (Worker #208)
# =============================================================================

@pytest.mark.integration
@pytest.mark.requires_binary
@pytest.mark.requires_models
class TestMPSResidency:
    """
    Verify MPS GPU is used for inference (no silent CPU fallbacks).

    NOTE: Moved from tests/smoke/ because it requires TTS execution with
    model loading, which takes ~17s. Not suitable for pre-commit hooks.
    """

    def test_no_mps_residency_warnings(self, cpp_binary, english_config, tmp_path):
        """
        Run TTS and verify no MPS_RESIDENCY warnings appear.

        This test catches silent CPU fallbacks that cause 10x slower inference.
        The MPS residency checks (Worker #208) log warnings when tensors are
        unexpectedly on CPU instead of MPS.
        """
        wav_file = tmp_path / "mps_residency_test.wav"
        input_json = '{"type":"content_block_delta","delta":{"type":"text_delta","text":"Testing MPS residency."}}'

        result = subprocess.run(
            [str(cpp_binary), "--save-audio", str(wav_file), str(english_config)],
            input=input_json,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(cpp_binary.parent.parent),
            env=get_tts_env()
        )

        # Check that audio was generated
        assert wav_file.exists() and wav_file.stat().st_size > 1000, \
            f"TTS failed to generate audio. stderr: {result.stderr[:500]}"

        # Check for MPS_RESIDENCY warnings in output
        # These indicate silent CPU fallbacks that hurt performance
        combined_output = (result.stdout or "") + (result.stderr or "")

        mps_warnings = [
            line for line in combined_output.split('\n')
            if 'MPS_RESIDENCY' in line
        ]

        if mps_warnings:
            pytest.fail(
                f"MPS residency check found {len(mps_warnings)} warning(s):\n" +
                "\n".join(f"  - {w}" for w in mps_warnings[:5]) +
                ("\n  ... and more" if len(mps_warnings) > 5 else "")
            )

    def test_mps_is_available(self):
        """
        Verify MPS (Metal Performance Shaders) is available on this system.

        This test is informational - it documents whether MPS is available
        but doesn't fail since MPS residency checks are automatically
        skipped when running on CPU-only systems.
        """
        import platform

        # MPS is only available on Apple Silicon Macs
        if platform.system() != "Darwin":
            pytest.skip("MPS only available on macOS")

        # Check for Apple Silicon
        cpu_info = platform.processor()
        is_apple_silicon = "arm" in cpu_info.lower() or platform.machine() == "arm64"

        if not is_apple_silicon:
            pytest.skip("MPS requires Apple Silicon (M1/M2/M3/M4)")

        # If we're on Apple Silicon, MPS should be available
        # The actual MPS usage is verified by test_no_mps_residency_warnings
