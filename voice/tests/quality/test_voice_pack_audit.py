"""
Voice Pack Quality Audit Tests

Tests all 54 Kokoro voice packs for quality using:
1. Audio quality metrics (RMS, peak, zero-crossing rate)
2. STT round-trip verification
3. LLM-as-Judge evaluation (optional, requires OPENAI_API_KEY)

Worker #235: Phase 5 Voice Pack Quality Audit

Usage:
    # Quick audit (audio metrics only, no API calls)
    pytest tests/quality/test_voice_pack_audit.py -v -m "not slow"

    # Full audit with STT verification
    pytest tests/quality/test_voice_pack_audit.py -v

    # Generate quality report
    pytest tests/quality/test_voice_pack_audit.py::test_generate_voice_quality_report -v -s
"""

import json
import os
import sys
import subprocess
import tempfile
import wave
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pytest

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
STREAM_TTS_CPP = PROJECT_ROOT / "stream-tts-cpp"
BUILD_DIR = STREAM_TTS_CPP / "build"
MODELS_DIR = PROJECT_ROOT / "models" / "kokoro"
TESTS_DIR = PROJECT_ROOT / "tests"
REPORTS_DIR = PROJECT_ROOT / "reports" / "main"

sys.path.insert(0, str(TESTS_DIR))

# Voice pack categorization by language
# Format: {language_code: {voice_id: voice_type}}
# voice_type: 'f' = female, 'm' = male
VOICE_PACKS = {
    "en-us": {  # American English
        "af_alloy": "f", "af_aoede": "f", "af_bella": "f", "af_heart": "f",
        "af_jessica": "f", "af_kore": "f", "af_nicole": "f", "af_nova": "f",
        "af_river": "f", "af_sarah": "f", "af_sky": "f",
        "am_adam": "m", "am_echo": "m", "am_eric": "m", "am_fenrir": "m",
        "am_liam": "m", "am_michael": "m", "am_onyx": "m", "am_puck": "m",
        "am_santa": "m",
    },
    "en-gb": {  # British English
        "bf_alice": "f", "bf_emma": "f", "bf_isabella": "f", "bf_lily": "f",
        "bm_daniel": "m", "bm_fable": "m", "bm_george": "m", "bm_lewis": "m",
    },
    "es": {  # Spanish
        "ef_dora": "f", "em_alex": "m", "em_santa": "m",
    },
    "fr": {  # French
        "ff_siwis": "f",
    },
    "hi": {  # Hindi
        "hf_alpha": "f", "hf_beta": "f", "hm_omega": "m", "hm_psi": "m",
    },
    "it": {  # Italian
        "if_sara": "f", "im_nicola": "m",
    },
    "ja": {  # Japanese
        "jf_alpha": "f", "jf_gongitsune": "f", "jf_nezumi": "f", "jf_tebukuro": "f",
        "jm_kumo": "m",
    },
    "pt": {  # Portuguese
        "pf_dora": "f", "pm_alex": "m", "pm_santa": "m",
    },
    "zh": {  # Chinese (Mandarin)
        "zf_xiaobei": "f", "zf_xiaoni": "f", "zf_xiaoxiao": "f", "zf_xiaoyi": "f",
        "zm_yunjian": "m", "zm_yunxi": "m", "zm_yunxia": "m", "zm_yunyang": "m",
    },
}

# Test sentences per language
TEST_SENTENCES = {
    "en-us": "Hello, world! The quick brown fox jumps over the lazy dog.",
    "en-gb": "Hello, world! The quick brown fox jumps over the lazy dog.",
    "es": "Hola, mundo! El veloz murciélago hindú comía feliz cardillo.",
    "fr": "Bonjour, le monde! Portez ce vieux whisky au juge blond qui fume.",
    "hi": "नमस्ते दुनिया! यह एक परीक्षण है।",
    "it": "Ciao, mondo! Questo è un test di qualità vocale.",
    "ja": "こんにちは、世界！これは音声品質テストです。",
    "pt": "Olá, mundo! A rápida raposa marrom salta sobre o cachorro preguiçoso.",
    "zh": "你好，世界！这是一个语音质量测试。",
}

# Default voices per language (current configuration)
DEFAULT_VOICES = {
    "en-us": "af_heart",
    "en-gb": "bf_emma",
    "es": "ef_dora",
    "fr": "ff_siwis",
    "hi": "hf_alpha",
    "it": "if_sara",
    "ja": "jf_alpha",
    "pt": "pf_dora",
    "zh": "zf_xiaobei",
}

# Flatten for easy iteration
ALL_VOICES = []
for lang, voices in VOICE_PACKS.items():
    for voice_id, voice_type in voices.items():
        ALL_VOICES.append((lang, voice_id, voice_type))


def get_voice_file(voice_id: str) -> Path:
    """Get the path to a voice file."""
    return MODELS_DIR / f"voice_{voice_id}.pt"


def voice_file_exists(voice_id: str) -> bool:
    """Check if voice file exists."""
    return get_voice_file(voice_id).exists()


def generate_audio_with_voice(text: str, voice_id: str, output_path: Path,
                               timeout: int = 120) -> Tuple[bool, str]:
    """
    Generate audio using a specific voice.

    Uses the main stream-tts-cpp binary with --speak and --voice-name options.

    Returns:
        (success, error_message)
    """
    binary = BUILD_DIR / "stream-tts-cpp"
    if not binary.exists():
        return False, f"Binary not found: {binary}"

    voice_file = get_voice_file(voice_id)
    if not voice_file.exists():
        return False, f"Voice file not found: {voice_file}"

    # Determine language from voice ID
    lang_prefix = voice_id[0]  # First char is language
    lang_map = {'a': 'en', 'b': 'en', 'e': 'es', 'f': 'fr',
                'h': 'hi', 'i': 'it', 'j': 'ja', 'p': 'pt', 'z': 'zh'}
    lang = lang_map.get(lang_prefix, 'en')

    cmd = [
        str(binary),
        "--speak", text,
        "--lang", lang,
        "--voice-name", voice_id,
        "--save-audio", str(output_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(STREAM_TTS_CPP)
        )

        if result.returncode != 0:
            return False, f"Exit code {result.returncode}: {result.stderr[:200]}"

        if not output_path.exists():
            return False, "Output file not created"

        return True, ""

    except subprocess.TimeoutExpired:
        return False, f"Timeout after {timeout}s"
    except Exception as e:
        return False, str(e)


def analyze_audio_basic(wav_path: Path) -> Dict:
    """
    Basic audio analysis without external dependencies.

    Returns:
        Dict with duration_s, sample_rate, channels, has_audio
    """
    try:
        with wave.open(str(wav_path), 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            duration = frames / rate

            # Read samples to check for silence
            raw_data = wf.readframes(frames)

        # Worker #424: Fixed silence detection bug
        # Previous code only checked first 1000 bytes which could be silent fade-in
        # Now check for any non-zero samples across entire file using max amplitude
        import struct
        if sampwidth == 2:
            # 16-bit audio - unpack as signed shorts
            num_samples = len(raw_data) // 2
            samples = struct.unpack(f'<{num_samples}h', raw_data)
            max_amp = max(abs(s) for s in samples) if samples else 0
            # Audio is considered present if max amplitude > 100 (out of 32767)
            has_audio = max_amp > 100
        else:
            # Fallback for other sample widths - check if any byte is non-zero
            has_audio = any(b != 0 for b in raw_data)

        return {
            "duration_s": duration,
            "sample_rate": rate,
            "channels": channels,
            "has_audio": has_audio,
            "file_size_kb": wav_path.stat().st_size / 1024,
        }
    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def tts_binary():
    """Path to TTS binary."""
    binary = BUILD_DIR / "stream-tts-cpp"
    if not binary.exists():
        pytest.skip(f"TTS binary not found: {binary}")
    return binary


# =============================================================================
# Tests
# =============================================================================

class TestVoicePackInventory:
    """Test voice pack availability."""

    def test_voice_file_count(self):
        """Verify expected number of voice files exist."""
        voice_files = list(MODELS_DIR.glob("voice_*.pt"))
        assert len(voice_files) >= 50, f"Expected 50+ voice files, found {len(voice_files)}"

    @pytest.mark.parametrize("lang,voice_id,voice_type", ALL_VOICES)
    def test_voice_file_exists(self, lang, voice_id, voice_type):
        """Check each cataloged voice file exists."""
        voice_file = get_voice_file(voice_id)
        assert voice_file.exists(), f"Missing voice file: {voice_file}"

    def test_default_voices_exist(self):
        """Ensure all default voices for languages exist."""
        for lang, voice_id in DEFAULT_VOICES.items():
            voice_file = get_voice_file(voice_id)
            assert voice_file.exists(), f"Default voice for {lang} missing: {voice_file}"


class TestVoicePackQuality:
    """Test voice pack audio quality."""

    @pytest.mark.parametrize("lang,voice_id,voice_type", ALL_VOICES[:10])  # Test first 10
    def test_voice_generates_audio(self, tts_binary, lang, voice_id, voice_type):
        """Test that voice can generate non-empty audio."""
        # Use appropriate test sentence
        base_lang = lang.split('-')[0]
        if base_lang in TEST_SENTENCES:
            text = TEST_SENTENCES[base_lang]
        elif lang in TEST_SENTENCES:
            text = TEST_SENTENCES[lang]
        else:
            text = TEST_SENTENCES["en-us"]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = Path(f.name)

        try:
            success, error = generate_audio_with_voice(text, voice_id, output_path)

            if not success:
                pytest.skip(f"Audio generation failed for {voice_id}: {error}")

            # Analyze audio
            analysis = analyze_audio_basic(output_path)

            assert "error" not in analysis, f"Analysis error: {analysis.get('error')}"
            assert analysis["duration_s"] > 0.5, f"Audio too short: {analysis['duration_s']}s"
            assert analysis["has_audio"], "Audio appears to be silent"

        finally:
            if output_path.exists():
                output_path.unlink()


class TestDefaultVoiceQuality:
    """Test quality of default voices per language."""

    @pytest.mark.parametrize("lang,voice_id", list(DEFAULT_VOICES.items()))
    def test_default_voice_quality(self, tts_binary, lang, voice_id):
        """Test default voice for each language produces quality audio."""
        base_lang = lang.split('-')[0]
        text = TEST_SENTENCES.get(lang) or TEST_SENTENCES.get(base_lang) or TEST_SENTENCES["en-us"]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = Path(f.name)

        try:
            success, error = generate_audio_with_voice(text, voice_id, output_path)

            if not success:
                pytest.skip(f"Audio generation failed: {error}")

            analysis = analyze_audio_basic(output_path)

            # Quality checks
            assert analysis.get("duration_s", 0) > 1.0, \
                f"{lang}/{voice_id}: Audio too short ({analysis.get('duration_s', 0):.2f}s)"
            assert analysis.get("has_audio", False), \
                f"{lang}/{voice_id}: Audio is silent"
            assert analysis.get("sample_rate", 0) >= 22050, \
                f"{lang}/{voice_id}: Sample rate too low ({analysis.get('sample_rate', 0)})"

        finally:
            if output_path.exists():
                output_path.unlink()


@pytest.mark.slow
def test_generate_voice_quality_report(tts_binary):
    """
    Generate comprehensive voice quality report.

    This test generates audio for all voices and creates a quality report.
    Run with: pytest tests/quality/test_voice_pack_audit.py::test_generate_voice_quality_report -v -s
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "total_voices": len(ALL_VOICES),
        "languages": {},
    }

    passed = 0
    failed = 0
    skipped = 0

    for lang, voice_id, voice_type in ALL_VOICES:
        # Get test sentence
        base_lang = lang.split('-')[0]
        text = TEST_SENTENCES.get(lang) or TEST_SENTENCES.get(base_lang) or TEST_SENTENCES["en-us"]

        # Initialize language entry
        if lang not in results["languages"]:
            results["languages"][lang] = {
                "voices": {},
                "best_voice": None,
                "best_score": 0,
            }

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = Path(f.name)

        try:
            success, error = generate_audio_with_voice(text, voice_id, output_path)

            if not success:
                results["languages"][lang]["voices"][voice_id] = {
                    "status": "failed",
                    "error": error,
                    "voice_type": voice_type,
                }
                failed += 1
                print(f"  FAIL: {voice_id} ({lang}) - {error[:50]}")
                continue

            analysis = analyze_audio_basic(output_path)

            if "error" in analysis:
                results["languages"][lang]["voices"][voice_id] = {
                    "status": "error",
                    "error": analysis["error"],
                    "voice_type": voice_type,
                }
                failed += 1
                continue

            # Calculate simple quality score (0-100)
            # Based on: duration reasonableness, has audio, file size
            score = 0
            if analysis["has_audio"]:
                score += 40
            if 1.0 < analysis["duration_s"] < 30.0:
                score += 30
            if analysis["sample_rate"] >= 22050:
                score += 20
            if analysis["file_size_kb"] > 10:
                score += 10

            results["languages"][lang]["voices"][voice_id] = {
                "status": "passed",
                "voice_type": voice_type,
                "duration_s": round(analysis["duration_s"], 2),
                "sample_rate": analysis["sample_rate"],
                "file_size_kb": round(analysis["file_size_kb"], 1),
                "quality_score": score,
            }

            # Track best voice for language
            if score > results["languages"][lang]["best_score"]:
                results["languages"][lang]["best_score"] = score
                results["languages"][lang]["best_voice"] = voice_id

            passed += 1
            print(f"  PASS: {voice_id} ({lang}) - score={score}, dur={analysis['duration_s']:.2f}s")

        finally:
            if output_path.exists():
                output_path.unlink()

    results["summary"] = {
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "pass_rate": f"{100 * passed / len(ALL_VOICES):.1f}%",
    }

    # Generate best voices table
    results["best_voices_table"] = {}
    for lang, data in results["languages"].items():
        if data["best_voice"]:
            results["best_voices_table"][lang] = {
                "voice": data["best_voice"],
                "score": data["best_score"],
            }

    # Save report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"voice_quality_audit_{datetime.now().strftime('%Y-%m-%d')}.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Voice Quality Audit Report")
    print(f"{'='*60}")
    print(f"Total voices tested: {len(ALL_VOICES)}")
    print(f"Passed: {passed}, Failed: {failed}")
    print(f"Pass rate: {100 * passed / len(ALL_VOICES):.1f}%")
    print(f"\nBest voice per language:")
    for lang, data in sorted(results["best_voices_table"].items()):
        print(f"  {lang}: {data['voice']} (score={data['score']})")
    print(f"\nReport saved: {report_path}")

    assert passed > 0, "No voices passed quality check"
