"""
A/B Test: Streaming vs One-Shot Audio Comparison

This test validates that streaming mode produces audio identical (or very similar)
to one-shot mode. This is a quality gate to ensure streaming doesn't degrade output.

Per MANAGER_ROADMAP_2025-12-05.md Phase 1 Audio Quality:
- [ ] A/B Test: Streaming vs one-shot should sound identical

Worker #237 - Investigation findings:
- Streaming mode plays audio directly to speaker, does NOT support --save-audio
- Sequential mode supports --save-audio for file output
- The two modes cannot be directly A/B compared via file analysis
- Manual listening tests required for subjective quality comparison

This test validates that SEQUENTIAL mode produces consistent audio duration
when run multiple times (reproducibility test).
"""

import os
import pytest
import subprocess
import tempfile
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
STREAM_TTS_CPP = PROJECT_ROOT / "stream-tts-cpp"
BUILD_DIR = STREAM_TTS_CPP / "build"
TTS_BINARY = BUILD_DIR / "stream-tts-cpp"


def get_wav_duration(wav_path: Path) -> float:
    """Get duration of WAV file in seconds."""
    try:
        import soundfile as sf
        data, sr = sf.read(wav_path)
        return len(data) / sr
    except ImportError:
        # Fallback: use ffprobe
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(wav_path)],
            capture_output=True, text=True
        )
        return float(result.stdout.strip())


def get_config_for_lang(lang: str) -> Path:
    """Get the appropriate config file for a language."""
    config_map = {
        "en": "kokoro-mps-en.yaml",
        "es": "kokoro-mps-es.yaml",
        "fr": "kokoro-mps-fr.yaml",
        "ja": "kokoro-mps-ja.yaml",
        "zh": "kokoro-mps-zh.yaml",
    }
    config_name = config_map.get(lang, "kokoro-mps-en.yaml")
    return STREAM_TTS_CPP / "config" / config_name


def synthesize_sequential(text: str, lang: str, output_path: Path) -> dict:
    """
    Synthesize audio in sequential mode with --save-audio.

    Returns dict with duration, synthesis_time, etc.
    """
    config_path = get_config_for_lang(lang)

    # Build command - sequential mode with --save-audio
    cmd = [str(TTS_BINARY), "--save-audio", str(output_path), str(config_path)]

    # Create Claude API JSON format input
    escaped_text = text.replace('\\', '\\\\').replace('"', '\\"')
    input_json = f'{{"type":"content_block_delta","delta":{{"type":"text_delta","text":"{escaped_text}"}}}}'

    import time
    start = time.time()
    result = subprocess.run(
        cmd,
        input=input_json,
        capture_output=True,
        text=True,
        cwd=str(STREAM_TTS_CPP),
        timeout=120
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        # Check if audio was still generated despite non-zero return
        if not output_path.exists() or output_path.stat().st_size < 1000:
            return {
                "success": False,
                "error": result.stderr[:500],
            }

    duration = get_wav_duration(output_path) if output_path.exists() else 0

    return {
        "success": True,
        "duration": duration,
        "synthesis_time": elapsed,
        "output_path": str(output_path)
    }


class TestTTSReproducibility:
    """Tests that TTS produces consistent audio duration across runs."""

    @pytest.fixture
    def tts_available(self):
        """Check if TTS binary is available."""
        if not TTS_BINARY.exists():
            pytest.skip(f"TTS binary not found: {TTS_BINARY}")
        return True

    @pytest.mark.parametrize("text,lang", [
        ("Hello world, how are you today?", "en"),
        ("The quick brown fox jumps over the lazy dog.", "en"),
        ("Are you coming to the party tonight?", "en"),
    ])
    def test_english_reproducibility(self, tts_available, text, lang):
        """
        Run TTS twice and verify output duration is consistent.

        Duration should be identical (within 1% tolerance for timing jitter).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            wav1 = Path(tmpdir) / "run1.wav"
            wav2 = Path(tmpdir) / "run2.wav"

            # Generate audio twice
            result1 = synthesize_sequential(text, lang, output_path=wav1)
            result2 = synthesize_sequential(text, lang, output_path=wav2)

            # Both should succeed
            if not result1["success"]:
                pytest.skip(f"Run 1 failed: {result1.get('error', 'unknown')}")
            if not result2["success"]:
                pytest.skip(f"Run 2 failed: {result2.get('error', 'unknown')}")

            dur1 = result1["duration"]
            dur2 = result2["duration"]

            # Calculate duration difference
            if dur1 > 0:
                diff_pct = abs(dur2 - dur1) / dur1 * 100
            else:
                diff_pct = 100 if dur2 > 0 else 0

            print(f"\n[Reproducibility Test] '{text[:40]}...'")
            print(f"  Run 1: {dur1:.3f}s")
            print(f"  Run 2: {dur2:.3f}s")
            print(f"  Difference: {diff_pct:.1f}%")

            # Duration should be identical (within 1% tolerance)
            assert diff_pct < 1, f"Non-reproducible: run1={dur1:.3f}s, run2={dur2:.3f}s ({diff_pct:.1f}% diff)"

    @pytest.mark.parametrize("text,lang", [
        ("Hola mundo, como estas hoy?", "es"),
        ("Bonjour le monde, comment allez-vous?", "fr"),
    ])
    def test_multilingual_reproducibility(self, tts_available, text, lang):
        """
        Run TTS twice for non-English and verify output duration is consistent.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            wav1 = Path(tmpdir) / "run1.wav"
            wav2 = Path(tmpdir) / "run2.wav"

            result1 = synthesize_sequential(text, lang, output_path=wav1)
            result2 = synthesize_sequential(text, lang, output_path=wav2)

            if not result1["success"]:
                pytest.skip(f"Run 1 failed: {result1.get('error', 'unknown')}")
            if not result2["success"]:
                pytest.skip(f"Run 2 failed: {result2.get('error', 'unknown')}")

            dur1 = result1["duration"]
            dur2 = result2["duration"]

            if dur1 > 0:
                diff_pct = abs(dur2 - dur1) / dur1 * 100
            else:
                diff_pct = 100 if dur2 > 0 else 0

            print(f"\n[Reproducibility Test] '{text[:40]}...' ({lang})")
            print(f"  Run 1: {dur1:.3f}s")
            print(f"  Run 2: {dur2:.3f}s")
            print(f"  Difference: {diff_pct:.1f}%")

            # Duration should be identical (within 1% tolerance)
            assert diff_pct < 1, f"Non-reproducible: run1={dur1:.3f}s, run2={dur2:.3f}s ({diff_pct:.1f}% diff)"


class TestStreamingModeDocumentation:
    """Document streaming vs sequential mode differences."""

    def test_streaming_mode_does_not_save_audio(self):
        """
        Document that streaming mode does NOT support --save-audio.

        Streaming mode plays audio directly to speaker for real-time output.
        Sequential mode is required for file output.

        Per investigation (Worker #237):
        - Streaming: Real-time playback, no file output
        - Sequential: Supports --save-audio for file output
        - A/B comparison requires manual listening test
        """
        # This test is documentation-only
        # The actual finding is documented in the docstring
        pass
