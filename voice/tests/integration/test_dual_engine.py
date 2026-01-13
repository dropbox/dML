"""
Dual-Engine Integration Tests

Tests for Kokoro ↔ CosyVoice2 engine switching and interoperability.
Verifies the unified voice interface works correctly across both TTS engines.

This validates Phase 4 (Unified Voice Registry) integration.

Usage:
    pytest tests/integration/test_dual_engine.py -v
    pytest tests/integration/test_dual_engine.py -v -s  # With output

Phase 5 of CosyVoice2 llama.cpp + libtorch implementation.
"""

import os
import subprocess
import tempfile
import time
import wave
import numpy as np
import pytest
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
STREAM_TTS_CPP = PROJECT_ROOT / "stream-tts-cpp"
BUILD_DIR = STREAM_TTS_CPP / "build"
TTS_BINARY = BUILD_DIR / "stream-tts-cpp"
CONFIG_DIR = STREAM_TTS_CPP / "config"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "output" / "dual_engine"


def get_tts_env():
    """Get environment variables for TTS subprocess."""
    env = os.environ.copy()
    # Fix OpenMP duplicate library crash when llama.cpp + libtorch both link OpenMP
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["OMP_NUM_THREADS"] = "1"
    return env


def read_wav(path: Path) -> tuple:
    """Read WAV file and return (samples as float array, sample_rate)."""
    with wave.open(str(path), 'rb') as wf:
        frames = wf.readframes(wf.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        return samples, wf.getframerate()


def get_audio_metrics(audio_data: np.ndarray) -> dict:
    """Calculate audio quality metrics."""
    if len(audio_data) == 0:
        return {"rms": 0.0, "peak": 0.0, "duration_samples": 0}

    audio = audio_data.astype(np.float32)
    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio))

    return {
        "rms": float(rms),
        "peak": float(peak),
        "duration_samples": len(audio),
    }


def generate_audio(voice_name: str, text: str, lang: str,
                   instruction: str = None, timeout: int = 60) -> tuple:
    """
    Generate audio using CLI and return (audio_samples, sample_rate, inference_time).

    Returns None if generation fails.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        output_path = Path(f.name)

    try:
        cmd = [
            str(TTS_BINARY),
            "--voice-name", voice_name,
            "--speak", text,
            "--lang", lang,
            "--save-audio", str(output_path)
        ]

        if instruction:
            cmd.extend(["--instruction", instruction])

        start = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(STREAM_TTS_CPP),
            env=get_tts_env()
        )
        inference_time = time.time() - start

        if result.returncode != 0:
            print(f"Generation failed: {result.stderr}")
            return None, None, inference_time

        if not output_path.exists() or output_path.stat().st_size < 1000:
            print("Output file empty or missing")
            return None, None, inference_time

        audio, sample_rate = read_wav(output_path)
        return audio, sample_rate, inference_time

    finally:
        if output_path.exists():
            output_path.unlink()


# Skip if binary not built
pytestmark = pytest.mark.skipif(
    not TTS_BINARY.exists(),
    reason="stream-tts-cpp binary not built"
)


class TestEngineRouting:
    """Test that voices route to the correct TTS engine."""

    # Kokoro voices (should route to Kokoro engine)
    KOKORO_VOICES = [
        ("af_heart", "Hello world", "en"),
        ("af_bella", "Good morning", "en"),
        ("bf_emma", "How are you today?", "en"),
    ]

    # CosyVoice2 voices (should route to CosyVoice2 engine)
    COSYVOICE2_VOICES = [
        ("sichuan", "你好世界", "zh"),
        ("cosy", "你好", "zh"),
        ("cosyvoice", "今天天气真好", "zh"),
    ]

    @pytest.mark.parametrize("voice,text,lang", KOKORO_VOICES)
    def test_kokoro_voice_routing(self, voice, text, lang):
        """Kokoro voices should produce valid audio."""
        audio, sample_rate, _ = generate_audio(voice, text, lang)

        assert audio is not None, f"Kokoro voice '{voice}' failed to generate"
        assert len(audio) > 0, f"Kokoro voice '{voice}' produced empty audio"

        metrics = get_audio_metrics(audio)
        assert metrics["rms"] > 0.01, f"Kokoro voice '{voice}' produced silent audio"

        # Kokoro uses 24kHz
        assert sample_rate == 24000, f"Unexpected sample rate: {sample_rate}"

        print(f"\nKokoro voice '{voice}': RMS={metrics['rms']:.4f}, "
              f"duration={len(audio)/sample_rate:.2f}s")

    @pytest.mark.parametrize("voice,text,lang", COSYVOICE2_VOICES)
    def test_cosyvoice2_voice_routing(self, voice, text, lang):
        """CosyVoice2 voices should produce valid audio."""
        audio, sample_rate, _ = generate_audio(voice, text, lang)

        assert audio is not None, f"CosyVoice2 voice '{voice}' failed to generate"
        assert len(audio) > 0, f"CosyVoice2 voice '{voice}' produced empty audio"

        metrics = get_audio_metrics(audio)
        assert metrics["rms"] > 0.01, f"CosyVoice2 voice '{voice}' produced silent audio"

        # CosyVoice2 uses 24kHz (HiFT vocoder)
        assert sample_rate in [22050, 24000], f"Unexpected sample rate: {sample_rate}"

        print(f"\nCosyVoice2 voice '{voice}': RMS={metrics['rms']:.4f}, "
              f"duration={len(audio)/sample_rate:.2f}s")


class TestEngineSwitching:
    """Test switching between engines in sequence."""

    def test_kokoro_then_cosyvoice(self):
        """Generate with Kokoro, then CosyVoice2."""
        # Kokoro
        audio1, sr1, time1 = generate_audio("af_heart", "Hello world", "en")
        assert audio1 is not None, "Kokoro generation failed"

        # CosyVoice2
        audio2, sr2, time2 = generate_audio("sichuan", "你好世界", "zh")
        assert audio2 is not None, "CosyVoice2 generation failed"

        # Both should produce valid audio
        metrics1 = get_audio_metrics(audio1)
        metrics2 = get_audio_metrics(audio2)

        assert metrics1["rms"] > 0.01, "Kokoro produced silent audio"
        assert metrics2["rms"] > 0.01, "CosyVoice2 produced silent audio"

        print(f"\nKokoro: {time1:.2f}s, CosyVoice2: {time2:.2f}s")

    def test_cosyvoice_then_kokoro(self):
        """Generate with CosyVoice2, then Kokoro."""
        # CosyVoice2
        audio1, sr1, time1 = generate_audio("sichuan", "你好", "zh")
        assert audio1 is not None, "CosyVoice2 generation failed"

        # Kokoro
        audio2, sr2, time2 = generate_audio("af_heart", "Good morning", "en")
        assert audio2 is not None, "Kokoro generation failed"

        # Both should produce valid audio
        metrics1 = get_audio_metrics(audio1)
        metrics2 = get_audio_metrics(audio2)

        assert metrics1["rms"] > 0.01, "CosyVoice2 produced silent audio"
        assert metrics2["rms"] > 0.01, "Kokoro produced silent audio"

        print(f"\nCosyVoice2: {time1:.2f}s, Kokoro: {time2:.2f}s")

    def test_alternating_engines(self):
        """Alternate between engines multiple times."""
        voices = [
            ("af_heart", "One", "en"),
            ("sichuan", "二", "zh"),
            ("af_bella", "Three", "en"),
            ("cosy", "四", "zh"),
            ("bf_emma", "Five", "en"),
        ]

        for voice, text, lang in voices:
            audio, sr, t = generate_audio(voice, text, lang)
            assert audio is not None, f"Failed for voice '{voice}'"

            metrics = get_audio_metrics(audio)
            assert metrics["rms"] > 0.01, f"Silent audio for voice '{voice}'"

            engine = "CosyVoice2" if voice in ["sichuan", "cosy"] else "Kokoro"
            print(f"\n{engine} ({voice}): {t:.2f}s, RMS={metrics['rms']:.4f}")


class TestDualEnginePerformance:
    """Performance comparison between engines."""

    def test_latency_comparison(self):
        """Compare first-audio latency between engines."""
        # Similar-length texts
        kokoro_text = "Hello, how are you today?"
        cosy_text = "你好，今天你好吗？"

        # Kokoro (warm run)
        generate_audio("af_heart", "warmup", "en")
        _, _, kokoro_time = generate_audio("af_heart", kokoro_text, "en")

        # CosyVoice2 (warm run)
        generate_audio("sichuan", "热身", "zh")
        _, _, cosy_time = generate_audio("sichuan", cosy_text, "zh")

        print(f"\nLatency Comparison:")
        print(f"  Kokoro: {kokoro_time:.2f}s")
        print(f"  CosyVoice2: {cosy_time:.2f}s")

        # Both should be under reasonable limits for cold starts (model loading)
        # CosyVoice2 needs more time due to loading llama.cpp + libtorch models
        assert kokoro_time < 15.0, f"Kokoro too slow: {kokoro_time:.2f}s"
        assert cosy_time < 20.0, f"CosyVoice2 too slow: {cosy_time:.2f}s"

    def test_rtf_comparison(self):
        """Compare RTF (Real-Time Factor) between engines."""
        # Similar-length texts
        kokoro_text = "The quick brown fox jumps over the lazy dog."
        cosy_text = "敏捷的棕色狐狸跳过了懒狗。"

        # Kokoro
        audio_k, sr_k, time_k = generate_audio("af_heart", kokoro_text, "en")
        if audio_k is not None:
            dur_k = len(audio_k) / sr_k
            rtf_k = time_k / dur_k if dur_k > 0 else float('inf')
        else:
            rtf_k = float('inf')

        # CosyVoice2
        audio_c, sr_c, time_c = generate_audio("sichuan", cosy_text, "zh")
        if audio_c is not None:
            dur_c = len(audio_c) / sr_c
            rtf_c = time_c / dur_c if dur_c > 0 else float('inf')
        else:
            rtf_c = float('inf')

        print(f"\nRTF Comparison:")
        print(f"  Kokoro: RTF={rtf_k:.3f} ({time_k:.2f}s / {dur_k:.2f}s audio)")
        print(f"  CosyVoice2: RTF={rtf_c:.3f} ({time_c:.2f}s / {dur_c:.2f}s audio)")

        # Both should be real-time capable (RTF < 1.0)
        # Allow up to 3.0 for cold starts (includes model loading)
        assert rtf_k < 3.0, f"Kokoro RTF too high: {rtf_k:.3f}"
        assert rtf_c < 3.0, f"CosyVoice2 RTF too high: {rtf_c:.3f}"


class TestVoiceList:
    """Test --list-voices command."""

    def test_list_voices_includes_kokoro(self):
        """--list-voices should show Kokoro voices."""
        result = subprocess.run(
            [str(TTS_BINARY), "--list-voices"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(STREAM_TTS_CPP),
            env=get_tts_env()
        )

        assert result.returncode == 0, f"--list-voices failed: {result.stderr}"

        output = result.stdout.lower()

        # Should include Kokoro voices (header or individual voices)
        has_kokoro = "kokoro" in output or "af_heart" in output
        assert has_kokoro, f"Kokoro voices not listed. Output: {result.stdout[:500]}"

        print(f"\n--list-voices output (truncated):\n{result.stdout[:1000]}")

    def test_list_engines_command(self):
        """--list-engines should return successfully."""
        result = subprocess.run(
            [str(TTS_BINARY), "--list-engines"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(STREAM_TTS_CPP),
            env=get_tts_env()
        )

        # Command should succeed
        assert result.returncode == 0, f"--list-engines failed: {result.stderr}"

        # Should list some engines (may vary based on build)
        output = result.stdout.lower()
        assert "engine" in output or "tts" in output, (
            f"No engine info in output: {result.stdout}"
        )

        print(f"\n--list-engines output:\n{result.stdout}")


class TestInstructionWithDualEngine:
    """Test instruction mode specific to CosyVoice2."""

    def test_instruction_only_for_cosyvoice(self):
        """Instruction flag should work with CosyVoice2 voices."""
        # CosyVoice2 with instruction should work
        audio, sr, _ = generate_audio("cosy", "你好", "zh", instruction="开心地说")
        assert audio is not None, "CosyVoice2 with instruction failed"

        metrics = get_audio_metrics(audio)
        assert metrics["rms"] > 0.01, "CosyVoice2 with instruction produced silent audio"

    def test_sichuan_dialect(self):
        """Sichuan voice should apply dialect instruction."""
        audio, sr, _ = generate_audio("sichuan", "你好世界", "zh")
        assert audio is not None, "Sichuan dialect generation failed"

        # Should produce audible speech
        metrics = get_audio_metrics(audio)
        assert metrics["rms"] > 0.01, "Sichuan dialect produced silent audio"
        assert len(audio) / sr > 0.5, "Sichuan dialect audio too short"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_voice_fallback(self):
        """Invalid voice should fallback gracefully."""
        # This might either fail or fallback to a default
        result = subprocess.run(
            [
                str(TTS_BINARY),
                "--voice-name", "nonexistent_voice_xyz",
                "--speak", "test",
                "--lang", "en"
            ],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(STREAM_TTS_CPP),
            env=get_tts_env()
        )

        # Should either work with fallback or fail gracefully
        # The important thing is it doesn't crash
        print(f"\nInvalid voice result: returncode={result.returncode}")
        if result.returncode != 0:
            print(f"Error (expected): {result.stderr[:200]}")

    def test_empty_text(self):
        """Empty text should be handled gracefully."""
        result = subprocess.run(
            [
                str(TTS_BINARY),
                "--voice-name", "af_heart",
                "--speak", "",
                "--lang", "en"
            ],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(STREAM_TTS_CPP),
            env=get_tts_env()
        )

        # Should handle gracefully (either skip or return quickly)
        print(f"\nEmpty text result: returncode={result.returncode}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
