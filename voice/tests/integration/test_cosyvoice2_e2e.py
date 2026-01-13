"""
CosyVoice2 End-to-End Integration Tests

Tests the complete CosyVoice2 pipeline via CLI:
- Basic synthesis
- Instruction mode
- Voice switching (Kokoro ↔ CosyVoice2)
- Error handling
- Unicode text handling

Based on PRODUCTION_ROADMAP_2025-12-10.md Phase 3 requirements.

Usage:
    pytest tests/integration/test_cosyvoice2_e2e.py -v
    pytest tests/integration/test_cosyvoice2_e2e.py -v -s  # With output
"""

import os
import subprocess
import tempfile
import wave
import numpy as np
import pytest
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
STREAM_TTS_CPP = PROJECT_ROOT / "stream-tts-cpp"
BUILD_DIR = STREAM_TTS_CPP / "build"
TTS_BINARY = BUILD_DIR / "stream-tts-cpp"

# Skip if binary not built
pytestmark = pytest.mark.skipif(
    not TTS_BINARY.exists(),
    reason="stream-tts-cpp binary not built"
)


def get_tts_env():
    """Get environment variables for TTS subprocess."""
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["OMP_NUM_THREADS"] = "1"
    return env


def read_wav_info(path: Path) -> dict:
    """Read WAV file and return info dict."""
    with wave.open(str(path), 'rb') as wf:
        frames = wf.readframes(wf.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        return {
            "sample_rate": wf.getframerate(),
            "channels": wf.getnchannels(),
            "samples": len(samples),
            "duration_sec": len(samples) / wf.getframerate(),
            "rms": float(np.sqrt(np.mean(samples ** 2))),
        }


class TestCosyVoice2CLIBasic:
    """Basic CLI synthesis tests."""

    def test_cli_synthesis_basic_chinese(self):
        """Test basic Chinese synthesis via CLI."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = Path(f.name)

        try:
            cmd = [
                str(TTS_BINARY),
                "--voice-name", "sichuan",
                "--speak", "你好世界",
                "--lang", "zh",
                "--save-audio", str(output_path)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(STREAM_TTS_CPP),
                env=get_tts_env()
            )

            assert result.returncode == 0, f"CLI failed: {result.stderr}"
            assert output_path.exists(), "Output file not created"

            info = read_wav_info(output_path)
            assert info["duration_sec"] >= 0.5, f"Audio too short: {info['duration_sec']:.2f}s"
            assert info["rms"] > 0.002, f"Audio is silent (RMS={info['rms']:.6f})"
            assert info["sample_rate"] == 24000, f"Unexpected sample rate: {info['sample_rate']}"
        finally:
            if output_path.exists():
                output_path.unlink()

    def test_cli_synthesis_with_instruction(self):
        """Test CLI synthesis with instruction mode."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = Path(f.name)

        try:
            cmd = [
                str(TTS_BINARY),
                "--voice-name", "cosy",
                "--speak", "今天天气真好",
                "--lang", "zh",
                "--instruction", "用四川话说",
                "--save-audio", str(output_path)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(STREAM_TTS_CPP),
                env=get_tts_env()
            )

            assert result.returncode == 0, f"CLI failed: {result.stderr}"
            assert output_path.exists()

            info = read_wav_info(output_path)
            assert info["rms"] > 0.002, "Audio is silent"
        finally:
            if output_path.exists():
                output_path.unlink()

    def test_cli_synthesis_cosy_voice(self):
        """Test CLI with 'cosy' voice name."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = Path(f.name)

        try:
            cmd = [
                str(TTS_BINARY),
                "--voice-name", "cosy",
                "--speak", "你好",
                "--lang", "zh",
                "--save-audio", str(output_path)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(STREAM_TTS_CPP),
                env=get_tts_env()
            )

            assert result.returncode == 0, f"CLI failed: {result.stderr}"
            info = read_wav_info(output_path)
            assert info["rms"] > 0.002
        finally:
            if output_path.exists():
                output_path.unlink()

    def test_cli_synthesis_cosyvoice_alias(self):
        """Test CLI with 'cosyvoice' voice alias."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = Path(f.name)

        try:
            cmd = [
                str(TTS_BINARY),
                "--voice-name", "cosyvoice",
                "--speak", "你好",
                "--lang", "zh",
                "--save-audio", str(output_path)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(STREAM_TTS_CPP),
                env=get_tts_env()
            )

            assert result.returncode == 0, f"CLI failed: {result.stderr}"
            info = read_wav_info(output_path)
            assert info["rms"] > 0.002
        finally:
            if output_path.exists():
                output_path.unlink()


class TestCosyVoice2VoiceSwitching:
    """Test switching between Kokoro and CosyVoice2 engines."""

    def test_kokoro_synthesis_after_cosyvoice(self):
        """Test Kokoro works after CosyVoice2 was used."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            cosy_path = Path(f.name)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            kokoro_path = Path(f.name)

        try:
            # First: CosyVoice2 synthesis
            cmd1 = [
                str(TTS_BINARY),
                "--voice-name", "sichuan",
                "--speak", "你好",
                "--lang", "zh",
                "--save-audio", str(cosy_path)
            ]
            result1 = subprocess.run(cmd1, capture_output=True, text=True,
                                      timeout=120, cwd=str(STREAM_TTS_CPP), env=get_tts_env())
            assert result1.returncode == 0, f"CosyVoice2 failed: {result1.stderr}"

            # Second: Kokoro synthesis
            cmd2 = [
                str(TTS_BINARY),
                "--voice-name", "af_heart",
                "--speak", "Hello world",
                "--lang", "en",
                "--save-audio", str(kokoro_path)
            ]
            result2 = subprocess.run(cmd2, capture_output=True, text=True,
                                      timeout=60, cwd=str(STREAM_TTS_CPP), env=get_tts_env())
            assert result2.returncode == 0, f"Kokoro failed: {result2.stderr}"

            # Verify both produced audio
            cosy_info = read_wav_info(cosy_path)
            kokoro_info = read_wav_info(kokoro_path)

            assert cosy_info["rms"] > 0.002, "CosyVoice2 audio is silent"
            assert kokoro_info["rms"] > 0.002, "Kokoro audio is silent"
        finally:
            for p in [cosy_path, kokoro_path]:
                if p.exists():
                    p.unlink()

    def test_sequential_cosyvoice_synthesis(self):
        """Test multiple sequential CosyVoice2 syntheses."""
        texts = ["你好", "今天天气真好", "很高兴认识你"]
        paths = []

        try:
            for i, text in enumerate(texts):
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    path = Path(f.name)
                    paths.append(path)

                cmd = [
                    str(TTS_BINARY),
                    "--voice-name", "sichuan",
                    "--speak", text,
                    "--lang", "zh",
                    "--save-audio", str(path)
                ]

                result = subprocess.run(cmd, capture_output=True, text=True,
                                         timeout=120, cwd=str(STREAM_TTS_CPP), env=get_tts_env())
                assert result.returncode == 0, f"Synthesis {i} failed: {result.stderr}"

                info = read_wav_info(path)
                assert info["rms"] > 0.002, f"Audio {i} is silent"
        finally:
            for p in paths:
                if p.exists():
                    p.unlink()


class TestCosyVoice2ErrorHandling:
    """Test error handling and edge cases."""

    def test_empty_text_handling(self):
        """Test graceful handling of empty text."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = Path(f.name)

        try:
            cmd = [
                str(TTS_BINARY),
                "--voice-name", "sichuan",
                "--speak", "",  # Empty text
                "--lang", "zh",
                "--save-audio", str(output_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True,
                                     timeout=30, cwd=str(STREAM_TTS_CPP), env=get_tts_env())

            # Should either fail gracefully or produce minimal audio
            # Not asserting specific behavior, just that it doesn't hang
        finally:
            if output_path.exists():
                output_path.unlink()

    def test_long_text_handling(self):
        """Test synthesis of long Chinese text."""
        long_text = "这是一个很长的句子，" * 10  # Repeat to make long

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = Path(f.name)

        try:
            cmd = [
                str(TTS_BINARY),
                "--voice-name", "sichuan",
                "--speak", long_text,
                "--lang", "zh",
                "--save-audio", str(output_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True,
                                     timeout=180,  # Longer timeout for long text
                                     cwd=str(STREAM_TTS_CPP), env=get_tts_env())

            assert result.returncode == 0, f"Long text synthesis failed: {result.stderr}"
            info = read_wav_info(output_path)
            assert info["rms"] > 0.002, "Audio is silent"
            assert info["duration_sec"] >= 3.0, "Audio too short for long text"
        finally:
            if output_path.exists():
                output_path.unlink()

    def test_missing_voice_error(self):
        """Test error handling for non-existent voice."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = Path(f.name)

        try:
            cmd = [
                str(TTS_BINARY),
                "--voice-name", "nonexistent_voice_xyz",
                "--speak", "Hello",
                "--lang", "en",
                "--save-audio", str(output_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True,
                                     timeout=30, cwd=str(STREAM_TTS_CPP), env=get_tts_env())

            # Should fail with error message
            # (Some implementations may fall back to default voice)
        finally:
            if output_path.exists():
                output_path.unlink()


class TestCosyVoice2UnicodeHandling:
    """Test Unicode text handling for Chinese and special characters."""

    UNICODE_TEST_CASES = [
        ("你好世界", "Basic Chinese"),
        ("今天天气真好！", "Chinese with punctuation"),
        ("北京、上海、广州", "Chinese with special punctuation"),
        ("中文English混合", "Chinese-English mix"),
        ("数字123测试", "Chinese with numbers"),
        ('引号"测试"文本', "Chinese with quotes"),
    ]

    @pytest.mark.parametrize("text,description", UNICODE_TEST_CASES)
    def test_unicode_text_synthesis(self, text, description):
        """Test proper Unicode handling for various Chinese text patterns."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = Path(f.name)

        try:
            cmd = [
                str(TTS_BINARY),
                "--voice-name", "sichuan",
                "--speak", text,
                "--lang", "zh",
                "--save-audio", str(output_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True,
                                     timeout=120, cwd=str(STREAM_TTS_CPP), env=get_tts_env())

            assert result.returncode == 0, f"Failed for {description}: {result.stderr}"
            assert output_path.exists()

            info = read_wav_info(output_path)
            assert info["rms"] > 0.002, f"Audio is silent for {description}"
        finally:
            if output_path.exists():
                output_path.unlink()


class TestCosyVoice2InstructionVariations:
    """Test various instruction mode variations."""

    INSTRUCTION_VARIATIONS = [
        ("用四川话说", "Sichuan dialect"),
        ("开心地说", "Happy emotion"),
        ("悲伤地说", "Sad emotion"),
        ("慢速地说", "Slow speed"),
        ("快速地说", "Fast speed"),
    ]

    @pytest.mark.parametrize("instruction,description", INSTRUCTION_VARIATIONS)
    def test_instruction_variation(self, instruction, description):
        """Test various instruction patterns produce valid audio."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = Path(f.name)

        try:
            cmd = [
                str(TTS_BINARY),
                "--voice-name", "cosy",
                "--speak", "你好",
                "--lang", "zh",
                "--instruction", instruction,
                "--save-audio", str(output_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True,
                                     timeout=120, cwd=str(STREAM_TTS_CPP), env=get_tts_env())

            assert result.returncode == 0, f"Failed for {description}: {result.stderr}"
            info = read_wav_info(output_path)
            assert info["rms"] > 0.002, f"Audio is silent for {description}"
        finally:
            if output_path.exists():
                output_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
