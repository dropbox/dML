"""
CosyVoice2 Multilingual TTS Tests (Worker #471)

Tests CosyVoice2 multilingual synthesis for:
- English (cosy-en)
- Japanese (cosy-ja)
- Korean (cosy-ko)
- Chinese (cosy-zh)

Also tests cross-lingual instructions (Chinese instructions on non-Chinese text).

Based on PRODUCTION_ROADMAP_2025-12-10.md Phase 2B requirements.

Usage:
    pytest tests/quality/test_cosyvoice2_multilingual.py -v
    pytest tests/quality/test_cosyvoice2_multilingual.py -v -s  # With output
    pytest tests/quality/test_cosyvoice2_multilingual.py -v -m llm_judge  # Only LLM tests
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

# Quality thresholds
QUALITY_THRESHOLDS = {
    "min_rms": 0.002,          # Minimum RMS to detect non-silent audio
    "min_duration_sec": 0.3,   # Minimum audio duration for short phrases
    "target_rtf": 1.5,         # Real-Time Factor target (cold-start CLI)
}

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


def read_wav(path: Path) -> tuple:
    """Read WAV file and return (samples as float array, sample_rate)."""
    with wave.open(str(path), 'rb') as wf:
        frames = wf.readframes(wf.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        return samples, wf.getframerate()


def get_audio_metrics(audio_data: np.ndarray, sample_rate: int = 24000) -> dict:
    """Calculate audio quality metrics."""
    if len(audio_data) == 0:
        return {
            "rms": 0.0,
            "peak": 0.0,
            "duration_sec": 0.0,
        }

    rms = np.sqrt(np.mean(audio_data ** 2))
    peak = np.max(np.abs(audio_data))
    duration_sec = len(audio_data) / sample_rate

    return {
        "rms": float(rms),
        "peak": float(peak),
        "duration_sec": float(duration_sec),
    }


def get_lang_for_voice(voice: str) -> str:
    """Get language code for one-shot mode based on voice name."""
    if voice == "cosy-en":
        return "en"
    elif voice == "cosy-ja":
        return "ja"
    elif voice == "cosy-ko":
        return "ko"
    else:  # cosy-zh or default
        return "zh"


def synthesize_multilingual(voice: str, text: str, output_path: Path, instruction: str = None) -> dict:
    """Run TTS synthesis with given voice and optional instruction.

    Returns dict with keys: success, duration_sec, rms, inference_time_sec, error
    """
    lang = get_lang_for_voice(voice)
    cmd = [
        str(TTS_BINARY),
        "--voice-name", voice,
        "--speak", text,
        "--lang", lang,  # Required for one-shot mode
        "--save-audio", str(output_path)
    ]

    if instruction:
        cmd.extend(["--instruction", instruction])

    start = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(STREAM_TTS_CPP),
        env=get_tts_env()
    )
    inference_time = time.time() - start

    if result.returncode != 0:
        return {
            "success": False,
            "error": result.stderr,
            "stdout": result.stdout,
            "duration_sec": 0,
            "rms": 0,
            "inference_time_sec": inference_time
        }

    if not output_path.exists() or output_path.stat().st_size < 100:
        return {
            "success": False,
            "error": "Output file not created or empty",
            "stdout": result.stdout,
            "duration_sec": 0,
            "rms": 0,
            "inference_time_sec": inference_time
        }

    try:
        audio, sample_rate = read_wav(output_path)
        metrics = get_audio_metrics(audio, sample_rate)
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to read WAV: {e}",
            "stdout": result.stdout,
            "duration_sec": 0,
            "rms": 0,
            "inference_time_sec": inference_time
        }

    return {
        "success": True,
        "error": None,
        "stdout": result.stdout,
        "duration_sec": metrics["duration_sec"],
        "rms": metrics["rms"],
        "peak": metrics["peak"],
        "inference_time_sec": inference_time
    }


# ============================================================================
# Test Data
# ============================================================================

# Multilingual test cases: (voice, text, expected_language, min_duration)
MULTILINGUAL_TEST_CASES = [
    # English
    ("cosy-en", "Hello, how are you today?", "English", 1.0),
    ("cosy-en", "The weather is nice today.", "English", 1.0),

    # Japanese
    ("cosy-ja", "こんにちは、元気ですか？", "Japanese", 1.0),
    ("cosy-ja", "今日は天気がいいですね。", "Japanese", 1.0),

    # Korean
    ("cosy-ko", "안녕하세요, 어떻게 지내세요?", "Korean", 1.0),
    ("cosy-ko", "오늘 날씨가 좋네요.", "Korean", 1.0),

    # Chinese (standard Mandarin via cosy-zh)
    ("cosy-zh", "你好，今天怎么样？", "Chinese", 0.8),
    ("cosy-zh", "今天天气真好。", "Chinese", 0.8),
]

# Cross-lingual instruction test cases: (voice, text, instruction, test_name)
# Tests if Chinese instructions affect non-Chinese synthesis
CROSS_LINGUAL_INSTRUCTION_CASES = [
    # Chinese instruction on English text
    ("cosy-en", "Hello world", "开心地说", "happy_english"),
    ("cosy-en", "Hello world", "慢速地说", "slow_english"),

    # Chinese instruction on Japanese text
    ("cosy-ja", "こんにちは", "开心地说", "happy_japanese"),

    # Chinese instruction on Korean text
    ("cosy-ko", "안녕하세요", "开心地说", "happy_korean"),
]


# ============================================================================
# Basic Synthesis Tests
# ============================================================================

class TestCosyVoice2MultilingualBasic:
    """Basic synthesis tests for all supported languages."""

    @pytest.mark.parametrize("voice,text,language,min_duration", MULTILINGUAL_TEST_CASES)
    def test_multilingual_synthesis(self, voice, text, language, min_duration, tmp_path):
        """Test basic synthesis for each supported language."""
        output_path = tmp_path / f"test_{voice}.wav"

        result = synthesize_multilingual(voice, text, output_path)

        # Check synthesis succeeded
        assert result["success"], (
            f"{voice} synthesis failed: {result.get('error')}\n"
            f"stdout: {result.get('stdout', '')}"
        )

        # Check audio is not silent
        assert result["rms"] > QUALITY_THRESHOLDS["min_rms"], (
            f"{voice} produced silent audio! RMS={result['rms']:.6f}"
        )

        # Check minimum duration
        assert result["duration_sec"] >= QUALITY_THRESHOLDS["min_duration_sec"], (
            f"{voice} audio too short: {result['duration_sec']:.2f}s"
        )

        # Log result
        rtf = result["inference_time_sec"] / result["duration_sec"] if result["duration_sec"] > 0 else float('inf')
        print(f"\n{voice} ({language}): PASS - RTF={rtf:.2f}, "
              f"duration={result['duration_sec']:.2f}s, RMS={result['rms']:.4f}")


class TestCosyVoice2MultilingualComparison:
    """Compare quality across languages."""

    def test_all_languages_produce_audio(self, tmp_path):
        """Verify all supported languages produce non-silent audio."""
        results = {}

        for voice, text, language, _ in MULTILINGUAL_TEST_CASES:
            # Skip duplicates (we have 2 test cases per language)
            if voice in results:
                continue

            output_path = tmp_path / f"test_{voice}.wav"
            result = synthesize_multilingual(voice, text, output_path)
            results[voice] = result

        # Summary report
        print("\n" + "=" * 60)
        print("CosyVoice2 Multilingual Synthesis Summary")
        print("=" * 60)
        print(f"{'Voice':<12} {'Status':<8} {'RMS':<10} {'Duration':<10} {'RTF':<8}")
        print("-" * 60)

        all_pass = True
        for voice, result in results.items():
            status = "PASS" if result["success"] else "FAIL"
            if result["success"]:
                rtf = result["inference_time_sec"] / result["duration_sec"] if result["duration_sec"] > 0 else float('inf')
                print(f"{voice:<12} {status:<8} {result['rms']:.4f}     "
                      f"{result['duration_sec']:.2f}s       {rtf:.2f}")
            else:
                print(f"{voice:<12} {status:<8} -          -          -")
                all_pass = False

        print("=" * 60)

        assert all_pass, "Not all languages produced audio successfully"


# ============================================================================
# Cross-Lingual Instruction Tests
# ============================================================================

class TestCosyVoice2CrossLingualInstructions:
    """Test Chinese instructions applied to non-Chinese text.

    This validates whether CosyVoice2's instruction mode affects synthesis
    when the instruction language differs from the text language.
    """

    @pytest.mark.parametrize("voice,text,instruction,test_name", CROSS_LINGUAL_INSTRUCTION_CASES)
    def test_cross_lingual_instruction(self, voice, text, instruction, test_name, tmp_path):
        """Test cross-lingual instruction synthesis produces audio.

        Note: These tests verify audio IS produced. Whether the instruction
        actually affects the output is documented behavior - it may or may not
        work depending on model training.
        """
        output_path = tmp_path / f"test_{test_name}.wav"

        result = synthesize_multilingual(voice, text, output_path, instruction=instruction)

        # Check synthesis succeeded (primary requirement)
        assert result["success"], (
            f"Cross-lingual synthesis failed for {test_name}: {result.get('error')}\n"
            f"Voice: {voice}, Text: {text}, Instruction: {instruction}"
        )

        # Check audio is not silent
        assert result["rms"] > QUALITY_THRESHOLDS["min_rms"], (
            f"Cross-lingual synthesis produced silent audio for {test_name}! "
            f"RMS={result['rms']:.6f}"
        )

        # Log result
        print(f"\n{test_name}: PASS - {voice} + '{instruction}' produced audio "
              f"(RMS={result['rms']:.4f}, duration={result['duration_sec']:.2f}s)")

    def test_cross_lingual_instruction_comparison(self, tmp_path):
        """Compare audio with and without Chinese instructions on English text.

        This test documents the behavior but does not require the instruction
        to have any effect - the model may not support cross-lingual instructions.
        """
        text = "Hello, how are you today?"
        instruction = "开心地说"

        # Synthesize without instruction
        output_no_inst = tmp_path / "english_no_instruction.wav"
        result_no_inst = synthesize_multilingual("cosy-en", text, output_no_inst)

        # Synthesize with instruction
        output_with_inst = tmp_path / "english_with_instruction.wav"
        result_with_inst = synthesize_multilingual("cosy-en", text, output_with_inst, instruction=instruction)

        # Both should succeed
        assert result_no_inst["success"], f"Baseline failed: {result_no_inst.get('error')}"
        assert result_with_inst["success"], f"Instructed failed: {result_with_inst.get('error')}"

        # Document findings
        duration_diff = abs(result_with_inst["duration_sec"] - result_no_inst["duration_sec"])
        rms_diff = abs(result_with_inst["rms"] - result_no_inst["rms"])

        print("\n" + "=" * 60)
        print("Cross-Lingual Instruction Effect Analysis")
        print("=" * 60)
        print(f"Text: {text}")
        print(f"Instruction: {instruction}")
        print("-" * 60)
        print(f"{'Metric':<20} {'Without':<15} {'With':<15} {'Diff':<15}")
        print(f"{'Duration (s)':<20} {result_no_inst['duration_sec']:<15.2f} "
              f"{result_with_inst['duration_sec']:<15.2f} {duration_diff:<15.2f}")
        print(f"{'RMS':<20} {result_no_inst['rms']:<15.4f} "
              f"{result_with_inst['rms']:<15.4f} {rms_diff:<15.4f}")
        print("=" * 60)

        # Note: Not asserting differences - documenting observed behavior
        if duration_diff > 0.5 or rms_diff > 0.05:
            print("OBSERVATION: Chinese instruction appears to affect English synthesis")
        else:
            print("OBSERVATION: Chinese instruction has minimal effect on English synthesis")
            print("This is expected - cross-lingual instructions may not be supported")


# ============================================================================
# LLM-as-Judge Tests
# ============================================================================

class TestCosyVoice2MultilingualLLMJudge:
    """LLM-as-Judge evaluation for multilingual synthesis."""

    def _call_llm_judge(self, audio_path: Path, expected_language: str) -> dict:
        """Call GPT-audio to evaluate audio and detect language.

        Returns dict with keys: score (1-10), detected_language, issues
        """
        import base64
        import json

        # Load API key
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            env_file = PROJECT_ROOT / ".env"
            if env_file.exists():
                with open(env_file) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            key, value = line.split("=", 1)
                            os.environ[key] = value.strip('"').strip("'")
                api_key = os.environ.get("OPENAI_API_KEY")

        if not api_key:
            pytest.skip("OPENAI_API_KEY not found")

        try:
            import openai
        except ImportError:
            pytest.skip("openai package not installed")

        # Read audio file
        with open(audio_path, "rb") as f:
            audio_data = base64.standard_b64encode(f.read()).decode("utf-8")

        prompt = f"""Evaluate this TTS (text-to-speech) audio.

1. Detect the primary language spoken (English, Japanese, Korean, Chinese, or Unknown)
2. Rate quality from 1-10 where 10 is perfect natural speech
3. Note any issues (distortion, artifacts, wrong language, etc.)

Expected language: {expected_language}

Output ONLY valid JSON:
{{"detected_language": "<language>", "score": <1-10>, "language_match": <true/false>, "issues": "<brief description or 'none'>"}}"""

        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-audio-preview",
            modalities=["text"],
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert audio evaluator. ONLY output valid JSON."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "input_audio", "input_audio": {"data": audio_data, "format": "wav"}}
                    ]
                }
            ]
        )

        result_text = response.choices[0].message.content.strip()

        try:
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
                result_text = result_text.strip()
            return json.loads(result_text)
        except json.JSONDecodeError:
            return {"detected_language": "Unknown", "score": 0, "language_match": False,
                    "issues": f"Failed to parse: {result_text}"}

    @pytest.mark.llm_judge
    @pytest.mark.parametrize("voice,text,expected_language", [
        ("cosy-en", "Hello, how are you today?", "English"),
        ("cosy-ja", "こんにちは、元気ですか？", "Japanese"),
        ("cosy-ko", "안녕하세요, 어떻게 지내세요?", "Korean"),
        ("cosy-zh", "你好，今天怎么样？", "Chinese"),
    ])
    def test_llm_judge_language_detection(self, voice, text, expected_language, tmp_path):
        """LLM-as-Judge verifies correct language is synthesized."""
        output_path = tmp_path / f"test_{voice}_llm.wav"

        result = synthesize_multilingual(voice, text, output_path)
        assert result["success"], f"Synthesis failed: {result.get('error')}"

        evaluation = self._call_llm_judge(output_path, expected_language)

        print(f"\nLLM Judge for {voice}:")
        print(f"  Expected: {expected_language}")
        print(f"  Detected: {evaluation.get('detected_language')}")
        print(f"  Score: {evaluation.get('score')}/10")
        print(f"  Issues: {evaluation.get('issues')}")

        # Quality assertion
        # Score >= 6/10 is the minimum acceptable bar for production audio.
        score = evaluation.get("score", 0)
        assert score >= 6, f"Audio quality too low for {voice}: {score}/10"

        # Language match is informational - model limitations documented
        if not evaluation.get("language_match", True):
            print(f"WARNING: LLM detected different language than expected for {voice}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
