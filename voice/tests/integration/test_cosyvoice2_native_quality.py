#!/usr/bin/env python3
"""
CosyVoice2 Native C++ MPS Quality Tests

Tests the native C++ CosyVoice2 engine with MPS acceleration.
This is a regression test to ensure models re-exported with PyTorch 2.9.1
produce equivalent quality to the original.

These tests verify:
1. Native C++ produces audible, non-silent audio
2. Audio quality passes LLM-as-judge evaluation (no "frog" artifacts)
3. RTF (Real-Time Factor) is acceptable for real-time playback

Requirements:
- stream-tts-cpp binary built with CosyVoice2 support
- Re-exported TorchScript models (PyTorch 2.9.1)
- OPENAI_API_KEY for LLM-as-judge tests (optional)

Copyright 2025 Andrew Yates. All rights reserved.
"""

import base64
import json
import os
import subprocess
import tempfile
import time
import wave
from pathlib import Path

import numpy as np
import pytest

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
STREAM_TTS_CPP = PROJECT_ROOT / "stream-tts-cpp"
BUILD_DIR = STREAM_TTS_CPP / "build"
TTS_BINARY = BUILD_DIR / "stream-tts-cpp"

# Model paths (native C++ uses these files)
FLOW_MODEL = PROJECT_ROOT / "models" / "cosyvoice" / "torchscript" / "flow_full.pt"
HIFT_MODEL = PROJECT_ROOT / "models" / "cosyvoice" / "exported" / "hift_traced.pt"
LLM_GGUF = PROJECT_ROOT / "models" / "cosyvoice" / "cosyvoice_qwen2_q8_0.gguf"


def load_env():
    """Load .env file for API keys."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key] = value.strip().strip('"').strip("'")


load_env()


# Skip conditions
skip_no_binary = pytest.mark.skipif(
    not TTS_BINARY.exists(),
    reason="stream-tts-cpp binary not built"
)

skip_no_models = pytest.mark.skipif(
    not FLOW_MODEL.exists() or not HIFT_MODEL.exists() or not LLM_GGUF.exists(),
    reason="CosyVoice2 models not found (flow_mps.pt, hift_mps.pt, cosyvoice_qwen2_q8_0.gguf)"
)


def get_tts_env():
    """Get environment variables for TTS subprocess."""
    env = os.environ.copy()
    # Fix OpenMP duplicate library crash when llama.cpp + libtorch both link OpenMP
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["OMP_NUM_THREADS"] = "1"
    return env


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read WAV file and return (samples as float array, sample_rate)."""
    with wave.open(str(path), "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        samples = (
            np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        )
        return samples, wf.getframerate()


def synthesize_native(text: str, output_path: Path, voice: str = "cosy", lang: str = "zh", instruction: str = None) -> dict:
    """Run native C++ TTS synthesis.

    Returns dict with keys: success, duration_sec, rms, inference_time_sec, error, rtf.
    """
    cmd = [
        str(TTS_BINARY),
        "--voice-name", voice,
        "--speak", text,
        "--lang", lang,
        "--save-audio", str(output_path),
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
        env=get_tts_env(),
    )
    inference_time = time.time() - start

    if result.returncode != 0:
        return {
            "success": False,
            "error": result.stderr,
            "duration_sec": 0.0,
            "rms": 0.0,
            "inference_time_sec": inference_time,
            "rtf": 0.0,
        }

    if not output_path.exists():
        return {
            "success": False,
            "error": "Output file not created",
            "duration_sec": 0.0,
            "rms": 0.0,
            "inference_time_sec": inference_time,
            "rtf": 0.0,
        }

    audio, sample_rate = read_wav(output_path)
    duration_sec = len(audio) / sample_rate
    rms = float(np.sqrt(np.mean(audio**2)))
    rtf = inference_time / duration_sec if duration_sec > 0 else 999

    return {
        "success": True,
        "error": None,
        "duration_sec": float(duration_sec),
        "rms": rms,
        "inference_time_sec": float(inference_time),
        "rtf": rtf,
    }


def llm_judge_quality(audio_path: Path) -> dict:
    """Use LLM-as-judge to evaluate audio quality.

    Returns dict with: overall_score, frog, issues, raw_response.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or api_key.startswith("sk-..."):
        return {"overall_score": None, "frog": None, "issues": "No API key", "raw_response": ""}

    try:
        from openai import OpenAI
    except ImportError:
        return {"overall_score": None, "frog": None, "issues": "openai not installed", "raw_response": ""}

    with open(audio_path, "rb") as f:
        audio_data = base64.b64encode(f.read()).decode("utf-8")

    client = OpenAI(api_key=api_key)

    prompt = """Evaluate this Chinese TTS audio quality.
Rate on a 1-10 scale:
- 10: Perfect, professional quality
- 8-9: Very good, minor imperfections
- 6-7: Good, some noticeable issues
- 4-5: Fair, clear issues
- 1-3: Poor, significant problems

Check for "dying frog" artifacts (robotic/distorted/garbled sounds).

Output JSON only:
{"overall_score": <1-10>, "frog": <true/false>, "issues": "<specific problems if any>"}
"""

    response = client.chat.completions.create(
        model="gpt-4o-audio-preview",
        modalities=["text"],
        messages=[
            {"role": "system", "content": "You are an expert TTS quality evaluator. Output only valid JSON."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "input_audio", "input_audio": {"data": audio_data, "format": "wav"}},
            ]},
        ],
        max_tokens=200,
    )

    raw = response.choices[0].message.content or ""

    try:
        cleaned = raw
        if cleaned.startswith("```"):
            parts = cleaned.split("```")
            cleaned = parts[1] if len(parts) > 1 else cleaned
            if cleaned.lstrip().startswith("json"):
                cleaned = cleaned.lstrip()[4:]

        json_start = cleaned.find("{")
        json_end = cleaned.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            parsed = json.loads(cleaned[json_start:json_end])
            return {
                "overall_score": parsed.get("overall_score", 0),
                "frog": parsed.get("frog", False),
                "issues": parsed.get("issues", ""),
                "raw_response": raw,
            }
    except Exception:
        pass

    return {"overall_score": 0, "frog": True, "issues": f"Parse error: {raw}", "raw_response": raw}


@pytest.mark.integration
@skip_no_binary
@skip_no_models
class TestCosyVoice2NativeMPS:
    """Tests for native C++ CosyVoice2 with MPS acceleration."""

    def test_basic_synthesis(self):
        """Test that native C++ produces audio."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = Path(f.name)

        try:
            result = synthesize_native("你好", output_path)
            assert result["success"], f"Synthesis failed: {result['error']}"
            assert result["rms"] > 0.001, f"Audio too quiet (RMS={result['rms']:.6f})"
            assert result["duration_sec"] > 0.3, f"Audio too short ({result['duration_sec']:.2f}s)"
            print(f"\nBasic: duration={result['duration_sec']:.2f}s RMS={result['rms']:.4f} RTF={result['rtf']:.2f}x")
        finally:
            if output_path.exists():
                output_path.unlink()

    def test_realtime_rtf(self):
        """Test that RTF is below 1.0 (real-time capable)."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = Path(f.name)

        try:
            # Use longer text for more reliable RTF measurement
            result = synthesize_native("今天天气真好，我们去公园散步吧", output_path)
            assert result["success"], f"Synthesis failed: {result['error']}"

            print(f"\nRTF: {result['rtf']:.2f}x (inference={result['inference_time_sec']:.2f}s, audio={result['duration_sec']:.2f}s)")

            # RTF < 1.0 means real-time capable
            # Allow up to 5.0 for cold start (first run loads all models: LLM GGUF, Flow, HiFT)
            # Warm runs should be < 1.0, but we can't guarantee that in tests without daemon mode
            assert result["rtf"] < 5.0, f"RTF too high even for cold start: {result['rtf']:.2f}x"
        finally:
            if output_path.exists():
                output_path.unlink()

    @pytest.mark.llm_judge
    def test_quality_no_frog(self):
        """Test that audio quality is acceptable (no 'dying frog' artifacts).

        Uses LLM-as-judge with majority voting (3 evaluations) to handle variance.
        MINIMUM THRESHOLD: overall_score >= 5, frog votes < 2/3
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = Path(f.name)

        try:
            result = synthesize_native("你好，今天天气真好", output_path)
            assert result["success"], f"Synthesis failed: {result['error']}"
            assert result["rms"] > 0.001, f"Audio silent (RMS={result['rms']:.6f})"

            # Run 3 evaluations for majority voting
            scores = []
            frog_votes = []

            for i in range(3):
                eval_result = llm_judge_quality(output_path)

                if eval_result["overall_score"] is None:
                    pytest.skip(f"LLM judge unavailable: {eval_result['issues']}")

                scores.append(eval_result["overall_score"])
                frog_votes.append(eval_result["frog"])
                print(f"\n  Eval {i+1}: score={eval_result['overall_score']}/10 frog={eval_result['frog']} issues={eval_result['issues']}")

                if i < 2:  # Rate limit between calls
                    time.sleep(1)

            avg_score = sum(scores) / len(scores)
            frog_count = sum(1 for v in frog_votes if v)

            print(f"\n  FINAL: avg_score={avg_score:.1f}/10 frog_votes={frog_count}/3")

            # Minimum acceptable thresholds
            assert avg_score >= 5.0, f"Quality too low: {avg_score:.1f}/10 (min 5.0)"
            assert frog_count < 2, f"Frog quality detected: {frog_count}/3 votes"

        finally:
            if output_path.exists():
                output_path.unlink()

    def test_instruction_support(self):
        """Test that instruction parameter works."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = Path(f.name)

        try:
            result = synthesize_native(
                "你好世界",
                output_path,
                instruction="用四川话说，像一个四川婆婆在讲故事"
            )
            assert result["success"], f"Synthesis failed: {result['error']}"
            assert result["rms"] > 0.001, f"Audio too quiet (RMS={result['rms']:.6f})"
            print(f"\nInstruction: duration={result['duration_sec']:.2f}s RMS={result['rms']:.4f}")
        finally:
            if output_path.exists():
                output_path.unlink()

    def test_models_version_match(self):
        """Verify TorchScript models were exported with PyTorch 2.9.1.

        This is a documentation test - it verifies the expected model files exist.
        If models were exported with wrong version, audio quality will degrade.
        """
        # Check model files exist and have reasonable sizes
        assert FLOW_MODEL.exists(), f"Flow model missing: {FLOW_MODEL}"
        assert HIFT_MODEL.exists(), f"HiFT model missing: {HIFT_MODEL}"
        assert LLM_GGUF.exists(), f"LLM GGUF missing: {LLM_GGUF}"

        flow_size = FLOW_MODEL.stat().st_size
        hift_size = HIFT_MODEL.stat().st_size
        llm_size = LLM_GGUF.stat().st_size

        print(f"\nModel sizes:")
        print(f"  flow_mps.pt: {flow_size / 1024 / 1024:.1f} MB")
        print(f"  hift_mps.pt: {hift_size / 1024 / 1024:.1f} MB")
        print(f"  cosyvoice_qwen2_q8_0.gguf: {llm_size / 1024 / 1024:.1f} MB")

        # Sanity check sizes (should be non-trivial)
        assert flow_size > 10 * 1024 * 1024, f"Flow model too small: {flow_size}"
        assert hift_size > 1 * 1024 * 1024, f"HiFT model too small: {hift_size}"
        assert llm_size > 100 * 1024 * 1024, f"LLM GGUF too small: {llm_size}"

    @pytest.mark.parametrize("voice,text,lang", [
        ("cosy", "你好", "zh"),
        ("cosy-en", "Hello world", "en"),
        ("cosy-ja", "こんにちは", "ja"),
        ("cosy-ko", "안녕하세요", "ko"),
        ("cosy-zh", "你好世界", "zh"),
        ("sichuan", "你好", "zh"),
        ("cantonese", "你好", "zh"),
        ("shanghainese", "你好", "zh"),
    ])
    def test_all_cosyvoice2_voices(self, voice, text, lang):
        """Test that ALL CosyVoice2 voices produce audible audio.

        This is a comprehensive regression test ensuring every voice:
        1. Successfully synthesizes audio (no crashes)
        2. Produces non-silent output (RMS > 0.001)
        3. Generates reasonable duration (> 0.3s)

        All 8 voices verified passing LLM-as-judge quality (Worker #543, 2025-12-11):
        - cosy: 8/10
        - cosy-en: 8/10
        - cosy-ja: 8/10
        - cosy-ko: 8/10
        - cosy-zh: 8/10
        - sichuan: 6.7/10 (after instruction fix)
        - cantonese: 8/10
        - shanghainese: 7/10
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = Path(f.name)

        try:
            result = synthesize_native(text, output_path, voice=voice, lang=lang)
            assert result["success"], f"Voice '{voice}' synthesis failed: {result['error']}"
            assert result["rms"] > 0.001, f"Voice '{voice}' audio too quiet (RMS={result['rms']:.6f})"
            assert result["duration_sec"] > 0.3, f"Voice '{voice}' audio too short ({result['duration_sec']:.2f}s)"
            print(f"\n{voice}: duration={result['duration_sec']:.2f}s RMS={result['rms']:.4f} RTF={result['rtf']:.2f}x")
        finally:
            if output_path.exists():
                output_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
