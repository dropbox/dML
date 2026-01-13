#!/usr/bin/env python3
"""
CosyVoice2 Instruction Override Integration Tests

Verifies that different instruction tags produce detectably different
emotional tones using LLM-as-judge evaluation.

This test ensures the --instruction CLI flag actually affects the synthesized
audio output, not just passes silently through the system.

Worker #551: Created to prove instruction system works after fixing
the set_instruction/set_voice ordering bug in main.cpp.

Copyright 2025 Andrew Yates. All rights reserved.
"""

import base64
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
STREAM_TTS_CPP = PROJECT_ROOT / "stream-tts-cpp"
BUILD_DIR = STREAM_TTS_CPP / "build"
TTS_BINARY = BUILD_DIR / "stream-tts-cpp"


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


def get_tts_env():
    """Get environment variables for TTS subprocess."""
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["OMP_NUM_THREADS"] = "1"
    return env


def read_wav(path: Path) -> tuple:
    """Read WAV file and return (samples as float array, sample_rate)."""
    import wave
    with wave.open(str(path), "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        samples = (
            np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        )
        return samples, wf.getframerate()


def synthesize_with_instruction(text: str, output_path: Path, voice: str = "sichuan", instruction: str = None) -> dict:
    """Run native C++ TTS synthesis with optional instruction."""
    cmd = [
        str(TTS_BINARY),
        "--voice-name", voice,
        "--speak", text,
        "--lang", "zh",
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
            "stdout": result.stdout,
        }

    if not output_path.exists():
        return {
            "success": False,
            "error": "Output file not created",
        }

    audio, sample_rate = read_wav(output_path)
    rms = float(np.sqrt(np.mean(audio**2)))

    return {
        "success": True,
        "rms": rms,
        "inference_time": inference_time,
    }


def llm_judge_emotion(audio_path: Path, expected_emotion: str) -> dict:
    """Use LLM-as-judge to evaluate emotional tone.

    Returns dict with: detected_tone, match_score, description.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or api_key.startswith("sk-..."):
        return {"detected_tone": None, "match_score": None, "error": "No API key"}

    try:
        from openai import OpenAI
    except ImportError:
        return {"detected_tone": None, "match_score": None, "error": "openai not installed"}

    with open(audio_path, "rb") as f:
        audio_data = base64.b64encode(f.read()).decode("utf-8")

    client = OpenAI(api_key=api_key)

    prompt = f"""Analyze this Chinese speech audio for EMOTIONAL TONE.

Rate how well the speech matches the expected emotion: {expected_emotion}

Score from 1-10:
- 10: Clear, unmistakable emotional tone matching expected
- 7: Noticeable emotional tone, somewhat matches
- 5: Neutral/unclear tone
- 3: Different emotional tone than expected
- 1: Completely wrong emotional expression

Output JSON only:
{{"detected_tone": "<what tone you hear>", "match_score": <1-10>, "description": "<brief explanation>"}}"""

    response = client.chat.completions.create(
        model="gpt-4o-audio-preview-2024-12-17",
        modalities=["text"],
        messages=[
            {"role": "system", "content": "You are an expert audio emotion analyst. Output only valid JSON."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "input_audio", "input_audio": {"data": audio_data, "format": "wav"}},
            ]},
        ],
        max_tokens=200,
    )

    raw = response.choices[0].message.content or ""

    try:
        # Handle markdown-wrapped JSON
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
                "detected_tone": parsed.get("detected_tone", "unknown"),
                "match_score": parsed.get("match_score", 0),
                "description": parsed.get("description", ""),
            }
    except Exception:
        pass

    return {"detected_tone": "error", "match_score": 0, "description": f"Parse error: {raw}"}


@pytest.mark.integration
@pytest.mark.llm_judge
@skip_no_binary
class TestInstructionOverride:
    """Tests verifying instruction tags produce different emotional outputs."""

    # Test cases: (instruction, expected_emotion, english_name)
    INSTRUCTION_CASES = [
        (None, "neutral", "default"),
        ("凶猛", "ferocious/fierce", "ferocious"),
        ("用开心的语气说", "happy", "happy"),
    ]

    TEST_TEXT = "你好，今天天气真好，我们一起去公园玩吧"

    @pytest.mark.parametrize("instruction,expected,name", INSTRUCTION_CASES)
    def test_instruction_produces_emotion(self, instruction, expected, name):
        """Test that a specific instruction produces the expected emotion."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = Path(f.name)

        try:
            result = synthesize_with_instruction(
                self.TEST_TEXT,
                output_path,
                instruction=instruction
            )
            assert result["success"], f"Synthesis failed: {result.get('error')}"
            assert result["rms"] > 0.001, f"Audio too quiet (RMS={result['rms']:.6f})"

            eval_result = llm_judge_emotion(output_path, expected)

            if eval_result.get("error"):
                pytest.skip(f"LLM judge unavailable: {eval_result['error']}")

            print(f"\n  Instruction: {instruction or 'default'} ({name})")
            print(f"  Expected: {expected}")
            print(f"  Detected: {eval_result['detected_tone']}")
            print(f"  Match score: {eval_result['match_score']}/10")
            print(f"  Description: {eval_result['description']}")

            # For emotional instructions (ferocious, happy), require high match
            if instruction is not None:
                assert eval_result["match_score"] >= 6, \
                    f"Emotion mismatch: expected {expected}, got {eval_result['detected_tone']} (score={eval_result['match_score']})"

        finally:
            if output_path.exists():
                output_path.unlink()

    def test_different_instructions_produce_different_tones(self):
        """Test that different instructions produce detectably different audio.

        This is the key integration test: if instructions don't affect output,
        all samples would have the same detected tone.
        """
        results = {}
        temp_files = []

        try:
            for instruction, expected, name in self.INSTRUCTION_CASES:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    output_path = Path(f.name)
                    temp_files.append(output_path)

                synth_result = synthesize_with_instruction(
                    self.TEST_TEXT,
                    output_path,
                    instruction=instruction
                )
                assert synth_result["success"], f"Synthesis failed for {name}: {synth_result.get('error')}"

                eval_result = llm_judge_emotion(output_path, expected)

                if eval_result.get("error"):
                    pytest.skip(f"LLM judge unavailable: {eval_result['error']}")

                results[name] = {
                    "instruction": instruction,
                    "expected": expected,
                    "detected_tone": eval_result["detected_tone"],
                    "match_score": eval_result["match_score"],
                }
                print(f"\n  {name}: detected={eval_result['detected_tone']} score={eval_result['match_score']}")

                time.sleep(1)  # Rate limit

            # Check that we got different detected tones
            detected_tones = [r["detected_tone"].lower() for r in results.values()]
            unique_tones = len(set(detected_tones))

            print(f"\n  SUMMARY:")
            print(f"  Detected tones: {detected_tones}")
            print(f"  Unique tones: {unique_tones}")

            # Require at least 2 distinct tones (proves instructions affect output)
            assert unique_tones >= 2, \
                f"Instructions not affecting output: all samples had similar tone ({detected_tones})"

            # Check at least one emotional instruction got a reasonable score
            # Note: Some instructions may not work well with certain voice prompts
            # (e.g., "ferocious" may not suit a grandmother voice)
            emotional_scores = [results[name]["match_score"] for name in ["ferocious", "happy"] if name in results]
            if emotional_scores:
                max_emotional_score = max(emotional_scores)
                print(f"  Max emotional score: {max_emotional_score}/10")
                # At least one emotional instruction should score >= 5
                assert max_emotional_score >= 5, \
                    f"No emotional instruction worked: max_score={max_emotional_score}"

        finally:
            for path in temp_files:
                if path.exists():
                    path.unlink()

    def test_instruction_in_log_output(self):
        """Test that CLI logs show the instruction being used."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = Path(f.name)

        try:
            cmd = [
                str(TTS_BINARY),
                "--voice-name", "sichuan",
                "--speak", "测试",
                "--lang", "zh",
                "--instruction", "凶猛",
                "--save-audio", str(output_path),
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(STREAM_TTS_CPP),
                env=get_tts_env(),
            )

            assert result.returncode == 0, f"Command failed: {result.stderr}"

            # Check stderr for instruction logging
            combined_output = result.stdout + result.stderr

            # Should see the instruction override log
            assert "凶猛" in combined_output or "instruction" in combined_output.lower(), \
                f"Instruction not logged in output:\n{combined_output[:500]}"

        finally:
            if output_path.exists():
                output_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
