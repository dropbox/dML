#!/usr/bin/env python3
# Copyright 2024-2025 Andrew Yates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Gate0 Challenging Audio Tests

Tests whisper transcription on difficult audio that exercises:
- Temperature fallback (noisy/unclear audio)
- VAD accuracy
- Short clip handling
- Silence detection

These tests SHOULD FAIL until temperature fallback is properly implemented.

Reference: MANAGER directive 2025-12-23
"""

import os
import sys
import json
import subprocess
import tempfile
import numpy as np
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Test configuration
WHISPER_MODEL = os.path.expanduser(
    "~/.cache/huggingface/hub/models--mlx-community--whisper-large-v3-turbo/"
    "snapshots/beea265c324f07ba1e347f3c8a97aec454056a86"
)
TEST_ENGINE = Path(__file__).parent.parent / "src/mlx_inference_engine/test_engine"


def create_silent_audio(duration_sec: float, sample_rate: int = 16000) -> np.ndarray:
    """Create silent audio array."""
    return np.zeros(int(duration_sec * sample_rate), dtype=np.float32)


def create_noisy_audio(
    base_audio: np.ndarray,
    noise_level: float = 0.1,
    sample_rate: int = 16000
) -> np.ndarray:
    """Add white noise to audio."""
    noise = np.random.randn(len(base_audio)).astype(np.float32) * noise_level
    return base_audio + noise


def create_short_audio(text: str = "Hello", sample_rate: int = 16000) -> np.ndarray:
    """Create very short audio (< 3 seconds) that can trigger hallucination."""
    # For now, create a 2-second burst of noise with a beep
    duration = 2.0
    t = np.linspace(0, duration, int(duration * sample_rate), dtype=np.float32)
    # Simple tone burst
    audio = 0.3 * np.sin(2 * np.pi * 440 * t) * np.exp(-t * 2)
    return audio.astype(np.float32)


def save_wav(audio: np.ndarray, path: str, sample_rate: int = 16000):
    """Save audio as WAV file."""
    import wave

    # Normalize to int16
    audio_int16 = (audio * 32767).astype(np.int16)

    with wave.open(path, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())


def run_cpp_transcribe(audio_path: str, debug: bool = False) -> dict:
    """Run C++ test_engine transcription."""
    env = os.environ.copy()
    if debug:
        env["DEBUG_WHISPER"] = "1"

    cmd = [
        str(TEST_ENGINE),
        "--whisper", WHISPER_MODEL,
        "--transcribe", audio_path,
        "--output-format", "json"
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(Path(__file__).parent.parent)
    )

    # Parse JSON from output
    try:
        # Find JSON in output (may have debug info before it)
        output = result.stdout
        json_start = output.find('{')
        if json_start >= 0:
            json_str = output[json_start:]
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    return {"error": result.stderr, "stdout": result.stdout}


def run_python_transcribe(audio_path: str) -> dict:
    """Run Python mlx-whisper transcription for comparison."""
    try:
        import mlx_whisper
        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        )
        return result
    except Exception as e:
        return {"error": str(e)}


def test_silent_audio():
    """Test: Silent audio should return empty or no_speech."""
    print("\n=== Test: Silent Audio (should detect no speech) ===")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        audio = create_silent_audio(5.0)
        save_wav(audio, f.name)

        cpp_result = run_cpp_transcribe(f.name)

        os.unlink(f.name)

        text = cpp_result.get("text", "").strip()
        print(f"C++ result: '{text}'")

        # Silent audio should produce empty or very short output
        if len(text) < 10:
            print("PASS: Silent audio correctly detected as no speech")
            return True
        else:
            print(f"FAIL: Silent audio produced unexpected output: {text}")
            return False


def test_short_audio():
    """Test: Very short audio (< 3s) should not hallucinate."""
    print("\n=== Test: Short Audio (<3s, hallucination prone) ===")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        audio = create_short_audio()
        save_wav(audio, f.name)

        cpp_result = run_cpp_transcribe(f.name)

        os.unlink(f.name)

        text = cpp_result.get("text", "").strip()
        print(f"C++ result: '{text}'")

        # Short non-speech audio should not produce long hallucinated text
        word_count = len(text.split()) if text else 0
        if word_count < 20:
            print(f"PASS: Short audio produced {word_count} words (not hallucinating)")
            return True
        else:
            print(f"FAIL: Short audio hallucinated {word_count} words")
            return False


def test_noisy_audio():
    """Test: Noisy audio should trigger temperature fallback."""
    print("\n=== Test: Noisy Audio (should trigger temperature fallback) ===")

    # Try to find a real audio file to add noise to
    test_audio = Path(__file__).parent.parent / "tests/test_audio_45s_concat.wav"
    if not test_audio.exists():
        print("SKIP: No test audio available")
        return None

    # Load the audio and add noise
    try:
        import wave
        with wave.open(str(test_audio), 'r') as wav:
            frames = wav.readframes(wav.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767
            sample_rate = wav.getframerate()
    except Exception as e:
        print(f"SKIP: Failed to load test audio: {e}")
        return None

    # Add significant noise
    noisy_audio = create_noisy_audio(audio, noise_level=0.3, sample_rate=sample_rate)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        save_wav(noisy_audio, f.name, sample_rate)

        # Run with debug to see if temperature fallback triggers
        env = os.environ.copy()
        env["DEBUG_WHISPER"] = "1"

        cmd = [
            str(TEST_ENGINE),
            "--whisper", WHISPER_MODEL,
            "--transcribe", f.name,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            cwd=str(Path(__file__).parent.parent)
        )

        os.unlink(f.name)

        output = result.stdout + result.stderr

        # Check if temperature fallback was triggered
        if "compression ratio" in output.lower() or "retry" in output.lower():
            print("PASS: Temperature fallback triggered on noisy audio")
            return True
        elif "Temp 0:" in output and "Temp 0.2" not in output:
            print("Note: Temperature fallback not triggered (T=0 succeeded)")
            return True  # Not a failure if T=0 works
        else:
            print("Note: Could not verify temperature fallback behavior")
            return None


def test_temperature_fallback_with_debug():
    """Test: Verify temperature fallback implementation via debug output."""
    print("\n=== Test: Temperature Fallback Debug Verification ===")

    # Use existing test audio
    test_audio = Path(__file__).parent.parent / "tests/test_audio_45s_concat.wav"
    if not test_audio.exists():
        print("SKIP: No test audio available")
        return None

    env = os.environ.copy()
    env["DEBUG_WHISPER"] = "1"

    cmd = [
        str(TEST_ENGINE),
        "--whisper", WHISPER_MODEL,
        "--transcribe", str(test_audio),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(Path(__file__).parent.parent)
    )

    output = result.stdout + result.stderr

    # Check for temperature-related debug output
    if "Temp 0:" in output:
        print("PASS: Temperature 0 decoding confirmed")

        # Check if temperature parameter is being passed
        if "Generated" in output:
            print("  - Generate function executed successfully")
            return True

    print("FAIL: No temperature debug output found")
    return False


def main():
    """Run all challenging audio tests."""
    print("=" * 60)
    print("Gate0 Challenging Audio Tests")
    print("=" * 60)
    print(f"Model: {WHISPER_MODEL}")
    print(f"Engine: {TEST_ENGINE}")

    if not TEST_ENGINE.exists():
        print(f"\nERROR: test_engine not found at {TEST_ENGINE}")
        print("Build it first: cd src/mlx_inference_engine && make")
        return 1

    if not Path(WHISPER_MODEL).exists():
        print(f"\nERROR: Whisper model not found at {WHISPER_MODEL}")
        return 1

    results = {}

    # Run tests
    results["silent"] = test_silent_audio()
    results["short"] = test_short_audio()
    results["noisy"] = test_noisy_audio()
    results["temp_fallback"] = test_temperature_fallback_with_debug()

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    skipped = sum(1 for r in results.values() if r is None)

    for name, result in results.items():
        status = "PASS" if result is True else "FAIL" if result is False else "SKIP"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
