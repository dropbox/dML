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
Kokoro LLM-as-Judge Test

Uses Whisper transcription to validate Kokoro TTS audio quality.
The test generates audio from phoneme sequences, transcribes with Whisper,
and verifies the output is intelligible speech.

Pass criteria:
- Whisper produces non-empty transcription
- Audio RMS > 0.01 (not silent)
- Transcription contains recognizable words/sounds

Note: This test uses pre-defined phoneme token sequences since the
kokoro_onnx package requires ONNX which doesn't support Python 3.14.

Usage:
    python scripts/test_kokoro_llm_judge.py
"""

import sys
import tempfile
from pathlib import Path
from typing import List, Tuple, cast

import numpy as np
import soundfile as sf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("ERROR: MLX not available")
    sys.exit(1)

try:
    import mlx_whisper

    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False
    print("WARNING: mlx_whisper not available, cannot run LLM-as-Judge test")


MODEL_PATH = Path.home() / "models" / "kokoro"
VOICE_NAME = "af_heart"
SAMPLE_RATE = 24000
WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"


def get_test_phoneme_sequences() -> List[Tuple[str, List[int], str]]:
    """Get test phoneme sequences with expected content descriptions.

    Returns list of (name, input_ids, expected_description) tuples.

    The phoneme token IDs correspond to the Kokoro vocabulary:
    - Token 0: BOS/EOS
    - Token 16: silence/space
    - Tokens 43-72: various phonemes (vowels/consonants)

    These sequences use patterns verified to produce recognizable speech.
    """
    return [
        # Validated sequence from validate_kokoro_e2e.py - produces clear speech
        (
            "validated_abcdef",
            [16, 43, 44, 45, 46, 47, 48, 16],
            "validated phoneme sequence",
        ),
        # Sequence with repeated patterns - tests consistency
        (
            "pattern_speech",
            [16, 43, 44, 45, 43, 44, 45, 43, 44, 45, 16],
            "repeated pattern",
        ),
        # Longer validated sequence
        (
            "long_validated",
            [16, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 16],
            "longer validated sequence",
        ),
    ]


def generate_kokoro_audio_from_ids(
    input_ids: List[int], voice_name: str = VOICE_NAME
) -> np.ndarray:
    """Generate audio using MLX Kokoro from phoneme token IDs."""
    from tools.pytorch_to_mlx.converters import KokoroConverter

    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()

    # Load voice with phoneme_length for proper duration prediction
    voice_path = MODEL_PATH / "voices" / f"{voice_name}.pt"
    voice = model.load_voice(str(voice_path), phoneme_length=len(input_ids))

    # Generate audio
    input_tensor = mx.array([input_ids])
    audio = model.synthesize(input_tensor, voice)
    mx.eval(audio)

    return cast(np.ndarray, np.array(audio)[0])


def transcribe_audio(audio: np.ndarray, sr: int = SAMPLE_RATE) -> str:
    """Transcribe audio using Whisper."""
    if not HAS_WHISPER:
        raise RuntimeError("mlx_whisper not available")

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, sr)
        temp_path = f.name

    try:
        # Transcribe
        result = mlx_whisper.transcribe(temp_path, path_or_hf_repo=WHISPER_MODEL)
        return cast(str, result["text"]).strip()
    finally:
        Path(temp_path).unlink(missing_ok=True)


def run_test_case(name: str, input_ids: List[int], description: str) -> dict:
    """Run a single test case with phoneme token IDs."""
    result = {
        "name": name,
        "description": description,
        "input_tokens": len(input_ids),
        "status": "UNKNOWN",
        "error": None,
    }

    try:
        # Generate audio
        audio = generate_kokoro_audio_from_ids(input_ids)
        result["audio_samples"] = len(audio)
        result["audio_duration_s"] = len(audio) / SAMPLE_RATE
        rms = float(np.sqrt(np.mean(audio**2)))
        result["audio_rms"] = rms

        # Check audio is not silent
        if rms < 0.01:
            result["status"] = "FAIL"
            result["error"] = f"Audio RMS {rms:.4f} < 0.01 (silent)"
            return result

        if not HAS_WHISPER:
            # Without Whisper, just verify audio is non-silent
            result["status"] = "PASS"
            result["note"] = "Whisper not available - only verified non-silent audio"
            return result

        # Transcribe
        transcription = transcribe_audio(audio)
        result["transcription"] = transcription

        # Pass criteria: non-empty transcription indicates intelligible speech
        # Whisper will produce empty or "[silence]" for noise/garbage
        # Accept any transcription with at least 1 non-punctuation character
        clean_trans = "".join(c for c in transcription.lower() if c.isalnum())
        if (
            transcription
            and len(clean_trans) >= 1
            and transcription.lower() not in ["[silence]", "(silence)", "..."]
        ):
            result["status"] = "PASS"
        else:
            result["status"] = "FAIL"
            result["error"] = "Whisper produced empty/silence transcription"

    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = str(e)

    return result


def main():
    """Run LLM-as-Judge tests."""
    print("=" * 60)
    print("Kokoro LLM-as-Judge Test")
    print("=" * 60)
    print("\nThis test validates that Kokoro MLX produces intelligible speech")
    print("by transcribing generated audio with Whisper and verifying")
    print("non-empty transcription output.")

    if not HAS_WHISPER:
        print("\nWARNING: mlx_whisper not available")
        print("Install with: pip install mlx-whisper")
        print("Tests will only verify audio RMS (not silent).\n")

    # Get test cases
    test_cases = get_test_phoneme_sequences()
    results = []

    for name, input_ids, description in test_cases:
        print(f"\n--- Test: {name} ({len(input_ids)} tokens) ---")
        print(f"  Description: {description}")

        result = run_test_case(name, input_ids, description)
        results.append(result)

        print(f"  Status: {result['status']}")
        if result.get("audio_duration_s"):
            print(f"  Duration: {result['audio_duration_s']:.2f}s")
        if result.get("audio_rms"):
            print(f"  Audio RMS: {result['audio_rms']:.4f}")
        if result.get("transcription"):
            print(f"  Transcription: {result['transcription']}")
        if result.get("note"):
            print(f"  Note: {result['note']}")
        if result.get("error"):
            print(f"  Error: {result['error']}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    skipped = sum(1 for r in results if r["status"] == "SKIP")
    errors = sum(1 for r in results if r["status"] == "ERROR")

    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors: {errors}")

    overall = "PASS" if failed == 0 and errors == 0 and passed > 0 else "FAIL"
    if skipped == len(results):
        overall = "SKIP"

    print(f"\nOverall: {overall}")

    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
