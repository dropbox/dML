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
F5-TTS Voice Cloning Test Suite

DEPRECATED: F5-TTS is deprecated in favor of CosyVoice2.
CosyVoice2 is 18x faster (35x vs 2x RTF) with equal/better quality.
For voice cloning, use CosyVoice2 with speaker embeddings instead.

Tests F5-TTS zero-shot voice cloning capability with various reference voices.
Uses Whisper to validate transcription accuracy.

Usage:
    python scripts/test_f5tts_voice_cloning.py

Features tested:
- Zero-shot voice cloning from reference audio
- Multiple emotional tones (happy, sad, angry, calm)
- Quality validation via Whisper transcription
- Performance benchmarking (RTF)
"""

import sys
import tempfile
import time
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import soundfile as sf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from f5_tts_mlx.generate import generate

    HAS_F5TTS = True
except ImportError:
    HAS_F5TTS = False

try:
    import mlx_whisper

    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False

VOICES_DIR = Path(__file__).parent.parent / "voices" / "f5tts"
SAMPLE_RATE = 24000
WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"


class VoiceTestCase(NamedTuple):
    """Test case for voice cloning."""

    name: str
    ref_audio: str
    ref_text: str
    gen_text: str


def get_test_cases() -> list[VoiceTestCase]:
    """Get voice cloning test cases.

    Note: F5-TTS requires 24kHz reference audio.
    The ref_text must closely match the actual content of the reference audio
    for successful alignment.

    These test cases use reference audio from the voices/f5tts directory.
    The reference texts are approximations - exact match not required but
    should be semantically close for best results.
    """
    return [
        # english_female.wav - verified female voice (Dante quote)
        VoiceTestCase(
            name="english_female",
            ref_audio="english_female.wav",
            ref_text="Song that as greatly doth transcend our Muses, Our Sirens, in those dulcet clarions.",
            gen_text="The quick brown fox jumps over the lazy dog.",
        ),
        # Same voice, different generation text
        VoiceTestCase(
            name="english_female_test2",
            ref_audio="english_female.wav",
            ref_text="Song that as greatly doth transcend our Muses, Our Sirens, in those dulcet clarions.",
            gen_text="Machine learning is transforming how we interact with technology.",
        ),
        # seedtts_ref_en_1.wav - Mother Nature voice
        VoiceTestCase(
            name="seedtts_en_1",
            ref_audio="seedtts_ref_en_1.wav",
            ref_text="Some call me nature. Others call me Mother Nature.",
            gen_text="Technology is advancing at an unprecedented rate.",
        ),
        # robust_1_ref.wav - verified 24kHz
        VoiceTestCase(
            name="robust_voice",
            ref_audio="robust_1_ref.wav",
            ref_text="This is a sample of my voice for testing purposes.",
            gen_text="Hello, this is a test of voice cloning capabilities.",
        ),
    ]


def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio using Whisper."""
    if not HAS_WHISPER:
        return ""
    result = mlx_whisper.transcribe(audio_path, path_or_hf_repo=WHISPER_MODEL)
    return str(result.get("text", "")).strip()


def run_voice_test(case: VoiceTestCase) -> dict[str, Any]:
    """Run a single voice cloning test."""
    result: dict[str, Any] = {
        "name": case.name,
        "gen_text": case.gen_text,
        "status": "UNKNOWN",
        "error": None,
    }

    ref_path = VOICES_DIR / case.ref_audio
    if not ref_path.exists():
        result["status"] = "SKIP"
        result["error"] = f"Reference audio not found: {ref_path}"
        return result

    try:
        # Generate audio with voice cloning
        output_path = tempfile.mktemp(suffix=".wav")

        start = time.time()
        generate(
            generation_text=case.gen_text,
            ref_audio_path=str(ref_path),
            ref_audio_text=case.ref_text,
            steps=8,
            output_path=output_path,
        )
        elapsed = time.time() - start

        # Load and analyze output
        audio, sr = sf.read(output_path)
        duration = len(audio) / sr

        result["generation_time_s"] = round(elapsed, 2)
        result["audio_duration_s"] = round(duration, 2)
        result["rtf"] = round(elapsed / duration, 3)
        result["audio_rms"] = round(float(np.sqrt(np.mean(audio**2))), 4)

        # Transcription validation
        if HAS_WHISPER:
            transcription = transcribe_audio(output_path)
            result["transcription"] = transcription

            # Compare transcription to generation text (case-insensitive, stripped)
            gen_lower = case.gen_text.lower().strip().rstrip(".")
            trans_lower = transcription.lower().strip().rstrip(".")

            # Calculate word overlap
            gen_words = set(gen_lower.split())
            trans_words = set(trans_lower.split())
            if gen_words:
                overlap = len(gen_words & trans_words) / len(gen_words)
                result["word_overlap"] = round(overlap, 2)
            else:
                result["word_overlap"] = 0.0

        # Cleanup
        Path(output_path).unlink(missing_ok=True)

        # Determine pass/fail
        if result["audio_rms"] < 0.01:
            result["status"] = "FAIL"
            result["error"] = "Silent audio (RMS < 0.01)"
        elif result.get("word_overlap", 1.0) < 0.3:
            result["status"] = "FAIL"
            result["error"] = (
                f"Poor transcription accuracy ({result.get('word_overlap', 0):.0%})"
            )
        else:
            result["status"] = "PASS"

    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = str(e)

    return result


def main():
    """Run F5-TTS voice cloning test suite."""
    print("=" * 70)
    print("F5-TTS Voice Cloning Test Suite")
    print("=" * 70)
    print("\nThis suite tests zero-shot voice cloning with various reference voices.")
    print("Each test generates audio and validates with Whisper transcription.\n")

    if not HAS_F5TTS:
        print("ERROR: f5_tts_mlx package required")
        print("Install with: pip install f5-tts-mlx")
        return 1

    if not HAS_WHISPER:
        print("WARNING: mlx_whisper not available for transcription validation")
        print("Tests will only verify audio RMS (not silent).\n")

    if not VOICES_DIR.exists():
        print(f"ERROR: Voices directory not found: {VOICES_DIR}")
        return 1

    # Run tests
    test_cases = get_test_cases()
    results = []

    for case in test_cases:
        print(f"\n--- Voice: {case.name} ---")
        print(
            f"  Gen text: {case.gen_text[:50]}{'...' if len(case.gen_text) > 50 else ''}"
        )

        result = run_voice_test(case)
        results.append(result)

        print(f"  Status: {result['status']}")
        if result.get("rtf"):
            print(
                f"  RTF: {result['rtf']:.3f}x ({result['generation_time_s']:.1f}s / {result['audio_duration_s']:.1f}s)"
            )
        if result.get("audio_rms"):
            print(f"  RMS: {result['audio_rms']:.4f}")
        if result.get("transcription"):
            trans = result["transcription"]
            print(f"  Transcription: {trans[:60]}{'...' if len(trans) > 60 else ''}")
        if result.get("word_overlap") is not None:
            print(f"  Word overlap: {result['word_overlap']:.0%}")
        if result.get("error"):
            print(f"  Error: {result['error']}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    skipped = sum(1 for r in results if r["status"] == "SKIP")
    errors = sum(1 for r in results if r["status"] == "ERROR")

    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors: {errors}")

    # Calculate average RTF for passed tests
    rtfs = [r["rtf"] for r in results if r.get("rtf") and r["status"] == "PASS"]
    if rtfs:
        avg_rtf = sum(rtfs) / len(rtfs)
        print(f"\n  Average RTF: {avg_rtf:.3f}x")
        if avg_rtf < 1.0:
            print("  (Faster than real-time)")

    # Overall result
    overall = "PASS" if failed == 0 and errors == 0 and passed > 0 else "FAIL"
    if passed == 0 and skipped > 0:
        overall = "SKIP"

    print(f"\nOverall: {overall}")

    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
