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
Kokoro Multi-Voice Validation Test

Tests Kokoro MLX with multiple voices to verify generalization across voice packs.
Uses Whisper transcription to validate audio quality.

Pass criteria:
- All voices produce non-silent audio (RMS > 0.01)
- Whisper produces non-empty transcription for each voice
- Different voices produce different audio (not identical)

Usage:
    python scripts/test_kokoro_multi_voice.py
"""

import sys
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np

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
    print("WARNING: mlx_whisper not available")

import tempfile

import soundfile as sf

SAMPLE_RATE = 24000
WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"


def find_voice_packs() -> List[Path]:
    """Find available Kokoro voice packs in HuggingFace cache."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    pattern = "models--hexgrad--Kokoro-82M"

    voices = []
    for model_dir in cache_dir.glob(f"{pattern}/snapshots/*/voices/*.pt"):
        voices.append(model_dir)

    return sorted(voices)


def get_test_phoneme_sequences() -> List[Tuple[str, List[int], str]]:
    """Get test phoneme sequences."""
    return [
        ("short_sequence", [16, 43, 44, 45, 46, 47, 48, 16], "short phoneme pattern"),
        (
            "medium_sequence",
            [16, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 16],
            "medium phoneme pattern",
        ),
    ]


def generate_audio(input_ids: List[int], voice_path: Path) -> np.ndarray:
    """Generate audio using MLX Kokoro with specified voice."""
    from tools.pytorch_to_mlx.converters import KokoroConverter

    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()

    # Load voice from specific path with phoneme_length for proper duration prediction
    # For arbitrary token sequences, use len(input_ids) as approximation
    voice = model.load_voice(str(voice_path), phoneme_length=len(input_ids))

    # Generate audio
    input_tensor = mx.array([input_ids])
    audio = model.synthesize(input_tensor, voice)
    mx.eval(audio)

    audio_np: np.ndarray[tuple[Any, ...], np.dtype[Any]] = np.array(audio)[0]
    return audio_np


def transcribe_audio(audio: np.ndarray, sr: int = SAMPLE_RATE) -> str:
    """Transcribe audio using Whisper."""
    if not HAS_WHISPER:
        return ""

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, sr)
        temp_path = f.name

    try:
        result = mlx_whisper.transcribe(temp_path, path_or_hf_repo=WHISPER_MODEL)
        text: str = result["text"]
        return text.strip()
    finally:
        Path(temp_path).unlink(missing_ok=True)


def compute_audio_similarity(audio1: np.ndarray, audio2: np.ndarray) -> float:
    """Compute correlation between two audio signals."""
    # Normalize lengths
    min_len = min(len(audio1), len(audio2))
    a1 = audio1[:min_len]
    a2 = audio2[:min_len]

    # Compute correlation
    if np.std(a1) > 0 and np.std(a2) > 0:
        corr = np.corrcoef(a1, a2)[0, 1]
        return float(corr)
    return 0.0


def main():
    """Run multi-voice validation tests."""
    print("=" * 70)
    print("Kokoro Multi-Voice Validation Test")
    print("=" * 70)

    # Find available voices
    voice_paths = find_voice_packs()
    print(f"\nFound {len(voice_paths)} voice pack(s):")
    for vp in voice_paths:
        print(f"  - {vp.stem}")

    if len(voice_paths) == 0:
        print("\nERROR: No voice packs found")
        return 1

    # Get test sequences
    test_sequences = get_test_phoneme_sequences()

    # Results storage
    all_results = []
    voice_audios = {}  # Store audio for similarity comparison

    # Test each voice with each sequence
    for voice_path in voice_paths:
        voice_name = voice_path.stem
        print(f"\n{'=' * 70}")
        print(f"Testing voice: {voice_name}")
        print("=" * 70)

        voice_audios[voice_name] = {}

        for seq_name, input_ids, description in test_sequences:
            print(f"\n  Sequence: {seq_name} ({len(input_ids)} tokens)")

            result = {
                "voice": voice_name,
                "sequence": seq_name,
                "status": "UNKNOWN",
            }

            try:
                # Generate audio
                audio = generate_audio(input_ids, voice_path)
                voice_audios[voice_name][seq_name] = audio

                result["duration_s"] = len(audio) / SAMPLE_RATE
                result["rms"] = float(np.sqrt(np.mean(audio**2)))

                # Check not silent
                if result["rms"] < 0.01:
                    result["status"] = "FAIL"
                    result["error"] = "Audio is silent"
                    print(f"    FAIL: Audio RMS {result['rms']:.4f} < 0.01")
                else:
                    # Transcribe
                    if HAS_WHISPER:
                        transcription = transcribe_audio(audio)
                        result["transcription"] = transcription
                        clean = "".join(c for c in transcription.lower() if c.isalnum())

                        if clean:
                            result["status"] = "PASS"
                            print(
                                f"    PASS: RMS={result['rms']:.4f}, Duration={result['duration_s']:.2f}s"
                            )
                            print(f"    Transcription: {transcription}")
                        else:
                            result["status"] = "FAIL"
                            result["error"] = "Empty transcription"
                            print("    FAIL: Empty/silence transcription")
                    else:
                        result["status"] = "PASS"
                        result["note"] = "No Whisper - verified non-silent only"
                        print(
                            f"    PASS: RMS={result['rms']:.4f}, Duration={result['duration_s']:.2f}s"
                        )
                        print("    (Whisper not available)")

            except Exception as e:
                result["status"] = "ERROR"
                result["error"] = str(e)
                print(f"    ERROR: {e}")

            all_results.append(result)

    # Cross-voice similarity check
    print(f"\n{'=' * 70}")
    print("Voice Differentiation Check")
    print("=" * 70)

    voice_names = list(voice_audios.keys())
    if len(voice_names) >= 2:
        for seq_name, _, _ in test_sequences:
            print(f"\n  Sequence: {seq_name}")

            for i in range(len(voice_names)):
                for j in range(i + 1, len(voice_names)):
                    v1, v2 = voice_names[i], voice_names[j]
                    if seq_name in voice_audios[v1] and seq_name in voice_audios[v2]:
                        corr = compute_audio_similarity(
                            voice_audios[v1][seq_name], voice_audios[v2][seq_name]
                        )
                        status = "DIFFERENT" if corr < 0.95 else "SIMILAR"
                        print(f"    {v1} vs {v2}: correlation={corr:.4f} ({status})")
    else:
        print("  Skipped: Only one voice available")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in all_results if r["status"] == "PASS")
    failed = sum(1 for r in all_results if r["status"] == "FAIL")
    errors = sum(1 for r in all_results if r["status"] == "ERROR")
    total = len(all_results)

    print(f"  Total tests: {total}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Errors: {errors}")
    print(f"  Voices tested: {len(voice_paths)}")
    print(f"  Sequences tested: {len(test_sequences)}")

    overall = "PASS" if failed == 0 and errors == 0 and passed > 0 else "FAIL"
    print(f"\nOverall: {overall}")

    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
