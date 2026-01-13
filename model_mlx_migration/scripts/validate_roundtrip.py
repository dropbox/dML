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
Validate Round-Trip Verification hypothesis.

Tests whether the round-trip verification can distinguish between:
1. Correct transcripts (should have HIGH similarity)
2. Wrong transcripts (should have LOW similarity)

If the hypothesis holds:
- confidence(audio, correct_transcript) >> confidence(audio, wrong_transcript)
"""

import sys
import numpy as np
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.whisper_mlx.roundtrip_verification import (
    RoundTripVerifier,
)


def create_synthetic_test():
    """Create synthetic test with known ground truth using Whisper."""
    print("=== Creating Synthetic Test ===\n")

    # Use LibriSpeech dev-clean samples
    librispeech_dir = Path("/Users/ayates/model_mlx_migration/data/benchmarks/librispeech/LibriSpeech/dev-clean")

    if not librispeech_dir.exists():
        print(f"LibriSpeech not found at {librispeech_dir}")
        return []

    # Find some audio files
    flac_files = list(librispeech_dir.rglob("*.flac"))[:5]

    if not flac_files:
        print("No FLAC files found")
        return []

    test_cases = []
    for flac in flac_files:
        # Find corresponding transcript
        trans_file = flac.parent / f"{flac.parent.name}.trans.txt"
        if trans_file.exists():
            with open(trans_file) as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) == 2 and parts[0] == flac.stem:
                        correct = parts[1].lower()
                        # Create wrong transcript by shuffling words
                        words = correct.split()
                        if len(words) > 2:
                            wrong = " ".join(words[1:] + words[:1])  # Rotate words
                        else:
                            wrong = "something completely different"

                        test_cases.append({
                            "audio_path": str(flac),
                            "correct": correct,
                            "wrong": wrong,
                        })
                        break

    return test_cases


def load_audio(path: str, sample_rate: int = 16000) -> np.ndarray:
    """Load audio file and resample to target rate."""
    import soundfile as sf

    audio, sr = sf.read(path)

    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != sample_rate:
        import scipy.signal
        audio = scipy.signal.resample(audio, int(len(audio) * sample_rate / sr))

    return audio.astype(np.float32)


def test_mel_similarity_hypothesis():
    """Test if mel similarity distinguishes correct vs wrong."""
    print("=== Testing Mel Similarity Hypothesis ===\n")

    # Create mock verifier (doesn't need Kokoro)
    verifier = RoundTripVerifier.from_mock()

    test_cases = create_synthetic_test()

    if not test_cases:
        print("No test cases available. Using synthetic audio.\n")
        # Fall back to synthetic test
        test_cases = [
            {
                "audio_path": None,
                "audio": np.random.randn(32000).astype(np.float32),  # 2 seconds
                "correct": "hello world how are you",
                "wrong": "goodbye moon i am fine",
            }
        ]

    results = []
    for i, case in enumerate(test_cases):
        print(f"Test case {i+1}/{len(test_cases)}")

        if case.get("audio_path"):
            try:
                audio = load_audio(case["audio_path"])
                print(f"  Audio: {case['audio_path']}")
            except Exception as e:
                print(f"  Failed to load audio: {e}")
                continue
        else:
            audio = case["audio"]
            print("  Audio: synthetic")

        print(f"  Correct: {case['correct'][:50]}...")
        print(f"  Wrong: {case['wrong'][:50]}...")

        # Get confidence for correct and wrong
        try:
            correct_result = verifier.compute_confidence(audio, case["correct"])
            wrong_result = verifier.compute_confidence(audio, case["wrong"])

            print(f"  Confidence (correct): {correct_result.confidence:.4f}")
            print(f"  Confidence (wrong):   {wrong_result.confidence:.4f}")
            print(f"  Difference: {correct_result.confidence - wrong_result.confidence:+.4f}")

            results.append({
                "correct_conf": correct_result.confidence,
                "wrong_conf": wrong_result.confidence,
                "difference": correct_result.confidence - wrong_result.confidence,
            })
        except Exception as e:
            print(f"  Error: {e}")

        print()

    # Summary
    if results:
        print("=== Summary ===")
        avg_correct = np.mean([r["correct_conf"] for r in results])
        avg_wrong = np.mean([r["wrong_conf"] for r in results])
        avg_diff = np.mean([r["difference"] for r in results])

        print(f"Average correct confidence: {avg_correct:.4f}")
        print(f"Average wrong confidence:   {avg_wrong:.4f}")
        print(f"Average difference:         {avg_diff:+.4f}")

        if avg_diff > 0.1:
            print("\n✓ HYPOTHESIS SUPPORTED: Correct transcripts have higher confidence")
        elif avg_diff > 0:
            print("\n~ WEAK SUPPORT: Small difference, may need tuning")
        else:
            print("\n✗ HYPOTHESIS NOT SUPPORTED: Wrong transcripts have higher confidence")


def test_with_kokoro():
    """Test with actual Kokoro TTS (if available)."""
    print("\n=== Testing with Kokoro TTS ===\n")

    try:
        verifier = RoundTripVerifier.from_kokoro()
        print("Kokoro verifier created successfully!")

        # Quick test
        audio = np.random.randn(16000).astype(np.float32)
        result = verifier.compute_confidence(audio, "hello world")
        print(f"Test confidence: {result.confidence:.4f}")
        print("✓ Kokoro integration working")

    except Exception as e:
        print(f"Kokoro not available: {e}")
        print("Using mock verifier instead (returns random confidence)")


if __name__ == "__main__":
    print("Round-Trip Verification Validation\n")
    print("=" * 50)

    # Test basic mel similarity
    test_mel_similarity_hypothesis()

    # Test with Kokoro if available
    test_with_kokoro()

    print("\n" + "=" * 50)
    print("Validation complete.")
    print("\nNOTE: The mock verifier returns random confidence.")
    print("For real validation, Kokoro TTS integration is needed.")
    print("The key metric is whether correct > wrong consistently.")
