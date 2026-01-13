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
Test script for Round-Trip Spectrogram Verification.

Demonstrates:
1. Basic verification with synthetic audio
2. Integration potential with streaming ASR
3. Confidence-based commit decisions
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_basic_verification():
    """Test basic verification functionality."""
    print("=" * 60)
    print("Test 1: Basic Verification with Mock TTS")
    print("=" * 60)

    from tools.whisper_mlx.roundtrip_verification import RoundTripVerifier

    # Create mock verifier (no TTS model needed)
    verifier = RoundTripVerifier.from_mock(similarity_method="cosine")

    # Test with different audio types
    test_cases = [
        ("Sine wave 440Hz", lambda: np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)).astype(np.float32) * 0.5),
        ("White noise", lambda: np.random.randn(16000).astype(np.float32) * 0.1),
        ("Silent", lambda: np.zeros(16000, dtype=np.float32)),
        ("Speech-like (multi-freq)", lambda: (
            np.sin(2 * np.pi * 200 * np.linspace(0, 1, 16000)) * 0.3 +
            np.sin(2 * np.pi * 400 * np.linspace(0, 1, 16000)) * 0.2 +
            np.random.randn(16000) * 0.05
        ).astype(np.float32)),
    ]

    for name, audio_fn in test_cases:
        audio = audio_fn()
        result = verifier.compute_confidence(audio, "hello world")
        print(f"\n{name}:")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Input frames: {result.input_mel_frames}")
        print(f"  Generated frames: {result.generated_mel_frames}")
        print(f"  Reliable: {result.is_reliable}")


def test_kokoro_verification(audio_path: str = None):
    """Test verification with actual Kokoro TTS."""
    print("\n" + "=" * 60)
    print("Test 2: Kokoro TTS Verification")
    print("=" * 60)

    try:
        from tools.whisper_mlx.roundtrip_verification import RoundTripVerifier

        print("\nLoading Kokoro model...")
        t0 = time.perf_counter()
        verifier = RoundTripVerifier.from_kokoro(similarity_method="dtw_fast")
        print(f"Model loaded in {time.perf_counter() - t0:.2f}s")

        # Test with various transcripts
        test_transcripts = [
            "hello",
            "hello world",
            "the quick brown fox jumps over the lazy dog",
            "one two three four five",
        ]

        # Generate test audio (speech-like)
        t = np.linspace(0, 2.0, int(2.0 * 16000))
        test_audio = (
            np.sin(2 * np.pi * 150 * t) * 0.3 +
            np.sin(2 * np.pi * 300 * t) * 0.2 +
            np.sin(2 * np.pi * 450 * t) * 0.1 +
            np.random.randn(len(t)) * 0.02
        ).astype(np.float32)

        print("\nTesting confidence scores with synthetic audio:")
        for transcript in test_transcripts:
            t0 = time.perf_counter()
            result = verifier.compute_confidence(test_audio, transcript)
            elapsed = time.perf_counter() - t0

            print(f"\n  '{transcript[:40]}...' if len(transcript) > 40 else '{transcript}'")
            print(f"    Confidence: {result.confidence:.3f}")
            print(f"    Alignment cost: {result.alignment_cost:.3f}")
            print(f"    Latency: {elapsed*1000:.1f}ms")

        # If audio file provided, test with real audio
        if audio_path:
            print(f"\n\nTesting with real audio: {audio_path}")
            from tools.whisper_mlx.audio import load_audio
            real_audio = load_audio(audio_path)

            # Test a few transcripts
            for transcript in ["hello", "test audio", "this is a test"]:
                result = verifier.compute_confidence(real_audio[:32000], transcript)
                print(f"\n  '{transcript}':")
                print(f"    Confidence: {result.confidence:.3f}")

    except Exception as e:
        print(f"Kokoro test failed: {e}")
        import traceback
        traceback.print_exc()


def test_streaming_integration():
    """Demonstrate how this could integrate with streaming ASR."""
    print("\n" + "=" * 60)
    print("Test 3: Streaming Integration Demo")
    print("=" * 60)

    from tools.whisper_mlx.roundtrip_verification import RoundTripVerifier

    verifier = RoundTripVerifier.from_mock(similarity_method="cosine")

    # Simulate streaming scenario
    print("\nSimulated streaming scenario:")
    print("  - Audio arrives in chunks")
    print("  - STT produces partial transcriptions")
    print("  - Verifier decides when to commit")

    # Simulated transcription history (what STT might produce)
    simulation = [
        (0.5, "hel", 0.3),      # 0.5s: partial, low confidence expected
        (1.0, "hello", 0.5),    # 1.0s: more complete
        (1.5, "hello wor", 0.4),  # 1.5s: partial word
        (2.0, "hello world", 0.6),  # 2.0s: complete
    ]

    threshold = 0.4
    committed_text = ""

    print(f"\nCommit threshold: {threshold}")
    print("-" * 40)

    for time_sec, transcript, _expected in simulation:
        # Generate audio chunk (just for demo)
        audio_chunk = np.random.randn(int(time_sec * 16000)).astype(np.float32) * 0.1

        should_commit, confidence = verifier.should_commit(
            audio_chunk, transcript, threshold=threshold
        )

        status = "COMMIT" if should_commit else "HOLD"
        print(f"t={time_sec:.1f}s: '{transcript}' -> {status} (conf={confidence:.2f})")

        if should_commit and transcript not in committed_text:
            committed_text = transcript

    print(f"\nFinal committed: '{committed_text}'")


def test_comparison_methods():
    """Compare different similarity methods."""
    print("\n" + "=" * 60)
    print("Test 4: Similarity Method Comparison")
    print("=" * 60)

    from tools.whisper_mlx.roundtrip_verification import MelSimilarity

    # Create test mel spectrograms
    np.random.seed(42)

    # Similar mels (small perturbation)
    mel_base = np.random.randn(100, 128).astype(np.float32)
    mel_similar = mel_base + np.random.randn(100, 128).astype(np.float32) * 0.1

    # Different mels
    mel_different = np.random.randn(100, 128).astype(np.float32)

    # Different length
    mel_longer = np.random.randn(150, 128).astype(np.float32)

    print("\nComparing mel_base vs mel_similar (small noise added):")
    print(f"  Cosine: {MelSimilarity.cosine_similarity_pooled(mel_base, mel_similar):.3f}")
    dtw_sim, dtw_cost = MelSimilarity.dtw_similarity_fast(mel_base, mel_similar)
    print(f"  DTW: {dtw_sim:.3f} (cost: {dtw_cost:.3f})")
    print(f"  Correlation: {MelSimilarity.correlation_similarity(mel_base, mel_similar):.3f}")

    print("\nComparing mel_base vs mel_different (random):")
    print(f"  Cosine: {MelSimilarity.cosine_similarity_pooled(mel_base, mel_different):.3f}")
    dtw_sim, dtw_cost = MelSimilarity.dtw_similarity_fast(mel_base, mel_different)
    print(f"  DTW: {dtw_sim:.3f} (cost: {dtw_cost:.3f})")
    print(f"  Correlation: {MelSimilarity.correlation_similarity(mel_base, mel_different):.3f}")

    print("\nComparing mel_base vs mel_longer (different length):")
    print(f"  Cosine: {MelSimilarity.cosine_similarity_pooled(mel_base, mel_longer):.3f}")
    dtw_sim, dtw_cost = MelSimilarity.dtw_similarity_fast(mel_base, mel_longer)
    print(f"  DTW: {dtw_sim:.3f} (cost: {dtw_cost:.3f})")
    print(f"  Correlation: {MelSimilarity.correlation_similarity(mel_base, mel_longer):.3f}")


def main():
    parser = argparse.ArgumentParser(description="Test Round-Trip Verification")
    parser.add_argument("--audio", type=str, help="Path to audio file for testing")
    parser.add_argument("--kokoro", action="store_true", help="Run Kokoro TTS tests")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    args = parser.parse_args()

    # Always run basic tests
    test_basic_verification()
    test_comparison_methods()
    test_streaming_integration()

    # Kokoro tests (optional, requires model)
    if args.kokoro or args.all:
        test_kokoro_verification(args.audio)

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
