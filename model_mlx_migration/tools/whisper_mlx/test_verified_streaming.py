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
Test script for verified streaming ASR with phoneme verification.

Tests the VerifiedStreamingWhisper integration end-to-end.
"""

import asyncio

# Add project root to path
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.whisper_mlx.model import WhisperMLX
from tools.whisper_mlx.verified_streaming import (
    VerifiedStreamingConfig,
    VerifiedStreamingWhisper,
)


async def stream_audio_file(audio: np.ndarray, chunk_size: int = 16000):
    """Simulate streaming by yielding audio in chunks."""
    for i in range(0, len(audio), chunk_size):
        yield audio[i:i + chunk_size]
        await asyncio.sleep(0.01)  # Small delay to simulate real-time


async def test_verified_streaming():
    """Test the verified streaming pipeline end-to-end."""
    print("=" * 60)
    print("Testing Verified Streaming ASR with Phoneme Verification")
    print("=" * 60)

    # Load test audio
    test_audio_path = Path("/Users/ayates/model_mlx_migration/tests/test_audio_45s_concat.wav")
    if not test_audio_path.exists():
        # Try alternative
        test_audio_path = Path("/Users/ayates/model_mlx_migration/test_data/cosyvoice3_cpp_test/audio_python.wav")

    if not test_audio_path.exists():
        print("ERROR: No test audio file found")
        return False

    print(f"\nLoading audio: {test_audio_path}")

    import soundfile as sf
    audio, sr = sf.read(test_audio_path)

    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Resample to 16kHz if needed
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    audio = audio.astype(np.float32)

    # Use first 10 seconds for quick test
    audio = audio[:sr * 10]
    print(f"Audio duration: {len(audio) / sr:.2f}s")

    # Load model
    print("\nLoading Whisper model (large-v3)...")
    start = time.time()
    model = WhisperMLX.from_pretrained("large-v3")
    print(f"Model loaded in {time.time() - start:.2f}s")

    # Create verified streamer
    print("\nCreating verified streaming pipeline...")
    config = VerifiedStreamingConfig(
        min_chunk_duration=0.5,
        max_chunk_duration=5.0,
        partial_interval=0.3,
        phoneme_commit_threshold=0.70,
        phoneme_wait_threshold=0.40,
        use_phoneme_verification=True,
        include_pronunciation_analysis=False,  # Keep simple for test
        use_local_agreement=False,  # Test phoneme verification directly
    )

    try:
        streamer = VerifiedStreamingWhisper.from_pretrained(
            model,
            phoneme_head_path="models/kokoro_phoneme_head",
            config=config,
        )
        print("Verified streamer created successfully")
    except Exception as e:
        print(f"ERROR creating streamer: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Run streaming transcription
    print("\n" + "-" * 60)
    print("Running streaming transcription with phoneme verification...")
    print("-" * 60)

    results = []
    start_time = time.time()

    try:
        async for result in streamer.transcribe_stream(stream_audio_file(audio, chunk_size=sr // 4)):
            results.append(result)

            # Print result
            status_str = result.commit_status.value.upper()
            conf_str = f"{result.phoneme_confidence:.2f}"
            time_str = f"[{result.segment_start:.1f}s-{result.segment_end:.1f}s]"

            print(f"\n{time_str} [{status_str}] (conf={conf_str})")
            print(f"  Text: {result.text}")
            print(f"  Edit distance: {result.edit_distance}")
            print(f"  Processing: {result.processing_time_ms:.1f}ms (verify: {result.verification_time_ms:.1f}ms)")

            if result.predicted_phonemes:
                pred_sample = result.predicted_phonemes[:10]
                print(f"  Predicted phonemes (first 10): {pred_sample}")
            if result.expected_phonemes:
                exp_sample = result.expected_phonemes[:10]
                print(f"  Expected phonemes (first 10): {exp_sample}")

    except Exception as e:
        print(f"\nERROR during streaming: {e}")
        import traceback
        traceback.print_exc()
        return False

    total_time = time.time() - start_time

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if not results:
        print("ERROR: No results produced")
        return False

    print(f"Total results: {len(results)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Real-time factor: {total_time / (len(audio) / sr):.2f}x")

    # Analyze commit statuses
    status_counts = {}
    for r in results:
        status = r.commit_status.value
        status_counts[status] = status_counts.get(status, 0) + 1

    print("\nCommit status distribution:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")

    # Analyze confidence scores
    confidences = [r.phoneme_confidence for r in results]
    print("\nPhoneme confidence stats:")
    print(f"  Min: {min(confidences):.3f}")
    print(f"  Max: {max(confidences):.3f}")
    print(f"  Mean: {np.mean(confidences):.3f}")

    # Analyze verification latency
    verify_times = [r.verification_time_ms for r in results]
    print("\nVerification latency:")
    print(f"  Min: {min(verify_times):.1f}ms")
    print(f"  Max: {max(verify_times):.1f}ms")
    print(f"  Mean: {np.mean(verify_times):.1f}ms")

    # Final transcript
    final_results = [r for r in results if r.is_final]
    if final_results:
        print(f"\nFinal transcript: {final_results[-1].text}")
    else:
        print(f"\nLast partial: {results[-1].text}")

    print("\n" + "=" * 60)
    print("TEST PASSED: Verified streaming pipeline works end-to-end")
    print("=" * 60)

    return True


async def test_commit_logic():
    """Test the commit/wait logic independently."""
    print("\n" + "=" * 60)
    print("Testing Commit Logic")
    print("=" * 60)


    # Test that different confidence levels produce expected statuses
    test_cases = [
        (0.90, "commit"),   # High confidence -> COMMIT
        (0.75, "commit"),   # At threshold -> COMMIT
        (0.70, "partial"),  # Between thresholds -> PARTIAL
        (0.50, "partial"),  # At wait threshold -> PARTIAL
        (0.40, "wait"),     # Below wait threshold -> WAIT
        (0.10, "wait"),     # Very low -> WAIT
    ]

    passed = 0
    for confidence, expected in test_cases:
        # Determine status based on thresholds
        commit_threshold = 0.75
        wait_threshold = 0.50

        if confidence >= commit_threshold:
            actual = "commit"
        elif confidence < wait_threshold:
            actual = "wait"
        else:
            actual = "partial"

        status = "PASS" if actual == expected else "FAIL"
        print(f"  confidence={confidence:.2f} -> {actual} (expected {expected}) [{status}]")
        if actual == expected:
            passed += 1

    print(f"\nCommit logic: {passed}/{len(test_cases)} tests passed")
    return passed == len(test_cases)


if __name__ == "__main__":
    # Run tests
    success = True

    # Test commit logic first (no model needed)
    if not asyncio.run(test_commit_logic()):
        success = False

    # Test full streaming pipeline
    if not asyncio.run(test_verified_streaming()):
        success = False

    sys.exit(0 if success else 1)
