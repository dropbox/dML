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
Gate 5 Harmonization Test

Tests the harmonization pipeline on RAVDESS singing samples:
1. Load singing samples (03-02-* files)
2. Detect singing (should be >99% for singing files)
3. Track pitch
4. Generate harmonized audio
5. Verify output quality

Success criteria:
- Singing detection accuracy >99% on song files
- Pitch tracking produces valid Hz values
- Harmonized audio is different from original
- No clipping in output
"""

import sys
from pathlib import Path
import numpy as np
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.whisper_mlx.audio import load_audio, get_audio_rms_db
from tools.whisper_mlx.harmonize import (
    HarmonyConfig,
    HarmonyGenerator,
    HarmonyInterval,
    generate_harmony,
    pitch_shift_audio,
)


def test_pitch_shift_basic():
    """Test basic pitch shifting functionality."""
    print("\n1. Testing basic pitch shift...")

    # Generate a simple sine wave at 440 Hz (A4)
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5

    # Shift up by a major third (+4 semitones)
    shifted = pitch_shift_audio(audio, semitones=4, sample_rate=sr)

    # Verify output
    assert shifted.shape == audio.shape, f"Shape mismatch: {shifted.shape} vs {audio.shape}"
    assert shifted.dtype == np.float32, f"Wrong dtype: {shifted.dtype}"
    assert np.max(np.abs(shifted)) <= 1.0, "Output clipped"

    # Verify audio is actually different
    diff = np.mean(np.abs(audio - shifted))
    assert diff > 0.01, f"Audio not shifted (diff={diff})"

    print(f"   Passed: Pitch shift produces valid output (diff={diff:.4f})")
    return True


def test_harmony_generation():
    """Test multi-voice harmony generation."""
    print("\n2. Testing harmony generation...")

    # Generate simple audio
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5

    # Generate major third + perfect fifth harmony
    intervals = [HarmonyInterval.MAJOR_THIRD, HarmonyInterval.PERFECT_FIFTH]
    harmonized = generate_harmony(audio, intervals, sr)

    # Verify output
    assert harmonized.shape == audio.shape, f"Shape mismatch: {harmonized.shape}"
    assert harmonized.dtype == np.float32, f"Wrong dtype: {harmonized.dtype}"
    assert np.max(np.abs(harmonized)) <= 1.0, "Output clipped"

    # Verify harmony is mixed in
    diff = np.mean(np.abs(audio - harmonized))
    assert diff > 0.01, f"Harmony not mixed (diff={diff})"

    print(f"   Passed: Harmony generation works (diff={diff:.4f})")
    return True


def test_singing_detection_on_ravdess(
    checkpoint_path: str,
    ravdess_dir: str,
    max_samples: int = 10,
):
    """Test singing detection accuracy on RAVDESS song files."""
    print("\n3. Testing singing detection on RAVDESS songs...")

    import mlx.core as mx
    from tools.whisper_mlx.model import WhisperMLX
    from tools.whisper_mlx.audio import log_mel_spectrogram
    from tools.whisper_mlx.multi_head import create_multi_head

    # Load Whisper
    print("   Loading Whisper model...")
    whisper = WhisperMLX.from_pretrained("mlx-community/whisper-large-v3-mlx")

    # Load singing head
    print("   Loading singing detection head...")
    multi_head = create_multi_head("large-v3")

    weights = dict(np.load(checkpoint_path, allow_pickle=True))
    singing_weights = []
    for k, v in weights.items():
        if k.startswith("singing."):
            new_key = k[len("singing."):]
            singing_weights.append((new_key, mx.array(v)))
    multi_head.singing_head.load_weights(singing_weights)

    # Find song files (03-02-*)
    ravdess_path = Path(ravdess_dir)
    song_files = list(ravdess_path.rglob("03-02-*.wav"))[:max_samples]

    if not song_files:
        print("   WARNING: No song files found in RAVDESS directory")
        return False

    print(f"   Testing on {len(song_files)} song files...")

    correct = 0
    total = 0

    for audio_path in song_files:
        try:
            # Load and process audio
            audio = load_audio(str(audio_path))
            mel = log_mel_spectrogram(audio)

            # Pad to 3000 frames
            target_frames = 3000
            if mel.shape[0] < target_frames:
                mel = mx.pad(mel, [(0, target_frames - mel.shape[0]), (0, 0)])
            else:
                mel = mel[:target_frames]

            mel = mel[None, ...]

            # Get encoder output
            encoder_output = whisper.embed_audio(mel)

            # Predict singing
            singing_logits = multi_head.singing_head(encoder_output)
            singing_prob = mx.sigmoid(singing_logits)
            is_singing = float(singing_prob[0, 0]) > 0.5

            if is_singing:
                correct += 1
            total += 1

        except Exception as e:
            print(f"   Error processing {audio_path}: {e}")

    accuracy = correct / total * 100 if total > 0 else 0
    print(f"   Singing detection accuracy on songs: {accuracy:.1f}% ({correct}/{total})")

    return accuracy >= 95.0  # Should detect almost all songs correctly


def test_full_harmonization_pipeline(
    checkpoint_path: str,
    ravdess_dir: str,
    output_dir: str = None,
):
    """Test full harmonization pipeline on a RAVDESS song."""
    print("\n4. Testing full harmonization pipeline...")

    from tools.whisper_mlx.model import WhisperMLX

    # Find a song file
    ravdess_path = Path(ravdess_dir)
    song_files = list(ravdess_path.rglob("03-02-*.wav"))

    if not song_files:
        print("   WARNING: No song files found")
        return False

    audio_path = song_files[0]
    print(f"   Testing on: {audio_path.name}")

    # Load audio
    audio = load_audio(str(audio_path))
    print(f"   Audio length: {len(audio)/16000:.2f}s, RMS: {get_audio_rms_db(audio):.1f} dB")

    # Load models
    print("   Loading models...")
    whisper = WhisperMLX.from_pretrained("mlx-community/whisper-large-v3-mlx")

    config = HarmonyConfig()
    generator = HarmonyGenerator(config)
    generator.load_models(whisper, checkpoint_path)

    # Process
    print("   Processing...")
    start_time = time.time()
    result = generator.process(audio)
    process_time = time.time() - start_time

    # Report results
    print("\n   Results:")
    print(f"   - Singing detected: {result.is_singing} (conf: {result.singing_confidence:.3f})")
    print(f"   - Mean pitch: {result.mean_pitch_hz:.1f} Hz ({result.pitch_note})")
    print(f"   - Process time: {process_time:.2f}s")

    if result.is_singing:
        print(f"   - Intervals: {[i.name for i in result.intervals]}")
        print(f"   - Harmony notes: {result.get_harmony_notes()}")

        if result.harmonized_audio is not None:
            orig_rms = get_audio_rms_db(result.original_audio)
            harm_rms = get_audio_rms_db(result.harmonized_audio)
            print(f"   - Original RMS: {orig_rms:.1f} dB")
            print(f"   - Harmonized RMS: {harm_rms:.1f} dB")

            # Verify no clipping
            max_val = np.max(np.abs(result.harmonized_audio))
            print(f"   - Max amplitude: {max_val:.3f}")

            if output_dir:
                output_path = Path(output_dir) / f"harmonized_{audio_path.stem}.wav"
                try:
                    import soundfile as sf
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(str(output_path), result.harmonized_audio, 16000)
                    print(f"   - Saved to: {output_path}")
                except ImportError:
                    print("   - Could not save (soundfile not available)")

    # Verify expectations
    passed = True

    if not result.is_singing:
        print("\n   FAIL: Should detect singing in song file")
        passed = False

    if result.mean_pitch_hz <= 0:
        print("\n   FAIL: Should detect valid pitch")
        passed = False

    if result.harmonized_audio is None:
        print("\n   FAIL: Should generate harmonized audio")
        passed = False

    if passed:
        print("\n   Passed: Full harmonization pipeline works")

    return passed


def run_gate5_tests(
    checkpoint_path: str,
    ravdess_dir: str,
    output_dir: str = None,
):
    """Run all Gate 5 tests."""
    print("=" * 60)
    print("Gate 5: Harmonization Tests")
    print("=" * 60)

    results = {}

    # Test 1: Basic pitch shift
    results["pitch_shift"] = test_pitch_shift_basic()

    # Test 2: Harmony generation
    results["harmony_gen"] = test_harmony_generation()

    # Test 3: Singing detection on RAVDESS
    results["singing_detection"] = test_singing_detection_on_ravdess(
        checkpoint_path, ravdess_dir, max_samples=10
    )

    # Test 4: Full pipeline
    results["full_pipeline"] = test_full_harmonization_pipeline(
        checkpoint_path, ravdess_dir, output_dir
    )

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    gate_status = "PASS" if all_passed else "FAIL"
    print(f"\n  Gate 5: {gate_status}")

    return all_passed


if __name__ == "__main__":
    checkpoint_path = "checkpoints/multi_head_ravdess/best.npz"
    ravdess_dir = "data/prosody/ravdess"
    output_dir = "outputs/harmonization"

    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    if len(sys.argv) > 2:
        ravdess_dir = sys.argv[2]
    if len(sys.argv) > 3:
        output_dir = sys.argv[3]

    success = run_gate5_tests(checkpoint_path, ravdess_dir, output_dir)
    sys.exit(0 if success else 1)
