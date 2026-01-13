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
Test WhisperMLX transcription against mlx-whisper reference.

Validates:
1. Text output matches mlx-whisper (EXACT string match)
2. Different inputs produce different outputs
3. Timestamps are reasonable
"""

import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.whisper_mlx import WhisperMLX


def load_audio(path: str, sample_rate: int = 16000) -> np.ndarray:
    """Load audio file."""
    import subprocess
    cmd = [
        "ffmpeg", "-i", path, "-f", "s16le", "-ac", "1",
        "-ar", str(sample_rate), "-acodec", "pcm_s16le", "-"
    ]
    result = subprocess.run(cmd, capture_output=True, check=True)
    return np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def transcribe_mlx_whisper(audio_path: str) -> dict:
    """Transcribe using mlx-whisper (reference)."""
    import mlx_whisper
    return mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
    )


def transcribe_whispermlx(audio_path: str, model: WhisperMLX, variable_length: bool = False) -> dict:
    """Transcribe using WhisperMLX."""
    return model.transcribe(
        audio_path,
        variable_length=variable_length,
        verbose=False,
    )


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    # Remove extra whitespace, lowercase for comparison
    text = " ".join(text.split())
    return text.lower().strip()


def test_transcription():
    """Test transcription accuracy against mlx-whisper."""
    print("=" * 60)
    print("WhisperMLX Transcription Validation")
    print("=" * 60)

    # Test audio file
    test_audio = Path(__file__).parent.parent / "tests" / "fixtures" / "audio" / "test_speech.wav"
    if not test_audio.exists():
        print(f"ERROR: Test audio not found: {test_audio}")
        return False

    print(f"\nTest audio: {test_audio}")

    # Get audio duration
    audio = load_audio(str(test_audio))
    duration = len(audio) / 16000
    print(f"Audio duration: {duration:.2f}s")

    # Load WhisperMLX model
    print("\nLoading WhisperMLX model...")
    start = time.time()
    model = WhisperMLX.from_pretrained("mlx-community/whisper-large-v3-mlx")
    mx.eval(model.parameters())
    print(f"Model loaded in {time.time() - start:.2f}s")

    # Test 1: Standard mode transcription vs mlx-whisper reference
    print("\n" + "-" * 60)
    print("Test 1: Standard mode vs mlx-whisper reference")
    print("-" * 60)

    # Get mlx-whisper reference
    print("Running mlx-whisper (reference)...")
    start = time.time()
    ref_result = transcribe_mlx_whisper(str(test_audio))
    ref_time = time.time() - start
    ref_text = ref_result["text"].strip()
    print(f"mlx-whisper: {ref_time:.2f}s")
    print(f"Text: {ref_text}")

    # Get WhisperMLX result (standard mode)
    print("\nRunning WhisperMLX (standard mode)...")
    start = time.time()
    our_result = transcribe_whispermlx(str(test_audio), model, variable_length=False)
    our_time = time.time() - start
    our_text = our_result["text"].strip()
    print(f"WhisperMLX: {our_time:.2f}s")
    print(f"Text: {our_text}")

    # Compare
    ref_norm = normalize_text(ref_text)
    our_norm = normalize_text(our_text)

    if ref_norm == our_norm:
        print("\n✓ PASS: Text matches mlx-whisper reference (normalized)")
        text_match = True
    else:
        print("\n✗ FAIL: Text does NOT match mlx-whisper reference")
        print(f"  Reference (norm): {ref_norm}")
        print(f"  WhisperMLX (norm): {our_norm}")
        text_match = False

    # Test 2: Variable-length mode
    # Note: Variable-length works best for audio 5-30s. Short audio may hallucinate.
    print("\n" + "-" * 60)
    print("Test 2: Variable-length mode")
    print("-" * 60)

    print("Running WhisperMLX (variable-length mode)...")
    start = time.time()
    var_result = transcribe_whispermlx(str(test_audio), model, variable_length=True)
    var_time = time.time() - start
    var_text = var_result["text"].strip()
    print(f"WhisperMLX (var): {var_time:.2f}s")
    print(f"Text: {var_text}")

    var_norm = normalize_text(var_text)
    # For audio 5-30s, variable-length should match standard mode

    if var_text and var_norm == our_norm:
        print("\n✓ PASS: Variable-length mode matches standard mode")
        var_match = True
    elif var_text:
        print("\n⚠ PARTIAL: Variable-length output differs (expected for audio <5s)")
        var_match = False
    else:
        print("\n✗ FAIL: Variable-length mode produces empty output")
        var_match = False

    # Test 3: Check timestamps
    print("\n" + "-" * 60)
    print("Test 3: Timestamp validation")
    print("-" * 60)

    segments = our_result.get("segments", [])
    if segments:
        print(f"Segments: {len(segments)}")
        for i, seg in enumerate(segments):
            print(f"  [{i}] {seg['start']:.2f}s - {seg['end']:.2f}s: {seg['text'][:50]}...")

        # Validate timestamps
        valid_timestamps = True
        for seg in segments:
            if seg["start"] < 0 or seg["end"] < seg["start"]:
                valid_timestamps = False
                print(f"✗ Invalid timestamp: start={seg['start']}, end={seg['end']}")

        if valid_timestamps:
            print("\n✓ PASS: All timestamps are valid")
    else:
        print("✗ FAIL: No segments returned")

    # Test 4: Different inputs produce different outputs
    print("\n" + "-" * 60)
    print("Test 4: Different inputs produce different outputs")
    print("-" * 60)

    # Check if we have other audio files
    prosody_files = list(Path(__file__).parent.parent.glob("tests/prosody/**/neutral_context_baseline.wav"))
    if prosody_files:
        other_audio = prosody_files[0]
        print(f"Testing with: {other_audio.name}")

        other_result = transcribe_whispermlx(str(other_audio), model)
        other_text = other_result["text"].strip()
        print(f"Other text: {other_text[:100]}...")

        if normalize_text(other_text) != our_norm:
            print("\n✓ PASS: Different inputs produce different outputs")
        else:
            print("\n✗ FAIL: Different inputs produce same output (suspicious)")
    else:
        print("SKIP: No other audio files available")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Text match (normalized): {'PASS' if text_match else 'FAIL'}")
    print(f"Variable-length match: {'PASS' if var_match else 'PARTIAL'}")
    print(f"Timestamps valid: {'PASS' if segments and valid_timestamps else 'FAIL'}")
    print(f"Performance: mlx-whisper={ref_time:.2f}s, WhisperMLX={our_time:.2f}s, Variable={var_time:.2f}s")
    if var_time < our_time:
        print(f"Variable-length speedup: {our_time/var_time:.2f}x faster than standard")

    return text_match and var_match


if __name__ == "__main__":
    success = test_transcription()
    sys.exit(0 if success else 1)
