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

"""Compare mel spectrograms between Python and C++ for file 0004."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import mlx.core as mx


TEST_FILE = "data/librispeech/dev-clean/1272/128104/1272-128104-0004.flac"


def get_python_mel_and_audio():
    """Get mel spectrogram from Python WhisperMLX."""
    from tools.whisper_mlx import WhisperMLX
    from tools.whisper_mlx.silero_vad import SileroVADProcessor
    from tools.whisper_mlx.audio import load_audio, log_mel_spectrogram

    # Load audio
    audio = load_audio(TEST_FILE, sample_rate=16000)
    print(f"Raw audio: {len(audio)} samples ({len(audio)/16000:.2f}s)")

    # Load model to get mel filters
    model = WhisperMLX.from_pretrained("mlx-community/whisper-large-v3-turbo", dtype=mx.float16)

    # VAD preprocess (matching Python default)
    vad = SileroVADProcessor()
    result = vad.get_speech_segments(audio)

    # Concatenate speech segments
    vad_audio = np.concatenate([audio[seg.start_sample:seg.end_sample] for seg in result.segments])
    print(f"VAD audio: {len(vad_audio)} samples ({len(vad_audio)/16000:.2f}s)")
    print(f"VAD segments: {len(result.segments)}")
    for i, seg in enumerate(result.segments):
        print(f"  Seg {i}: {seg.start_sample/16000:.2f}s - {seg.end_sample/16000:.2f}s ({(seg.end_sample-seg.start_sample)/16000:.2f}s)")

    # Compute mel spectrogram
    # Pad to 30 seconds (480000 samples) like Python does
    padded_audio = np.zeros(480000, dtype=np.float32)
    padded_audio[:min(len(vad_audio), 480000)] = vad_audio[:min(len(vad_audio), 480000)]

    # Compute mel spectrogram using model's n_mels (128 for large-v3)
    n_mels = 128  # large-v3 uses 128 mel bins
    mel = log_mel_spectrogram(padded_audio, n_mels=n_mels)
    print(f"Mel shape: {mel.shape}")

    return mel, vad_audio, result


def main():
    print("=" * 60)
    print("MEL SPECTROGRAM COMPARISON - File 0004")
    print("=" * 60)

    # Get Python mel
    print("\n--- Python ---")
    py_mel, py_audio, vad_result = get_python_mel_and_audio()

    # Convert to numpy for analysis
    py_mel_np = np.array(py_mel)

    print("\nPython mel stats:")
    print(f"  Shape: {py_mel_np.shape}")
    print(f"  Min: {py_mel_np.min():.4f}")
    print(f"  Max: {py_mel_np.max():.4f}")
    print(f"  Mean: {py_mel_np.mean():.4f}")
    print(f"  Std: {py_mel_np.std():.4f}")

    # Save audio for C++ to read
    print("\nSaving VAD-processed audio to /tmp/vad_audio_0004.raw...")
    py_audio_f32 = np.array(py_audio, dtype=np.float32)
    py_audio_f32.tofile("/tmp/vad_audio_0004.raw")
    print(f"  Written {len(py_audio_f32)} samples")

    # Save mel for comparison
    print("Saving Python mel to /tmp/python_mel_0004.npy...")
    np.save("/tmp/python_mel_0004.npy", py_mel_np)

    # Print first few mel values at key positions
    print("\nFirst 10 mel values at position 0:")
    print(f"  {py_mel_np[0, :10]}")

    print("\nFirst 10 mel values at position 500 (middle of audio):")
    print(f"  {py_mel_np[500, :10]}")

    # Check for NaN or Inf
    nan_count = np.isnan(py_mel_np).sum()
    inf_count = np.isinf(py_mel_np).sum()
    print(f"\nNaN count: {nan_count}, Inf count: {inf_count}")


if __name__ == "__main__":
    main()
