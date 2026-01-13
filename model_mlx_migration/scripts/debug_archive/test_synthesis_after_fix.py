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

"""Test full synthesis after dilation fix."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "pytorch_to_mlx"))

import mlx.core as mx
import numpy as np
import soundfile as sf
import torch
from converters.kokoro_converter import KokoroConverter

print("=== Loading model ===")
converter = KokoroConverter()
model, config, pt_state = converter.load_from_hf("hexgrad/Kokoro-82M")
mx.eval(model.parameters())

# Load voice
voice_path = Path.home() / "models" / "kokoro" / "voices" / "af_heart.pt"
voice_pack = torch.load(voice_path, map_location="cpu", weights_only=False)
style = mx.array(voice_pack[5].numpy())

# Test tokens
tokens = mx.array([[0, 47, 44, 51, 51, 54, 0]])

print("\n=== Running synthesize ===")
audio = model.synthesize(tokens, style)
mx.eval(audio)

print(f"Audio shape: {audio.shape}")
print(f"Audio range: [{float(audio.min()):.4f}, {float(audio.max()):.4f}]")
print(f"Audio RMS: {float(mx.sqrt(mx.mean(audio**2))):.4f}")

# Check clipping
clipped = (mx.abs(audio) > 0.99).astype(mx.float32)
print(f"Clipping percentage: {float(mx.mean(clipped)) * 100:.2f}%")

# Save audio for manual inspection
audio_np = np.array(audio.squeeze())
sf.write("/tmp/test_synthesis.wav", audio_np, 24000)
print("\nSaved to /tmp/test_synthesis.wav")

# Check frequency content
print("\n=== Frequency analysis ===")
fft = np.fft.rfft(audio_np)
freqs = np.fft.rfftfreq(len(audio_np), 1 / 24000)
mag = np.abs(fft)

# Find top 5 frequencies
top_indices = np.argsort(mag)[-5:][::-1]
print("Top 5 frequencies by magnitude:")
for idx in top_indices:
    print(f"  {freqs[idx]:.1f} Hz: magnitude {mag[idx]:.2f}")

# Check if energy is in speech range (80-400 Hz)
speech_mask = (freqs >= 80) & (freqs <= 400)
speech_energy = np.sum(mag[speech_mask] ** 2)
total_energy = np.sum(mag**2)
speech_ratio = speech_energy / (total_energy + 1e-8)
print(f"\nSpeech range (80-400 Hz) energy ratio: {speech_ratio * 100:.1f}%")

# Run whisper
print("\n=== Whisper transcription ===")
try:
    import mlx_whisper

    result = mlx_whisper.transcribe(
        "/tmp/test_synthesis.wav", path_or_hf_repo="mlx-community/whisper-small"
    )
    print(f"Transcription: '{result['text']}'")
except Exception as e:
    print(f"Whisper error: {e}")
