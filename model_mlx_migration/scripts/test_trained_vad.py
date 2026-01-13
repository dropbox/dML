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
Test trained Encoder VAD head on audio files.
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import mlx.core as mx


def main():
    print("Loading trained VAD head...")
    from tools.whisper_mlx.encoder_vad import load_encoder_vad_head
    from tools.whisper_mlx.model import WhisperMLX
    from tools.whisper_mlx.audio import log_mel_spectrogram

    # Load model
    model = WhisperMLX.from_pretrained("large-v3", warmup=True)

    # Load trained VAD head
    vad_head = load_encoder_vad_head(
        "checkpoints/encoder_vad/encoder_vad_best.npz",
        n_state=model.config.n_audio_state,
        hidden_dim=256,
        dtype=mx.float32,
    )

    # Test on audio file
    test_file = "tests/fixtures/audio/test_speech.wav"
    print(f"\nTesting on: {test_file}")

    audio, sr = sf.read(test_file)
    if sr != 16000:
        ratio = 16000 / sr
        new_len = int(len(audio) * ratio)
        audio = np.interp(
            np.linspace(0, len(audio), new_len),
            np.arange(len(audio)),
            audio
        )

    audio = audio.astype(np.float32)
    duration_s = len(audio) / 16000
    print(f"Audio duration: {duration_s:.2f}s")

    # Compute mel spectrogram
    mel = log_mel_spectrogram(audio, n_mels=model.config.n_mels)

    # Pad to 30s
    target_len = 3000
    if mel.shape[0] < target_len:
        mel = mx.pad(mel, [(0, target_len - mel.shape[0]), (0, 0)])
    elif mel.shape[0] > target_len:
        mel = mel[:target_len, :]

    # Encode
    mel_batch = mel[None].astype(mx.float32)
    encoder_output = model.embed_audio(mel_batch, variable_length=False)
    mx.eval(encoder_output)

    # Run VAD head
    probs = vad_head(encoder_output, training=False)
    mx.eval(probs)

    # Analyze results
    probs_np = np.array(probs[0])

    # Compute metrics for actual audio portion (not padding)
    n_actual_frames = int(duration_s / 30.0 * 1500)
    actual_probs = probs_np[:n_actual_frames]

    speech_ratio = (actual_probs >= 0.5).mean()

    print("\n=== VAD Results ===")
    print(f"Total encoder positions: {len(probs_np)}")
    print(f"Actual audio positions: {n_actual_frames}")
    print(f"Speech probability range: [{actual_probs.min():.3f}, {actual_probs.max():.3f}]")
    print(f"Mean speech probability: {actual_probs.mean():.3f}")
    print(f"Speech ratio (>0.5): {speech_ratio:.1%}")
    print(f"Silence ratio: {1 - speech_ratio:.1%}")

    # Show histogram
    print("\nProbability distribution:")
    for thresh in [0.1, 0.3, 0.5, 0.7, 0.9]:
        count = (actual_probs >= thresh).sum()
        pct = count / len(actual_probs) * 100
        print(f"  >= {thresh}: {count}/{len(actual_probs)} ({pct:.1f}%)")

    # Visualize as simple text timeline
    print("\nTimeline (. = silence, # = speech):")
    timeline_chars = 60
    bins = np.linspace(0, len(actual_probs), timeline_chars + 1).astype(int)
    timeline = ""
    for i in range(timeline_chars):
        bin_probs = actual_probs[bins[i]:bins[i+1]]
        if len(bin_probs) > 0 and bin_probs.mean() >= 0.5:
            timeline += "#"
        else:
            timeline += "."
    print(f"  |{timeline}|")
    print(f"  0s{' ' * (timeline_chars - 6)}{duration_s:.1f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
