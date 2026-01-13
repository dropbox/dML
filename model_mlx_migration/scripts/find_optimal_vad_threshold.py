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
Find optimal threshold for Encoder VAD vs Silero comparison.
"""

import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import mlx.core as mx


def get_silero_vad_labels(audio: np.ndarray, n_encoder_positions: int, vad_model, get_speech_timestamps) -> np.ndarray:
    import torch
    audio_tensor = torch.from_numpy(audio).float()
    speech_timestamps = get_speech_timestamps(
        audio_tensor, vad_model, sampling_rate=16000, threshold=0.5,
        min_speech_duration_ms=100, min_silence_duration_ms=100, return_seconds=True,
    )
    duration_s = len(audio) / 16000
    frame_duration = duration_s / n_encoder_positions
    labels = np.zeros(n_encoder_positions, dtype=np.float32)
    for segment in speech_timestamps:
        start_frame = int(segment['start'] / frame_duration)
        end_frame = int(segment['end'] / frame_duration)
        labels[start_frame:min(end_frame + 1, n_encoder_positions)] = 1.0
    return labels


def main():
    print("Finding optimal threshold for Encoder VAD")
    print("=" * 60)

    # Load data
    from datasets import load_dataset
    ds = load_dataset("librispeech_asr", "clean", split="validation")
    test_samples = list(ds.select(range(min(50, len(ds)))))  # Quick test with 50 samples
    print(f"Loaded {len(test_samples)} samples")

    # Load models
    from tools.whisper_mlx.model import WhisperMLX
    from tools.whisper_mlx.encoder_vad import load_encoder_vad_head
    from tools.whisper_mlx.audio import log_mel_spectrogram

    model = WhisperMLX.from_pretrained("large-v3", warmup=True)
    vad_head = load_encoder_vad_head(
        "checkpoints/encoder_vad/encoder_vad_best.npz",
        n_state=model.config.n_audio_state, hidden_dim=256, dtype=mx.float32,
    )

    import torch
    silero_model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True, verbose=False)
    get_speech_timestamps = utils[0]

    # Collect all predictions and labels
    all_probs = []
    all_labels = []

    for sample in test_samples:
        audio = sample["audio"]["array"].astype(np.float32)
        duration_s = len(audio) / 16000

        mel = log_mel_spectrogram(audio, n_mels=model.config.n_mels)
        target_len = 3000
        if mel.shape[0] < target_len:
            mel = mx.pad(mel, [(0, target_len - mel.shape[0]), (0, 0)])
        elif mel.shape[0] > target_len:
            mel = mel[:target_len, :]

        mel_batch = mel[None].astype(mx.float32)
        encoder_output = model.embed_audio(mel_batch, variable_length=False)
        mx.eval(encoder_output)

        probs = vad_head(encoder_output, training=False)
        mx.eval(probs)
        probs_np = np.array(probs[0])

        n_actual_frames = min(int(duration_s / 30.0 * 1500), 1500)
        actual_probs = probs_np[:n_actual_frames]

        silero_labels = get_silero_vad_labels(audio, n_actual_frames, silero_model, get_speech_timestamps)

        all_probs.extend(actual_probs.tolist())
        all_labels.extend(silero_labels.tolist())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    print(f"\nTotal frames: {len(all_probs)}")
    print(f"Speech frames (Silero): {all_labels.sum():.0f} ({all_labels.mean():.1%})")
    print("\nProb distribution:")
    for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        count = (all_probs >= p).sum()
        print(f"  >= {p}: {count} ({count/len(all_probs):.1%})")

    print(f"\n{'Threshold':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Accuracy':<10}")
    print("-" * 50)

    best_f1 = 0
    best_thresh = 0.5
    best_acc = 0
    best_acc_thresh = 0.5

    for thresh in np.arange(0.1, 0.91, 0.05):
        preds = (all_probs >= thresh).astype(np.float32)
        tp = np.sum((preds == 1) & (all_labels == 1))
        fp = np.sum((preds == 1) & (all_labels == 0))
        fn = np.sum((preds == 0) & (all_labels == 1))
        tn = np.sum((preds == 0) & (all_labels == 0))

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)
        accuracy = (tp + tn) / len(all_probs)

        print(f"{thresh:<10.2f} {precision:<10.3f} {recall:<10.3f} {f1:<10.3f} {accuracy:<10.3f}")

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
        if accuracy > best_acc:
            best_acc = accuracy
            best_acc_thresh = thresh

    print(f"\nBest F1: {best_f1:.3f} at threshold {best_thresh:.2f}")
    print(f"Best Accuracy: {best_acc:.3f} at threshold {best_acc_thresh:.2f}")


if __name__ == "__main__":
    main()
