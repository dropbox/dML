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
Validate Encoder VAD Head accuracy against Silero VAD on LibriSpeech.

Uses LibriSpeech test-clean samples for proper validation.
"""

import sys
import time
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import mlx.core as mx


def get_silero_vad_labels(audio: np.ndarray, n_encoder_positions: int, vad_model, get_speech_timestamps) -> np.ndarray:
    """Get per-position VAD labels from Silero VAD."""
    import torch

    audio_tensor = torch.from_numpy(audio).float()

    speech_timestamps = get_speech_timestamps(
        audio_tensor,
        vad_model,
        sampling_rate=16000,
        threshold=0.5,
        min_speech_duration_ms=100,
        min_silence_duration_ms=100,
        return_seconds=True,
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
    print("=" * 70)
    print("Encoder VAD vs Silero Validation (LibriSpeech)")
    print("=" * 70)

    # Load LibriSpeech test samples
    print("\n[1/5] Loading LibriSpeech validation set...")
    from datasets import load_dataset

    ds = load_dataset("librispeech_asr", "clean", split="validation")
    # Limit to 100 samples for quick validation
    test_samples = list(ds.select(range(min(100, len(ds)))))
    print(f"  Loaded {len(test_samples)} samples")

    # Load models
    print("\n[2/5] Loading Whisper model...")
    from tools.whisper_mlx.model import WhisperMLX
    from tools.whisper_mlx.encoder_vad import load_encoder_vad_head

    model = WhisperMLX.from_pretrained("large-v3", warmup=True)
    print("  Whisper loaded.")

    print("\n[3/5] Loading Encoder VAD head...")
    vad_head = load_encoder_vad_head(
        "checkpoints/encoder_vad/encoder_vad_best.npz",
        n_state=model.config.n_audio_state,
        hidden_dim=256,
        dtype=mx.float32,
    )
    print("  VAD head loaded.")

    print("\n[4/5] Loading Silero VAD...")
    import torch
    silero_model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        trust_repo=True,
        verbose=False
    )
    get_speech_timestamps = utils[0]
    print("  Silero loaded.")

    # Run validation
    print("\n[5/5] Running validation...")
    from tools.whisper_mlx.audio import log_mel_spectrogram

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    total_vad_time = 0.0
    total_audio_duration = 0.0

    for i, sample in enumerate(test_samples):
        audio = sample["audio"]["array"].astype(np.float32)
        duration_s = len(audio) / 16000
        total_audio_duration += duration_s

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

        # Run VAD head (timed)
        t0 = time.perf_counter()
        probs = vad_head(encoder_output, training=False)
        mx.eval(probs)
        t1 = time.perf_counter()
        total_vad_time += (t1 - t0)

        probs_np = np.array(probs[0])

        # Get actual frame count
        n_actual_frames = min(int(duration_s / 30.0 * 1500), 1500)
        actual_probs = probs_np[:n_actual_frames]

        # Get Silero labels
        silero_labels = get_silero_vad_labels(audio, n_actual_frames, silero_model, get_speech_timestamps)

        # Compute metrics at 0.5 threshold
        preds = (actual_probs >= 0.5).astype(np.float32)

        tp = np.sum((preds == 1) & (silero_labels == 1))
        fp = np.sum((preds == 1) & (silero_labels == 0))
        fn = np.sum((preds == 0) & (silero_labels == 1))
        tn = np.sum((preds == 0) & (silero_labels == 0))

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

        if (i + 1) % 20 == 0:
            running_precision = total_tp / max(total_tp + total_fp, 1)
            running_recall = total_tp / max(total_tp + total_fn, 1)
            running_f1 = 2 * running_precision * running_recall / max(running_precision + running_recall, 1e-6)
            print(f"  Progress: {i+1}/{len(test_samples)}, F1={running_f1:.3f}")

    # Final metrics
    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn)

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print("\nAccuracy vs Silero VAD @ threshold=0.5:")
    print(f"  Precision: {precision:.1%} (how much of detected speech is correct)")
    print(f"  Recall:    {recall:.1%} (how much speech was detected)")
    print(f"  F1 Score:  {f1:.1%}")
    print(f"  Accuracy:  {accuracy:.1%}")

    print("\nVAD Head Overhead:")
    print(f"  Total VAD time: {total_vad_time * 1000:.2f}ms")
    print(f"  Total audio:    {total_audio_duration:.2f}s")
    print(f"  Overhead:       {(total_vad_time / total_audio_duration) * 100:.4f}%")

    print("\nConfusion Matrix:")
    print(f"  TP={int(total_tp)}, FP={int(total_fp)}")
    print(f"  FN={int(total_fn)}, TN={int(total_tn)}")

    # Check target
    print("\n" + "-" * 70)
    target_accuracy = 0.95
    if accuracy >= target_accuracy:
        print(f"PASS: Accuracy {accuracy:.1%} >= {target_accuracy:.0%} target")
        return 0
    else:
        print(f"FAIL: Accuracy {accuracy:.1%} < {target_accuracy:.0%} target")
        print(f"  Gap: {(target_accuracy - accuracy) * 100:.1f}pp")
        return 1


if __name__ == "__main__":
    sys.exit(main())
