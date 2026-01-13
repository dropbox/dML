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

"""Evaluate singing detection accuracy on RAVDESS validation set."""

import sys
from pathlib import Path
import numpy as np
import mlx.core as mx

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.whisper_mlx.audio import load_audio, log_mel_spectrogram
from tools.whisper_mlx.model import WhisperMLX
from tools.whisper_mlx.multi_head import create_multi_head


def load_checkpoint(checkpoint_path: str, singing_head):
    """Load singing head weights from checkpoint."""

    weights = dict(np.load(checkpoint_path, allow_pickle=True))

    # Extract singing head weights (prefixed with "singing.")
    singing_weights = {}
    for k, v in weights.items():
        if k.startswith("singing."):
            new_key = k[len("singing."):]
            singing_weights[new_key] = mx.array(v)

    # Load weights into head
    flat_params = [(k, v) for k, v in singing_weights.items()]
    singing_head.load_weights(flat_params)

    return singing_head


def evaluate_ravdess(
    model_path: str,
    ravdess_dir: str,
    whisper_model: str = "mlx-community/whisper-large-v3-mlx"
):
    """Evaluate singing detection on RAVDESS."""
    print("=" * 60)
    print("Singing Detection Evaluation - RAVDESS")
    print("=" * 60)

    # Load Whisper
    print("\n1. Loading Whisper model...")
    whisper = WhisperMLX.from_pretrained(whisper_model)
    d_model = whisper.config.n_audio_state

    # Create and load singing head
    print("2. Loading singing detection head...")
    multi_head = create_multi_head("large-v3", ctc_head=None)
    multi_head.singing_head = load_checkpoint(model_path, multi_head.singing_head)

    # Collect RAVDESS test files
    ravdess_path = Path(ravdess_dir)
    audio_files = list(ravdess_path.rglob("*.wav"))

    # Use 10% as test set (same seed as training)
    np.random.seed(42)
    indices = np.random.permutation(len(audio_files))
    val_size = int(len(audio_files) * 0.1)
    test_files = [audio_files[i] for i in indices[:val_size]]

    print(f"3. Testing on {len(test_files)} files...")

    correct = 0
    total = 0
    confusion = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

    for i, audio_path in enumerate(test_files):
        # Parse RAVDESS filename for ground truth
        filename = audio_path.stem
        parts = filename.split("-")
        if len(parts) < 2:
            continue

        vocal_channel = int(parts[1])
        is_singing_gt = vocal_channel == 2  # 02 = song

        # Load and process audio
        try:
            audio = load_audio(str(audio_path))
            mel = log_mel_spectrogram(audio)

            # Pad/trim to 3000 frames
            target_frames = 3000
            if mel.shape[0] < target_frames:
                mel = np.pad(mel, ((0, target_frames - mel.shape[0]), (0, 0)))
            else:
                mel = mel[:target_frames]

            mel_mx = mx.array(mel)[None, ...]  # (1, 3000, n_mels)

            # Get encoder output
            encoder_output = whisper.embed_audio(mel_mx)

            # Predict singing
            singing_logits = multi_head.singing_head(encoder_output)
            singing_prob = mx.sigmoid(singing_logits)
            is_singing_pred = float(singing_prob[0, 0]) > 0.5

            # Track accuracy
            if is_singing_pred == is_singing_gt:
                correct += 1
                if is_singing_gt:
                    confusion["TP"] += 1
                else:
                    confusion["TN"] += 1
            else:
                if is_singing_gt:
                    confusion["FN"] += 1
                else:
                    confusion["FP"] += 1
            total += 1

            if (i + 1) % 50 == 0:
                print(f"   Processed {i + 1}/{len(test_files)}...")

        except Exception as e:
            print(f"   Error processing {audio_path}: {e}")
            continue

    # Results
    accuracy = correct / total * 100 if total > 0 else 0
    precision = confusion["TP"] / (confusion["TP"] + confusion["FP"]) if (confusion["TP"] + confusion["FP"]) > 0 else 0
    recall = confusion["TP"] / (confusion["TP"] + confusion["FN"]) if (confusion["TP"] + confusion["FN"]) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total samples: {total}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print("\nConfusion Matrix:")
    print(f"  TP (Singing correctly detected): {confusion['TP']}")
    print(f"  TN (Speaking correctly detected): {confusion['TN']}")
    print(f"  FP (Speaking classified as Singing): {confusion['FP']}")
    print(f"  FN (Singing classified as Speaking): {confusion['FN']}")

    gate_pass = accuracy >= 90.0
    print(f"\nGate 5 Target (>90%): {'PASS' if gate_pass else 'FAIL'}")

    return accuracy, precision, recall, f1


if __name__ == "__main__":
    evaluate_ravdess(
        model_path="checkpoints/multi_head_ravdess/best.npz",
        ravdess_dir="data/prosody/ravdess"
    )
