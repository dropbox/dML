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

"""Evaluate emotion head checkpoint on prosody validation data."""

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.whisper_mlx.audio import load_audio, log_mel_spectrogram
from tools.whisper_mlx.model import WhisperMLX
from tools.whisper_mlx.multi_head import create_multi_head, EXTENDED_EMOTIONS
from tools.whisper_mlx.ctc_head import create_ctc_draft_head

# Mapping from prosody_type in JSON manifests to EXTENDED_EMOTIONS indices
PROSODY_TYPE_TO_EXTENDED = {
    0: 0,   # neutral
    40: 4,  # angry
    41: 3,  # sad
    42: 2,  # happy
    48: 6,  # disgust
    49: 5,  # fearful
    50: 7,  # surprise
}


def load_checkpoint(checkpoint_path: str, multi_head):
    """Load emotion head weights from checkpoint."""
    from mlx.utils import tree_unflatten

    weights = dict(mx.load(checkpoint_path))

    # Extract emotion head weights
    emotion_weights = {}
    for k, v in weights.items():
        if k.startswith("emotion."):
            # Remove 'emotion.' prefix
            new_key = k[len("emotion."):]
            emotion_weights[new_key] = v

    if emotion_weights:
        # Unflatten and update
        flat_weights = list(emotion_weights.items())
        params = tree_unflatten(flat_weights)
        multi_head.emotion_head.update(params)
        print(f"Loaded {len(emotion_weights)} emotion head weights")

    return multi_head


def evaluate_checkpoint(checkpoint_path: str, prosody_dir: str, max_samples: int = 500):
    """Evaluate emotion checkpoint on validation data."""

    print("Loading Whisper model...")
    whisper_model = WhisperMLX.from_pretrained("mlx-community/whisper-large-v3-mlx")

    print("Creating multi-head...")
    ctc_head = create_ctc_draft_head("large-v3")
    multi_head = create_multi_head("large-v3", ctc_head=ctc_head)

    print(f"Loading checkpoint: {checkpoint_path}")
    multi_head = load_checkpoint(checkpoint_path, multi_head)

    # Load validation samples from prosody manifests
    prosody_path = Path(prosody_dir)
    val_samples = []

    for manifest in ["crema.json", "esd.json", "ravdess.json"]:
        manifest_path = prosody_path / manifest
        if not manifest_path.exists():
            continue

        with open(manifest_path) as f:
            data = json.load(f)

        # Take 10% as validation (matching training split seed=42)
        np.random.seed(42)
        indices = np.random.permutation(len(data))
        val_size = int(len(data) * 0.1)
        val_indices = set(indices[:val_size])

        for i in val_indices:
            entry = data[i]
            audio_path = entry.get("audio_path", "")
            if audio_path and Path(audio_path).exists():
                prosody_type = entry.get("prosody_type", -1)
                emotion_id = PROSODY_TYPE_TO_EXTENDED.get(prosody_type, -1)
                if emotion_id >= 0:
                    val_samples.append({
                        "audio_path": audio_path,
                        "emotion_id": emotion_id,
                        "emotion_label": EXTENDED_EMOTIONS[emotion_id]
                    })

    print(f"Loaded {len(val_samples)} validation samples")

    # Limit samples for quick eval
    if len(val_samples) > max_samples:
        np.random.shuffle(val_samples)
        val_samples = val_samples[:max_samples]
        print(f"Using {max_samples} samples for evaluation")

    # Evaluate
    correct = 0
    total = 0
    confusion = {}  # (true, pred) -> count

    for i, sample in enumerate(val_samples):
        try:
            # Load audio
            audio = load_audio(sample["audio_path"])

            # Compute mel spectrogram
            mel = log_mel_spectrogram(audio)

            # Pad/trim to 30s
            target_frames = 3000
            if mel.shape[0] < target_frames:
                pad_len = target_frames - mel.shape[0]
                mel = np.pad(mel, ((0, pad_len), (0, 0)), mode='constant')
            else:
                mel = mel[:target_frames]

            # Encode
            mel_mx = mx.array(mel)[None, ...]
            encoder_output = whisper_model.embed_audio(mel_mx)

            # Predict emotion
            emotion_logits = multi_head.emotion_head(encoder_output)
            pred_id = int(mx.argmax(emotion_logits, axis=-1)[0])

            true_id = sample["emotion_id"]
            if pred_id == true_id:
                correct += 1

            key = (EXTENDED_EMOTIONS[true_id], EXTENDED_EMOTIONS[min(pred_id, len(EXTENDED_EMOTIONS)-1)])
            confusion[key] = confusion.get(key, 0) + 1

            total += 1

            if (i + 1) % 50 == 0:
                print(f"  Evaluated {i+1}/{len(val_samples)}, accuracy so far: {100*correct/total:.1f}%")

        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            continue

    accuracy = 100 * correct / total if total > 0 else 0
    print("\n=== Results ===")
    print(f"Total samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("Random baseline (7 classes): 14.3%")

    # Print confusion matrix summary
    print("\n=== Top Confusions ===")
    sorted_confusion = sorted(confusion.items(), key=lambda x: -x[1])
    for (true_label, pred_label), count in sorted_confusion[:10]:
        pct = 100 * count / total
        marker = "✓" if true_label == pred_label else "✗"
        print(f"  {marker} {true_label} -> {pred_label}: {count} ({pct:.1f}%)")

    return accuracy


if __name__ == "__main__":
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/emotion_consolidated/step_1000.npz"
    prosody_dir = sys.argv[2] if len(sys.argv) > 2 else "data/prosody"
    max_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 200

    evaluate_checkpoint(checkpoint, prosody_dir, max_samples)
