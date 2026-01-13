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
Evaluate singing detection accuracy on RAVDESS dataset.

RAVDESS filename format: Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor.wav
- VocalChannel 1 = Speech
- VocalChannel 2 = Song
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .audio import load_audio, log_mel_spectrogram
from .model import WhisperMLX


def parse_ravdess_label(filename: str) -> bool:
    """Parse RAVDESS filename to extract singing label.

    Returns True if singing (vocal_channel=2), False if speech (vocal_channel=1).
    Returns None if invalid.
    """
    stem = Path(filename).stem
    parts = stem.split("-")
    if len(parts) < 2:
        return None
    try:
        vocal_channel = int(parts[1])
        return vocal_channel == 2  # 2 = Song
    except ValueError:
        return None


def load_singing_head(checkpoint_path: str) -> nn.Module:
    """Load the trained singing head from checkpoint.

    Automatically infers architecture parameters (hidden_dim, num_singing_styles)
    from checkpoint weight shapes, making it compatible with any checkpoint
    regardless of how it was trained.
    """
    from .multi_head import ExtendedSingingHead, MultiHeadConfig, SingingHead

    # Load checkpoint
    checkpoint = dict(mx.load(checkpoint_path))

    # Extract singing head weights (prefix is "singing." not "singing_head.")
    singing_params = {
        k.replace("singing.", ""): v
        for k, v in checkpoint.items()
        if k.startswith("singing.")
    }

    if not singing_params:
        raise ValueError(f"No singing weights found in checkpoint. Keys: {list(checkpoint.keys())[:10]}")

    # Detect if this is ExtendedSingingHead (has shared_fc, singing_fc, etc.)
    use_extended = "shared_fc.weight" in singing_params

    # Create config for singing head (large-v3 dims)
    if use_extended:
        # ExtendedSingingHead config - infer all parameters from weights
        # hidden_dim from shared_fc.weight shape[0]
        if "shared_fc.weight" in singing_params:
            hidden_dim = singing_params["shared_fc.weight"].shape[0]
        else:
            hidden_dim = 256  # Fallback default

        # Infer num_singing_styles from style_fc.weight shape[0]
        if "style_fc.weight" in singing_params:
            num_styles = singing_params["style_fc.weight"].shape[0]
        else:
            num_styles = 10  # Fallback default

        config = MultiHeadConfig(
            d_model=1280,  # large-v3 encoder dim
            singing_hidden_dim=hidden_dim,
            singing_style_hidden_dim=hidden_dim,  # ExtendedSingingHead uses this
            num_singing_styles=num_styles,
        )

        # Initialize extended singing head
        singing_head = ExtendedSingingHead(config)
        print(f"Loading ExtendedSingingHead (hidden_dim={hidden_dim}, num_styles={num_styles})")
    else:
        # Basic SingingHead config - binary classification, no styles
        if "fc1.weight" in singing_params:
            hidden_dim = singing_params["fc1.weight"].shape[0]
        else:
            hidden_dim = 128  # Fallback default

        # SingingHead is binary (singing vs speaking), num_singing_styles not used
        config = MultiHeadConfig(
            d_model=1280,  # large-v3 encoder dim
            singing_hidden_dim=hidden_dim,
            num_singing_styles=2,  # Not used by SingingHead, but required by config
        )

        # Initialize singing head
        singing_head = SingingHead(config)
        print(f"Loading SingingHead (hidden_dim={hidden_dim}, binary classification)")

    # Load into model
    singing_head.load_weights(list(singing_params.items()))
    mx.eval(singing_head.parameters())

    return singing_head


def evaluate_singing(
    data_dir: str,
    checkpoint_path: str,
    model_size: str = "large-v3",
    threshold: float = 0.5,
    max_samples: int = 0,
    actors_held_out: list[int] = None,
) -> dict:
    """Evaluate singing detection accuracy.

    Args:
        data_dir: Path to RAVDESS data
        checkpoint_path: Path to singing head checkpoint
        model_size: Whisper model size
        threshold: Classification threshold
        max_samples: Max samples to evaluate (0=all)
        actors_held_out: List of actor IDs for test set (default: [23, 24])

    Returns:
        Dict with accuracy metrics
    """
    if actors_held_out is None:
        actors_held_out = [23, 24]  # Last 2 actors for test

    # Find all wav files
    data_path = Path(data_dir)
    wav_files = list(data_path.rglob("*.wav"))
    print(f"Found {len(wav_files)} total wav files")

    # Filter to test set (held-out actors)
    test_files = []
    for f in wav_files:
        # Extract actor from filename
        parts = f.stem.split("-")
        if len(parts) >= 7:
            try:
                actor = int(parts[6])
                if actor in actors_held_out:
                    label = parse_ravdess_label(f.name)
                    if label is not None:
                        test_files.append((f, label))
            except ValueError:
                continue

    print(f"Test set (actors {actors_held_out}): {len(test_files)} files")

    if max_samples > 0 and len(test_files) > max_samples:
        # Random subsample (not class-balanced, just random)
        import random
        random.seed(42)
        random.shuffle(test_files)
        test_files = test_files[:max_samples]

    # Count labels
    n_speech = sum(1 for _, is_singing in test_files if not is_singing)
    n_song = sum(1 for _, is_singing in test_files if is_singing)
    print(f"  Speech: {n_speech}, Song: {n_song}")

    # Load Whisper encoder
    print(f"\nLoading Whisper {model_size} encoder...")
    model = WhisperMLX.from_pretrained(model_size)

    # Load singing head
    print(f"Loading singing head from {checkpoint_path}...")
    singing_head = load_singing_head(checkpoint_path)

    # Evaluate
    results = {
        "true_positives": 0,  # Correctly predicted singing
        "true_negatives": 0,  # Correctly predicted speech
        "false_positives": 0,  # Predicted singing but was speech
        "false_negatives": 0,  # Predicted speech but was singing
        "predictions": [],
    }

    start_time = time.time()

    for i, (audio_path, is_singing_true) in enumerate(test_files):
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"  Processed {i + 1}/{len(test_files)} ({rate:.1f} files/sec)")

        try:
            # Load and preprocess audio
            audio = load_audio(str(audio_path))

            # Mel spectrogram
            mel = log_mel_spectrogram(audio)

            # Pad/trim to 30s (3000 mel frames -> 1500 encoder frames)
            target_frames = 3000
            if mel.shape[0] < target_frames:
                pad_len = target_frames - mel.shape[0]
                mel = np.pad(mel, ((0, pad_len), (0, 0)), mode='constant')
            else:
                mel = mel[:target_frames]

            # Convert to MLX - shape (batch, n_frames, n_mels)
            mel_mx = mx.array(mel)[None, ...]  # (1, 3000, n_mels)

            # Run encoder via embed_audio
            encoder_output = model.embed_audio(mel_mx)
            mx.eval(encoder_output)

            # Run singing head
            # ExtendedSingingHead returns (singing_logit, style_logits, intensity)
            # Basic SingingHead returns just logits
            result = singing_head(encoder_output)
            if isinstance(result, tuple):
                # ExtendedSingingHead
                logits = result[0]  # singing_logit (batch, 2)
            else:
                logits = result  # (batch, 1)

            # Handle different output shapes
            if logits.shape[-1] == 2:
                # ExtendedSingingHead uses 2-class softmax
                probs = mx.softmax(logits, axis=-1)
                singing_prob = float(probs[0, 1])  # prob of class 1 (singing)
            else:
                # Basic SingingHead uses sigmoid
                singing_prob = float(mx.sigmoid(logits)[0, 0])

            is_singing_pred = singing_prob > threshold

            # Update counts
            if is_singing_true and is_singing_pred:
                results["true_positives"] += 1
            elif not is_singing_true and not is_singing_pred:
                results["true_negatives"] += 1
            elif not is_singing_true and is_singing_pred:
                results["false_positives"] += 1
            else:  # is_singing_true and not is_singing_pred
                results["false_negatives"] += 1

            results["predictions"].append({
                "file": str(audio_path),
                "true_label": is_singing_true,
                "predicted": is_singing_pred,
                "probability": singing_prob,
            })

        except Exception as e:
            print(f"  Error processing {audio_path}: {e}")
            continue

    elapsed = time.time() - start_time

    # Calculate metrics
    tp, tn, fp, fn = (
        results["true_positives"],
        results["true_negatives"],
        results["false_positives"],
        results["false_negatives"],
    )

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    results["metrics"] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_samples": total,
        "elapsed_seconds": elapsed,
        "samples_per_second": total / elapsed if elapsed > 0 else 0,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate singing detection")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/prosody/ravdess",
        help="Path to RAVDESS data",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/multi_head_ravdess/best.npz",
        help="Path to singing head checkpoint",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="large-v3",
        help="Whisper model size",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Max samples (0=all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output JSON path (optional)",
    )
    args = parser.parse_args()

    results = evaluate_singing(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        model_size=args.model_size,
        threshold=args.threshold,
        max_samples=args.max_samples,
    )

    # Print summary
    m = results["metrics"]
    print("\n" + "=" * 60)
    print("SINGING DETECTION EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total samples:    {m['total_samples']}")
    print(f"Accuracy:         {m['accuracy']*100:.2f}%")
    print(f"Precision:        {m['precision']*100:.2f}%")
    print(f"Recall:           {m['recall']*100:.2f}%")
    print(f"F1 Score:         {m['f1']*100:.2f}%")
    print(f"Processing time:  {m['elapsed_seconds']:.1f}s ({m['samples_per_second']:.1f} files/sec)")
    print("-" * 60)
    print("Confusion Matrix:")
    print(f"  True Positives (Song→Song):     {results['true_positives']}")
    print(f"  True Negatives (Speech→Speech): {results['true_negatives']}")
    print(f"  False Positives (Speech→Song):  {results['false_positives']}")
    print(f"  False Negatives (Song→Speech):  {results['false_negatives']}")
    print("=" * 60)

    if args.output:
        # Remove raw predictions for JSON (too large)
        results_json = {k: v for k, v in results.items() if k != "predictions"}
        with open(args.output, "w") as f:
            json.dump(results_json, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
