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
Evaluate emotion classification accuracy on emotion datasets.

Supports evaluation on:
- RAVDESS: 8-class emotion (actors 23, 24 held out for test)
- Consolidated: 7-class emotion with validation split
- Unified: 10-class emotion with validation split (when available)
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .audio import load_audio, log_mel_spectrogram
from .model import WhisperMLX
from .multi_head import EXTENDED_EMOTIONS, RAVDESS_EMOTIONS


def load_emotion_head(checkpoint_path: str) -> tuple[nn.Module, int]:
    """Load the trained emotion head from checkpoint.

    Returns:
        Tuple of (emotion_head, num_emotions)
    """
    from .multi_head import EmotionHead, MultiHeadConfig

    # Load checkpoint
    checkpoint = dict(mx.load(checkpoint_path))

    # Determine num_emotions from checkpoint weights
    # fc2 output dim = num_emotions
    if "emotion.fc2.weight" in checkpoint:
        num_emotions = checkpoint["emotion.fc2.weight"].shape[0]
    else:
        # Try to find emotion weights with different prefix
        for key in checkpoint.keys():
            if key.endswith("fc2.weight") and "emotion" in key.lower():
                num_emotions = checkpoint[key].shape[0]
                break
        else:
            num_emotions = 34  # Default

    # Check hidden dim from fc1
    if "emotion.fc1.weight" in checkpoint:
        hidden_dim = checkpoint["emotion.fc1.weight"].shape[0]
    else:
        hidden_dim = 512  # Default

    config = MultiHeadConfig(
        d_model=1280,  # large-v3 encoder dim
        emotion_hidden_dim=hidden_dim,
        num_emotions=num_emotions,
    )

    # Initialize emotion head
    emotion_head = EmotionHead(config)

    # Extract emotion head weights
    emotion_params = {
        k.replace("emotion.", ""): v
        for k, v in checkpoint.items()
        if k.startswith("emotion.")
    }

    if not emotion_params:
        raise ValueError(f"No emotion weights found in checkpoint. Keys: {list(checkpoint.keys())[:10]}")

    # Load into model
    emotion_head.load_weights(list(emotion_params.items()))
    mx.eval(emotion_head.parameters())

    return emotion_head, num_emotions


def parse_ravdess_emotion(filename: str) -> int | None:
    """Parse RAVDESS filename to extract emotion label.

    RAVDESS filename format: Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor.wav
    Emotion codes: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised

    Returns 0-indexed emotion ID (0-7) or None if invalid.
    """
    stem = Path(filename).stem
    parts = stem.split("-")
    if len(parts) < 3:
        return None
    try:
        emotion_code = int(parts[2])
        if 1 <= emotion_code <= 8:
            return emotion_code - 1  # Convert to 0-indexed
        return None
    except ValueError:
        return None


def evaluate_ravdess(
    data_dir: str,
    checkpoint_path: str,
    model_size: str = "large-v3",
    actors_held_out: list[int] = None,
    max_samples: int = 0,
) -> dict:
    """Evaluate emotion classification on RAVDESS dataset.

    Args:
        data_dir: Path to RAVDESS data
        checkpoint_path: Path to emotion head checkpoint
        model_size: Whisper model size
        actors_held_out: List of actor IDs for test set (default: [23, 24])
        max_samples: Max samples to evaluate (0=all)

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
        parts = f.stem.split("-")
        if len(parts) >= 7:
            try:
                actor = int(parts[6])
                if actor in actors_held_out:
                    emotion_id = parse_ravdess_emotion(f.name)
                    if emotion_id is not None:
                        test_files.append((f, emotion_id))
            except ValueError:
                continue

    print(f"Test set (actors {actors_held_out}): {len(test_files)} files")

    if max_samples > 0 and len(test_files) > max_samples:
        import random
        random.seed(42)
        random.shuffle(test_files)
        test_files = test_files[:max_samples]

    # Count labels
    emotion_counts = defaultdict(int)
    for _, eid in test_files:
        emotion_counts[eid] += 1
    print("Emotion distribution in test set:")
    for eid, count in sorted(emotion_counts.items()):
        print(f"  {RAVDESS_EMOTIONS[eid]}: {count}")

    # Load Whisper encoder
    print(f"\nLoading Whisper {model_size} encoder...")
    model = WhisperMLX.from_pretrained(model_size)

    # Load emotion head
    print(f"Loading emotion head from {checkpoint_path}...")
    emotion_head, num_emotions = load_emotion_head(checkpoint_path)
    print(f"Emotion head has {num_emotions} classes")

    # Evaluate
    results = {
        "correct": 0,
        "total": 0,
        "per_class_correct": defaultdict(int),
        "per_class_total": defaultdict(int),
        "confusion_matrix": defaultdict(lambda: defaultdict(int)),
        "predictions": [],
    }

    start_time = time.time()

    for i, (audio_path, true_emotion_id) in enumerate(test_files):
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

            # Convert to MLX
            mel_mx = mx.array(mel)[None, ...]  # (1, 3000, n_mels)

            # Run encoder
            encoder_output = model.embed_audio(mel_mx)
            mx.eval(encoder_output)

            # Run emotion head
            logits = emotion_head(encoder_output)  # (batch, num_emotions)
            probs = mx.softmax(logits, axis=-1)
            pred_emotion_id = int(mx.argmax(logits, axis=-1)[0])
            confidence = float(probs[0, pred_emotion_id])

            # Update counts
            is_correct = pred_emotion_id == true_emotion_id
            results["total"] += 1
            if is_correct:
                results["correct"] += 1
                results["per_class_correct"][true_emotion_id] += 1
            results["per_class_total"][true_emotion_id] += 1
            results["confusion_matrix"][true_emotion_id][pred_emotion_id] += 1

            results["predictions"].append({
                "file": str(audio_path),
                "true_emotion": true_emotion_id,
                "predicted_emotion": pred_emotion_id,
                "confidence": confidence,
                "correct": is_correct,
            })

        except Exception as e:
            print(f"  Error processing {audio_path}: {e}")
            continue

    elapsed = time.time() - start_time

    # Calculate metrics
    total = results["total"]
    accuracy = results["correct"] / total if total > 0 else 0

    # Per-class accuracy
    per_class_accuracy = {}
    for eid in range(8):  # RAVDESS has 8 emotions
        class_total = results["per_class_total"].get(eid, 0)
        class_correct = results["per_class_correct"].get(eid, 0)
        if class_total > 0:
            per_class_accuracy[RAVDESS_EMOTIONS[eid]] = class_correct / class_total

    results["metrics"] = {
        "accuracy": accuracy,
        "per_class_accuracy": per_class_accuracy,
        "total_samples": total,
        "num_emotions": num_emotions,
        "elapsed_seconds": elapsed,
        "samples_per_second": total / elapsed if elapsed > 0 else 0,
    }

    return results


def evaluate_consolidated(
    data_dir: str,
    checkpoint_path: str,
    model_size: str = "large-v3",
    split: str = "validation",
    max_samples: int = 0,
) -> dict:
    """Evaluate emotion classification on consolidated emotion dataset.

    Args:
        data_dir: Path to consolidated dataset
        checkpoint_path: Path to emotion head checkpoint
        model_size: Whisper model size
        split: Dataset split to evaluate ("train" or "validation")
        max_samples: Max samples to evaluate (0=all)

    Returns:
        Dict with accuracy metrics
    """
    from datasets import load_from_disk

    # Load dataset
    print(f"Loading consolidated dataset from {data_dir}...")
    try:
        ds = load_from_disk(data_dir)
        if split in ds:
            dataset = ds[split]
        else:
            # Single split dataset
            dataset = ds
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return {"error": str(e)}

    print(f"Dataset has {len(dataset)} samples")

    # Get emotion column name
    if "emotion" in dataset.column_names:
        emotion_col = "emotion"
    elif "label" in dataset.column_names:
        emotion_col = "label"
    else:
        print(f"No emotion column found. Columns: {dataset.column_names}")
        return {"error": "No emotion column"}

    # Build emotion mapping from dataset
    unique_emotions = sorted(set(dataset[emotion_col]))
    print(f"Found {len(unique_emotions)} unique emotions: {unique_emotions[:10]}...")

    # Create mapping
    if all(isinstance(e, int) for e in unique_emotions):
        emotion_to_id = {e: e for e in unique_emotions}
    else:
        emotion_to_id = {e: i for i, e in enumerate(unique_emotions)}

    if max_samples > 0 and len(dataset) > max_samples:
        import random
        indices = list(range(len(dataset)))
        random.seed(42)
        random.shuffle(indices)
        dataset = dataset.select(indices[:max_samples])

    # Load Whisper encoder
    print(f"\nLoading Whisper {model_size} encoder...")
    model = WhisperMLX.from_pretrained(model_size)

    # Load emotion head
    print(f"Loading emotion head from {checkpoint_path}...")
    emotion_head, num_emotions = load_emotion_head(checkpoint_path)
    print(f"Emotion head has {num_emotions} classes")

    # Evaluate
    results = {
        "correct": 0,
        "total": 0,
        "per_class_correct": defaultdict(int),
        "per_class_total": defaultdict(int),
        "confusion_matrix": defaultdict(lambda: defaultdict(int)),
    }

    start_time = time.time()

    for i, example in enumerate(dataset):
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"  Processed {i + 1}/{len(dataset)} ({rate:.1f} samples/sec)")

        try:
            # Get audio
            audio_data = example["audio"]
            if isinstance(audio_data, dict):
                audio = np.array(audio_data["array"], dtype=np.float32)
                sr = audio_data["sampling_rate"]
                # Resample if needed
                if sr != 16000:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            elif hasattr(audio_data, 'get_all_samples'):
                # Handle datasets AudioDecoder format
                samples = audio_data.get_all_samples()
                audio = samples.data.numpy().squeeze().astype(np.float32)
                sr = samples.sample_rate
                if sr != 16000:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            else:
                audio = np.array(audio_data, dtype=np.float32)

            # Get true emotion
            true_emotion = example[emotion_col]
            if isinstance(true_emotion, str):
                true_emotion_id = emotion_to_id.get(true_emotion, 0)
            else:
                true_emotion_id = int(true_emotion)

            # Mel spectrogram
            mel = log_mel_spectrogram(audio)

            # Pad/trim
            target_frames = 3000
            if mel.shape[0] < target_frames:
                pad_len = target_frames - mel.shape[0]
                mel = np.pad(mel, ((0, pad_len), (0, 0)), mode='constant')
            else:
                mel = mel[:target_frames]

            # Convert to MLX
            mel_mx = mx.array(mel)[None, ...]

            # Run encoder
            encoder_output = model.embed_audio(mel_mx)
            mx.eval(encoder_output)

            # Run emotion head
            logits = emotion_head(encoder_output)
            pred_emotion_id = int(mx.argmax(logits, axis=-1)[0])

            # Update counts
            is_correct = pred_emotion_id == true_emotion_id
            results["total"] += 1
            if is_correct:
                results["correct"] += 1
                results["per_class_correct"][true_emotion_id] += 1
            results["per_class_total"][true_emotion_id] += 1
            results["confusion_matrix"][true_emotion_id][pred_emotion_id] += 1

        except Exception as e:
            if i < 5:  # Only print first few errors
                print(f"  Error processing sample {i}: {e}")
            continue

    elapsed = time.time() - start_time

    # Calculate metrics
    total = results["total"]
    accuracy = results["correct"] / total if total > 0 else 0

    # Per-class accuracy
    per_class_accuracy = {}
    for eid in results["per_class_total"]:
        class_total = results["per_class_total"][eid]
        class_correct = results["per_class_correct"][eid]
        if class_total > 0:
            # Get label name
            if eid < len(EXTENDED_EMOTIONS):
                label = EXTENDED_EMOTIONS[eid]
            else:
                label = f"emotion_{eid}"
            per_class_accuracy[label] = class_correct / class_total

    results["metrics"] = {
        "accuracy": accuracy,
        "per_class_accuracy": per_class_accuracy,
        "total_samples": total,
        "num_emotions": num_emotions,
        "elapsed_seconds": elapsed,
        "samples_per_second": total / elapsed if elapsed > 0 else 0,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate emotion classification")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/prosody/ravdess",
        help="Path to emotion data",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        choices=["ravdess", "consolidated", "unified"],
        default="ravdess",
        help="Dataset type to evaluate",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/emotion_consolidated/step_11500.npz",
        help="Path to emotion head checkpoint",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="large-v3",
        help="Whisper model size",
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

    if args.dataset_type == "ravdess":
        results = evaluate_ravdess(
            data_dir=args.data_dir,
            checkpoint_path=args.checkpoint,
            model_size=args.model_size,
            max_samples=args.max_samples,
        )
    else:
        results = evaluate_consolidated(
            data_dir=args.data_dir,
            checkpoint_path=args.checkpoint,
            model_size=args.model_size,
            max_samples=args.max_samples,
        )

    # Print summary
    if "error" in results:
        print(f"\nERROR: {results['error']}")
        return

    m = results["metrics"]
    print("\n" + "=" * 60)
    print("EMOTION CLASSIFICATION EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total samples:      {m['total_samples']}")
    print(f"Num emotion classes: {m['num_emotions']}")
    print(f"Overall Accuracy:   {m['accuracy']*100:.2f}%")
    print(f"Processing time:    {m['elapsed_seconds']:.1f}s ({m['samples_per_second']:.1f} samples/sec)")
    print("-" * 60)
    print("Per-class accuracy:")
    for emotion, acc in sorted(m['per_class_accuracy'].items(), key=lambda x: -x[1]):
        print(f"  {emotion}: {acc*100:.1f}%")
    print("=" * 60)

    # Print confusion matrix summary
    if "confusion_matrix" in results:
        cm = results["confusion_matrix"]
        print("\nConfusion Matrix (rows=true, cols=predicted):")
        print("  Most common confusions:")
        confusions = []
        for true_id, preds in cm.items():
            for pred_id, count in preds.items():
                if true_id != pred_id and count > 0:
                    if true_id < len(EXTENDED_EMOTIONS):
                        true_label = EXTENDED_EMOTIONS[true_id]
                    else:
                        true_label = f"emotion_{true_id}"
                    if pred_id < len(EXTENDED_EMOTIONS):
                        pred_label = EXTENDED_EMOTIONS[pred_id]
                    else:
                        pred_label = f"emotion_{pred_id}"
                    confusions.append((true_label, pred_label, count))

        confusions.sort(key=lambda x: -x[2])
        for true_label, pred_label, count in confusions[:10]:
            print(f"    {true_label} -> {pred_label}: {count}")

    if args.output:
        # Remove raw predictions for JSON
        results_json = {k: v for k, v in results.items() if k != "predictions"}
        # Convert defaultdicts to regular dicts for JSON serialization
        if "confusion_matrix" in results_json:
            results_json["confusion_matrix"] = {
                str(k): dict(v) for k, v in results_json["confusion_matrix"].items()
            }
        if "per_class_correct" in results_json:
            results_json["per_class_correct"] = dict(results_json["per_class_correct"])
        if "per_class_total" in results_json:
            results_json["per_class_total"] = dict(results_json["per_class_total"])

        with open(args.output, "w") as f:
            json.dump(results_json, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
