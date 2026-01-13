#!/usr/bin/env python3
# Copyright 2024-2026 Andrew Yates
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
Evaluate trained rich audio heads (emotion, paralinguistics) on test sets.

Usage:
    python scripts/evaluate_rich_audio_head.py \
        --checkpoint checkpoints/paralinguistics_v8_extended/head_best.npz \
        --encoder checkpoints/zipformer/en-streaming/exp/pretrained.pt \
        --head paralinguistics \
        --data-dir data/paralinguistics/vocalsound_labeled

    python scripts/evaluate_rich_audio_head.py \
        --checkpoint checkpoints/rich_audio_heads/emotion_v8_final/head_best.npz \
        --encoder checkpoints/zipformer/en-streaming/exp/pretrained.pt \
        --head emotion \
        --data-dir data/emotion/crema-d
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
_PROJECT_ROOT = Path(__file__).parent.parent.absolute()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from src.models.zipformer import load_checkpoint
from src.models.heads import (
    EmotionHead,
    EmotionConfig,
    ParalinguisticsHead,
    ParalinguisticsConfig,
)
from src.training import (
    create_emotion_loader,
    create_combined_emotion_loader,
    create_paralinguistics_loader,
    create_meld_loader,
    PARALINGUISTIC_LABELS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

CREMA_D_AND_COMBINED_EMOTION_LABELS = [
    "anger",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
]

MELD_EMOTION_LABELS = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise",
]


def create_head(head_type: str, encoder_dim: int, num_classes: int) -> nn.Module:
    """Create the appropriate head module."""
    if head_type == "emotion":
        config = EmotionConfig(
            encoder_dim=encoder_dim,
            num_classes=num_classes,
            hidden_dim=256,
            dropout_rate=0.0,  # Disable dropout for evaluation
        )
        return EmotionHead(config)
    elif head_type == "paralinguistics":
        config = ParalinguisticsConfig(
            encoder_dim=encoder_dim,
            num_classes=num_classes,
            hidden_dim=256,
            dropout_rate=0.0,  # Disable dropout for evaluation
        )
        return ParalinguisticsHead(config)
    else:
        raise ValueError(f"Unknown head type: {head_type}")


def load_head_checkpoint(head: nn.Module, weights_path: Path) -> None:
    """Load head weights from checkpoint."""
    data = np.load(str(weights_path))
    flat_dict = {k: data[k] for k in data.files}
    mlx_params = {k: mx.array(v) for k, v in flat_dict.items()}
    head.load_weights(list(mlx_params.items()))
    logger.info(f"Loaded head checkpoint from {weights_path}")


def load_encoder_weights(encoder: nn.Module, weights_path: Path) -> None:
    """Load fine-tuned encoder weights from .npz checkpoint.

    The checkpoint saves encoder stage weights with 'stage_N' prefix (e.g., stage_4.xxx)
    but the encoder model uses 'encoders.N' (e.g., encoders.4.xxx). We need to remap
    the keys to match the model structure.
    """
    data = np.load(str(weights_path))
    flat_dict = {}

    for k in data.files:
        # Remap stage_4.xxx -> encoders.4.xxx, stage_5.xxx -> encoders.5.xxx, etc.
        new_key = k
        for i in range(6):  # Zipformer has 6 stages (0-5)
            if k.startswith(f"stage_{i}."):
                new_key = k.replace(f"stage_{i}.", f"encoders.{i}.", 1)
                break
        flat_dict[new_key] = data[k]

    mlx_params = {k: mx.array(v) for k, v in flat_dict.items()}
    encoder.load_weights(list(mlx_params.items()), strict=False)
    logger.info(f"Loaded fine-tuned encoder weights from {weights_path} ({len(flat_dict)} arrays)")


def evaluate_head(
    encoder: nn.Module,
    head: nn.Module,
    dataloader,
    label_key: str,
    labels_list: List[str],
) -> Tuple[float, Dict]:
    """
    Evaluate head on dataset and compute metrics.

    Returns:
        Tuple of (accuracy, metrics_dict)
    """
    all_predictions = []
    all_labels = []
    num_batches = 0

    for batch in dataloader:
        # Forward through encoder (frozen)
        encoder_out, encoder_lengths = encoder(
            batch["features"],
            batch["feature_lengths"],
        )

        # Forward through head
        logits = head(encoder_out, encoder_lengths)

        # Get predictions
        predictions = mx.argmax(logits, axis=-1)
        labels = batch[label_key]

        mx.eval(predictions, labels)

        all_predictions.extend(predictions.tolist())
        all_labels.extend(labels.tolist())
        num_batches += 1

        if num_batches % 10 == 0:
            logger.info(f"Processed {num_batches} batches ({len(all_labels)} samples)")

    # Compute accuracy
    total = len(all_labels)
    correct = sum(p == l for p, l in zip(all_predictions, all_labels))
    accuracy = correct / total if total > 0 else 0.0

    # Compute per-class metrics
    class_correct = Counter()
    class_total = Counter()
    class_predicted = Counter()

    for pred, label in zip(all_predictions, all_labels):
        class_total[label] += 1
        class_predicted[pred] += 1
        if pred == label:
            class_correct[label] += 1

    # Compute precision, recall, F1 for each class
    class_metrics = {}
    for i, label_name in enumerate(labels_list):
        tp = class_correct[i]
        total_pred = class_predicted[i]
        total_actual = class_total[i]

        precision = tp / total_pred if total_pred > 0 else 0.0
        recall = tp / total_actual if total_actual > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        class_metrics[label_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": total_actual,
            "correct": tp,
        }

    # Compute macro-averaged metrics
    macro_precision = sum(m["precision"] for m in class_metrics.values()) / len(class_metrics)
    macro_recall = sum(m["recall"] for m in class_metrics.values()) / len(class_metrics)
    macro_f1 = sum(m["f1"] for m in class_metrics.values()) / len(class_metrics)

    # Create confusion matrix
    num_classes = len(labels_list)
    confusion = [[0] * num_classes for _ in range(num_classes)]
    for pred, label in zip(all_predictions, all_labels):
        if 0 <= pred < num_classes and 0 <= label < num_classes:
            confusion[label][pred] += 1

    return accuracy, {
        "total_samples": total,
        "correct": correct,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "class_metrics": class_metrics,
        "confusion_matrix": confusion,
    }


def print_results(accuracy: float, metrics: Dict, labels_list: List[str]) -> None:
    """Print evaluation results in a formatted way."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Total Samples: {metrics['total_samples']}")
    print(f"Correct: {metrics['correct']}")
    print("\nMacro-averaged Metrics:")
    print(f"  Precision: {metrics['macro_precision']:.4f}")
    print(f"  Recall: {metrics['macro_recall']:.4f}")
    print(f"  F1-Score: {metrics['macro_f1']:.4f}")

    print("\nPer-Class Metrics:")
    print("-" * 60)
    print(f"{'Class':<20} {'Prec':<8} {'Recall':<8} {'F1':<8} {'Support':<8}")
    print("-" * 60)
    for label_name in labels_list:
        if label_name in metrics["class_metrics"]:
            m = metrics["class_metrics"][label_name]
            print(f"{label_name:<20} {m['precision']:.4f}   {m['recall']:.4f}   {m['f1']:.4f}   {m['support']:<8}")

    print("\nConfusion Matrix:")
    print("-" * 60)
    # Header
    header = "True\\Pred".ljust(16)
    for label in labels_list:
        header += label[:6].ljust(8)
    print(header)
    print("-" * 60)

    # Rows
    for i, row_label in enumerate(labels_list):
        row = row_label[:14].ljust(16)
        for j in range(len(labels_list)):
            row += str(metrics["confusion_matrix"][i][j]).ljust(8)
        print(row)

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained rich audio heads on labeled datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to head checkpoint (.npz file)",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="checkpoints/zipformer/en-streaming/exp/pretrained.pt",
        help="Path to Zipformer encoder checkpoint",
    )
    parser.add_argument(
        "--encoder-weights",
        type=str,
        help="Path to fine-tuned encoder weights (.npz file). Overrides weights from --encoder.",
    )
    parser.add_argument(
        "--head",
        type=str,
        required=True,
        choices=["emotion", "paralinguistics"],
        help="Head type to evaluate",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Path to test data directory",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="crema-d",
        choices=["crema-d", "combined", "meld"],
        help="Dataset type for emotion head",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split to evaluate (emotion: train/val/validation/dev/test; paralinguistics: train/test)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save results JSON",
    )

    args = parser.parse_args()

    # Set default data directories
    if args.data_dir is None:
        if args.head == "paralinguistics":
            args.data_dir = "data/paralinguistics/vocalsound_labeled"
        elif args.head == "emotion":
            if args.dataset == "combined":
                args.data_dir = "data/emotion/combined_emotion_hf"
            elif args.dataset == "meld":
                args.data_dir = "data/emotion_punctuation/MELD.Raw"
            else:
                args.data_dir = "data/emotion/crema-d"

    # Choose default split if not provided
    if args.split is None:
        if args.head == "paralinguistics":
            args.split = "test"
        elif args.head == "emotion":
            if args.dataset == "combined":
                args.split = "validation"
            elif args.dataset == "meld":
                args.split = "test"
            else:
                args.split = "val"

    # Normalize split names across loaders
    if args.head == "emotion":
        if args.dataset == "combined" and args.split == "val":
            args.split = "validation"
        if args.dataset == "crema-d" and args.split == "validation":
            args.split = "val"

    # Load encoder
    logger.info(f"Loading encoder from {args.encoder}")
    model, config = load_checkpoint(args.encoder)
    encoder = model.encoder
    encoder_dim = encoder.output_dim
    logger.info(f"Encoder loaded: output_dim={encoder_dim}")

    # Load fine-tuned encoder weights if provided
    if args.encoder_weights:
        load_encoder_weights(encoder, Path(args.encoder_weights))

    # Set up head and dataloader based on type
    if args.head == "paralinguistics":
        labels_list = PARALINGUISTIC_LABELS
        label_key = "paralinguistic_labels"
        dataloader = create_paralinguistics_loader(
            args.data_dir,
            split=args.split,
            batch_size=args.batch_size,
            shuffle=False,
        )
    elif args.head == "emotion":
        label_key = "emotion_labels"
        if args.dataset == "meld":
            labels_list = MELD_EMOTION_LABELS
            dataloader = create_meld_loader(
                args.data_dir,
                split=args.split,
                batch_size=args.batch_size,
                shuffle=False,
            )
        elif args.dataset == "combined":
            labels_list = CREMA_D_AND_COMBINED_EMOTION_LABELS
            dataloader = create_combined_emotion_loader(
                args.data_dir,
                split=args.split,
                batch_size=args.batch_size,
                shuffle=False,
            )
        else:
            labels_list = CREMA_D_AND_COMBINED_EMOTION_LABELS
            dataloader = create_emotion_loader(
                args.data_dir,
                split=args.split,
                batch_size=args.batch_size,
                shuffle=False,
            )

    logger.info(f"Test set batches: {len(dataloader)}")

    # Create and load head
    head = create_head(args.head, encoder_dim, len(labels_list))
    load_head_checkpoint(head, Path(args.checkpoint))

    # Evaluate
    logger.info("Starting evaluation...")
    accuracy, metrics = evaluate_head(
        encoder, head, dataloader, label_key, labels_list
    )

    # Print results
    print_results(accuracy, metrics, labels_list)

    # Save results if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
