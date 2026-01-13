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
Pseudo-label LibriSpeech features using trained RichDecoder v3 model.

This script generates emotion pseudo-labels for unlabeled LibriSpeech
encoder features using a trained model. High-confidence predictions
are kept and can be combined with labeled data for semi-supervised learning.

IMPORTANT: Uses full decoder inference (cross-attention over encoder features)
to match the V3 training architecture. The emotion head is trained on decoder
hidden states, not raw encoder features.

Usage:
    python scripts/pseudo_label_librispeech.py \
        --checkpoint checkpoints/rich_decoder_v3_cached/best.npz \
        --features-dir data/emotion_punctuation/librispeech_features \
        --output data/pseudo_labels/librispeech_manifest.json \
        --confidence-threshold 0.9

Expected yield at 0.9 threshold: ~12,000-17,000 samples from 23,943 total
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.whisper_mlx.rich_decoder import RichDecoder, RichDecoderConfig
from tools.whisper_mlx.model import WhisperMLX


# Emotion class names (34 classes used in V3)
EMOTION_CLASSES_34 = [
    "neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised",
    "amused", "bored", "confused", "contempt", "desire", "disappointment",
    "embarrassment", "excited", "interested", "pain", "proud", "relieved",
    "satisfied", "sympathetic", "tired", "default", "enunciated", "laughing",
    "nonverbal", "projected", "singing", "sleepy", "whisper",
    "confused_voice", "concerned", "thoughtful"
]

# 8-class simplified version (V3 uses this)
EMOTION_CLASSES_8 = [
    "neutral", "angry", "disgust", "fear", "happy", "sad", "surprise", "other"
]


class FullRichDecoderInference:
    """
    Full RichDecoder inference using proper decoder cross-attention.

    This matches the V3 training architecture:
    1. Encoder features -> Decoder cross-attention -> Decoder hidden states
    2. Decoder hidden states -> Emotion head -> Predictions

    The emotion head is trained on decoder hidden states, not raw encoder features.
    Using encoder features directly gives incorrect predictions.
    """

    def __init__(
        self,
        whisper_model: WhisperMLX,
        rich_decoder: RichDecoder,
        sot_token: int,
        num_emotions: int = 8,
    ):
        self.whisper = whisper_model
        self.rich_decoder = rich_decoder
        self.sot_token = sot_token
        self.num_emotions = num_emotions

    def __call__(self, encoder_features: mx.array) -> mx.array:
        """
        Predict emotion using full decoder inference.

        Args:
            encoder_features: (T, 1280) encoder output

        Returns:
            emotion_probs: (num_emotions,) probability distribution
        """
        # Add batch dimension
        xa = encoder_features[None, ...]  # (1, T, 1280)

        # Create SOT token input (V3 training uses just SOT)
        x = mx.array([[self.sot_token]])  # (1, 1)

        # Run full decoder forward pass
        outputs = self.rich_decoder(x=x, xa=xa)

        # Get emotion logits and pool over sequence
        emotion_logits = outputs["emotion"]  # (1, seq_len, num_emotions)
        emotion_pooled = mx.mean(emotion_logits, axis=1)  # (1, num_emotions)

        # Softmax for probabilities
        probs = mx.softmax(emotion_pooled, axis=-1)
        return probs.squeeze(0)  # (num_emotions,)


def load_full_model(
    checkpoint_path: str,
    whisper_model_name: str = "mlx-community/whisper-large-v3-mlx",
    num_emotions: int = 8,
    lora_rank: int = 32,
    lora_alpha: int = 64,
) -> Tuple[FullRichDecoderInference, int]:
    """
    Load full RichDecoder with Whisper base model and trained LoRA weights.

    Args:
        checkpoint_path: Path to trained LoRA checkpoint
        whisper_model_name: Hugging Face model ID for base Whisper
        num_emotions: Number of emotion classes (8 for V3)
        lora_rank: LoRA rank (32 for V3)
        lora_alpha: LoRA alpha (64 for V3)

    Returns:
        Tuple of (model, num_weights_loaded)
    """
    # Load base Whisper model
    print(f"Loading {whisper_model_name}...")
    whisper = WhisperMLX.from_pretrained(whisper_model_name)

    # Create RichDecoder config matching V3 training
    # Note: lora_rank in checkpoint is 8 (V3 default), not 32
    config = RichDecoderConfig(
        n_vocab=51865,
        n_ctx=448,
        n_state=1280,
        n_head=20,
        n_layer=32,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        num_emotions=num_emotions,
    )

    # Create RichDecoder wrapping Whisper decoder
    rich_decoder = RichDecoder.from_whisper_decoder(whisper.decoder, config)

    # Load trained weights using RichDecoder's built-in method
    # This correctly parses checkpoint format: "lora.layer_X_Y.A.weight", "rich_heads.X.weight"
    print(f"Loading checkpoint {checkpoint_path}...")
    weights = dict(mx.load(checkpoint_path))
    num_weights = len(weights)

    # Use RichDecoder's load_trainable method which knows the checkpoint format
    rich_decoder.load_trainable(checkpoint_path)

    print(f"  Checkpoint contained {num_weights} weight tensors")

    # Get SOT token from tokenizer
    from tools.whisper_mlx.tokenizer import get_whisper_tokenizer
    tokenizer = get_whisper_tokenizer()
    sot_token = tokenizer.sot

    # Create inference wrapper
    model = FullRichDecoderInference(
        whisper_model=whisper,
        rich_decoder=rich_decoder,
        sot_token=sot_token,
        num_emotions=num_emotions,
    )

    return model, num_weights


def pseudo_label_features(
    model: FullRichDecoderInference,
    features_dir: Path,
    confidence_threshold: float = 0.9,
    max_samples: Optional[int] = None,
) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Generate pseudo-labels for all feature files.

    Args:
        model: Trained model for inference
        features_dir: Directory containing .npz encoder features
        confidence_threshold: Minimum confidence to keep prediction
        max_samples: Optional limit on number of samples to process

    Returns:
        Tuple of (pseudo_labels list, statistics dict)
    """
    feature_files = sorted(features_dir.glob("*.npz"))
    if max_samples:
        feature_files = feature_files[:max_samples]

    pseudo_labels = []
    # Use appropriate class list based on model config
    emotion_classes = EMOTION_CLASSES_34 if model.num_emotions == 34 else EMOTION_CLASSES_8

    stats = {
        "total": 0,
        "accepted": 0,
        "rejected": 0,
        "errors": 0,
        "per_class": {cls: 0 for cls in emotion_classes},
    }

    for npz_path in tqdm(feature_files, desc="Pseudo-labeling"):
        stats["total"] += 1

        try:
            # Load encoder features
            data = np.load(npz_path, allow_pickle=True)
            encoder_features = mx.array(data["encoder_features"])

            # Get metadata if available
            transcript = str(data["transcript"]) if "transcript" in data else ""
            speaker = str(data["speaker"]) if "speaker" in data else ""
            duration_s = float(data["duration_s"]) if "duration_s" in data else 0.0
            utterance_id = str(data["utterance_id"]) if "utterance_id" in data else npz_path.stem

            # Run inference
            probs = model(encoder_features)
            mx.eval(probs)  # Ensure computation completes

            # Get prediction and confidence
            probs_np = np.array(probs)
            pred_class = int(np.argmax(probs_np))
            confidence = float(probs_np[pred_class])

            # Filter by confidence
            if confidence >= confidence_threshold:
                emotion_label = emotion_classes[pred_class] if pred_class < len(emotion_classes) else f"unknown_{pred_class}"
                pseudo_labels.append({
                    "path": str(npz_path),
                    "utterance_id": utterance_id,
                    "emotion_id": pred_class,
                    "emotion_label": emotion_label,
                    "confidence": round(confidence, 4),
                    "transcript": transcript,
                    "speaker": speaker,
                    "duration_s": round(duration_s, 2),
                    "source": "pseudo_librispeech",
                })
                stats["accepted"] += 1
                if emotion_label in stats["per_class"]:
                    stats["per_class"][emotion_label] += 1
            else:
                stats["rejected"] += 1

        except Exception as e:
            stats["errors"] += 1
            if stats["errors"] <= 10:
                print(f"Error processing {npz_path.name}: {e}")

    return pseudo_labels, stats


def analyze_distribution(pseudo_labels: List[Dict], emotion_classes: List[str] = None) -> Dict[str, float]:
    """Analyze class distribution of pseudo-labels."""
    if emotion_classes is None:
        emotion_classes = EMOTION_CLASSES_34

    counts = {cls: 0 for cls in emotion_classes}

    for label in pseudo_labels:
        if label["emotion_label"] in counts:
            counts[label["emotion_label"]] += 1

    total = len(pseudo_labels) if pseudo_labels else 1
    distribution = {cls: count / total * 100 for cls, count in counts.items()}

    return distribution


def main():
    parser = argparse.ArgumentParser(
        description="Generate pseudo-labels for LibriSpeech features"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/rich_decoder_v3_cached/best.npz",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--features-dir",
        type=str,
        default="data/emotion_punctuation/librispeech_features",
        help="Directory containing encoder feature .npz files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/pseudo_labels/librispeech_manifest.json",
        help="Output manifest file path",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.9,
        help="Minimum confidence to accept pseudo-label (default: 0.9)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to process (for testing)",
    )
    parser.add_argument(
        "--num-emotions",
        type=int,
        default=34,
        help="Number of emotion classes (default: 34 to match V3 checkpoint)",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=8,
        help="LoRA rank (default: 8 for V3 checkpoint)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha (default: 16 for V3 checkpoint)",
    )

    args = parser.parse_args()

    # Validate paths
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    features_dir = Path(args.features_dir)
    if not features_dir.exists():
        print(f"Error: Features directory not found: {features_dir}")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Count available features
    num_features = len(list(features_dir.glob("*.npz")))
    print("\nPseudo-labeling Configuration:")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Features directory: {features_dir}")
    print(f"  Total feature files: {num_features}")
    print(f"  Confidence threshold: {args.confidence_threshold}")
    print(f"  Output: {output_path}")

    # Load full model (Whisper + RichDecoder with LoRA)
    print("\nLoading model...")
    model, loaded = load_full_model(
        checkpoint_path=str(checkpoint_path),
        num_emotions=args.num_emotions,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )

    if loaded == 0:
        print("\nWarning: No weights loaded! Check checkpoint format.")
        print("Available keys in checkpoint:")
        weights = dict(mx.load(str(checkpoint_path)))
        for key in sorted(weights.keys())[:20]:
            print(f"  {key}: {weights[key].shape}")
        if len(weights) > 20:
            print(f"  ... and {len(weights) - 20} more")
        sys.exit(1)

    # Run pseudo-labeling
    print("\nGenerating pseudo-labels...")
    pseudo_labels, stats = pseudo_label_features(
        model=model,
        features_dir=features_dir,
        confidence_threshold=args.confidence_threshold,
        max_samples=args.max_samples,
    )

    # Print statistics
    print("\nResults:")
    print(f"  Total processed: {stats['total']}")
    print(f"  Accepted (conf >= {args.confidence_threshold}): {stats['accepted']}")
    print(f"  Rejected (low confidence): {stats['rejected']}")
    print(f"  Errors: {stats['errors']}")
    print(f"  Acceptance rate: {stats['accepted'] / max(stats['total'], 1) * 100:.1f}%")

    # Class distribution
    emotion_classes = EMOTION_CLASSES_34 if args.num_emotions == 34 else EMOTION_CLASSES_8
    print("\nClass distribution:")
    distribution = analyze_distribution(pseudo_labels, emotion_classes)
    for cls, pct in sorted(distribution.items(), key=lambda x: -x[1]):
        if pct > 0:  # Only show classes with samples
            count = stats["per_class"].get(cls, 0)
            print(f"  {cls}: {count} ({pct:.1f}%)")

    # Save manifest
    print(f"\nSaving {len(pseudo_labels)} pseudo-labels to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(pseudo_labels, f, indent=2)

    # Also save statistics
    stats_path = output_path.parent / f"{output_path.stem}_stats.json"
    with open(stats_path, "w") as f:
        json.dump({
            "total_processed": stats["total"],
            "accepted": stats["accepted"],
            "rejected": stats["rejected"],
            "errors": stats["errors"],
            "acceptance_rate": stats["accepted"] / max(stats["total"], 1),
            "confidence_threshold": args.confidence_threshold,
            "checkpoint": str(checkpoint_path),
            "class_distribution": distribution,
            "per_class_counts": stats["per_class"],
        }, f, indent=2)
    print(f"Saved statistics to {stats_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
