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
Pseudo-Label LibriSpeech Features

Uses a trained RichDecoder model to generate pseudo-labels for unlabeled
LibriSpeech encoder features. Implements Phase 3 of RICHDECODER_V4_ROADMAP.md.

Strategy:
1. Load trained V3 model (91.02% emotion accuracy)
2. Run inference on LibriSpeech features
3. Keep only high-confidence predictions (>0.9)
4. Generate manifest for combined training

Expected yield:
- LibriSpeech: 23,943 samples
- At 0.9 confidence: ~50-70% acceptance rate
- Expected pseudo-labels: ~12,000-17,000 samples

Usage:
    python -m tools.whisper_mlx.pseudo_label_librispeech \
        --checkpoint checkpoints/rich_decoder_v3_cached/best.npz \
        --features-dir data/emotion_punctuation/librispeech_features \
        --output-manifest data/pseudo_labels/librispeech_pseudo_manifest.json \
        --confidence-threshold 0.9
"""

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

from .model import WhisperMLX
from .rich_ctc_head import EMOTION_CLASSES_34
from .rich_decoder import RichDecoder, RichDecoderConfig
from .tokenizer import get_whisper_tokenizer


class PseudoLabeler:
    """
    Generate pseudo-labels for unlabeled audio features.

    Uses a trained RichDecoder to predict emotion labels with confidence scores.
    Only samples exceeding the confidence threshold are kept.
    """

    def __init__(
        self,
        checkpoint_path: str,
        whisper_model: str = "large-v3",
        confidence_threshold: float = 0.9,
        batch_size: int = 16,
    ):
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size

        print(f"Loading Whisper model: {whisper_model}")
        self.whisper = WhisperMLX.from_pretrained(whisper_model)
        self.tokenizer = get_whisper_tokenizer(whisper_model)

        # Create RichDecoder config using WhisperMLX.config
        config = RichDecoderConfig(
            n_vocab=self.whisper.config.n_vocab,
            n_ctx=self.whisper.config.n_text_ctx,
            n_state=self.whisper.config.n_text_state,
            n_head=self.whisper.config.n_text_head,
            n_layer=self.whisper.config.n_text_layer,
            lora_rank=32,  # Match V3 training
            lora_alpha=64,
            num_emotions=34,
        )

        # Create RichDecoder from Whisper decoder
        print("Creating RichDecoder from Whisper decoder...")
        self.rich_decoder = RichDecoder.from_whisper_decoder(
            self.whisper.decoder,
            config=config,
        )

        # Load trained weights
        print(f"Loading checkpoint: {checkpoint_path}")
        self.rich_decoder.load_trainable(checkpoint_path)
        mx.eval(self.rich_decoder.parameters())

        # SOT token for decoder input
        self.sot_token = self.tokenizer.sot

        # Emotion class names
        self.emotion_classes = EMOTION_CLASSES_34

        print(f"Model loaded. Confidence threshold: {confidence_threshold}")

    def predict_emotion(
        self,
        encoder_features: mx.array,
    ) -> tuple[int, float, np.ndarray]:
        """
        Predict emotion from encoder features.

        Args:
            encoder_features: Shape (T, 1280) encoder output

        Returns:
            Tuple of (predicted_class, confidence, all_probs)
        """
        # Add batch dimension
        encoder_out = encoder_features[None, ...]  # (1, T, 1280)

        # Create decoder input (SOT token)
        token_ids = mx.array([[self.sot_token]])  # (1, 1)

        # Forward pass
        outputs = self.rich_decoder(
            x=token_ids,
            xa=encoder_out,
        )

        # Get emotion prediction
        emotion_logits = outputs["emotion"]  # (1, 1, 34)
        emotion_pooled = mx.mean(emotion_logits, axis=1)  # (1, 34)

        # Softmax to get probabilities
        probs = mx.softmax(emotion_pooled, axis=-1)  # (1, 34)
        probs = np.array(probs[0])  # (34,)

        # Get prediction and confidence
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])

        return pred_class, confidence, probs

    def predict_batch(
        self,
        encoder_features_list: list[mx.array],
    ) -> list[tuple[int, float, np.ndarray]]:
        """
        Predict emotions for a batch of encoder features.

        Args:
            encoder_features_list: List of (T, 1280) encoder outputs

        Returns:
            List of (predicted_class, confidence, all_probs) tuples
        """
        if not encoder_features_list:
            return []

        # Pad to same length
        max_len = max(f.shape[0] for f in encoder_features_list)
        padded = []
        for features in encoder_features_list:
            if features.shape[0] < max_len:
                padding = mx.zeros((max_len - features.shape[0], features.shape[1]))
                features = mx.concatenate([features, padding], axis=0)
            padded.append(features)

        # Stack into batch
        encoder_out = mx.stack(padded)  # (B, T, 1280)

        # Create decoder input (SOT token for each)
        batch_size = len(encoder_features_list)
        token_ids = mx.array([[self.sot_token]] * batch_size)  # (B, 1)

        # Forward pass
        outputs = self.rich_decoder(
            x=token_ids,
            xa=encoder_out,
        )

        # Get emotion predictions
        emotion_logits = outputs["emotion"]  # (B, 1, 34)
        emotion_pooled = mx.mean(emotion_logits, axis=1)  # (B, 34)

        # Softmax to get probabilities
        probs = mx.softmax(emotion_pooled, axis=-1)  # (B, 34)
        probs_np = np.array(probs)  # (B, 34)

        # Get predictions and confidences
        results = []
        for i in range(batch_size):
            pred_class = int(np.argmax(probs_np[i]))
            confidence = float(probs_np[i, pred_class])
            results.append((pred_class, confidence, probs_np[i]))

        return results

    def process_features_dir(
        self,
        features_dir: str,
        output_manifest: str,
        limit: int | None = None,
    ) -> dict:
        """
        Process all feature files in a directory.

        Args:
            features_dir: Directory with .npz feature files
            output_manifest: Path to save pseudo-label manifest
            limit: Optional limit on number of files to process

        Returns:
            Statistics dictionary
        """
        features_path = Path(features_dir)
        npz_files = sorted(features_path.glob("*.npz"))

        if limit:
            npz_files = npz_files[:limit]

        total = len(npz_files)
        print(f"Processing {total} feature files...")

        pseudo_labels = []
        stats = {
            "total_processed": 0,
            "accepted": 0,
            "rejected": 0,
            "emotion_distribution": {},
            "confidence_histogram": {
                "0.5-0.6": 0,
                "0.6-0.7": 0,
                "0.7-0.8": 0,
                "0.8-0.9": 0,
                "0.9-1.0": 0,
            },
        }

        # Process in batches
        batch_features = []
        batch_files = []
        batch_metadata = []

        for i, npz_file in enumerate(npz_files):
            if (i + 1) % 1000 == 0:
                accept_rate = stats["accepted"] / max(stats["total_processed"], 1) * 100
                print(f"  Processed {i + 1}/{total} ({accept_rate:.1f}% accepted)")

            try:
                data = np.load(npz_file)
                encoder_features = mx.array(data["encoder_features"])

                batch_features.append(encoder_features)
                batch_files.append(npz_file)
                batch_metadata.append({
                    "transcript": str(data.get("transcript", "")),
                    "duration_s": float(data.get("duration_s", 0)),
                    "utterance_id": str(data.get("utterance_id", "")),
                    "speaker": str(data.get("speaker", "")),
                })

                # Process batch when full
                if len(batch_features) >= self.batch_size:
                    self._process_batch(
                        batch_features, batch_files, batch_metadata,
                        pseudo_labels, stats,
                    )
                    batch_features = []
                    batch_files = []
                    batch_metadata = []

            except Exception as e:
                print(f"  Error processing {npz_file.name}: {e}")
                continue

        # Process remaining
        if batch_features:
            self._process_batch(
                batch_features, batch_files, batch_metadata,
                pseudo_labels, stats,
            )

        # Save manifest
        output_path = Path(output_manifest)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(pseudo_labels, f, indent=2)

        # Print statistics
        print("\nPseudo-labeling complete:")
        print(f"  Total processed: {stats['total_processed']}")
        print(f"  Accepted (>={self.confidence_threshold}): {stats['accepted']}")
        print(f"  Rejected: {stats['rejected']}")
        print(f"  Acceptance rate: {stats['accepted'] / max(stats['total_processed'], 1) * 100:.1f}%")
        print("\nConfidence distribution:")
        for bucket, count in stats["confidence_histogram"].items():
            print(f"  {bucket}: {count}")
        print("\nEmotion distribution:")
        sorted_emotions = sorted(
            stats["emotion_distribution"].items(),
            key=lambda x: x[1],
            reverse=True,
        )
        for emotion, count in sorted_emotions[:10]:
            print(f"  {emotion}: {count}")
        print(f"\nSaved {len(pseudo_labels)} pseudo-labels to: {output_manifest}")

        return stats

    def _process_batch(
        self,
        batch_features: list[mx.array],
        batch_files: list[Path],
        batch_metadata: list[dict],
        pseudo_labels: list[dict],
        stats: dict,
    ):
        """Process a batch of features and update results."""
        results = self.predict_batch(batch_features)

        for (pred_class, confidence, _probs), npz_file, metadata in zip(
            results, batch_files, batch_metadata, strict=False,
        ):
            stats["total_processed"] += 1

            # Update confidence histogram
            if 0.5 <= confidence < 0.6:
                stats["confidence_histogram"]["0.5-0.6"] += 1
            elif 0.6 <= confidence < 0.7:
                stats["confidence_histogram"]["0.6-0.7"] += 1
            elif 0.7 <= confidence < 0.8:
                stats["confidence_histogram"]["0.7-0.8"] += 1
            elif 0.8 <= confidence < 0.9:
                stats["confidence_histogram"]["0.8-0.9"] += 1
            else:
                stats["confidence_histogram"]["0.9-1.0"] += 1

            # Filter by confidence
            if confidence >= self.confidence_threshold:
                stats["accepted"] += 1
                emotion_name = self.emotion_classes[pred_class]

                # Update emotion distribution
                stats["emotion_distribution"][emotion_name] = (
                    stats["emotion_distribution"].get(emotion_name, 0) + 1
                )

                # Add to pseudo-labels
                pseudo_labels.append({
                    "path": str(npz_file),
                    "emotion_id": pred_class,
                    "emotion": emotion_name,
                    "confidence": round(confidence, 4),
                    "source": "pseudo_librispeech",
                    **metadata,
                })
            else:
                stats["rejected"] += 1


def main():
    parser = argparse.ArgumentParser(
        description="Generate pseudo-labels for LibriSpeech features",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained RichDecoder checkpoint",
    )
    parser.add_argument(
        "--features-dir",
        type=str,
        required=True,
        help="Directory with .npz encoder feature files",
    )
    parser.add_argument(
        "--output-manifest",
        type=str,
        required=True,
        help="Path to save pseudo-label manifest JSON",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="large-v3",
        help="Whisper model name (default: large-v3)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.9,
        help="Minimum confidence to accept pseudo-label (default: 0.9)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for inference (default: 16)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of files to process (for testing)",
    )

    args = parser.parse_args()

    # Create labeler
    labeler = PseudoLabeler(
        checkpoint_path=args.checkpoint,
        whisper_model=args.whisper_model,
        confidence_threshold=args.confidence_threshold,
        batch_size=args.batch_size,
    )

    # Process features
    labeler.process_features_dir(
        features_dir=args.features_dir,
        output_manifest=args.output_manifest,
        limit=args.limit,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
