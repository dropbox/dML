#!/usr/bin/env python3
"""
Knowledge Distillation Training Script

Trains Whisper-based heads using both hard labels and soft labels from SOTA models.
This enables the student heads to learn from SOTA model knowledge, improving accuracy
beyond what hard labels alone can achieve.

Loss function:
    L = α * CE(pred, hard_label) + (1-α) * KL(pred, soft_label) * T²

Where:
- α: Interpolation weight (default 0.5)
- T: Temperature for softening distributions (default 2.0)
- CE: Cross-entropy loss with hard labels
- KL: KL divergence with soft labels

Usage:
    # Train with soft labels from wav2vec2-xlsr-ser
    python scripts/train_with_distillation.py \
        --manifest data/v4_expanded/train_manifest.json \
        --soft-labels data/soft_labels \
        --teacher wav2vec2-xlsr-ser \
        --task emotion

    # Test run with smaller dataset
    python scripts/train_with_distillation.py --limit 1000 --epochs 2

References:
    - Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)
    - https://arxiv.org/abs/1503.02531
"""

import argparse
import gc
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from tools.whisper_mlx.multi_head import (
    EmotionHead,
    ParalinguisticsHead,
    MultiHeadConfig,
    PARALINGUISTICS_CLASSES,
)
from tools.whisper_mlx.rich_ctc_head import EMOTION_CLASSES_8 as EMOTION_CLASSES
from tools.whisper_mlx.encoder_cache import TrainingEncoderCache


@dataclass
class DistillationConfig:
    """Configuration for distillation training."""

    # Data
    manifest_path: str = "data/v4_expanded/train_manifest.json"
    val_manifest_path: Optional[str] = None
    soft_labels_dir: str = "data/soft_labels"
    encoder_cache_dir: str = "data/v4_expanded/encoder_cache"

    # Distillation
    teacher_model: str = "wav2vec2-xlsr-ser"  # SOTA model for soft labels
    task: str = "emotion"  # emotion or paralinguistics
    alpha: float = 0.5  # Weight for hard label loss (1-alpha for soft)
    temperature: float = 2.0  # Temperature for softening

    # Model
    d_model: int = 1280  # Whisper large-v3 encoder dim

    # Training
    epochs: int = 5
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500

    # Output
    output_dir: str = "checkpoints/distillation"
    log_interval: int = 50
    save_interval: int = 500
    eval_interval: int = 500


def log_softmax(x: mx.array, axis: int = -1) -> mx.array:
    """Log-softmax function."""
    # For numerical stability: log_softmax(x) = x - log(sum(exp(x)))
    max_x = mx.max(x, axis=axis, keepdims=True)
    x_shifted = x - max_x
    return x_shifted - mx.log(mx.sum(mx.exp(x_shifted), axis=axis, keepdims=True))


def kl_divergence_loss(
    student_logits: mx.array,
    teacher_probs: mx.array,
    temperature: float = 2.0,
) -> mx.array:
    """
    Compute KL divergence loss for knowledge distillation.

    Args:
        student_logits: Raw logits from student model [batch, num_classes]
        teacher_probs: Soft probabilities from teacher [batch, num_classes]
            NOTE: teacher_probs should already be softmaxed at temperature T
            during soft label generation (generate_soft_labels.py)
        temperature: Temperature for softening distributions

    Returns:
        KL divergence loss scaled by T²
    """
    # Soften student logits with temperature
    student_log_probs = log_softmax(student_logits / temperature, axis=-1)

    # Teacher probs are already softmaxed at temperature T during soft label generation
    # DO NOT apply temperature again - this was a bug (double temperature application)
    # Just use them directly
    teacher_probs_T = teacher_probs

    # KL(teacher || student) = sum(teacher * log(teacher/student))
    kl = mx.sum(teacher_probs_T * (mx.log(teacher_probs_T + 1e-10) - student_log_probs), axis=-1)

    # Scale by T² as per Hinton et al.
    return mx.mean(kl) * (temperature ** 2)


def cross_entropy_loss(logits: mx.array, labels: mx.array) -> mx.array:
    """Standard cross-entropy loss."""
    log_probs = log_softmax(logits, axis=-1)
    # Gather the log probs at the label indices
    batch_indices = mx.arange(len(labels))
    return -mx.mean(log_probs[batch_indices, labels])


@dataclass
class DistillationSample:
    """A training sample with hard label and optional soft label.

    NOTE: encoder_output is loaded lazily via get_encoder_output() to avoid
    loading all 84K+ encoder outputs into memory at once (ISSUE-002 fix).
    """
    audio_path: str
    cache_key: str = ""  # Key to load encoder output on demand
    hard_label: int = 0  # Ground truth class ID
    soft_probs: Optional[np.ndarray] = None  # Teacher model probabilities (small, kept in memory)
    has_soft_label: bool = False

    def get_encoder_output(self, encoder_cache: 'TrainingEncoderCache') -> Optional[mx.array]:
        """Load encoder output on demand from cache."""
        result = encoder_cache.load(self.audio_path)
        if result is not None:
            return result[0]  # (encoder_output, actual_frames) -> encoder_output
        return None


class DistillationTrainer:
    """Trainer for knowledge distillation."""

    def __init__(self, config: DistillationConfig):
        self.config = config
        self.step = 0
        self.best_val_acc = 0.0

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

        # Initialize model head based on task
        if config.task == "emotion":
            head_config = MultiHeadConfig(
                d_model=config.d_model,
                num_emotions=len(EMOTION_CLASSES),
                emotion_hidden_dim=512,
            )
            self.head = EmotionHead(head_config)
            self.num_classes = len(EMOTION_CLASSES)
            self.class_names = EMOTION_CLASSES
        else:
            head_config = MultiHeadConfig(
                d_model=config.d_model,
                num_paralinguistic_classes=len(PARALINGUISTICS_CLASSES),
                paralinguistic_hidden_dim=512,
            )
            self.head = ParalinguisticsHead(head_config)
            self.num_classes = len(PARALINGUISTICS_CLASSES)
            self.class_names = PARALINGUISTICS_CLASSES

        # Initialize optimizer
        self.optimizer = optim.AdamW(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Load encoder cache
        self.encoder_cache = TrainingEncoderCache(config.encoder_cache_dir)
        stats = self.encoder_cache.get_stats()
        print(f"Loaded encoder cache: {stats.get('cached_files', 0)} entries")

        # Soft label index
        self.soft_label_index = self._load_soft_label_index()

    def _load_soft_label_index(self) -> Dict[str, str]:
        """Load index of available soft labels."""
        index_path = Path(self.config.soft_labels_dir) / "index.json"
        if index_path.exists():
            with open(index_path, 'r') as f:
                data = json.load(f)
                # Index could be a list or dict
                # Check if it's a valid file-path mapping (not stats from generate_soft_labels.py)
                if isinstance(data, dict):
                    # Skip if this looks like stats file (has 'total_processed', 'errors', etc.)
                    stats_keys = {'total_processed', 'total_errors', 'errors'}
                    if not stats_keys.intersection(data.keys()):
                        return data
                elif isinstance(data, list):
                    return {item: str(Path(self.config.soft_labels_dir) / f"{item}.json") for item in data}

        # Build index from individual files
        index = {}
        soft_dir = Path(self.config.soft_labels_dir)
        if soft_dir.exists():
            for json_file in soft_dir.glob("*.json"):
                if json_file.name == "index.json":
                    continue
                index[json_file.stem] = str(json_file)

        return index

    def _get_soft_label(self, audio_path: str) -> Optional[np.ndarray]:
        """Get soft label probabilities for an audio file."""
        stem = Path(audio_path).stem

        if stem in self.soft_label_index:
            try:
                with open(self.soft_label_index[stem], 'r') as f:
                    data = json.load(f)

                soft_labels = data.get("soft_labels", {})
                teacher_data = soft_labels.get(self.config.teacher_model, {})

                if "probabilities" in teacher_data:
                    return np.array(teacher_data["probabilities"], dtype=np.float32)
            except Exception:
                pass

        return None

    def load_samples(self, manifest_path: str, limit: Optional[int] = None) -> List[DistillationSample]:
        """Load training samples from manifest.

        NOTE: Encoder outputs are NOT loaded here to avoid memory exhaustion with
        large datasets (84K+ samples). They are loaded lazily in _prepare_batch.
        """
        import hashlib

        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        if limit:
            manifest = manifest[:limit]

        samples = []
        soft_count = 0
        cache_miss_count = 0

        for entry in manifest:
            audio_path = entry.get("audio_path", "")

            # Get hard label
            if self.config.task == "emotion":
                label_str = entry.get("emotion", "neutral")
                # Map common label variants
                label_map = {
                    'fear': 'fearful',
                    'surprise': 'surprised',
                }
                label_str = label_map.get(label_str, label_str)
                if label_str in EMOTION_CLASSES:
                    hard_label = EMOTION_CLASSES.index(label_str)
                else:
                    continue
            else:
                label_str = entry.get("para", "speech")
                if label_str in PARALINGUISTICS_CLASSES:
                    hard_label = PARALINGUISTICS_CLASSES.index(label_str)
                else:
                    continue

            # Generate cache key for lazy loading (don't load encoder output here)
            cache_key = hashlib.sha256(audio_path.encode()).hexdigest()[:16]

            # Check if cache entry exists (quick path check, no memory load)
            if not self.encoder_cache.has(audio_path):
                cache_miss_count += 1
                continue

            # Get soft label (these are small, OK to keep in memory)
            soft_probs = self._get_soft_label(audio_path)
            has_soft = soft_probs is not None
            if has_soft:
                soft_count += 1

            samples.append(DistillationSample(
                audio_path=audio_path,
                cache_key=cache_key,
                hard_label=hard_label,
                soft_probs=soft_probs,
                has_soft_label=has_soft,
            ))

        print(f"Loaded {len(samples)} samples ({soft_count} with soft labels)")
        if cache_miss_count > 0:
            print(f"  Skipped {cache_miss_count} samples (not in encoder cache)")
        return samples

    def _prepare_batch(
        self,
        samples: List[DistillationSample]
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
        """Prepare a batch for training.

        Loads encoder outputs lazily to avoid memory exhaustion (ISSUE-002 fix).
        """
        encoder_outputs = []
        hard_labels = []
        soft_probs_list = []
        has_soft = []
        valid_samples = []

        teacher_classes = 8  # wav2vec2-xlsr-ser has 8 emotion classes

        for sample in samples:
            # Load encoder output on demand (lazy loading)
            encoder_output = sample.get_encoder_output(self.encoder_cache)
            if encoder_output is None:
                continue  # Skip if cache load failed

            encoder_outputs.append(encoder_output)
            hard_labels.append(sample.hard_label)
            has_soft.append(sample.has_soft_label)
            valid_samples.append(sample)

            if sample.has_soft_label:
                soft_probs_list.append(sample.soft_probs)
            else:
                soft_probs_list.append(np.zeros(teacher_classes, dtype=np.float32))

        if not encoder_outputs:
            # Return empty batch if all samples failed to load
            return mx.zeros((0, 1, 1280)), mx.zeros((0,), dtype=mx.int32), mx.zeros((0, teacher_classes)), mx.zeros((0,))

        # Pad encoder outputs to same length
        max_frames = max(e.shape[0] for e in encoder_outputs)
        padded = []
        for e in encoder_outputs:
            if e.shape[0] < max_frames:
                pad = mx.zeros((max_frames - e.shape[0], e.shape[1]))
                e = mx.concatenate([e, pad], axis=0)
            padded.append(e)

        encoder_batch = mx.stack(padded)
        hard_labels_arr = mx.array(hard_labels)
        soft_probs_arr = mx.array(np.stack(soft_probs_list))
        has_soft_arr = mx.array(has_soft)

        return encoder_batch, hard_labels_arr, soft_probs_arr, has_soft_arr

    def train_step(self, samples: List[DistillationSample]) -> Tuple[float, float, float]:
        """Single training step with distillation loss."""
        encoder_outputs, hard_labels, soft_probs, has_soft_mask = self._prepare_batch(samples)

        config = self.config

        # Track individual losses separately for logging
        hard_loss_val = 0.0
        soft_loss_val = 0.0

        def loss_fn(head):
            nonlocal hard_loss_val, soft_loss_val

            logits = head(encoder_outputs, return_frame_logits=False)

            # Hard label loss (CE)
            hard_loss = cross_entropy_loss(logits, hard_labels)

            # Soft label loss (KL) - only for samples with soft labels
            num_with_soft = mx.sum(has_soft_mask.astype(mx.float32))
            if float(num_with_soft) > 0:
                soft_loss = kl_divergence_loss(
                    logits,
                    soft_probs,
                    temperature=config.temperature,
                )
                soft_loss = soft_loss * (num_with_soft / len(samples))
            else:
                soft_loss = mx.array(0.0)

            total_loss = config.alpha * hard_loss + (1 - config.alpha) * soft_loss

            # Store for logging
            mx.eval(hard_loss, soft_loss)
            hard_loss_val = float(hard_loss)
            soft_loss_val = float(soft_loss)

            return total_loss

        total_loss, grads = nn.value_and_grad(self.head, loss_fn)(self.head)

        self.optimizer.update(self.head, grads)
        mx.eval(self.head.parameters())

        return float(total_loss), hard_loss_val, soft_loss_val

    def evaluate(self, samples: List[DistillationSample]) -> Tuple[float, float]:
        """Evaluate on validation set."""
        total_loss = 0.0
        correct = 0
        total = 0

        batch_size = self.config.batch_size

        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            encoder_outputs, hard_labels, _, _ = self._prepare_batch(batch)

            logits = self.head(encoder_outputs, return_frame_logits=False)

            loss = cross_entropy_loss(logits, hard_labels)
            total_loss += float(loss) * len(batch)

            preds = mx.argmax(logits, axis=-1)
            correct += int(mx.sum(preds == hard_labels))
            total += len(batch)

        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        return avg_loss, accuracy

    def train(
        self,
        train_samples: List[DistillationSample],
        val_samples: Optional[List[DistillationSample]] = None,
    ):
        """Main training loop."""
        config = self.config

        print(f"\n{'='*60}")
        print("Knowledge Distillation Training")
        print(f"{'='*60}")
        print(f"  Task: {config.task}")
        print(f"  Teacher: {config.teacher_model}")
        print(f"  Alpha (hard weight): {config.alpha}")
        print(f"  Temperature: {config.temperature}")
        print(f"  Train samples: {len(train_samples)}")
        if val_samples:
            print(f"  Val samples: {len(val_samples)}")
        print(f"  Epochs: {config.epochs}")
        print(f"  Batch size: {config.batch_size}")
        print()

        log_path = Path(config.output_dir) / "training.log"
        log_file = open(log_path, 'a')

        for epoch in range(config.epochs):
            print(f"Epoch {epoch + 1}/{config.epochs}")

            np.random.shuffle(train_samples)
            epoch_losses = []

            for i in range(0, len(train_samples), config.batch_size):
                self.step += 1
                batch = train_samples[i:i + config.batch_size]

                total_loss, hard_loss, soft_loss = self.train_step(batch)
                epoch_losses.append(total_loss)

                if self.step % config.log_interval == 0:
                    msg = f"  Step {self.step}: loss={total_loss:.4f} (hard={hard_loss:.4f}, soft={soft_loss:.4f})"
                    print(msg)
                    log_file.write(f"{msg}\n")
                    log_file.flush()

                    gc.collect()
                    mx.clear_cache()

                if self.step % config.save_interval == 0:
                    self._save_checkpoint(f"step_{self.step}.npz")

                if val_samples and self.step % config.eval_interval == 0:
                    val_loss, val_acc = self.evaluate(val_samples)
                    msg = f"  Validation: loss={val_loss:.4f}, acc={val_acc:.2%}"
                    print(msg)
                    log_file.write(f"{msg}\n")

                    if val_acc > self.best_val_acc:
                        self.best_val_acc = val_acc
                        self._save_checkpoint("best.npz")
                        print(f"  New best model! acc={val_acc:.2%}")

                    gc.collect()
                    mx.clear_cache()

            avg_loss = np.mean(epoch_losses)
            print(f"  Epoch {epoch + 1} complete. Avg loss: {avg_loss:.4f}")

        self._save_checkpoint("final.npz")
        print(f"\nTraining complete. Best val accuracy: {self.best_val_acc:.2%}")
        log_file.close()

    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        from mlx.utils import tree_flatten

        path = Path(self.config.output_dir) / filename
        # Flatten nested parameters for saving
        flat_params = {}
        for key, value in tree_flatten(self.head.parameters()):
            flat_params[f"head.{key}"] = value
        flat_params["_step"] = mx.array(self.step)
        mx.savez(str(path), **flat_params)
        print(f"  Saved checkpoint: {path}")


def main():
    parser = argparse.ArgumentParser(description="Knowledge Distillation Training")

    parser.add_argument("--manifest", type=str, default="data/v4_expanded/train_manifest.json")
    parser.add_argument("--val-manifest", type=str, default=None)
    parser.add_argument("--soft-labels", type=str, default="data/soft_labels")
    parser.add_argument("--encoder-cache", type=str, default="data/v4_expanded/encoder_cache")

    parser.add_argument("--teacher", type=str, default="wav2vec2-xlsr-ser")
    parser.add_argument("--task", type=str, default="emotion", choices=["emotion", "paralinguistics"])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=2.0)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=str, default="checkpoints/distillation")

    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    config = DistillationConfig(
        manifest_path=args.manifest,
        val_manifest_path=args.val_manifest,
        soft_labels_dir=args.soft_labels,
        encoder_cache_dir=args.encoder_cache,
        teacher_model=args.teacher,
        task=args.task,
        alpha=args.alpha,
        temperature=args.temperature,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
    )

    trainer = DistillationTrainer(config)

    print("\nLoading training samples...")
    train_samples = trainer.load_samples(config.manifest_path, limit=args.limit)

    val_samples = None
    if config.val_manifest_path:
        print("\nLoading validation samples...")
        val_samples = trainer.load_samples(config.val_manifest_path)

    trainer.train(train_samples, val_samples)


if __name__ == "__main__":
    main()
