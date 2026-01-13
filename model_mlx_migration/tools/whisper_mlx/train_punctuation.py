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
Train Emotion-Aware Punctuation Head.

This script trains a punctuation prediction head on top of the frozen
Whisper encoder, optionally using emotion and pitch features.

6 Output Classes:
    0: PERIOD (.)
    1: COMMA (,)
    2: QUESTION (?)
    3: EXCLAMATION (!)
    4: ELLIPSIS (...)
    5: NONE (no punctuation)

The key insight is that questions are often distinguishable by rising
intonation (pitch) and certain emotional states. By conditioning on
emotion and pitch features, we can improve question detection accuracy.

Usage:
    python -m tools.whisper_mlx.train_punctuation \
        --data-dir data/LibriSpeech/train-clean-100 \
        --output-dir checkpoints/punctuation_head \
        --epochs 10

Architecture:
    Audio -> Mel -> Whisper Encoder (frozen)
                      |
                      v
              [encoder_output]
                      |
           +----------+----------+
           |          |          |
           v          v          v
      PunctuationHead  EmotionHead  PitchHead
           |          (frozen)   (frozen)
           |              |          |
           +--------------+----------+
                      |
                      v
              [punctuation logits]
"""

import argparse
import gc
import json
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .audio import load_audio, log_mel_spectrogram
from .encoder_cache import TrainingEncoderCache
from .multi_head import (
    PUNCTUATION_CLASSES,
    EmotionHead,
    MultiHeadConfig,
    PunctuationHead,
    focal_loss,
)

# ==============================================================================
# Configuration
# ==============================================================================


@dataclass
class PunctuationTrainingConfig:
    """Configuration for punctuation head training."""

    # Data
    data_dir: str = "data/LibriSpeech/dev-clean"
    output_dir: str = "checkpoints/punctuation_head"
    max_audio_len: float = 30.0  # Max audio length in seconds
    max_samples: int = 0  # 0 = all samples

    # Model
    whisper_model: str = "mlx-community/whisper-large-v3-mlx"
    model_size: str = "large-v3"
    d_model: int = 1280  # large-v3 dimension
    n_mels: int = 128  # large-v3 uses 128, others use 80
    hidden_dim: int = 256
    num_classes: int = 6
    dropout: float = 0.1

    # Feature conditioning
    use_emotion: bool = True  # Use emotion head features
    use_pitch: bool = True  # Use pitch head features
    emotion_checkpoint: str | None = None  # Pre-trained emotion head
    pitch_checkpoint: str | None = None  # Pre-trained pitch head

    # Training
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 3e-4
    warmup_steps: int = 500
    weight_decay: float = 0.01
    label_smoothing: float = 0.1

    # Focal loss (for class imbalance)
    use_focal_loss: bool = True
    focal_gamma: float = 2.0

    # Data augmentation
    spec_augment: bool = False
    freq_mask_param: int = 27
    time_mask_param: int = 100
    num_freq_masks: int = 2
    num_time_masks: int = 2

    # Optimization
    encoder_cache_dir: str | None = None
    length_sorted_batching: bool = True
    bucket_size_multiplier: int = 8

    # Logging
    log_interval: int = 50
    save_interval: int = 500


# ==============================================================================
# Data Loading
# ==============================================================================


@dataclass
class PunctuationSample:
    """Single training sample with punctuation labels."""

    audio_path: Path
    text: str
    punct_labels: list[int]  # Per-word punctuation labels
    duration: float = 0.0


def extract_punctuation_from_text(text: str) -> list[int]:
    """
    Extract punctuation labels from text.

    Returns a list of punctuation class indices, one per word.

    Punctuation classes:
        0: PERIOD (.)
        1: COMMA (,)
        2: QUESTION (?)
        3: EXCLAMATION (!)
        4: ELLIPSIS (...)
        5: NONE (no punctuation)
    """
    punct_map = {
        ".": 0,
        ",": 1,
        "?": 2,
        "!": 3,
        "...": 4,
    }

    # Split into words, preserving punctuation
    words = text.split()
    labels = []

    for word in words:
        label = 5  # NONE default

        # Check for ellipsis first (before checking for period)
        if word.endswith("..."):
            label = 4
        # Check other punctuation
        else:
            for punct, idx in punct_map.items():
                if punct != "..." and word.endswith(punct):
                    label = idx
                    break

        labels.append(label)

    return labels


def load_librispeech_samples(
    data_dir: Path, max_samples: int = 0,
) -> list[PunctuationSample]:
    """
    Load LibriSpeech samples with punctuation labels.

    LibriSpeech transcripts already have punctuation marks.
    """
    samples = []

    # Find all .trans.txt files
    trans_files = list(data_dir.glob("**/*.trans.txt"))
    print(f"Found {len(trans_files)} transcript files")

    for trans_path in trans_files:
        audio_dir = trans_path.parent

        with open(trans_path) as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) != 2:
                    continue

                audio_id, text = parts
                audio_path = audio_dir / f"{audio_id}.flac"

                if not audio_path.exists():
                    continue

                # Extract punctuation from transcript
                punct_labels = extract_punctuation_from_text(text)

                sample = PunctuationSample(
                    audio_path=audio_path,
                    text=text,
                    punct_labels=punct_labels,
                )
                samples.append(sample)

                if max_samples > 0 and len(samples) >= max_samples:
                    print(f"Loaded {len(samples)} samples (limited)")
                    return samples

    print(f"Loaded {len(samples)} samples")
    return samples


def load_meld_samples(
    meld_dir: Path, split: str = "train", max_samples: int = 0,
) -> list[PunctuationSample]:
    """
    Load MELD dataset samples with punctuation and emotion labels.

    MELD provides conversational speech from Friends TV show with:
    - Ground-truth emotion labels (7 classes)
    - Natural punctuation in transcripts
    - Pre-extracted audio files

    Args:
        meld_dir: Path to MELD.Raw directory
        split: One of "train", "dev", "test"
        max_samples: Max samples to load (0 = all)

    Returns:
        List of PunctuationSample objects
    """
    import csv

    samples = []

    # Map split name to audio directory and CSV file
    split_map = {
        "train": ("audio_train", "train_sent_emo.csv"),
        "dev": ("audio_dev", "dev_sent_emo.csv"),
        "test": ("audio_test", "test_sent_emo.csv"),
    }

    if split not in split_map:
        raise ValueError(f"Unknown split: {split}. Use one of: {list(split_map.keys())}")

    audio_dir_name, csv_name = split_map[split]
    audio_dir = meld_dir / audio_dir_name
    csv_path = meld_dir / csv_name

    if not csv_path.exists():
        raise FileNotFoundError(f"MELD CSV not found: {csv_path}")
    if not audio_dir.exists():
        raise FileNotFoundError(
            f"MELD audio directory not found: {audio_dir}\n"
            f"Run scripts/extract_meld_audio.sh to extract audio from videos.",
        )

    print(f"Loading MELD {split} from {csv_path}")
    print(f"Audio directory: {audio_dir}")

    # Count available audio files
    audio_files = {p.stem for p in audio_dir.glob("*.wav")}
    print(f"Found {len(audio_files)} audio files")

    # Read CSV annotations
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Get dialogue and utterance IDs to construct filename
            dia_id = row["Dialogue_ID"]
            utt_id = row["Utterance_ID"]

            # MELD naming convention: dia{dia_id}_utt{utt_id}.wav
            audio_name = f"dia{dia_id}_utt{utt_id}"
            audio_path = audio_dir / f"{audio_name}.wav"

            if not audio_path.exists():
                # Try alternate naming
                if audio_name not in audio_files:
                    continue
                audio_path = audio_dir / f"{audio_name}.wav"

            # Get transcript with natural punctuation
            text = row["Utterance"].strip()

            # Skip empty utterances
            if not text:
                continue

            # Extract punctuation labels from text
            punct_labels = extract_punctuation_from_text(text)

            # Get emotion label (7 classes in MELD)
            # Store as metadata for analysis (not used in training directly)
            row.get("Emotion", "neutral").lower()

            sample = PunctuationSample(
                audio_path=audio_path,
                text=text,
                punct_labels=punct_labels,
            )
            samples.append(sample)

            if max_samples > 0 and len(samples) >= max_samples:
                print(f"Loaded {len(samples)} MELD samples (limited)")
                return samples

    print(f"Loaded {len(samples)} MELD samples")

    # Analyze punctuation distribution
    punct_counts = dict.fromkeys(range(6), 0)
    for s in samples:
        for label in s.punct_labels:
            punct_counts[label] += 1

    print("MELD punctuation distribution:")
    total = sum(punct_counts.values())
    for i, name in enumerate(PUNCTUATION_CLASSES):
        count = punct_counts[i]
        pct = 100 * count / total if total > 0 else 0
        print(f"  {name}: {count} ({pct:.1f}%)")

    return samples


def load_combined_samples(
    librispeech_dir: Path | None = None,
    meld_dir: Path | None = None,
    max_samples: int = 0,
) -> list[PunctuationSample]:
    """
    Load samples from multiple datasets.

    Args:
        librispeech_dir: Path to LibriSpeech directory (optional)
        meld_dir: Path to MELD.Raw directory (optional)
        max_samples: Max total samples (0 = all)

    Returns:
        Combined list of samples
    """
    samples = []

    if librispeech_dir and librispeech_dir.exists():
        libri_samples = load_librispeech_samples(
            librispeech_dir,
            max(0, max_samples),
        )
        samples.extend(libri_samples)
        print(f"LibriSpeech: {len(libri_samples)} samples")

    if meld_dir and meld_dir.exists():
        # Load all MELD splits
        remaining = max_samples - len(samples) if max_samples > 0 else 0
        for split in ["train", "dev"]:  # Don't include test in training
            meld_samples = load_meld_samples(
                meld_dir,
                split=split,
                max_samples=max(0, remaining),
            )
            samples.extend(meld_samples)
            if max_samples > 0:
                remaining = max_samples - len(samples)
                if remaining <= 0:
                    break

    print(f"Total combined samples: {len(samples)}")
    return samples


# ==============================================================================
# Model Utilities
# ==============================================================================


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss / 1024 / 1024  # macOS returns bytes


def clear_memory():
    """Clear MLX and Python memory."""
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
        mx.metal.clear_cache()
    gc.collect()


def load_whisper_model(model_name: str):
    """Load frozen Whisper model."""
    try:
        from mlx_whisper import load_models
        return load_models.load_model(model_name)
    except ImportError:
        # Fallback to manual loading
        from .model import WhisperMLX

        return WhisperMLX.from_pretrained(model_name)


# ==============================================================================
# Trainer
# ==============================================================================


class PunctuationTrainer:
    """
    Trainer for emotion-aware punctuation head.

    Uses cross-entropy (or focal) loss for 6-class frame-level prediction.
    """

    def __init__(
        self,
        config: PunctuationTrainingConfig,
        whisper_model: Any,
        punctuation_head: PunctuationHead,
        emotion_head: EmotionHead | None = None,
        pitch_head: nn.Module | None = None,
    ):
        self.config = config
        self.whisper_model = whisper_model
        self.punctuation_head = punctuation_head
        self.emotion_head = emotion_head
        self.pitch_head = pitch_head

        # Optimizer (only for punctuation head)
        self.optimizer = optim.AdamW(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Training state
        self.step = 0
        self.epoch = 0
        self.best_loss = float("inf")
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Log file
        self.log_file = self.output_dir / "training.log"

        # Encoder cache
        self.encoder_cache = None
        if config.encoder_cache_dir:
            self.encoder_cache = TrainingEncoderCache(
                cache_dir=config.encoder_cache_dir,
            )

        # Compute class weights for imbalanced data
        # Expected distribution: NONE >> PERIOD > COMMA > QUESTION > EXCLAMATION > ELLIPSIS
        self.class_weights = mx.array([
            1.0,   # PERIOD - common
            1.0,   # COMMA - common
            2.0,   # QUESTION - less common, important
            3.0,   # EXCLAMATION - rare
            3.0,   # ELLIPSIS - rare
            0.5,   # NONE - very common, downweight
        ], dtype=mx.float32)

    def log(self, message: str):
        """Log to console and file."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(message)
        with open(self.log_file, "a") as f:
            f.write(f"{timestamp} | {message}\n")

    def _get_encoder_output(
        self, audio_path: Path, mel: mx.array | None = None,
    ) -> mx.array:
        """Get encoder output, using cache if available."""
        # Try cache first
        if self.encoder_cache is not None:
            cached = self.encoder_cache.load(str(audio_path))
            if cached is not None:
                enc_out, _ = cached
                return mx.array(enc_out)

        # Compute if not cached
        if mel is None:
            audio = load_audio(str(audio_path))
            mel = log_mel_spectrogram(audio, n_mels=self.config.n_mels)
            mel = mx.array(mel)

        # Pad to 3000 frames (required by Whisper encoder)
        encoder_frames_total = 3000
        if mel.shape[0] < encoder_frames_total:
            pad_amount = encoder_frames_total - mel.shape[0]
            mel = mx.pad(mel, [(0, pad_amount), (0, 0)])
        else:
            mel = mel[:encoder_frames_total]

        # Add batch dimension if needed
        if mel.ndim == 2:
            mel = mel[None, :, :]

        # Run encoder
        encoder_out = self.whisper_model.encoder(mel)
        mx.eval(encoder_out)

        # Cache for next time
        if self.encoder_cache is not None:
            self.encoder_cache.save(
                str(audio_path),
                np.array(encoder_out[0]),  # Remove batch dim
                encoder_out.shape[1],
            )

        return encoder_out

    def _prepare_batch(
        self, samples: list[PunctuationSample],
    ) -> tuple[mx.array, mx.array, mx.array | None, mx.array | None]:
        """
        Prepare a batch of samples.

        Returns:
            encoder_output: (batch, T, d_model)
            labels: (batch, T) punctuation labels
            emotion_probs: (batch, T, num_emotions) or None
            pitch_values: (batch, T, 1) or None
        """
        encoder_outputs = []
        label_sequences = []

        for sample in samples:
            # Get encoder output
            enc_out = self._get_encoder_output(sample.audio_path)
            if enc_out.ndim == 3:
                enc_out = enc_out[0]  # Remove batch dim

            encoder_outputs.append(enc_out)

            # Expand word-level labels to frame-level
            # Naive approach: distribute labels evenly across frames
            T = enc_out.shape[0]
            num_words = len(sample.punct_labels)

            if num_words > 0:
                # Simple linear interpolation of labels
                frame_labels = np.zeros(T, dtype=np.int32)
                for t in range(T):
                    word_idx = min(int(t * num_words / T), num_words - 1)
                    frame_labels[t] = sample.punct_labels[word_idx]
            else:
                frame_labels = np.full(T, 5, dtype=np.int32)  # NONE

            label_sequences.append(frame_labels)

        # Pad sequences to same length
        max_T = max(enc.shape[0] for enc in encoder_outputs)

        batch_enc = np.zeros((len(samples), max_T, self.config.d_model), dtype=np.float32)
        batch_labels = np.full((len(samples), max_T), 5, dtype=np.int32)  # NONE padding

        for i, (enc, labels) in enumerate(zip(encoder_outputs, label_sequences, strict=False)):
            T = enc.shape[0]
            batch_enc[i, :T] = np.array(enc)
            batch_labels[i, :T] = labels

        encoder_output = mx.array(batch_enc)
        labels = mx.array(batch_labels)

        # Get emotion and pitch features if using
        emotion_probs = None
        pitch_values = None

        if self.config.use_emotion and self.emotion_head is not None:
            emotion_probs = self.emotion_head(encoder_output, return_frame_logits=True)
            emotion_probs = mx.softmax(emotion_probs, axis=-1)

        if self.config.use_pitch and self.pitch_head is not None:
            pitch_values = self.pitch_head(encoder_output)
            if pitch_values.ndim == 2:
                pitch_values = pitch_values[:, :, None]

        return encoder_output, labels, emotion_probs, pitch_values

    def _compute_loss(
        self,
        logits: mx.array,
        labels: mx.array,
    ) -> mx.array:
        """Compute cross-entropy or focal loss."""
        if self.config.use_focal_loss:
            return focal_loss(
                logits,
                labels,
                gamma=self.config.focal_gamma,
                alpha=self.class_weights,
                reduction="mean",
            )
        # Standard cross-entropy with label smoothing
        batch, T, num_classes = logits.shape
        logits_flat = logits.reshape(batch * T, num_classes)
        labels_flat = labels.reshape(batch * T)

        # One-hot with smoothing
        n_samples = logits_flat.shape[0]
        smooth = self.config.label_smoothing
        target_probs = mx.zeros((n_samples, num_classes))
        target_probs = target_probs + smooth / num_classes
        indices = mx.arange(n_samples)
        target_probs = mx.where(
            mx.expand_dims(indices, -1) == mx.expand_dims(labels_flat, 0)[:, :, None],
            target_probs + (1 - smooth),
            target_probs,
        )

        # Cross-entropy
        log_probs = mx.log_softmax(logits_flat, axis=-1)
        loss = -mx.sum(target_probs * log_probs, axis=-1)

        # Apply class weights
        sample_weights = self.class_weights[labels_flat]
        loss = loss * sample_weights

        return mx.mean(loss)

    def train_step(
        self, batch: tuple[mx.array, mx.array, mx.array | None, mx.array | None],
    ) -> tuple[float, dict[str, float]]:
        """
        Single training step.

        Returns:
            loss: float loss value
            metrics: dict of metrics
        """
        encoder_output, labels, emotion_probs, pitch_values = batch

        def loss_fn(head):
            logits = head(encoder_output, emotion_probs, pitch_values)
            return self._compute_loss(logits, labels)

        # Compute loss and gradients
        loss, grads = mx.value_and_grad(loss_fn)(self.punctuation_head)
        mx.eval(loss)

        # Update parameters
        self.optimizer.update(self.punctuation_head, grads)
        mx.eval(self.punctuation_head.parameters())

        # Compute accuracy (no gradients needed for metrics)
        logits_for_acc = self.punctuation_head(encoder_output, emotion_probs, pitch_values)
        predictions = mx.argmax(logits_for_acc, axis=-1)
        accuracy = mx.mean((predictions == labels).astype(mx.float32))

        return float(loss), {"accuracy": float(accuracy)}

    def validate_step(
        self, batch: tuple[mx.array, mx.array, mx.array | None, mx.array | None],
    ) -> tuple[float, dict[str, float]]:
        """Single validation step."""
        encoder_output, labels, emotion_probs, pitch_values = batch

        logits = self.punctuation_head(encoder_output, emotion_probs, pitch_values)
        loss = self._compute_loss(logits, labels)

        predictions = mx.argmax(logits, axis=-1)
        accuracy = mx.mean((predictions == labels).astype(mx.float32))

        # Per-class accuracy (especially question recall)
        metrics = {"accuracy": float(accuracy)}

        # Question recall (class 2)
        question_mask = labels == 2
        if mx.sum(question_mask) > 0:
            question_correct = mx.sum((predictions == 2) & question_mask)
            question_total = mx.sum(question_mask)
            metrics["question_recall"] = float(question_correct / question_total)

        return float(loss), metrics

    def _populate_sample_durations(self, samples: list[PunctuationSample]):
        """Populate duration field for length-sorted batching."""
        import soundfile as sf

        for sample in samples:
            try:
                info = sf.info(str(sample.audio_path))
                sample.duration = info.duration
            except Exception:
                sample.duration = 0.0

    def _create_length_sorted_batches(
        self, samples: list[PunctuationSample],
    ) -> list[list[PunctuationSample]]:
        """Create batches of similar-length samples."""
        # Sort by duration
        sorted_samples = sorted(samples, key=lambda s: s.duration)

        batch_size = self.config.batch_size
        bucket_size = batch_size * self.config.bucket_size_multiplier

        batches = []
        rng = np.random.default_rng()

        # Process in buckets
        for bucket_start in range(0, len(sorted_samples), bucket_size):
            bucket_end = min(bucket_start + bucket_size, len(sorted_samples))
            bucket = sorted_samples[bucket_start:bucket_end]

            # Shuffle within bucket
            rng.shuffle(bucket)

            # Create batches from bucket
            for i in range(0, len(bucket), batch_size):
                batch = bucket[i : i + batch_size]
                if batch:
                    batches.append(batch)

        # Shuffle batches
        rng.shuffle(batches)

        return batches

    def train(
        self,
        train_samples: list[PunctuationSample],
        val_samples: list[PunctuationSample],
    ):
        """Main training loop."""
        self.log("=" * 60)
        self.log("Punctuation Head Training")
        self.log("=" * 60)
        self.log(f"Model: {self.config.whisper_model}")
        self.log(f"Data: {self.config.data_dir}")
        self.log(f"Output: {self.config.output_dir}")
        self.log(f"Epochs: {self.config.epochs}")
        self.log(f"Batch size: {self.config.batch_size}")
        self.log(f"Learning rate: {self.config.learning_rate}")
        self.log(f"Use emotion: {self.config.use_emotion}")
        self.log(f"Use pitch: {self.config.use_pitch}")
        self.log(f"Train samples: {len(train_samples)}")
        self.log(f"Val samples: {len(val_samples)}")
        self.log("=" * 60)

        # Populate durations for length-sorted batching
        if self.config.length_sorted_batching:
            self.log("Populating sample durations...")
            self._populate_sample_durations(train_samples)
            self._populate_sample_durations(val_samples)

        for epoch in range(self.config.epochs):
            self.epoch = epoch
            epoch_start = time.time()

            # Train epoch
            train_loss = self._train_epoch(train_samples)
            self.train_losses.append(train_loss)

            # Validate
            val_loss, val_metrics = self._validate(val_samples)
            self.val_losses.append(val_loss)

            epoch_time = time.time() - epoch_start

            self.log(f"\nEpoch {epoch + 1}/{self.config.epochs} ({epoch_time:.1f}s)")
            self.log(f"  Train loss: {train_loss:.4f}")
            self.log(f"  Val loss: {val_loss:.4f}")
            for k, v in val_metrics.items():
                self.log(f"  {k}: {v:.4f}")

            # Save best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self._save_checkpoint("best.npz")
                self.log("  New best model saved!")

            # Save epoch checkpoint
            self._save_checkpoint(f"epoch_{epoch + 1}.npz")

        # Save final model
        self._save_checkpoint("final.npz")
        self.log("\nTraining complete!")

        # Save history
        self._save_history()

    def _train_epoch(self, samples: list[PunctuationSample]) -> float:
        """Train for one epoch."""
        total_loss = 0.0
        num_batches = 0

        # Create batches
        if self.config.length_sorted_batching:
            batches = self._create_length_sorted_batches(samples)
        else:
            # Random batches
            shuffled = samples.copy()
            rng = np.random.default_rng()
            rng.shuffle(shuffled)
            batches = [
                shuffled[i : i + self.config.batch_size]
                for i in range(0, len(shuffled), self.config.batch_size)
            ]

        for batch_samples in batches:
            # Prepare batch
            batch = self._prepare_batch(batch_samples)

            # Training step
            loss, metrics = self.train_step(batch)

            if loss > 0 and not np.isnan(loss):
                total_loss += loss
                num_batches += 1

            self.step += 1

            # Logging
            if self.step % self.config.log_interval == 0:
                mem_mb = get_memory_usage_mb()
                acc = metrics.get("accuracy", 0)
                self.log(
                    f"  Step {self.step}: loss={loss:.4f}, acc={acc:.4f}, mem={mem_mb:.0f}MB",
                )

            # Save checkpoint
            if self.step % self.config.save_interval == 0:
                self._save_checkpoint(f"step_{self.step}.npz")

            # Clear memory
            if self.step % 100 == 0:
                clear_memory()

        return total_loss / max(num_batches, 1)

    def _validate(
        self, samples: list[PunctuationSample],
    ) -> tuple[float, dict[str, float]]:
        """Run validation."""
        # Set to eval mode to disable dropout
        self.punctuation_head.eval()

        total_loss = 0.0
        total_metrics: dict[str, float] = {}
        num_batches = 0

        # Create batches (no shuffling for validation)
        batches = [
            samples[i : i + self.config.batch_size]
            for i in range(0, len(samples), self.config.batch_size)
        ]

        for batch_samples in batches:
            batch = self._prepare_batch(batch_samples)
            loss, metrics = self.validate_step(batch)

            if loss > 0 and not np.isnan(loss):
                total_loss += loss
                num_batches += 1
                for k, v in metrics.items():
                    total_metrics[k] = total_metrics.get(k, 0.0) + v

        avg_loss = total_loss / max(num_batches, 1)
        avg_metrics = {k: v / max(num_batches, 1) for k, v in total_metrics.items()}

        # Restore training mode
        self.punctuation_head.train()

        return avg_loss, avg_metrics

    def _save_checkpoint(self, filename: str):
        """Save checkpoint."""
        path = self.output_dir / filename
        print(f"Saving: {path}")

        # Flatten nested parameters for saving
        def flatten_params(params, prefix=""):
            flat = {}
            for name, value in params.items():
                key = f"{prefix}{name}" if prefix else name
                if isinstance(value, dict):
                    flat.update(flatten_params(value, f"{key}."))
                else:
                    mx.eval(value)
                    flat[key] = value
            return flat

        params = flatten_params(self.punctuation_head.parameters())
        mx.savez(str(path), **params)

        # Save training state
        state_path = self.output_dir / f"{filename.replace('.npz', '_state.json')}"
        state = {
            "step": self.step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "config": {
                "d_model": self.config.d_model,
                "hidden_dim": self.config.hidden_dim,
                "num_classes": self.config.num_classes,
                "use_emotion": self.config.use_emotion,
                "use_pitch": self.config.use_pitch,
            },
        }
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)

    def _save_history(self):
        """Save training history."""
        history_path = self.output_dir / "history.json"
        history = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_loss": self.best_loss,
            "final_step": self.step,
            "epochs": self.epoch + 1,
        }
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    def evaluate_test_set(
        self,
        test_samples: list[PunctuationSample],
    ) -> dict[str, Any]:
        """
        Comprehensive evaluation on test set.

        Returns per-class metrics including:
        - Overall accuracy
        - Per-class precision, recall, F1
        - Question detection metrics (critical)
        - Confusion matrix
        """
        print(f"\nEvaluating on {len(test_samples)} test samples...")

        all_predictions = []
        all_targets = []
        total_loss = 0.0
        num_batches = 0

        # Create batches
        for batch_start in range(0, len(test_samples), self.config.batch_size):
            batch_end = min(batch_start + self.config.batch_size, len(test_samples))
            batch = test_samples[batch_start:batch_end]

            # Prepare batch
            encoder_outputs, labels, emotion_probs, pitch_values = self._prepare_batch(batch)

            # Get predictions
            encoder_output = mx.array(encoder_outputs)
            labels_mx = mx.array(labels)
            emotion_probs_mx = mx.array(emotion_probs) if emotion_probs is not None else None
            pitch_values_mx = mx.array(pitch_values) if pitch_values is not None else None

            logits = self.punctuation_head(encoder_output, emotion_probs_mx, pitch_values_mx)

            # Compute loss
            loss = self._compute_loss(logits, labels_mx)
            total_loss += float(loss)
            num_batches += 1

            # Get predictions
            predictions = mx.argmax(logits, axis=-1)
            mx.eval(predictions)

            # Flatten and store
            pred_np = np.array(predictions).flatten()
            target_np = np.array(labels).flatten()

            all_predictions.extend(pred_np.tolist())
            all_targets.extend(target_np.tolist())

            if (batch_start // self.config.batch_size) % 50 == 0:
                print(f"  Evaluated {batch_start + len(batch)}/{len(test_samples)} samples")

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Compute metrics
        results = self._compute_detailed_metrics(all_predictions, all_targets)
        results["loss"] = avg_loss
        results["num_samples"] = len(test_samples)

        # Print results
        print("\n" + "=" * 60)
        print("TEST SET EVALUATION RESULTS")
        print("=" * 60)
        print(f"Samples: {len(test_samples)}")
        print(f"Loss: {avg_loss:.4f}")
        print(f"Overall Accuracy: {results['accuracy']:.4f}")
        print()

        print("Per-Class Metrics:")
        print("-" * 60)
        print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print("-" * 60)

        for class_name in PUNCTUATION_CLASSES:
            m = results["per_class"][class_name]
            print(f"{class_name:<15} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['support']:>10}")

        print("-" * 60)
        print(f"{'MACRO AVG':<15} {results['macro_precision']:>10.4f} {results['macro_recall']:>10.4f} {results['macro_f1']:>10.4f}")
        print(f"{'WEIGHTED AVG':<15} {results['weighted_precision']:>10.4f} {results['weighted_recall']:>10.4f} {results['weighted_f1']:>10.4f}")
        print()

        # Highlight question detection (critical metric)
        q_metrics = results["per_class"]["QUESTION"]
        print("=" * 60)
        print("QUESTION DETECTION (Critical Metric)")
        print("=" * 60)
        print(f"  Precision: {q_metrics['precision']:.4f}")
        print(f"  Recall:    {q_metrics['recall']:.4f} {'<-- TARGET: >0.80' if q_metrics['recall'] < 0.80 else '✓ GOOD!'}")
        print(f"  F1 Score:  {q_metrics['f1']:.4f}")
        print(f"  Support:   {q_metrics['support']}")
        print("=" * 60)

        return results

    def _compute_detailed_metrics(
        self,
        predictions: list[int],
        targets: list[int],
    ) -> dict[str, Any]:
        """Compute detailed metrics for all classes."""

        # Count predictions and targets per class
        num_classes = len(PUNCTUATION_CLASSES)
        tp = [0] * num_classes  # True positives
        fp = [0] * num_classes  # False positives
        fn = [0] * num_classes  # False negatives
        support = [0] * num_classes  # Total actual per class

        for pred, target in zip(predictions, targets, strict=False):
            support[target] += 1
            if pred == target:
                tp[pred] += 1
            else:
                fp[pred] += 1
                fn[target] += 1

        # Compute per-class metrics
        per_class = {}
        for i, name in enumerate(PUNCTUATION_CLASSES):
            precision = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0.0
            recall = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            per_class[name] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support[i],
                "tp": tp[i],
                "fp": fp[i],
                "fn": fn[i],
            }

        # Compute overall accuracy
        correct = sum(1 for p, t in zip(predictions, targets, strict=False) if p == t)
        accuracy = correct / len(predictions) if predictions else 0.0

        # Compute macro and weighted averages
        macro_precision = sum(per_class[n]["precision"] for n in PUNCTUATION_CLASSES) / num_classes
        macro_recall = sum(per_class[n]["recall"] for n in PUNCTUATION_CLASSES) / num_classes
        macro_f1 = sum(per_class[n]["f1"] for n in PUNCTUATION_CLASSES) / num_classes

        total_support = sum(support)
        weighted_precision = sum(per_class[n]["precision"] * per_class[n]["support"] for n in PUNCTUATION_CLASSES) / total_support if total_support > 0 else 0.0
        weighted_recall = sum(per_class[n]["recall"] * per_class[n]["support"] for n in PUNCTUATION_CLASSES) / total_support if total_support > 0 else 0.0
        weighted_f1 = sum(per_class[n]["f1"] * per_class[n]["support"] for n in PUNCTUATION_CLASSES) / total_support if total_support > 0 else 0.0

        return {
            "accuracy": accuracy,
            "per_class": per_class,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "weighted_precision": weighted_precision,
            "weighted_recall": weighted_recall,
            "weighted_f1": weighted_f1,
        }


# ==============================================================================
# Main
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Train emotion-aware punctuation head",
    )

    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/LibriSpeech/dev-clean",
        help="Path to LibriSpeech data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/punctuation_head",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Max samples to load (0 = all)",
    )
    parser.add_argument(
        "--meld-dir",
        type=str,
        default=None,
        help="Path to MELD.Raw directory (for MELD or combined dataset)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="librispeech",
        choices=["librispeech", "meld", "combined"],
        help="Dataset to use: librispeech, meld, or combined",
    )

    # Model arguments
    parser.add_argument(
        "--model-size",
        type=str,
        default="large-v3",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Hidden dimension for punctuation head",
    )

    # Feature conditioning
    parser.add_argument(
        "--use-emotion",
        action="store_true",
        default=True,
        help="Use emotion head features",
    )
    parser.add_argument(
        "--no-emotion",
        action="store_true",
        help="Disable emotion features",
    )
    parser.add_argument(
        "--use-pitch",
        action="store_true",
        default=True,
        help="Use pitch head features",
    )
    parser.add_argument(
        "--no-pitch",
        action="store_true",
        help="Disable pitch features",
    )
    parser.add_argument(
        "--emotion-checkpoint",
        type=str,
        help="Path to pre-trained emotion head checkpoint",
    )
    parser.add_argument(
        "--pitch-checkpoint",
        type=str,
        help="Path to pre-trained pitch head checkpoint",
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        help="Label smoothing factor",
    )

    # Focal loss
    parser.add_argument(
        "--use-focal-loss",
        action="store_true",
        default=True,
        help="Use focal loss for class imbalance",
    )
    parser.add_argument(
        "--no-focal-loss",
        action="store_true",
        help="Disable focal loss",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focal loss gamma parameter",
    )

    # Optimization
    parser.add_argument(
        "--encoder-cache-dir",
        type=str,
        help="Directory for encoder output cache",
    )
    parser.add_argument(
        "--length-sorted-batching",
        action="store_true",
        default=True,
        help="Enable length-sorted batching",
    )
    parser.add_argument(
        "--no-length-sorted-batching",
        action="store_true",
        help="Disable length-sorted batching",
    )

    # Resume
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume from checkpoint",
    )

    # Evaluation mode
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation on test set instead of training",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Checkpoint to evaluate (required with --evaluate)",
    )

    args = parser.parse_args()

    # Build config
    config = PunctuationTrainingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        model_size=args.model_size,
        hidden_dim=args.hidden_dim,
        use_emotion=args.use_emotion and not args.no_emotion,
        use_pitch=args.use_pitch and not args.no_pitch,
        emotion_checkpoint=args.emotion_checkpoint,
        pitch_checkpoint=args.pitch_checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        label_smoothing=args.label_smoothing,
        use_focal_loss=args.use_focal_loss and not args.no_focal_loss,
        focal_gamma=args.focal_gamma,
        encoder_cache_dir=args.encoder_cache_dir,
        length_sorted_batching=args.length_sorted_batching and not args.no_length_sorted_batching,
    )

    # Set model name based on size
    model_map = {
        "tiny": "mlx-community/whisper-tiny-mlx",
        "base": "mlx-community/whisper-base-mlx",
        "small": "mlx-community/whisper-small-mlx",
        "medium": "mlx-community/whisper-medium-mlx",
        "large-v3": "mlx-community/whisper-large-v3-mlx",
    }
    config.whisper_model = model_map.get(config.model_size, config.whisper_model)

    # Set d_model based on model size
    d_model_map = {
        "tiny": 384,
        "base": 512,
        "small": 768,
        "medium": 1024,
        "large-v3": 1280,
    }
    config.d_model = d_model_map.get(config.model_size, config.d_model)

    # Set n_mels based on model size (v3 uses 128, others use 80)
    n_mels_map = {
        "tiny": 80,
        "base": 80,
        "small": 80,
        "medium": 80,
        "large-v3": 128,
    }
    config.n_mels = n_mels_map.get(config.model_size, 80)

    print("=" * 60)
    print("Loading models...")
    print("=" * 60)

    # Load Whisper model
    print("1. Loading Whisper model (frozen encoder)...")
    whisper_model = load_whisper_model(config.whisper_model)

    # Create punctuation head
    multi_head_config = MultiHeadConfig(
        d_model=config.d_model,
        num_punctuation_classes=config.num_classes,
        punctuation_hidden_dim=config.hidden_dim,
        punctuation_use_emotion=config.use_emotion,
        punctuation_use_pitch=config.use_pitch,
        num_emotions=34,  # Extended taxonomy for emotion features
    )

    print("2. Creating punctuation head...")
    punctuation_head = PunctuationHead(multi_head_config)

    # Load emotion head if using
    emotion_head = None
    if config.use_emotion:
        print("3. Loading emotion head (frozen)...")
        emotion_head = EmotionHead(multi_head_config)
        if config.emotion_checkpoint:
            weights = mx.load(config.emotion_checkpoint)
            emotion_head.load_weights(list(weights.items()))
        # Freeze emotion head
        for param in emotion_head.parameters().values():
            param.freeze()

    # Load pitch head if using
    pitch_head = None
    if config.use_pitch:
        print("4. Loading pitch head (frozen)...")
        # Using simple PitchHeadMLP for now
        from .multi_head import PitchHeadMLP

        pitch_head = PitchHeadMLP(multi_head_config)
        if config.pitch_checkpoint:
            weights = mx.load(config.pitch_checkpoint)
            pitch_head.load_weights(list(weights.items()))
        # Freeze pitch head
        for param in pitch_head.parameters().values():
            param.freeze()

    # Load data
    print("=" * 60)
    print(f"Loading data (dataset={args.dataset})...")
    print("=" * 60)

    data_dir = Path(config.data_dir) if config.data_dir else None
    meld_dir = Path(args.meld_dir) if args.meld_dir else None

    # Load samples based on dataset selection
    if args.dataset == "meld":
        if not meld_dir:
            raise ValueError("--meld-dir required when using MELD dataset")
        # Load MELD train for training, dev for validation
        train_samples = load_meld_samples(meld_dir, split="train", max_samples=config.max_samples)
        val_samples = load_meld_samples(meld_dir, split="dev", max_samples=0)
        all_samples = train_samples + val_samples
    elif args.dataset == "combined":
        all_samples = load_combined_samples(
            librispeech_dir=data_dir,
            meld_dir=meld_dir,
            max_samples=config.max_samples,
        )
        # Split into train/val
        rng = np.random.default_rng(42)
        rng.shuffle(all_samples)
        val_size = max(1, int(len(all_samples) * 0.1))
        val_samples = all_samples[:val_size]
        train_samples = all_samples[val_size:]
    else:  # librispeech (default)
        if not data_dir:
            raise ValueError("--data-dir required when using LibriSpeech dataset")
        all_samples = load_librispeech_samples(data_dir, config.max_samples)
        # Split into train/val
        rng = np.random.default_rng(42)
        rng.shuffle(all_samples)
        val_size = max(1, int(len(all_samples) * 0.1))
        val_samples = all_samples[:val_size]
        train_samples = all_samples[val_size:]

    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")

    # Analyze class distribution
    all_labels = []
    for s in all_samples:
        all_labels.extend(s.punct_labels)
    print("\nClass distribution:")
    for i, name in enumerate(PUNCTUATION_CLASSES):
        count = sum(1 for label in all_labels if label == i)
        pct = 100 * count / len(all_labels) if all_labels else 0
        print(f"  {name}: {count} ({pct:.1f}%)")

    # Create trainer
    trainer = PunctuationTrainer(
        config=config,
        whisper_model=whisper_model,
        punctuation_head=punctuation_head,
        emotion_head=emotion_head,
        pitch_head=pitch_head,
    )

    # Load checkpoint if specified (for resume or evaluate)
    checkpoint_path = args.checkpoint or args.resume
    if checkpoint_path:
        print(f"\nLoading checkpoint: {checkpoint_path}...")
        weights = mx.load(checkpoint_path)
        punctuation_head.load_weights(list(weights.items()))

        # Load state if resuming (not evaluating)
        if args.resume and not args.evaluate:
            state_path = checkpoint_path.replace(".npz", "_state.json")
            if Path(state_path).exists():
                with open(state_path) as f:
                    state = json.load(f)
                trainer.step = state.get("step", 0)
                trainer.epoch = state.get("epoch", 0)
                trainer.best_loss = state.get("best_loss", float("inf"))

    # Evaluation mode
    if args.evaluate:
        if not checkpoint_path:
            raise ValueError("--checkpoint required for evaluation mode")

        # Load test set
        if args.dataset == "meld":
            if not meld_dir:
                raise ValueError("--meld-dir required for MELD evaluation")
            test_samples = load_meld_samples(meld_dir, split="test", max_samples=0)
            print(f"Loaded {len(test_samples)} MELD test samples")
        else:
            print("Using validation set for evaluation (no separate test set)")
            test_samples = val_samples

        # Run evaluation
        results = trainer.evaluate_test_set(test_samples)

        # Save results
        results_path = Path(args.output_dir) / "test_evaluation_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_path}")
        return

    # Train
    trainer.train(train_samples, val_samples)


if __name__ == "__main__":
    main()
