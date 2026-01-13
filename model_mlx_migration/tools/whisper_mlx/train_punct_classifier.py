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
Utterance-level Punctuation Classifier Training.

Trains a simple classifier to predict sentence-ending punctuation from audio.
This approach avoids the CTC loss complexity by treating punctuation as an
utterance-level classification task.

Key insight: MELD has complete utterances with punctuation in transcripts.
We can train a classifier: audio features -> punctuation class

Architecture:
    Audio -> Whisper Encoder (frozen) -> Mean Pool ->
        concat(encoder_pool, emotion_pool, pitch_stats) ->
        MLP -> punctuation_class

Classes:
    0: period (.)
    1: question (?)
    2: exclamation (!)
    3: comma (,)
    4: none (no punctuation)

Usage:
    python -m tools.whisper_mlx.train_punct_classifier \
        --meld-dir data/emotion_punctuation/MELD.Raw \
        --output-dir checkpoints/punct_classifier_v1 \
        --epochs 10

This is a simpler alternative to ProsodyCTC that provides immediate punctuation
prediction capability while we work on proper CTC loss implementation.
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pandas as pd
from mlx.utils import tree_flatten, tree_map

from .audio import load_audio, log_mel_spectrogram
from .model import WhisperMLX
from .rich_ctc_head import CREPEPitchHead, EmotionHead

# =============================================================================
# Constants
# =============================================================================

# Punctuation classes
PUNCT_CLASSES = {
    ".": 0,
    "?": 1,
    "!": 2,
    ",": 3,  # For internal commas, less common for sentence-end
}
NUM_PUNCT_CLASSES = 5  # 4 punct + 1 "none"
PUNCT_CLASS_NAMES = ["period", "question", "exclamation", "comma", "none"]


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class PunctClassifierConfig:
    """Configuration for punctuation classifier training."""

    # Data
    meld_dir: str = "data/emotion_punctuation/MELD.Raw"
    output_dir: str = "checkpoints/punct_classifier_v1"
    max_audio_len: float = 30.0

    # Model
    whisper_model: str = "mlx-community/whisper-large-v3-mlx"
    d_model: int = 1280

    # Classifier architecture
    hidden_dim: int = 512
    dropout: float = 0.1

    # Prosody features
    use_emotion: bool = True
    use_pitch: bool = True
    emotion_weights: str = "checkpoints/emotion_unified_v2/best.npz"
    pitch_weights: str = "checkpoints/pitch_combined_v4/best.npz"
    emotion_dim: int = 34
    pitch_features: int = 4  # mean, std, max, min

    # Training
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-4
    warmup_steps: int = 200
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # Class weights for imbalanced data
    use_class_weights: bool = True

    # Logging
    log_interval: int = 50
    save_interval: int = 500
    val_split: float = 0.1


# =============================================================================
# Model
# =============================================================================


class PunctuationClassifier(nn.Module):
    """
    Utterance-level punctuation classifier.

    Takes pooled encoder features and optional prosody features,
    predicts sentence-ending punctuation class.
    """

    def __init__(
        self,
        d_model: int = 1280,
        hidden_dim: int = 512,
        num_classes: int = NUM_PUNCT_CLASSES,
        use_emotion: bool = True,
        use_pitch: bool = True,
        emotion_dim: int = 34,
        pitch_features: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_emotion = use_emotion
        self.use_pitch = use_pitch

        # Input dimension
        input_dim = d_model
        if use_emotion:
            input_dim += emotion_dim
        if use_pitch:
            input_dim += pitch_features

        # MLP classifier
        self.ln_input = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def __call__(
        self,
        encoder_pool: mx.array,  # (batch, d_model)
        emotion_pool: mx.array | None = None,  # (batch, emotion_dim)
        pitch_stats: mx.array | None = None,  # (batch, pitch_features)
    ) -> mx.array:
        """
        Forward pass.

        Returns:
            logits: (batch, num_classes)
        """
        # Concatenate features
        features = [encoder_pool]
        if self.use_emotion and emotion_pool is not None:
            features.append(emotion_pool)
        if self.use_pitch and pitch_stats is not None:
            features.append(pitch_stats)

        x = mx.concatenate(features, axis=-1)

        # MLP
        x = self.ln_input(x)
        x = nn.gelu(self.fc1(x))
        x = self.dropout(x)
        x = nn.gelu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)



# =============================================================================
# Dataset
# =============================================================================


@dataclass
class PunctSample:
    """Single sample for punctuation classification."""
    audio_path: str
    transcript: str
    punct_class: int
    emotion: str


def get_sentence_end_punct(text: str) -> int:
    """
    Get the sentence-ending punctuation class from text.

    Returns:
        Class index (0-4)
    """
    text = text.strip()
    if not text:
        return 4  # none

    last_char = text[-1]
    if last_char == ".":
        return 0
    if last_char == "?":
        return 1
    if last_char == "!":
        return 2
    if last_char == ",":
        return 3
    return 4  # none


class MELDPunctDataset:
    """MELD dataset for punctuation classification."""

    def __init__(
        self,
        meld_dir: str,
        split: str = "train",
        max_audio_len: float = 30.0,
        val_split: float = 0.1,
    ):
        self.meld_dir = Path(meld_dir)
        self.split = split
        self.max_audio_len = max_audio_len
        self.samples: list[PunctSample] = []

        print(f"Loading MELD dataset from: {self.meld_dir}")
        self._load_meld()

        # Split into train/val
        rng = np.random.default_rng(42)
        indices = rng.permutation(len(self.samples))
        val_size = int(len(self.samples) * val_split)

        self.val_indices = set(indices[:val_size])
        self.train_indices = set(indices[val_size:])

        # Compute class distribution
        class_counts = [0] * NUM_PUNCT_CLASSES
        for s in self.samples:
            class_counts[s.punct_class] += 1
        print(f"Class distribution: {dict(zip(PUNCT_CLASS_NAMES, class_counts, strict=False))}")

        print(f"Total samples: {len(self.samples)}")
        print(f"Train: {len(self.train_indices)}, Val: {len(self.val_indices)}")

    def _load_meld(self):
        """Load MELD samples."""
        csv_file = self.meld_dir / f"{self.split}_sent_emo.csv"
        if not csv_file.exists():
            print(f"WARNING: CSV not found: {csv_file}")
            return

        audio_dir = self.meld_dir / f"audio_{self.split}"
        if not audio_dir.exists():
            print(f"WARNING: Audio dir not found: {audio_dir}")
            return

        df = pd.read_csv(csv_file)
        print(f"  CSV has {len(df)} rows")

        for _, row in df.iterrows():
            dialogue_id = row['Dialogue_ID']
            utterance_id = row['Utterance_ID']
            audio_path = audio_dir / f"dia{dialogue_id}_utt{utterance_id}.wav"

            if audio_path.exists():
                transcript = str(row['Utterance'])
                transcript = transcript.replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"')

                punct_class = get_sentence_end_punct(transcript)

                self.samples.append(PunctSample(
                    audio_path=str(audio_path),
                    transcript=transcript,
                    punct_class=punct_class,
                    emotion=row['Emotion'],
                ))

        print(f"  Loaded {len(self.samples)} samples with audio")

    def get_train_samples(self) -> list[PunctSample]:
        return [self.samples[i] for i in self.train_indices]

    def get_val_samples(self) -> list[PunctSample]:
        return [self.samples[i] for i in self.val_indices]

    def get_class_weights(self) -> mx.array:
        """Compute inverse frequency class weights."""
        counts = [0] * NUM_PUNCT_CLASSES
        for s in self.samples:
            counts[s.punct_class] += 1

        total = sum(counts)
        weights = [total / (NUM_PUNCT_CLASSES * max(c, 1)) for c in counts]
        return mx.array(weights)


# =============================================================================
# Training
# =============================================================================


class PunctClassifierTrainer:
    """Trainer for punctuation classifier."""

    def __init__(self, config: PunctClassifierConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.step = 0
        self.epoch = 0
        self.best_acc = 0.0

        # Load Whisper encoder
        print(f"Loading Whisper model: {config.whisper_model}")
        self.whisper = WhisperMLX.from_pretrained(config.whisper_model)

        # Initialize emotion head if using
        self.emotion_head = None
        if config.use_emotion:
            self.emotion_head = EmotionHead(
                d_model=config.d_model,
                num_emotions=config.emotion_dim,
                hidden_dim=512,
            )
            if Path(config.emotion_weights).exists():
                self._load_emotion_weights(config.emotion_weights)
                print(f"  Loaded emotion weights from {config.emotion_weights}")

        # Initialize pitch head if using
        self.pitch_head = None
        if config.use_pitch:
            self.pitch_head = CREPEPitchHead(
                d_model=config.d_model,
                hidden_dim=256,
                num_bins=361,
            )
            if Path(config.pitch_weights).exists():
                self._load_pitch_weights(config.pitch_weights)
                print(f"  Loaded pitch weights from {config.pitch_weights}")

        # Initialize classifier
        self.classifier = PunctuationClassifier(
            d_model=config.d_model,
            hidden_dim=config.hidden_dim,
            num_classes=NUM_PUNCT_CLASSES,
            use_emotion=config.use_emotion,
            use_pitch=config.use_pitch,
            emotion_dim=config.emotion_dim,
            pitch_features=config.pitch_features,
            dropout=config.dropout,
        )

        # Count parameters
        all_params = tree_flatten(self.classifier.parameters())
        num_params = sum(p.size for _, p in all_params)
        print(f"  Classifier parameters: {num_params:,}")

        self.optimizer = optim.AdamW(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        mx.eval(self.classifier.parameters())

    def _load_emotion_weights(self, path: str):
        """Load emotion head weights."""
        weights = dict(mx.load(path))
        if "emotion.ln.weight" in weights:
            self.emotion_head.ln.weight = weights["emotion.ln.weight"]
            self.emotion_head.ln.bias = weights["emotion.ln.bias"]
            self.emotion_head.fc1.weight = weights["emotion.fc1.weight"]
            self.emotion_head.fc1.bias = weights["emotion.fc1.bias"]
            self.emotion_head.fc2.weight = weights["emotion.fc2.weight"]
            self.emotion_head.fc2.bias = weights["emotion.fc2.bias"]
        mx.eval(self.emotion_head.parameters())

    def _load_pitch_weights(self, path: str):
        """Load pitch head weights."""
        weights = dict(mx.load(path))
        if "pitch.ln_input.weight" in weights:
            self.pitch_head.ln_input.weight = weights["pitch.ln_input.weight"]
            self.pitch_head.ln_input.bias = weights["pitch.ln_input.bias"]
            self.pitch_head.input_proj.weight = weights["pitch.input_proj.weight"]
            self.pitch_head.input_proj.bias = weights["pitch.input_proj.bias"]
            self.pitch_head.output_proj.weight = weights["pitch.output_proj.weight"]
            self.pitch_head.output_proj.bias = weights["pitch.output_proj.bias"]
        mx.eval(self.pitch_head.parameters())

    def _encode_audio(self, audio_path: str) -> tuple[mx.array, mx.array, mx.array] | None:
        """
        Encode audio and extract pooled features.

        Returns:
            (encoder_pool, emotion_pool, pitch_stats) or None if failed
        """
        try:
            audio = load_audio(audio_path)
            if audio is None or len(audio) == 0:
                return None

            duration = len(audio) / 16000
            if duration > self.config.max_audio_len:
                audio = audio[:int(self.config.max_audio_len * 16000)]
                duration = self.config.max_audio_len

            mel = log_mel_spectrogram(audio)
            if mel.shape[0] < 10:
                return None

            # Pad to 3000 frames
            target_frames = 3000
            if mel.shape[0] < target_frames:
                padding = mx.zeros((target_frames - mel.shape[0], mel.shape[1]))
                mel = mx.concatenate([mel, padding], axis=0)
            else:
                mel = mel[:target_frames]

            # Calculate actual frames (not padding)
            actual_encoder_frames = max(1, min(1500, int(duration * 50)))

            # Encode
            mel = mel[None, :]  # (1, T, n_mels)
            encoder_out = self.whisper.encoder(mel)  # (1, 1500, 1280)
            mx.eval(encoder_out)

            # Pool only over actual audio frames
            encoder_actual = encoder_out[0, :actual_encoder_frames]  # (T, 1280)
            encoder_pool = mx.mean(encoder_actual, axis=0)  # (1280,)

            # Emotion features
            emotion_pool = None
            if self.emotion_head is not None:
                emotion_logits = self.emotion_head(encoder_out)  # (1, 1500, 34)
                emotion_actual = emotion_logits[0, :actual_encoder_frames]  # (T, 34)
                emotion_pool = mx.mean(emotion_actual, axis=0)  # (34,)
                mx.eval(emotion_pool)

            # Pitch features (mean, std, max, min)
            pitch_stats = None
            if self.pitch_head is not None:
                _, pitch_hz = self.pitch_head(encoder_out)  # (1, 1500, 1)
                pitch_actual = pitch_hz[0, :actual_encoder_frames, 0]  # (T,)
                pitch_stats = mx.array([
                    mx.mean(pitch_actual),
                    mx.std(pitch_actual),
                    mx.max(pitch_actual),
                    mx.min(pitch_actual),
                ])  # (4,)
                mx.eval(pitch_stats)

            return encoder_pool, emotion_pool, pitch_stats

        except Exception as e:
            print(f"  Warning: Failed to encode {audio_path}: {e}")
            return None

    def _compute_loss(
        self,
        encoder_pool: mx.array,
        emotion_pool: mx.array | None,
        pitch_stats: mx.array | None,
        target: int,
        class_weights: mx.array | None = None,
    ) -> tuple[float, dict]:
        """Compute cross-entropy loss and gradients."""

        mx.array([target])

        def loss_fn(model):
            # Add batch dimension
            enc = encoder_pool[None, :]
            emo = emotion_pool[None, :] if emotion_pool is not None else None
            pit = pitch_stats[None, :] if pitch_stats is not None else None

            logits = model(enc, emo, pit)  # (1, num_classes)

            # Cross-entropy loss
            log_probs = nn.log_softmax(logits, axis=-1)
            loss = -log_probs[0, target]

            # Apply class weight if provided
            if class_weights is not None:
                loss = loss * class_weights[target]

            return loss

        loss, grads = nn.value_and_grad(self.classifier, loss_fn)(self.classifier)
        mx.eval(loss)
        mx.eval(grads)

        return float(loss), grads

    def _train_batch(
        self,
        samples: list[PunctSample],
        class_weights: mx.array | None = None,
    ) -> tuple[float, int]:
        """Train on a batch of samples."""
        total_loss = 0.0
        valid_samples = 0
        accumulated_grads = None

        for sample in samples:
            result = self._encode_audio(sample.audio_path)
            if result is None:
                continue

            encoder_pool, emotion_pool, pitch_stats = result
            loss, grads = self._compute_loss(
                encoder_pool, emotion_pool, pitch_stats,
                sample.punct_class, class_weights,
            )

            if loss > 0 and not np.isnan(loss) and not np.isinf(loss) and grads:
                total_loss += loss
                valid_samples += 1

                if accumulated_grads is None:
                    accumulated_grads = grads
                else:
                    accumulated_grads = tree_map(
                        lambda a, b: a + b if isinstance(a, mx.array) and isinstance(b, mx.array) else a,
                        accumulated_grads, grads,
                    )

        if valid_samples == 0 or accumulated_grads is None:
            return 0.0, 0

        # Average gradients
        accumulated_grads = tree_map(
            lambda g: g / valid_samples if isinstance(g, mx.array) else g,
            accumulated_grads,
        )

        # Clip gradients
        flat_grads = tree_flatten(accumulated_grads)
        grad_norm = sum(float(mx.sum(g ** 2)) for _, g in flat_grads if isinstance(g, mx.array))
        grad_norm = float(np.sqrt(grad_norm))

        if grad_norm > self.config.grad_clip:
            scale = self.config.grad_clip / grad_norm
            accumulated_grads = tree_map(
                lambda g: g * scale if isinstance(g, mx.array) else g,
                accumulated_grads,
            )

        # Update parameters
        self.optimizer.update(self.classifier, accumulated_grads)
        mx.eval(self.classifier.parameters())

        return total_loss / valid_samples, valid_samples

    def _validate(self, samples: list[PunctSample]) -> tuple[float, float, dict[str, float]]:
        """
        Validate model.

        Returns:
            (average_loss, accuracy, per_class_accuracy)
        """
        # Set to eval mode to disable dropout
        self.classifier.eval()

        total_loss = 0.0
        valid_samples = 0
        correct = 0
        per_class_correct = [0] * NUM_PUNCT_CLASSES
        per_class_total = [0] * NUM_PUNCT_CLASSES

        for sample in samples[:200]:  # Limit validation samples
            result = self._encode_audio(sample.audio_path)
            if result is None:
                continue

            encoder_pool, emotion_pool, pitch_stats = result

            # Forward pass
            enc = encoder_pool[None, :]
            emo = emotion_pool[None, :] if emotion_pool is not None else None
            pit = pitch_stats[None, :] if pitch_stats is not None else None

            logits = self.classifier(enc, emo, pit)
            mx.eval(logits)

            # Loss
            log_probs = nn.log_softmax(logits, axis=-1)
            loss = -float(log_probs[0, sample.punct_class])

            if not np.isnan(loss) and not np.isinf(loss):
                total_loss += loss
                valid_samples += 1

            # Accuracy
            pred = int(mx.argmax(logits[0]))
            if pred == sample.punct_class:
                correct += 1
                per_class_correct[sample.punct_class] += 1
            per_class_total[sample.punct_class] += 1

        avg_loss = total_loss / max(valid_samples, 1)
        accuracy = correct / max(valid_samples, 1)

        per_class_acc = {}
        for i, name in enumerate(PUNCT_CLASS_NAMES):
            if per_class_total[i] > 0:
                per_class_acc[name] = per_class_correct[i] / per_class_total[i]
            else:
                per_class_acc[name] = 0.0

        # Restore training mode
        self.classifier.train()

        return avg_loss, accuracy, per_class_acc

    def _save_checkpoint(self, filename: str):
        """Save checkpoint."""
        path = self.output_dir / filename
        weights = {}

        flat_params = tree_flatten(self.classifier.parameters())
        for name, param in flat_params:
            weights[f"classifier.{name}"] = param

        mx.savez(str(path), **weights)
        self.log(f"Saved checkpoint: {path}")

    def log(self, msg: str):
        """Log message."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {msg}")

        log_file = self.output_dir / "training.log"
        with open(log_file, "a") as f:
            f.write(f"[{timestamp}] {msg}\n")

    def train(self, dataset: MELDPunctDataset):
        """Main training loop."""
        train_samples = dataset.get_train_samples()
        val_samples = dataset.get_val_samples()

        class_weights = None
        if self.config.use_class_weights:
            class_weights = dataset.get_class_weights()
            self.log(f"Class weights: {list(np.array(class_weights))}")

        self.log("Starting punctuation classifier training")
        self.log(f"Epochs: {self.config.epochs}")
        self.log(f"Train samples: {len(train_samples)}")
        self.log(f"Val samples: {len(val_samples)}")
        self.log(f"Batch size: {self.config.batch_size}")

        for epoch in range(self.config.epochs):
            self.epoch = epoch

            rng = np.random.default_rng()
            rng.shuffle(train_samples)

            total_loss = 0.0
            total_samples = 0
            batch_samples = []

            for sample in train_samples:
                batch_samples.append(sample)

                if len(batch_samples) >= self.config.batch_size:
                    loss, num_valid = self._train_batch(batch_samples, class_weights)
                    if num_valid > 0:
                        total_loss += loss * num_valid
                        total_samples += num_valid

                    batch_samples = []
                    self.step += 1

                    if self.step % self.config.log_interval == 0:
                        avg_loss = total_loss / max(total_samples, 1)
                        self.log(f"  Step {self.step}: loss={avg_loss:.4f}")

                    if self.step % self.config.save_interval == 0:
                        self._save_checkpoint(f"step_{self.step}.npz")

            # Process remaining
            if batch_samples:
                loss, num_valid = self._train_batch(batch_samples, class_weights)
                if num_valid > 0:
                    total_loss += loss * num_valid
                    total_samples += num_valid

            # Validation
            val_loss, val_acc, per_class_acc = self._validate(val_samples)

            self.log(f"Epoch {epoch + 1}/{self.config.epochs}")
            self.log(f"  Train loss: {total_loss / max(total_samples, 1):.4f}")
            self.log(f"  Val loss: {val_loss:.4f}")
            self.log(f"  Val accuracy: {val_acc:.3f}")
            for name, acc in per_class_acc.items():
                self.log(f"    {name}: {acc:.3f}")

            # Save best model
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self._save_checkpoint("best.npz")
                self.log("  New best model saved!")

            self._save_checkpoint(f"epoch_{epoch + 1}.npz")

        self.log("Training complete!")
        self.log(f"Best validation accuracy: {self.best_acc:.3f}")


# =============================================================================
# Main
# =============================================================================


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train punctuation classifier")

    parser.add_argument(
        "--meld-dir",
        type=str,
        default="data/emotion_punctuation/MELD.Raw",
        help="Path to MELD.Raw directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/punct_classifier_v1",
        help="Output directory",
    )
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
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--no-emotion",
        action="store_true",
        help="Disable emotion features",
    )
    parser.add_argument(
        "--no-pitch",
        action="store_true",
        help="Disable pitch features",
    )

    args = parser.parse_args()

    config = PunctClassifierConfig(
        meld_dir=args.meld_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_emotion=not args.no_emotion,
        use_pitch=not args.no_pitch,
    )

    dataset = MELDPunctDataset(
        meld_dir=config.meld_dir,
        split="train",
        max_audio_len=config.max_audio_len,
        val_split=config.val_split,
    )

    trainer = PunctClassifierTrainer(config)
    trainer.train(dataset)


if __name__ == "__main__":
    main()
