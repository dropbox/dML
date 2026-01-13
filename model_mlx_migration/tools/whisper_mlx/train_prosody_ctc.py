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
Train ProsodyConditionedCTC - CTC head conditioned on emotion and pitch.

This script trains the prosody projection layer of ProsodyConditionedCTC while
keeping the base CTC weights initialized from a pretrained CTC head.

Key insight: CTC can "hear" prosodic cues (rising pitch = question, emotion = emphasis)
before decoding text. By conditioning on per-frame prosody features, we can improve
punctuation prediction.

Architecture:
    Audio -> Whisper Encoder (frozen) -> encoder_out
                                          |
                                          +-> EmotionHead (frozen) -> emotion_logits
                                          +-> PitchHead (frozen) -> pitch_values
                                          |
                                          v
                              ProsodyConditionedCTC -> text_logits
                                          |
                              concat(encoder_out, prosody_emb)
                                          |
                                          v
                                      CTC Loss

Usage:
    python -m tools.whisper_mlx.train_prosody_ctc \
        --meld-dir data/emotion_punctuation/MELD.Raw \
        --ctc-weights checkpoints/ctc_english_full/step_49000.npz \
        --output-dir checkpoints/prosody_ctc_v1 \
        --epochs 10

References:
    - UNIFIED_RICH_AUDIO_ARCHITECTURE.md
    - Commit #2119: ProsodyConditionedCTC implementation
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
from mlx.utils import tree_flatten

# Import proxy loss and helpers from train_ctc
from .train_ctc import compute_ctc_loss_mlx_efficient

# PyTorch availability check (for validation)
try:
    import torch  # noqa: F401 - used for HAS_TORCH detection
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: PyTorch not available. Validation will use proxy loss.")

from .audio import load_audio, log_mel_spectrogram
from .model import WhisperMLX
from .rich_ctc_head import (
    WHISPER_VOCAB_SIZE,
    CREPEPitchHead,
    EmotionHead,
    ProsodyConditionedCTC,
)
from .tokenizer import get_whisper_tokenizer

# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ProsodyCTCConfig:
    """Configuration for ProsodyConditionedCTC training."""

    # Data
    meld_dir: str = "data/emotion_punctuation/MELD.Raw"
    output_dir: str = "checkpoints/prosody_ctc_v1"
    max_audio_len: float = 30.0  # Max audio length in seconds

    # Model
    whisper_model: str = "mlx-community/whisper-large-v3-mlx"
    d_model: int = 1280

    # CTC initialization
    ctc_weights: str = "checkpoints/ctc_english_full/step_49000.npz"
    emotion_weights: str = "checkpoints/emotion_unified_v2/best.npz"
    pitch_weights: str = "checkpoints/pitch_combined_v4/best.npz"

    # Prosody CTC architecture
    emotion_dim: int = 34  # From trained emotion head
    pitch_dim: int = 1
    prosody_dim: int = 64

    # Training
    epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 1e-4
    warmup_steps: int = 500
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # Freeze settings
    freeze_encoder: bool = True
    freeze_emotion: bool = True
    freeze_pitch: bool = True
    freeze_ctc_base: bool = False  # Fine-tune the CTC projection too

    # Logging
    log_interval: int = 50
    save_interval: int = 500
    val_split: float = 0.1


# Punctuation characters for F1 evaluation
PUNCT_CHARS = {'.', ',', '?', '!'}


# =============================================================================
# MELD Dataset
# =============================================================================


@dataclass
class MELDSample:
    """Single MELD sample."""
    audio_path: str
    transcript: str
    emotion: str
    dialogue_id: int
    utterance_id: int


class MELDDataset:
    """
    MELD Dataset loader.

    MELD (Multimodal EmotionLines Dataset) contains TV show clips with
    emotion labels and punctuated transcripts.
    """

    def __init__(
        self,
        meld_dir: str,
        split: str = "train",
        max_audio_len: float = 30.0,
        val_split: float = 0.1,
    ):
        """
        Initialize MELD dataset.

        Args:
            meld_dir: Path to MELD.Raw directory
            split: Data split (train, dev, test)
            max_audio_len: Maximum audio length in seconds
            val_split: Validation split ratio
        """
        self.meld_dir = Path(meld_dir)
        self.split = split
        self.max_audio_len = max_audio_len
        self.samples: list[MELDSample] = []

        print(f"Loading MELD dataset from: {self.meld_dir}")
        self._load_meld()

        # Split into train/val
        rng = np.random.default_rng(42)
        indices = rng.permutation(len(self.samples))
        val_size = int(len(self.samples) * val_split)

        self.val_indices = set(indices[:val_size])
        self.train_indices = set(indices[val_size:])

        print(f"Total samples: {len(self.samples)}")
        print(f"Train: {len(self.train_indices)}, Val: {len(self.val_indices)}")

    def _load_meld(self):
        """Load MELD samples from CSV and audio files."""
        # CSV file with transcripts
        csv_file = self.meld_dir / f"{self.split}_sent_emo.csv"
        if not csv_file.exists():
            print(f"WARNING: CSV not found: {csv_file}")
            return

        # Audio directory
        audio_dir = self.meld_dir / f"audio_{self.split}"
        if not audio_dir.exists():
            print(f"WARNING: Audio dir not found: {audio_dir}")
            return

        # Load CSV
        df = pd.read_csv(csv_file)
        print(f"  CSV has {len(df)} rows")

        # Build samples
        for _, row in df.iterrows():
            dialogue_id = row['Dialogue_ID']
            utterance_id = row['Utterance_ID']

            # Audio file naming: dia{dialogue_id}_utt{utterance_id}.wav
            audio_path = audio_dir / f"dia{dialogue_id}_utt{utterance_id}.wav"

            if audio_path.exists():
                transcript = str(row['Utterance'])
                # Clean transcript (remove smart quotes, etc.)
                transcript = transcript.replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"')

                self.samples.append(MELDSample(
                    audio_path=str(audio_path),
                    transcript=transcript,
                    emotion=row['Emotion'],
                    dialogue_id=dialogue_id,
                    utterance_id=utterance_id,
                ))

        print(f"  Loaded {len(self.samples)} samples with audio")

    def get_train_samples(self) -> list[MELDSample]:
        """Get training samples."""
        return [self.samples[i] for i in self.train_indices]

    def get_val_samples(self) -> list[MELDSample]:
        """Get validation samples."""
        return [self.samples[i] for i in self.val_indices]


# =============================================================================
# Punctuation Metrics
# =============================================================================


def extract_punct_positions(text: str) -> dict[str, list[int]]:
    """
    Extract positions of punctuation marks in text.

    Args:
        text: Input text with punctuation

    Returns:
        Dict mapping punct char to list of character positions
    """
    positions = {p: [] for p in PUNCT_CHARS}
    for i, char in enumerate(text):
        if char in PUNCT_CHARS:
            positions[char].append(i)
    return positions


def compute_punct_f1(
    pred_text: str,
    ref_text: str,
) -> dict[str, float]:
    """
    Compute punctuation F1 scores.

    Args:
        pred_text: Predicted text with punctuation
        ref_text: Reference text with punctuation

    Returns:
        Dict with per-class and macro F1 scores
    """
    # Normalize texts (lowercase, normalize whitespace)
    pred_text = ' '.join(pred_text.lower().split())
    ref_text = ' '.join(ref_text.lower().split())

    # Extract punctuation positions
    pred_punct = extract_punct_positions(pred_text)
    ref_punct = extract_punct_positions(ref_text)

    f1_scores = {}
    total_f1 = 0.0
    num_classes = 0

    for punct_char in PUNCT_CHARS:
        pred_set = set(pred_punct[punct_char])
        ref_set = set(ref_punct[punct_char])

        if len(ref_set) == 0 and len(pred_set) == 0:
            # No instances of this class
            continue

        # True positives: correct predictions
        tp = len(pred_set & ref_set)

        # Precision and recall
        precision = tp / len(pred_set) if len(pred_set) > 0 else 0.0
        recall = tp / len(ref_set) if len(ref_set) > 0 else 0.0

        # F1
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        f1_scores[punct_char] = f1
        total_f1 += f1
        num_classes += 1

    # Macro F1
    f1_scores['macro'] = total_f1 / max(num_classes, 1)

    return f1_scores


# =============================================================================
# Training
# =============================================================================


class ProsodyCTCTrainer:
    """Trainer for ProsodyConditionedCTC."""

    def __init__(self, config: ProsodyCTCConfig):
        """Initialize trainer."""
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.step = 0
        self.epoch = 0
        self.best_loss = float('inf')

        # Load Whisper model (encoder only)
        print(f"Loading Whisper model: {config.whisper_model}")
        self.whisper = WhisperMLX.from_pretrained(config.whisper_model)
        self.tokenizer = get_whisper_tokenizer(config.whisper_model)

        # Note: In MLX, we don't explicitly freeze parameters.
        # We just don't include them in the optimizer.
        # The encoder is automatically frozen because we only train prosody_ctc parameters.
        if config.freeze_encoder:
            print("  Encoder frozen (not included in optimizer)")

        # Initialize emotion head (for prosody features)
        self.emotion_head = EmotionHead(
            d_model=config.d_model,
            num_emotions=config.emotion_dim,
            hidden_dim=512,
        )
        if Path(config.emotion_weights).exists():
            self._load_emotion_weights(config.emotion_weights)
            print(f"  Loaded emotion weights from {config.emotion_weights}")

        # Initialize pitch head (for prosody features)
        self.pitch_head = CREPEPitchHead(
            d_model=config.d_model,
            hidden_dim=256,
            num_bins=361,
        )
        if Path(config.pitch_weights).exists():
            self._load_pitch_weights(config.pitch_weights)
            print(f"  Loaded pitch weights from {config.pitch_weights}")

        # Initialize ProsodyConditionedCTC
        if Path(config.ctc_weights).exists():
            print(f"  Initializing from CTC weights: {config.ctc_weights}")
            self.prosody_ctc = ProsodyConditionedCTC.from_ctc_head(
                config.ctc_weights,
                emotion_dim=config.emotion_dim,
                pitch_dim=config.pitch_dim,
                prosody_dim=config.prosody_dim,
            )
        else:
            print("  WARNING: No CTC weights found, using random initialization")
            self.prosody_ctc = ProsodyConditionedCTC(
                d_model=config.d_model,
                emotion_dim=config.emotion_dim,
                pitch_dim=config.pitch_dim,
                prosody_dim=config.prosody_dim,
                vocab_size=WHISPER_VOCAB_SIZE,
            )

        # Count trainable parameters
        all_params = tree_flatten(self.prosody_ctc.parameters())
        num_params = sum(p.size for _, p in all_params)
        print(f"  Trainable parameters: {num_params:,}")

        self.optimizer = optim.AdamW(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        mx.eval(self.prosody_ctc.parameters())

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

    def _encode_audio(self, audio_path: str) -> mx.array | None:
        """
        Encode audio to get Whisper encoder features.

        Args:
            audio_path: Path to audio file

        Returns:
            Encoder output (T, d_model) or None if failed
        """
        try:
            # Load and preprocess audio
            audio = load_audio(audio_path)
            if audio is None or len(audio) == 0:
                return None

            # Check duration
            duration = len(audio) / 16000
            if duration > self.config.max_audio_len:
                audio = audio[:int(self.config.max_audio_len * 16000)]

            # Compute mel spectrogram
            mel = log_mel_spectrogram(audio)
            if mel.shape[0] < 10:
                return None

            # Pad or truncate to 3000 frames (Whisper expects 30s = 3000 mel frames)
            target_frames = 3000
            if mel.shape[0] < target_frames:
                # Pad with zeros
                padding = mx.zeros((target_frames - mel.shape[0], mel.shape[1]))
                mel = mx.concatenate([mel, padding], axis=0)
            elif mel.shape[0] > target_frames:
                # Truncate
                mel = mel[:target_frames]

            # Store actual audio length for CTC (in encoder frames = mel_frames / 2)
            min(mel.shape[0], int(duration * 50))  # 50Hz encoder rate

            # Add batch dimension
            mel = mel[None, :]  # (1, T, n_mels)

            # Encode
            encoder_out = self.whisper.encoder(mel)
            mx.eval(encoder_out)

            # Return only the actual audio portion (not padding)
            # Encoder outputs 1500 frames for 3000 mel frames
            actual_encoder_frames = max(1, min(1500, int(duration * 50)))

            return encoder_out[0, :actual_encoder_frames]  # (actual_T, d_model)

        except Exception as e:
            print(f"  Warning: Failed to encode {audio_path}: {e}")
            return None

    def _tokenize(self, text: str) -> list[int]:
        """Tokenize text using Whisper tokenizer."""
        # Clean text
        text = text.strip()
        if not text:
            return []

        # Tokenize (without special tokens)
        tokens = self.tokenizer.encode(text)

        # Filter valid tokens
        return [t for t in tokens if 0 < t < WHISPER_VOCAB_SIZE]


    def _compute_loss_and_grad(
        self,
        encoder_out: mx.array,
        targets: list[int],
    ) -> tuple[float, dict]:
        """
        Compute CTC proxy loss and gradients using differentiable MLX operations.

        Uses soft monotonic alignment proxy loss that's differentiable through MLX's autograd.

        Args:
            encoder_out: (T, d_model) encoder output
            targets: List of target token IDs

        Returns:
            (loss_value, gradients)
        """
        # Get prosody features (frozen - not included in gradient computation)
        encoder_out_batched = encoder_out[None, :]  # Add batch dim (1, T, d_model)

        # Emotion features (frozen)
        emotion_logits = self.emotion_head(encoder_out_batched)  # (1, T, 34)
        mx.eval(emotion_logits)

        # Pitch features (frozen)
        _, pitch_hz = self.pitch_head(encoder_out_batched)  # (1, T, 1)
        pitch_normalized = pitch_hz / 500.0  # Rough normalization
        mx.eval(pitch_normalized)

        T = encoder_out.shape[0]
        input_lengths = [T]
        target_lengths = [len(targets)]

        # Define loss function for value_and_grad
        def loss_fn(model):
            # Forward pass
            logits = model(encoder_out_batched, emotion_logits, pitch_normalized)
            # logits shape: (1, T, vocab)

            # Use proxy loss (differentiable through MLX autograd)
            return compute_ctc_loss_mlx_efficient(
                logits,
                [targets],  # List of target sequences
                input_lengths,
                target_lengths,
                blank_id=0,
            )

        # Get loss and gradients
        loss, grads = nn.value_and_grad(self.prosody_ctc, loss_fn)(self.prosody_ctc)
        mx.eval(loss)
        mx.eval(grads)

        loss_value = float(loss)

        # Check for NaN/inf
        if np.isnan(loss_value) or np.isinf(loss_value):
            return 0.0, {}

        return loss_value, grads

    def _train_batch(self, samples: list[MELDSample]) -> float:
        """Train on a batch of samples."""
        total_loss = 0.0
        valid_samples = 0
        accumulated_grads = None

        for sample in samples:
            # Encode audio
            encoder_out = self._encode_audio(sample.audio_path)
            if encoder_out is None:
                continue

            # Tokenize transcript (with punctuation)
            targets = self._tokenize(sample.transcript)
            if len(targets) == 0:
                continue

            # Compute loss and gradients
            loss, grads = self._compute_loss_and_grad(encoder_out, targets)

            if loss > 0 and not np.isnan(loss) and not np.isinf(loss) and grads:
                total_loss += loss
                valid_samples += 1

                # Accumulate gradients using tree_map for nested dicts
                if accumulated_grads is None:
                    accumulated_grads = grads
                else:
                    def add_grads(a, b):
                        if isinstance(a, mx.array) and isinstance(b, mx.array):
                            return a + b
                        return a
                    from mlx.utils import tree_map
                    accumulated_grads = tree_map(add_grads, accumulated_grads, grads)

        if valid_samples == 0 or accumulated_grads is None:
            return 0.0

        # Average gradients using tree_map
        from mlx.utils import tree_map
        accumulated_grads = tree_map(lambda g: g / valid_samples if isinstance(g, mx.array) else g, accumulated_grads)

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
        self.optimizer.update(self.prosody_ctc, accumulated_grads)
        mx.eval(self.prosody_ctc.parameters())

        return total_loss / valid_samples

    def _validate(self, samples: list[MELDSample]) -> tuple[float, dict[str, float]]:
        """
        Validate model.

        Returns:
            (average_loss, punct_f1_scores)
        """
        # Set to eval mode to disable dropout
        self.emotion_head.eval()
        self.pitch_head.eval()
        self.prosody_ctc.eval()

        total_loss = 0.0
        valid_samples = 0
        all_pred_texts = []
        all_ref_texts = []

        for sample in samples[:100]:  # Limit validation samples
            # Encode audio
            encoder_out = self._encode_audio(sample.audio_path)
            if encoder_out is None:
                continue

            # Tokenize transcript
            targets = self._tokenize(sample.transcript)
            if len(targets) == 0:
                continue

            # Forward pass
            encoder_out = encoder_out[None, :]
            emotion_logits = self.emotion_head(encoder_out)
            _, pitch_hz = self.pitch_head(encoder_out)
            pitch_normalized = pitch_hz / 500.0

            logits = self.prosody_ctc(encoder_out, emotion_logits, pitch_normalized)
            mx.eval(logits)

            # Compute loss using proxy loss
            T = logits.shape[1]
            input_lengths = [T]
            target_lengths = [len(targets)]

            loss = compute_ctc_loss_mlx_efficient(
                logits,
                [targets],
                input_lengths,
                target_lengths,
                blank_id=0,
            )
            mx.eval(loss)
            loss_val = float(loss)

            if not np.isnan(loss_val) and not np.isinf(loss_val):
                total_loss += loss_val
                valid_samples += 1

            # CTC greedy decode
            logits_np = np.array(logits[0])
            predictions = np.argmax(logits_np, axis=-1)
            decoded_tokens = []
            prev = 0
            for t in predictions:
                if t != 0 and t != prev:
                    decoded_tokens.append(int(t))
                prev = t

            # Decode to text
            try:
                pred_text = self.tokenizer.decode(decoded_tokens)
                all_pred_texts.append(pred_text)
                all_ref_texts.append(sample.transcript)
            except Exception:
                pass

        # Compute average loss
        avg_loss = total_loss / max(valid_samples, 1)

        # Compute punctuation F1
        total_f1 = {'macro': 0.0}
        for punct in PUNCT_CHARS:
            total_f1[punct] = 0.0

        for pred, ref in zip(all_pred_texts, all_ref_texts, strict=False):
            f1_scores = compute_punct_f1(pred, ref)
            for k, v in f1_scores.items():
                total_f1[k] += v

        num_samples = max(len(all_pred_texts), 1)
        punct_f1 = {k: v / num_samples for k, v in total_f1.items()}

        # Restore training mode
        self.emotion_head.train()
        self.pitch_head.train()
        self.prosody_ctc.train()

        return avg_loss, punct_f1

    def _save_checkpoint(self, filename: str):
        """Save checkpoint."""
        path = self.output_dir / filename
        weights = {}

        # Save ProsodyConditionedCTC weights (flatten nested dict)
        flat_params = tree_flatten(self.prosody_ctc.parameters())
        for name, param in flat_params:
            weights[f"prosody_ctc.{name}"] = param

        mx.savez(str(path), **weights)
        self.log(f"Saved checkpoint: {path}")

    def log(self, msg: str):
        """Log message."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {msg}")

        # Also log to file
        log_file = self.output_dir / "training.log"
        with open(log_file, "a") as f:
            f.write(f"[{timestamp}] {msg}\n")

    def train(self, dataset: MELDDataset):
        """Main training loop."""
        train_samples = dataset.get_train_samples()
        val_samples = dataset.get_val_samples()

        self.log("Starting ProsodyConditionedCTC training")
        self.log(f"Epochs: {self.config.epochs}")
        self.log(f"Train samples: {len(train_samples)}")
        self.log(f"Val samples: {len(val_samples)}")
        self.log(f"Batch size: {self.config.batch_size}")

        for epoch in range(self.config.epochs):
            self.epoch = epoch

            # Shuffle training samples
            rng = np.random.default_rng()
            rng.shuffle(train_samples)

            # Training
            total_loss = 0.0
            num_batches = 0
            batch_samples = []

            for sample in train_samples:
                batch_samples.append(sample)

                if len(batch_samples) >= self.config.batch_size:
                    loss = self._train_batch(batch_samples)
                    if loss > 0:
                        total_loss += loss
                        num_batches += 1

                    batch_samples = []
                    self.step += 1

                    # Log progress
                    if self.step % self.config.log_interval == 0:
                        avg_loss = total_loss / max(num_batches, 1)
                        self.log(f"  Step {self.step}: loss={avg_loss:.4f}")

                    # Save checkpoint
                    if self.step % self.config.save_interval == 0:
                        self._save_checkpoint(f"step_{self.step}.npz")

            # Process remaining
            if batch_samples:
                loss = self._train_batch(batch_samples)
                if loss > 0:
                    total_loss += loss
                    num_batches += 1

            # Validation
            val_loss, punct_f1 = self._validate(val_samples)

            self.log(f"Epoch {epoch + 1}/{self.config.epochs}")
            self.log(f"  Train loss: {total_loss / max(num_batches, 1):.4f}")
            self.log(f"  Val loss: {val_loss:.4f}")
            self.log(f"  Punct F1 (macro): {punct_f1['macro']:.3f}")
            for punct_char in PUNCT_CHARS:
                if punct_char in punct_f1:
                    self.log(f"    {punct_char}: {punct_f1[punct_char]:.3f}")

            # Save best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self._save_checkpoint("best.npz")
                self.log("  New best model saved!")

            # Save epoch checkpoint
            self._save_checkpoint(f"epoch_{epoch + 1}.npz")

        self.log("Training complete!")


# =============================================================================
# Main
# =============================================================================


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train ProsodyConditionedCTC")

    parser.add_argument(
        "--meld-dir",
        type=str,
        default="data/emotion_punctuation/MELD.Raw",
        help="Path to MELD.Raw directory",
    )
    parser.add_argument(
        "--ctc-weights",
        type=str,
        default="checkpoints/ctc_english_full/step_49000.npz",
        help="Path to pretrained CTC weights",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/prosody_ctc_v1",
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
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )

    args = parser.parse_args()

    # Create config
    config = ProsodyCTCConfig(
        meld_dir=args.meld_dir,
        ctc_weights=args.ctc_weights,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    # Load dataset
    dataset = MELDDataset(
        meld_dir=config.meld_dir,
        split="train",
        max_audio_len=config.max_audio_len,
        val_split=config.val_split,
    )

    # Create trainer
    trainer = ProsodyCTCTrainer(config)

    # Train
    trainer.train(dataset)


if __name__ == "__main__":
    main()
