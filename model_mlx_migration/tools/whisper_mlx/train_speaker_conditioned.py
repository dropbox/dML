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
Speaker-Conditioned Encoder Training Script (Phase 10.4).

Trains SpeakerQueryAttention layers to inject speaker information into
the Whisper encoder for improved recognition of target speakers in
multi-speaker or noisy environments.

Architecture:
- Whisper encoder: FROZEN (use pretrained weights)
- Speaker encoder (ECAPA-TDNN): FROZEN (use pretrained weights)
- SpeakerQueryAttention layers: TRAINABLE (4 layers at blocks 8, 16, 24, 32)

Training strategy:
1. For each utterance, extract speaker embedding from enrollment audio
2. Condition encoder on speaker embedding via cross-attention
3. Use CTC loss on transcription to train speaker attention layers
4. Gated residual ensures smooth blending with original features

The model learns to focus on the target speaker's voice characteristics,
improving recognition in cocktail party scenarios.

Usage:
    python -m tools.whisper_mlx.train_speaker_conditioned \
        --data-dir data/LibriSpeech/dev-clean \
        --output-dir checkpoints/speaker_conditioned \
        --epochs 5

References:
    - SQ-Whisper (Guo et al., 2024)
    - ECAPA-TDNN (Desplanques et al., 2020)
"""

import argparse
import gc
import json
import resource
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss / 1024 / 1024


def clear_memory():
    """Aggressively clear memory between batches."""
    if hasattr(mx, 'clear_cache'):
        mx.clear_cache()
    elif hasattr(mx, 'metal') and hasattr(mx.metal, 'clear_cache'):
        mx.metal.clear_cache()
    gc.collect()


# PyTorch CTC loss bridge
try:
    import torch
    from torch.nn.functional import ctc_loss as torch_ctc_loss
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch_ctc_loss = None
    print("WARNING: PyTorch not available. Using MLX CTC loss.")

from .audio import load_audio, log_mel_spectrogram
from .ctc_head import create_ctc_draft_head
from .model import WhisperMLX
from .sota.speaker_encoder import SpeakerEncoder
from .sota.speaker_query_attention import (
    SpeakerConditionedEncoder,
    SpeakerQueryConfig,
)
from .tokenizer import get_whisper_tokenizer


@dataclass
class SpeakerConditionedConfig:
    """Configuration for speaker-conditioned encoder training."""

    # Data
    data_dir: str = "data/LibriSpeech/dev-clean"
    output_dir: str = "checkpoints/speaker_conditioned"

    # Model
    whisper_model: str = "mlx-community/whisper-large-v3-mlx"
    model_size: str = "large-v3"
    d_model: int = 1280  # large-v3 dimension
    speaker_dim: int = 192  # ECAPA-TDNN embedding dimension
    speaker_model_path: str = "models/ecapa-spkver-mlx"

    # Speaker query attention
    injection_layers: tuple[int, ...] = (8, 16, 24, 32)
    n_attention_heads: int = 8
    gate_init_bias: float = -2.0  # Start with minimal speaker influence

    # Training
    epochs: int = 5
    batch_size: int = 2  # Small due to memory constraints
    gradient_accumulation_steps: int = 8  # Effective batch = 16
    learning_rate: float = 1e-4  # Lower LR for attention layers
    warmup_steps: int = 200
    grad_clip: float = 1.0
    weight_decay: float = 0.01
    dropout: float = 0.1

    # CTC
    blank_id: int = 0
    max_audio_len: float = 15.0

    # Logging
    log_interval: int = 50
    save_interval: int = 500
    eval_interval: int = 200
    val_split: float = 0.1

    # Speaker enrollment
    enrollment_seconds: float = 3.0  # Seconds of audio for speaker enrollment


@dataclass
class SpeakerSample:
    """Audio sample with speaker information."""

    audio_path: str
    transcript: str
    speaker_id: str
    language: str = "en"
    duration: float = 0.0


class SpeakerDataset:
    """Dataset loader for speaker-labeled audio (LibriSpeech format).

    LibriSpeech structure:
        split/speaker_id/chapter_id/speaker-chapter-utterance.flac
    """

    def __init__(
        self,
        data_dir: str,
        max_audio_len: float = 15.0,
        val_split: float = 0.1,
    ):
        self.data_dir = Path(data_dir)
        self.max_audio_len = max_audio_len
        self.samples: list[SpeakerSample] = []
        self.speaker_to_samples: dict[str, list[int]] = {}

        print(f"Loading speaker-labeled data from: {self.data_dir}")
        self._load_data()

        # Split into train/val (by speaker to avoid data leakage)
        speakers = list(self.speaker_to_samples.keys())
        rng = np.random.default_rng(42)
        rng.shuffle(speakers)

        val_speaker_count = max(1, int(len(speakers) * val_split))
        self.val_speakers = set(speakers[:val_speaker_count])
        self.train_speakers = set(speakers[val_speaker_count:])

        self.val_indices = {
            idx for spk in self.val_speakers
            for idx in self.speaker_to_samples[spk]
        }
        self.train_indices = {
            idx for spk in self.train_speakers
            for idx in self.speaker_to_samples[spk]
        }

        print(f"Total samples: {len(self.samples)}")
        print(f"Speakers: {len(speakers)} (train: {len(self.train_speakers)}, val: {len(self.val_speakers)})")
        print(f"Train: {len(self.train_indices)}, Val: {len(self.val_indices)}")

    def _load_data(self):
        """Load LibriSpeech format data with speaker IDs."""
        # Find all transcript files
        trans_files = list(self.data_dir.rglob("*.trans.txt"))

        if not trans_files:
            # Check if this is a symlinked/combined directory
            trans_files = list(self.data_dir.rglob("*.txt"))
            trans_files = [f for f in trans_files if f.name.endswith(".trans.txt")]

        if not trans_files:
            raise ValueError(f"No transcript files found in {self.data_dir}")

        for trans_path in sorted(trans_files):
            # Parse transcript file
            with open(trans_path) as f:
                for line in f:
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) != 2:
                        continue

                    utterance_id, transcript = parts

                    # Extract speaker ID from utterance ID (format: speaker-chapter-utterance)
                    speaker_id = utterance_id.split("-")[0]

                    # Find audio file
                    audio_dir = trans_path.parent
                    audio_path = None
                    for ext in [".flac", ".wav", ".mp3", ".ogg"]:
                        candidate = audio_dir / f"{utterance_id}{ext}"
                        if candidate.exists():
                            audio_path = str(candidate)
                            break

                    if audio_path is None:
                        continue

                    sample = SpeakerSample(
                        audio_path=audio_path,
                        transcript=transcript.lower(),
                        speaker_id=speaker_id,
                    )

                    sample_idx = len(self.samples)
                    self.samples.append(sample)

                    # Track speaker -> sample mapping
                    if speaker_id not in self.speaker_to_samples:
                        self.speaker_to_samples[speaker_id] = []
                    self.speaker_to_samples[speaker_id].append(sample_idx)

    def get_speaker_enrollment(self, speaker_id: str, exclude_idx: int | None = None) -> str | None:
        """Get an audio file for speaker enrollment (different from current sample).

        Args:
            speaker_id: Speaker to get enrollment for
            exclude_idx: Sample index to exclude (the current training sample)

        Returns:
            Path to enrollment audio, or None if not available
        """
        if speaker_id not in self.speaker_to_samples:
            return None

        candidates = [
            idx for idx in self.speaker_to_samples[speaker_id]
            if idx != exclude_idx
        ]

        if not candidates:
            return None

        # Pick a random enrollment sample
        rng = np.random.default_rng()
        enrollment_idx = rng.choice(candidates)
        return self.samples[enrollment_idx].audio_path

    def get_train_samples(self) -> list[tuple[int, SpeakerSample]]:
        """Get training samples with indices."""
        return [(idx, self.samples[idx]) for idx in sorted(self.train_indices)]

    def get_val_samples(self) -> list[tuple[int, SpeakerSample]]:
        """Get validation samples with indices."""
        return [(idx, self.samples[idx]) for idx in sorted(self.val_indices)]


class SpeakerConditionedTrainer:
    """Trainer for speaker-conditioned encoder."""

    def __init__(self, config: SpeakerConditionedConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize logging
        self.log_file = self.output_dir / "training.log"
        self.log("Starting speaker-conditioned encoder training")
        self.log(f"Output directory: {self.output_dir}")

        # Load tokenizer
        self.tokenizer = get_whisper_tokenizer(config.model_size)

        # Load Whisper model
        print("1. Loading Whisper model (encoder will be frozen)...")
        self.whisper = WhisperMLX.from_pretrained(config.whisper_model)

        # Create speaker-conditioned encoder
        print("2. Creating speaker-conditioned encoder...")
        speaker_config = SpeakerQueryConfig(
            d_model=config.d_model,
            speaker_dim=config.speaker_dim,
            n_heads=config.n_attention_heads,
            injection_layers=config.injection_layers,
            dropout=config.dropout,
            gate_init_bias=config.gate_init_bias,
        )
        self.conditioned_encoder = SpeakerConditionedEncoder(
            self.whisper.encoder, speaker_config,
        )
        self.conditioned_encoder.freeze_encoder()

        # Load speaker encoder
        print("3. Loading speaker encoder (ECAPA-TDNN)...")
        self.speaker_encoder = SpeakerEncoder.from_pretrained(config.speaker_model_path)
        self.speaker_encoder.freeze()

        # Create CTC head
        print("4. Creating CTC head...")
        self.ctc_head = create_ctc_draft_head(
            model_size=config.model_size,
            use_layer_norm=True,
        )

        # Set up optimizer (only for trainable parameters)
        trainable_params = self._get_trainable_params()
        n_trainable = self._count_params(trainable_params)
        print(f"5. Trainable parameters: {n_trainable:,}")
        self.log(f"Trainable parameters: {n_trainable:,}")

        # AdamW optimizer with weight decay
        self.optimizer = optim.AdamW(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Training state
        self.step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        self.accumulated_grads = None
        self.accumulated_count = 0

    def _get_trainable_params(self):
        """Get trainable parameters (speaker query layers + CTC head)."""
        params = {}
        # Speaker query attention layers
        for layer_idx, layer in self.conditioned_encoder.speaker_query_layers.items():
            for name, param in layer.parameters().items():
                params[f"speaker_query_{layer_idx}.{name}"] = param
        # CTC head
        for name, param in self.ctc_head.parameters().items():
            params[f"ctc_head.{name}"] = param
        return params

    def _count_params(self, params: dict) -> int:
        """Count total parameters in a nested dict."""
        total = 0
        for v in params.values():
            if isinstance(v, dict):
                total += self._count_params(v)
            else:
                total += v.size
        return total

    def log(self, message: str):
        """Log message to file and stdout."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"{timestamp} | {message}"
        print(log_line)
        with open(self.log_file, "a") as f:
            f.write(log_line + "\n")

    def _extract_speaker_embedding(self, audio_path: str) -> mx.array:
        """Extract speaker embedding from audio file."""
        # Load audio and compute mel spectrogram
        audio = load_audio(audio_path)

        # Use first N seconds for enrollment
        max_samples = int(self.config.enrollment_seconds * 16000)
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        # ECAPA-TDNN uses 80 mels (not 128 like Whisper large-v3)
        mel = log_mel_spectrogram(audio, n_mels=80, padding=0)

        # Extract speaker embedding
        if mel.ndim == 2:
            mel = mel[None]  # Add batch dimension

        embedding = self.speaker_encoder.encode(mel, normalize=True)
        return embedding[0]  # Remove batch dimension

    def _ctc_loss_pytorch(
        self,
        log_probs: mx.array,
        targets: mx.array,
        input_lengths: mx.array,
        target_lengths: mx.array,
    ) -> mx.array:
        """Compute CTC loss using PyTorch for numerical stability."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for CTC loss")

        # Convert MLX -> NumPy -> PyTorch
        log_probs_np = np.array(log_probs)
        targets_np = np.array(targets)
        input_lengths_np = np.array(input_lengths)
        target_lengths_np = np.array(target_lengths)

        log_probs_pt = torch.tensor(log_probs_np, requires_grad=False)
        targets_pt = torch.tensor(targets_np, dtype=torch.long)
        input_lengths_pt = torch.tensor(input_lengths_np, dtype=torch.long)
        target_lengths_pt = torch.tensor(target_lengths_np, dtype=torch.long)

        # CTC loss expects (T, N, C) format
        log_probs_pt = log_probs_pt.transpose(0, 1)

        loss = torch_ctc_loss(
            log_probs_pt,
            targets_pt,
            input_lengths_pt,
            target_lengths_pt,
            blank=self.config.blank_id,
            reduction='mean',
            zero_infinity=True,
        )

        return mx.array(loss.item())

    def _forward_step(
        self,
        mel: mx.array,
        speaker_embedding: mx.array,
        targets: mx.array,
        input_lengths: mx.array,
        target_lengths: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Forward pass for speaker-conditioned encoding.

        Returns:
            (loss, log_probs) tuple
        """
        # Encode with speaker conditioning
        encoder_output = self.conditioned_encoder(mel, speaker_embedding)

        # CTC head
        logits = self.ctc_head(encoder_output)  # (B, T, vocab)
        log_probs = mx.log_softmax(logits, axis=-1)

        # Compute loss
        loss = self._ctc_loss_pytorch(log_probs, targets, input_lengths, target_lengths)

        return loss, log_probs

    def train_step(
        self,
        mel: mx.array,
        speaker_embedding: mx.array,
        targets: mx.array,
        input_lengths: mx.array,
        target_lengths: mx.array,
    ) -> float:
        """Single training step with gradient accumulation."""

        def loss_fn(params):
            # Temporarily apply params to model
            # (In production, use proper param passing)
            encoder_output = self.conditioned_encoder(mel, speaker_embedding)
            logits = self.ctc_head(encoder_output)
            log_probs = mx.log_softmax(logits, axis=-1)

            # CTC loss
            return self._ctc_loss_pytorch(
                log_probs, targets, input_lengths, target_lengths,
            )

        # Get trainable params
        params = self._get_trainable_params()

        # Compute loss and gradients
        loss, grads = mx.value_and_grad(loss_fn)(params)
        mx.eval(loss, grads)

        # Accumulate gradients
        if self.accumulated_grads is None:
            self.accumulated_grads = grads
        else:
            self.accumulated_grads = mx.tree_map(
                lambda a, g: a + g, self.accumulated_grads, grads,
            )
        self.accumulated_count += 1

        # Apply gradients if accumulation complete
        if self.accumulated_count >= self.config.gradient_accumulation_steps:
            # Average gradients
            avg_grads = mx.tree_map(
                lambda g: g / self.accumulated_count, self.accumulated_grads,
            )

            # Clip gradients
            grad_norm = mx.sqrt(sum(
                mx.sum(g * g) for g in mx.utils.tree_flatten(avg_grads)
            ))
            if grad_norm > self.config.grad_clip:
                scale = self.config.grad_clip / (grad_norm + 1e-8)
                avg_grads = mx.tree_map(lambda g: g * scale, avg_grads)

            # Update parameters
            self.optimizer.update(params, avg_grads)
            mx.eval(params)

            # Reset accumulation
            self.accumulated_grads = None
            self.accumulated_count = 0
            self.step += 1

        return float(loss)

    def train(self, dataset: SpeakerDataset):
        """Train speaker-conditioned encoder."""
        self.log(f"Starting training for {self.config.epochs} epochs")
        self.log(f"Train samples: {len(dataset.train_indices)}")
        self.log(f"Val samples: {len(dataset.val_indices)}")
        self.log(f"Batch size: {self.config.batch_size}")
        self.log(f"Gradient accumulation: {self.config.gradient_accumulation_steps}")
        self.log(f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")

        for epoch in range(self.config.epochs):
            self.epoch = epoch
            self.log(f"\n=== Epoch {epoch + 1}/{self.config.epochs} ===")

            # Get training samples and shuffle
            train_data = dataset.get_train_samples()
            rng = np.random.default_rng()
            rng.shuffle(train_data)

            running_loss = 0.0
            n_samples = 0

            for idx, sample in train_data:
                # Skip if audio too long
                try:
                    audio = load_audio(sample.audio_path)
                except Exception as e:
                    print(f"  Skip {sample.audio_path}: {e}")
                    continue

                duration = len(audio) / 16000
                if duration > self.config.max_audio_len:
                    continue

                # Get enrollment audio for speaker
                enrollment_path = dataset.get_speaker_enrollment(
                    sample.speaker_id, exclude_idx=idx,
                )
                if enrollment_path is None:
                    # Use same audio for enrollment (not ideal but works)
                    enrollment_path = sample.audio_path

                try:
                    # Extract speaker embedding
                    speaker_emb = self._extract_speaker_embedding(enrollment_path)

                    # Compute mel spectrogram
                    mel = log_mel_spectrogram(audio, n_mels=128)
                    if mel.ndim == 2:
                        mel = mel[None]

                    # Tokenize transcript
                    tokens = self.tokenizer.encode(sample.transcript)
                    if len(tokens) == 0:
                        continue

                    targets = mx.array([tokens], dtype=mx.int32)
                    input_lengths = mx.array([mel.shape[1] // 2], dtype=mx.int32)
                    target_lengths = mx.array([len(tokens)], dtype=mx.int32)

                    # Training step
                    loss = self.train_step(
                        mel, speaker_emb[None], targets, input_lengths, target_lengths,
                    )

                    running_loss += loss
                    n_samples += 1

                    # Log progress
                    if n_samples % self.config.log_interval == 0:
                        avg_loss = running_loss / n_samples
                        mem_mb = get_memory_usage_mb()
                        self.log(f"  Step {self.step}: loss={avg_loss:.4f}, mem={mem_mb:.0f}MB")

                    # Save checkpoint
                    if self.step > 0 and self.step % self.config.save_interval == 0:
                        self._save_checkpoint()

                    # Clear memory
                    clear_memory()

                except Exception as e:
                    print(f"  Error processing {sample.audio_path}: {e}")
                    continue

            # End of epoch
            if n_samples > 0:
                epoch_loss = running_loss / n_samples
                self.log(f"Epoch {epoch + 1} complete: avg_loss={epoch_loss:.4f}")

                # Validation
                val_loss = self._validate(dataset)
                self.log(f"Validation loss: {val_loss:.4f}")

                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self._save_checkpoint(is_best=True)

        self.log("Training complete!")
        self._save_checkpoint(final=True)

    def _validate(self, dataset: SpeakerDataset) -> float:
        """Run validation."""
        # Set to eval mode to disable dropout
        self.ctc_head.eval()

        val_data = dataset.get_val_samples()
        total_loss = 0.0
        n_samples = 0

        for idx, sample in val_data[:100]:  # Limit validation samples
            try:
                audio = load_audio(sample.audio_path)
                duration = len(audio) / 16000
                if duration > self.config.max_audio_len:
                    continue

                enrollment_path = dataset.get_speaker_enrollment(
                    sample.speaker_id, exclude_idx=idx,
                ) or sample.audio_path

                speaker_emb = self._extract_speaker_embedding(enrollment_path)
                mel = log_mel_spectrogram(audio, n_mels=128)
                if mel.ndim == 2:
                    mel = mel[None]

                tokens = self.tokenizer.encode(sample.transcript)
                if len(tokens) == 0:
                    continue

                targets = mx.array([tokens], dtype=mx.int32)
                input_lengths = mx.array([mel.shape[1] // 2], dtype=mx.int32)
                target_lengths = mx.array([len(tokens)], dtype=mx.int32)

                loss, _ = self._forward_step(
                    mel, speaker_emb[None], targets, input_lengths, target_lengths,
                )

                total_loss += float(loss)
                n_samples += 1

            except Exception:
                continue

        # Restore training mode
        self.ctc_head.train()

        return total_loss / max(1, n_samples)

    def _save_checkpoint(self, is_best: bool = False, final: bool = False):
        """Save model checkpoint."""
        if final:
            path = self.output_dir / "final"
        elif is_best:
            path = self.output_dir / "best"
        else:
            path = self.output_dir / f"step_{self.step}"

        path.mkdir(parents=True, exist_ok=True)

        # Save speaker query weights
        self.conditioned_encoder.save_speaker_query_weights(str(path))

        # Save CTC head
        mx.savez(str(path / "ctc_head.npz"), **dict(self.ctc_head.parameters()))

        # Save training state
        state = {
            "step": self.step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "config": {
                "d_model": self.config.d_model,
                "speaker_dim": self.config.speaker_dim,
                "injection_layers": list(self.config.injection_layers),
            },
        }
        with open(path / "training_state.json", "w") as f:
            json.dump(state, f, indent=2)

        self.log(f"Checkpoint saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Train speaker-conditioned encoder")

    # Data arguments
    parser.add_argument("--data-dir", type=str, default="data/LibriSpeech/dev-clean",
                        help="Path to LibriSpeech-format data")
    parser.add_argument("--output-dir", type=str, default="checkpoints/speaker_conditioned",
                        help="Output directory for checkpoints")

    # Model arguments
    parser.add_argument("--whisper-model", type=str, default="mlx-community/whisper-large-v3-mlx",
                        help="Whisper model to use")
    parser.add_argument("--speaker-model", type=str, default="models/ecapa-spkver-mlx",
                        help="Speaker encoder model path")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)

    # Logging
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=500)

    args = parser.parse_args()

    # Create config
    config = SpeakerConditionedConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        whisper_model=args.whisper_model,
        speaker_model_path=args.speaker_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
    )

    # Load dataset
    print("\n" + "=" * 60)
    print("Speaker-Conditioned Encoder Training")
    print("=" * 60)

    dataset = SpeakerDataset(
        data_dir=config.data_dir,
        max_audio_len=config.max_audio_len,
        val_split=config.val_split,
    )

    # Create trainer
    trainer = SpeakerConditionedTrainer(config)

    # Train
    trainer.train(dataset)


if __name__ == "__main__":
    main()
