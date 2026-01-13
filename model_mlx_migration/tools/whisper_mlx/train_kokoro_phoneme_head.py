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
Kokoro Phoneme Head Training Script.

Trains a small CTC head to predict Kokoro phoneme tokens from Whisper encoder output.
This enables fast, voice-agnostic transcript verification by comparing predicted
phonemes with phonemized text (edit distance comparison).

Key insight: Both Whisper encoder and Kokoro phonemizer are voice-agnostic.
By bridging them with a learned head, we can verify transcripts in phoneme space.

Architecture:
    Audio → Whisper Encoder (frozen) → Kokoro Phoneme Head → Phoneme Tokens
                                            ↓
    Text → Kokoro Phonemizer → Phoneme Tokens
                                            ↓
                            Compare with Edit Distance

Usage:
    # Quick test on dev-clean
    python -m tools.whisper_mlx.train_kokoro_phoneme_head \
        --data-dir data/LibriSpeech/dev-clean \
        --output-dir checkpoints/kokoro_phoneme_head \
        --epochs 5

    # Full training on train-clean-100
    python -m tools.whisper_mlx.train_kokoro_phoneme_head \
        --data-dir data/LibriSpeech/train-clean-100 \
        --output-dir checkpoints/kokoro_phoneme_head \
        --epochs 10 \
        --batch-size 8

References:
    - kokoro_phoneme_head.py: Model architecture
    - ROUNDTRIP_VERIFICATION_FINDINGS_2025-12-28.md: Research context
"""

import argparse
import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None
    nn = None
    optim = None

# PyTorch CTC loss bridge (numerically stable)
try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: PyTorch not available. Training requires PyTorch for CTC loss.")

from .audio import get_audio_duration, load_audio, log_mel_spectrogram
from .encoder_cache import TrainingEncoderCache
from .kokoro_phoneme_head import create_kokoro_phoneme_head
from .model import WhisperMLX

# =============================================================================
# Configuration
# =============================================================================


@dataclass
class TrainingConfig:
    """Configuration for Kokoro Phoneme Head training."""

    # Data
    data_dir: str = "data/LibriSpeech/dev-clean"
    output_dir: str = "checkpoints/kokoro_phoneme_head"

    # Model
    whisper_model: str = "large-v3"  # Use large-v3 for best encoder quality
    hidden_dim: int = 512  # Hidden layer dimension for phoneme head
    dropout: float = 0.0  # Dropout rate for regularization (0.1-0.2 recommended)

    # Training
    epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 1e-3  # Higher LR for small head
    warmup_steps: int = 100
    grad_clip: float = 1.0
    weight_decay: float = 0.01

    # CTC
    blank_id: int = 0  # CTC blank token (same as Kokoro PAD)
    max_audio_len: float = 15.0  # Max 15 seconds per sample

    # Logging
    log_interval: int = 20
    save_interval: int = 500
    eval_interval: int = 100

    # Validation
    val_split: float = 0.1

    # Encoder caching (~3x speedup by caching frozen encoder outputs)
    encoder_cache_dir: str | None = None

    # Length-sorted batching (~1.3x speedup by reducing padding waste)
    length_sorted_batching: bool = False
    bucket_size_multiplier: int = 100  # Bucket size = batch_size * multiplier

    # Resume from checkpoint
    resume_checkpoint: str | None = None
    resume_step: int = 0


# =============================================================================
# Dataset
# =============================================================================


@dataclass
class AudioSample:
    """Single audio sample for training."""
    audio_path: str
    transcript: str
    language: str = "en"
    duration: float = 0.0  # Audio duration in seconds (for length-sorted batching)


class LibriSpeechDataset:
    """
    Dataset loader for LibriSpeech format.

    LibriSpeech structure:
        dev-clean/
            speaker_id/
                chapter_id/
                    speaker-chapter-utterance.flac
                    speaker-chapter.trans.txt
    """

    def __init__(
        self,
        data_dir: str,
        max_audio_len: float = 15.0,
        val_split: float = 0.1,
    ):
        self.data_dir = Path(data_dir)
        self.max_audio_len = max_audio_len
        self.samples: list[AudioSample] = []

        print(f"Loading LibriSpeech from: {self.data_dir}")
        self._load_librispeech()

        # Split into train/val
        rng = np.random.default_rng(42)
        indices = rng.permutation(len(self.samples))
        val_size = int(len(self.samples) * val_split)

        self.val_indices = set(indices[:val_size])
        self.train_indices = set(indices[val_size:])

        print(f"Total samples: {len(self.samples)}")
        print(f"Train: {len(self.train_indices)}, Val: {len(self.val_indices)}")

    def _load_librispeech(self):
        """Load LibriSpeech format dataset."""
        import glob as glob_module
        trans_pattern = str(self.data_dir / "**" / "*.trans.txt")
        trans_files = [Path(p) for p in glob_module.glob(trans_pattern, recursive=True)]

        print(f"Found {len(trans_files)} transcript files")

        for trans_file in trans_files:
            self._load_transcript_file(trans_file)

    def _load_transcript_file(self, trans_file: Path):
        """Load samples from a transcript file."""
        chapter_dir = trans_file.parent

        with open(trans_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(" ", 1)
                if len(parts) < 2:
                    continue

                utterance_id = parts[0]
                transcript = parts[1]

                audio_path = chapter_dir / f"{utterance_id}.flac"
                if not audio_path.exists():
                    audio_path = chapter_dir / f"{utterance_id}.wav"

                if audio_path.exists():
                    self.samples.append(AudioSample(
                        audio_path=str(audio_path),
                        transcript=transcript,
                        language="en",
                    ))

    def get_train_samples(self) -> list[AudioSample]:
        return [self.samples[i] for i in self.train_indices]

    def get_val_samples(self) -> list[AudioSample]:
        return [self.samples[i] for i in self.val_indices]

    def __len__(self) -> int:
        return len(self.samples)


# =============================================================================
# CTC Loss (via PyTorch bridge)
# =============================================================================


def ctc_loss_pytorch(
    logits: mx.array,
    targets: list[list[int]],
    input_lengths: list[int],
    target_lengths: list[int],
    blank_id: int = 0,
) -> tuple[float, np.ndarray]:
    """
    Compute CTC loss using PyTorch (numerically stable).

    Args:
        logits: (batch, T, vocab) log probabilities from MLX
        targets: List of target token sequences
        input_lengths: Length of each input sequence
        target_lengths: Length of each target sequence
        blank_id: Blank token ID

    Returns:
        Tuple of (loss_value, gradient_array)
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch required for CTC loss")

    # Convert MLX to numpy for PyTorch bridge
    logits_np = np.array(logits)
    batch_size, T, vocab_size = logits_np.shape

    # Convert to PyTorch tensors with gradient tracking
    logits_pt = torch.tensor(logits_np, dtype=torch.float32, requires_grad=True)

    # Log softmax for CTC (expects log probabilities)
    log_probs = F.log_softmax(logits_pt, dim=-1)

    # Transpose to (T, batch, vocab) as expected by CTC loss
    log_probs = log_probs.transpose(0, 1)

    # Pad targets to same length
    max_target_len = max(target_lengths) if target_lengths else 1
    targets_padded = np.zeros((batch_size, max_target_len), dtype=np.int64)
    for i, tgt in enumerate(targets):
        targets_padded[i, :len(tgt)] = tgt

    targets_pt = torch.tensor(targets_padded, dtype=torch.long)
    input_lengths_pt = torch.tensor(input_lengths, dtype=torch.long)
    target_lengths_pt = torch.tensor(target_lengths, dtype=torch.long)

    # Compute CTC loss
    loss = F.ctc_loss(
        log_probs,
        targets_pt,
        input_lengths_pt,
        target_lengths_pt,
        blank=blank_id,
        reduction='mean',
        zero_infinity=True,
    )

    # Backward pass for gradients
    loss.backward()

    # Return loss value and gradient
    loss_value = loss.item()
    grad_np = logits_pt.grad.numpy()

    return loss_value, grad_np


# =============================================================================
# Trainer
# =============================================================================


class KokoroPhonemeTrainer:
    """Trainer for Kokoro Phoneme Head."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load Whisper model (encoder only)
        print(f"Loading Whisper encoder: {config.whisper_model}")
        self.whisper = WhisperMLX.from_pretrained(
            config.whisper_model,
            warmup=True,
        )
        # Freeze encoder
        self.whisper.freeze()
        print(f"  Encoder dimension: {self.whisper.config.n_audio_state}")

        # Get encoder dimension
        d_model = self.whisper.config.n_audio_state

        # Create phoneme head
        print(f"Creating Kokoro Phoneme Head (d_model={d_model}, hidden={config.hidden_dim}, dropout={config.dropout})")
        self.head = create_kokoro_phoneme_head(
            model_size=config.whisper_model,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )
        mx.eval(self.head.parameters())

        # Resume from checkpoint if provided
        if config.resume_checkpoint:
            print(f"Loading checkpoint: {config.resume_checkpoint}")
            weights = mx.load(config.resume_checkpoint)
            self.head.load_weights(list(weights.items()))
            mx.eval(self.head.parameters())
            print("  Loaded weights from checkpoint")

        # Load Kokoro phonemizer
        self._init_phonemizer()

        # Optimizer
        self.optimizer = optim.AdamW(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Training state
        self.step = config.resume_step
        self.best_val_loss = float('inf')

        # Encoder caching (~3x speedup by caching frozen encoder outputs)
        self.encoder_cache = None
        if config.encoder_cache_dir:
            self.encoder_cache = TrainingEncoderCache(
                cache_dir=config.encoder_cache_dir,
                use_compression=True,
            )
            print(f"  Encoder cache: {config.encoder_cache_dir}")

    def _init_phonemizer(self):
        """Initialize Kokoro phonemizer."""
        try:
            from tools.pytorch_to_mlx.converters.models.kokoro_phonemizer import (
                PAD_TOKEN,
                get_vocab_size,
                phonemize_text,
            )
            self.phonemize_text = phonemize_text
            self.phoneme_vocab_size = get_vocab_size()
            self.pad_token = PAD_TOKEN
            print(f"  Kokoro phoneme vocab size: {self.phoneme_vocab_size}")
        except ImportError as e:
            raise RuntimeError(f"Kokoro phonemizer not available: {e}") from e

    def _populate_sample_durations(self, samples: list[AudioSample]) -> None:
        """
        Populate duration field for samples that have duration=0.

        Uses soundfile.info() for fast metadata-only access.
        Modifies samples in-place.
        """
        samples_needing_duration = [s for s in samples if s.duration == 0.0]
        if not samples_needing_duration:
            return

        print(f"Populating durations for {len(samples_needing_duration)} samples...")
        start_time = time.time()

        for i, sample in enumerate(samples_needing_duration):
            sample.duration = get_audio_duration(sample.audio_path)
            if (i + 1) % 10000 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                print(f"  {i + 1}/{len(samples_needing_duration)} ({rate:.0f} samples/sec)")

        elapsed = time.time() - start_time
        print(f"Duration population complete in {elapsed:.1f}s")

    def _create_length_sorted_batches(
        self, samples: list[AudioSample],
    ) -> list[list[AudioSample]]:
        """
        Create batches of similar-length samples.

        Strategy:
        1. Sort all samples by duration
        2. Divide into buckets of size (batch_size * bucket_size_multiplier)
        3. Shuffle samples within each bucket
        4. Create batches from each bucket
        5. Shuffle the batches

        This reduces padding waste while maintaining some randomness.
        Expected speedup: 1.2-1.3x (less computation wasted on padding).

        Args:
            samples: List of AudioSample (must have duration populated)

        Returns:
            List of batches, each batch is a list of AudioSample
        """
        # Sort by duration (shortest first reduces peak memory)
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
                batch = bucket[i:i + batch_size]
                if batch:
                    batches.append(batch)

        # Shuffle batches for training randomness
        rng.shuffle(batches)

        return batches

    def _get_mel(self, audio_path: str) -> mx.array:
        """Load audio and compute mel spectrogram."""
        audio = load_audio(audio_path)

        # Trim to max length
        max_samples = int(self.config.max_audio_len * 16000)
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        # Use correct n_mels for this model (80 for small/medium, 128 for large-v3)
        n_mels = self.whisper.config.n_mels
        return log_mel_spectrogram(audio, n_mels=n_mels)

    def _get_encoder_output(self, mel: mx.array) -> mx.array:
        """Get Whisper encoder output (frozen)."""
        if mel.ndim == 2:
            mel = mel[None, :]  # Add batch dimension

        encoder_output = self.whisper.encoder(mel, variable_length=True)
        mx.eval(encoder_output)
        return encoder_output

    def _get_encoder_output_cached(
        self, mel: mx.array, audio_path: str,
    ) -> tuple[mx.array, int]:
        """
        Get Whisper encoder output with caching support.

        Returns:
            Tuple of (encoder_output, actual_frames) where encoder_output has
            batch dimension removed.
        """
        # Check cache first
        if self.encoder_cache is not None:
            cached = self.encoder_cache.load(audio_path)
            if cached is not None:
                enc_out, actual_frames = cached
                return enc_out, actual_frames

        # Cache miss - compute encoder output
        if mel.ndim == 2:
            mel = mel[None, :]  # Add batch dimension

        encoder_output = self.whisper.encoder(mel, variable_length=True)
        mx.eval(encoder_output)
        enc_out = encoder_output[0]  # Remove batch dim

        # Actual frames = mel frames // 2 (Whisper encoder downsamples 2x)
        actual_frames = mel.shape[1] // 2

        # Save to cache
        if self.encoder_cache is not None:
            self.encoder_cache.save(audio_path, enc_out, actual_frames)

        return enc_out, actual_frames

    def _get_phoneme_tokens(self, text: str, language: str = "en") -> list[int]:
        """Get Kokoro phoneme tokens for text."""
        _, token_ids = self.phonemize_text(text, language=language)
        # Remove PAD tokens at start/end for CTC (they're BOS/EOS markers)
        # Keep only the actual phoneme tokens
        return [t for t in token_ids if t != self.pad_token]

    def train_step(
        self,
        batch_mels: list[mx.array],
        batch_transcripts: list[str],
        batch_audio_paths: list[str] | None = None,
    ) -> float:
        """
        Single training step with manual gradient computation.

        Uses PyTorch CTC loss for numerical stability, then manually
        computes gradients through the head using chain rule.
        This approach is needed because MLX doesn't have native CTC loss.

        Args:
            batch_mels: List of mel spectrograms
            batch_transcripts: List of text transcripts
            batch_audio_paths: Optional list of audio paths for encoder caching
        """
        # Get encoder outputs (with caching if audio_paths provided)
        encoder_outputs = []
        input_lengths = []

        if batch_audio_paths is not None and self.encoder_cache is not None:
            # Use cached encoder outputs
            for mel, audio_path in zip(batch_mels, batch_audio_paths, strict=False):
                enc_out, actual_frames = self._get_encoder_output_cached(mel, audio_path)
                encoder_outputs.append(enc_out)
                input_lengths.append(enc_out.shape[0])
        else:
            # Original path without caching
            for mel in batch_mels:
                enc_out = self._get_encoder_output(mel)
                encoder_outputs.append(enc_out[0])  # Remove batch dim
                input_lengths.append(enc_out.shape[1])

        # Pad encoder outputs to same length
        max_len = max(input_lengths)
        padded_outputs = []
        for enc_out in encoder_outputs:
            T = enc_out.shape[0]
            if T < max_len:
                pad = mx.zeros((max_len - T, enc_out.shape[1]), dtype=enc_out.dtype)
                enc_out = mx.concatenate([enc_out, pad], axis=0)
            padded_outputs.append(enc_out)

        # Stack to batch
        encoder_batch = mx.stack(padded_outputs, axis=0)
        mx.eval(encoder_batch)

        # Get phoneme targets
        targets = []
        target_lengths = []
        for transcript in batch_transcripts:
            tokens = self._get_phoneme_tokens(transcript)
            targets.append(tokens)
            target_lengths.append(len(tokens))

        # Forward pass through head (training=True enables dropout)
        logits = self.head(encoder_batch, training=True)
        mx.eval(logits)

        # CTC loss via PyTorch
        loss_value, grad_np = ctc_loss_pytorch(
            logits,
            targets,
            input_lengths,
            target_lengths,
            blank_id=self.config.blank_id,
        )

        if not np.isfinite(loss_value):
            return 0.0

        # Convert gradient back to MLX
        logits_grad = mx.array(grad_np)

        # MANUAL GRADIENT COMPUTATION through head layers
        # This is necessary because MLX doesn't have native CTC loss
        # and the proxy loss approach doesn't propagate gradients correctly.

        grads = {}
        d_model = self.head.d_model
        vocab_size = self.head.phoneme_vocab

        # Get intermediate activations for backprop
        x = encoder_batch
        if self.head._use_layer_norm:
            x = self.head.ln(x)
        mx.eval(x)

        if self.head.hidden is not None:
            # Two-layer head: input -> hidden -> (dropout) -> proj
            hidden_input = x
            hidden_out = nn.gelu(self.head.hidden(hidden_input))
            mx.eval(hidden_out)

            # Apply dropout if enabled (recomputed, different mask than forward pass)
            # This is an approximation but still provides regularization benefit
            if self.head._dropout_rate > 0:
                hidden_out = self.head.dropout(hidden_out)
                mx.eval(hidden_out)

            # Gradient through proj layer
            # grad_W_proj = grad_logits.T @ hidden_out (post-dropout)
            # grad_b_proj = sum(grad_logits)
            encoder_batch.shape[0]
            hidden_out_flat = mx.reshape(hidden_out, (-1, self.head.hidden_dim))
            grad_flat = mx.reshape(logits_grad, (-1, vocab_size))

            grad_W_proj = mx.matmul(mx.transpose(grad_flat), hidden_out_flat)
            grad_b_proj = mx.sum(grad_flat, axis=0)

            # Gradient through hidden layer (chain rule through GELU)
            # d_loss/d_hidden_out = grad_logits @ proj.weight.T
            grad_hidden_out = mx.matmul(grad_flat, self.head.proj.weight)
            grad_hidden_out = mx.reshape(grad_hidden_out, hidden_out.shape)

            # GELU derivative: gelu'(x) ≈ sigmoid(1.702 * x) + x * sigmoid(1.702 * x) * (1 - sigmoid(1.702 * x)) * 1.702
            # For simplicity, use approximate: gelu'(x) ≈ 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            # Or just use the simple approximation that works well in practice
            hidden_pre = self.head.hidden(hidden_input)
            mx.eval(hidden_pre)
            gelu_grad = 0.5 * (1 + mx.tanh(0.7978845608 * (hidden_pre + 0.044715 * hidden_pre ** 3)))
            grad_hidden_pre = grad_hidden_out * gelu_grad

            # Gradient through hidden linear
            hidden_input_flat = mx.reshape(hidden_input, (-1, d_model))
            grad_hidden_pre_flat = mx.reshape(grad_hidden_pre, (-1, self.head.hidden_dim))

            grad_W_hidden = mx.matmul(mx.transpose(grad_hidden_pre_flat), hidden_input_flat)
            grad_b_hidden = mx.sum(grad_hidden_pre_flat, axis=0)

            grads = {
                'hidden': {'weight': grad_W_hidden, 'bias': grad_b_hidden},
                'proj': {'weight': grad_W_proj, 'bias': grad_b_proj},
            }
        else:
            # Single layer head: input -> proj
            x_flat = mx.reshape(x, (-1, d_model))
            grad_flat = mx.reshape(logits_grad, (-1, vocab_size))

            grad_W_proj = mx.matmul(mx.transpose(grad_flat), x_flat)
            grad_b_proj = mx.sum(grad_flat, axis=0)

            grads = {
                'proj': {'weight': grad_W_proj, 'bias': grad_b_proj},
            }

        # Add zero gradients for layer norm (not training it for stability)
        if self.head._use_layer_norm:
            grads['ln'] = {
                'weight': mx.zeros_like(self.head.ln.weight),
                'bias': mx.zeros_like(self.head.ln.bias),
            }

        # Clip gradients
        def clip_grads_recursive(g, max_val):
            if isinstance(g, dict):
                return {k: clip_grads_recursive(v, max_val) for k, v in g.items()}
            if isinstance(g, mx.array):
                return mx.clip(g, -max_val, max_val)
            return g

        grads = clip_grads_recursive(grads, self.config.grad_clip)

        # Apply optimizer
        self.optimizer.update(self.head, grads)
        mx.eval(self.head.parameters())

        self.step += 1
        return loss_value

    def evaluate(self, samples: list[AudioSample]) -> tuple[float, float]:
        """
        Evaluate on validation set.

        Returns:
            Tuple of (average_loss, phoneme_edit_rate)
        """
        total_loss = 0.0
        total_edit_dist = 0
        total_phonemes = 0
        n_samples = min(len(samples), 100)  # Cap eval at 100 samples

        for sample in samples[:n_samples]:
            try:
                mel = self._get_mel(sample.audio_path)
                enc_out = self._get_encoder_output(mel)

                # Get predictions (training=False disables dropout)
                logits = self.head(enc_out, training=False)
                mx.eval(logits)

                # Greedy decode
                preds = mx.argmax(logits, axis=-1)
                mx.eval(preds)
                preds_np = np.array(preds).squeeze()

                # Collapse CTC output
                collapsed = []
                prev = -1
                for t in preds_np:
                    if t != self.config.blank_id and t != prev:
                        collapsed.append(int(t))
                    prev = t

                # Get reference
                ref_tokens = self._get_phoneme_tokens(sample.transcript)

                # Edit distance
                edit_dist = self._levenshtein(collapsed, ref_tokens)
                total_edit_dist += edit_dist
                total_phonemes += max(len(ref_tokens), 1)

                # Compute loss
                targets = [ref_tokens]
                input_lengths = [enc_out.shape[1]]
                target_lengths = [len(ref_tokens)]

                loss, _ = ctc_loss_pytorch(
                    logits, targets, input_lengths, target_lengths,
                    blank_id=self.config.blank_id,
                )
                total_loss += loss

            except Exception as e:
                print(f"  Eval error on {sample.audio_path}: {e}")
                continue

        avg_loss = total_loss / n_samples if n_samples > 0 else 0.0
        per = total_edit_dist / total_phonemes if total_phonemes > 0 else 1.0
        return avg_loss, per

    def _levenshtein(self, s1: list[int], s2: list[int]) -> int:
        """Levenshtein edit distance."""
        if len(s1) < len(s2):
            return self._levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)

        prev = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev[j + 1] + 1
                deletions = curr[j] + 1
                substitutions = prev[j] + (c1 != c2)
                curr.append(min(insertions, deletions, substitutions))
            prev = curr

        return prev[-1]

    def save_checkpoint(self, suffix: str = ""):
        """Save model checkpoint."""
        filename = f"kokoro_phoneme_head_{self.step}{suffix}.npz"
        path = self.output_dir / filename
        self.head.save(str(path))
        print(f"  Saved checkpoint: {path}")

        # Save config
        config_path = self.output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump({
                "whisper_model": self.config.whisper_model,
                "hidden_dim": self.config.hidden_dim,
                "d_model": self.head.d_model,
                "phoneme_vocab": self.head.phoneme_vocab,
                "step": self.step,
            }, f, indent=2)

    def train(self, dataset: LibriSpeechDataset):
        """Main training loop."""
        train_samples = dataset.get_train_samples()
        val_samples = dataset.get_val_samples()

        print("\nStarting training:")
        print(f"  Train samples: {len(train_samples)}")
        print(f"  Val samples: {len(val_samples)}")
        print(f"  Epochs: {self.config.epochs}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"  Length-sorted batching: {self.config.length_sorted_batching}")
        print()

        # Populate durations if using length-sorted batching
        if self.config.length_sorted_batching:
            self._populate_sample_durations(train_samples)
            self._populate_sample_durations(val_samples)
            # Log duration statistics
            durations = [s.duration for s in train_samples]
            print(f"Duration stats: min={min(durations):.1f}s, max={max(durations):.1f}s, mean={np.mean(durations):.1f}s")

        for epoch in range(self.config.epochs):
            print(f"Epoch {epoch + 1}/{self.config.epochs}")

            epoch_loss = 0.0
            n_batches = 0
            start_time = time.time()

            # Track padding efficiency for length-sorted batching
            total_frames = 0
            total_padded_frames = 0

            if self.config.length_sorted_batching:
                # Use length-sorted batches
                batches = self._create_length_sorted_batches(train_samples)
            else:
                # Standard random shuffle
                rng = np.random.default_rng()
                rng.shuffle(train_samples)
                batches = [
                    train_samples[i:i + self.config.batch_size]
                    for i in range(0, len(train_samples), self.config.batch_size)
                ]

            # Process batches
            for batch in batches:

                # Load batch
                batch_mels = []
                batch_transcripts = []
                batch_audio_paths = []
                for sample in batch:
                    try:
                        mel = self._get_mel(sample.audio_path)
                        batch_mels.append(mel)
                        batch_transcripts.append(sample.transcript)
                        batch_audio_paths.append(sample.audio_path)
                    except Exception as e:
                        print(f"  Error loading {sample.audio_path}: {e}")
                        continue

                if not batch_mels:
                    continue

                # Train step (with encoder caching if enabled)
                loss = self.train_step(
                    batch_mels, batch_transcripts, batch_audio_paths,
                )
                epoch_loss += loss
                n_batches += 1

                # Track padding efficiency for length-sorted batching
                if self.config.length_sorted_batching and batch:
                    batch_durations = [s.duration for s in batch]
                    max_duration = max(batch_durations)
                    actual_frames = sum(int(d * 100) for d in batch_durations)  # ~100 frames/sec
                    padded_frames = len(batch) * int(max_duration * 100)
                    total_frames += actual_frames
                    total_padded_frames += padded_frames

                # Log
                if self.step % self.config.log_interval == 0:
                    avg_loss = epoch_loss / n_batches
                    elapsed = time.time() - start_time
                    samples_per_sec = (n_batches * self.config.batch_size) / elapsed
                    log_msg = f"  Step {self.step}: loss={avg_loss:.4f}, samples/s={samples_per_sec:.1f}"

                    # Add encoder cache stats if enabled
                    if self.encoder_cache is not None:
                        stats = self.encoder_cache.get_stats()
                        hit_rate = stats["hit_rate"]
                        log_msg += f", cache={stats['cached_files']} files ({hit_rate:.0%} hits)"

                    # Add padding efficiency if using length-sorted batching
                    if self.config.length_sorted_batching and total_padded_frames > 0:
                        efficiency = total_frames / total_padded_frames * 100
                        log_msg += f", pad_eff={efficiency:.1f}%"

                    print(log_msg)

                # Evaluate
                if self.step % self.config.eval_interval == 0:
                    val_loss, per = self.evaluate(val_samples)
                    print(f"  [Eval] val_loss={val_loss:.4f}, PER={per:.3f}")

                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint("_best")

                # Save checkpoint
                if self.step % self.config.save_interval == 0:
                    self.save_checkpoint()

                # Clear memory periodically
                if self.step % 50 == 0:
                    gc.collect()
                    if hasattr(mx, 'clear_cache'):
                        mx.clear_cache()

            # End of epoch
            epoch_time = time.time() - start_time
            avg_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
            epoch_msg = f"  Epoch {epoch + 1} complete: avg_loss={avg_loss:.4f}, time={epoch_time:.1f}s"

            # Add padding efficiency summary for length-sorted batching
            if self.config.length_sorted_batching and total_padded_frames > 0:
                efficiency = total_frames / total_padded_frames * 100
                epoch_msg += f", pad_eff={efficiency:.1f}%"

            print(epoch_msg)

            # Save end-of-epoch checkpoint
            self.save_checkpoint(f"_epoch{epoch + 1}")

        # Final save
        self.save_checkpoint("_final")
        print("\nTraining complete!")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Train Kokoro Phoneme Head for transcript verification",
    )

    # Data
    parser.add_argument(
        "--data-dir", type=str, default="data/LibriSpeech/dev-clean",
        help="Path to LibriSpeech data directory",
    )
    parser.add_argument(
        "--output-dir", type=str, default="checkpoints/kokoro_phoneme_head",
        help="Output directory for checkpoints",
    )

    # Model
    parser.add_argument(
        "--whisper-model", type=str, default="large-v3",
        help="Whisper model size (small, medium, large-v3)",
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=512,
        help="Hidden layer dimension for phoneme head",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.0,
        help="Dropout rate for regularization (0.1-0.2 recommended to prevent overfitting)",
    )

    # Training
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--max-audio-len", type=float, default=15.0,
        help="Maximum audio length in seconds",
    )

    # Logging
    parser.add_argument(
        "--log-interval", type=int, default=20,
        help="Steps between log outputs",
    )
    parser.add_argument(
        "--eval-interval", type=int, default=100,
        help="Steps between evaluations",
    )
    parser.add_argument(
        "--save-interval", type=int, default=500,
        help="Steps between checkpoint saves",
    )

    # Encoder caching (~3x speedup)
    parser.add_argument(
        "--encoder-cache-dir", type=str, default=None,
        help="Directory for cached encoder outputs (~3x speedup for multi-epoch training)",
    )

    # Length-sorted batching (~1.3x speedup)
    parser.add_argument(
        "--length-sorted-batching", action="store_true",
        help="Enable length-sorted batching (reduces padding waste ~1.3x speedup)",
    )
    parser.add_argument(
        "--bucket-size-multiplier", type=int, default=100,
        help="Bucket size = batch_size * multiplier (default: 100)",
    )

    # Resume from checkpoint
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint file to resume training from",
    )
    parser.add_argument(
        "--resume-step", type=int, default=0,
        help="Step number to resume from (for logging/checkpointing)",
    )

    args = parser.parse_args()

    # Check requirements
    if not HAS_MLX:
        print("ERROR: MLX required for training")
        return 1

    if not HAS_TORCH:
        print("ERROR: PyTorch required for CTC loss computation")
        return 1

    # Create config
    config = TrainingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        whisper_model=args.whisper_model,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_audio_len=args.max_audio_len,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        encoder_cache_dir=args.encoder_cache_dir,
        length_sorted_batching=args.length_sorted_batching,
        bucket_size_multiplier=args.bucket_size_multiplier,
        resume_checkpoint=args.resume,
        resume_step=args.resume_step,
    )

    # Load dataset
    dataset = LibriSpeechDataset(
        data_dir=config.data_dir,
        max_audio_len=config.max_audio_len,
        val_split=config.val_split,
    )

    if len(dataset) == 0:
        print(f"ERROR: No samples found in {config.data_dir}")
        return 1

    # Create trainer and train
    trainer = KokoroPhonemeTrainer(config)
    trainer.train(dataset)

    return 0


if __name__ == "__main__":
    exit(main())
