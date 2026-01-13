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
Medusa head training utilities for WhisperMLX.

This module implements self-distillation training for Medusa heads:
1. Run Whisper (teacher) on audio-text pairs (frozen)
2. Collect hidden states at each decoder position
3. Train Medusa heads to predict teacher's output distribution at shifted positions
4. Use KL divergence loss to match teacher distributions

The key insight is that Medusa head i should learn to predict what the teacher
model would predict at position n+i given the hidden state at position n.

Training Data:
- LibriSpeech (primary): 960 hours of read English speech
- Requires: audio files, text transcriptions, tokenization

References:
- Medusa paper: https://arxiv.org/abs/2401.10774
- Whisper-Medusa: https://github.com/aiola-lab/whisper-medusa
"""

import json
import os
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim

from .medusa import MedusaModule


@dataclass
class MedusaTrainingConfig:
    """Configuration for Medusa head training."""

    # Model
    n_medusa_heads: int = 5
    use_block: bool = False  # Medusa-Linear (False) vs Medusa-Block (True)

    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 4
    max_epochs: int = 10
    warmup_steps: int = 1000
    max_steps: int | None = None  # If set, overrides epochs
    gradient_accumulation_steps: int = 1

    # Loss
    kl_temperature: float = 1.0  # Temperature for KL divergence
    loss_weights: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0, 1.0])

    # Data
    max_audio_length: float = 30.0  # Max audio duration in seconds
    max_text_length: int = 448  # Max decoder tokens

    # Checkpointing
    checkpoint_dir: str = "checkpoints/medusa"
    save_every_steps: int = 1000
    eval_every_steps: int = 500

    # Logging
    log_every_steps: int = 100


def log_softmax(x: mx.array, axis: int = -1) -> mx.array:
    """Compute log softmax (x - logsumexp(x)) for numerical stability."""
    return x - mx.logsumexp(x, axis=axis, keepdims=True)


def kl_divergence_loss(
    student_logits: mx.array,
    teacher_logits: mx.array,
    temperature: float = 1.0,
    reduction: str = "mean",
) -> mx.array:
    """
    Compute KL divergence loss for knowledge distillation.

    KL(P_teacher || P_student) = sum(P_teacher * log(P_teacher / P_student))
                                = sum(P_teacher * (log(P_teacher) - log(P_student)))

    Args:
        student_logits: Medusa head logits, shape (batch, seq_len, vocab)
        teacher_logits: Teacher (main decoder) logits, shape (batch, seq_len, vocab)
        temperature: Softmax temperature (higher = softer distribution)
        reduction: "mean", "sum", or "none"

    Returns:
        KL divergence loss
    """
    # Scale by temperature
    student_scaled = student_logits / temperature
    teacher_scaled = teacher_logits / temperature

    # Teacher and student log probabilities
    teacher_log_probs = log_softmax(teacher_scaled, axis=-1)
    student_log_probs = log_softmax(student_scaled, axis=-1)

    # Teacher probabilities for weighting
    teacher_probs = mx.softmax(teacher_scaled, axis=-1)

    # KL divergence: sum over vocab dimension
    # KL = sum(p_teacher * (log(p_teacher) - log(p_student)))
    kl = mx.sum(teacher_probs * (teacher_log_probs - student_log_probs), axis=-1)

    # Scale by temperature^2 (standard KD scaling)
    kl = kl * (temperature ** 2)

    if reduction == "mean":
        return mx.mean(kl)
    if reduction == "sum":
        return mx.sum(kl)
    return kl


def compute_medusa_loss(
    medusa_logits: list[mx.array],
    teacher_logits: mx.array,
    config: MedusaTrainingConfig,
) -> tuple[mx.array, dict[str, float]]:
    """
    Compute total Medusa training loss.

    Each Medusa head i should predict teacher's output at position n+i+1
    given hidden state at position n.

    For head i (1-indexed in paper, 0-indexed here):
    - Input: hidden_state[:, :-i-1] (all positions except last i+1)
    - Target: teacher_logits[:, i+1:] (shifted by i+1 positions)

    Args:
        medusa_logits: List of logits from each Medusa head
                      Each shape: (batch, seq_len, vocab)
        teacher_logits: Teacher model logits, shape (batch, seq_len, vocab)
        config: Training configuration

    Returns:
        Tuple of:
        - Total weighted loss (scalar)
        - Dict of per-head losses for logging
    """
    # n_heads = len(medusa_logits) - used implicitly in enumerate loop
    seq_len = teacher_logits.shape[1]

    total_loss = mx.array(0.0)
    loss_dict = {}

    for head_idx, head_logits in enumerate(medusa_logits):
        # Head i predicts position n+i+1 from hidden state at position n
        shift = head_idx + 1

        # Ensure we have enough sequence length
        if seq_len <= shift:
            continue

        # Student: logits from head at positions 0 to seq_len-shift-1
        # These are predictions for positions shift to seq_len-1
        student = head_logits[:, :-shift]

        # Teacher: logits at positions shift to seq_len-1
        teacher = teacher_logits[:, shift:]

        # Compute KL divergence loss
        head_loss = kl_divergence_loss(
            student,
            teacher,
            temperature=config.kl_temperature,
            reduction="mean",
        )

        # Weight this head's contribution
        weight = config.loss_weights[head_idx] if head_idx < len(config.loss_weights) else 1.0
        total_loss = total_loss + weight * head_loss

        loss_dict[f"head_{head_idx}_loss"] = head_loss.item()

    loss_dict["total_loss"] = total_loss.item()

    return total_loss, loss_dict


@dataclass
class TrainingBatch:
    """A single training batch."""

    audio_features: mx.array  # Encoder output (batch, encoder_len, n_state)
    decoder_tokens: mx.array  # Tokenized text (batch, seq_len)
    attention_mask: mx.array | None = None  # Padding mask if needed


class MedusaTrainer:
    """
    Trainer for Medusa heads using self-distillation.

    Training loop:
    1. For each (audio, text) pair:
       a. Encode audio with frozen encoder
       b. Run frozen decoder to get teacher logits and hidden states
       c. Run Medusa heads on hidden states
       d. Compute KL loss between Medusa predictions and shifted teacher logits
       e. Update only Medusa head parameters
    """

    def __init__(
        self,
        model,  # WhisperMLX model (frozen except Medusa heads)
        medusa_module: MedusaModule,
        config: MedusaTrainingConfig,
        tokenizer=None,
    ):
        """
        Args:
            model: WhisperMLX model with encoder and decoder
            medusa_module: MedusaModule to train
            config: Training configuration
            tokenizer: Whisper tokenizer for text processing
        """
        self.model = model
        self.medusa = medusa_module
        self.config = config
        self.tokenizer = tokenizer

        # Freeze base model - only train Medusa heads
        self._freeze_base_model()

        # Initialize optimizer for Medusa parameters only
        self.optimizer = optim.AdamW(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float("inf")

        # Loss and gradient functions
        self._loss_fn = None
        self._grad_fn = None

    def _freeze_base_model(self):
        """Freeze all parameters except Medusa heads."""
        # In MLX, we control which parameters get updated through the optimizer
        # We'll only pass Medusa parameters to the optimizer

    def _get_trainable_params(self) -> dict[str, mx.array]:
        """Get parameters that should be trained (Medusa heads only)."""
        params = {}
        for i, head in enumerate(self.medusa.heads):
            head_params = head.parameters()
            for key, value in head_params.items():
                params[f"medusa.heads.{i}.{key}"] = value
        return params

    def _set_trainable_params(self, params: dict[str, mx.array]):
        """Update trainable parameters."""
        for i, head in enumerate(self.medusa.heads):
            head_prefix = f"medusa.heads.{i}."
            head_params = {}
            for key, value in params.items():
                if key.startswith(head_prefix):
                    param_name = key[len(head_prefix):]
                    head_params[param_name] = value
            if head_params:
                head.update(head_params)

    def _forward_teacher(
        self,
        audio_features: mx.array,
        decoder_tokens: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """
        Run frozen teacher model to get logits and hidden states.

        Args:
            audio_features: Encoder output (batch, encoder_len, n_state)
            decoder_tokens: Decoder input tokens (batch, seq_len)

        Returns:
            Tuple of:
            - teacher_logits: (batch, seq_len, vocab)
            - hidden_states: (batch, seq_len, n_state)
        """
        # Run decoder with return_hidden=True
        teacher_logits, _, _, hidden_states = self.model.decoder(
            decoder_tokens,
            audio_features,
            kv_cache=None,
            return_hidden=True,
        )

        return teacher_logits, hidden_states

    def _compute_loss_and_grads(
        self,
        audio_features: mx.array,
        decoder_tokens: mx.array,
        params: dict[str, mx.array],
    ) -> tuple[mx.array, dict[str, mx.array], dict[str, float]]:
        """
        Compute loss and gradients for a batch.

        Args:
            audio_features: Encoder output
            decoder_tokens: Decoder input tokens
            params: Current trainable parameters

        Returns:
            Tuple of:
            - loss: Scalar loss value
            - grads: Gradients for each parameter
            - loss_dict: Per-head loss values for logging
        """

        def loss_fn(params):
            # Update Medusa heads with current params
            self._set_trainable_params(params)

            # Get teacher outputs (frozen)
            teacher_logits, hidden_states = self._forward_teacher(
                audio_features, decoder_tokens,
            )

            # Get Medusa head predictions
            medusa_logits = self.medusa(hidden_states)

            # Compute loss
            loss, _ = compute_medusa_loss(
                medusa_logits, teacher_logits, self.config,
            )

            return loss

        # Compute loss and gradients
        loss, grads = mx.value_and_grad(loss_fn)(params)

        # Re-run to get loss dict for logging
        self._set_trainable_params(params)
        teacher_logits, hidden_states = self._forward_teacher(
            audio_features, decoder_tokens,
        )
        medusa_logits = self.medusa(hidden_states)
        _, loss_dict = compute_medusa_loss(
            medusa_logits, teacher_logits, self.config,
        )

        return loss, grads, loss_dict

    def train_step(
        self,
        batch: TrainingBatch,
    ) -> dict[str, float]:
        """
        Execute a single training step.

        Args:
            batch: Training batch with audio features and decoder tokens

        Returns:
            Dict of metrics for logging
        """
        params = self._get_trainable_params()

        # Compute loss and gradients
        loss, grads, loss_dict = self._compute_loss_and_grads(
            batch.audio_features,
            batch.decoder_tokens,
            params,
        )

        # Update parameters
        updated_params = self.optimizer.apply_gradients(grads, params)
        self._set_trainable_params(updated_params)

        self.global_step += 1

        # Add step info to metrics
        loss_dict["step"] = self.global_step
        loss_dict["learning_rate"] = self.config.learning_rate

        return loss_dict

    def evaluate(
        self,
        eval_batches: Iterator[TrainingBatch],
        max_batches: int = 100,
    ) -> dict[str, float]:
        """
        Evaluate on a validation set.

        Args:
            eval_batches: Iterator of evaluation batches
            max_batches: Maximum batches to evaluate

        Returns:
            Dict of evaluation metrics
        """
        total_loss = 0.0
        head_losses = [0.0] * self.config.n_medusa_heads
        n_batches = 0

        for batch in eval_batches:
            if n_batches >= max_batches:
                break

            # Forward pass only
            teacher_logits, hidden_states = self._forward_teacher(
                batch.audio_features,
                batch.decoder_tokens,
            )
            medusa_logits = self.medusa(hidden_states)
            loss, loss_dict = compute_medusa_loss(
                medusa_logits, teacher_logits, self.config,
            )

            total_loss += loss.item()
            for i in range(self.config.n_medusa_heads):
                key = f"head_{i}_loss"
                if key in loss_dict:
                    head_losses[i] += loss_dict[key]

            n_batches += 1

        # Compute averages
        metrics = {
            "eval_loss": total_loss / max(n_batches, 1),
        }
        for i, hl in enumerate(head_losses):
            metrics[f"eval_head_{i}_loss"] = hl / max(n_batches, 1)

        return metrics

    def save_checkpoint(self, path: str | None = None):
        """
        Save training checkpoint.

        Args:
            path: Optional custom path, otherwise uses config checkpoint_dir
        """
        if path is None:
            os.makedirs(self.config.checkpoint_dir, exist_ok=True)
            path = os.path.join(
                self.config.checkpoint_dir,
                f"medusa_step_{self.global_step}.npz",
            )

        # Get Medusa parameters (flattened)
        medusa_params = {}
        for i, head in enumerate(self.medusa.heads):
            flat_params = _flatten_params(head.parameters())
            for key, value in flat_params.items():
                medusa_params[f"heads.{i}.{key}"] = value

        # Save with training state
        checkpoint = {
            "global_step": mx.array([self.global_step]),
            "epoch": mx.array([self.epoch]),
            "best_loss": mx.array([self.best_loss]),
            **medusa_params,
        }

        mx.savez(path, **checkpoint)
        print(f"Saved checkpoint to {path}")

        # Also save config
        config_path = path.replace(".npz", "_config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.__dict__, f, indent=2)

    def load_checkpoint(self, path: str):
        """
        Load training checkpoint.

        Args:
            path: Path to checkpoint file
        """
        checkpoint = mx.load(path)

        # Restore training state
        self.global_step = int(checkpoint.pop("global_step").item())
        self.epoch = int(checkpoint.pop("epoch").item())
        self.best_loss = float(checkpoint.pop("best_loss").item())

        # Restore Medusa parameters
        # Helper to convert flat dict keys ("linear.bias") to nested dict ({"linear": {"bias": ...}})
        def make_nested(flat_dict):
            result = {}
            for key, value in flat_dict.items():
                parts = key.split(".")
                current = result
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            return result

        for i, head in enumerate(self.medusa.heads):
            head_prefix = f"heads.{i}."
            head_params = {}
            for key, value in checkpoint.items():
                if key.startswith(head_prefix):
                    param_name = key[len(head_prefix):]
                    head_params[param_name] = value
            if head_params:
                # Convert flat keys to nested structure for nn.Module.update()
                nested_params = make_nested(head_params)
                head.update(nested_params)

        print(f"Loaded checkpoint from {path}, step {self.global_step}")

    def train(
        self,
        train_batches: Iterator[TrainingBatch],
        eval_batches: Iterator[TrainingBatch] | None = None,
        callback: Callable[[dict[str, float]], None] | None = None,
    ):
        """
        Run full training loop.

        Args:
            train_batches: Iterator of training batches
            eval_batches: Optional iterator of evaluation batches
            callback: Optional callback called after each step with metrics
        """
        print("Starting Medusa training")
        print(f"  Heads: {self.config.n_medusa_heads}")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"  Max steps: {self.config.max_steps or 'unlimited'}")

        start_time = time.time()

        for batch in train_batches:
            # Training step
            metrics = self.train_step(batch)

            # Logging
            if self.global_step % self.config.log_every_steps == 0:
                elapsed = time.time() - start_time
                steps_per_sec = self.global_step / max(elapsed, 1)
                print(
                    f"Step {self.global_step}: "
                    f"loss={metrics['total_loss']:.4f}, "
                    f"steps/sec={steps_per_sec:.2f}",
                )

            # Evaluation
            if eval_batches and self.global_step % self.config.eval_every_steps == 0:
                eval_metrics = self.evaluate(eval_batches)
                print(f"  Eval loss: {eval_metrics['eval_loss']:.4f}")

                if eval_metrics["eval_loss"] < self.best_loss:
                    self.best_loss = eval_metrics["eval_loss"]
                    self.save_checkpoint()

            # Checkpointing
            if self.global_step % self.config.save_every_steps == 0:
                self.save_checkpoint()

            # Callback
            if callback:
                callback(metrics)

            # Check stopping condition
            if self.config.max_steps and self.global_step >= self.config.max_steps:
                break

        # Final checkpoint
        self.save_checkpoint()
        print(f"Training complete. Final step: {self.global_step}")


class LibriSpeechDataLoader:
    """
    Data loader for LibriSpeech dataset.

    LibriSpeech structure:
    - train-clean-100/: 100 hours of clean speech
    - train-clean-360/: 360 hours of clean speech
    - train-other-500/: 500 hours of other speech

    Each split contains speaker directories with chapter subdirectories.
    Each chapter has:
    - .flac audio files
    - .trans.txt transcription file (format: "file_id text")
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer,
        encoder_fn: Callable,
        config: MedusaTrainingConfig,
        splits: list[str] | None = None,
    ):
        """
        Args:
            data_dir: Path to LibriSpeech root directory
            tokenizer: Whisper tokenizer
            encoder_fn: Function to encode audio -> features
            config: Training configuration
            splits: List of splits to use (e.g., ["train-clean-100"])
        """
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.encoder_fn = encoder_fn
        self.config = config
        self.splits = splits or ["train-clean-100"]

        # Collect all samples
        self.samples = self._collect_samples()
        print(f"Found {len(self.samples)} samples in {self.splits}")

    def _collect_samples(self) -> list[dict]:
        """Collect all (audio_path, text) pairs from dataset."""
        samples = []

        for split in self.splits:
            split_dir = self.data_dir / split
            if not split_dir.exists():
                print(f"Warning: Split {split} not found at {split_dir}")
                continue

            # Walk through speaker/chapter directories
            for speaker_dir in split_dir.iterdir():
                if not speaker_dir.is_dir():
                    continue

                for chapter_dir in speaker_dir.iterdir():
                    if not chapter_dir.is_dir():
                        continue

                    # Find transcription file
                    trans_files = list(chapter_dir.glob("*.trans.txt"))
                    if not trans_files:
                        continue

                    trans_file = trans_files[0]

                    # Parse transcriptions
                    with open(trans_file) as f:
                        for line in f:
                            parts = line.strip().split(" ", 1)
                            if len(parts) != 2:
                                continue

                            file_id, text = parts
                            audio_path = chapter_dir / f"{file_id}.flac"

                            if audio_path.exists():
                                samples.append({
                                    "audio_path": str(audio_path),
                                    "text": text,
                                })

        return samples

    def _load_audio(self, audio_path: str) -> mx.array:
        """Load and preprocess audio file."""
        import soundfile as sf

        audio, sr = sf.read(audio_path)

        # Resample to 16kHz if needed
        if sr != 16000:
            # Simple resampling (for production, use proper resampler)
            import numpy as np
            ratio = 16000 / sr
            n_samples = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, n_samples)
            audio = np.interp(indices, np.arange(len(audio)), audio)

        # Truncate to max length
        max_samples = int(self.config.max_audio_length * 16000)
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        return mx.array(audio, dtype=mx.float32)

    def _tokenize_text(self, text: str) -> mx.array:
        """Tokenize text for decoder input."""
        # Add special tokens for Whisper decoder
        tokens = self.tokenizer.encode(text)

        # Prepend SOT tokens (simplified - actual Whisper has more complex SOT)
        sot_sequence = [self.tokenizer.sot]
        tokens = sot_sequence + tokens + [self.tokenizer.eot]

        # Truncate to max length
        if len(tokens) > self.config.max_text_length:
            tokens = tokens[:self.config.max_text_length]

        return mx.array(tokens)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[TrainingBatch]:
        """Iterate over batches for multiple epochs."""
        import random

        for epoch in range(self.config.max_epochs):
            print(f"Starting epoch {epoch + 1}/{self.config.max_epochs}")

            indices = list(range(len(self.samples)))
            random.shuffle(indices)

            batch_audio = []
            batch_tokens = []

            for idx in indices:
                sample = self.samples[idx]

                try:
                    # Load and encode audio
                    audio = self._load_audio(sample["audio_path"])
                    audio_features = self.encoder_fn(audio)

                    # Tokenize text
                    tokens = self._tokenize_text(sample["text"])

                    batch_audio.append(audio_features)
                    batch_tokens.append(tokens)

                    # Yield batch when full
                    if len(batch_audio) >= self.config.batch_size:
                        yield self._collate_batch(batch_audio, batch_tokens)
                        batch_audio = []
                        batch_tokens = []

                except Exception as e:
                    print(f"Error processing {sample['audio_path']}: {e}")
                    continue

            # Yield remaining samples from this epoch
            if batch_audio:
                yield self._collate_batch(batch_audio, batch_tokens)
                batch_audio = []
                batch_tokens = []

    def _collate_batch(
        self,
        audio_features: list[mx.array],
        tokens: list[mx.array],
    ) -> TrainingBatch:
        """Collate samples into a batch with padding."""
        # Stack audio features (assume same encoder output length)
        # For variable lengths, would need padding
        audio_batch = mx.stack(audio_features, axis=0)

        # Pad tokens to same length
        # Use 0 as pad token (Whisper tokenizer may not have explicit pad token)
        pad_token = getattr(self.tokenizer, 'pad', None) or 0
        max_len = max(len(t) for t in tokens)
        padded_tokens = []
        for t in tokens:
            if len(t) < max_len:
                padding = mx.full((max_len - len(t),), pad_token)
                t = mx.concatenate([t, padding])
            padded_tokens.append(t)

        tokens_batch = mx.stack(padded_tokens, axis=0)

        return TrainingBatch(
            audio_features=audio_batch,
            decoder_tokens=tokens_batch,
        )


def _flatten_params(params: dict, prefix: str = "") -> dict[str, mx.array]:
    """Flatten nested parameter dict to flat dict with dotted keys."""
    flat = {}
    for key, value in params.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_params(value, full_key))
        else:
            flat[full_key] = value
    return flat


def _unflatten_params(flat: dict[str, mx.array]) -> dict:
    """Unflatten dotted keys back to nested dict structure."""
    result = {}
    for key, value in flat.items():
        parts = key.split(".")
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return result


def load_medusa_weights(
    medusa_module: MedusaModule,
    weights_path: str,
) -> MedusaModule:
    """
    Load trained Medusa weights from checkpoint.

    Args:
        medusa_module: MedusaModule to load weights into
        weights_path: Path to .npz checkpoint file

    Returns:
        MedusaModule with loaded weights
    """
    checkpoint = dict(mx.load(weights_path))

    # Load head parameters
    for i, head in enumerate(medusa_module.heads):
        head_prefix = f"heads.{i}."
        head_params_flat = {}
        for key, value in checkpoint.items():
            if key.startswith(head_prefix):
                param_name = key[len(head_prefix):]
                head_params_flat[param_name] = value
        if head_params_flat:
            # Unflatten to match head's parameter structure
            head_params = _unflatten_params(head_params_flat)
            head.update(head_params)

    return medusa_module


def save_medusa_weights(
    medusa_module: MedusaModule,
    weights_path: str,
):
    """
    Save Medusa weights to file.

    Args:
        medusa_module: MedusaModule to save
        weights_path: Path to save .npz file
    """
    params = {}
    for i, head in enumerate(medusa_module.heads):
        # Flatten nested parameters
        flat_params = _flatten_params(head.parameters())
        for key, value in flat_params.items():
            params[f"heads.{i}.{key}"] = value

    mx.savez(weights_path, **params)
    print(f"Saved Medusa weights to {weights_path}")
