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
Base trainer class for all Whisper MLX training jobs.

Provides shared infrastructure for:
- Training loop with logging
- Checkpoint save/load
- Validation loop
- Memory management
- Learning rate scheduling
- Gradient accumulation
"""

import gc
import json
import resource
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from .data_loader import AudioSample, OptimizedDataLoader


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss / 1024 / 1024  # Convert to MB (macOS)


def clear_memory():
    """Clear MLX and Python memory."""
    if hasattr(mx, 'clear_cache'):
        mx.clear_cache()
    elif hasattr(mx, 'metal') and hasattr(mx.metal, 'clear_cache'):
        mx.metal.clear_cache()
    gc.collect()


@dataclass
class TrainingConfig:
    """Base configuration for all training jobs."""

    # Data
    data_dir: str = "data/LibriSpeech/dev-clean"
    output_dir: str = "checkpoints/head"
    mel_cache_dir: str | None = None

    # Model
    whisper_model: str = "mlx-community/whisper-large-v3-mlx"
    model_size: str = "large-v3"
    d_model: int = 1280  # large-v3 dimension

    # Training
    epochs: int = 10
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    warmup_steps: int = 500
    grad_clip: float = 1.0
    weight_decay: float = 0.01

    # Data loading
    max_audio_len: float = 30.0
    prefetch_workers: int = 2
    sort_by_length: bool = True

    # Logging
    log_interval: int = 50
    save_interval: int = 500
    eval_interval: int = 500

    # Validation
    val_split: float = 0.1

    # SpecAugment
    spec_augment: bool = False
    freq_mask_param: int = 27
    time_mask_param: int = 100
    num_freq_masks: int = 2
    num_time_masks: int = 2


class BaseTrainer(ABC):
    """
    Abstract base class for Whisper head training.

    Subclasses implement train_step() for specific loss computation.
    """

    def __init__(
        self,
        config: TrainingConfig,
        whisper_model: Any,  # WhisperMLX
        head: nn.Module,
        start_step: int = 0,
        start_epoch: int = 0,
        best_loss: float = float("inf"),
    ):
        """
        Initialize trainer.

        Args:
            config: Training configuration
            whisper_model: Frozen Whisper model (for encoder)
            head: Head module to train
            start_step: Starting step (for resuming)
            start_epoch: Starting epoch (for resuming)
            best_loss: Best loss so far (for resuming)
        """
        self.config = config
        self.whisper_model = whisper_model
        self.head = head

        # Optimizer
        self.optimizer = optim.AdamW(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Training state
        self.step = start_step
        self.epoch = start_epoch
        self.best_loss = best_loss
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Log file
        self.log_file = self.output_dir / "training.log"

        # Data loader
        self.data_loader = OptimizedDataLoader(
            batch_size=config.batch_size,
            max_audio_len=config.max_audio_len,
            prefetch_workers=config.prefetch_workers,
            sort_by_length=config.sort_by_length,
        )

    def log(self, message: str):
        """Log to console and file."""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print(message)
        with open(self.log_file, "a") as f:
            f.write(f"{timestamp} | {message}\n")

    @abstractmethod
    def train_step(self, batch: Any) -> tuple[float, dict[str, Any]]:
        """
        Single training step.

        Args:
            batch: Prepared batch data

        Returns:
            Tuple of (loss value, metrics dict)
        """

    @abstractmethod
    def validate_step(self, batch: Any) -> tuple[float, dict[str, Any]]:
        """
        Single validation step.

        Args:
            batch: Prepared batch data

        Returns:
            Tuple of (loss value, metrics dict)
        """

    @abstractmethod
    def prepare_batch(self, samples: list[AudioSample]) -> Any:
        """
        Prepare a batch of samples for training.

        Args:
            samples: List of audio samples

        Returns:
            Prepared batch data (implementation-specific)
        """

    def get_learning_rate(self) -> float:
        """Get current learning rate with warmup."""
        if self.step < self.config.warmup_steps:
            # Linear warmup
            return self.config.learning_rate * (self.step + 1) / self.config.warmup_steps
        return self.config.learning_rate

    def train(
        self,
        train_samples: list[AudioSample],
        val_samples: list[AudioSample],
    ):
        """
        Main training loop.

        Args:
            train_samples: Training samples
            val_samples: Validation samples
        """
        self.log("Starting training")
        self.log(f"  Epochs: {self.config.epochs}")
        self.log(f"  Train samples: {len(train_samples)}")
        self.log(f"  Val samples: {len(val_samples)}")
        self.log(f"  Batch size: {self.config.batch_size}")
        self.log(f"  Learning rate: {self.config.learning_rate}")

        for epoch in range(self.epoch, self.config.epochs):
            self.epoch = epoch

            # Train one epoch
            train_loss = self._train_epoch(train_samples)

            # Validate
            val_loss, val_metrics = self._validate(val_samples)

            self.log(f"Epoch {epoch + 1}/{self.config.epochs}")
            self.log(f"  Train loss: {train_loss:.4f}")
            self.log(f"  Val loss: {val_loss:.4f}")
            for k, v in val_metrics.items():
                self.log(f"  {k}: {v:.4f}")

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # Save best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self._save_checkpoint("best.npz")
                self.log("  New best model saved!")

            # Save epoch checkpoint
            self._save_checkpoint(f"epoch_{epoch + 1}.npz")

        # Save final model
        self._save_checkpoint("final.npz")
        self.log("Training complete!")

        # Save history
        self._save_history()

    def _train_epoch(self, samples: list[AudioSample]) -> float:
        """Train for one epoch."""
        total_loss = 0.0
        num_batches = 0
        start_time = time.time()

        # Iterate with prefetching
        for batch_data in self.data_loader.iterate(
            samples, self.prepare_batch, shuffle=True,
        ):
            # Update learning rate
            lr = self.get_learning_rate()
            self.optimizer.learning_rate = lr

            # Training step
            loss, metrics = self.train_step(batch_data)

            if loss > 0:
                total_loss += loss
                num_batches += 1

            self.step += 1

            # Logging
            if self.step % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                steps_per_sec = self.step / elapsed if elapsed > 0 else 0
                mem_mb = get_memory_usage_mb()
                self.log(
                    f"Step {self.step}: loss={loss:.4f}, "
                    f"lr={lr:.2e}, steps/s={steps_per_sec:.2f}, mem={mem_mb:.0f}MB",
                )

            # Save checkpoint
            if self.step % self.config.save_interval == 0:
                self._save_checkpoint(f"step_{self.step}.npz")

            # Clear memory periodically
            if self.step % 100 == 0:
                clear_memory()

        return total_loss / num_batches if num_batches > 0 else 0.0

    def _validate(
        self, samples: list[AudioSample],
    ) -> tuple[float, dict[str, Any]]:
        """Run validation."""
        total_loss = 0.0
        total_metrics: dict[str, float] = {}
        num_batches = 0

        for batch_data in self.data_loader.iterate(
            samples, self.prepare_batch, shuffle=False,
        ):
            loss, metrics = self.validate_step(batch_data)

            if loss > 0:
                total_loss += loss
                num_batches += 1
                for k, v in metrics.items():
                    total_metrics[k] = total_metrics.get(k, 0.0) + v

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}

        return avg_loss, avg_metrics

    def _save_checkpoint(self, filename: str):
        """Save checkpoint."""
        path = self.output_dir / filename
        mx.savez(
            str(path),
            **dict(self.head.parameters()),
        )
        # Save training state
        state_path = self.output_dir / f"{filename.replace('.npz', '_state.json')}"
        state = {
            "step": self.step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
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

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config: TrainingConfig,
        whisper_model: Any,
        head: nn.Module,
    ) -> "BaseTrainer":
        """
        Load trainer from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            config: Training configuration
            whisper_model: Whisper model
            head: Head module

        Returns:
            Trainer instance with loaded state
        """
        # Load weights
        weights = mx.load(checkpoint_path)
        head.load_weights(list(weights.items()))

        # Load training state
        state_path = checkpoint_path.replace('.npz', '_state.json')
        if Path(state_path).exists():
            with open(state_path) as f:
                state = json.load(f)
            return cls(
                config=config,
                whisper_model=whisper_model,
                head=head,
                start_step=state.get("step", 0),
                start_epoch=state.get("epoch", 0),
                best_loss=state.get("best_loss", float("inf")),
            )

        return cls(config=config, whisper_model=whisper_model, head=head)
