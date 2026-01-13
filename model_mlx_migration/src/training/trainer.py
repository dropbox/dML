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
Training loop for Zipformer ASR model.

Implements stability mitigations for streaming Zipformer-large:
- Gradient clipping
- Warmup scheduler
- Loss scaling and NaN detection
- Checkpointing
"""

import json
import math
import os
import shutil
import tarfile
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from .config import TrainingConfig
from .loss import CRCTCLoss, CTCLoss, RichAudioLoss
from .scheduler import get_scheduler


@dataclass
class TrainingState:
    """Current state of training for checkpointing."""

    step: int = 0
    epoch: int = 0
    best_loss: float = float("inf")
    best_val_wer: float = float("inf")

    # E13 fix: Use deque with maxlen to prevent unbounded growth
    train_losses: deque = field(default_factory=lambda: deque(maxlen=1000))
    val_losses: deque = field(default_factory=lambda: deque(maxlen=1000))
    learning_rates: deque = field(default_factory=lambda: deque(maxlen=1000))

    # Timing
    total_train_time: float = 0.0
    samples_processed: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "step": self.step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "best_val_wer": self.best_val_wer,
            "train_losses": list(self.train_losses),
            "val_losses": list(self.val_losses),
            "learning_rates": list(self.learning_rates),
            "total_train_time": self.total_train_time,
            "samples_processed": self.samples_processed,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainingState":
        """Create state from dictionary."""
        state = cls()
        state.step = d.get("step", 0)
        state.epoch = d.get("epoch", 0)
        state.best_loss = d.get("best_loss", float("inf"))
        state.best_val_wer = d.get("best_val_wer", float("inf"))
        state.train_losses.extend(d.get("train_losses", []))
        state.val_losses.extend(d.get("val_losses", []))
        state.learning_rates.extend(d.get("learning_rates", []))
        state.total_train_time = d.get("total_train_time", 0.0)
        state.samples_processed = d.get("samples_processed", 0)
        return state


class Trainer:
    """
    Trainer for Zipformer ASR model.

    Features:
    - CR-CTC joint loss training
    - Gradient clipping for stability
    - Learning rate warmup
    - NaN/Inf loss detection and skipping
    - Checkpointing and resumption
    - Validation during training
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataloader: Iterator | None = None,
        val_dataloader: Iterator | None = None,
        rich_audio_heads: nn.Module | None = None,
    ):
        """
        Initialize trainer.

        Args:
            model: ASR model to train.
            config: Training configuration.
            train_dataloader: Training data iterator.
            val_dataloader: Validation data iterator (optional).
            rich_audio_heads: Optional RichAudioHeads module for multi-head training.
        """
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.rich_audio_heads = rich_audio_heads

        # Initialize optimizer
        self.optimizer = self._create_optimizer()

        # Initialize scheduler
        self.scheduler = get_scheduler(
            config.scheduler,
            config.optimizer.learning_rate,
        )

        # Initialize loss function
        self.loss_fn = self._create_loss_fn()

        # Initialize rich audio loss if enabled
        self.rich_audio_loss_fn: RichAudioLoss | None = None
        if config.rich_audio_heads.enabled and rich_audio_heads is not None:
            cfg = config.rich_audio_heads
            self.rich_audio_loss_fn = RichAudioLoss(
                emotion_weight=cfg.emotion_weight,
                language_weight=cfg.language_weight,
                paralinguistics_weight=cfg.paralinguistics_weight,
                pitch_weight=cfg.pitch_weight,
                phoneme_weight=cfg.phoneme_weight,
                singing_weight=cfg.singing_weight,
                timestamp_weight=cfg.timestamp_weight,
                pitch_f0_weight=cfg.pitch_f0_weight,
                pitch_voiced_weight=cfg.pitch_voiced_weight,
                singing_binary_weight=cfg.singing_binary_weight,
                singing_technique_weight=cfg.singing_technique_weight,
                timestamp_boundary_weight=cfg.timestamp_boundary_weight,
                timestamp_offset_weight=cfg.timestamp_offset_weight,
                label_smoothing=cfg.label_smoothing,
            )

        # Training state
        self.state = TrainingState()

        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # E2 fix: Create CTCLoss once in __init__, not every train_step
        self.ctc_loss_fn = CTCLoss(blank_id=self.config.blank_id, reduction="mean")

        # E12 fix: Cache hasattr checks in __init__
        self._has_ctc_head = hasattr(model, "ctc_head")

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config."""
        cfg = self.config.optimizer

        if cfg.optimizer_type == "adamw":
            return optim.AdamW(
                learning_rate=cfg.learning_rate,
                betas=[cfg.beta1, cfg.beta2],
                eps=cfg.eps,
                weight_decay=cfg.weight_decay,
            )
        if cfg.optimizer_type == "sgd":
            return optim.SGD(
                learning_rate=cfg.learning_rate,
                momentum=0.9,
                weight_decay=cfg.weight_decay,
            )
        raise ValueError(f"Unknown optimizer: {cfg.optimizer_type}")

    def _create_loss_fn(self) -> nn.Module:
        """Create loss function based on config."""
        if self.config.loss_type == "cr_ctc":
            return CRCTCLoss(
                blank_id=self.config.blank_id,
                ctc_weight=self.config.ctc_weight,
                reduction="mean",
            )
        if self.config.loss_type == "ctc":
            return CTCLoss(
                blank_id=self.config.blank_id,
                reduction="mean",
            )
        raise ValueError(f"Unknown loss type: {self.config.loss_type}")

    def _clip_gradients(self, grads: dict, max_norm: float) -> tuple[dict, float]:
        """
        Clip gradients by global norm.

        M12 fix: Compute norm incrementally without full concatenation.
        This avoids creating a massive temporary array for large models.

        Args:
            grads: Dictionary of gradients.
            max_norm: Maximum norm threshold.

        Returns:
            Tuple of (clipped_grads, grad_norm).
        """
        # M12 fix: Compute global norm incrementally without concatenation
        total_norm_sq = 0.0

        def accumulate_norm_sq(g):
            nonlocal total_norm_sq
            if isinstance(g, dict):
                for v in g.values():
                    accumulate_norm_sq(v)
            elif isinstance(g, list):
                for v in g:
                    accumulate_norm_sq(v)
            elif g is not None:
                # Compute sum of squares for this gradient tensor
                total_norm_sq += mx.sum(g * g).item()

        accumulate_norm_sq(grads)

        if total_norm_sq == 0.0:
            return grads, 0.0

        grad_norm = math.sqrt(total_norm_sq)

        # Clip if necessary
        if grad_norm > max_norm:
            scale = max_norm / (grad_norm + 1e-6)

            def clip(g):
                if isinstance(g, dict):
                    return {k: clip(v) for k, v in g.items()}
                if isinstance(g, list):
                    return [clip(v) for v in g]
                if g is not None:
                    return g * scale
                return g

            grads = clip(grads)

        return grads, grad_norm

    def _check_loss_valid(self, loss: mx.array) -> bool:
        """Check if loss is valid (not NaN or Inf)."""
        loss_val = loss.item()
        if loss_val != loss_val:  # NaN check
            return False
        if abs(loss_val) > self.config.max_loss:
            return False
        return True

    def _has_rich_audio_labels(self, batch: dict[str, mx.array]) -> bool:
        """Check if batch contains any rich audio labels."""
        rich_label_keys = {
            "emotion_labels",
            "language_labels",
            "paralinguistics_labels",
            "pitch_f0_targets",
            "phoneme_labels",
            "singing_binary_labels",
            "timestamp_boundary_targets",
        }
        return any(key in batch for key in rich_label_keys)

    def _extract_rich_audio_labels(
        self, batch: dict[str, mx.array],
    ) -> dict[str, mx.array]:
        """Extract rich audio labels from batch."""
        label_keys = [
            "emotion_labels",
            "language_labels",
            "paralinguistics_labels",
            "pitch_f0_targets",
            "pitch_voiced_targets",
            "pitch_mask",
            "phoneme_labels",
            "phoneme_mask",
            "singing_binary_labels",
            "singing_technique_labels",
            "timestamp_boundary_targets",
            "timestamp_offset_targets",
            "timestamp_mask",
        ]
        return {k: batch[k] for k in label_keys if k in batch}

    def train_step(self, batch: dict[str, mx.array]) -> dict[str, float]:
        """
        Execute single training step.

        Args:
            batch: Dictionary containing:
                - features: Mel features (batch, time, mel_dim)
                - feature_lengths: Feature lengths (batch,)
                - targets: Target tokens (batch, target_len)
                - target_lengths: Target lengths (batch,)
                - (optional) Rich audio labels for multi-head training

        Returns:
            Dictionary of metrics (loss, grad_norm, etc.)
        """
        # B5 fix: Handle empty batch
        if batch["features"].shape[0] == 0:
            return {
                "loss": 0.0,
                "skipped": True,
                "reason": "empty_batch",
            }

        # Check if we should compute rich audio losses
        compute_rich_audio = (
            self.rich_audio_heads is not None
            and self.rich_audio_loss_fn is not None
            and self._has_rich_audio_labels(batch)
        )

        # M7 fix: Use closure to capture loss components for logging
        # instead of doing a second forward pass
        loss_components = {"ctc": None, "rich_audio": None, "rich_losses": {}}

        def loss_fn(model):
            # Forward pass through encoder
            encoder_out, encoder_lengths = model.encoder(
                batch["features"],
                batch["feature_lengths"],
            )

            # E12 fix: Use cached hasattr check
            if self._has_ctc_head:
                ctc_output = model.ctc_head(encoder_out)
            else:
                # Use joiner output as CTC proxy
                ctc_output = model.joiner.output_linear(encoder_out)

            # E2 fix: Use cached CTCLoss instance instead of creating new one
            ctc_loss = self.ctc_loss_fn(
                logits=ctc_output,
                targets=batch["targets"],
                input_lengths=encoder_lengths,
                target_lengths=batch["target_lengths"],
            )

            total_loss = ctc_loss
            # M7 fix: Save loss component for logging (avoid second forward pass)
            loss_components["ctc"] = ctc_loss

            # Add rich audio head losses if enabled and labels present
            if compute_rich_audio:
                # Get rich audio outputs
                rich_outputs = self.rich_audio_heads(
                    encoder_out,
                    encoder_lengths,
                )

                # Extract labels from batch
                rich_labels = self._extract_rich_audio_labels(batch)

                # Compute rich audio loss
                rich_loss_out = self.rich_audio_loss_fn(rich_outputs, rich_labels)

                # Add to total loss
                total_loss = total_loss + rich_loss_out.total_loss

                # M7 fix: Save for logging (avoid second forward pass)
                loss_components["rich_audio"] = rich_loss_out.total_loss
                loss_components["rich_losses"] = rich_loss_out.losses

            return total_loss

        # Compute loss and gradients
        loss_and_grad_fn = nn.value_and_grad(self.model, loss_fn)
        loss, grads = loss_and_grad_fn(self.model)

        # M7 fix: Use saved loss components for logging (no second forward pass)
        trans_loss_val = 0.0
        ctc_loss_val = loss_components["ctc"].item() if loss_components["ctc"] is not None else loss.item()
        rich_audio_loss_val = loss_components["rich_audio"].item() if loss_components["rich_audio"] is not None else 0.0
        rich_losses: dict[str, float] = {
            k: v.item() for k, v in loss_components["rich_losses"].items()
        } if loss_components["rich_losses"] else {}

        # Check for invalid loss
        if not self._check_loss_valid(loss):
            if self.config.skip_nan_loss:
                return {
                    "loss": float("nan"),
                    "skipped": True,
                    "reason": "invalid_loss",
                }
            raise ValueError(f"Invalid loss: {loss.item()}")

        # Clip gradients
        grads, grad_norm = self._clip_gradients(
            grads,
            self.config.optimizer.grad_clip,
        )

        # Update learning rate
        lr = self.scheduler.step()
        self.optimizer.learning_rate = lr

        # Apply gradients
        self.optimizer.update(self.model, grads)
        mx.eval(self.model.parameters())

        # Update state
        self.state.step += 1
        self.state.train_losses.append(loss.item())
        self.state.learning_rates.append(lr)
        self.state.samples_processed += batch["features"].shape[0]

        metrics = {
            "loss": loss.item(),
            "trans_loss": trans_loss_val,
            "ctc_loss": ctc_loss_val,
            "rich_audio_loss": rich_audio_loss_val,
            "grad_norm": grad_norm,
            "lr": lr,
            "skipped": False,
        }

        # Add individual rich audio losses
        for k, v in rich_losses.items():
            metrics[f"rich_{k}_loss"] = v

        return metrics

    def validate(self) -> dict[str, float]:
        """
        Run validation loop.

        M6 fix: Call mx.eval() after each batch to release computation graph memory.

        Returns:
            Dictionary of validation metrics.
        """
        if self.val_dataloader is None:
            return {}

        total_loss = 0.0
        total_rich_loss = 0.0
        rich_loss_accum: dict[str, float] = {}
        num_batches = 0

        for batch in self.val_dataloader:
            # B5 fix: Skip empty batches
            if batch["features"].shape[0] == 0:
                continue

            # Forward pass (no gradients)
            encoder_out, encoder_lengths = self.model.encoder(
                batch["features"],
                batch["feature_lengths"],
            )

            # E12 fix: Use cached hasattr check
            if self._has_ctc_head:
                ctc_output = self.model.ctc_head(encoder_out)
            else:
                ctc_output = self.model.joiner.output_linear(encoder_out)

            # E2 fix: Use cached CTCLoss instance
            loss = self.ctc_loss_fn(
                logits=ctc_output,
                targets=batch["targets"],
                input_lengths=encoder_lengths,
                target_lengths=batch["target_lengths"],
            )

            # M6 fix: Call mx.eval() after each batch to release computation graph memory
            mx.eval(loss)
            total_loss += loss.item()

            # Compute rich audio losses if enabled
            if (
                self.rich_audio_heads is not None
                and self.rich_audio_loss_fn is not None
                and self._has_rich_audio_labels(batch)
            ):
                rich_outputs = self.rich_audio_heads(encoder_out, encoder_lengths)
                rich_labels = self._extract_rich_audio_labels(batch)
                rich_loss_out = self.rich_audio_loss_fn(rich_outputs, rich_labels)
                # M6 fix: Eval rich audio loss too
                mx.eval(rich_loss_out.total_loss)
                total_rich_loss += rich_loss_out.total_loss.item()

                for k, v in rich_loss_out.losses.items():
                    rich_loss_accum[k] = rich_loss_accum.get(k, 0.0) + v.item()

            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        self.state.val_losses.append(avg_loss)

        metrics = {"val_loss": avg_loss}

        # Add rich audio metrics if computed
        if total_rich_loss > 0:
            metrics["val_rich_audio_loss"] = total_rich_loss / max(num_batches, 1)
            for k, v in rich_loss_accum.items():
                metrics[f"val_rich_{k}_loss"] = v / max(num_batches, 1)

        return metrics

    def train(
        self,
        num_steps: int | None = None,
        callback: Callable[[dict], None] | None = None,
    ) -> TrainingState:
        """
        Main training loop.

        Args:
            num_steps: Number of steps to train (or use config).
            callback: Optional callback function called after each step.

        Returns:
            Final training state.
        """
        if self.train_dataloader is None:
            raise ValueError("Train dataloader not provided")

        num_steps = num_steps or self.config.scheduler.total_steps
        start_step = self.state.step

        print(f"Starting training from step {start_step}")
        print(f"Training for {num_steps - start_step} steps")
        print(f"Config: {self.config.model_variant}, LR={self.config.optimizer.learning_rate}")

        train_start = time.time()

        for batch in self.train_dataloader:
            if self.state.step >= num_steps:
                break

            step_start = time.time()

            # Training step
            metrics = self.train_step(batch)

            step_time = time.time() - step_start
            self.state.total_train_time += step_time

            # Logging
            if self.state.step % self.config.log_every_n_steps == 0:
                samples_per_sec = self.state.samples_processed / self.state.total_train_time
                log_msg = (
                    f"Step {self.state.step}: "
                    f"loss={metrics['loss']:.4f}, "
                    f"lr={metrics['lr']:.2e}, "
                    f"grad_norm={metrics['grad_norm']:.2f}, "
                    f"samples/s={samples_per_sec:.1f}"
                )
                # Add rich audio loss if present
                if metrics.get("rich_audio_loss", 0) > 0:
                    log_msg += f", rich_loss={metrics['rich_audio_loss']:.4f}"
                print(log_msg)

            # Validation
            if (
                self.val_dataloader is not None
                and self.state.step % self.config.val_every_n_steps == 0
            ):
                val_metrics = self.validate()
                val_msg = f"Validation: loss={val_metrics.get('val_loss', 0):.4f}"
                if val_metrics.get("val_rich_audio_loss", 0) > 0:
                    val_msg += f", rich_loss={val_metrics['val_rich_audio_loss']:.4f}"
                print(val_msg)

                # Check for best model
                if val_metrics.get("val_loss", float("inf")) < self.state.best_loss:
                    self.state.best_loss = val_metrics["val_loss"]
                    self.save_checkpoint("best")

            # Checkpointing
            if self.state.step % self.config.save_every_n_steps == 0:
                self.save_checkpoint(f"step_{self.state.step}")

            # Callback
            if callback is not None:
                callback(metrics)

        total_time = time.time() - train_start
        print(f"Training complete. Total time: {total_time:.1f}s")

        # Final checkpoint
        self.save_checkpoint("final")

        return self.state

    def _flatten_params(self, params: dict, prefix: str = "") -> dict[str, mx.array]:
        """Flatten nested parameter dict for safetensors."""
        flat = {}
        for k, v in params.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(self._flatten_params(v, key))
            elif isinstance(v, mx.array):
                flat[key] = v
        return flat

    def save_checkpoint(self, name: str) -> Path:
        """
        Save training checkpoint.

        Args:
            name: Checkpoint name (e.g., "step_1000", "best", "final").

        Returns:
            Path to saved checkpoint.
        """
        ckpt_path = self.checkpoint_dir / name
        ckpt_path.mkdir(parents=True, exist_ok=True)

        # Save model weights (flatten nested params)
        weights_path = ckpt_path / "model.safetensors"
        flat_params = self._flatten_params(dict(self.model.parameters()))
        mx.save_safetensors(str(weights_path), flat_params)

        # Save optimizer state (flatten if necessary)
        opt_path = ckpt_path / "optimizer.safetensors"
        opt_state = self.optimizer.state
        if opt_state:
            flat_opt_state = self._flatten_params(opt_state)
            if flat_opt_state:
                mx.save_safetensors(str(opt_path), flat_opt_state)

        # Save scheduler state
        sched_path = ckpt_path / "scheduler.json"
        with open(sched_path, "w") as f:
            json.dump(self.scheduler.state_dict(), f)

        # Save training state
        state_path = ckpt_path / "training_state.json"
        with open(state_path, "w") as f:
            json.dump(self.state.to_dict(), f)

        # E15 fix: Save config once (it is immutable during training)
        config_path = self.checkpoint_dir / "config.json"
        if not config_path.exists():
            self.config.save(str(config_path))
        ckpt_config_path = ckpt_path / "config.json"
        if not ckpt_config_path.exists():
            try:
                rel = os.path.relpath(config_path, start=ckpt_path)
                ckpt_config_path.symlink_to(rel)
            except Exception:
                shutil.copy2(config_path, ckpt_config_path)

        print(f"Saved checkpoint to {ckpt_path}")

        # Clean up old checkpoints
        self._cleanup_checkpoints()

        return ckpt_path

    def _unflatten_params(self, flat: dict[str, mx.array]) -> dict:
        """Unflatten dotted keys back to nested dict."""
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

    def load_checkpoint(self, path: str) -> None:
        """
        Load training checkpoint.

        Args:
            path: Path to checkpoint directory.
        """
        ckpt_path = Path(path)

        # Load model weights
        weights_path = ckpt_path / "model.safetensors"
        if weights_path.exists():
            flat_weights = mx.load(str(weights_path))
            weights = self._unflatten_params(flat_weights)
            self.model.update(weights)

        # Load optimizer state
        opt_path = ckpt_path / "optimizer.safetensors"
        if opt_path.exists():
            opt_state = mx.load(str(opt_path))
            self.optimizer.state = opt_state

        # Load scheduler state
        sched_path = ckpt_path / "scheduler.json"
        if sched_path.exists():
            with open(sched_path) as f:
                self.scheduler.load_state_dict(json.load(f))

        # Load training state
        state_path = ckpt_path / "training_state.json"
        if state_path.exists():
            with open(state_path) as f:
                self.state = TrainingState.from_dict(json.load(f))

        print(f"Loaded checkpoint from {ckpt_path}")
        print(f"Resuming from step {self.state.step}")

    def _cleanup_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the most recent."""
        # Get all step checkpoints
        step_ckpts = sorted(
            [p for p in self.checkpoint_dir.iterdir() if p.name.startswith("step_")],
            key=lambda p: int(p.name.split("_")[1]),
        )

        # Keep only last N
        to_remove = step_ckpts[: -self.config.keep_last_n_checkpoints]
        for ckpt in to_remove:
            # E9 fix (optional): archive older checkpoints with gzip compression instead of deleting raw dirs.
            if getattr(self.config, "archive_old_checkpoints", False):
                archive_path = ckpt.parent / f"{ckpt.name}.tar.gz"
                if not archive_path.exists():
                    with tarfile.open(archive_path, "w:gz") as tar:
                        tar.add(ckpt, arcname=ckpt.name)

            # E5 fix: shutil is now imported at module level
            shutil.rmtree(ckpt)
