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
Speaker embedding trainer with AAM-Softmax loss.

Trains DELULU-style speaker embeddings targeting <0.8% EER.
"""

# Import speaker head and loss from models
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from .config import OptimizerConfig, SchedulerConfig, SpeakerTrainingConfig
from .scheduler import get_scheduler
from .speaker_dataloader import (
    CNCelebDataset,
    CombinedSpeakerDataset,
    LibriSpeechSpeakerDataset,
    SpeakerDataLoader,
    SpeakerDataset,
    VerificationTrialLoader,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.heads.speaker import (
    SpeakerConfig,
    SpeakerHead,
    aam_softmax_loss,
    verification_eer,
)


@dataclass
class SpeakerTrainingState:
    """Training state for speaker embedding model."""

    step: int = 0
    epoch: int = 0
    best_eer: float = 1.0
    total_loss: float = 0.0
    num_batches: int = 0


@dataclass
class SpeakerTrainingMetrics:
    """Metrics from a training step or validation."""

    loss: float
    accuracy: float = 0.0
    eer: float | None = None
    num_samples: int = 0
    time_ms: float = 0.0


class SpeakerTrainer:
    """
    Trainer for DELULU-style speaker embeddings.

    Supports:
    - AAM-Softmax classification training
    - Periodic EER validation
    - Checkpoint saving/loading
    - Learning rate scheduling
    """

    def __init__(
        self,
        speaker_head: SpeakerHead,
        encoder: nn.Module | None = None,
        config: SpeakerTrainingConfig | None = None,
        optimizer_config: OptimizerConfig | None = None,
        scheduler_config: SchedulerConfig | None = None,
    ):
        """
        Initialize speaker trainer.

        Args:
            speaker_head: DELULU speaker embedding head.
            encoder: Optional encoder (e.g., Zipformer). If None, trains
                     speaker head standalone with raw features.
            config: Speaker training configuration.
            optimizer_config: Optimizer configuration.
            scheduler_config: Learning rate scheduler configuration.
        """
        self.speaker_head = speaker_head
        self.encoder = encoder
        self.config = config or SpeakerTrainingConfig()
        self.optimizer_config = optimizer_config or OptimizerConfig(
            learning_rate=1e-3,
            weight_decay=1e-4,
        )
        self.scheduler_config = scheduler_config or SchedulerConfig(
            scheduler_type="warmup_cosine",
            warmup_steps=5000,
            total_steps=100000,
        )

        # Initialize optimizer
        self.optimizer = self._create_optimizer()

        # Create scheduler
        self.scheduler = get_scheduler(
            self.scheduler_config,
            self.optimizer_config.learning_rate,
        )

        # Training state
        self.state = SpeakerTrainingState()

        # Checkpoint directory
        self.checkpoint_dir = Path("checkpoints/speaker_embedding")

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer for speaker head (and encoder if provided)."""
        if self.optimizer_config.optimizer_type == "adamw":
            return optim.AdamW(
                learning_rate=self.optimizer_config.learning_rate,
                betas=[self.optimizer_config.beta1, self.optimizer_config.beta2],
                eps=self.optimizer_config.eps,
                weight_decay=self.optimizer_config.weight_decay,
            )
        if self.optimizer_config.optimizer_type == "sgd":
            return optim.SGD(
                learning_rate=self.optimizer_config.learning_rate,
                momentum=0.9,
                weight_decay=self.optimizer_config.weight_decay,
            )
        raise ValueError(
            f"Unknown optimizer: {self.optimizer_config.optimizer_type}",
        )

    def _get_trainable_params(self) -> dict:
        """Get trainable parameters from speaker head and encoder."""
        params = {"speaker_head": self.speaker_head.trainable_parameters()}
        if self.encoder is not None:
            params["encoder"] = self.encoder.trainable_parameters()
        return params

    def _forward(
        self,
        features: mx.array,
        feature_lengths: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """
        Forward pass through encoder and speaker head.

        Args:
            features: Input features (batch, time, feat_dim).
            feature_lengths: Sequence lengths (batch,).

        Returns:
            Tuple of (embeddings, logits).
        """
        # If we have an encoder, use it
        if self.encoder is not None:
            encoder_out = self.encoder(features, feature_lengths)
            if isinstance(encoder_out, tuple):
                encoder_out = encoder_out[0]  # Get primary output
        else:
            # Use features directly (for standalone training)
            encoder_out = features

        # Forward through speaker head
        embeddings, logits = self.speaker_head(
            encoder_out,
            encoder_lengths=feature_lengths,
            return_embeddings_only=False,
        )

        return embeddings, logits

    def train_step(
        self,
        batch: dict[str, mx.array],
    ) -> SpeakerTrainingMetrics:
        """
        Perform a single training step.

        Args:
            batch: Batch dictionary with features, feature_lengths, speaker_indices.

        Returns:
            Training metrics for this step.
        """
        start_time = time.time()

        features = batch["features"]
        feature_lengths = batch["feature_lengths"]
        speaker_indices = batch["speaker_indices"]

        # Flatten speaker indices if needed (batch, 1) -> (batch,)
        if speaker_indices.ndim > 1:
            speaker_indices = speaker_indices.reshape(-1)

        # Store logits for accuracy computation (to avoid double forward pass)
        logits_holder = [None]

        def loss_fn(params):
            # Update parameters
            self.speaker_head.update(params.get("speaker_head", {}))
            if self.encoder is not None and "encoder" in params:
                self.encoder.update(params["encoder"])

            # Forward pass
            embeddings, logits = self._forward(features, feature_lengths)

            # Store logits for accuracy computation (stop gradient)
            logits_holder[0] = mx.stop_gradient(logits)

            # AAM-Softmax loss
            loss = aam_softmax_loss(
                embeddings=embeddings,
                logits=logits,
                targets=speaker_indices,
                margin=self.config.aam_margin,
                scale=self.config.aam_scale,
            )

            return loss

        # Get parameters
        params = self._get_trainable_params()

        # Compute loss and gradients using MLX's value_and_grad
        loss_and_grad_fn = mx.value_and_grad(loss_fn)
        loss, grads = loss_and_grad_fn(params)

        # Compute accuracy from stored logits (no extra forward pass)
        logits = logits_holder[0]
        preds = mx.argmax(logits, axis=-1)
        correct = mx.sum(preds == speaker_indices)
        accuracy = correct / speaker_indices.shape[0]

        # Clip gradients
        if self.optimizer_config.grad_clip > 0:
            grads = self._clip_gradients(grads, self.optimizer_config.grad_clip)

        # Update learning rate
        lr = self.scheduler.get_lr(self.state.step)
        self.optimizer.learning_rate = lr

        # Apply gradients
        self.optimizer.update(self.speaker_head, grads.get("speaker_head", {}))
        if self.encoder is not None and "encoder" in grads:
            self.optimizer.update(self.encoder, grads["encoder"])

        # Evaluate to get actual values
        mx.eval(loss, accuracy)

        # Clear cache periodically to prevent memory buildup
        if self.state.step % 100 == 0:
            mx.clear_cache()

        # Update state
        self.state.step += 1
        self.state.total_loss += float(loss)
        self.state.num_batches += 1

        elapsed_ms = (time.time() - start_time) * 1000

        return SpeakerTrainingMetrics(
            loss=float(loss),
            accuracy=float(accuracy),
            num_samples=features.shape[0],
            time_ms=elapsed_ms,
        )

    def _clip_gradients(
        self,
        grads: dict,
        max_norm: float,
    ) -> dict:
        """Clip gradients by global norm."""
        # Flatten all gradients
        flat_grads = []
        for key in grads:
            if isinstance(grads[key], dict):
                for subkey in grads[key]:
                    if isinstance(grads[key][subkey], mx.array):
                        flat_grads.append(grads[key][subkey].reshape(-1))
            elif isinstance(grads[key], mx.array):
                flat_grads.append(grads[key].reshape(-1))

        if not flat_grads:
            return grads

        # Compute global norm
        total_norm = mx.sqrt(
            sum(mx.sum(g * g) for g in flat_grads),
        )

        # Clip if necessary
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef = mx.minimum(clip_coef, mx.array(1.0))

        # Apply clipping
        def clip_grad(g):
            if isinstance(g, mx.array):
                return g * clip_coef
            if isinstance(g, dict):
                return {k: clip_grad(v) for k, v in g.items()}
            return g

        return {k: clip_grad(v) for k, v in grads.items()}

    def validate(
        self,
        val_loader: SpeakerDataLoader,
        trial_loader: VerificationTrialLoader | None = None,
        max_batches: int = 100,
    ) -> SpeakerTrainingMetrics:
        """
        Run validation and optionally compute EER.

        Args:
            val_loader: Validation data loader.
            trial_loader: Optional trial loader for EER computation.
            max_batches: Maximum batches to process.

        Returns:
            Validation metrics including EER if trial_loader provided.
        """
        start_time = time.time()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # Classification validation
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break

            features = batch["features"]
            feature_lengths = batch["feature_lengths"]
            speaker_indices = batch["speaker_indices"]

            # Flatten speaker indices if needed (batch, 1) -> (batch,)
            if speaker_indices.ndim > 1:
                speaker_indices = speaker_indices.reshape(-1)

            # Forward pass (no gradients needed)
            embeddings, logits = self._forward(features, feature_lengths)

            # Compute loss
            loss = aam_softmax_loss(
                embeddings=embeddings,
                logits=logits,
                targets=speaker_indices,
                margin=self.config.aam_margin,
                scale=self.config.aam_scale,
            )

            # Compute accuracy
            preds = mx.argmax(logits, axis=-1)
            correct = mx.sum(preds == speaker_indices)

            mx.eval(loss, correct)

            total_loss += float(loss) * features.shape[0]
            total_correct += int(correct)
            total_samples += features.shape[0]

        avg_loss = total_loss / max(total_samples, 1)
        accuracy = total_correct / max(total_samples, 1)

        # EER computation
        eer = None
        if trial_loader is not None:
            eer = self._compute_eer(trial_loader)

        elapsed_ms = (time.time() - start_time) * 1000

        return SpeakerTrainingMetrics(
            loss=avg_loss,
            accuracy=accuracy,
            eer=eer,
            num_samples=total_samples,
            time_ms=elapsed_ms,
        )

    def _compute_eer(
        self,
        trial_loader: VerificationTrialLoader,
        max_trials: int = 200,
    ) -> float:
        """
        Compute Equal Error Rate on verification trials.

        Args:
            trial_loader: Trial pair generator.
            max_trials: Maximum number of trials to use (to manage memory).
                       Reduced from 1000 to 200 to prevent Metal memory exhaustion.

        Returns:
            EER as a float (0.0 to 1.0).
        """
        import gc
        trials = list(trial_loader.generate_trials())

        # Limit trials to prevent memory exhaustion
        if len(trials) > max_trials:
            import random
            trials = random.sample(trials, max_trials)

        similarities = []
        labels = []

        # Clear memory at start
        mx.clear_cache()
        gc.collect()

        for i, (idx1, idx2, label) in enumerate(trials):
            # Get samples
            sample1 = trial_loader.dataset[idx1]
            sample2 = trial_loader.dataset[idx2]

            # Get embeddings - evaluate immediately to free graph
            emb1 = self._get_embedding(sample1)
            mx.eval(emb1)
            emb2 = self._get_embedding(sample2)
            mx.eval(emb2)

            # Compute similarity
            sim = self.speaker_head.similarity(emb1, emb2)
            mx.eval(sim)

            similarities.append(float(sim))
            labels.append(label)

            # Clear memory every 20 trials to prevent exhaustion
            if (i + 1) % 20 == 0:
                mx.clear_cache()
                gc.collect()

        # Final cleanup before EER computation
        mx.clear_cache()
        gc.collect()

        # Compute EER
        eer = verification_eer(
            mx.array(similarities),
            mx.array(labels),
        )

        return eer

    def _get_embedding(self, sample: dict[str, mx.array]) -> mx.array:
        """Get embedding for a single sample."""
        features = sample["features"]
        if features.ndim == 2:
            features = mx.expand_dims(features, axis=0)

        feature_lengths = sample["feature_lengths"]
        if feature_lengths.ndim == 0:
            feature_lengths = mx.expand_dims(feature_lengths, axis=0)

        # Forward through encoder if available
        if self.encoder is not None:
            encoder_out = self.encoder(features, feature_lengths)
            if isinstance(encoder_out, tuple):
                encoder_out = encoder_out[0]
        else:
            encoder_out = features

        # Get embedding
        embedding = self.speaker_head.extract_embedding(
            encoder_out, feature_lengths,
        )

        return embedding

    def train_epoch(
        self,
        train_loader: SpeakerDataLoader,
        val_loader: SpeakerDataLoader | None = None,
        trial_loader: VerificationTrialLoader | None = None,
        log_every: int = 100,
        val_every: int = 1000,
    ) -> dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader.
            val_loader: Optional validation data loader.
            trial_loader: Optional trial loader for EER.
            log_every: Log every N steps.
            val_every: Validate every N steps.

        Returns:
            Dictionary of epoch metrics.
        """
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0

        for batch in train_loader:
            metrics = self.train_step(batch)

            epoch_loss += metrics.loss
            epoch_accuracy += metrics.accuracy
            num_batches += 1

            # Logging
            if self.state.step % log_every == 0:
                avg_loss = epoch_loss / num_batches
                epoch_accuracy / num_batches
                lr = self.scheduler.get_lr(self.state.step)
                print(
                    f"Step {self.state.step} | "
                    f"Loss: {metrics.loss:.4f} | "
                    f"Acc: {metrics.accuracy:.4f} | "
                    f"Avg Loss: {avg_loss:.4f} | "
                    f"LR: {lr:.2e} | "
                    f"Time: {metrics.time_ms:.1f}ms",
                )

            # Validation (skip EER during training for speed)
            if val_loader is not None and self.state.step % val_every == 0:
                # Skip EER computation during training - too slow
                val_metrics = self.validate(val_loader, trial_loader=None)
                print(
                    f"Validation | "
                    f"Loss: {val_metrics.loss:.4f} | "
                    f"Acc: {val_metrics.accuracy:.4f}",
                )

                # Save checkpoint periodically (based on validation accuracy)
                # Best model tracking uses accuracy until we have proper EER
                self.save_checkpoint("best")
                print(f"Saved checkpoint at step {self.state.step}")

        self.state.epoch += 1

        return {
            "loss": epoch_loss / max(num_batches, 1),
            "accuracy": epoch_accuracy / max(num_batches, 1),
            "best_eer": self.state.best_eer,
        }

    def save_checkpoint(self, name: str = "latest") -> Path:
        """
        Save model checkpoint.

        Args:
            name: Checkpoint name.

        Returns:
            Path to saved checkpoint.
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / f"speaker_{name}.safetensors"

        def flatten_dict(d: dict, prefix: str = "") -> dict[str, mx.array]:
            """Recursively flatten nested dicts to key.subkey format."""
            result = {}
            for key, value in d.items():
                full_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    result.update(flatten_dict(value, full_key))
                elif isinstance(value, mx.array):
                    result[full_key] = value
            return result

        # Collect and flatten weights
        flat_weights = {}
        flat_weights.update(
            flatten_dict(dict(self.speaker_head.parameters()), "speaker_head"),
        )
        if self.encoder is not None:
            flat_weights.update(
                flatten_dict(dict(self.encoder.parameters()), "encoder"),
            )

        mx.save_safetensors(str(path), flat_weights)

        # Save training state
        state_path = self.checkpoint_dir / f"speaker_{name}_state.json"
        import json
        with open(state_path, "w") as f:
            json.dump({
                "step": self.state.step,
                "epoch": self.state.epoch,
                "best_eer": self.state.best_eer,
            }, f)

        print(f"Saved checkpoint to {path}")
        return path

    def load_checkpoint(self, name: str = "latest") -> bool:
        """
        Load model checkpoint.

        Args:
            name: Checkpoint name.

        Returns:
            True if loaded successfully.
        """
        path = self.checkpoint_dir / f"speaker_{name}.safetensors"
        if not path.exists():
            print(f"No checkpoint found at {path}")
            return False

        # Load weights
        flat_weights = mx.load(str(path))

        def unflatten_dict(flat: dict[str, mx.array], prefix: str) -> dict:
            """Convert flat keys to nested dict matching model structure."""
            result = {}
            prefix_len = len(prefix) + 1  # +1 for the dot

            for key, value in flat.items():
                if not key.startswith(prefix + "."):
                    continue

                # Get key without prefix
                subkey = key[prefix_len:]
                # Split into parts and build nested dict
                parts = subkey.split(".")
                current = result

                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]

                current[parts[-1]] = value

            return result

        # Unflatten and update models
        speaker_weights = unflatten_dict(flat_weights, "speaker_head")
        if speaker_weights:
            self.speaker_head.update(speaker_weights)

        encoder_weights = unflatten_dict(flat_weights, "encoder")
        if self.encoder is not None and encoder_weights:
            self.encoder.update(encoder_weights)

        # Load training state
        state_path = self.checkpoint_dir / f"speaker_{name}_state.json"
        if state_path.exists():
            import json
            with open(state_path) as f:
                state = json.load(f)
                self.state.step = state.get("step", 0)
                self.state.epoch = state.get("epoch", 0)
                self.state.best_eer = state.get("best_eer", 1.0)

        print(f"Loaded checkpoint from {path}")
        return True


def create_speaker_trainer(
    config: SpeakerTrainingConfig,
    encoder: nn.Module | None = None,
    optimizer_config: OptimizerConfig | None = None,
    scheduler_config: SchedulerConfig | None = None,
) -> SpeakerTrainer:
    """
    Create a speaker trainer from configuration.

    Args:
        config: Speaker training configuration.
        encoder: Optional encoder model.
        optimizer_config: Optimizer configuration.
        scheduler_config: Scheduler configuration.

    Returns:
        Configured SpeakerTrainer.
    """
    # Create speaker head config
    speaker_config = SpeakerConfig(
        encoder_dim=config.encoder_dim,
        embedding_dim=config.embedding_dim,
        num_speakers=config.num_speakers,
        aam_margin=config.aam_margin,
        aam_scale=config.aam_scale,
        use_ssl=config.use_ssl,
        mask_ratio=config.mask_ratio,
        denoise_prob=config.denoise_prob,
    )

    # Create speaker head
    speaker_head = SpeakerHead(speaker_config)

    return SpeakerTrainer(
        speaker_head=speaker_head,
        encoder=encoder,
        config=config,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
    )


def create_speaker_datasets(
    config: SpeakerTrainingConfig,
) -> tuple[SpeakerDataset, SpeakerDataset, VerificationTrialLoader]:
    """
    Create training and validation datasets from configuration.

    Args:
        config: Speaker training configuration.

    Returns:
        Tuple of (train_dataset, val_dataset, trial_loader).
    """
    datasets = []

    # Try CN-Celeb
    cnceleb_dir = Path(config.cnceleb_dir)
    if cnceleb_dir.exists():
        try:
            train_cnceleb = CNCelebDataset(str(cnceleb_dir), split="train")
            datasets.append(train_cnceleb)
            print(f"Loaded CN-Celeb: {len(train_cnceleb)} samples, {train_cnceleb.num_speakers} speakers")
        except Exception as e:
            print(f"Warning: Could not load CN-Celeb: {e}")

    # Try LibriSpeech speakers
    if config.use_librispeech_speakers:
        libri_dir = Path(config.librispeech_dir)
        if libri_dir.exists():
            try:
                train_libri = LibriSpeechSpeakerDataset(str(libri_dir))
                datasets.append(train_libri)
                print(f"Loaded LibriSpeech: {len(train_libri)} samples, {train_libri.num_speakers} speakers")
            except Exception as e:
                print(f"Warning: Could not load LibriSpeech: {e}")

    if not datasets:
        raise ValueError("No speaker datasets found")

    # Combine datasets
    if len(datasets) > 1:
        train_dataset = CombinedSpeakerDataset(datasets)
    else:
        train_dataset = datasets[0]

    # Update config with actual speaker count
    config.num_speakers = train_dataset.num_speakers
    print(f"Total: {len(train_dataset)} samples, {train_dataset.num_speakers} speakers")

    # Validation dataset (CN-Celeb dev or subset of train)
    val_dataset = None
    if cnceleb_dir.exists():
        try:
            val_dataset = CNCelebDataset(str(cnceleb_dir), split="dev")
            print(f"Loaded CN-Celeb dev: {len(val_dataset)} samples")
        except Exception:
            pass

    if val_dataset is None:
        # Use subset of train for validation
        val_dataset = train_dataset

    # Trial loader for EER
    trial_loader = VerificationTrialLoader(
        val_dataset,
        num_positive_pairs=1000,
        num_negative_pairs=1000,
    )

    return train_dataset, val_dataset, trial_loader
