#!/usr/bin/env python3
"""
Multi-Head Training Utilities

Provides loss weighting and sampling strategies for multi-head training
to prevent text head from drowning out rare heads (emotion/paralinguistics/singing).

Per Blocker 4 from TRAINING_PLAN_REVIEW_2026-01-02-15-05.md:
- Add per-head loss weights
- Use temperature sampling / task-balanced batches

Usage:
    from tools.whisper_mlx.multi_head_training import (
        MultiHeadLossWeighter,
        TaskBalancedSampler,
    )

    # Configure loss weighting
    weighter = MultiHeadLossWeighter(
        weights={
            "text": 1.0,
            "emotion": 2.0,      # Upweight rare heads
            "paralinguistics": 2.0,
            "singing": 3.0,
        }
    )

    # In training loop:
    total_loss = weighter.compute_total_loss(head_losses)
"""

import random
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np


@dataclass
class MultiHeadLossConfig:
    """Configuration for multi-head loss weighting."""

    # Per-head loss weights (default to balanced)
    head_weights: dict[str, float] = field(default_factory=lambda: {
        "text": 1.0,
        "emotion": 1.0,
        "paralinguistics": 1.0,
        "pitch": 1.0,
        "singing": 1.0,
        "phoneme": 1.0,
    })

    # Dynamic weight adjustment based on head accuracy
    use_dynamic_weights: bool = False
    dynamic_weight_ema: float = 0.99  # EMA decay for accuracy tracking

    # Temperature for softmax-based weight normalization
    weight_temperature: float = 1.0

    # Gradient balancing (normalize gradients per head)
    use_gradient_balancing: bool = False
    gradient_clip_per_head: float = 1.0


class MultiHeadLossWeighter:
    """
    Computes weighted multi-head loss.

    Supports:
    - Static per-head weights
    - Dynamic weight adjustment based on head performance
    - Gradient balancing to prevent dominant heads
    """

    def __init__(self, config: MultiHeadLossConfig | None = None, **kwargs):
        """
        Initialize loss weighter.

        Args:
            config: MultiHeadLossConfig or None for defaults
            **kwargs: Override config fields
        """
        self.config = config or MultiHeadLossConfig()

        # Allow kwargs to override config
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            elif key == "weights":
                self.config.head_weights = value

        # Tracking for dynamic weights
        self._head_accuracies: dict[str, float] = {}
        self._head_losses: dict[str, float] = {}

    def compute_total_loss(
        self,
        head_losses: dict[str, mx.array],
        head_accuracies: dict[str, float] | None = None,
    ) -> tuple[mx.array, dict[str, float]]:
        """
        Compute weighted total loss.

        Args:
            head_losses: Dict of head_name -> loss value
            head_accuracies: Optional dict of head_name -> accuracy (for dynamic weights)

        Returns:
            (total_loss, effective_weights) tuple
        """
        if self.config.use_dynamic_weights and head_accuracies:
            weights = self._compute_dynamic_weights(head_accuracies)
        else:
            weights = self.config.head_weights.copy()

        # Normalize weights
        total_weight = sum(weights.get(name, 0) for name in head_losses.keys())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        # Apply temperature
        if self.config.weight_temperature != 1.0:
            weights = self._apply_temperature(weights)

        # Compute weighted sum
        total_loss = mx.array(0.0)
        effective_weights = {}

        for head_name, loss in head_losses.items():
            weight = weights.get(head_name, 1.0)
            effective_weights[head_name] = weight
            total_loss = total_loss + weight * loss

            # Update tracking
            self._head_losses[head_name] = float(loss)

        return total_loss, effective_weights

    def _compute_dynamic_weights(self, accuracies: dict[str, float]) -> dict[str, float]:
        """
        Compute dynamic weights based on head accuracy.

        Heads with lower accuracy get higher weights (focus on weak spots).
        """
        weights = self.config.head_weights.copy()

        for head_name, accuracy in accuracies.items():
            # Update EMA
            prev_acc = self._head_accuracies.get(head_name, 0.5)
            new_acc = self.config.dynamic_weight_ema * prev_acc + \
                      (1 - self.config.dynamic_weight_ema) * accuracy
            self._head_accuracies[head_name] = new_acc

            # Inverse accuracy weighting: lower accuracy -> higher weight
            # Clamp to prevent extreme weights
            inv_acc = max(0.1, min(2.0, 1.0 / (new_acc + 0.1)))
            base_weight = weights.get(head_name, 1.0)
            weights[head_name] = base_weight * inv_acc

        return weights

    def _apply_temperature(self, weights: dict[str, float]) -> dict[str, float]:
        """Apply temperature scaling to weights (softmax-style)."""
        if not weights:
            return weights

        T = self.config.weight_temperature
        values = np.array(list(weights.values()))

        # Softmax with temperature
        exp_values = np.exp(values / T)
        softmax_values = exp_values / exp_values.sum()

        # Scale back to original sum
        original_sum = values.sum()
        scaled_values = softmax_values * original_sum

        return {k: float(v) for k, v in zip(weights.keys(), scaled_values, strict=False)}

    def get_stats(self) -> dict[str, any]:
        """Get current weighting stats."""
        return {
            "head_weights": self.config.head_weights,
            "head_accuracies": self._head_accuracies,
            "head_losses": self._head_losses,
        }


@dataclass
class SamplingConfig:
    """Configuration for task-balanced sampling."""

    # Sampling temperature (higher = more uniform, lower = more focused on rare)
    temperature: float = 1.0

    # Minimum samples per task per batch
    min_samples_per_task: int = 1

    # Task weights for sampling probability
    task_weights: dict[str, float] = field(default_factory=lambda: {
        "text": 1.0,
        "emotion": 2.0,
        "paralinguistics": 2.0,
        "singing": 3.0,
    })


class TaskBalancedSampler:
    """
    Samples batches to balance across tasks/heads.

    Prevents rare tasks (emotion, singing) from being drowned out by
    abundant text samples.
    """

    def __init__(self, config: SamplingConfig | None = None, **kwargs):
        self.config = config or SamplingConfig()

        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # Track samples per task
        self._task_counts: dict[str, int] = {}
        self._epoch_samples: dict[str, list[int]] = {}

    def register_samples(
        self,
        samples: list[dict],
        task_field: str = "task",
    ):
        """
        Register available samples for each task.

        Args:
            samples: List of sample dicts
            task_field: Field name containing task label
        """
        self._epoch_samples = {}
        self._task_counts = {}

        for i, sample in enumerate(samples):
            task = sample.get(task_field, "text")
            if task not in self._epoch_samples:
                self._epoch_samples[task] = []
                self._task_counts[task] = 0
            self._epoch_samples[task].append(i)
            self._task_counts[task] += 1

    def sample_batch(self, batch_size: int) -> list[int]:
        """
        Sample a task-balanced batch of indices.

        Returns:
            List of sample indices
        """
        if not self._epoch_samples:
            return list(range(batch_size))

        # Compute sampling probabilities
        tasks = list(self._epoch_samples.keys())
        weights = []

        for task in tasks:
            base_weight = self.config.task_weights.get(task, 1.0)
            count = len(self._epoch_samples[task])
            # Inverse frequency weighting
            inv_freq = 1.0 / (count + 1)
            weights.append(base_weight * inv_freq)

        # Apply temperature
        weights = np.array(weights)
        weights = np.power(weights, 1.0 / self.config.temperature)
        probs = weights / weights.sum()

        # Sample tasks, then sample indices from each task
        batch_indices = []
        max(
            self.config.min_samples_per_task,
            batch_size // len(tasks),
        )

        for task, prob in zip(tasks, probs, strict=False):
            n_samples = max(1, int(batch_size * prob))
            n_samples = min(n_samples, len(self._epoch_samples[task]))

            task_indices = random.sample(self._epoch_samples[task], n_samples)
            batch_indices.extend(task_indices)

        # Trim or pad to exact batch size
        if len(batch_indices) > batch_size:
            batch_indices = random.sample(batch_indices, batch_size)
        elif len(batch_indices) < batch_size:
            # Pad with random samples
            all_indices = [i for indices in self._epoch_samples.values() for i in indices]
            remaining = batch_size - len(batch_indices)
            batch_indices.extend(random.sample(all_indices, remaining))

        random.shuffle(batch_indices)
        return batch_indices

    def get_task_distribution(self) -> dict[str, float]:
        """Get current task distribution."""
        total = sum(self._task_counts.values())
        if total == 0:
            return {}
        return {task: count / total for task, count in self._task_counts.items()}


def compute_gradient_norms(grads: dict[str, mx.array]) -> dict[str, float]:
    """
    Compute gradient norms per head for gradient balancing.

    Args:
        grads: Dict of parameter_name -> gradient

    Returns:
        Dict of head_name -> gradient norm
    """
    head_norms = {}

    for param_name, grad in grads.items():
        # Extract head name from parameter name (e.g., "emotion_head.fc.weight" -> "emotion")
        parts = param_name.split(".")
        if len(parts) > 0:
            head_name = parts[0].replace("_head", "")

            norm = float(mx.sqrt(mx.sum(grad * grad)))
            if head_name not in head_norms:
                head_norms[head_name] = 0.0
            head_norms[head_name] += norm

    return head_norms


def balance_gradients(
    grads: dict[str, mx.array],
    target_norm: float = 1.0,
) -> dict[str, mx.array]:
    """
    Balance gradients across heads to prevent dominant heads.

    Scales gradients so each head contributes equally to the update.

    Args:
        grads: Dict of parameter_name -> gradient
        target_norm: Target gradient norm per head

    Returns:
        Balanced gradients
    """
    head_norms = compute_gradient_norms(grads)
    balanced_grads = {}

    for param_name, grad in grads.items():
        parts = param_name.split(".")
        if len(parts) > 0:
            head_name = parts[0].replace("_head", "")
            head_norm = head_norms.get(head_name, 1.0)

            if head_norm > 0:
                scale = target_norm / head_norm
                # Clamp scale to prevent extreme values
                scale = max(0.1, min(10.0, scale))
                balanced_grads[param_name] = grad * scale
            else:
                balanced_grads[param_name] = grad
        else:
            balanced_grads[param_name] = grad

    return balanced_grads


# Convenience function for quick setup
def create_multi_head_training_config(
    text_weight: float = 1.0,
    emotion_weight: float = 2.0,
    para_weight: float = 2.0,
    singing_weight: float = 3.0,
    phoneme_weight: float = 1.5,
    use_dynamic: bool = False,
    use_gradient_balance: bool = False,
) -> tuple[MultiHeadLossConfig, SamplingConfig]:
    """
    Create standard multi-head training configuration.

    Returns:
        (loss_config, sampling_config) tuple
    """
    loss_config = MultiHeadLossConfig(
        head_weights={
            "text": text_weight,
            "emotion": emotion_weight,
            "paralinguistics": para_weight,
            "singing": singing_weight,
            "phoneme": phoneme_weight,
        },
        use_dynamic_weights=use_dynamic,
        use_gradient_balancing=use_gradient_balance,
    )

    sampling_config = SamplingConfig(
        task_weights={
            "text": text_weight,
            "emotion": emotion_weight,
            "paralinguistics": para_weight,
            "singing": singing_weight,
        },
    )

    return loss_config, sampling_config
