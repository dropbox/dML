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
Learning rate schedulers for Zipformer ASR training.

Implements warmup + decay schedules critical for stable training of
streaming Zipformer-large models.
"""

import math

from .config import SchedulerConfig


class WarmupScheduler:
    """
    Learning rate scheduler with warmup and decay.

    Supports:
    - Linear warmup from 0 to base_lr
    - Constant, cosine, or linear decay after warmup
    - Optional cooldown phase at the end

    Based on the scheduler used in k2-fsa/icefall.
    """

    def __init__(
        self,
        base_lr: float,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
        cooldown_steps: int = 0,
        scheduler_type: str = "warmup_cosine",
    ):
        """
        Initialize scheduler.

        Args:
            base_lr: Peak learning rate after warmup.
            warmup_steps: Number of warmup steps.
            total_steps: Total training steps.
            min_lr: Minimum learning rate.
            cooldown_steps: Steps for final linear cooldown.
            scheduler_type: Type of decay: "warmup_constant", "warmup_cosine", "warmup_linear".
        """
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.cooldown_steps = cooldown_steps
        self.scheduler_type = scheduler_type

        # Decay steps (between warmup and cooldown)
        self.decay_steps = total_steps - warmup_steps - cooldown_steps

        self._step = 0

    def get_lr(self, step: int | None = None) -> float:
        """
        Get learning rate for given step.

        Args:
            step: Step number. If None, uses internal step counter.

        Returns:
            Learning rate for the given step.
        """
        if step is None:
            step = self._step

        # Warmup phase
        if step < self.warmup_steps:
            return self.base_lr * (step + 1) / self.warmup_steps

        # Cooldown phase (linear decay to min_lr)
        if step >= self.total_steps - self.cooldown_steps:
            cooldown_step = step - (self.total_steps - self.cooldown_steps)
            progress = cooldown_step / max(self.cooldown_steps, 1)
            current_lr = self._get_decay_lr(self.total_steps - self.cooldown_steps - 1)
            return current_lr - progress * (current_lr - self.min_lr)

        # Decay phase
        return self._get_decay_lr(step)

    def _get_decay_lr(self, step: int) -> float:
        """Get LR during decay phase."""
        decay_step = step - self.warmup_steps
        progress = decay_step / max(self.decay_steps, 1)

        if self.scheduler_type == "warmup_constant":
            return self.base_lr

        if self.scheduler_type == "warmup_cosine":
            # Cosine annealing
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return self.min_lr + (self.base_lr - self.min_lr) * cosine_decay

        if self.scheduler_type == "warmup_linear":
            # Linear decay
            return self.base_lr - progress * (self.base_lr - self.min_lr)

        raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")

    def step(self) -> float:
        """Increment step and return new learning rate."""
        lr = self.get_lr()
        self._step += 1
        return lr

    @property
    def current_lr(self) -> float:
        """Current learning rate."""
        return self.get_lr()

    @property
    def current_step(self) -> int:
        """Current step."""
        return self._step

    def state_dict(self) -> dict:
        """Get scheduler state for checkpointing."""
        return {
            "step": self._step,
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr": self.min_lr,
            "cooldown_steps": self.cooldown_steps,
            "scheduler_type": self.scheduler_type,
        }

    def load_state_dict(self, state: dict) -> None:
        """Load scheduler state from checkpoint."""
        self._step = state["step"]


class NoamScheduler:
    """
    Noam learning rate scheduler (Transformer paper).

    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))

    Alternative to warmup scheduler, sometimes used for transformers.
    """

    def __init__(
        self,
        d_model: int,
        warmup_steps: int,
        scale: float = 1.0,
    ):
        """
        Initialize Noam scheduler.

        Args:
            d_model: Model dimension (for scaling).
            warmup_steps: Number of warmup steps.
            scale: Additional scaling factor.
        """
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.scale = scale
        self._step = 0

    def get_lr(self, step: int | None = None) -> float:
        """Get learning rate for given step."""
        if step is None:
            step = self._step

        step = max(step, 1)  # Avoid division by zero

        return self.scale * (
            self.d_model ** (-0.5)
            * min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
        )

    def step(self) -> float:
        """Increment step and return new learning rate."""
        lr = self.get_lr()
        self._step += 1
        return lr

    @property
    def current_lr(self) -> float:
        """Current learning rate."""
        return self.get_lr()

    @property
    def current_step(self) -> int:
        """Current step."""
        return self._step

    def state_dict(self) -> dict:
        """Get scheduler state for checkpointing."""
        return {
            "step": self._step,
            "d_model": self.d_model,
            "warmup_steps": self.warmup_steps,
            "scale": self.scale,
        }

    def load_state_dict(self, state: dict) -> None:
        """Load scheduler state from checkpoint."""
        self._step = state["step"]


def get_scheduler(config: SchedulerConfig, base_lr: float) -> WarmupScheduler:
    """
    Create scheduler from config.

    Args:
        config: Scheduler configuration.
        base_lr: Base learning rate from optimizer config.

    Returns:
        Configured scheduler instance.
    """
    return WarmupScheduler(
        base_lr=base_lr,
        warmup_steps=config.warmup_steps,
        total_steps=config.total_steps,
        min_lr=config.min_lr,
        cooldown_steps=config.cooldown_steps,
        scheduler_type=config.scheduler_type,
    )
