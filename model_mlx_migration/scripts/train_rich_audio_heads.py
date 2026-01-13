#!/usr/bin/env python3
# Copyright 2024-2026 Andrew Yates
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
Rich Audio Heads Training Script

Trains emotion, paralinguistics, and other rich audio heads on a frozen
Zipformer encoder. This script is separate from the main ASR training
because rich audio datasets don't have ASR targets.

Usage:
    python scripts/train_rich_audio_heads.py \\
        --checkpoint checkpoints/zipformer/en-streaming/exp/pretrained.pt \\
        --head emotion \\
        --data-dir data/emotion/crema-d \\
        --output-dir checkpoints/rich_audio_heads/emotion_v1 \\
        --epochs 20

    python scripts/train_rich_audio_heads.py \\
        --checkpoint checkpoints/zipformer/en-streaming/exp/pretrained.pt \\
        --head paralinguistics \\
        --data-dir data/paralinguistics/vocalsound_labeled \\
        --output-dir checkpoints/rich_audio_heads/paralinguistics_v1 \\
        --epochs 20
"""

import argparse
import fcntl
import json
import logging
import math
import os
import random
import sys
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

# Add project root to path
_PROJECT_ROOT = Path(__file__).parent.parent.absolute()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from src.models.heads import (
    EmotionConfig,
    EmotionHead,
    LanguageConfig,
    LanguageHead,
    ParalinguisticsConfig,
    ParalinguisticsHead,
)
from src.models.zipformer import load_checkpoint
from src.training import (
    EMOTION_LABELS,
    PARALINGUISTIC_LABELS,
    create_combined_emotion_loader,
    create_emotion_loader,
    create_meld_loader,
    create_paralinguistics_loader,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingArgs:
    """Command-line arguments for rich audio head training."""

    # Model
    checkpoint: str = "checkpoints/zipformer/en-streaming/exp/pretrained.pt"
    output_dir: str = "checkpoints/rich_audio_heads/v1"

    # Head selection
    head: str = "emotion"  # emotion, paralinguistics, language

    # Data
    data_dir: str = "data/emotion/crema-d"
    dataset: str = "crema-d"  # crema-d or combined (for emotion head)

    # Training
    epochs: int = 20
    batch_size: int = 16
    learning_rate: float = 1e-3
    encoder_lr: float = 1e-6  # Very low LR for pretrained encoder layers
    warmup_steps: int = 500
    weight_decay: float = 0.01
    grad_clip: float = 1.0  # Gradient clipping max norm

    # Encoder fine-tuning
    unfreeze_layers: int = 0  # Number of top encoder layers to unfreeze (0 = frozen)

    # Memory management (for encoder unfreezing stability)
    accumulation_steps: int = 1  # Gradient accumulation for effective batch size
    clear_cache_every: int = 100  # Clear MLX metal cache every N steps (0 = disabled)

    # Checkpointing
    save_every: int = 500
    eval_every: int = 100
    log_every: int = 10

    # Training quality
    label_smoothing: float = 0.1
    use_class_weights: bool = True  # Enable by default for imbalanced datasets
    focal_loss: bool = False  # Use focal loss for class imbalance
    focal_gamma: float = 2.0  # Focal loss focusing parameter (higher = more focus on hard examples)
    ema_decay: float = 0.0

    # Mixed precision control
    dtype: str = "float32"  # float32, float16, bfloat16

    # SpecAugment
    spec_augment: bool = True
    num_time_masks: int = 2
    time_mask_max: int = 50
    num_freq_masks: int = 2
    freq_mask_max: int = 20

    # Hardware
    seed: int = 42

    # Resume training
    resume: bool = False  # Resume from latest checkpoint in output_dir
    resume_from: str | None = None  # Specific checkpoint path to resume from


def parse_args() -> TrainingArgs:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train rich audio heads on frozen Zipformer encoder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/zipformer/en-streaming/exp/pretrained.pt")
    parser.add_argument("--output-dir", type=str,
                        default="checkpoints/rich_audio_heads/v1")
    parser.add_argument("--head", type=str, default="emotion",
                        choices=["emotion", "paralinguistics", "language"])
    parser.add_argument("--data-dir", type=str, default="data/emotion/crema-d")
    parser.add_argument("--dataset", type=str, default="crema-d",
                        choices=["crema-d", "combined", "meld"],
                        help="Dataset to use for emotion head (crema-d, combined, or meld)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--encoder-lr", type=float, default=1e-5,
                        help="Learning rate for unfrozen encoder layers")
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--unfreeze-layers", type=int, default=0,
                        help="Number of top encoder layers to unfreeze (0 = frozen)")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Maximum gradient norm for clipping")
    parser.add_argument("--accumulation-steps", type=int, default=1,
                        help="Gradient accumulation steps (effective batch = batch_size * accumulation)")
    parser.add_argument("--clear-cache-every", type=int, default=100,
                        help="Clear MLX metal cache every N steps (0 = disabled)")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                        help="Label smoothing for classification heads")
    parser.add_argument("--use-class-weights", action="store_true",
                        help="Use inverse-frequency class weights for imbalanced datasets")
    parser.add_argument("--focal-loss", action="store_true",
                        help="Use focal loss instead of cross-entropy (better for class imbalance)")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Focal loss focusing parameter (higher = more focus on hard examples)")
    parser.add_argument("--ema-decay", type=float, default=0.0,
                        help="EMA decay for head weights (0 disables EMA)")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"],
                        help="Parameter dtype for mixed precision control")
    parser.add_argument("--spec-augment", action="store_true", default=True,
                        help="Enable SpecAugment on training features")
    parser.add_argument("--no-spec-augment", action="store_false", dest="spec_augment",
                        help="Disable SpecAugment")
    parser.add_argument("--num-time-masks", type=int, default=2)
    parser.add_argument("--time-mask-max", type=int, default=50)
    parser.add_argument("--num-freq-masks", type=int, default=2)
    parser.add_argument("--freq-mask-max", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from latest checkpoint in output-dir")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Specific checkpoint path to resume from (e.g., head_step_8000.npz)")

    args = parser.parse_args()

    return TrainingArgs(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        head=args.head,
        data_dir=args.data_dir,
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        encoder_lr=args.encoder_lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        unfreeze_layers=args.unfreeze_layers,
        grad_clip=args.grad_clip,
        accumulation_steps=args.accumulation_steps,
        clear_cache_every=args.clear_cache_every,
        save_every=args.save_every,
        eval_every=args.eval_every,
        log_every=args.log_every,
        label_smoothing=args.label_smoothing,
        use_class_weights=args.use_class_weights,
        focal_loss=args.focal_loss,
        focal_gamma=args.focal_gamma,
        ema_decay=args.ema_decay,
        dtype=args.dtype,
        spec_augment=args.spec_augment,
        num_time_masks=args.num_time_masks,
        time_mask_max=args.time_mask_max,
        num_freq_masks=args.num_freq_masks,
        freq_mask_max=args.freq_mask_max,
        seed=args.seed,
        resume=args.resume,
        resume_from=args.resume_from,
    )


def create_head(head_type: str, encoder_dim: int, num_classes: int | None = None) -> nn.Module:
    """Create the appropriate head module."""
    if head_type == "emotion":
        if num_classes is None:
            num_classes = len(EMOTION_LABELS)
        config = EmotionConfig(
            encoder_dim=encoder_dim,
            num_classes=num_classes,
            hidden_dim=256,
            dropout_rate=0.1,
        )
        return EmotionHead(config)
    if head_type == "paralinguistics":
        if num_classes is None:
            num_classes = len(PARALINGUISTIC_LABELS)
        config = ParalinguisticsConfig(
            encoder_dim=encoder_dim,
            num_classes=num_classes,
            hidden_dim=256,
            dropout_rate=0.1,
        )
        return ParalinguisticsHead(config)
    if head_type == "language":
        config = LanguageConfig(
            encoder_dim=encoder_dim,
            num_languages=9,
            hidden_dim=256,
            dropout_rate=0.1,
        )
        return LanguageHead(config)
    raise ValueError(f"Unknown head type: {head_type}")


def create_dataloader(args: TrainingArgs, split: str) -> Iterator:
    """Create dataloader for the specified head type."""
    if args.head == "emotion":
        if args.dataset == "combined":
            # Use combined emotion HF dataset
            data_dir = args.data_dir
            if data_dir == "data/emotion/crema-d":
                # Use default combined path
                data_dir = "data/emotion/combined_emotion_hf"
            return create_combined_emotion_loader(
                data_dir,
                batch_size=args.batch_size,
                split=split if split == "train" else "validation",
                shuffle=(split == "train"),
            )
        if args.dataset == "meld":
            # Use MELD dataset (Friends TV show)
            data_dir = args.data_dir
            if data_dir == "data/emotion/crema-d":
                # Use default MELD path
                data_dir = "data/emotion_punctuation/MELD.Raw"
            return create_meld_loader(
                data_dir,
                batch_size=args.batch_size,
                split=split if split == "train" else "dev",
                shuffle=(split == "train"),
            )
        # Use CREMA-D dataset
        return create_emotion_loader(
            args.data_dir,
            batch_size=args.batch_size,
            split=split,
            shuffle=(split == "train"),
        )
    if args.head == "paralinguistics":
        return create_paralinguistics_loader(
            args.data_dir,
            batch_size=args.batch_size,
            split=split if split == "train" else "test",  # VocalSound uses test
            shuffle=(split == "train"),
        )
    raise ValueError(f"No dataloader for head type: {args.head}")


def get_label_key(head_type: str) -> str:
    """Get the batch key for labels based on head type."""
    if head_type == "emotion":
        return "emotion_labels"
    if head_type == "paralinguistics":
        return "paralinguistic_labels"
    if head_type == "language":
        return "language_labels"
    raise ValueError(f"Unknown head type: {head_type}")


def warmup_cosine_lr(step: int, warmup_steps: int, max_steps: int,
                     base_lr: float, min_lr: float = 1e-6) -> float:
    """Warmup + cosine decay learning rate schedule."""
    if step < warmup_steps:
        return base_lr * step / warmup_steps

    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    cosine_decay = 0.5 * (1 + mx.cos(mx.array(mx.pi * progress))).item()
    return min_lr + (base_lr - min_lr) * cosine_decay


class EncoderHeadModel(nn.Module):
    """Combined encoder + head model for fine-tuning.

    When unfreeze_layers > 0, gradients flow through top encoder layers.
    The encoder stages are numbered 0-5 (bottom to top), and we unfreeze
    from the top (stage 5 first, then 4, etc.)
    """

    def __init__(self, encoder, head, unfreeze_layers: int = 0):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.unfreeze_layers = unfreeze_layers

    def __call__(self, features, feature_lengths):
        """Forward pass with selective gradient freezing."""
        # Forward through encoder
        try:
            encoder_out, encoder_lengths = self.encoder(
                features,
                feature_lengths,
                unfreeze_layers=self.unfreeze_layers,
            )
        except TypeError:
            encoder_out, encoder_lengths = self.encoder(features, feature_lengths)
            if self.unfreeze_layers == 0:
                encoder_out = mx.stop_gradient(encoder_out)

        # Forward through head
        logits = self.head(encoder_out, encoder_lengths)
        return logits, encoder_out, encoder_lengths


def get_unfrozen_params(encoder, unfreeze_layers: int) -> dict:
    """Get parameters from top encoder stages to unfreeze.

    Zipformer has 6 encoder stages (0-5). We unfreeze from the top:
    - unfreeze_layers=1: stage 5 only
    - unfreeze_layers=2: stages 4-5
    - unfreeze_layers=3: stages 3-5
    etc.

    Returns dict of encoder parameters to train.
    """
    if unfreeze_layers <= 0:
        return {}

    params = {}
    num_stages = len(encoder.encoders)
    start_stage = max(0, num_stages - unfreeze_layers)

    for stage_idx in range(start_stage, num_stages):
        stage_key = f"encoders.{stage_idx}"
        stage_encoder = encoder.encoders[stage_idx]
        for name, param in stage_encoder.parameters().items():
            full_name = f"{stage_key}.{name}"
            params[full_name] = param

    return params


def clip_grad_norm(grads, max_norm: float = 1.0):
    """Clip gradient norm for stability."""
    def _clip_recursive(g):
        if isinstance(g, dict):
            return {k: _clip_recursive(v) for k, v in g.items()}
        if isinstance(g, mx.array):
            norm = mx.sqrt(mx.sum(g * g))
            scale = mx.minimum(mx.array(1.0), mx.array(max_norm) / (norm + 1e-6))
            return g * scale
        return g
    return _clip_recursive(grads)


def _tree_add(a, b):
    if isinstance(a, dict):
        return {k: _tree_add(a[k], b[k]) for k in a}
    if isinstance(a, list):
        return [_tree_add(x, y) for x, y in zip(a, b, strict=False)]
    if isinstance(a, mx.array):
        return a + b
    return a


def _tree_scale(a, scale: float):
    if isinstance(a, dict):
        return {k: _tree_scale(v, scale) for k, v in a.items()}
    if isinstance(a, list):
        return [_tree_scale(v, scale) for v in a]
    if isinstance(a, mx.array):
        return a * scale
    return a


def _tree_leaves(a) -> list[mx.array]:
    leaves: list[mx.array] = []

    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, mx.array):
            leaves.append(x)

    walk(a)
    return leaves


def _tree_eval(a) -> None:
    leaves = _tree_leaves(a)
    if leaves:
        mx.eval(*leaves)


def _is_nonfinite(x: mx.array) -> bool:
    return bool(mx.any(mx.isnan(x) | mx.isinf(x)).item())


def _tree_has_nonfinite(a) -> bool:
    for leaf in _tree_leaves(a):
        if _is_nonfinite(leaf):
            return True
    return False


def _ema_update(ema_params, params, decay: float):
    if isinstance(ema_params, dict):
        return {k: _ema_update(ema_params[k], params[k], decay) for k in ema_params}
    if isinstance(ema_params, list):
        return [_ema_update(e, p, decay) for e, p in zip(ema_params, params, strict=False)]
    if isinstance(ema_params, mx.array):
        if ema_params.dtype in (mx.float16, mx.float32, mx.bfloat16):
            return ema_params * decay + params * (1.0 - decay)
        return ema_params
    return ema_params


def focal_loss(
    logits: mx.array,
    labels: mx.array,
    gamma: float = 2.0,
    alpha: mx.array | None = None,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
) -> mx.array:
    """Compute focal loss for classification.

    Focal loss addresses class imbalance by down-weighting easy examples
    and focusing on hard examples. Defined as:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        logits: Raw logits with shape (batch_size, num_classes)
        labels: Integer class labels with shape (batch_size,)
        gamma: Focusing parameter (default 2.0). Higher values increase focus
               on hard examples. gamma=0 reduces to standard cross-entropy.
        alpha: Optional class weights with shape (num_classes,). If provided,
               they are applied as per-sample weights based on the true label.
        label_smoothing: Amount of label smoothing to apply (default 0.0).
        reduction: 'mean', 'sum', or 'none' (default 'mean').

    Returns:
        Focal loss value.

    Reference:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """
    num_classes = logits.shape[-1]
    batch_size = logits.shape[0]

    # Compute softmax probabilities
    probs = mx.softmax(logits, axis=-1)
    # Clamp for numerical stability
    probs = mx.clip(probs, 1e-10, 1.0 - 1e-10)
    log_probs = mx.log(probs)

    # Apply label smoothing if requested
    if label_smoothing > 0:
        # Create smoothed one-hot targets
        # Start with uniform distribution
        smooth_labels = mx.full((batch_size, num_classes), label_smoothing / num_classes)
        # Create one-hot for true labels
        one_hot = mx.zeros((batch_size, num_classes))
        one_hot = one_hot.at[mx.arange(batch_size), labels].add(1.0)
        # Combine: smooth + (1-smooth)*one_hot
        smooth_labels = smooth_labels + (1.0 - label_smoothing) * one_hot

        # Weighted focal loss with smoothed labels
        focal_weight = (1.0 - probs) ** gamma
        loss = -mx.sum(smooth_labels * focal_weight * log_probs, axis=-1)
    else:
        # Standard focal loss for hard labels
        # Get probability of the true class
        p_t = probs[mx.arange(batch_size), labels]
        # Get log probability of the true class
        log_p_t = log_probs[mx.arange(batch_size), labels]
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1.0 - p_t) ** gamma
        # Focal loss
        loss = -focal_weight * log_p_t

    # Apply class-based alpha weighting if provided
    if alpha is not None:
        alpha_t = alpha[labels]
        loss = alpha_t * loss

    # Apply reduction
    if reduction == "mean":
        return mx.mean(loss)
    if reduction == "sum":
        return mx.sum(loss)
    return loss


def _apply_spec_augment(
    features: mx.array,
    feature_lengths: mx.array,
    num_time_masks: int,
    time_mask_max: int,
    num_freq_masks: int,
    freq_mask_max: int,
) -> mx.array:
    """Apply SpecAugment (time/frequency masking) to fbank features.

    Uses efficient vectorized operations with numpy for random generation
    and MLX for the actual masking. Masked regions are filled with the
    per-sample mean feature value to avoid creating out-of-distribution
    inputs (filling with literal zeros can trigger NaNs in the Zipformer
    encoder on some batches).
    """
    import numpy as np

    batch_size, max_time, num_bins = features.shape

    # Use numpy for random generation (fast), then convert mask to MLX
    np_mask = np.ones((batch_size, max_time, num_bins), dtype=np.float32)

    for b in range(batch_size):
        t_len = int(feature_lengths[b].item())
        if t_len <= 1:
            continue

        # Time masks - mask out horizontal bands
        for _ in range(num_time_masks):
            width = random.randint(0, min(time_mask_max, t_len - 1))
            if width <= 0:
                continue
            start = random.randint(0, max(0, t_len - width - 1))
            np_mask[b, start : start + width, :] = 0.0

        # Frequency masks - mask out vertical bands (only in valid time range)
        for _ in range(num_freq_masks):
            width = random.randint(0, min(freq_mask_max, num_bins - 1))
            if width <= 0:
                continue
            start = random.randint(0, max(0, num_bins - width - 1))
            np_mask[b, :t_len, start : start + width] = 0.0

    # Convert to MLX and apply
    mask = mx.array(np_mask)
    fill_value = mx.mean(features, axis=(1, 2), keepdims=True)
    return features * mask + fill_value * (1.0 - mask)


def _infer_num_classes_from_loader(train_loader, head_type: str) -> int | None:
    dataset = getattr(train_loader, "dataset", None)
    samples = getattr(dataset, "samples", None)

    if head_type == "emotion" and isinstance(samples, list):
        labels = [s.emotion_label for s in samples if getattr(s, "emotion_label", None) is not None]
        return (max(labels) + 1) if labels else None
    if head_type == "paralinguistics" and isinstance(samples, list):
        labels = [
            s.paralinguistic_label
            for s in samples
            if getattr(s, "paralinguistic_label", None) is not None
        ]
        return (max(labels) + 1) if labels else None

    hf = getattr(dataset, "dataset", None)
    if head_type == "emotion" and hf is not None and hasattr(dataset, "EMOTION_TO_INT"):
        try:
            emotions = hf["emotion"]
            labels = [dataset.EMOTION_TO_INT.get(str(e).lower(), None) for e in emotions]
            labels = [x for x in labels if x is not None]
            return (max(labels) + 1) if labels else None
        except Exception:
            return None

    return None


def _compute_class_weights(train_loader, head_type: str, num_classes: int) -> mx.array | None:
    dataset = getattr(train_loader, "dataset", None)
    samples = getattr(dataset, "samples", None)

    counts = [0] * num_classes
    if head_type == "emotion" and isinstance(samples, list):
        for s in samples:
            y = getattr(s, "emotion_label", None)
            if y is not None and 0 <= y < num_classes:
                counts[int(y)] += 1
    elif head_type == "paralinguistics" and isinstance(samples, list):
        for s in samples:
            y = getattr(s, "paralinguistic_label", None)
            if y is not None and 0 <= y < num_classes:
                counts[int(y)] += 1
    else:
        return None

    total = sum(counts)
    if total == 0:
        return None

    weights = []
    for c in counts:
        if c <= 0:
            weights.append(1.0)
        else:
            weights.append(total / (num_classes * c))

    return mx.array(weights, dtype=mx.float32)


def _compute_loss_and_grads(
    encoder: nn.Module,
    head: nn.Module,
    batch: dict[str, mx.array],
    label_key: str,
    combined_model: Optional["EncoderHeadModel"],
    label_smoothing: float,
    class_weights: mx.array | None,
    encoder_grad_scale: float,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
) -> tuple[mx.array, mx.array, dict]:
    labels = batch[label_key]

    if combined_model is None:
        encoder_out, encoder_lengths = encoder(
            batch["features"],
            batch["feature_lengths"],
        )
        encoder_out = mx.stop_gradient(encoder_out)

        logits_holder = [None]

        def loss_fn(head_model):
            logits = head_model(encoder_out, encoder_lengths)
            logits_holder[0] = logits
            if use_focal_loss:
                return focal_loss(
                    logits,
                    labels,
                    gamma=focal_gamma,
                    alpha=class_weights,
                    label_smoothing=label_smoothing,
                    reduction="mean",
                )
            # Convert class weights to per-sample weights if provided
            sample_weights = None
            if class_weights is not None:
                sample_weights = class_weights[labels]
            return nn.losses.cross_entropy(
                logits,
                labels,
                weights=sample_weights,
                label_smoothing=label_smoothing,
                reduction="mean",
            )

        loss_and_grad_fn = nn.value_and_grad(head, loss_fn)
        loss, grads = loss_and_grad_fn(head)
        logits = logits_holder[0]
        return loss, logits, grads

    logits_holder = [None]

    def loss_fn(model):
        logits, _, _ = model(batch["features"], batch["feature_lengths"])
        logits_holder[0] = logits
        if use_focal_loss:
            return focal_loss(
                logits,
                labels,
                gamma=focal_gamma,
                alpha=class_weights,
                label_smoothing=label_smoothing,
                reduction="mean",
            )
        # Convert class weights to per-sample weights if provided
        sample_weights = None
        if class_weights is not None:
            sample_weights = class_weights[labels]
        return nn.losses.cross_entropy(
            logits,
            labels,
            weights=sample_weights,
            label_smoothing=label_smoothing,
            reduction="mean",
        )

    loss_and_grad_fn = nn.value_and_grad(combined_model, loss_fn)
    loss, grads = loss_and_grad_fn(combined_model)

    if encoder_grad_scale != 1.0 and isinstance(grads, dict) and "encoder" in grads:
        grads = dict(grads)
        grads["encoder"] = _tree_scale(grads["encoder"], encoder_grad_scale)

    logits = logits_holder[0]
    return loss, logits, grads


def train_step(
    encoder: nn.Module,
    head: nn.Module,
    optimizer: optim.Optimizer,
    batch: dict[str, mx.array],
    label_key: str,
    grad_clip: float = 1.0,
    combined_model: Optional["EncoderHeadModel"] = None,
    label_smoothing: float = 0.0,
    class_weights: mx.array | None = None,
    encoder_grad_scale: float = 1.0,
) -> tuple[float, float, mx.array]:
    """
    Execute a single training step.

    Args:
        encoder: Zipformer encoder module.
        head: Classification head module.
        optimizer: Optimizer for updating parameters.
        batch: Input batch with features and labels.
        label_key: Key for label tensor in batch.
        grad_clip: Maximum gradient norm.
        combined_model: Pre-created EncoderHeadModel for fine-tuning (reused).
                        If None, uses frozen encoder mode.

    Returns:
        Tuple of (loss, accuracy, logits).
    """
    labels = batch[label_key]
    loss, logits, grads = _compute_loss_and_grads(
        encoder=encoder,
        head=head,
        batch=batch,
        label_key=label_key,
        combined_model=combined_model,
        label_smoothing=label_smoothing,
        class_weights=class_weights,
        encoder_grad_scale=encoder_grad_scale,
    )

    grads = clip_grad_norm(grads, max_norm=grad_clip)

    _tree_eval(grads)
    mx.eval(loss, logits)
    loss_val = loss.item()

    if (not math.isfinite(loss_val)) or _is_nonfinite(logits) or _tree_has_nonfinite(grads):
        return float("nan"), 0.0, mx.array([])

    # Apply update
    if combined_model is None:
        optimizer.update(head, grads)
        mx.eval(head.parameters())
    else:
        optimizer.update(combined_model, grads)
        mx.eval(combined_model.parameters())

    # Compute accuracy using saved logits (no second forward pass)
    predictions = mx.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean().item()

    return loss_val, accuracy, logits


def evaluate(
    encoder: nn.Module,
    head: nn.Module,
    dataloader: Iterator,
    label_key: str,
    max_batches: int = 100,
    class_weights: mx.array | None = None,
    ema_params: dict | None = None,
) -> tuple[float, float]:
    """
    Evaluate the model on validation data.

    Returns:
        Tuple of (average loss, accuracy).
    """
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    num_batches = 0
    skipped_batches = 0

    # Set head to eval mode to disable dropout during validation
    head.eval()

    old_params = None
    if ema_params is not None:
        old_params = head.parameters()
        head.update(ema_params)

    for batch in dataloader:
        if num_batches >= max_batches:
            break

        # Forward through frozen encoder
        encoder_out, encoder_lengths = encoder(
            batch["features"],
            batch["feature_lengths"],
        )

        # Forward through head
        logits = head(encoder_out, encoder_lengths)

        # Get labels
        labels = batch[label_key]

        # Convert class weights to per-sample weights if provided
        sample_weights = None
        if class_weights is not None:
            sample_weights = class_weights[labels]

        # Compute loss
        loss = nn.losses.cross_entropy(
            logits,
            labels,
            weights=sample_weights,
            label_smoothing=0.0,
            reduction="mean",
        )

        # Compute accuracy
        predictions = mx.argmax(logits, axis=-1)
        correct = (predictions == labels).sum()

        # M5: Evaluate after EACH batch to release computation graph memory
        mx.eval(loss, correct)

        # Skip batches with non-finite loss/logits (can occur with edge case inputs)
        loss_val = loss.item()
        if (not math.isfinite(loss_val)) or _is_nonfinite(logits):
            skipped_batches += 1
            continue

        total_loss += loss_val
        total_correct += correct.item()
        total_samples += labels.shape[0]

        num_batches += 1

    if num_batches == 0:
        logger.error(
            "Evaluation produced no valid batches "
            f"(skipped={skipped_batches}, max_batches={max_batches}).",
        )
        avg_loss = float("nan")
        accuracy = float("nan")
    else:
        avg_loss = total_loss / num_batches
        accuracy = total_correct / max(total_samples, 1)

    if old_params is not None:
        head.update(old_params)

    # Restore training mode after evaluation
    head.train()

    return avg_loss, accuracy


def flatten_params(params, prefix=""):
    """Flatten nested parameter dict for saving, handling dicts and lists."""
    flat = {}
    if isinstance(params, dict):
        for k, v in params.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, (dict, list)):
                flat.update(flatten_params(v, key))
            else:
                flat[key] = v
    elif isinstance(params, list):
        for i, v in enumerate(params):
            key = f"{prefix}.{i}" if prefix else str(i)
            if isinstance(v, (dict, list)):
                flat.update(flatten_params(v, key))
            else:
                flat[key] = v
    return flat


def save_checkpoint(
    head: nn.Module,
    optimizer: optim.Optimizer,
    step: int,
    epoch: int,
    output_dir: Path,
    is_best: bool = False,
    encoder: nn.Module | None = None,
    unfreeze_layers: int = 0,
):
    """Save model checkpoint.

    When unfreeze_layers > 0, also saves the unfrozen encoder layer weights
    for proper resumption of fine-tuning.
    """
    import numpy as np

    checkpoint = {
        "step": step,
        "epoch": epoch,
        "unfreeze_layers": unfreeze_layers,
    }

    suffix = "best" if is_best else f"step_{step}"

    # Flatten and convert to numpy for saving
    flat_params = flatten_params(head.parameters())
    np_params = {k: np.array(v) for k, v in flat_params.items()}

    weights_path = output_dir / f"head_{suffix}.npz"
    np.savez(str(weights_path), **np_params)

    # Save encoder weights if fine-tuning is enabled
    if encoder is not None and unfreeze_layers > 0:
        num_stages = len(encoder.encoders)
        start_stage = max(0, num_stages - unfreeze_layers)
        encoder_params = {}
        for stage_idx in range(start_stage, num_stages):
            stage_params = flatten_params(
                encoder.encoders[stage_idx].parameters(),
                prefix=f"stage_{stage_idx}",
            )
            encoder_params.update({k: np.array(v) for k, v in stage_params.items()})

        encoder_path = output_dir / f"encoder_{suffix}.npz"
        np.savez(str(encoder_path), **encoder_params)
        logger.info(f"Saved encoder weights ({len(encoder_params)} arrays) to {encoder_path}")

    meta_path = output_dir / f"meta_{suffix}.json"
    with open(meta_path, "w") as f:
        json.dump(checkpoint, f, indent=2)


def find_latest_checkpoint(output_dir: Path) -> tuple[Path, int, int] | None:
    """Find the latest checkpoint in output_dir.

    Returns:
        Tuple of (weights_path, step, epoch) or None if no checkpoint found.
    """
    # Look for step checkpoints (not 'best')
    pattern = output_dir / "meta_step_*.json"
    import glob

    meta_files = glob.glob(str(pattern))
    if not meta_files:
        return None

    # Parse step numbers and find latest
    latest_step = -1
    latest_meta = None
    for meta_file in meta_files:
        try:
            # Extract step from filename: meta_step_8000.json -> 8000
            fname = Path(meta_file).stem  # meta_step_8000
            step_str = fname.replace("meta_step_", "")
            step = int(step_str)
            if step > latest_step:
                latest_step = step
                latest_meta = Path(meta_file)
        except (ValueError, IndexError):
            continue

    if latest_meta is None:
        return None

    # Read metadata
    with open(latest_meta) as f:
        meta = json.load(f)

    step = meta.get("step", latest_step)
    epoch = meta.get("epoch", 0)

    # Find corresponding weights file
    weights_path = output_dir / f"head_step_{step}.npz"
    if not weights_path.exists():
        logger.warning(f"Checkpoint metadata found but weights missing: {weights_path}")
        return None

    return weights_path, step, epoch


def load_head_checkpoint(
    head: nn.Module,
    weights_path: Path,
    encoder: nn.Module | None = None,
    unfreeze_layers: int = 0,
) -> None:
    """Load head weights from checkpoint.

    Also loads encoder weights if they exist and fine-tuning is enabled.

    Args:
        head: Head module to load weights into.
        weights_path: Path to .npz file with saved weights.
        encoder: Optional encoder module for loading fine-tuned weights.
        unfreeze_layers: Number of encoder layers that were unfrozen.
    """
    import numpy as np

    # Load numpy arrays
    data = np.load(str(weights_path))

    # Get flat dict from numpy file
    flat_dict = {k: data[k] for k in data.files}

    # For simple flat structure, directly convert and update
    mlx_params = {k: mx.array(v) for k, v in flat_dict.items()}
    head.load_weights(list(mlx_params.items()))

    logger.info(f"Loaded head checkpoint from {weights_path}")

    # Try to load encoder weights if fine-tuning
    if encoder is not None and unfreeze_layers > 0:
        # Check for encoder weights file with same suffix
        suffix = weights_path.stem.replace("head_", "")  # step_8000 or best
        encoder_path = weights_path.parent / f"encoder_{suffix}.npz"

        if encoder_path.exists():
            encoder_data = np.load(str(encoder_path))
            num_stages = len(encoder.encoders)
            start_stage = max(0, num_stages - unfreeze_layers)

            loaded_stages = 0
            for stage_idx in range(start_stage, num_stages):
                prefix = f"stage_{stage_idx}."
                stage_params = {}
                for k in encoder_data.files:
                    if k.startswith(prefix):
                        # Remove stage prefix to get parameter name
                        param_name = k[len(prefix):]
                        stage_params[param_name] = mx.array(encoder_data[k])

                if stage_params:
                    encoder.encoders[stage_idx].load_weights(list(stage_params.items()))
                    loaded_stages += 1

            logger.info(f"Loaded encoder weights for {loaded_stages} stages from {encoder_path}")
        else:
            logger.warning(
                f"Encoder weights not found at {encoder_path}. "
                f"Encoder will use pretrained weights (accuracy may be lower).",
            )


def acquire_training_lock() -> int:
    """
    Acquire exclusive training lock to prevent parallel training jobs.

    Returns:
        File descriptor of the lock file (must be kept open).

    Raises:
        SystemExit: If another training job is already running.
    """
    LOCK_FILE = "/tmp/rich_audio_training.lock"
    lock_fd = os.open(LOCK_FILE, os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Write PID to lock file for debugging
        os.write(lock_fd, f"{os.getpid()}\n".encode())
        return lock_fd
    except BlockingIOError:
        os.close(lock_fd)
        logger.error("ERROR: Another training job is already running")
        logger.error("If you believe this is incorrect, remove /tmp/rich_audio_training.lock")
        sys.exit(1)


def train(args: TrainingArgs) -> None:
    """Main training function."""

    # Acquire exclusive lock to prevent parallel training
    lock_fd = acquire_training_lock()
    logger.info("Acquired training lock")

    # Set random seeds for reproducibility (Q6)
    mx.random.seed(args.seed)
    random.seed(args.seed)
    try:
        import numpy as np
        np.random.seed(args.seed)
    except ImportError:
        pass

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up file logging to output_dir/training.log
    # This ensures all training runs have a persistent log file
    log_file = output_dir / "training.log"
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)
    logger.info(f"Logging to {log_file}")

    # Load pretrained encoder
    logger.info(f"Loading encoder from {args.checkpoint}")
    model, config = load_checkpoint(args.checkpoint)
    encoder = model.encoder
    encoder_dim = encoder.output_dim
    logger.info(f"Encoder loaded: output_dim={encoder_dim}")

    # Create dataloaders
    # Compute actual data path for logging
    actual_data_dir = args.data_dir
    if args.head == "emotion" and args.dataset == "combined":
        if args.data_dir == "data/emotion/crema-d":
            actual_data_dir = "data/emotion/combined_emotion_hf"
    elif args.head == "emotion" and args.dataset == "meld":
        if args.data_dir == "data/emotion/crema-d":
            actual_data_dir = "data/emotion_punctuation/MELD.Raw"
    logger.info(f"Loading data from {actual_data_dir} (dataset={args.dataset})")
    train_loader = create_dataloader(args, "train")
    val_loader = create_dataloader(args, "val")

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")

    label_key = get_label_key(args.head)
    inferred_num_classes = _infer_num_classes_from_loader(train_loader, args.head)
    head_num_classes = inferred_num_classes
    if head_num_classes is None:
        head_num_classes = len(EMOTION_LABELS) if args.head == "emotion" else len(PARALINGUISTIC_LABELS)
    logger.info(f"Head num_classes: {head_num_classes}")

    # Create head (Q10: avoid hard-coded class counts in head config)
    head = create_head(args.head, encoder_dim, num_classes=head_num_classes)

    # Mixed precision control (Q7)
    dtype_map = {
        "float32": mx.float32,
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
    }
    param_dtype = dtype_map[args.dtype]
    encoder.set_dtype(param_dtype)
    head.set_dtype(param_dtype)

    def count_params(params):
        """Count parameters recursively, handling dicts and lists."""
        total = 0
        if isinstance(params, dict):
            for v in params.values():
                total += count_params(v)
        elif isinstance(params, list):
            for v in params:
                total += count_params(v)
        elif hasattr(params, "size"):
            total += params.size
        return total

    num_head_params = count_params(head.parameters())
    logger.info(f"Created {args.head} head with {num_head_params:,} parameters")

    # Calculate trainable encoder parameters if unfreezing
    num_encoder_params = 0
    if args.unfreeze_layers > 0:
        num_stages = len(encoder.encoders)
        start_stage = max(0, num_stages - args.unfreeze_layers)
        for stage_idx in range(start_stage, num_stages):
            stage_params = count_params(encoder.encoders[stage_idx].parameters())
            num_encoder_params += stage_params
            logger.info(f"Stage {stage_idx}: {stage_params:,} params (UNFROZEN)")
        logger.info(f"Total unfrozen encoder params: {num_encoder_params:,}")

    total_trainable = num_head_params + num_encoder_params
    logger.info(f"Total trainable parameters: {total_trainable:,}")

    # Q5: Optional class weights for imbalanced datasets
    class_weights = None
    if args.use_class_weights and args.head in ("emotion", "paralinguistics"):
        class_weights = _compute_class_weights(train_loader, args.head, head_num_classes)
        if class_weights is not None:
            logger.info(f"Using class weights for {args.head}: {class_weights.tolist()}")

    # Calculate total optimizer update steps (M11: gradient accumulation)
    micro_steps_per_epoch = len(train_loader)
    updates_per_epoch = micro_steps_per_epoch
    if args.accumulation_steps > 1:
        updates_per_epoch = (micro_steps_per_epoch + args.accumulation_steps - 1) // args.accumulation_steps
    max_steps = args.epochs * updates_per_epoch

    # Q1: Adjust warmup for head-only training
    # Head-only training needs much shorter warmup (100 steps vs 500 for full model)
    effective_warmup = args.warmup_steps
    if args.unfreeze_layers == 0:
        # Head-only: shorter warmup since we're just training a small classifier
        effective_warmup = min(100, args.warmup_steps)
        logger.info(f"Head-only training: using shorter warmup ({effective_warmup} steps)")

    # Create optimizer
    optimizer = optim.AdamW(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Training state
    step = 0
    start_epoch = 0
    best_val_acc = 0.0

    # Resume from checkpoint if requested
    if args.resume or args.resume_from:
        if args.resume_from:
            # Load from specific checkpoint
            resume_path = Path(args.resume_from)
            if not resume_path.exists():
                # Try interpreting as relative to output_dir
                resume_path = output_dir / args.resume_from
            if not resume_path.exists():
                logger.error(f"Resume checkpoint not found: {args.resume_from}")
                sys.exit(1)
            # Extract step and epoch from filename if possible
            try:
                step_str = resume_path.stem.replace("head_step_", "")
                step = int(step_str)
                meta_path = output_dir / f"meta_step_{step}.json"
                if meta_path.exists():
                    with open(meta_path) as f:
                        meta = json.load(f)
                        start_epoch = meta.get("epoch", 0)
            except (ValueError, AttributeError):
                logger.warning("Could not parse step from checkpoint filename, starting from step 0")
        else:
            # Find latest checkpoint
            ckpt_info = find_latest_checkpoint(output_dir)
            if ckpt_info is None:
                logger.warning("No checkpoint found to resume from, starting fresh")
            else:
                resume_path, step, start_epoch = ckpt_info

        if 'resume_path' in locals() and resume_path.exists():
            load_head_checkpoint(
                head, resume_path,
                encoder=encoder if args.unfreeze_layers > 0 else None,
                unfreeze_layers=args.unfreeze_layers,
            )
            logger.info(f"Resuming training from step {step}, epoch {start_epoch}")
            # Check if there's a best checkpoint to get best_val_acc
            best_meta_path = output_dir / "meta_best.json"
            if best_meta_path.exists():
                # Try to get best accuracy from eval logs or estimate from checkpoint
                logger.info("Found best checkpoint - will compare against it during training")
            # The step counter continues from where we left off

    # Q2: Learning rate safety for unfrozen encoder stages
    # NOTE: Gradient scaling (encoder_grad_scale = encoder_lr / head_lr) doesn't work
    # correctly with AdamW because AdamW normalizes by gradient variance. Instead,
    # we cap the learning rate to 3e-5 when encoder unfreezing is enabled to prevent
    # numerical instability that causes NaN losses.
    # Evidence from training runs:
    # - emotion_v9: LR < 8e-5: 0 skipped batches; LR > 9e-5: Many skipped batches
    # - paralinguistics_v5: LR = 4e-5: 382/485 (79%) skipped by epoch 10
    # The instability compounds over epochs, requiring lower base LR + adaptive reduction.
    encoder_grad_scale = 1.0  # Disabled - doesn't work correctly with AdamW
    # Q2: Learning rate safety - track for adaptive reduction
    adaptive_lr_factor = 1.0  # Will be reduced if NaN rate is high
    nan_rate_threshold = 0.1  # Reduce LR if >10% batches have NaN
    consecutive_high_nan_epochs = 0  # Track consecutive bad epochs

    if args.unfreeze_layers > 0:
        # Learning rate history for encoder fine-tuning:
        # - paralinguistics_v6: 1.5e-5 was safe, paralinguistics_v9: 4e-5 worked (0.49% NaN)
        # - emotion_v11: 4e-5 achieved 46.70%, emotion_v14: 4e-5 had 20.2% NaN rate (unstable)
        # Conclusion: 4e-5 is too aggressive for some tasks (emotion). Using 2.5e-5 as
        # a compromise that works across all head types without excessive NaN skips.
        max_safe_lr = 2.5e-5  # Maximum safe LR for encoder fine-tuning
        if args.learning_rate > max_safe_lr:
            logger.warning(
                f"Learning rate {args.learning_rate} is too high for encoder fine-tuning. "
                f"This can cause NaN values during training. "
                f"Automatically reducing to {max_safe_lr}.",
            )
            args.learning_rate = max_safe_lr
            # Recreate optimizer with safe learning rate
            optimizer = optim.AdamW(
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
            )
        logger.info(
            f"Fine-tuning with encoder unfreezing: lr={args.learning_rate} "
            f"(encoder_lr parameter is ignored - both encoder and head use same lr)",
        )
        logger.info(
            f"Adaptive LR enabled: will reduce LR by 0.5x if NaN rate > {nan_rate_threshold*100:.0f}%",
        )

    # Q8: Optional EMA over head parameters
    ema_params = None
    if args.ema_decay and args.ema_decay > 0.0:
        ema_params = head.parameters()
        logger.info(f"EMA enabled: decay={args.ema_decay}")

    logger.info("=" * 60)
    logger.info("Training Configuration")
    logger.info("=" * 60)
    logger.info(f"Head type: {args.head}")
    logger.info(f"Encoder dim: {encoder_dim}")
    logger.info(f"Batch size: {args.batch_size}")
    if args.accumulation_steps > 1:
        logger.info(f"Gradient accumulation: {args.accumulation_steps} steps")
        logger.info(f"Effective batch size: {args.batch_size * args.accumulation_steps}")
    logger.info(f"Head learning rate: {args.learning_rate}")
    if args.unfreeze_layers > 0:
        logger.info(f"Encoder learning rate: {args.encoder_lr}")
        logger.info(f"Unfrozen encoder stages: {args.unfreeze_layers}")
    else:
        logger.info("Encoder: FROZEN")
    logger.info(f"Gradient clipping: {args.grad_clip}")
    if args.clear_cache_every > 0:
        logger.info(f"Cache clearing: every {args.clear_cache_every} steps")
    logger.info(f"Label smoothing: {args.label_smoothing}")
    if args.focal_loss:
        logger.info(f"Loss function: Focal Loss (gamma={args.focal_gamma})")
    else:
        logger.info("Loss function: Cross-Entropy")
    logger.info(f"SpecAugment: {args.spec_augment}")
    logger.info(f"Param dtype: {args.dtype}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Max steps: {max_steps}")
    logger.info(f"Label key: {label_key}")
    logger.info("=" * 60)

    # Save final args after any automatic adjustments (e.g., LR capping).
    with open(output_dir / "training_args.json", "w") as f:
        json.dump(asdict(args), f, indent=2)

    # M2: Create combined model ONCE before training loop (not every step)
    combined_model: EncoderHeadModel | None = None
    if args.unfreeze_layers > 0:
        combined_model = EncoderHeadModel(encoder, head, args.unfreeze_layers)
        combined_model.set_dtype(param_dtype)
        logger.info("Created EncoderHeadModel for fine-tuning (reused across all steps)")

    start_time = time.time()

    # Cumulative NaN tracking across all epochs
    total_batches_processed = 0
    total_batches_skipped_loss = 0
    total_batches_skipped_logits = 0
    total_batches_skipped_grads = 0
    epoch_skip_rates: list[float] = []  # Track skip rate per epoch for summary

    for epoch in range(start_epoch, args.epochs):
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")

        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_updates = 0

        accum_grads = None
        accum_count = 0
        accum_loss = 0.0
        accum_acc = 0.0

        epoch_micro_batches = 0
        epoch_skipped_loss = 0
        epoch_skipped_logits = 0
        epoch_skipped_grads = 0

        for batch in train_loader:
            epoch_micro_batches += 1
            # Update learning rate (Q1: use effective_warmup for head-only training)
            # Apply adaptive_lr_factor for NaN stability when encoder unfreezing is enabled
            base_lr = warmup_cosine_lr(step, effective_warmup, max_steps, args.learning_rate)
            lr = base_lr * adaptive_lr_factor
            optimizer.learning_rate = lr

            train_batch = batch
            if args.spec_augment:
                train_batch = dict(batch)
                train_batch["features"] = _apply_spec_augment(
                    train_batch["features"],
                    train_batch["feature_lengths"],
                    num_time_masks=args.num_time_masks,
                    time_mask_max=args.time_mask_max,
                    num_freq_masks=args.num_freq_masks,
                    freq_mask_max=args.freq_mask_max,
                )

            loss, logits, grads = _compute_loss_and_grads(
                encoder=encoder,
                head=head,
                batch=train_batch,
                label_key=label_key,
                combined_model=combined_model,
                label_smoothing=args.label_smoothing,
                class_weights=class_weights,
                encoder_grad_scale=encoder_grad_scale,
                use_focal_loss=args.focal_loss,
                focal_gamma=args.focal_gamma,
            )

            # Materialize to avoid graph accumulation
            _tree_eval(grads)
            mx.eval(loss, logits)

            loss_val = loss.item()
            if not math.isfinite(loss_val):
                epoch_skipped_loss += 1
                if epoch_skipped_loss <= 10 or epoch_skipped_loss % 100 == 0:
                    logger.warning(
                        f"Skipping batch due to non-finite loss at step={step} "
                        f"(loss={loss_val}, epoch={epoch + 1}).",
                    )
                continue

            if _is_nonfinite(logits):
                epoch_skipped_logits += 1
                if epoch_skipped_logits <= 10 or epoch_skipped_logits % 100 == 0:
                    logger.warning(
                        f"Skipping batch due to non-finite logits at step={step} "
                        f"(loss={loss_val:.4f}, epoch={epoch + 1}).",
                    )
                continue

            if _tree_has_nonfinite(grads):
                epoch_skipped_grads += 1
                if epoch_skipped_grads <= 10 or epoch_skipped_grads % 100 == 0:
                    logger.warning(
                        f"Skipping batch due to non-finite gradients at step={step} "
                        f"(loss={loss_val:.4f}, epoch={epoch + 1}).",
                    )
                continue

            labels = train_batch[label_key]
            predictions = mx.argmax(logits, axis=-1)
            acc = (predictions == labels).mean().item()

            accum_grads = grads if accum_grads is None else _tree_add(accum_grads, grads)
            accum_count += 1
            accum_loss += loss_val
            accum_acc += acc

            if accum_count < args.accumulation_steps:
                continue

            module_to_update = combined_model if combined_model is not None else head
            mean_grads = _tree_scale(accum_grads, 1.0 / accum_count)
            mean_grads = clip_grad_norm(mean_grads, max_norm=args.grad_clip)

            optimizer.update(module_to_update, mean_grads)
            mx.eval(module_to_update.parameters())

            if ema_params is not None:
                ema_params = _ema_update(ema_params, head.parameters(), args.ema_decay)

            update_loss = accum_loss / max(accum_count, 1)
            update_acc = accum_acc / max(accum_count, 1)

            epoch_loss += update_loss
            epoch_acc += update_acc
            epoch_updates += 1
            step += 1

            accum_grads = None
            accum_count = 0
            accum_loss = 0.0
            accum_acc = 0.0

            # M4: Clear MLX cache periodically to prevent memory fragmentation
            if args.clear_cache_every > 0 and step % args.clear_cache_every == 0:
                mx.clear_cache()

            # Logging
            if step % args.log_every == 0:
                logger.info(
                    f"Step {step}/{max_steps} | "
                    f"Loss: {update_loss:.4f} | "
                    f"Acc: {update_acc:.4f} | "
                    f"LR: {lr:.2e}",
                )

            # Evaluation
            if step % args.eval_every == 0:
                val_loss, val_acc = evaluate(
                    encoder,
                    head,
                    val_loader,
                    label_key,
                    ema_params=ema_params,
                )
                logger.info(
                    f"[EVAL] Step {step} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_acc:.4f}",
                )

                # Save best model (use EMA weights if enabled)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    if ema_params is not None:
                        old_params = head.parameters()
                        head.update(ema_params)
                        save_checkpoint(head, optimizer, step, epoch, output_dir, is_best=True, encoder=encoder, unfreeze_layers=args.unfreeze_layers)
                        head.update(old_params)
                    else:
                        save_checkpoint(head, optimizer, step, epoch, output_dir, is_best=True, encoder=encoder, unfreeze_layers=args.unfreeze_layers)
                    logger.info(f"New best validation accuracy: {val_acc:.4f}")

            # Checkpointing
            if step % args.save_every == 0:
                if ema_params is not None:
                    old_params = head.parameters()
                    head.update(ema_params)
                    save_checkpoint(head, optimizer, step, epoch, output_dir, encoder=encoder, unfreeze_layers=args.unfreeze_layers)
                    head.update(old_params)
                else:
                    save_checkpoint(head, optimizer, step, epoch, output_dir, encoder=encoder, unfreeze_layers=args.unfreeze_layers)

        # Apply leftover accumulated gradients at end of epoch
        if accum_grads is not None and accum_count > 0:
            module_to_update = combined_model if combined_model is not None else head
            mean_grads = _tree_scale(accum_grads, 1.0 / accum_count)
            mean_grads = clip_grad_norm(mean_grads, max_norm=args.grad_clip)
            optimizer.update(module_to_update, mean_grads)
            mx.eval(module_to_update.parameters())
            if ema_params is not None:
                ema_params = _ema_update(ema_params, head.parameters(), args.ema_decay)

            epoch_loss += accum_loss / max(accum_count, 1)
            epoch_acc += accum_acc / max(accum_count, 1)
            epoch_updates += 1
            step += 1

        # Epoch summary
        if epoch_updates == 0:
            avg_loss = float("nan")
            avg_acc = float("nan")
        else:
            avg_loss = epoch_loss / epoch_updates
            avg_acc = epoch_acc / epoch_updates

        total_skipped = epoch_skipped_loss + epoch_skipped_logits + epoch_skipped_grads
        epoch_skip_rate = total_skipped / max(epoch_micro_batches, 1)
        logger.info(
            f"Epoch {epoch + 1} complete | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"Avg Acc: {avg_acc:.4f} | "
            f"Updates: {epoch_updates} | "
            f"Micro-batches: {epoch_micro_batches} | "
            f"Skipped: {total_skipped} ({epoch_skip_rate*100:.1f}%) "
            f"(loss={epoch_skipped_loss}, logits={epoch_skipped_logits}, grads={epoch_skipped_grads})",
        )

        # Update cumulative NaN tracking
        total_batches_processed += epoch_micro_batches
        total_batches_skipped_loss += epoch_skipped_loss
        total_batches_skipped_logits += epoch_skipped_logits
        total_batches_skipped_grads += epoch_skipped_grads
        epoch_skip_rates.append(epoch_skip_rate)

        # Adaptive LR reduction when encoder unfreezing is enabled
        if args.unfreeze_layers > 0 and epoch_micro_batches > 0:
            nan_rate = total_skipped / epoch_micro_batches

            # Early stopping: halt training if skip rate exceeds 50%
            # At this point, the model is learning from less than half the data
            early_stop_threshold = 0.5
            if nan_rate > early_stop_threshold:
                logger.error(
                    f"EARLY STOPPING: NaN rate ({nan_rate*100:.1f}%) exceeds {early_stop_threshold*100:.0f}% threshold. "
                    f"Model is skipping more than half of all batches. "
                    f"Training is unlikely to produce a useful model. "
                    f"Consider: (1) lowering learning rate, (2) reducing unfreeze_layers, "
                    f"(3) checking data for problematic samples.",
                )
                # Save checkpoint before stopping
                save_checkpoint(head, optimizer, step, epoch, output_dir, encoder=encoder, unfreeze_layers=args.unfreeze_layers)
                logger.info(f"Saved checkpoint at step {step} before early stopping")
                break  # Exit the epoch loop

            if nan_rate > nan_rate_threshold:
                consecutive_high_nan_epochs += 1
                if consecutive_high_nan_epochs >= 1:
                    # Reduce LR by 50% to stabilize training (trigger on first bad epoch)
                    adaptive_lr_factor *= 0.5
                    logger.warning(
                        f"High NaN rate ({nan_rate*100:.1f}%) for {consecutive_high_nan_epochs} epoch(s). "
                        f"Reducing effective LR by factor of {1/adaptive_lr_factor:.1f}x "
                        f"(base_lr={args.learning_rate}, effective_lr={args.learning_rate * adaptive_lr_factor:.2e})",
                    )
                    # Reset counter after reduction
                    consecutive_high_nan_epochs = 0
            else:
                consecutive_high_nan_epochs = 0  # Reset counter on good epoch

    # Final evaluation
    val_loss, val_acc = evaluate(
        encoder,
        head,
        val_loader,
        label_key,
        ema_params=ema_params,
    )
    logger.info(f"Final validation: Loss={val_loss:.4f}, Acc={val_acc:.4f}")
    if math.isfinite(val_acc) and val_acc > best_val_acc:
        best_val_acc = val_acc
        if ema_params is not None:
            old_params = head.parameters()
            head.update(ema_params)
            save_checkpoint(head, optimizer, step, args.epochs - 1, output_dir, is_best=True, encoder=encoder, unfreeze_layers=args.unfreeze_layers)
            head.update(old_params)
        else:
            save_checkpoint(head, optimizer, step, args.epochs - 1, output_dir, is_best=True, encoder=encoder, unfreeze_layers=args.unfreeze_layers)

    # Save final checkpoint
    if ema_params is not None:
        old_params = head.parameters()
        head.update(ema_params)
        save_checkpoint(head, optimizer, step, args.epochs - 1, output_dir, encoder=encoder, unfreeze_layers=args.unfreeze_layers)
        head.update(old_params)
    else:
        save_checkpoint(head, optimizer, step, args.epochs - 1, output_dir, encoder=encoder, unfreeze_layers=args.unfreeze_layers)

    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed:.2f}s ({elapsed/3600:.2f}h)")
    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")

    # NaN/Skip Statistics Summary
    total_skipped_all = total_batches_skipped_loss + total_batches_skipped_logits + total_batches_skipped_grads
    overall_skip_rate = total_skipped_all / max(total_batches_processed, 1)
    logger.info("=" * 60)
    logger.info("NaN/Skip Statistics Summary")
    logger.info("=" * 60)
    logger.info(f"Total batches processed: {total_batches_processed}")
    logger.info(f"Total batches skipped: {total_skipped_all} ({overall_skip_rate*100:.1f}%)")
    logger.info(f"  - Skipped due to NaN loss: {total_batches_skipped_loss}")
    logger.info(f"  - Skipped due to NaN logits: {total_batches_skipped_logits}")
    logger.info(f"  - Skipped due to NaN gradients: {total_batches_skipped_grads}")
    if epoch_skip_rates:
        logger.info(f"Skip rate progression by epoch: {[f'{r*100:.1f}%' for r in epoch_skip_rates]}")
        max_skip_epoch = epoch_skip_rates.index(max(epoch_skip_rates)) + 1
        logger.info(f"Worst epoch: {max_skip_epoch} ({max(epoch_skip_rates)*100:.1f}% skipped)")
    if overall_skip_rate > 0.2:
        logger.warning(
            f"High overall skip rate ({overall_skip_rate*100:.1f}%). "
            f"Model may have trained on incomplete data. Consider lowering learning rate.",
        )
    logger.info("=" * 60)


def main():
    """Entry point."""
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
