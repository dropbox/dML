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
Training configuration for Zipformer ASR model.

Based on k2-fsa/icefall training parameters with stability mitigations
for streaming Zipformer-large.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SpeakerTrainingConfig:
    """Configuration for DELULU-style speaker embedding training."""

    # Whether to enable speaker embedding training
    enabled: bool = False

    # Number of speakers (set from dataset)
    num_speakers: int = 997  # CN-Celeb default

    # Embedding dimension
    embedding_dim: int = 256

    # AAM-Softmax margin (angular margin in radians)
    # Higher margin = harder training = better separation
    aam_margin: float = 0.2

    # AAM-Softmax scale (logit scaling factor)
    aam_scale: float = 30.0

    # Loss weight when training jointly with ASR
    loss_weight: float = 0.5

    # Self-supervised learning objectives
    use_ssl: bool = True
    mask_ratio: float = 0.15
    denoise_prob: float = 0.5

    # Data augmentation
    aug_speed_perturb: bool = True
    aug_noise: bool = True
    aug_reverberation: bool = True

    # Batch settings for speaker verification
    batch_size: int = 32
    samples_per_speaker: int = 2

    # Dataset paths
    cnceleb_dir: str = "data/cn-celeb"
    librispeech_dir: str = "data/LibriSpeech"
    voxceleb_dir: str = "data/voxceleb"

    # Use LibriSpeech speakers as additional training data
    use_librispeech_speakers: bool = True

    # Encoder dimension (must match Zipformer output)
    encoder_dim: int = 384


@dataclass
class RichAudioHeadsTrainingConfig:
    """Configuration for training rich audio heads alongside ASR."""

    # Whether to enable rich audio heads training
    enabled: bool = False

    # Loss weights for each head (0 = disabled)
    emotion_weight: float = 0.1
    language_weight: float = 0.1
    paralinguistics_weight: float = 0.1
    pitch_weight: float = 0.05  # Lower weight for regression tasks
    phoneme_weight: float = 0.2  # Important for ROVER
    singing_weight: float = 0.05
    timestamp_weight: float = 0.1

    # Pitch head specifics
    pitch_f0_weight: float = 0.5  # Within pitch loss
    pitch_voiced_weight: float = 0.5

    # Singing head specifics
    singing_binary_weight: float = 0.5
    singing_technique_weight: float = 0.5

    # Timestamp head specifics
    timestamp_boundary_weight: float = 0.7
    timestamp_offset_weight: float = 0.3

    # Label smoothing for classification heads
    label_smoothing: float = 0.0

    # Encoder dimension (must match model)
    encoder_dim: int = 384


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""

    # Optimizer type: "adamw", "sgd"
    optimizer_type: str = "adamw"

    # Base learning rate
    # NOTE: For streaming Zipformer-large, use lower LR (1e-4) vs standard (4e-3)
    # to avoid training divergence at 700-1000 steps
    learning_rate: float = 1e-4

    # Weight decay
    weight_decay: float = 0.01

    # Adam betas
    beta1: float = 0.9
    beta2: float = 0.98

    # Adam epsilon
    eps: float = 1e-9

    # Gradient clipping (max norm)
    # Critical for stability - set to 1.0 for streaming Zipformer
    grad_clip: float = 1.0

    # Whether to use gradient accumulation
    grad_accumulation_steps: int = 1


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""

    # Scheduler type: "warmup_constant", "warmup_cosine", "warmup_linear"
    scheduler_type: str = "warmup_cosine"

    # Warmup steps
    # Use longer warmup (10k) for stability
    warmup_steps: int = 10000

    # Total training steps
    total_steps: int = 500000

    # Minimum learning rate (for cosine decay)
    min_lr: float = 1e-6

    # Cooldown steps (linear decay at end)
    cooldown_steps: int = 50000


@dataclass
class TrainingConfig:
    """Complete training configuration for Zipformer ASR."""

    # === Model Configuration ===
    # Zipformer architecture variant
    model_variant: str = "streaming-large"  # "streaming-small", "streaming-medium", "streaming-large"

    # Vocab size (Whisper tokenizer: 51865)
    vocab_size: int = 51865

    # Blank token ID
    blank_id: int = 0

    # === Streaming Configuration ===
    # Chunk sizes for streaming (in frames, 10ms each)
    chunk_sizes: tuple[int, ...] = (32, 64)  # 320ms, 640ms

    # Whether to use causal attention
    causal: bool = True

    # === Loss Configuration ===
    # Loss type: "transducer", "ctc", "cr_ctc" (joint transducer + CTC)
    loss_type: str = "cr_ctc"

    # Weight for CTC loss in CR-CTC
    # Paper recommends 0.3 for joint training
    ctc_weight: float = 0.3

    # Transducer loss pruning bounds
    # Larger bounds = more computation but potentially better gradients
    prune_range: int = 5

    # Simple loss computation (faster, less memory)
    simple_loss: bool = True

    # === Training Parameters ===
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    # === Rich Audio Heads ===
    rich_audio_heads: RichAudioHeadsTrainingConfig = field(
        default_factory=RichAudioHeadsTrainingConfig,
    )

    # === Speaker Embedding Training ===
    speaker: SpeakerTrainingConfig = field(
        default_factory=SpeakerTrainingConfig,
    )

    # Batch size per device
    batch_size: int = 8

    # Max audio duration in seconds (for bucketing)
    max_duration: float = 30.0

    # Number of epochs
    num_epochs: int = 100

    # Steps per epoch (if not using full dataset)
    steps_per_epoch: int | None = None

    # === Stability Mitigations ===
    # Enable gradient checkpointing (reduce memory, slower)
    gradient_checkpointing: bool = False

    # Skip batches with NaN/Inf loss
    skip_nan_loss: bool = True

    # Loss scaling for mixed precision (not used in MLX)
    loss_scale: float = 1.0

    # Maximum loss value before skipping batch
    max_loss: float = 100.0

    # === Regularization ===
    dropout: float = 0.1

    # Label smoothing
    label_smoothing: float = 0.0

    # SpecAugment configuration
    spec_augment: bool = True
    freq_mask_max: int = 27
    time_mask_max: int = 100
    num_freq_masks: int = 2
    num_time_masks: int = 10

    # === Checkpointing ===
    checkpoint_dir: str = "checkpoints/zipformer_training"

    # Save checkpoint every N steps
    save_every_n_steps: int = 5000

    # Keep only last N checkpoints
    keep_last_n_checkpoints: int = 5

    # E9 fix (optional): archive older step checkpoints as `.tar.gz` before deletion.
    archive_old_checkpoints: bool = False

    # === Logging ===
    log_every_n_steps: int = 100

    # Validation every N steps
    val_every_n_steps: int = 2000

    # === Dataset ===
    train_manifest: str = "data/LibriSpeech/train-960/manifest.json"
    val_manifest: str = "data/LibriSpeech/dev-clean/manifest.json"

    # Number of data loading workers
    num_workers: int = 4

    # Seed for reproducibility
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "model_variant": self.model_variant,
            "vocab_size": self.vocab_size,
            "blank_id": self.blank_id,
            "chunk_sizes": self.chunk_sizes,
            "causal": self.causal,
            "loss_type": self.loss_type,
            "ctc_weight": self.ctc_weight,
            "prune_range": self.prune_range,
            "simple_loss": self.simple_loss,
            "optimizer": {
                "optimizer_type": self.optimizer.optimizer_type,
                "learning_rate": self.optimizer.learning_rate,
                "weight_decay": self.optimizer.weight_decay,
                "beta1": self.optimizer.beta1,
                "beta2": self.optimizer.beta2,
                "eps": self.optimizer.eps,
                "grad_clip": self.optimizer.grad_clip,
                "grad_accumulation_steps": self.optimizer.grad_accumulation_steps,
            },
            "scheduler": {
                "scheduler_type": self.scheduler.scheduler_type,
                "warmup_steps": self.scheduler.warmup_steps,
                "total_steps": self.scheduler.total_steps,
                "min_lr": self.scheduler.min_lr,
                "cooldown_steps": self.scheduler.cooldown_steps,
            },
            "rich_audio_heads": {
                "enabled": self.rich_audio_heads.enabled,
                "emotion_weight": self.rich_audio_heads.emotion_weight,
                "language_weight": self.rich_audio_heads.language_weight,
                "paralinguistics_weight": self.rich_audio_heads.paralinguistics_weight,
                "pitch_weight": self.rich_audio_heads.pitch_weight,
                "phoneme_weight": self.rich_audio_heads.phoneme_weight,
                "singing_weight": self.rich_audio_heads.singing_weight,
                "timestamp_weight": self.rich_audio_heads.timestamp_weight,
                "pitch_f0_weight": self.rich_audio_heads.pitch_f0_weight,
                "pitch_voiced_weight": self.rich_audio_heads.pitch_voiced_weight,
                "singing_binary_weight": self.rich_audio_heads.singing_binary_weight,
                "singing_technique_weight": self.rich_audio_heads.singing_technique_weight,
                "timestamp_boundary_weight": self.rich_audio_heads.timestamp_boundary_weight,
                "timestamp_offset_weight": self.rich_audio_heads.timestamp_offset_weight,
                "label_smoothing": self.rich_audio_heads.label_smoothing,
                "encoder_dim": self.rich_audio_heads.encoder_dim,
            },
            "speaker": {
                "enabled": self.speaker.enabled,
                "num_speakers": self.speaker.num_speakers,
                "embedding_dim": self.speaker.embedding_dim,
                "aam_margin": self.speaker.aam_margin,
                "aam_scale": self.speaker.aam_scale,
                "loss_weight": self.speaker.loss_weight,
                "use_ssl": self.speaker.use_ssl,
                "mask_ratio": self.speaker.mask_ratio,
                "denoise_prob": self.speaker.denoise_prob,
                "aug_speed_perturb": self.speaker.aug_speed_perturb,
                "aug_noise": self.speaker.aug_noise,
                "aug_reverberation": self.speaker.aug_reverberation,
                "batch_size": self.speaker.batch_size,
                "samples_per_speaker": self.speaker.samples_per_speaker,
                "cnceleb_dir": self.speaker.cnceleb_dir,
                "librispeech_dir": self.speaker.librispeech_dir,
                "voxceleb_dir": self.speaker.voxceleb_dir,
                "use_librispeech_speakers": self.speaker.use_librispeech_speakers,
                "encoder_dim": self.speaker.encoder_dim,
            },
            "batch_size": self.batch_size,
            "max_duration": self.max_duration,
            "num_epochs": self.num_epochs,
            "steps_per_epoch": self.steps_per_epoch,
            "gradient_checkpointing": self.gradient_checkpointing,
            "skip_nan_loss": self.skip_nan_loss,
            "max_loss": self.max_loss,
            "dropout": self.dropout,
            "label_smoothing": self.label_smoothing,
            "spec_augment": self.spec_augment,
            "checkpoint_dir": self.checkpoint_dir,
            "save_every_n_steps": self.save_every_n_steps,
            "log_every_n_steps": self.log_every_n_steps,
            "val_every_n_steps": self.val_every_n_steps,
            "seed": self.seed,
        }

    def save(self, path: str) -> None:
        """Save config to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainingConfig":
        """Create config from dictionary."""
        optimizer = OptimizerConfig(**d.get("optimizer", {}))
        scheduler = SchedulerConfig(**d.get("scheduler", {}))
        rich_audio_heads = RichAudioHeadsTrainingConfig(
            **d.get("rich_audio_heads", {}),
        )
        speaker = SpeakerTrainingConfig(**d.get("speaker", {}))

        # Remove nested configs before creating main config
        d = {
            k: v
            for k, v in d.items()
            if k not in ("optimizer", "scheduler", "rich_audio_heads", "speaker")
        }

        return cls(
            optimizer=optimizer,
            scheduler=scheduler,
            rich_audio_heads=rich_audio_heads,
            speaker=speaker,
            **d,
        )

    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        """Load config from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


# Predefined configurations for different model variants
STREAMING_SMALL_CONFIG = TrainingConfig(
    model_variant="streaming-small",
    optimizer=OptimizerConfig(learning_rate=4e-3),  # Standard LR for small
    scheduler=SchedulerConfig(warmup_steps=5000),
    batch_size=16,
)

STREAMING_MEDIUM_CONFIG = TrainingConfig(
    model_variant="streaming-medium",
    optimizer=OptimizerConfig(learning_rate=2e-3),
    scheduler=SchedulerConfig(warmup_steps=7500),
    batch_size=12,
)

STREAMING_LARGE_CONFIG = TrainingConfig(
    model_variant="streaming-large",
    optimizer=OptimizerConfig(
        learning_rate=1e-4,  # Lower LR to avoid divergence
        grad_clip=1.0,
    ),
    scheduler=SchedulerConfig(
        warmup_steps=10000,  # Longer warmup
        total_steps=500000,
    ),
    batch_size=8,
    gradient_checkpointing=True,
)


def get_config(variant: str = "streaming-large") -> TrainingConfig:
    """Get predefined config by variant name."""
    configs = {
        "streaming-small": STREAMING_SMALL_CONFIG,
        "streaming-medium": STREAMING_MEDIUM_CONFIG,
        "streaming-large": STREAMING_LARGE_CONFIG,
    }
    if variant not in configs:
        raise ValueError(f"Unknown variant: {variant}. Available: {list(configs.keys())}")
    return configs[variant]
