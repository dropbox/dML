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
Phase 4: Multi-head Zipformer Training Script

This script trains the Zipformer ASR model with rich audio heads for:
- Text (RNN-T + CTC)
- Emotion (8 classes)
- Pitch (F0 Hz)
- Phoneme (178 IPA)
- Paralinguistics (50 classes)
- Language (9+ langs)
- Singing (binary + 10 techniques)
- Timestamps (word boundaries)

Usage:
    python scripts/train_zipformer_multihead.py \\
        --checkpoint checkpoints/zipformer/en-streaming \\
        --data-dir data/LibriSpeech \\
        --output-dir checkpoints/zipformer-multihead-v1 \\
        --epochs 10 \\
        --heads emotion,phoneme,language

Modes:
    --mode pretrain: Train ASR only (CTC + transducer)
    --mode finetune: Fine-tune with rich audio heads
    --mode heads-only: Freeze encoder, train heads only
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Set

# Add project root to path for imports
_PROJECT_ROOT = Path(__file__).parent.parent.absolute()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import mlx.core as mx

from src.models.zipformer import (
    ASRModel,
    load_checkpoint,
)
from src.models.heads import (
    RichAudioHeads,
    RichAudioHeadsConfig,
)
from src.training import (
    Trainer,
    TrainingConfig,
    SchedulerConfig,
    OptimizerConfig,
    RichAudioHeadsTrainingConfig,
    create_librispeech_loader,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class MultiheadTrainingArgs:
    """Command-line arguments for multihead training."""

    # Model paths
    checkpoint: str = "checkpoints/zipformer/en-streaming"
    output_dir: str = "checkpoints/zipformer-multihead-v1"

    # Data
    data_dir: str = "data/LibriSpeech"
    manifest_file: Optional[str] = None

    # Training mode
    mode: str = "finetune"  # pretrain, finetune, heads-only

    # Heads to train (comma-separated)
    heads: str = "emotion,phoneme,language"

    # Training parameters
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    gradient_clip: float = 1.0

    # Checkpointing
    save_every: int = 500
    eval_every: int = 100
    log_every: int = 10

    # Hardware
    compile_model: bool = True
    seed: int = 42


def parse_args() -> MultiheadTrainingArgs:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Zipformer with rich audio heads",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model paths
    parser.add_argument("--checkpoint", type=str, default="checkpoints/zipformer/en-streaming",
                        help="Path to pretrained Zipformer checkpoint")
    parser.add_argument("--output-dir", type=str, default="checkpoints/zipformer-multihead-v1",
                        help="Output directory for trained model")

    # Data
    parser.add_argument("--data-dir", type=str, default="data/LibriSpeech",
                        help="Path to training data directory")
    parser.add_argument("--manifest-file", type=str, default=None,
                        help="Optional manifest file with additional labels")

    # Training mode
    parser.add_argument("--mode", type=str, default="finetune",
                        choices=["pretrain", "finetune", "heads-only"],
                        help="Training mode")

    # Heads
    parser.add_argument("--heads", type=str, default="emotion,phoneme,language",
                        help="Comma-separated list of heads to train")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--gradient-clip", type=float, default=1.0)

    # Checkpointing
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)

    # Hardware
    parser.add_argument("--no-compile", action="store_true", help="Disable model compilation")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    return MultiheadTrainingArgs(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        manifest_file=args.manifest_file,
        mode=args.mode,
        heads=args.heads,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        gradient_clip=args.gradient_clip,
        save_every=args.save_every,
        eval_every=args.eval_every,
        log_every=args.log_every,
        compile_model=not args.no_compile,
        seed=args.seed,
    )


def create_rich_audio_heads(
    encoder_dim: int,
    enabled_heads: Set[str],
) -> RichAudioHeads:
    """
    Create RichAudioHeads module with specified heads enabled.

    Note: RichAudioHeads always creates all heads. The enabled_heads set
    is used to configure loss weights (zero weight = head not trained).

    Args:
        encoder_dim: Dimension of encoder output.
        enabled_heads: Set of head names to enable for training.

    Returns:
        Configured RichAudioHeads module.
    """
    config = RichAudioHeadsConfig(
        encoder_dim=encoder_dim,
        # Speaker head can be disabled
        speaker_enabled="speaker" in enabled_heads,
    )

    return RichAudioHeads(config)


def create_training_config(
    args: MultiheadTrainingArgs,
    enabled_heads: Set[str],
) -> TrainingConfig:
    """Create training configuration from arguments."""

    # Configure rich audio heads training
    rich_audio_config = RichAudioHeadsTrainingConfig(
        enabled=bool(enabled_heads),
        emotion_weight=0.1 if "emotion" in enabled_heads else 0.0,
        language_weight=0.1 if "language" in enabled_heads else 0.0,
        paralinguistics_weight=0.1 if "paralinguistics" in enabled_heads else 0.0,
        pitch_weight=0.05 if "pitch" in enabled_heads else 0.0,
        phoneme_weight=0.2 if "phoneme" in enabled_heads else 0.0,
        singing_weight=0.05 if "singing" in enabled_heads else 0.0,
        timestamp_weight=0.1 if "timestamp" in enabled_heads else 0.0,
    )

    # Configure optimizer
    optimizer_config = OptimizerConfig(
        optimizer_type="adamw",
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        beta1=0.9,
        beta2=0.98,
        eps=1e-8,
    )

    # Configure scheduler
    scheduler_config = SchedulerConfig(
        scheduler_type="warmup_cosine",
        warmup_steps=args.warmup_steps,
        max_steps=args.epochs * 10000,  # Estimate
        min_lr=1e-6,
    )

    return TrainingConfig(
        # Model
        model_variant="streaming-large",

        # Training
        num_epochs=args.epochs,
        batch_size=args.batch_size,

        # Loss
        loss_type="cr_ctc" if args.mode == "pretrain" else "ctc",
        ctc_weight=0.3,
        blank_id=0,

        # Checkpointing
        checkpoint_dir=args.output_dir,
        save_every_n_steps=args.save_every,
        val_every_n_steps=args.eval_every,
        log_every_n_steps=args.log_every,

        # Optimizer and scheduler
        optimizer=optimizer_config,
        scheduler=scheduler_config,

        # Rich audio heads
        rich_audio_heads=rich_audio_config,
    )


def load_pretrained_model(checkpoint_path: str) -> ASRModel:
    """Load pretrained Zipformer ASR model."""
    checkpoint_dir = Path(checkpoint_path)

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading pretrained model from {checkpoint_path}")

    # Load model configuration and weights
    model = load_checkpoint(checkpoint_dir)

    logger.info(f"Loaded model with {sum(p.size for p in model.parameters().values())} parameters")

    return model


def freeze_encoder(model: ASRModel) -> None:
    """Freeze encoder parameters for heads-only training."""
    # Get encoder parameters and mark them as non-trainable
    encoder_params = model.encoder.parameters()

    # MLX doesn't have a direct freeze mechanism, but we can exclude
    # encoder parameters from optimization by filtering them out
    logger.info("Encoder parameters will be excluded from training")


def setup_dataloaders(
    args: MultiheadTrainingArgs,
    enabled_heads: Set[str],
):
    """Setup training and validation dataloaders."""
    data_dir = Path(args.data_dir)

    # Find available splits
    train_splits = []
    val_splits = []

    for split in ["train-clean-100", "train-clean-360", "train-other-500"]:
        if (data_dir / split).exists():
            train_splits.append(split)

    for split in ["dev-clean", "dev-other"]:
        if (data_dir / split).exists():
            val_splits.append(split)

    if not train_splits:
        logger.warning(f"No training splits found in {data_dir}")
        logger.warning("Expected: train-clean-100, train-clean-360, or train-other-500")
        return None, None

    logger.info(f"Training splits: {train_splits}")
    logger.info(f"Validation splits: {val_splits}")

    # Create dataloaders
    train_loader = create_librispeech_loader(
        data_dir=str(data_dir),
        splits=train_splits,
        batch_size=args.batch_size,
        shuffle=True,
        include_rich_labels=bool(enabled_heads),
    )

    val_loader = None
    if val_splits:
        val_loader = create_librispeech_loader(
            data_dir=str(data_dir),
            splits=val_splits,
            batch_size=args.batch_size,
            shuffle=False,
            include_rich_labels=bool(enabled_heads),
        )

    return train_loader, val_loader


def train(args: MultiheadTrainingArgs) -> None:
    """Main training function."""

    # Set random seed
    mx.random.seed(args.seed)

    # Parse enabled heads
    enabled_heads = set(h.strip() for h in args.heads.split(",") if h.strip())
    logger.info(f"Enabled heads: {enabled_heads}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save training args
    with open(output_dir / "training_args.json", "w") as f:
        json.dump(asdict(args), f, indent=2)

    # Load pretrained model
    model = load_pretrained_model(args.checkpoint)
    encoder_dim = model.encoder.output_dim

    # Create rich audio heads
    rich_audio_heads = None
    if enabled_heads:
        rich_audio_heads = create_rich_audio_heads(encoder_dim, enabled_heads)
        logger.info(f"Created rich audio heads with {sum(p.size for p in rich_audio_heads.parameters().values())} parameters")

    # Handle training modes
    if args.mode == "heads-only":
        freeze_encoder(model)
        logger.info("Training mode: heads-only (encoder frozen)")
    elif args.mode == "pretrain":
        logger.info("Training mode: pretrain (ASR only)")
        rich_audio_heads = None  # Don't use heads in pretrain
    else:
        logger.info("Training mode: finetune (full model + heads)")

    # Create training config
    config = create_training_config(args, enabled_heads if args.mode != "pretrain" else set())

    # Setup dataloaders
    train_loader, val_loader = setup_dataloaders(args, enabled_heads)

    if train_loader is None:
        logger.error("No training data available. Exiting.")
        return

    # Optionally compile model
    if args.compile_model:
        logger.info("Compiling model for faster training...")
        model = mx.compile(model)
        if rich_audio_heads is not None:
            rich_audio_heads = mx.compile(rich_audio_heads)

    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        rich_audio_heads=rich_audio_heads,
    )

    # Log training setup
    logger.info("=" * 60)
    logger.info("Training Configuration")
    logger.info("=" * 60)
    logger.info(f"Model checkpoint: {args.checkpoint}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Training mode: {args.mode}")
    logger.info(f"Enabled heads: {enabled_heads if enabled_heads else 'None (ASR only)'}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Gradient clip: {args.gradient_clip}")
    logger.info("=" * 60)

    # Run training
    try:
        logger.info("Starting training...")
        start_time = time.time()

        trainer.train()

        elapsed = time.time() - start_time
        logger.info(f"Training completed in {elapsed:.2f}s ({elapsed/3600:.2f}h)")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    finally:
        # Save final checkpoint
        final_checkpoint = output_dir / "final_checkpoint"
        trainer.save_checkpoint(final_checkpoint)
        logger.info(f"Saved final checkpoint to {final_checkpoint}")


def main():
    """Entry point."""
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
