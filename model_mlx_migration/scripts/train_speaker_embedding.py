#!/usr/bin/env python3
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
Speaker embedding training script for Phase 6.3.

Trains DELULU-style speaker embeddings on CN-Celeb + LibriSpeech
targeting <0.8% EER (compared to ECAPA-TDNN baseline ~2% EER).

Usage:
    python scripts/train_speaker_embedding.py --epochs 50
    python scripts/train_speaker_embedding.py --dry-run  # Test data loading
    python scripts/train_speaker_embedding.py --quick-test  # Quick validation run
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


from src.training.config import (
    SpeakerTrainingConfig,
    OptimizerConfig,
    SchedulerConfig,
)
from src.training.speaker_trainer import (
    create_speaker_trainer,
)
from src.training.speaker_dataloader import (
    SpeakerDataLoader,
    CNCelebDataset,
    LibriSpeechSpeakerDataset,
    CombinedSpeakerDataset,
    VerificationTrialLoader,
)


# Default paths for datasets
DEFAULT_CNCELEB_DIR = "data/cn-celeb"
DEFAULT_LIBRISPEECH_DIR = "data/LibriSpeech"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train DELULU-style speaker embeddings"
    )

    # Data paths
    parser.add_argument(
        "--cnceleb-dir",
        type=str,
        default=DEFAULT_CNCELEB_DIR,
        help="Path to CN-Celeb dataset",
    )
    parser.add_argument(
        "--librispeech-dir",
        type=str,
        default=DEFAULT_LIBRISPEECH_DIR,
        help="Path to LibriSpeech dataset",
    )
    parser.add_argument(
        "--use-librispeech",
        action="store_true",
        default=True,
        help="Include LibriSpeech speakers in training",
    )
    parser.add_argument(
        "--cnceleb-only",
        action="store_true",
        help="Only use CN-Celeb (disable LibriSpeech)",
    )

    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5000,
        help="Warmup steps",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=5.0,
        help="Gradient clipping max norm",
    )

    # Model parameters
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=256,
        help="Speaker embedding dimension",
    )
    parser.add_argument(
        "--encoder-dim",
        type=int,
        default=80,  # Use fbank features directly for standalone training
        help="Input feature dimension (80 for fbank)",
    )
    parser.add_argument(
        "--aam-margin",
        type=float,
        default=0.2,
        help="AAM-Softmax margin",
    )
    parser.add_argument(
        "--aam-scale",
        type=float,
        default=30.0,
        help="AAM-Softmax scale",
    )

    # Logging
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Log every N steps",
    )
    parser.add_argument(
        "--val-every",
        type=int,
        default=1000,
        help="Validate every N steps",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/speaker_embedding",
        help="Checkpoint directory",
    )

    # Testing modes
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only test data loading, no training",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Quick test with 10 batches",
    )

    # Resume training
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint (e.g., 'latest' or 'best')",
    )

    return parser.parse_args()


def load_datasets(args) -> tuple:
    """
    Load training and validation datasets.

    Returns:
        Tuple of (train_dataset, val_dataset, trial_loader).
    """
    datasets = []

    # Load CN-Celeb (primary)
    cnceleb_path = Path(args.cnceleb_dir)
    if cnceleb_path.exists():
        print(f"Loading CN-Celeb from {cnceleb_path}...")
        try:
            train_cnceleb = CNCelebDataset(str(cnceleb_path), split="train")
            print(f"  CN-Celeb train: {len(train_cnceleb)} samples, "
                  f"{train_cnceleb.num_speakers} speakers")
            datasets.append(train_cnceleb)

            # Load dev set for validation
            dev_cnceleb = CNCelebDataset(str(cnceleb_path), split="dev")
            print(f"  CN-Celeb dev: {len(dev_cnceleb)} samples, "
                  f"{dev_cnceleb.num_speakers} speakers")
        except Exception as e:
            print(f"  Warning: Could not load CN-Celeb: {e}")
            dev_cnceleb = None
    else:
        print(f"CN-Celeb not found at {cnceleb_path}")
        dev_cnceleb = None

    # Load LibriSpeech (optional)
    if args.use_librispeech and not args.cnceleb_only:
        libri_path = Path(args.librispeech_dir)
        if libri_path.exists():
            print(f"Loading LibriSpeech from {libri_path}...")
            try:
                # Use train-clean-100 and train-clean-360 for speaker training
                train_libri = LibriSpeechSpeakerDataset(
                    str(libri_path),
                    splits=["train-clean-100", "train-clean-360"],
                )
                print(f"  LibriSpeech: {len(train_libri)} samples, "
                      f"{train_libri.num_speakers} speakers")
                datasets.append(train_libri)
            except Exception as e:
                print(f"  Warning: Could not load LibriSpeech: {e}")
        else:
            print(f"LibriSpeech not found at {libri_path}")

    if not datasets:
        raise ValueError("No datasets found! Check paths.")

    # Combine datasets
    if len(datasets) > 1:
        train_dataset = CombinedSpeakerDataset(datasets)
        print(f"\nCombined: {len(train_dataset)} samples, "
              f"{train_dataset.num_speakers} speakers")
    else:
        train_dataset = datasets[0]

    # Validation dataset (use dev if available and non-empty, else train)
    if dev_cnceleb is not None and len(dev_cnceleb) > 0:
        val_dataset = dev_cnceleb
    else:
        # Use subset of training data for validation
        print("  Using training data subset for validation")
        val_dataset = train_dataset

    # Trial loader for EER computation
    trial_loader = VerificationTrialLoader(
        val_dataset,
        num_positive_pairs=2000,
        num_negative_pairs=2000,
    )

    return train_dataset, val_dataset, trial_loader


def run_dry_run(train_dataset, val_dataset, args):
    """Test data loading without training."""
    print("\n=== DRY RUN: Testing data loading ===")

    # Create data loader
    train_loader = SpeakerDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    print(f"\nTrain loader: ~{len(train_loader)} batches")

    # Load a few batches
    num_batches = 5
    print(f"\nLoading {num_batches} batches...")

    for i, batch in enumerate(train_loader):
        if i >= num_batches:
            break

        features = batch["features"]
        feature_lengths = batch["feature_lengths"]
        speaker_indices = batch["speaker_indices"]

        print(f"  Batch {i+1}:")
        print(f"    features: {features.shape}")
        print(f"    feature_lengths: {feature_lengths.shape}, "
              f"range [{int(feature_lengths.min())}, {int(feature_lengths.max())}]")
        print(f"    speaker_indices: {speaker_indices.shape}, "
              f"unique speakers: {len(set(int(x) for x in speaker_indices.tolist()))}")

    print("\n=== DRY RUN COMPLETE ===")
    return True


def run_quick_test(train_dataset, val_dataset, trial_loader, args):
    """Quick training test with 10 batches."""
    print("\n=== QUICK TEST: Training 10 batches ===")

    # Create config
    config = SpeakerTrainingConfig(
        encoder_dim=args.encoder_dim,
        embedding_dim=args.embedding_dim,
        num_speakers=train_dataset.num_speakers,
        aam_margin=args.aam_margin,
        aam_scale=args.aam_scale,
    )

    optimizer_config = OptimizerConfig(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
    )

    scheduler_config = SchedulerConfig(
        scheduler_type="warmup_cosine",
        warmup_steps=10,  # Quick warmup for test
        total_steps=100,
        cooldown_steps=0,
        min_lr=1e-6,
    )

    # Create trainer (standalone, no encoder)
    trainer = create_speaker_trainer(
        config=config,
        encoder=None,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
    )
    trainer.checkpoint_dir = Path(args.checkpoint_dir)

    # Create data loader
    train_loader = SpeakerDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    print("\nTraining 10 batches...")
    losses = []
    accuracies = []

    for i, batch in enumerate(train_loader):
        if i >= 10:
            break

        metrics = trainer.train_step(batch)
        losses.append(metrics.loss)
        accuracies.append(metrics.accuracy)

        print(f"  Step {i+1}: loss={metrics.loss:.4f}, "
              f"acc={metrics.accuracy:.4f}, time={metrics.time_ms:.1f}ms")

    # Summary
    print("\nResults:")
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Loss reduction: {losses[0] - losses[-1]:.4f}")
    print(f"  Avg accuracy: {sum(accuracies) / len(accuracies):.4f}")

    # Quick validation
    print("\nRunning validation...")
    val_loader = SpeakerDataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    val_metrics = trainer.validate(val_loader, trial_loader=None, max_batches=5)
    print(f"  Val loss: {val_metrics.loss:.4f}, Val acc: {val_metrics.accuracy:.4f}")

    print("\n=== QUICK TEST COMPLETE ===")
    return True


def train(train_dataset, val_dataset, trial_loader, args):
    """Full training loop."""
    print("\n=== TRAINING SPEAKER EMBEDDINGS ===")
    print("Target: <0.8% EER (ECAPA-TDNN baseline: ~2.0% EER)")

    # Calculate total steps
    steps_per_epoch = len(train_dataset) // args.batch_size
    total_steps = steps_per_epoch * args.epochs

    print("\nTraining configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Warmup steps: {args.warmup_steps}")
    print(f"  Speakers: {train_dataset.num_speakers}")

    # Create config
    config = SpeakerTrainingConfig(
        encoder_dim=args.encoder_dim,
        embedding_dim=args.embedding_dim,
        num_speakers=train_dataset.num_speakers,
        aam_margin=args.aam_margin,
        aam_scale=args.aam_scale,
    )

    optimizer_config = OptimizerConfig(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
    )

    scheduler_config = SchedulerConfig(
        scheduler_type="warmup_cosine",
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
        cooldown_steps=0,  # No cooldown - cosine decay to min_lr
        min_lr=1e-6,
    )

    # Create trainer
    trainer = create_speaker_trainer(
        config=config,
        encoder=None,  # Standalone training on fbank features
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
    )
    trainer.checkpoint_dir = Path(args.checkpoint_dir)

    # Resume if specified
    if args.resume:
        if trainer.load_checkpoint(args.resume):
            print(f"Resumed from checkpoint '{args.resume}'")
            print(f"  Step: {trainer.state.step}, Epoch: {trainer.state.epoch}, "
                  f"Best EER: {trainer.state.best_eer:.4f}")

    # Create data loaders
    train_loader = SpeakerDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_loader = SpeakerDataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Training log file
    log_dir = Path(args.checkpoint_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    training_log = {
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "warmup_steps": args.warmup_steps,
            "num_speakers": train_dataset.num_speakers,
            "total_samples": len(train_dataset),
        },
        "metrics": [],
    }

    print("\nStarting training...")
    print(f"  Logging to: {log_file}")
    start_time = time.time()

    # Training loop
    for epoch in range(trainer.state.epoch, args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")

        epoch_metrics = trainer.train_epoch(
            train_loader,
            val_loader=val_loader,
            trial_loader=trial_loader,
            log_every=args.log_every,
            val_every=args.val_every,
        )

        # Log epoch results
        training_log["metrics"].append({
            "epoch": epoch + 1,
            "loss": epoch_metrics["loss"],
            "accuracy": epoch_metrics["accuracy"],
            "best_eer": epoch_metrics["best_eer"],
            "time": time.time() - start_time,
        })

        # Save log
        with open(log_file, "w") as f:
            json.dump(training_log, f, indent=2)

        # Save checkpoint every epoch
        trainer.save_checkpoint(f"epoch_{epoch + 1}")
        trainer.save_checkpoint("latest")

        print(f"Epoch {epoch + 1} complete:")
        print(f"  Loss: {epoch_metrics['loss']:.4f}")
        print(f"  Accuracy: {epoch_metrics['accuracy']:.4f}")
        print(f"  Best EER: {epoch_metrics['best_eer']:.4f}")

    # Final validation with full EER computation
    print("\n=== FINAL VALIDATION ===")
    final_metrics = trainer.validate(val_loader, trial_loader, max_batches=1000)

    total_time = time.time() - start_time

    print("\nTraining complete!")
    print(f"  Total time: {total_time / 3600:.2f} hours")
    print(f"  Final loss: {final_metrics.loss:.4f}")
    print(f"  Final accuracy: {final_metrics.accuracy:.4f}")
    final_eer_str = f"{final_metrics.eer:.4f}" if final_metrics.eer is not None else "N/A"
    print(f"  Final EER: {final_eer_str}")
    print(f"  Best EER: {trainer.state.best_eer:.4f}")

    # Check if target achieved
    target_eer = 0.008  # 0.8%
    if trainer.state.best_eer <= target_eer:
        print(f"\n*** TARGET ACHIEVED: EER {trainer.state.best_eer:.4f} <= {target_eer:.4f} ***")
    else:
        print(f"\n  Target not yet achieved: {trainer.state.best_eer:.4f} > {target_eer:.4f}")
        print(f"  Gap: {(trainer.state.best_eer - target_eer) * 100:.2f}% absolute")

    return trainer.state.best_eer


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 60)
    print("DELULU Speaker Embedding Training")
    print("Phase 6.3: Train on CN-Celeb + LibriSpeech")
    print("Target: <0.8% EER")
    print("=" * 60)

    # Load datasets
    train_dataset, val_dataset, trial_loader = load_datasets(args)

    # Run appropriate mode
    if args.dry_run:
        run_dry_run(train_dataset, val_dataset, args)
    elif args.quick_test:
        run_quick_test(train_dataset, val_dataset, trial_loader, args)
    else:
        train(train_dataset, val_dataset, trial_loader, args)


if __name__ == "__main__":
    main()
