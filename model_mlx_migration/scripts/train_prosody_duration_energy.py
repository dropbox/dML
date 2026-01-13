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
Train Prosody Duration/Energy Predictor.

Trains a model to predict duration and energy multipliers from emotion type.
Uses the same architecture as ProsodyContourPredictorV2 with arousal-aware paths.

Usage:
    # Train with mock data (pipeline test)
    python scripts/train_prosody_duration_energy.py --mock --epochs 20

    # Train with real data
    python scripts/train_prosody_duration_energy.py \
        --data data/prosody/multilingual_train.json \
        --epochs 50

    # Continue training from checkpoint
    python scripts/train_prosody_duration_energy.py \
        --data data/prosody/multilingual_train.json \
        --checkpoint models/prosody_duration_energy/checkpoint_20.npz \
        --epochs 30
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from tools.pytorch_to_mlx.converters.models.kokoro import (
    ProsodyDurationEnergyPredictor,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Target multipliers (data-driven from multilingual analysis)
# Note: These are perceptual targets, may differ from raw audio measurements
DURATION_TARGETS = {
    0: 1.00,   # NEUTRAL
    40: 0.93,  # ANGRY (faster)
    41: 1.15,  # SAD (slower)
    42: 0.95,  # EXCITED (faster)
    45: 1.00,  # CALM
    48: 1.05,  # FRUSTRATED (slower)
    49: 0.90,  # NERVOUS (faster)
    50: 0.92,  # SURPRISED (faster)
}

ENERGY_TARGETS = {
    0: 1.00,   # NEUTRAL
    40: 1.40,  # ANGRY (louder)
    41: 0.90,  # SAD (quieter)
    42: 1.30,  # EXCITED (louder)
    45: 0.95,  # CALM (quieter)
    48: 1.35,  # FRUSTRATED (louder)
    49: 1.20,  # NERVOUS (louder)
    50: 1.50,  # SURPRISED (louder)
}


def generate_mock_data(num_samples: int = 200) -> List[Dict]:
    """Generate mock training data for pipeline testing."""
    data = []
    emotions = list(DURATION_TARGETS.keys())

    for _ in range(num_samples):
        emotion = np.random.choice(emotions)
        dur_target = DURATION_TARGETS[emotion]
        energy_target = ENERGY_TARGETS[emotion]

        # Add some noise to targets
        dur_noise = np.random.normal(0, 0.05)
        energy_noise = np.random.normal(0, 0.1)

        data.append({
            "prosody_type": emotion,
            "duration_mult": dur_target + dur_noise,
            "energy_mult": energy_target + energy_noise,
        })

    return data


def load_training_data(data_path: Path, use_perceptual_targets: bool = True) -> List[Dict]:
    """
    Load training data with target multipliers.

    Args:
        data_path: Path to JSON training data
        use_perceptual_targets: If True, use hardcoded perceptual targets.
                               If False, derive targets from data statistics.
    """
    logger.info(f"Loading data from {data_path}...")

    with open(data_path) as f:
        raw_data = json.load(f)

    # Count samples per emotion
    emotion_counts = defaultdict(int)
    for sample in raw_data:
        pid = sample.get("prosody_type", 0)
        if pid in DURATION_TARGETS:
            emotion_counts[pid] += 1

    if use_perceptual_targets:
        # Use hardcoded perceptual targets (better for TTS quality)
        logger.info("Using perceptual targets (hardcoded)")
        data = []
        for pid in DURATION_TARGETS.keys():
            count = emotion_counts.get(pid, 0)
            if count == 0:
                continue

            # Use predefined perceptual targets
            dur_target = DURATION_TARGETS[pid]
            energy_target = ENERGY_TARGETS[pid]

            # Add some noise for regularization
            for _ in range(min(count, 1000)):
                dur_noise = np.random.normal(0, 0.02)
                energy_noise = np.random.normal(0, 0.05)
                data.append({
                    "prosody_type": pid,
                    "duration_mult": np.clip(dur_target + dur_noise, 0.7, 1.5),
                    "energy_mult": np.clip(energy_target + energy_noise, 0.7, 1.8),
                })
    else:
        # Derive targets from data statistics
        logger.info("Deriving targets from data statistics")
        duration_by_emotion = defaultdict(list)
        energy_by_emotion = defaultdict(list)

        for sample in raw_data:
            pid = sample.get("prosody_type", 0)
            duration = sample.get("duration_s", 0)
            energy = sample.get("energy_rms", 0)
            text = sample.get("text", "")

            if pid not in DURATION_TARGETS:
                continue

            if duration > 0 and energy > 0 and len(text) > 0:
                dur_per_char = duration / len(text)
                duration_by_emotion[pid].append(dur_per_char)
                energy_by_emotion[pid].append(energy)

        neutral_dur = np.mean(duration_by_emotion[0]) if 0 in duration_by_emotion else 1.0
        neutral_energy = np.mean(energy_by_emotion[0]) if 0 in energy_by_emotion else 1.0

        data = []
        for pid in DURATION_TARGETS.keys():
            if pid not in duration_by_emotion:
                continue

            dur_mult = np.mean(duration_by_emotion[pid]) / neutral_dur
            energy_mult = np.mean(energy_by_emotion[pid]) / neutral_energy

            dur_mult = np.clip(dur_mult, 0.7, 1.5)
            energy_mult = np.clip(energy_mult, 0.7, 1.8)

            count = len(duration_by_emotion[pid])
            for _ in range(min(count, 1000)):
                data.append({
                    "prosody_type": pid,
                    "duration_mult": dur_mult,
                    "energy_mult": energy_mult,
                })

    logger.info(f"Created {len(data)} training samples from {len(raw_data)} raw samples")
    return data


def compute_loss(
    model: ProsodyDurationEnergyPredictor,
    batch: Dict[str, mx.array],
) -> Tuple[mx.array, Dict[str, float]]:
    """Compute loss for a batch."""
    prosody_ids = batch["prosody_type"]
    dur_targets = batch["duration_mult"]
    energy_targets = batch["energy_mult"]

    # Forward pass
    dur_pred, energy_pred = model(prosody_ids)
    dur_pred = dur_pred.squeeze(-1)
    energy_pred = energy_pred.squeeze(-1)

    # MSE loss
    dur_loss = mx.mean((dur_pred - dur_targets) ** 2)
    energy_loss = mx.mean((energy_pred - energy_targets) ** 2)

    # Combined loss (equal weight)
    total_loss = dur_loss + energy_loss

    metrics = {
        "dur_loss": float(dur_loss),
        "energy_loss": float(energy_loss),
        "total_loss": float(total_loss),
        "dur_mae": float(mx.mean(mx.abs(dur_pred - dur_targets))),
        "energy_mae": float(mx.mean(mx.abs(energy_pred - energy_targets))),
    }

    return total_loss, metrics


def train(
    model: ProsodyDurationEnergyPredictor,
    train_data: List[Dict],
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    output_dir: Path = Path("models/prosody_duration_energy"),
    checkpoint_path: Path = None,
):
    """Train the model."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize optimizer
    optimizer = optim.Adam(learning_rate=lr)

    # Load checkpoint if provided
    start_epoch = 0
    if checkpoint_path and checkpoint_path.exists():
        logger.info(f"Loading checkpoint from {checkpoint_path}...")
        weights = mx.load(str(checkpoint_path))
        model.update(weights)
        start_epoch = int(checkpoint_path.stem.split("_")[-1])

    # Create batches
    def get_batches():
        np.random.shuffle(train_data)
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + batch_size]
            yield {
                "prosody_type": mx.array([int(s["prosody_type"]) for s in batch], dtype=mx.int32),
                "duration_mult": mx.array([float(s["duration_mult"]) for s in batch], dtype=mx.float32),
                "energy_mult": mx.array([float(s["energy_mult"]) for s in batch], dtype=mx.float32),
            }

    # Loss function with gradient
    loss_fn = nn.value_and_grad(model, compute_loss)

    best_loss = float('inf')

    for epoch in range(start_epoch, start_epoch + epochs):
        epoch_metrics = defaultdict(list)

        for batch in get_batches():
            (loss, metrics), grads = loss_fn(model, batch)
            optimizer.update(model, grads)
            mx.eval(model.parameters())

            for k, v in metrics.items():
                epoch_metrics[k].append(v)

        # Compute epoch averages
        avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}

        logger.info(
            f"Epoch {epoch + 1}: "
            f"loss={avg_metrics['total_loss']:.4f}, "
            f"dur_mae={avg_metrics['dur_mae']:.4f}, "
            f"energy_mae={avg_metrics['energy_mae']:.4f}"
        )

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            ckpt_path = output_dir / f"checkpoint_{epoch + 1}.npz"
            flat_params = dict(tree_flatten(model.parameters()))
            mx.savez(str(ckpt_path), **flat_params)
            logger.info(f"Saved checkpoint to {ckpt_path}")

        # Save best model
        if avg_metrics['total_loss'] < best_loss:
            best_loss = avg_metrics['total_loss']
            best_path = output_dir / "best_model.npz"
            # Flatten parameters for saving
            flat_params = dict(tree_flatten(model.parameters()))
            mx.savez(str(best_path), **flat_params)
            logger.info(f"Saved best model (loss={best_loss:.4f})")

    # Save final model
    final_path = output_dir / "final_model.npz"
    flat_params = dict(tree_flatten(model.parameters()))
    mx.savez(str(final_path), **flat_params)
    logger.info(f"Saved final model to {final_path}")

    return model


def evaluate(model: ProsodyDurationEnergyPredictor):
    """Evaluate model on all emotion types."""
    logger.info("\nEvaluation Results:")
    logger.info(f"{'Emotion':<12} {'Dur Pred':<10} {'Dur Target':<12} {'Dur Err':<10} {'Energy Pred':<12} {'Energy Target':<14} {'Energy Err':<10}")
    logger.info("-" * 90)

    for pid, dur_target in DURATION_TARGETS.items():
        energy_target = ENERGY_TARGETS[pid]

        prosody_id = mx.array([pid])
        dur_pred, energy_pred = model(prosody_id)
        mx.eval(dur_pred, energy_pred)

        dur_val = float(dur_pred.squeeze())
        energy_val = float(energy_pred.squeeze())

        dur_err = abs(dur_val - dur_target)
        energy_err = abs(energy_val - energy_target)

        name = {
            0: "NEUTRAL", 40: "ANGRY", 41: "SAD", 42: "EXCITED",
            45: "CALM", 48: "FRUSTRATED", 49: "NERVOUS", 50: "SURPRISED"
        }.get(pid, f"TYPE_{pid}")

        logger.info(
            f"{name:<12} {dur_val:<10.3f} {dur_target:<12.3f} {dur_err:<10.3f} "
            f"{energy_val:<12.3f} {energy_target:<14.3f} {energy_err:<10.3f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Train prosody duration/energy predictor")
    parser.add_argument("--data", type=Path, help="Path to training data JSON")
    parser.add_argument("--mock", action="store_true", help="Use mock data for testing")
    parser.add_argument("--mock-samples", type=int, default=200, help="Number of mock samples")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output-dir", type=Path, default=Path("models/prosody_duration_energy"))
    parser.add_argument("--checkpoint", type=Path, help="Path to checkpoint to resume from")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate, no training")
    parser.add_argument("--prosody-embeddings", type=Path,
                       default=Path("models/prosody_embeddings_orthogonal/final.safetensors"),
                       help="Path to prosody embeddings")
    args = parser.parse_args()

    # Create model
    model = ProsodyDurationEnergyPredictor()

    # Load prosody embeddings
    if args.prosody_embeddings.exists():
        logger.info(f"Loading prosody embeddings from {args.prosody_embeddings}...")
        emb_weights = mx.load(str(args.prosody_embeddings))
        if "embedding.weight" in emb_weights:
            emb_weight = emb_weights["embedding.weight"]
            num_types, dim = emb_weight.shape
            model.embedding = nn.Embedding(num_types, dim)
            model.embedding.weight = emb_weight
            model.num_prosody_types = num_types
            logger.info(f"Loaded embeddings: {num_types} types, {dim} dims")

    if args.eval_only:
        if args.checkpoint and args.checkpoint.exists():
            weights = mx.load(str(args.checkpoint))
            model.update(weights)
        evaluate(model)
        return

    # Load or generate training data
    if args.mock:
        logger.info(f"Generating {args.mock_samples} mock training samples...")
        train_data = generate_mock_data(args.mock_samples)
    elif args.data and args.data.exists():
        train_data = load_training_data(args.data)
    else:
        logger.error("Must provide --data or --mock")
        return

    # Train
    model = train(
        model,
        train_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
    )

    # Final evaluation
    evaluate(model)


if __name__ == "__main__":
    main()
