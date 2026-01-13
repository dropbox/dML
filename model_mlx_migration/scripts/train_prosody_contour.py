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
Train Prosody Contour Model (Path C)

Instead of static F0 multipliers, learns actual F0 contour patterns
from real emotional speech data.

Key improvements over v5:
1. Learns contour SHAPES, not just averages
2. Per-frame F0 predictions
3. Uses real samples instead of synthetic sentences

Usage:
    python scripts/train_prosody_contour.py --epochs 50
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProsodyContourPredictor(nn.Module):
    """
    Predicts F0 contour offsets based on prosody embedding.

    Instead of a scalar multiplier (v5), outputs per-frame F0 offsets
    that capture the dynamic nature of emotional speech.
    """

    def __init__(self, prosody_dim: int = 768, hidden_dim: int = 256, contour_len: int = 50):
        super().__init__()
        self.contour_len = contour_len

        # Project prosody embedding
        self.prosody_proj = nn.Linear(prosody_dim, hidden_dim)

        # Learn contour pattern
        self.contour_fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.contour_fc2 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.contour_out = nn.Linear(hidden_dim * 2, contour_len)

        # Also predict scalar statistics for regularization
        self.stats_fc = nn.Linear(hidden_dim, 3)  # mean, std, range

    def __call__(self, prosody_emb: mx.array) -> Tuple[mx.array, mx.array]:
        """
        Args:
            prosody_emb: (batch, prosody_dim) - emotion embedding

        Returns:
            contour: (batch, contour_len) - normalized F0 contour [0, 1]
            stats: (batch, 3) - predicted mean, std, range
        """
        # Project
        h = self.prosody_proj(prosody_emb)
        h = nn.gelu(h)

        # Contour prediction
        c = self.contour_fc1(h)
        c = nn.gelu(c)
        c = self.contour_fc2(c)
        c = nn.gelu(c)
        contour = mx.sigmoid(self.contour_out(c))  # [0, 1] range

        # Stats prediction
        stats = self.stats_fc(h)

        return contour, stats


def load_contour_data(path: str) -> List[Dict]:
    """Load contour training data."""
    with open(path, 'r') as f:
        data = json.load(f)

    # Filter samples with valid contours
    valid = [s for s in data if s.get('f0_contour') and len(s['f0_contour']) == 50]
    logger.info(f"Loaded {len(valid)}/{len(data)} valid samples from {path}")
    return valid


def load_prosody_embeddings(path: str) -> mx.array:
    """Load prosody embedding table."""
    weights = mx.load(str(path))
    for key in weights:
        if 'embedding' in key.lower():
            logger.info(f"Loaded embedding from key: {key}")
            return weights[key]
    raise ValueError(f"No embedding found in {path}")


# Prosody type mapping
PROSODY_TYPES = {
    'neutral': 0,
    'angry': 40,
    'sad': 41,
    'excited': 42,
    'happy': 42,
    'calm': 45,
    'frustrated': 48,
    'disgust': 48,
    'nervous': 49,
    'fear': 49,
    'fearful': 49,
    'surprised': 50,
    'surprise': 50,
}


def get_prosody_id(emotion) -> int:
    """Map emotion name to prosody ID."""
    if isinstance(emotion, int):
        return emotion
    emotion_lower = str(emotion).lower().strip()
    return PROSODY_TYPES.get(emotion_lower, 0)


def train_epoch(
    model: ProsodyContourPredictor,
    embedding_table: mx.array,
    data: List[Dict],
    optimizer: optim.Optimizer,
    batch_size: int = 32,
) -> Tuple[float, float]:
    """Train for one epoch."""

    random.shuffle(data)
    total_loss = 0.0
    total_contour_loss = 0.0
    num_batches = 0

    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        if len(batch) < batch_size // 2:
            continue

        # Prepare batch
        prosody_ids = []
        target_contours = []
        target_stats = []

        for sample in batch:
            emotion = sample.get('emotion', sample.get('prosody_type', 'neutral'))
            pid = get_prosody_id(emotion)

            prosody_ids.append(pid)
            target_contours.append(sample['f0_contour'])
            target_stats.append([
                sample.get('f0_mean', 150) / 300,  # Normalize to ~[0, 1]
                sample.get('f0_std', 30) / 100,
                sample.get('f0_range', 100) / 200,
            ])

        prosody_ids = mx.array(prosody_ids)
        target_contours = mx.array(target_contours)
        target_stats = mx.array(target_stats)

        # Get prosody embeddings
        prosody_emb = embedding_table[prosody_ids]

        def loss_fn(model):
            pred_contour, pred_stats = model(prosody_emb)

            # Contour loss (main objective)
            contour_loss = mx.mean((pred_contour - target_contours) ** 2)

            # Stats loss (regularization)
            stats_loss = mx.mean((pred_stats - target_stats) ** 2)

            # Smoothness loss (encourage smooth contours)
            contour_diff = pred_contour[:, 1:] - pred_contour[:, :-1]
            smoothness_loss = mx.mean(contour_diff ** 2)

            total = contour_loss + 0.1 * stats_loss + 0.01 * smoothness_loss
            return total

        loss, grads = nn.value_and_grad(model, loss_fn)(model)
        contour_loss = loss  # Simplify for logging
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        total_loss += float(loss)
        total_contour_loss += float(contour_loss)
        num_batches += 1

    return total_loss / max(num_batches, 1), total_contour_loss / max(num_batches, 1)


def evaluate(
    model: ProsodyContourPredictor,
    embedding_table: mx.array,
    data: List[Dict],
    batch_size: int = 64,
) -> Tuple[float, float]:
    """Evaluate on validation set."""

    total_loss = 0.0
    num_batches = 0

    for i in range(0, min(len(data), 1000), batch_size):
        batch = data[i:i + batch_size]

        prosody_ids = []
        target_contours = []

        for sample in batch:
            emotion = sample.get('emotion', sample.get('prosody_type', 'neutral'))
            pid = get_prosody_id(emotion)

            prosody_ids.append(pid)
            target_contours.append(sample['f0_contour'])

        prosody_ids = mx.array(prosody_ids)
        target_contours = mx.array(target_contours)

        prosody_emb = embedding_table[prosody_ids]
        pred_contour, _ = model(prosody_emb)

        contour_loss = mx.mean((pred_contour - target_contours) ** 2)
        mx.eval(contour_loss)

        total_loss += float(contour_loss)
        num_batches += 1

    return total_loss / max(num_batches, 1), total_loss / max(num_batches, 1)


def main():
    parser = argparse.ArgumentParser(description='Train prosody contour model')
    parser.add_argument('--train', default='data/prosody/contours_train.json')
    parser.add_argument('--val', default='data/prosody/contours_val.json')
    parser.add_argument('--prosody-embeddings', default='models/prosody_embeddings_orthogonal/final.safetensors')
    parser.add_argument('--output-dir', default='models/prosody_contour_v1')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden-dim', type=int, default=256)
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading training data...")
    train_data = load_contour_data(args.train)
    val_data = load_contour_data(args.val)

    # Load prosody embeddings
    logger.info("Loading prosody embeddings...")
    embedding_table = load_prosody_embeddings(args.prosody_embeddings)
    prosody_dim = embedding_table.shape[1]
    logger.info(f"Embedding table shape: {embedding_table.shape}")

    # Create model
    logger.info("Creating contour model...")
    model = ProsodyContourPredictor(
        prosody_dim=prosody_dim,
        hidden_dim=args.hidden_dim,
        contour_len=50,
    )

    # Count parameters
    flat_params = tree_flatten(model.parameters())
    num_params = sum(p.size for _, p in flat_params)
    logger.info(f"Model parameters: {num_params:,}")

    # Optimizer
    optimizer = optim.Adam(learning_rate=args.lr)

    # Training loop
    logger.info(f"\nStarting training for {args.epochs} epochs...")
    logger.info(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")

    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        train_loss, train_contour = train_epoch(
            model, embedding_table, train_data, optimizer, args.batch_size
        )

        if epoch % 5 == 0 or epoch == 1:
            val_loss, val_contour = evaluate(model, embedding_table, val_data)
            logger.info(
                f"Epoch {epoch}/{args.epochs} - "
                f"Train: {train_loss:.4f}, Val: {val_loss:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save best model
                weights = dict(tree_flatten(model.parameters()))
                mx.savez(str(output_dir / 'best_model.npz'), **weights)
                logger.info(f"  Saved best model (val_loss: {val_loss:.4f})")
        else:
            logger.info(f"Epoch {epoch}/{args.epochs} - Train: {train_loss:.4f}")

    # Save final model
    weights = dict(tree_flatten(model.parameters()))
    mx.savez(str(output_dir / 'final_model.npz'), **weights)

    # Save config
    config = {
        'prosody_dim': prosody_dim,
        'hidden_dim': args.hidden_dim,
        'contour_len': 50,
        'epochs': args.epochs,
        'best_val_loss': float(best_val_loss),
        'train_samples': len(train_data),
        'val_samples': len(val_data),
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    logger.info("\nTraining complete!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Model saved to: {output_dir}")


if __name__ == '__main__':
    main()
