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
Train UNIFIED Prosody Model - THE BEST VERSION

This model predicts ALL emotional speech features simultaneously:
1. F0 multiplier (pitch level)
2. F0 contour (pitch shape over time)
3. Duration multiplier (speaking rate)
4. Energy multiplier (loudness)

Architecture: Emotion-aware transformer with multi-head outputs.

Usage:
    python scripts/train_prosody_unified.py \
        --train data/prosody/contours_train.json \
        --epochs 100 \
        --output-dir models/prosody_unified_v1
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

# THE BEST target values (data-driven + perceptual tuning)
# NOTE: CALM is combined with NEUTRAL as they are perceptually similar
TARGETS = {
    # prosody_id: (f0_mult, duration_mult, energy_mult)
    0:  (1.00, 1.00, 1.00),  # NEUTRAL (also used for CALM)
    40: (1.12, 0.80, 1.50),  # ANGRY - higher pitch, much faster, louder (reduced energy)
    41: (0.94, 1.25, 0.78),  # SAD - lower pitch, slower, quieter
    42: (1.10, 0.88, 1.35),  # EXCITED/HAPPY - moderate pitch, faster, louder (reduced to avoid artifacts)
    # CALM (45) -> mapped to NEUTRAL (0)
    48: (1.06, 0.92, 1.35),  # FRUSTRATED - slightly higher, slightly faster, louder
    49: (1.08, 0.90, 1.15),  # NERVOUS - higher pitch, faster
    50: (1.15, 0.82, 1.45),  # SURPRISED - high pitch, fast (reduced to avoid artifacts)
}

# Emotion groupings - CALM combined with NEUTRAL
HIGH_AROUSAL = {40, 42, 49, 50}  # angry, excited, nervous, surprised
LOW_AROUSAL = {41, 48}          # sad, frustrated (CALM removed)
NEUTRAL_LIKE = {0, 45}          # neutral AND calm combined


class ResidualBlock(nn.Module):
    """Residual block with layer norm."""
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        x = self.norm(x)
        x = nn.gelu(self.fc1(x))
        x = self.fc2(x)
        return x + residual


class UnifiedProsodyPredictor(nn.Module):
    """
    THE BEST prosody predictor - unified architecture for all features.

    Outputs:
    - f0_mult: (batch, 1) - F0 multiplier
    - f0_contour: (batch, contour_len) - F0 shape [0, 1]
    - duration_mult: (batch, 1) - Duration multiplier
    - energy_mult: (batch, 1) - Energy multiplier
    """

    def __init__(
        self,
        prosody_dim: int = 768,
        hidden_dim: int = 512,
        contour_len: int = 50,
        num_blocks: int = 4,
    ):
        super().__init__()
        self.contour_len = contour_len
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(prosody_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        # Emotion-type embedding (learnable)
        self.arousal_embedding = nn.Embedding(3, hidden_dim // 4)

        # Shared trunk with residual blocks
        self.trunk = [ResidualBlock(hidden_dim) for _ in range(num_blocks)]

        # High arousal path
        self.high_arousal_fc = nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim)
        self.high_arousal_block = ResidualBlock(hidden_dim)

        # Low arousal path
        self.low_arousal_fc = nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim)
        self.low_arousal_block = ResidualBlock(hidden_dim)

        # Feature-specific heads
        # F0 multiplier head
        self.f0_mult_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # F0 contour head (shape over time)
        self.f0_contour_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, contour_len),
        )

        # Duration multiplier head
        self.duration_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Energy multiplier head
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def __call__(
        self,
        prosody_emb: mx.array,
        prosody_ids: mx.array,
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
        """
        Forward pass.

        Returns:
            f0_mult: (batch, 1)
            f0_contour: (batch, contour_len)
            duration_mult: (batch, 1)
            energy_mult: (batch, 1)
        """
        # Input projection
        h = self.input_proj(prosody_emb)
        h = nn.gelu(h)
        h = self.input_norm(h)

        # Shared trunk
        for block in self.trunk:
            h = block(h)

        # Determine arousal type
        arousal_types = []
        for pid in prosody_ids.tolist():
            if pid in NEUTRAL_LIKE:
                arousal_types.append(0)
            elif pid in HIGH_AROUSAL:
                arousal_types.append(1)
            else:
                arousal_types.append(2)
        arousal_types = mx.array(arousal_types)

        # Get arousal embedding
        arousal_emb = self.arousal_embedding(arousal_types)
        h_with_arousal = mx.concatenate([h, arousal_emb], axis=-1)

        # Process through arousal-specific paths
        h_high = nn.gelu(self.high_arousal_fc(h_with_arousal))
        h_high = self.high_arousal_block(h_high)

        h_low = nn.gelu(self.low_arousal_fc(h_with_arousal))
        h_low = self.low_arousal_block(h_low)

        # Route based on arousal type
        is_high = (arousal_types == 1).astype(mx.float32).reshape(-1, 1)
        is_low = (arousal_types == 2).astype(mx.float32).reshape(-1, 1)
        is_neutral = (arousal_types == 0).astype(mx.float32).reshape(-1, 1)

        h_final = is_high * h_high + is_low * h_low + is_neutral * (h_high + h_low) / 2

        # Output heads
        # F0 multiplier: range [0.85, 1.25] via tanh
        f0_mult_raw = self.f0_mult_head(h_final)
        f0_mult = 1.0 + 0.2 * mx.tanh(f0_mult_raw)

        # F0 contour: [0, 1] range
        f0_contour = mx.sigmoid(self.f0_contour_head(h_final))

        # Duration multiplier: range [0.8, 1.25] via sigmoid
        duration_raw = self.duration_head(h_final)
        duration_mult = 0.8 + 0.45 * mx.sigmoid(duration_raw)

        # Energy multiplier: range [0.7, 1.6] via sigmoid
        energy_raw = self.energy_head(h_final)
        energy_mult = 0.7 + 0.9 * mx.sigmoid(energy_raw)

        return f0_mult, f0_contour, duration_mult, energy_mult


def load_prosody_embeddings(path: str) -> mx.array:
    """Load prosody embedding table."""
    weights = mx.load(str(path))
    for key in weights:
        if 'embedding' in key.lower():
            logger.info(f"Loaded embedding from key: {key}")
            return weights[key]
    raise ValueError(f"No embedding found in {path}")


PROSODY_TYPES = {
    'neutral': 0, 'angry': 40, 'sad': 41, 'excited': 42, 'happy': 42,
    'calm': 0,  # CALM -> NEUTRAL (combined!)
    'frustrated': 48, 'disgust': 48, 'nervous': 49,
    'fear': 49, 'fearful': 49, 'surprised': 50, 'surprise': 50,
}


def get_prosody_id(emotion) -> int:
    if isinstance(emotion, int):
        # Map CALM (45) to NEUTRAL (0)
        if emotion == 45:
            return 0
        return emotion
    return PROSODY_TYPES.get(str(emotion).lower().strip(), 0)


def load_training_data(path: str) -> List[Dict]:
    """Load training data with contours."""
    with open(path, 'r') as f:
        data = json.load(f)
    valid = [s for s in data if s.get('f0_contour') and len(s['f0_contour']) == 50]
    logger.info(f"Loaded {len(valid)}/{len(data)} valid samples from {path}")
    return valid


def train_epoch(
    model: UnifiedProsodyPredictor,
    embedding_table: mx.array,
    data: List[Dict],
    optimizer: optim.Optimizer,
    batch_size: int = 64,
) -> Dict[str, float]:
    """Train for one epoch."""
    random.shuffle(data)
    total_loss = 0
    num_batches = 0

    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        if len(batch) < batch_size // 2:
            continue

        # Prepare batch
        prosody_ids = []
        target_f0_mults = []
        target_dur_mults = []
        target_energy_mults = []
        target_contours = []

        for sample in batch:
            emotion = sample.get('emotion', sample.get('prosody_type', 'neutral'))
            pid = get_prosody_id(emotion)
            prosody_ids.append(pid)

            targets = TARGETS.get(pid, (1.0, 1.0, 1.0))
            target_f0_mults.append(targets[0])
            target_dur_mults.append(targets[1])
            target_energy_mults.append(targets[2])
            target_contours.append(sample['f0_contour'])

        prosody_ids_arr = mx.array(prosody_ids)
        target_f0 = mx.array(target_f0_mults).reshape(-1, 1)
        target_dur = mx.array(target_dur_mults).reshape(-1, 1)
        target_energy = mx.array(target_energy_mults).reshape(-1, 1)
        target_contour = mx.array(target_contours)

        prosody_emb = embedding_table[prosody_ids_arr]

        def loss_fn(model):
            f0_mult, f0_contour, dur_mult, energy_mult = model(prosody_emb, prosody_ids_arr)

            # Per-emotion weighted loss
            # F0 multiplier loss (high weight)
            f0_loss = mx.mean((f0_mult - target_f0) ** 2) * 50

            # Duration loss (high weight)
            dur_loss = mx.mean((dur_mult - target_dur) ** 2) * 50

            # Energy loss (high weight)
            energy_loss = mx.mean((energy_mult - target_energy) ** 2) * 50

            # Contour loss (shape matching)
            contour_loss = mx.mean((f0_contour - target_contour) ** 2) * 10

            return f0_loss + dur_loss + energy_loss + contour_loss

        loss, grads = mx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        total_loss += float(loss)
        num_batches += 1

    return {'total_loss': total_loss / max(num_batches, 1)}


def evaluate(
    model: UnifiedProsodyPredictor,
    embedding_table: mx.array,
) -> Dict[str, float]:
    """Evaluate model on target emotions (CALM combined with NEUTRAL)."""
    results = {}

    # CALM removed - combined with NEUTRAL
    for name, pid in [('NEUTRAL', 0), ('ANGRY', 40), ('SAD', 41), ('EXCITED', 42)]:
        prosody_id = mx.array([pid])
        prosody_emb = embedding_table[prosody_id]

        f0_mult, _, dur_mult, energy_mult = model(prosody_emb, prosody_id)

        targets = TARGETS[pid]
        results[f'{name}_f0'] = float(f0_mult[0, 0])
        results[f'{name}_dur'] = float(dur_mult[0, 0])
        results[f'{name}_energy'] = float(energy_mult[0, 0])
        results[f'{name}_f0_target'] = targets[0]
        results[f'{name}_dur_target'] = targets[1]
        results[f'{name}_energy_target'] = targets[2]

    return results


def main():
    parser = argparse.ArgumentParser(description='Train unified prosody model')
    parser.add_argument('--train', default='data/prosody/contours_train.json')
    parser.add_argument('--prosody-embeddings', default='models/prosody_embeddings_orthogonal/final.safetensors')
    parser.add_argument('--output-dir', default='models/prosody_unified_v1')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--hidden-dim', type=int, default=512)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading training data...")
    train_data = load_training_data(args.train)

    # Load embeddings
    logger.info("Loading prosody embeddings...")
    embedding_table = load_prosody_embeddings(args.prosody_embeddings)
    prosody_dim = embedding_table.shape[1]

    # Create model
    logger.info("Creating unified prosody model...")
    model = UnifiedProsodyPredictor(
        prosody_dim=prosody_dim,
        hidden_dim=args.hidden_dim,
        contour_len=50,
        num_blocks=4,
    )

    flat_params = tree_flatten(model.parameters())
    num_params = sum(p.size for _, p in flat_params)
    logger.info(f"Model parameters: {num_params:,}")

    optimizer = optim.Adam(learning_rate=args.lr)

    logger.info(f"\nStarting training for {args.epochs} epochs...")
    logger.info(f"Train samples: {len(train_data)}")
    logger.info("-" * 70)

    best_score = float('inf')

    for epoch in range(1, args.epochs + 1):
        # LR schedule
        if epoch <= 5:
            lr = args.lr * epoch / 5
        else:
            lr = args.lr * (0.95 ** ((epoch - 5) // 10))
        optimizer.learning_rate = lr

        metrics = train_epoch(model, embedding_table, train_data, optimizer, args.batch_size)

        if epoch % 10 == 0 or epoch == 1:
            eval_results = evaluate(model, embedding_table)

            logger.info(f"Epoch {epoch:3d}/{args.epochs} | Loss: {metrics['total_loss']:.4f} | LR: {lr:.2e}")
            logger.info(f"  {'Emotion':<10} {'F0':>8} {'F0 tgt':>8} {'Dur':>8} {'Dur tgt':>8} {'Energy':>8} {'E tgt':>8}")

            score = 0
            for name in ['NEUTRAL', 'ANGRY', 'SAD', 'EXCITED']:  # CALM combined with NEUTRAL
                f0 = eval_results[f'{name}_f0']
                f0_t = eval_results[f'{name}_f0_target']
                dur = eval_results[f'{name}_dur']
                dur_t = eval_results[f'{name}_dur_target']
                energy = eval_results[f'{name}_energy']
                energy_t = eval_results[f'{name}_energy_target']

                score += abs(f0 - f0_t) + abs(dur - dur_t) + abs(energy - energy_t)

                logger.info(f"  {name:<10} {f0:>8.3f} {f0_t:>8.2f} {dur:>8.3f} {dur_t:>8.2f} {energy:>8.3f} {energy_t:>8.2f}")

            if score < best_score:
                best_score = score
                weights = {k: v for k, v in tree_flatten(model.parameters())}
                mx.savez(str(output_dir / 'best_model.npz'), **weights)
                logger.info(f"  -> New best! Score: {score:.4f}")

    # Save final
    weights = {k: v for k, v in tree_flatten(model.parameters())}
    mx.savez(str(output_dir / 'final_model.npz'), **weights)

    # Save config
    config = {
        'version': 'unified_v1',
        'prosody_dim': prosody_dim,
        'hidden_dim': args.hidden_dim,
        'contour_len': 50,
        'num_blocks': 4,
        'epochs': args.epochs,
        'best_score': float(best_score),
        'features': ['f0_mult', 'f0_contour', 'duration_mult', 'energy_mult'],
        'targets': {str(k): v for k, v in TARGETS.items()},
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"\nTraining complete! Best score: {best_score:.4f}")
    logger.info(f"Model saved to: {output_dir}")


if __name__ == '__main__':
    main()
