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
Prosody Embedding Training Script

Phase B of Kokoro Prosody Annotation Roadmap:
Train prosody embedding layer on prepared training data.

Architecture:
- Freeze all Kokoro weights
- Add prosody embedding table (num_prosody_types x hidden_dim) ~6K params
- Add learnable scale factor
- Inject embeddings after BERT encoder: bert_enc = bert_enc + scale * prosody_emb

Training:
- Loss: MSE on F0 prediction + MSE on duration prediction
- Optimizer: Adam
- Data: JSON samples from prepare_prosody_training_data.py

Usage:
    # Train on synthetic data
    python scripts/train_prosody_embeddings.py --data data/prosody/train.json

    # Train with validation
    python scripts/train_prosody_embeddings.py --data data/prosody/train.json --val data/prosody/val.json

    # Test mode with mock data
    python scripts/train_prosody_embeddings.py --mock --mock-samples 100

See: reports/main/PROSODY_ROADMAP.md
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Prosody Types (matching C++ prosody_types.h and prepare_prosody_training_data.py)
# =============================================================================

PROSODY_TYPES = {
    "NEUTRAL": 0,
    "EMPHASIS": 1,
    "STRONG_EMPHASIS": 2,
    "REDUCED_EMPHASIS": 3,
    "RATE_X_SLOW": 10,
    "RATE_SLOW": 11,
    "RATE_FAST": 12,
    "RATE_X_FAST": 13,
    "PITCH_X_LOW": 20,
    "PITCH_LOW": 21,
    "PITCH_HIGH": 22,
    "PITCH_X_HIGH": 23,
    "VOLUME_X_SOFT": 30,
    "VOLUME_SOFT": 31,
    "VOLUME_LOUD": 32,
    "VOLUME_X_LOUD": 33,
    "VOLUME_WHISPER": 34,
    "EMOTION_ANGRY": 40,
    "EMOTION_SAD": 41,
    "EMOTION_EXCITED": 42,
    "EMOTION_WORRIED": 43,
    "EMOTION_ALARMED": 44,
    "EMOTION_CALM": 45,
    "EMOTION_EMPATHETIC": 46,
    "EMOTION_CONFIDENT": 47,
    "EMOTION_FRUSTRATED": 48,
    "EMOTION_NERVOUS": 49,
    "EMOTION_SURPRISED": 50,
    "EMOTION_DISAPPOINTED": 51,
    "QUESTION": 60,
    "WHISPER": 61,
    "LOUD": 62,
}

NUM_PROSODY_TYPES = max(PROSODY_TYPES.values()) + 1


# =============================================================================
# Training Sample
# =============================================================================

@dataclass
class TrainingSample:
    """A single training sample."""
    text: str
    prosody_type: int
    f0_mean: float
    f0_std: float
    duration_s: float
    energy_rms: float


# =============================================================================
# Prosody Embedding Module
# =============================================================================

class ProsodyEmbedding(nn.Module):
    """
    Prosody embedding table for Phase B training.

    Adds learned embeddings to BERT output based on prosody type.
    """

    def __init__(self, num_types: int, hidden_dim: int, init_scale: float = 0.01):
        super().__init__()
        self.num_types = num_types
        self.hidden_dim = hidden_dim

        # Embedding table: [num_types, hidden_dim]
        # Initialize with small random values
        self.embedding = nn.Embedding(num_types, hidden_dim)

        # Learnable scale factor (start small to not disrupt base model)
        self.scale = mx.array([init_scale])

    def __call__(self, prosody_mask: mx.array) -> mx.array:
        """
        Get prosody embeddings for input prosody mask.

        Args:
            prosody_mask: [batch, seq_len] int array of prosody type IDs

        Returns:
            embeddings: [batch, seq_len, hidden_dim] scaled prosody embeddings
        """
        emb = self.embedding(prosody_mask)  # [batch, seq_len, hidden_dim]
        return self.scale * emb


# =============================================================================
# Dataset Loading
# =============================================================================

def load_dataset(path: Path) -> List[TrainingSample]:
    """Load training samples from JSON file."""
    with open(path) as f:
        data = json.load(f)

    samples = []
    for item in data:
        sample = TrainingSample(
            text=item.get("text", ""),
            prosody_type=item.get("prosody_type", 0),
            f0_mean=item.get("f0_mean", 0.0),
            f0_std=item.get("f0_std", 0.0),
            duration_s=item.get("duration_s", 0.0),
            energy_rms=item.get("energy_rms", 0.0),
        )
        samples.append(sample)

    return samples


def create_mock_dataset(num_samples: int = 100) -> List[TrainingSample]:
    """
    Create mock dataset for testing training pipeline.

    Generates samples with prosody-correlated features:
    - Emotions affect F0 and duration
    - Rate affects duration
    - Emphasis affects F0
    """
    samples = []

    # Base values
    base_f0 = 180.0  # Hz
    base_f0_std = 30.0
    base_duration = 1.0  # seconds
    base_energy = 0.05

    # Prosody effects (type -> (f0_mult, f0_std_mult, dur_mult, energy_mult))
    prosody_effects = {
        PROSODY_TYPES["NEUTRAL"]: (1.0, 1.0, 1.0, 1.0),
        PROSODY_TYPES["EMPHASIS"]: (1.1, 1.2, 1.2, 1.2),
        PROSODY_TYPES["STRONG_EMPHASIS"]: (1.2, 1.4, 1.4, 1.4),
        PROSODY_TYPES["RATE_SLOW"]: (1.0, 1.0, 1.4, 1.0),
        PROSODY_TYPES["RATE_FAST"]: (1.0, 1.0, 0.7, 1.0),
        PROSODY_TYPES["EMOTION_ANGRY"]: (1.15, 1.3, 0.9, 1.3),
        PROSODY_TYPES["EMOTION_SAD"]: (0.9, 0.8, 1.3, 0.8),
        PROSODY_TYPES["EMOTION_EXCITED"]: (1.2, 1.5, 0.85, 1.4),
        PROSODY_TYPES["EMOTION_WORRIED"]: (1.05, 1.2, 1.1, 1.0),
        PROSODY_TYPES["EMOTION_CALM"]: (0.95, 0.7, 1.2, 0.9),
    }

    # Sample texts
    texts = [
        "Hello, how are you today?",
        "I really need your help with this.",
        "That's absolutely wonderful news!",
        "I'm not sure what to do.",
        "Please listen carefully.",
        "Everything will be alright.",
        "This is very important.",
        "Can you help me understand?",
        "I appreciate your patience.",
        "Let me explain the situation.",
    ]

    rng = np.random.default_rng(42)

    for i in range(num_samples):
        # Select random prosody type
        prosody_type = rng.choice(list(prosody_effects.keys()))
        effects = prosody_effects[prosody_type]

        # Apply effects with noise
        f0_mult, f0_std_mult, dur_mult, energy_mult = effects
        noise = rng.normal(0, 0.05, 4)

        sample = TrainingSample(
            text=texts[i % len(texts)],
            prosody_type=prosody_type,
            f0_mean=base_f0 * (f0_mult + noise[0]),
            f0_std=base_f0_std * (f0_std_mult + noise[1]),
            duration_s=base_duration * (dur_mult + noise[2]),
            energy_rms=base_energy * (energy_mult + noise[3]),
        )
        samples.append(sample)

    return samples


# =============================================================================
# Training Loop
# =============================================================================

@dataclass
class TrainingConfig:
    """Training configuration."""
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 100
    hidden_dim: int = 512  # Kokoro hidden dimension
    init_scale: float = 0.01
    log_interval: int = 10
    val_interval: int = 50


def prepare_batch(
    samples: List[TrainingSample],
    batch_indices: List[int],
) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
    """
    Prepare a training batch.

    Returns:
        prosody_types: [batch] int array
        f0_targets: [batch] float array (normalized F0 mean)
        f0_std_targets: [batch] float array
        duration_targets: [batch] float array
    """
    batch_samples = [samples[i] for i in batch_indices]

    prosody_types = mx.array([int(s.prosody_type) for s in batch_samples], dtype=mx.int32)

    # Normalize targets to reasonable ranges
    # F0: normalize to [0, 1] range (typical range 100-300 Hz)
    f0_targets = mx.array([float((s.f0_mean - 100) / 200) for s in batch_samples], dtype=mx.float32)
    f0_std_targets = mx.array([float(s.f0_std / 100) for s in batch_samples], dtype=mx.float32)
    duration_targets = mx.array([float(s.duration_s) for s in batch_samples], dtype=mx.float32)

    return prosody_types, f0_targets, f0_std_targets, duration_targets


def compute_loss(
    prosody_emb_module: ProsodyEmbedding,
    prosody_types: mx.array,
    f0_targets: mx.array,
    f0_std_targets: mx.array,
    duration_targets: mx.array,
) -> Tuple[mx.array, Dict[str, float]]:
    """
    Compute training loss.

    For Phase B, we predict prosody features directly from the embedding.
    This serves as a proxy for the full model training.

    The embedding learns to encode prosody-relevant information that,
    when added to BERT output, will influence F0 and duration predictions.
    """
    # Get embeddings for each prosody type
    # Shape: [batch, hidden_dim] (single token per sample for now)
    prosody_types_2d = prosody_types[:, None]  # [batch, 1]
    emb = prosody_emb_module(prosody_types_2d)  # [batch, 1, hidden_dim]
    emb = emb.squeeze(1)  # [batch, hidden_dim]

    # Simple linear projections for prediction
    # In full model, these would come from Kokoro's predictors
    # Here we learn what embeddings should produce what features

    # Mean embedding per sample predicts features
    emb_mean = mx.mean(emb, axis=-1)  # [batch]
    emb_std = mx.std(emb, axis=-1)    # [batch]

    # Normalize predictions to match target ranges
    # softplus(x) = log(1 + exp(x))
    def softplus(x):
        return mx.log(1 + mx.exp(x))

    f0_pred = mx.sigmoid(emb_mean)  # [0, 1]
    f0_std_pred = softplus(emb_std) * 0.5  # positive
    dur_pred = softplus(mx.mean(emb, axis=-1)) * 2  # positive, ~1-2 range

    # MSE losses
    loss_f0 = mx.mean((f0_pred - f0_targets) ** 2)
    loss_f0_std = mx.mean((f0_std_pred - f0_std_targets) ** 2)
    loss_dur = mx.mean((dur_pred - duration_targets) ** 2)

    # Combined loss
    total_loss = loss_f0 + 0.5 * loss_f0_std + loss_dur

    metrics = {
        "loss_f0": float(loss_f0),
        "loss_f0_std": float(loss_f0_std),
        "loss_dur": float(loss_dur),
        "total": float(total_loss),
        "scale": float(prosody_emb_module.scale),
    }

    return total_loss, metrics


def train(
    config: TrainingConfig,
    train_samples: List[TrainingSample],
    val_samples: Optional[List[TrainingSample]] = None,
    output_dir: Optional[Path] = None,
) -> ProsodyEmbedding:
    """
    Train prosody embeddings.

    Args:
        config: Training configuration
        train_samples: Training data
        val_samples: Optional validation data
        output_dir: Directory to save checkpoints

    Returns:
        Trained ProsodyEmbedding module
    """
    print("Training prosody embeddings")
    print(f"  Train samples: {len(train_samples)}")
    print(f"  Val samples: {len(val_samples) if val_samples else 0}")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Num prosody types: {NUM_PROSODY_TYPES}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.epochs}")
    print()

    # Create model
    model = ProsodyEmbedding(
        num_types=NUM_PROSODY_TYPES,
        hidden_dim=config.hidden_dim,
        init_scale=config.init_scale,
    )

    # Optimizer
    optimizer = optim.Adam(learning_rate=config.learning_rate)

    # Training state
    num_samples = len(train_samples)
    num_batches = (num_samples + config.batch_size - 1) // config.batch_size

    # Value and grad function
    def loss_fn(model, prosody_types, f0_targets, f0_std_targets, dur_targets):
        loss, _ = compute_loss(model, prosody_types, f0_targets, f0_std_targets, dur_targets)
        return loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    best_val_loss = float("inf")
    rng = np.random.default_rng(42)

    for epoch in range(config.epochs):
        epoch_start = time.time()
        epoch_loss = 0.0

        # Shuffle indices
        indices = rng.permutation(num_samples)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * config.batch_size
            end_idx = min(start_idx + config.batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx].tolist()

            # Prepare batch
            prosody_types, f0_targets, f0_std_targets, dur_targets = prepare_batch(
                train_samples, batch_indices
            )

            # Forward + backward
            loss, grads = loss_and_grad(
                model, prosody_types, f0_targets, f0_std_targets, dur_targets
            )

            # Update
            optimizer.update(model, grads)
            mx.eval(model.parameters())

            epoch_loss += float(loss)

            # Log
            step = epoch * num_batches + batch_idx
            if step > 0 and step % config.log_interval == 0:
                _, metrics = compute_loss(
                    model, prosody_types, f0_targets, f0_std_targets, dur_targets
                )
                print(
                    f"  Step {step}: loss={metrics['total']:.4f} "
                    f"(f0={metrics['loss_f0']:.4f}, dur={metrics['loss_dur']:.4f}, "
                    f"scale={metrics['scale']:.4f})"
                )

        # Epoch summary
        avg_epoch_loss = epoch_loss / num_batches
        epoch_time = time.time() - epoch_start

        # Validation
        val_loss_str = ""
        if val_samples and (epoch + 1) % (config.val_interval // num_batches + 1) == 0:
            val_indices = list(range(len(val_samples)))
            val_prosody, val_f0, val_f0_std, val_dur = prepare_batch(
                val_samples, val_indices
            )
            val_loss, val_metrics = compute_loss(
                model, val_prosody, val_f0, val_f0_std, val_dur
            )
            val_loss_str = f", val_loss={float(val_loss):.4f}"

            if float(val_loss) < best_val_loss:
                best_val_loss = float(val_loss)
                if output_dir:
                    save_checkpoint(model, output_dir / "best.safetensors")

        print(
            f"Epoch {epoch + 1}/{config.epochs}: "
            f"loss={avg_epoch_loss:.4f}{val_loss_str} "
            f"({epoch_time:.1f}s)"
        )

    # Save final checkpoint
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_checkpoint(model, output_dir / "final.safetensors")
        print(f"\nSaved checkpoint to {output_dir / 'final.safetensors'}")

    return model


def save_checkpoint(model: ProsodyEmbedding, path: Path):
    """Save model checkpoint."""
    import mlx.core as mx

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    weights = {
        "embedding.weight": model.embedding.weight,
        "scale": model.scale,
    }
    mx.save_safetensors(str(path), weights)


def load_checkpoint(path: Path, hidden_dim: int) -> ProsodyEmbedding:
    """Load model checkpoint."""
    import mlx.core as mx

    model = ProsodyEmbedding(
        num_types=NUM_PROSODY_TYPES,
        hidden_dim=hidden_dim,
    )

    weights = mx.load(str(path))
    model.embedding.weight = weights["embedding.weight"]
    model.scale = weights["scale"]

    return model


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(
    model: ProsodyEmbedding,
    samples: List[TrainingSample],
) -> Dict[str, float]:
    """Evaluate model on samples."""
    indices = list(range(len(samples)))
    prosody_types, f0_targets, f0_std_targets, dur_targets = prepare_batch(
        samples, indices
    )

    loss, metrics = compute_loss(
        model, prosody_types, f0_targets, f0_std_targets, dur_targets
    )

    return metrics


def analyze_embeddings(model: ProsodyEmbedding) -> Dict[str, Any]:
    """Analyze learned embeddings."""
    emb_weight = model.embedding.weight  # [num_types, hidden_dim]

    # Compute pairwise similarities
    # Normalize embeddings
    norms = mx.linalg.norm(emb_weight, axis=1, keepdims=True)
    normalized = emb_weight / (norms + 1e-8)
    similarities = normalized @ normalized.T  # [num_types, num_types]

    # Find most similar pairs (excluding diagonal)
    sim_np = np.array(similarities)
    np.fill_diagonal(sim_np, -1)  # Exclude self-similarity

    type_names = {v: k for k, v in PROSODY_TYPES.items()}

    # Top similar pairs
    flat_indices = np.argsort(sim_np.flatten())[::-1][:10]
    top_pairs = []
    for idx in flat_indices:
        i, j = divmod(idx, sim_np.shape[1])
        if i in type_names and j in type_names:
            top_pairs.append((type_names[i], type_names[j], sim_np[i, j]))

    return {
        "scale": float(model.scale),
        "embedding_norm_mean": float(mx.mean(norms)),
        "embedding_norm_std": float(mx.std(norms)),
        "top_similar_pairs": top_pairs,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train prosody embeddings")
    parser.add_argument("--data", type=Path, help="Training data JSON file")
    parser.add_argument("--val", type=Path, help="Validation data JSON file")
    parser.add_argument("--output", type=Path, default=Path("models/prosody_embeddings"),
                        help="Output directory for checkpoints")
    parser.add_argument("--mock", action="store_true", help="Use mock data for testing")
    parser.add_argument("--mock-samples", type=int, default=100,
                        help="Number of mock samples")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=512,
                        help="Kokoro hidden dimension")
    parser.add_argument("--analyze-only", type=Path,
                        help="Load checkpoint and analyze embeddings")

    args = parser.parse_args()

    # Analyze mode
    if args.analyze_only:
        print(f"Loading checkpoint from {args.analyze_only}")
        model = load_checkpoint(args.analyze_only, args.hidden_dim)
        analysis = analyze_embeddings(model)
        print("\nEmbedding Analysis:")
        print(f"  Scale: {analysis['scale']:.4f}")
        print(f"  Embedding norm mean: {analysis['embedding_norm_mean']:.4f}")
        print(f"  Embedding norm std: {analysis['embedding_norm_std']:.4f}")
        print("\n  Top similar prosody type pairs:")
        for t1, t2, sim in analysis['top_similar_pairs']:
            print(f"    {t1} <-> {t2}: {sim:.3f}")
        return

    # Load or create data
    if args.mock:
        print(f"Creating mock dataset with {args.mock_samples} samples")
        train_samples = create_mock_dataset(args.mock_samples)
        val_samples = create_mock_dataset(args.mock_samples // 5)
    elif args.data:
        print(f"Loading training data from {args.data}")
        train_samples = load_dataset(args.data)
        val_samples = load_dataset(args.val) if args.val else None
    else:
        print("Error: Must specify --data or --mock")
        sys.exit(1)

    # Training config
    config = TrainingConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
    )

    # Train
    model = train(
        config=config,
        train_samples=train_samples,
        val_samples=val_samples,
        output_dir=args.output,
    )

    # Final evaluation
    print("\nFinal evaluation:")
    metrics = evaluate(model, train_samples)
    print(f"  Train loss: {metrics['total']:.4f}")
    if val_samples:
        val_metrics = evaluate(model, val_samples)
        print(f"  Val loss: {val_metrics['total']:.4f}")

    # Analyze embeddings
    print("\nEmbedding analysis:")
    analysis = analyze_embeddings(model)
    print(f"  Scale: {analysis['scale']:.4f}")
    print(f"  Embedding norm mean: {analysis['embedding_norm_mean']:.4f}")
    print("\n  Top similar prosody type pairs:")
    for t1, t2, sim in analysis['top_similar_pairs'][:5]:
        print(f"    {t1} <-> {t2}: {sim:.3f}")


if __name__ == "__main__":
    main()
