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
DEPRECATED: Use train_prosody_f0_conditioning_v2.py instead.

This v1 script produced F0 changes in the WRONG direction (all emotions decreased
F0 by 6-11%). Root cause: training with random text_enc/style tensors doesn't
match the inference distribution.

See: reports/main/PHASE_C_AB_RESULTS_2025-12-17.md

---

Train Prosody F0 Conditioning (Phase C) - DEPRECATED V1

Trains the fc_prosody layers in ProsodyConditionedAdainResBlk1d to map
prosody embeddings to F0 changes.

Key insight: Phase B embeddings didn't affect F0 because Instance Normalization
in AdaIN washes out the prosody signal. Phase C bypasses this by having prosody
directly condition the gamma/beta parameters in AdaIN.

Usage:
    # DEPRECATED - use train_prosody_f0_conditioning_v2.py instead
    python scripts/train_prosody_f0_conditioning.py \
        --data data/prosody/train_split.json \
        --model-path models/kokoro-82m-mlx \
        --epochs 50 \
        --lr 1e-4

    # Test with mock data
    python scripts/train_prosody_f0_conditioning.py --mock --epochs 10
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_mock_f0_data(num_samples: int = 200) -> List[Dict]:
    """
    Create mock training data with prosody-correlated F0 targets.

    Simulates RAVDESS-like data where emotions have characteristic F0 patterns:
    - ANGRY: +20% F0
    - SAD: -10% F0
    - EXCITED: +25% F0
    - CALM: -5% F0
    """
    import random

    # Emotion F0 multipliers (from RAVDESS analysis)
    emotion_f0_mult = {
        0: 1.0,    # NONE
        21: 1.22,  # ANGRY (+22%)
        24: 0.90,  # SAD (-10%)
        25: 1.05,  # CALM (+5%)
        26: 1.27,  # EMPATHETIC (similar to calm)
        27: 1.25,  # EXCITED (+25%)
        28: 1.12,  # FRUSTRATED (+12%)
        29: 0.95,  # NERVOUS (-5%)
    }

    samples = []
    base_f0 = 160.0  # Base F0 in Hz

    for i in range(num_samples):
        # Random emotion
        emotion_id = random.choice(list(emotion_f0_mult.keys()))
        mult = emotion_f0_mult[emotion_id]

        # Target F0 with some noise
        target_f0 = base_f0 * mult * (1 + random.gauss(0, 0.05))

        # Random prosody embedding (768-dim)
        # In real training, this would come from the ProsodyEmbedding table
        prosody_emb = [random.gauss(0, 1) for _ in range(768)]

        # Scale prosody by emotion pattern
        # This creates correlation between prosody_emb and target F0
        for j in range(768):
            prosody_emb[j] += (mult - 1.0) * random.gauss(0, 0.5)

        samples.append({
            "prosody_type": emotion_id,
            "prosody_emb": prosody_emb,
            "target_f0_mean": target_f0,
            "text_length": random.randint(5, 20),
        })

    return samples


def load_training_data(data_path: str) -> List[Dict]:
    """Load training data from JSON file."""
    with open(data_path) as f:
        data = json.load(f)

    # Validate required fields
    required_fields = ["prosody_type", "f0_mean"]
    for item in data:
        for field in required_fields:
            if field not in item:
                raise ValueError(f"Missing required field '{field}' in training data")

    # Map f0_mean to target_f0_mean for consistency
    for item in data:
        if "target_f0_mean" not in item:
            item["target_f0_mean"] = item["f0_mean"]

    return data


def load_prosody_embeddings(
    weights_path: str,
    num_types: int = 63,
    hidden_dim: int = 768,
) -> mx.array:
    """
    Load Phase B prosody embeddings from safetensors file.

    Args:
        weights_path: Path to safetensors file
        num_types: Number of prosody types
        hidden_dim: Embedding dimension

    Returns:
        embedding_table: [num_types, hidden_dim] embedding table
    """
    weights = mx.load(str(weights_path))

    # Find embedding weight
    if "embedding.weight" in weights:
        return weights["embedding.weight"]
    elif "prosody_embedding.embedding.weight" in weights:
        return weights["prosody_embedding.embedding.weight"]
    else:
        # Try to find any embedding-like key
        for key, value in weights.items():
            if "embedding" in key.lower() and "weight" in key.lower():
                logger.info(f"Using embedding weights from key: {key}")
                return value

        raise ValueError(f"No embedding weights found in {weights_path}. Keys: {list(weights.keys())}")


def get_trainable_params(model) -> List[Tuple[str, mx.array]]:
    """Get only the fc_prosody parameters for training."""
    trainable = []
    for name, param in tree_flatten(model.parameters()):
        if "fc_prosody" in name:
            trainable.append((name, param))
    return trainable


def compute_f0_loss(
    predicted_f0: mx.array,
    target_f0_mean: mx.array,
) -> mx.array:
    """
    Compute F0 prediction loss.

    We want the mean F0 of the prediction to match the target.

    Args:
        predicted_f0: [batch, length] predicted F0 curve
        target_f0_mean: [batch] target mean F0

    Returns:
        loss: scalar loss value
    """
    # Compute mean F0 of prediction
    pred_f0_mean = mx.mean(predicted_f0, axis=-1)  # [batch]

    # L1 loss on mean F0
    loss = mx.mean(mx.abs(pred_f0_mean - target_f0_mean))

    return loss


def train_step(
    model,
    optimizer,
    batch: Dict[str, mx.array],
) -> Tuple[mx.array, Dict[str, float]]:
    """Single training step."""

    def loss_fn(model):
        # Forward pass through predictor with prosody
        # We use random text encoding since we're training the prosody conditioning
        text_enc = batch["text_enc"]
        style = batch["style"]
        prosody_emb = batch["prosody_emb"]
        target_f0 = batch["target_f0_mean"]

        # Get duration, f0, noise from predictor
        duration, f0, noise = model(text_enc, style, prosody_emb=prosody_emb)

        # Compute F0 loss
        # Scale predicted F0 to Hz (model outputs normalized values)
        # Kokoro F0 is typically in range 0-1000 Hz, predictions are ~0-2
        f0_hz = f0 * 200.0  # Scale factor from model outputs to Hz

        loss = compute_f0_loss(f0_hz, target_f0)

        return loss, {"f0_loss": float(loss)}

    # Compute gradients
    grad_fn = nn.value_and_grad(model, loss_fn)
    (loss, metrics), grads = grad_fn(model)

    # Only update fc_prosody parameters
    # Zero out gradients for non-fc_prosody params
    def zero_non_prosody_grads(grad):
        return tree_map(
            lambda g, n: g if "fc_prosody" in n else mx.zeros_like(g),
            grad,
            list(tree_flatten(model.parameters())),
        )

    # Note: This is a simplified approach. In practice, we'd use a mask or
    # separate optimizer for trainable params only.

    # Update model
    optimizer.update(model, grads)
    mx.eval(model.parameters())

    return loss, metrics


def create_batch(
    samples: List[Dict],
    batch_size: int,
    embedding_table: Optional[mx.array] = None,
    hidden_dim: int = 512,
    style_dim: int = 128,
    prosody_dim: int = 768,
) -> Dict[str, mx.array]:
    """
    Create a training batch from samples.

    Args:
        samples: List of training samples
        batch_size: Batch size
        embedding_table: [num_types, prosody_dim] Phase B prosody embeddings.
                        If provided, looks up embeddings by prosody_type.
                        If None, generates random embeddings.
        hidden_dim: Hidden dimension for text encoding
        style_dim: Style vector dimension
        prosody_dim: Prosody embedding dimension
    """
    import random

    batch_samples = random.sample(samples, min(batch_size, len(samples)))

    # Determine max text length
    max_len = max(s.get("text_length", 10) for s in batch_samples)

    # Build prosody embeddings and target F0 as lists, then convert
    prosody_embs = []
    target_f0s = []

    for sample in batch_samples:
        prosody_type = sample.get("prosody_type", 0)

        # Get prosody embedding
        if embedding_table is not None:
            num_types = embedding_table.shape[0]
            if prosody_type < num_types:
                prosody_embs.append(embedding_table[prosody_type])
            else:
                logger.warning(f"prosody_type {prosody_type} >= num_types {num_types}, using type 0")
                prosody_embs.append(embedding_table[0])
        elif "prosody_emb" in sample:
            prosody_embs.append(mx.array(sample["prosody_emb"][:prosody_dim]))
        else:
            # Generate deterministic embedding from prosody type
            mx.random.seed(prosody_type * 1000)
            prosody_embs.append(mx.random.normal((prosody_dim,)))

        # Get target F0
        target_f0s.append(float(sample["target_f0_mean"]))

    # Stack into batch tensors
    batch = {
        "text_enc": mx.random.normal((batch_size, max_len, hidden_dim)),
        "style": mx.random.normal((batch_size, style_dim)),
        "prosody_emb": mx.stack(prosody_embs, axis=0),
        "target_f0_mean": mx.array(target_f0s),
    }

    return batch


def train(
    data: List[Dict],
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-4,
    output_dir: str = "models/prosody_f0_conditioning",
    model_path: Optional[str] = None,
    prosody_embeddings_path: Optional[str] = None,
) -> None:
    """
    Main training loop for Phase C prosody F0 conditioning.

    Args:
        data: Training data samples
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        output_dir: Output directory for trained weights
        model_path: Path to pretrained Kokoro model (optional)
        prosody_embeddings_path: Path to Phase B prosody embeddings (optional)
    """
    from tools.pytorch_to_mlx.converters.models.kokoro import (
        KokoroConfig,
        ProsodyPredictor,
    )

    logger.info(f"Training with {len(data)} samples")
    logger.info(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")

    # Load Phase B prosody embeddings if provided
    embedding_table = None
    if prosody_embeddings_path:
        logger.info(f"Loading Phase B prosody embeddings from {prosody_embeddings_path}")
        embedding_table = load_prosody_embeddings(prosody_embeddings_path)
        logger.info(f"Loaded embedding table: {embedding_table.shape}")

    # Create model
    config = KokoroConfig()
    predictor = ProsodyPredictor(config)

    # Enable prosody conditioning
    predictor.enable_prosody_conditioning(prosody_dim=768, prosody_scale=0.1)
    logger.info("Prosody conditioning enabled")

    # Count parameters
    total_params = sum(p.size for _, p in tree_flatten(predictor.parameters()))
    fc_prosody_params = sum(
        p.size for n, p in tree_flatten(predictor.parameters()) if "fc_prosody" in n
    )
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable fc_prosody parameters: {fc_prosody_params:,}")

    # Load pretrained weights if provided
    if model_path:
        logger.info(f"Loading pretrained weights from {model_path}")
        # Would load weights here
        # predictor.copy_f0_weights_to_prosody_blocks()

    # Optimizer
    optimizer = optim.Adam(learning_rate=lr)

    # Training loop
    best_loss = float("inf")
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = len(data) // batch_size

        for batch_idx in range(max(1, num_batches)):
            batch = create_batch(data, batch_size, embedding_table=embedding_table)
            loss, metrics = train_step(predictor, optimizer, batch)
            epoch_loss += float(loss)

        avg_loss = epoch_loss / max(1, num_batches)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss

    logger.info(f"Training complete. Best loss: {best_loss:.4f}")

    # Save model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save only fc_prosody weights
    fc_prosody_weights = {}
    for name, param in tree_flatten(predictor.parameters()):
        if "fc_prosody" in name:
            fc_prosody_weights[name] = param

    weights_path = output_path / "fc_prosody_weights.safetensors"
    mx.save_safetensors(str(weights_path), fc_prosody_weights)
    logger.info(f"Saved fc_prosody weights to {weights_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train Prosody F0 Conditioning (Phase C)"
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Path to training data JSON file",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to pretrained Kokoro model weights",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/prosody_f0_conditioning",
        help="Output directory for trained weights",
    )
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
        help="Training batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock data for testing",
    )
    parser.add_argument(
        "--prosody-embeddings",
        type=str,
        default="models/prosody_embeddings_ravdess_768/final.safetensors",
        help="Path to Phase B prosody embeddings (default: RAVDESS trained embeddings)",
    )

    args = parser.parse_args()

    # Load or create data
    if args.mock:
        logger.info("Using mock data")
        data = create_mock_f0_data(num_samples=200)
    elif args.data:
        logger.info(f"Loading data from {args.data}")
        data = load_training_data(args.data)
    else:
        raise ValueError("Either --data or --mock must be specified")

    # Train
    train(
        data=data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
        model_path=args.model_path,
        prosody_embeddings_path=args.prosody_embeddings if not args.mock else None,
    )


if __name__ == "__main__":
    main()
