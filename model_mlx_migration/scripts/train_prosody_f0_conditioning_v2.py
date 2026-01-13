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
Train Prosody F0 Conditioning (Phase C) - Version 2

Fixed training methodology that uses real text through frozen Kokoro layers,
addressing the distributional mismatch issue from v1.

Key changes from v1:
1. Uses real text inputs (not random tensors)
2. Runs text through frozen Kokoro pipeline to get real text_enc
3. Trains fc_prosody to produce RELATIVE F0 changes (multipliers, not absolute Hz)
4. Target is baseline_F0 * emotion_multiplier, not absolute RAVDESS F0

Usage:
    # Train with synthetic sentences
    python scripts/train_prosody_f0_conditioning_v2.py --synthetic --epochs 50

    # Train with custom text file
    python scripts/train_prosody_f0_conditioning_v2.py --text-file data/prosody/sentences.txt --epochs 50
"""

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Emotion F0 multipliers based on RAVDESS analysis
# These are the target RELATIVE changes we want to achieve
EMOTION_F0_MULTIPLIERS = {
    0: 1.00,   # NEUTRAL
    40: 1.22,  # EMOTION_ANGRY (+22%)
    41: 1.12,  # EMOTION_SAD (+12%)
    42: 1.21,  # EMOTION_EXCITED (+21%)
    45: 1.03,  # EMOTION_CALM (+3%)
    48: 1.08,  # EMOTION_FRUSTRATED (+8%)
    49: 1.27,  # EMOTION_NERVOUS (+27%)
    50: 1.21,  # EMOTION_SURPRISED (+21%)
}

# Sample sentences for training
SAMPLE_SENTENCES = [
    "Hello, how are you today?",
    "I am very happy to see you.",
    "This is absolutely terrible news.",
    "Please stay calm, everything will be fine.",
    "I cannot believe this is happening right now!",
    "The weather is beautiful outside.",
    "Would you like some coffee?",
    "Let me explain what happened.",
    "I think we should go now.",
    "Thank you for your help today.",
    "That sounds like a great idea.",
    "I'm not sure what to do.",
    "Can you please repeat that?",
    "We need to hurry up.",
    "This is very important.",
    "I don't understand what you mean.",
    "Everything is going to be okay.",
    "Watch out for that car!",
    "I'm so proud of you.",
    "This doesn't make any sense.",
]


def load_kokoro_model():
    """Load Kokoro model and converter."""
    from tools.pytorch_to_mlx.converters import KokoroConverter

    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()
    return model, converter


def load_prosody_embeddings(path: str) -> mx.array:
    """Load Phase B prosody embedding table."""
    weights = mx.load(str(path))

    if "embedding.weight" in weights:
        return weights["embedding.weight"]
    elif "prosody_embedding.embedding.weight" in weights:
        return weights["prosody_embedding.embedding.weight"]
    else:
        for key, value in weights.items():
            if "embedding" in key.lower() and "weight" in key.lower():
                return value
        raise ValueError(f"No embedding weights found in {path}")


def text_to_tokens(text: str) -> Tuple[mx.array, int]:
    """Convert text to Kokoro tokens."""
    from tools.pytorch_to_mlx.converters.models.kokoro_phonemizer import phonemize_text

    phonemes, token_ids = phonemize_text(text)
    return mx.array([token_ids]), len(phonemes)


def get_baseline_f0(
    model,
    converter,
    text: str,
) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
    """
    Run text through frozen Kokoro to get intermediate representations.

    Returns:
        x_shared: [batch, T_audio, 512] - shared BiLSTM output (input to F0 blocks)
        speaker: [batch, 128] - speaker style vector
        baseline_f0: [batch, T_audio*2] - F0 prediction without prosody
        indices: [batch, T_audio] - duration alignment indices
    """
    input_ids, phoneme_length = text_to_tokens(text)
    voice = converter.load_voice("af_heart", phoneme_length=phoneme_length)
    mx.eval(voice)

    batch_size, seq_length = input_ids.shape

    # Split voice embedding
    if voice.shape[-1] == 256:
        _style = voice[:, :128]  # Style embedding (not used in text encoder)
        speaker = voice[:, 128:]
    else:
        _style = voice
        speaker = voice

    # Run through frozen BERT
    bert_out = model.bert(input_ids, None)
    bert_enc = model.bert_encoder(bert_out)

    # Run through text encoder
    duration_feats = model.predictor.text_encoder(bert_enc, speaker)

    # Duration prediction
    dur_enc = model.predictor.lstm(duration_feats)
    duration_logits = model.predictor.duration_proj(dur_enc)

    # Compute alignment
    indices, total_frames, _ = model._compute_alignment(duration_logits, speed=1.0)

    # Expand features
    en_expanded_640 = model._expand_features(duration_feats, indices, total_frames)

    # Run shared BiLSTM
    x_shared = model.predictor.shared(en_expanded_640)

    # Get baseline F0 (without prosody)
    x = x_shared
    x = model.predictor.F0_0(x, speaker)
    x = model.predictor.F0_1(x, speaker)
    x = model.predictor.F0_2(x, speaker)
    baseline_f0 = model.predictor.F0_proj(x).squeeze(-1)

    return x_shared, speaker, baseline_f0, indices


def predict_f0_with_prosody(
    predictor,
    x_shared: mx.array,
    speaker: mx.array,
    prosody_emb: mx.array,
) -> mx.array:
    """
    Run F0 prediction with prosody-conditioned blocks.

    Args:
        predictor: ProsodyPredictor with prosody conditioning enabled
        x_shared: [batch, T_audio, 512] - shared BiLSTM output
        speaker: [batch, 128] - speaker style vector
        prosody_emb: [batch, 768] - prosody embedding vector

    Returns:
        f0: [batch, T_audio*2] - prosody-conditioned F0 prediction
    """
    x = x_shared
    x = predictor.F0_0_prosody(x, speaker, prosody_emb)
    x = predictor.F0_1_prosody(x, speaker, prosody_emb)
    x = predictor.F0_2_prosody(x, speaker, prosody_emb)
    f0 = predictor.F0_proj(x).squeeze(-1)
    return f0


def compute_f0_loss(
    prosody_f0: mx.array,
    baseline_f0: mx.array,
    target_multiplier: float,
) -> Tuple[mx.array, Dict[str, float]]:
    """
    Compute loss for F0 prediction.

    We want: prosody_f0 ≈ baseline_f0 * target_multiplier

    Args:
        prosody_f0: [batch, T_audio*2] - F0 with prosody conditioning
        baseline_f0: [batch, T_audio*2] - baseline F0 (frozen)
        target_multiplier: Target F0 multiplier (e.g., 1.22 for ANGRY)

    Returns:
        loss: Scalar loss value
        metrics: Dictionary of metrics
    """
    # Target F0 is baseline scaled by emotion multiplier
    target_f0 = baseline_f0 * target_multiplier

    # L2 loss on F0 values
    loss = mx.mean((prosody_f0 - target_f0) ** 2)

    # Compute actual multiplier achieved
    # Avoid division by zero with small epsilon
    eps = 1e-6
    actual_mult = mx.mean(prosody_f0) / (mx.mean(baseline_f0) + eps)

    metrics = {
        "loss": float(loss),
        "target_mult": target_multiplier,
        "actual_mult": float(actual_mult),
        "baseline_f0_mean": float(mx.mean(baseline_f0)),
        "prosody_f0_mean": float(mx.mean(prosody_f0)),
    }

    return loss, metrics


def train_epoch(
    model,
    converter,
    predictor,
    optimizer,
    embedding_table: mx.array,
    sentences: List[str],
    emotion_types: List[int],
) -> Dict[str, float]:
    """
    Train one epoch.

    Args:
        model: Frozen Kokoro model
        converter: KokoroConverter
        predictor: ProsodyPredictor with prosody conditioning (trainable)
        optimizer: MLX optimizer
        embedding_table: [num_types, 768] prosody embedding table
        sentences: List of training sentences
        emotion_types: List of emotion type IDs to train on

    Returns:
        epoch_metrics: Average metrics for the epoch
    """
    total_loss = 0.0
    total_samples = 0
    mult_errors = []

    # Shuffle sentence-emotion pairs
    pairs = [(s, e) for s in sentences for e in emotion_types if e != 0]  # Exclude NEUTRAL
    random.shuffle(pairs)

    for sentence, emotion_type in pairs:
        try:
            # Get baseline F0 from frozen model
            x_shared, speaker, baseline_f0, _ = get_baseline_f0(model, converter, sentence)
            mx.eval(x_shared, speaker, baseline_f0)

            # Get prosody embedding for this emotion
            prosody_emb = embedding_table[emotion_type:emotion_type+1]  # [1, 768]

            # Get target multiplier
            target_mult = EMOTION_F0_MULTIPLIERS.get(emotion_type, 1.0)

            # Define loss function for this sample
            def loss_fn(pred):
                # Run F0 prediction with prosody
                prosody_f0 = predict_f0_with_prosody(pred, x_shared, speaker, prosody_emb)
                loss, metrics = compute_f0_loss(prosody_f0, baseline_f0, target_mult)
                return loss, metrics

            # Compute loss and gradients
            (loss, metrics), grads = nn.value_and_grad(predictor, loss_fn)(predictor)

            # Update only fc_prosody parameters
            optimizer.update(predictor, grads)
            mx.eval(predictor.parameters())

            total_loss += metrics["loss"]
            total_samples += 1
            mult_errors.append(abs(metrics["actual_mult"] - metrics["target_mult"]))

        except Exception as e:
            logger.warning(f"Error processing '{sentence[:30]}...': {e}")
            continue

    if total_samples == 0:
        return {"loss": 0.0, "mult_error": 0.0}

    return {
        "loss": total_loss / total_samples,
        "mult_error": sum(mult_errors) / len(mult_errors) if mult_errors else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train Prosody F0 Conditioning (Phase C) - Version 2"
    )
    parser.add_argument(
        "--prosody-embeddings",
        type=str,
        default="models/prosody_embeddings_ravdess_768/final.safetensors",
        help="Path to Phase B prosody embeddings",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/prosody_f0_conditioning_v2",
        help="Output directory for trained weights",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic sentences for training",
    )
    parser.add_argument(
        "--text-file",
        type=str,
        help="Path to file with training sentences (one per line)",
    )

    args = parser.parse_args()

    # Load sentences
    if args.text_file:
        with open(args.text_file) as f:
            sentences = [
                line.strip() for line in f
                if line.strip() and not line.strip().startswith("#")
            ]
        logger.info(f"Loaded {len(sentences)} sentences from {args.text_file}")
    elif args.synthetic:
        sentences = SAMPLE_SENTENCES
        logger.info(f"Using {len(sentences)} synthetic sentences")
    else:
        raise ValueError("Either --synthetic or --text-file must be specified")

    # Emotion types to train on (excluding NEUTRAL)
    emotion_types = list(EMOTION_F0_MULTIPLIERS.keys())
    logger.info(f"Training on {len(emotion_types)} emotion types")

    # Load model
    logger.info("Loading Kokoro model...")
    model, converter = load_kokoro_model()

    # Load prosody embeddings
    logger.info(f"Loading prosody embeddings from {args.prosody_embeddings}...")
    embedding_table = load_prosody_embeddings(args.prosody_embeddings)
    logger.info(f"Embedding table shape: {embedding_table.shape}")

    # Enable Phase C prosody conditioning
    logger.info("Enabling Phase C prosody conditioning...")
    model.predictor.enable_prosody_conditioning(prosody_dim=768, prosody_scale=0.1)

    # Copy original F0 weights to prosody blocks
    logger.info("Copying F0 weights to prosody-conditioned blocks...")
    model.predictor.copy_f0_weights_to_prosody_blocks()

    # Count parameters
    fc_prosody_params = sum(
        p.size for n, p in tree_flatten(model.predictor.parameters()) if "fc_prosody" in n
    )
    logger.info(f"Trainable fc_prosody parameters: {fc_prosody_params:,}")

    # Optimizer
    optimizer = optim.Adam(learning_rate=args.lr)

    # Training loop
    best_loss = float("inf")
    best_mult_error = float("inf")

    logger.info(f"\nStarting training for {args.epochs} epochs...")
    logger.info(f"Training pairs per epoch: {len(sentences) * (len(emotion_types) - 1)}")

    for epoch in range(args.epochs):
        metrics = train_epoch(
            model,
            converter,
            model.predictor,
            optimizer,
            embedding_table,
            sentences,
            emotion_types,
        )

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch + 1}/{args.epochs} - "
                f"Loss: {metrics['loss']:.6f}, "
                f"Mult Error: {metrics['mult_error']:.4f}"
            )

        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            best_mult_error = metrics["mult_error"]

    logger.info("\nTraining complete!")
    logger.info(f"Best loss: {best_loss:.6f}")
    logger.info(f"Best multiplier error: {best_mult_error:.4f}")

    # Save weights
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save only fc_prosody weights
    fc_prosody_weights = {}
    for name, param in tree_flatten(model.predictor.parameters()):
        if "fc_prosody" in name:
            fc_prosody_weights[name] = param

    weights_path = output_path / "fc_prosody_weights.safetensors"
    mx.save_safetensors(str(weights_path), fc_prosody_weights)
    logger.info(f"Saved fc_prosody weights to {weights_path}")


if __name__ == "__main__":
    main()
