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
Train Prosody Conditioning V6 - STRONG EMOTIONAL IMPACT

This version trains fc_prosody weights with MAXIMUM prosody influence so
emotions actually change how words are spoken, not just global F0/speed/volume.

Key differences from v5:
1. prosody_scale=1.0 during training (was 0.1) - emotions have FULL effect
2. Train on longer, more varied sentences
3. Use contrastive loss to ensure emotions sound DIFFERENT
4. Higher learning rate and more epochs for stronger differentiation

Usage:
    python scripts/train_prosody_conditioning_v6.py --epochs 100
"""

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import mlx.optimizers as optim

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Strong F0 targets - these define how different each emotion should sound
F0_TARGETS = {
    0:  1.00,   # NEUTRAL - baseline
    40: 1.08,   # ANGRY - moderate pitch increase (emphasis comes from contour)
    41: 0.92,   # SAD - noticeably lower
    42: 1.04,   # EXCITED - slightly higher (reduced to avoid distortion)
    43: 1.05,   # HAPPY - warm, slightly elevated (distinct from excited)
}

# Training sentences - diverse emotional content
TRAINING_SENTENCES = [
    "I cannot believe what you just said to me.",
    "This is absolutely wonderful news for everyone.",
    "Please leave me alone right now.",
    "Today has been such a difficult day.",
    "We finally did it after all this time!",
    "How could you do something like this?",
    "Everything is going to be okay.",
    "I am so disappointed in your behavior.",
    "This is the best thing that has ever happened.",
    "Why does everything always go wrong?",
    "Congratulations on your amazing achievement!",
    "I don't want to talk about this anymore.",
    "You have made me so proud today.",
    "Nothing ever works out the way I planned.",
    "Let's celebrate this incredible moment together!",
    "I trusted you and you let me down.",
    "What a beautiful and perfect day this is.",
    "I feel so empty and alone inside.",
    "This calls for a special celebration!",
    "How dare you speak to me that way!",
    # Technical examples from Claude Code style
    "The build completed successfully with zero errors.",
    "Critical failure detected in the authentication module.",
]


def load_models():
    """Load Kokoro model and prosody embeddings."""
    from tools.pytorch_to_mlx.converters import KokoroConverter

    logger.info("Loading Kokoro model...")
    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()

    logger.info("Loading prosody embeddings...")
    weights = mx.load("models/prosody_embeddings_orthogonal/final.safetensors")
    embedding_table = weights["embedding.weight"]

    return model, converter, embedding_table


def text_to_tokens(text: str, language: str = "en") -> Tuple[mx.array, int]:
    """Convert text to tokens."""
    from tools.pytorch_to_mlx.converters.models.kokoro_phonemizer import phonemize_text
    phonemes, token_ids = phonemize_text(text, language=language)
    return mx.array([token_ids]), len(phonemes)


def compute_f0_with_prosody(
    model,
    input_ids: mx.array,
    speaker: mx.array,
    prosody_emb: mx.array,
) -> mx.array:
    """Compute F0 using prosody-conditioned path."""
    bert_out = model.bert(input_ids, None)
    bert_enc = model.bert_encoder(bert_out)
    duration_feats = model.predictor.text_encoder(bert_enc, speaker)
    dur_enc = model.predictor.lstm(duration_feats)
    duration_logits = model.predictor.duration_proj(dur_enc)
    indices, total_frames, _ = model._compute_alignment(duration_logits, speed=1.0)
    en_expanded = model._expand_features(duration_feats, indices, total_frames)
    x_shared = model.predictor.shared(en_expanded)

    # F0 with prosody conditioning
    x = x_shared
    x = model.predictor.F0_0_prosody(x, speaker, prosody_emb)
    x = model.predictor.F0_1_prosody(x, speaker, prosody_emb)
    x = model.predictor.F0_2_prosody(x, speaker, prosody_emb)
    f0 = model.predictor.F0_proj(x).squeeze(-1)

    return f0


def compute_baseline_f0(model, input_ids: mx.array, speaker: mx.array) -> mx.array:
    """Compute baseline F0 without prosody conditioning."""
    bert_out = model.bert(input_ids, None)
    bert_enc = model.bert_encoder(bert_out)
    duration_feats = model.predictor.text_encoder(bert_enc, speaker)
    dur_enc = model.predictor.lstm(duration_feats)
    duration_logits = model.predictor.duration_proj(dur_enc)
    indices, total_frames, _ = model._compute_alignment(duration_logits, speed=1.0)
    en_expanded = model._expand_features(duration_feats, indices, total_frames)
    x_shared = model.predictor.shared(en_expanded)

    # Standard F0 path (no prosody)
    x = x_shared
    x = model.predictor.F0_0(x, speaker)
    x = model.predictor.F0_1(x, speaker)
    x = model.predictor.F0_2(x, speaker)
    f0 = model.predictor.F0_proj(x).squeeze(-1)

    return f0


def train_epoch(
    model,
    converter,
    embedding_table: mx.array,
    optimizer: optim.Optimizer,
    sentences: List[str],
    emotion_ids: List[int],
) -> dict:
    """Train for one epoch."""
    random.shuffle(sentences)

    total_loss = 0
    total_f0_loss = 0
    total_contrast_loss = 0
    num_batches = 0

    voice = converter.load_voice("af_heart", phoneme_length=100)
    mx.eval(voice)
    speaker = voice[:, 128:] if voice.shape[-1] == 256 else voice

    for sentence in sentences:
        try:
            input_ids, _ = text_to_tokens(sentence)
        except Exception:
            continue

        # Compute baseline F0
        baseline_f0 = compute_baseline_f0(model, input_ids, speaker)
        baseline_mean = mx.mean(mx.abs(baseline_f0))

        def loss_fn(model):
            total = mx.array(0.0)

            for emo_id in emotion_ids:
                prosody_emb = embedding_table[emo_id:emo_id+1]
                target_mult = F0_TARGETS.get(emo_id, 1.0)

                # Compute F0 with prosody
                f0_prosody = compute_f0_with_prosody(model, input_ids, speaker, prosody_emb)
                f0_mean = mx.mean(mx.abs(f0_prosody))

                # Target: prosody F0 should be target_mult * baseline
                target_f0 = baseline_mean * target_mult
                f0_loss = (f0_mean - target_f0) ** 2
                total = total + f0_loss * 100  # Strong weight

            # Contrastive loss: ensure different emotions produce different F0
            # Angry vs Sad should have maximum difference
            angry_emb = embedding_table[40:41]
            sad_emb = embedding_table[41:42]
            f0_angry = compute_f0_with_prosody(model, input_ids, speaker, angry_emb)
            f0_sad = compute_f0_with_prosody(model, input_ids, speaker, sad_emb)

            # Maximize difference between angry and sad
            angry_mean = mx.mean(mx.abs(f0_angry))
            sad_mean = mx.mean(mx.abs(f0_sad))
            diff = mx.abs(angry_mean - sad_mean)

            # We want diff to be at least 15% of baseline
            min_diff = baseline_mean * 0.15
            contrast_loss = mx.maximum(min_diff - diff, mx.array(0.0)) ** 2 * 200

            total = total + contrast_loss

            return total

        loss, grads = mx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        total_loss += float(loss)
        num_batches += 1

    return {
        "total_loss": total_loss / max(num_batches, 1),
        "f0_loss": total_f0_loss / max(num_batches, 1),
        "contrast_loss": total_contrast_loss / max(num_batches, 1),
    }


def evaluate(model, converter, embedding_table: mx.array) -> dict:
    """Evaluate F0 differences between emotions."""
    voice = converter.load_voice("af_heart", phoneme_length=100)
    mx.eval(voice)
    speaker = voice[:, 128:] if voice.shape[-1] == 256 else voice

    test_sentence = "I cannot believe what just happened today."
    input_ids, _ = text_to_tokens(test_sentence)

    baseline_f0 = compute_baseline_f0(model, input_ids, speaker)
    baseline_mean = float(mx.mean(mx.abs(baseline_f0)))

    results = {"baseline": baseline_mean}

    for name, emo_id in [("neutral", 0), ("angry", 40), ("sad", 41), ("excited", 42), ("happy", 43)]:
        prosody_emb = embedding_table[emo_id:emo_id+1]
        f0 = compute_f0_with_prosody(model, input_ids, speaker, prosody_emb)
        f0_mean = float(mx.mean(mx.abs(f0)))
        mult = f0_mean / baseline_mean if baseline_mean > 0 else 1.0
        results[name] = mult
        results[f"{name}_target"] = F0_TARGETS.get(emo_id, 1.0)

    return results


def main():
    parser = argparse.ArgumentParser(description="Train Prosody Conditioning V6")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output-dir", default="models/prosody_conditioning_v6")
    parser.add_argument("--prosody-scale", type=float, default=1.0, help="Prosody influence strength")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    model, converter, embedding_table = load_models()

    # Enable prosody conditioning with STRONG scale
    logger.info(f"Enabling prosody conditioning with scale={args.prosody_scale}")
    model.predictor.enable_prosody_conditioning(prosody_dim=768, prosody_scale=args.prosody_scale)
    model.predictor.copy_f0_weights_to_prosody_blocks()

    # Get fc_prosody parameters to train
    fc_prosody_params = {}
    for block_name in ["F0_0_prosody", "F0_1_prosody", "F0_2_prosody"]:
        if hasattr(model.predictor, block_name):
            block = getattr(model.predictor, block_name)
            if hasattr(block, "fc_prosody"):
                fc_prosody_params[f"{block_name}.fc_prosody.weight"] = block.fc_prosody.weight
                fc_prosody_params[f"{block_name}.fc_prosody.bias"] = block.fc_prosody.bias

    num_params = sum(p.size for p in fc_prosody_params.values())
    logger.info(f"Training {len(fc_prosody_params)} fc_prosody parameters ({num_params:,} total)")

    optimizer = optim.Adam(learning_rate=args.lr)

    emotion_ids = [0, 40, 41, 42, 43]  # neutral, angry, sad, excited, happy

    logger.info(f"\nStarting training for {args.epochs} epochs...")
    logger.info(f"Prosody scale: {args.prosody_scale}")
    logger.info(f"F0 targets: {F0_TARGETS}")
    logger.info("-" * 70)

    best_score = float('inf')

    for epoch in range(1, args.epochs + 1):
        # LR schedule
        if epoch <= 10:
            lr = args.lr * epoch / 10
        else:
            lr = args.lr * (0.95 ** ((epoch - 10) // 10))
        optimizer.learning_rate = lr

        metrics = train_epoch(model, converter, embedding_table, optimizer, TRAINING_SENTENCES, emotion_ids)

        if epoch % 10 == 0 or epoch == 1:
            eval_results = evaluate(model, converter, embedding_table)

            logger.info(f"Epoch {epoch:3d}/{args.epochs} | Loss: {metrics['total_loss']:.4f} | F0: {metrics['f0_loss']:.4f} | Contrast: {metrics['contrast_loss']:.4f}")
            logger.info(f"  Baseline F0: {eval_results['baseline']:.1f}")

            score = 0
            for name in ["neutral", "angry", "sad", "excited", "happy"]:
                mult = eval_results[name]
                target = eval_results[f"{name}_target"]
                diff = abs(mult - target)
                score += diff
                status = "OK" if diff < 0.03 else "MISS"
                logger.info(f"  {name.upper():8s}: {mult:.3f} (target: {target:.2f}) [{status}]")

            # Check angry vs sad difference
            angry_sad_diff = abs(eval_results["angry"] - eval_results["sad"])
            logger.info(f"  Angry-Sad diff: {angry_sad_diff:.3f} (want > 0.15)")

            if score < best_score:
                best_score = score
                # Save fc_prosody weights
                save_weights = {}
                for block_name in ["F0_0_prosody", "F0_1_prosody", "F0_2_prosody"]:
                    if hasattr(model.predictor, block_name):
                        block = getattr(model.predictor, block_name)
                        if hasattr(block, "fc_prosody"):
                            save_weights[f"{block_name}.fc_prosody.weight"] = block.fc_prosody.weight
                            save_weights[f"{block_name}.fc_prosody.bias"] = block.fc_prosody.bias
                mx.save_safetensors(str(output_dir / "fc_prosody_weights.safetensors"), save_weights)
                logger.info(f"  -> New best! Score: {score:.4f}")

    logger.info(f"\nTraining complete! Best score: {best_score:.4f}")
    logger.info(f"Weights saved to: {output_dir}")


if __name__ == "__main__":
    main()
