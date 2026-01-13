#!/usr/bin/env python3
"""
Evaluate RichDecoder v4 on cached encoder features.

Usage:
    # No-LoRA model:
    python -m tools.whisper_mlx.eval_rich_decoder_v4 \
        --checkpoint checkpoints/rich_decoder_v4_nolora/epoch_10.npz \
        --manifest data/v3_multitask/val_manifest.json \
        --encoder-cache data/v3_multitask/encoder_cache

    # With encoder LoRA:
    python -m tools.whisper_mlx.eval_rich_decoder_v4 \
        --checkpoint checkpoints/rich_decoder_v4_lora/best.npz \
        --manifest data/v3_multitask/val_manifest.json \
        --encoder-cache data/v3_multitask/encoder_cache \
        --use-encoder-lora --encoder-lora-rank 8 --encoder-lora-alpha 16
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .encoder_lora import EncoderBlockLoRA


class EmotionHead(nn.Module):
    """Emotion classification head."""

    def __init__(self, input_dim: int = 1280, num_classes: int = 8):
        super().__init__()
        self.pool = nn.Linear(input_dim, input_dim)
        self.classifier = nn.Linear(input_dim, num_classes)

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.mean(x, axis=1)
        x = nn.gelu(self.pool(x))
        return self.classifier(x)


class EncoderLoRAModule(nn.Module):
    """
    Encoder LoRA module for adapting cached encoder outputs.
    (Matches the one in train_rich_decoder_v4.py)
    """

    def __init__(
        self,
        n_state: int = 1280,
        rank: int = 16,
        alpha: int = 32,
        start_layer: int = 20,
        end_layer: int = 32,
    ):
        super().__init__()

        self.n_state = n_state
        self.rank = rank
        self.alpha = alpha
        self.start_layer = start_layer
        self.end_layer = end_layer

        self.adapters = []
        for i in range(end_layer - start_layer):
            adapter = EncoderBlockLoRA(
                n_state=n_state,
                rank=rank,
                alpha=alpha,
                adapt_query=True,
                adapt_key=True,
                adapt_value=True,
            )
            self.adapters.append(adapter)
            setattr(self, f'adapter_{i}', adapter)

    def __call__(self, encoder_out: mx.array) -> mx.array:
        delta = mx.zeros_like(encoder_out)

        for adapter in self.adapters:
            if adapter.q_lora is not None:
                delta = delta + adapter.q_lora(encoder_out)
            if adapter.k_lora is not None:
                delta = delta + adapter.k_lora(encoder_out)
            if adapter.v_lora is not None:
                delta = delta + adapter.v_lora(encoder_out)

        n_adapters = len(self.adapters)
        if n_adapters > 0:
            delta = delta / n_adapters

        return encoder_out + delta


class RichDecoderV4Eval(nn.Module):
    """RichDecoder v4 for evaluation."""

    def __init__(
        self,
        input_dim: int = 1280,
        num_emotions: int = 8,
        use_prosody: bool = True,
        use_encoder_lora: bool = False,
        encoder_lora_rank: int = 16,
        encoder_lora_alpha: int = 32,
        encoder_lora_start_layer: int = 20,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.use_prosody = use_prosody
        self.use_encoder_lora = use_encoder_lora
        self.prosody_dim = 4

        # Encoder LoRA
        if use_encoder_lora:
            self.encoder_lora = EncoderLoRAModule(
                n_state=input_dim,
                rank=encoder_lora_rank,
                alpha=encoder_lora_alpha,
                start_layer=encoder_lora_start_layer,
                end_layer=32,
            )
        else:
            self.encoder_lora = None

        # Prosody projection
        if use_prosody:
            self.prosody_projection = nn.Linear(input_dim + self.prosody_dim, input_dim)
        else:
            self.prosody_projection = None

        self.emotion_head = EmotionHead(input_dim, num_emotions)

    def __call__(self, encoder_out: mx.array, prosody: mx.array | None = None) -> mx.array:
        # Apply encoder LoRA if enabled
        if self.encoder_lora is not None:
            encoder_out = self.encoder_lora(encoder_out)

        if self.use_prosody and prosody is not None and self.prosody_projection is not None:
            combined = mx.concatenate([encoder_out, prosody], axis=-1)
            encoder_out = self.prosody_projection(combined)
        return self.emotion_head(encoder_out)


def get_cache_path(audio_path: str, cache_dir: Path) -> Path:
    """Get cache file path for audio."""
    cache_key = hashlib.sha256(audio_path.encode()).hexdigest()[:16]
    return cache_dir / cache_key[:2] / f"{cache_key}.npz"


def load_model(
    checkpoint_path: str,
    use_prosody: bool = True,
    use_encoder_lora: bool = False,
    encoder_lora_rank: int = 16,
    encoder_lora_alpha: int = 32,
) -> RichDecoderV4Eval:
    """Load model from checkpoint."""
    model = RichDecoderV4Eval(
        use_prosody=use_prosody,
        use_encoder_lora=use_encoder_lora,
        encoder_lora_rank=encoder_lora_rank,
        encoder_lora_alpha=encoder_lora_alpha,
    )

    weights = dict(mx.load(checkpoint_path))

    # Map flat keys to nested structure
    model_params = dict(model.parameters())

    def apply_weights(params, prefix=''):
        for k, v in params.items():
            key = f'{prefix}.{k}' if prefix else k
            if isinstance(v, dict):
                apply_weights(v, key)
            else:
                if key in weights and weights[key].shape == v.shape:
                    params[k] = weights[key]

    apply_weights(model_params)
    model.update(model_params)
    return model


def evaluate(
    model: RichDecoderV4Eval,
    manifest: list,
    cache_dir: Path,
    use_prosody: bool = True,
):
    """Evaluate model on manifest using cached features."""
    emotion_to_id = {
        'neutral': 0, 'happy': 1, 'sad': 2, 'angry': 3,
        'fear': 4, 'disgust': 5, 'surprise': 6, 'other': 7,
    }
    id_to_emotion = {v: k for k, v in emotion_to_id.items()}

    # Per-class stats
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    # Confusion matrix
    confusion = defaultdict(lambda: defaultdict(int))

    total_correct = 0
    total_samples = 0
    cached_samples = 0
    missing_samples = 0

    for sample in manifest:
        audio_path = sample.get("audio_path", sample.get("path", ""))
        emotion = sample.get("emotion", "neutral")
        emotion_id = emotion_to_id.get(emotion, 0)

        cache_path = get_cache_path(audio_path, cache_dir)
        if not cache_path.exists():
            missing_samples += 1
            continue

        # Load cached features
        try:
            data = np.load(cache_path)
            encoder_out = mx.array(data["encoder_output"].astype(np.float32))
            encoder_out = mx.expand_dims(encoder_out, axis=0)  # Add batch dim
        except Exception:
            # Corrupt or incomplete cache file
            missing_samples += 1
            continue

        prosody = None
        if use_prosody and "prosody" in data:
            prosody_np = data["prosody"].astype(np.float32)
            # Align prosody to encoder length (Whisper: 50fps, prosody: 100fps)
            encoder_len = encoder_out.shape[1]
            if prosody_np.shape[0] != encoder_len:
                # Resample prosody to match encoder length
                indices = np.linspace(0, prosody_np.shape[0] - 1, encoder_len).astype(int)
                prosody_np = prosody_np[indices]
            prosody = mx.array(prosody_np)
            prosody = mx.expand_dims(prosody, axis=0)

        # Forward pass
        logits = model(encoder_out, prosody=prosody)
        pred = int(mx.argmax(logits, axis=-1).item())

        # Update stats
        cached_samples += 1
        total_samples += 1
        class_total[emotion] += 1

        if pred == emotion_id:
            total_correct += 1
            class_correct[emotion] += 1

        # Confusion matrix
        pred_emotion = id_to_emotion.get(pred, "unknown")
        confusion[emotion][pred_emotion] += 1

    # Print results
    print(f"\n{'='*60}")
    print("Evaluation Results")
    print(f"{'='*60}")
    print(f"Total manifest samples: {len(manifest)}")
    print(f"Cached samples: {cached_samples}")
    print(f"Missing samples: {missing_samples}")
    print(f"\nOverall accuracy: {total_correct}/{total_samples} = {total_correct/max(total_samples,1)*100:.2f}%")

    print(f"\n{'='*60}")
    print("Per-class accuracy:")
    print(f"{'='*60}")
    for emotion in sorted(class_total.keys()):
        correct = class_correct[emotion]
        total = class_total[emotion]
        acc = correct / max(total, 1) * 100
        print(f"  {emotion:10s}: {correct:4d}/{total:4d} = {acc:.2f}%")

    print(f"\n{'='*60}")
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(f"{'='*60}")

    emotions = sorted(set(class_total.keys()))
    header = f"{'':12s}" + "".join(f"{e[:8]:>9s}" for e in emotions)
    print(header)

    for actual in emotions:
        row = f"{actual[:10]:12s}"
        for pred in emotions:
            count = confusion[actual][pred]
            row += f"{count:9d}"
        print(row)

    return total_correct / max(total_samples, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--encoder-cache", required=True)
    parser.add_argument("--no-prosody", action="store_true")
    parser.add_argument("--use-encoder-lora", action="store_true",
                       help="Enable encoder LoRA")
    parser.add_argument("--encoder-lora-rank", type=int, default=16)
    parser.add_argument("--encoder-lora-alpha", type=int, default=32)
    args = parser.parse_args()

    # Load manifest
    with open(args.manifest) as f:
        manifest = json.load(f)

    # Load model
    use_prosody = not args.no_prosody
    print(f"Loading model from {args.checkpoint}")
    print(f"Prosody: {'enabled' if use_prosody else 'disabled'}")
    print(f"Encoder LoRA: {'enabled' if args.use_encoder_lora else 'disabled'}")
    if args.use_encoder_lora:
        print(f"  Rank: {args.encoder_lora_rank}, Alpha: {args.encoder_lora_alpha}")
    model = load_model(
        args.checkpoint,
        use_prosody=use_prosody,
        use_encoder_lora=args.use_encoder_lora,
        encoder_lora_rank=args.encoder_lora_rank,
        encoder_lora_alpha=args.encoder_lora_alpha,
    )

    # Evaluate
    cache_dir = Path(args.encoder_cache)
    return evaluate(model, manifest, cache_dir, use_prosody=use_prosody)



if __name__ == "__main__":
    main()
