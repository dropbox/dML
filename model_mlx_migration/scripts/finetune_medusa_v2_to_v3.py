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
Fine-tune aiola's v2 Medusa heads for whisper-large-v3.

Track A with Fine-Tuning approach:
1. Initialize ResBlock heads from aiola/whisper-medusa-v1 weights
2. Use v3's proj_out (token_embedding.as_linear())
3. Fine-tune with low learning rate (1e-5)
4. Train on LibriSpeech train-clean-100

This combines the benefits of:
- Pre-trained Medusa architecture (aiola's trained ResBlocks)
- Correct target model (whisper-large-v3)
- Fast convergence (warm start vs random init)

Usage:
    python scripts/finetune_medusa_v2_to_v3.py \
        --aiola-weights checkpoints/aiola_medusa_v1/medusa_aiola_v1.npz \
        --max-steps 5000 \
        --learning-rate 1e-5

Previous Track Results (why fine-tuning is needed):
- Track A (direct use): 6.7% accuracy, 0.11x speedup (9x SLOWER)
- Track B (from scratch): loss ~15-18, 0.17x speedup (6x SLOWER)
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import mlx.nn as nn

from tools.whisper_mlx import WhisperMLX
from tools.whisper_mlx.audio import log_mel_spectrogram
from tools.whisper_mlx.medusa import MedusaModule
from tools.whisper_mlx.medusa_training import (
    LibriSpeechDataLoader,
    MedusaTrainer,
    MedusaTrainingConfig,
    save_medusa_weights,
)
from tools.whisper_mlx.tokenizer import get_whisper_tokenizer


class ProjOutWrapper(nn.Module):
    """Wrapper to use token_embedding.as_linear() as a callable layer.

    This allows Medusa heads to share the decoder's vocabulary projection,
    matching the aiola architecture where ResBlocks output hidden states
    that are then projected through the shared lm_head.

    Args:
        token_embedding: nn.Embedding layer to use for projection
        n_vocab: Original vocabulary size (before padding). If provided,
                output will be sliced to this size to match decoder output.
    """

    def __init__(self, token_embedding, n_vocab: int = None):
        super().__init__()
        self._token_embedding = token_embedding
        self._n_vocab = n_vocab

    def __call__(self, x):
        logits = self._token_embedding.as_linear(x)
        # Slice to original vocab size if specified (to match decoder output)
        if self._n_vocab is not None and logits.shape[-1] > self._n_vocab:
            logits = logits[..., :self._n_vocab]
        return logits


def load_aiola_weights_into_module(
    medusa: MedusaModule,
    aiola_weights_path: str,
    n_heads: int = 5,
) -> MedusaModule:
    """
    Load aiola v2 Medusa weights into a MedusaModule.

    aiola has 11 heads (indices 0-10), but we only need 5 for standard Medusa.
    The architecture is MedusaResBlock: Linear(1280->1280) + SiLU + residual.

    Args:
        medusa: MedusaModule with use_aiola=True
        aiola_weights_path: Path to aiola weights (.npz)
        n_heads: Number of heads to load (default 5, aiola has 11)

    Returns:
        MedusaModule with loaded weights
    """
    weights = dict(mx.load(aiola_weights_path))

    print(f"Loading aiola weights from {aiola_weights_path}")
    print(f"  Total keys in file: {len(weights)}")
    print(f"  Loading {n_heads} heads out of 11 available")

    loaded_count = 0
    for i in range(n_heads):
        head = medusa.heads[i]

        # Load linear layer weights
        weight_key = f"heads.{i}.linear.weight"
        bias_key = f"heads.{i}.linear.bias"

        if weight_key in weights and bias_key in weights:
            head.linear.weight = weights[weight_key]
            head.linear.bias = weights[bias_key]
            loaded_count += 1
            print(f"  Head {i}: weight shape {weights[weight_key].shape}, bias shape {weights[bias_key].shape}")
        else:
            print(f"  Head {i}: WARNING - weights not found!")

    print(f"Loaded {loaded_count}/{n_heads} heads from aiola weights")
    return medusa


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune aiola v2 Medusa heads for whisper-large-v3"
    )

    # Weights
    parser.add_argument(
        "--aiola-weights",
        type=str,
        default="checkpoints/aiola_medusa_v1/medusa_aiola_v1.npz",
        help="Path to aiola Medusa weights",
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/whisper-large-v3-mlx",
        help="Whisper model to fine-tune on",
    )
    parser.add_argument(
        "--n-heads",
        type=int,
        default=5,
        help="Number of Medusa heads (aiola has 11, we use 5)",
    )

    # Data
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/Users/ayates/model_mlx_migration/data/benchmarks/librispeech/LibriSpeech",
        help="Path to LibriSpeech dataset root",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train-clean-100"],
        help="LibriSpeech splits to use",
    )

    # Training
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5000,
        help="Maximum training steps",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate (low for fine-tuning)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Learning rate warmup steps",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/medusa_v3_finetuned",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=500,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=250,
        help="Evaluate every N steps",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Log metrics every N steps",
    )

    # Resume
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Medusa Fine-Tuning: aiola v2 -> whisper-large-v3")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Medusa heads: {args.n_heads}")
    print(f"aiola weights: {args.aiola_weights}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Max steps: {args.max_steps}")
    print(f"Data: {args.data_dir} ({args.splits})")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    # Verify aiola weights exist
    if not os.path.exists(args.aiola_weights):
        print(f"ERROR: aiola weights not found at {args.aiola_weights}")
        print("Run scripts/convert_aiola_medusa.py first to extract weights")
        sys.exit(1)

    # Load WhisperMLX model
    print("\nLoading WhisperMLX model...")
    model = WhisperMLX.from_pretrained(args.model)
    print(f"Loaded {args.model}")

    # Get model config
    config = model.config
    n_state = config.n_text_state
    n_vocab = config.n_vocab

    print(f"Model config: n_state={n_state}, n_vocab={n_vocab}")

    # Get tokenizer
    is_multilingual = config.n_vocab >= 51865
    num_langs = config.n_vocab - 51765 - int(is_multilingual)
    tokenizer = get_whisper_tokenizer(multilingual=is_multilingual, num_languages=num_langs)

    # Create proj_out from decoder's token embedding (aiola approach)
    # In aiola, proj_out is the decoder's lm_head (token_embedding used as linear)
    # Pass n_vocab to slice output to match decoder (which slices from padded to original)
    print("\nSetting up proj_out from decoder token_embedding...")
    proj_out = ProjOutWrapper(model.decoder.token_embedding, n_vocab=n_vocab)
    print(f"  proj_out: {n_state} -> {n_vocab}")

    # Initialize Medusa module with aiola architecture
    print(f"\nInitializing Medusa module with {args.n_heads} aiola-style ResBlock heads...")
    medusa = MedusaModule(
        n_state=n_state,
        n_vocab=n_vocab,
        n_heads=args.n_heads,
        use_aiola=True,  # Use MedusaResBlock architecture
        proj_out=proj_out,
    )

    # Load aiola weights
    medusa = load_aiola_weights_into_module(
        medusa,
        args.aiola_weights,
        n_heads=args.n_heads,
    )

    # Count parameters
    def count_params(params_dict):
        total = 0
        for v in params_dict.values():
            if isinstance(v, dict):
                total += count_params(v)
            elif hasattr(v, 'size'):
                total += v.size
        return total

    total_params = sum(count_params(head.parameters()) for head in medusa.heads)
    model_params = count_params(model.parameters())
    print(f"\nMedusa parameters: {total_params:,}")
    print(f"Parameter overhead: {total_params / model_params * 100:.2f}%")

    # Create training config
    train_config = MedusaTrainingConfig(
        n_medusa_heads=args.n_heads,
        use_block=False,  # Not using block variant
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_epochs=100,  # High number, but will stop at max_steps
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        checkpoint_dir=args.output_dir,
        save_every_steps=args.save_every,
        eval_every_steps=args.eval_every,
        log_every_steps=args.log_every,
    )

    # Create encoder function
    def encode_audio(audio: mx.array) -> mx.array:
        """Encode audio to features using frozen encoder."""
        # Pad or trim to 30s
        target_len = 30 * 16000
        if audio.shape[0] < target_len:
            padding = mx.zeros((target_len - audio.shape[0],))
            audio = mx.concatenate([audio, padding])
        else:
            audio = audio[:target_len]

        # Convert to mel spectrogram
        mel = log_mel_spectrogram(audio, n_mels=model.config.n_mels)
        # Pad/trim to standard length (3000 frames)
        target_mel_len = model.config.n_audio_ctx * 2
        if mel.shape[0] < target_mel_len:
            mel = mx.pad(mel, [(0, target_mel_len - mel.shape[0]), (0, 0)])
        else:
            mel = mel[:target_mel_len]

        # Encode audio
        features = model.embed_audio(mel)
        if features.ndim == 3 and features.shape[0] == 1:
            features = features.squeeze(0)
        return features

    # Create data loader
    print(f"\nLoading LibriSpeech from {args.data_dir}...")
    data_loader = LibriSpeechDataLoader(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        encoder_fn=encode_audio,
        config=train_config,
        splits=args.splits,
    )
    print(f"Found {len(data_loader)} samples")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create trainer
    trainer = MedusaTrainer(
        model=model,
        medusa_module=medusa,
        config=train_config,
        tokenizer=tokenizer,
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Run training
    print("\n" + "=" * 60)
    print("Starting fine-tuning...")
    print("=" * 60)
    start_time = time.time()

    try:
        trainer.train(data_loader)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer.save_checkpoint()

    elapsed = time.time() - start_time
    print(f"\nFine-tuning completed in {elapsed / 3600:.2f} hours")
    print(f"Final step: {trainer.global_step}")
    print(f"Best loss: {trainer.best_loss:.4f}")

    # Save final weights
    final_path = os.path.join(args.output_dir, "medusa_v3_finetuned.npz")
    save_medusa_weights(medusa, final_path)
    print(f"\nSaved final weights to {final_path}")

    # Save metadata
    metadata = {
        "source": "aiola/whisper-medusa-v1",
        "target": args.model,
        "n_heads": args.n_heads,
        "learning_rate": args.learning_rate,
        "max_steps": args.max_steps,
        "final_step": trainer.global_step,
        "best_loss": float(trainer.best_loss),
        "training_time_hours": elapsed / 3600,
        "data_splits": args.splits,
    }
    import json
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
