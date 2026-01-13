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
Train Medusa heads for WhisperMLX using self-distillation.

This script trains Medusa heads to predict multiple future tokens,
enabling speculative decoding for 1.5-2.5x inference speedup.

Training approach:
1. Load pretrained WhisperMLX model (frozen)
2. Initialize Medusa heads
3. For each audio-text pair:
   - Encode audio with frozen encoder
   - Get decoder hidden states and teacher logits
   - Train Medusa heads to predict shifted teacher distributions
4. Save trained Medusa weights

Usage:
    python scripts/train_medusa_heads.py \
        --model large-v3 \
        --data-dir /path/to/librispeech \
        --output-dir checkpoints/medusa \
        --epochs 10 \
        --batch-size 4

Requirements:
    - LibriSpeech dataset downloaded
    - WhisperMLX model (will download if not cached)
    - soundfile for audio loading
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Medusa heads for WhisperMLX"
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="large-v3",
        help="Whisper model size (tiny, base, small, medium, large-v3)",
    )
    parser.add_argument(
        "--n-heads",
        type=int,
        default=5,
        help="Number of Medusa heads to train",
    )
    parser.add_argument(
        "--use-block",
        action="store_true",
        help="Use Medusa-Block variant (default: Medusa-Linear)",
    )

    # Data
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
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
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum training steps (overrides epochs)",
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
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1000,
        help="Learning rate warmup steps",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/medusa",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1000,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=500,
        help="Evaluate every N steps",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
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
    print("Medusa Head Training for WhisperMLX")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Medusa heads: {args.n_heads}")
    print(f"Variant: {'Medusa-Block' if args.use_block else 'Medusa-Linear'}")
    print(f"Data: {args.data_dir} ({args.splits})")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    # Load WhisperMLX model
    print("\nLoading WhisperMLX model...")
    model = WhisperMLX.from_pretrained(args.model)
    print(f"Loaded {args.model} model")

    # Get model config
    config = model.config
    n_state = config.n_text_state
    n_vocab = config.n_vocab

    print(f"Model config: n_state={n_state}, n_vocab={n_vocab}")

    # Get tokenizer
    is_multilingual = config.n_vocab >= 51865
    num_langs = config.n_vocab - 51765 - int(is_multilingual)
    tokenizer = get_whisper_tokenizer(multilingual=is_multilingual, num_languages=num_langs)

    # Initialize Medusa module
    print(f"\nInitializing Medusa module with {args.n_heads} heads...")
    medusa = MedusaModule(
        n_state=n_state,
        n_vocab=n_vocab,
        n_heads=args.n_heads,
        use_block=args.use_block,
    )

    # Count parameters - helper to flatten nested param dicts
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
    print(f"Medusa parameters: {total_params:,}")
    print(f"Parameter overhead: {total_params / model_params * 100:.1f}%")

    # Create training config
    train_config = MedusaTrainingConfig(
        n_medusa_heads=args.n_heads,
        use_block=args.use_block,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
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

        # Encode audio (embed_audio expects (batch, n_mels, n_frames) or (n_mels, n_frames))
        # Returns (1, 1500, 1280) for single input - squeeze to (1500, 1280) for batching
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
    print("\nStarting training...")
    start_time = time.time()

    try:
        trainer.train(data_loader)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer.save_checkpoint()

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed / 3600:.2f} hours")
    print(f"Final step: {trainer.global_step}")
    print(f"Best loss: {trainer.best_loss:.4f}")

    # Save final weights
    final_path = os.path.join(args.output_dir, "medusa_final.npz")
    save_medusa_weights(medusa, final_path)
    print(f"\nSaved final weights to {final_path}")


if __name__ == "__main__":
    main()
