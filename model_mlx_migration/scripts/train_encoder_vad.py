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
Train Encoder VAD Head via Silero Distillation.

This script trains a lightweight VAD head on top of Whisper encoder outputs
by distilling knowledge from Silero VAD. The trained head enables skipping
decoder calls for silent positions, providing ~1.2x speedup.

Training data:
- LibriSpeech (clean speech with natural pauses)
- Augmented with silence and noise

Usage:
    # Quick training on small dataset (for testing)
    python scripts/train_encoder_vad.py --quick

    # Full training
    python scripts/train_encoder_vad.py --dataset librispeech-dev-clean --epochs 10

    # Resume training
    python scripts/train_encoder_vad.py --resume checkpoints/encoder_vad_epoch5.npz
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import mlx.core as mx
import mlx.optimizers as optim


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Encoder VAD Head via Silero Distillation"
    )

    # Data settings
    parser.add_argument(
        "--dataset",
        type=str,
        default="librispeech-dev-clean",
        help="Dataset to use (librispeech-dev-clean, librispeech-test-clean, etc.)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to local dataset directory (optional, downloads if not provided)",
    )
    parser.add_argument(
        "--local-audio-dir",
        type=str,
        default=None,
        help="Use local audio files from this directory instead of downloading",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to use (for debugging)",
    )

    # Model settings
    parser.add_argument(
        "--model",
        type=str,
        default="large-v3",
        help="Whisper model to use for encoder",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="VAD head hidden dimension",
    )

    # Training settings
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )

    # Output settings
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/encoder_vad",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint",
    )

    # Quick mode
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick training mode (10 samples, 2 epochs) for testing",
    )

    # Verbosity
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    return parser.parse_args()


def load_local_audio_samples(
    audio_dir: str,
    max_samples: Optional[int] = None,
) -> List[Tuple[np.ndarray, str]]:
    """
    Load audio samples from a local directory.

    Returns list of (audio_array, transcript) tuples.
    """
    import soundfile as sf
    from pathlib import Path

    audio_path = Path(audio_dir)
    if not audio_path.exists():
        raise ValueError(f"Audio directory not found: {audio_dir}")

    # Find all wav files
    audio_files = list(audio_path.rglob("*.wav"))
    print(f"Found {len(audio_files)} WAV files in {audio_dir}")

    if max_samples is not None:
        audio_files = audio_files[:max_samples]

    samples = []
    for audio_file in audio_files:
        audio, sr = sf.read(str(audio_file))

        # Resample to 16kHz if needed
        if sr != 16000:
            ratio = 16000 / sr
            new_len = int(len(audio) * ratio)
            audio = np.interp(
                np.linspace(0, len(audio), new_len),
                np.arange(len(audio)),
                audio
            )

        audio = audio.astype(np.float32)

        # Normalize
        if audio.max() > 0:
            audio = audio / np.abs(audio).max()

        # Use filename as pseudo-transcript
        transcript = audio_file.stem
        samples.append((audio, transcript))

    print(f"Loaded {len(samples)} samples")
    return samples


def load_librispeech_samples(
    dataset_name: str = "librispeech-dev-clean",
    data_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> List[Tuple[np.ndarray, str]]:
    """
    Load audio samples from LibriSpeech dataset.

    Returns list of (audio_array, transcript) tuples.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets library required: pip install datasets")

    print(f"Loading dataset: {dataset_name}")

    # Parse dataset name to HuggingFace format
    # LibriSpeech configs: 'clean', 'other', 'all'
    # LibriSpeech splits: 'train.clean.100', 'train.clean.360', 'train.other.500',
    #                     'validation.clean', 'validation.other', 'test.clean', 'test.other'
    if dataset_name.startswith("librispeech-"):
        # Map our names to HuggingFace format
        # e.g., 'librispeech-dev-clean' -> config='clean', split='validation.clean'
        # e.g., 'librispeech-test-clean' -> config='clean', split='test.clean'
        # e.g., 'librispeech-train-clean-100' -> config='clean', split='train.clean.100'
        if "clean" in dataset_name:
            config = "clean"
        elif "other" in dataset_name:
            config = "other"
        else:
            config = "all"

        if "dev" in dataset_name:
            split = "validation"
        elif "test" in dataset_name:
            split = "test"
        elif "train" in dataset_name:
            # Extract size if present (e.g., train-clean-100)
            if "100" in dataset_name:
                split = "train.100"
            elif "360" in dataset_name:
                split = "train.360"
            elif "500" in dataset_name:
                split = "train.500"
            else:
                split = "train.100"  # default to smallest
        else:
            split = "validation"

        ds = load_dataset(
            "librispeech_asr",
            config,
            split=split,
            cache_dir=data_dir,
        )
    else:
        # Try loading as-is
        ds = load_dataset(dataset_name, cache_dir=data_dir)
        if isinstance(ds, dict):
            ds = ds.get("train") or ds.get("validation") or list(ds.values())[0]

    # Limit samples if requested
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    print(f"Loaded {len(ds)} samples")

    # Extract audio and transcripts
    samples = []
    for item in ds:
        audio = item["audio"]["array"]
        transcript = item.get("text", "")
        samples.append((audio.astype(np.float32), transcript))

    return samples


def create_training_batch(
    samples: List[Tuple[np.ndarray, str]],
    batch_indices: List[int],
    model,
    distiller,
    n_mels: int = 128,
) -> Tuple[mx.array, mx.array]:
    """
    Create a training batch from samples.

    Returns:
        (encoder_outputs, vad_labels) batch
    """
    from tools.whisper_mlx.audio import log_mel_spectrogram

    encoder_outputs = []
    vad_labels = []

    for idx in batch_indices:
        audio, _ = samples[idx]

        # Compute mel spectrogram
        mel = log_mel_spectrogram(audio, n_mels=n_mels)

        # Pad/trim to 30s (3000 frames)
        target_len = 3000
        if mel.shape[0] < target_len:
            mel = mx.pad(mel, [(0, target_len - mel.shape[0]), (0, 0)])
        elif mel.shape[0] > target_len:
            mel = mel[:target_len, :]

        # Add batch dimension and encode
        mel_batch = mel[None]  # (1, 3000, n_mels)
        encoder_output = model.embed_audio(mel_batch, variable_length=False)
        mx.eval(encoder_output)

        # Get encoder output shape for VAD labels
        seq_len = encoder_output.shape[1]  # Should be 1500

        # Get VAD labels from Silero
        labels = distiller.get_vad_labels(audio, seq_len)

        encoder_outputs.append(encoder_output)
        vad_labels.append(labels)

    # Stack into batch
    encoder_batch = mx.concatenate(encoder_outputs, axis=0)
    labels_batch = mx.stack(vad_labels, axis=0)

    # Cast to float32 for stable training
    encoder_batch = encoder_batch.astype(mx.float32)
    labels_batch = labels_batch.astype(mx.float32)

    return encoder_batch, labels_batch


def train_epoch(
    vad_head,
    model,
    distiller,
    samples: List[Tuple[np.ndarray, str]],
    optimizer,
    batch_size: int,
    n_mels: int = 128,
    verbose: bool = False,
) -> float:
    """Train for one epoch, return average loss."""
    total_loss = 0.0
    n_batches = 0

    # Shuffle indices
    indices = list(range(len(samples)))
    np.random.shuffle(indices)

    # Process batches
    for batch_start in range(0, len(indices), batch_size):
        batch_indices = indices[batch_start:batch_start + batch_size]

        # Create batch
        encoder_outputs, labels = create_training_batch(
            samples, batch_indices, model, distiller, n_mels
        )

        # Forward and backward pass
        def loss_fn(vad_head):
            logits = vad_head.get_logits(encoder_outputs, training=True)
            loss = distiller.compute_loss(logits, labels)
            return loss

        loss, grads = mx.value_and_grad(loss_fn)(vad_head)
        optimizer.update(vad_head, grads)
        mx.eval(vad_head.parameters(), optimizer.state)

        total_loss += float(loss)
        n_batches += 1

        if verbose and n_batches % 10 == 0:
            print(f"  Batch {n_batches}: loss = {float(loss):.4f}")

    return total_loss / max(n_batches, 1)


def evaluate(
    vad_head,
    model,
    distiller,
    samples: List[Tuple[np.ndarray, str]],
    batch_size: int,
    n_mels: int = 128,
) -> Tuple[float, float, float]:
    """
    Evaluate VAD head on samples.

    Returns:
        (loss, precision, recall)
    """
    total_loss = 0.0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    n_batches = 0

    for batch_start in range(0, len(samples), batch_size):
        batch_indices = list(range(
            batch_start,
            min(batch_start + batch_size, len(samples))
        ))

        # Create batch
        encoder_outputs, labels = create_training_batch(
            samples, batch_indices, model, distiller, n_mels
        )

        # Forward pass (no training)
        logits = vad_head.get_logits(encoder_outputs, training=False)
        loss = distiller.compute_loss(logits, labels)

        # Compute metrics
        probs = mx.sigmoid(logits)
        preds = probs >= 0.5
        labels_bool = labels >= 0.5

        tp = mx.sum(preds & labels_bool)
        fp = mx.sum(preds & ~labels_bool)
        fn = mx.sum(~preds & labels_bool)

        true_positives += int(tp)
        false_positives += int(fp)
        false_negatives += int(fn)
        total_loss += float(loss)
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)

    return avg_loss, precision, recall


def main():
    args = parse_args()

    # Quick mode overrides
    if args.quick:
        args.max_samples = 10
        args.epochs = 2
        args.batch_size = 2
        print("Quick mode: 10 samples, 2 epochs")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load samples
    if args.local_audio_dir:
        samples = load_local_audio_samples(
            args.local_audio_dir,
            args.max_samples,
        )
    else:
        samples = load_librispeech_samples(
            args.dataset,
            args.data_dir,
            args.max_samples,
        )

    # Split into train/val (90/10)
    n_train = int(len(samples) * 0.9)
    train_samples = samples[:n_train]
    val_samples = samples[n_train:]

    print(f"Train: {len(train_samples)} samples, Val: {len(val_samples)} samples")

    # Load Whisper model (encoder only)
    print(f"Loading Whisper model: {args.model}")
    from tools.whisper_mlx.model import WhisperMLX
    model = WhisperMLX.from_pretrained(
        args.model,
        warmup=True,
    )

    # Get encoder config
    n_state = model.config.n_audio_state
    n_mels = model.config.n_mels

    print(f"Encoder config: n_state={n_state}, n_mels={n_mels}")

    # Create VAD head
    from tools.whisper_mlx.encoder_vad import (
        EncoderVADHead,
        SileroVADDistiller,
        save_encoder_vad_head,
        load_encoder_vad_head,
    )

    if args.resume:
        print(f"Resuming from: {args.resume}")
        vad_head = load_encoder_vad_head(
            args.resume,
            n_state=n_state,
            hidden_dim=args.hidden_dim,
            dtype=mx.float32,  # Use float32 for training
        )
    else:
        # Use float32 for training to avoid mixed-precision issues with AdamW
        vad_head = EncoderVADHead(
            n_state=n_state,
            hidden_dim=args.hidden_dim,
            dtype=mx.float32,
        )

    # Create distiller
    distiller = SileroVADDistiller()

    # Create optimizer
    optimizer = optim.AdamW(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )

    # Training loop
    print("\nStarting training...")
    best_loss = float("inf")

    for epoch in range(args.epochs):
        t0 = time.perf_counter()

        # Train
        train_loss = train_epoch(
            vad_head,
            model,
            distiller,
            train_samples,
            optimizer,
            args.batch_size,
            n_mels,
            args.verbose,
        )

        # Validate
        val_loss, precision, recall = evaluate(
            vad_head,
            model,
            distiller,
            val_samples,
            args.batch_size,
            n_mels,
        )

        t1 = time.perf_counter()
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)

        print(
            f"Epoch {epoch+1}/{args.epochs}: "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, "
            f"time={t1-t0:.1f}s"
        )

        # Save checkpoint
        ckpt_path = output_dir / f"encoder_vad_epoch{epoch+1}.npz"
        save_encoder_vad_head(vad_head, str(ckpt_path))

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_path = output_dir / "encoder_vad_best.npz"
            save_encoder_vad_head(vad_head, str(best_path))
            print(f"  New best model saved: {best_path}")

    # Final output
    print("\nTraining complete!")
    print(f"Best model: {output_dir / 'encoder_vad_best.npz'}")
    print(f"Best val_loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
