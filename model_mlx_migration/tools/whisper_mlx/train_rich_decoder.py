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
RichDecoder Training Script.

Trains RichDecoder (LoRA adapters + rich output heads) on frozen Whisper.
The encoder and decoder are frozen; only LoRA adapters and output heads train.

Output heads:
1. Emotion Head - 34-class emotion detection
2. Paralinguistics Head - 50-class non-speech vocalizations
3. Language Head - 100-class language detection (code-switching)
4. Phoneme Deviation Head - Hallucination signal (0-1 score)

Architecture:
    encoder_out (frozen) -> decoder (frozen + LoRA) -> rich outputs

Training strategy:
- Multi-task learning with weighted loss
- LoRA rank=8, alpha=16 (following LoRA paper)
- Prosody cross-attention for CTC features conditioning
- Use unified_emotion dataset for emotion training

Usage:
    python -m tools.whisper_mlx.train_rich_decoder \
        --emotion-data data/emotion/unified_emotion \
        --output-dir checkpoints/rich_decoder \
        --epochs 10

    # With paralinguistics data
    python -m tools.whisper_mlx.train_rich_decoder \
        --emotion-data data/emotion/unified_emotion \
        --para-checkpoint checkpoints/paralinguistics_v3/best.npz \
        --output-dir checkpoints/rich_decoder \
        --epochs 10

References:
    - LoRA: https://arxiv.org/abs/2106.09685
    - UNIFIED_RICH_AUDIO_ARCHITECTURE.md
"""

import argparse
import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

# Import Whisper components
from .audio import load_audio, log_mel_spectrogram
from .encoder_cache import TrainingEncoderCache
from .model import WhisperMLX
from .rich_decoder import RichDecoder, RichDecoderConfig
from .tokenizer import get_whisper_tokenizer


def clear_memory():
    """Clear memory between batches."""
    if hasattr(mx, 'clear_cache'):
        mx.clear_cache()
    gc.collect()


@dataclass
class RichDecoderTrainingConfig:
    """Configuration for RichDecoder training."""

    # Data
    emotion_data_dir: str | None = None  # HuggingFace dataset
    para_checkpoint: str | None = None   # Pretrained paralinguistics head

    # Output
    output_dir: str = "checkpoints/rich_decoder"

    # Model
    whisper_model: str = "mlx-community/whisper-large-v3-mlx"

    # LoRA configuration
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    # Training
    epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 1e-4
    warmup_steps: int = 500
    weight_decay: float = 0.01

    # Loss weights
    emotion_loss_weight: float = 1.0
    para_loss_weight: float = 0.5
    language_loss_weight: float = 0.3
    deviation_loss_weight: float = 0.2

    # Audio
    max_audio_len: float = 10.0  # seconds
    sample_rate: int = 16000

    # Logging
    log_interval: int = 50
    save_interval: int = 500

    # Decoder sequence length
    max_decoder_len: int = 64

    # Encoder caching
    encoder_cache_dir: str | None = None


@dataclass
class RichDecoderSample:
    """Sample for RichDecoder training."""

    audio_path: str
    audio_array: np.ndarray | None = None

    # Labels
    emotion_id: int = 0
    language_id: int = 0
    para_id: int | None = None

    # Transcription (for decoder)
    text: str | None = None
    token_ids: list[int] | None = None


class UnifiedEmotionDataset:
    """Load unified emotion dataset from HuggingFace format."""

    # Emotion label mapping (aligned with RichDecoder config)
    EMOTION_MAP = {
        'neutral': 0, 'happy': 1, 'sad': 2, 'angry': 3,
        'fear': 4, 'disgust': 5, 'surprise': 6, 'contempt': 7,
        # Additional emotions
        'amused': 8, 'frustrated': 9, 'annoyed': 10, 'confused': 11,
        'excited': 12, 'bored': 13, 'calm': 14, 'anxious': 15,
        # More from expresso
        'enunciated': 16, 'laughing': 17, 'whisper': 18, 'projected': 19,
        'narration': 20, 'singing': 21, 'sarcastic': 22, 'sleepy': 23,
        'emphatic': 24, 'hesitant': 25, 'soft': 26, 'aggressive': 27,
        'concerned': 28, 'loving': 29, 'stern': 30, 'encouraging': 31,
        'disappointed': 32, 'hopeful': 33,
    }

    # Language mapping
    LANGUAGE_MAP = {
        'en': 0, 'zh': 1, 'ja': 2, 'ko': 3, 'de': 4,
        'es': 5, 'fr': 6, 'ru': 7, 'hi': 8, 'pt': 9,
        # More languages
        'ar': 10, 'it': 11, 'nl': 12, 'pl': 13, 'tr': 14,
        'vi': 15, 'th': 16, 'id': 17, 'ms': 18, 'sv': 19,
    }

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        max_audio_len: float = 10.0,
        sample_rate: int = 16000,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_audio_len = max_audio_len
        self.sample_rate = sample_rate
        self.samples: list[RichDecoderSample] = []

        self._load_dataset()
        print(f"Loaded {len(self.samples)} {split} samples")

    def _load_dataset(self):
        """Load dataset from HuggingFace format."""
        from datasets import load_from_disk

        print(f"Loading unified emotion dataset from: {self.data_dir}")

        try:
            ds = load_from_disk(str(self.data_dir))
            split_ds = ds[self.split]

            # Note: Audio is loaded lazily via torchcodec AudioDecoder
            # We handle decoding in _process_item

            print(f"  Processing {len(split_ds)} samples...")

            for i in range(len(split_ds)):
                try:
                    item = split_ds[i]
                    sample = self._process_item(item)
                    if sample:
                        self.samples.append(sample)
                except Exception:
                    pass  # Skip corrupted samples

                if (i + 1) % 1000 == 0:
                    print(f"    Processed {i + 1}/{len(split_ds)} samples")

        except Exception as e:
            print(f"Error loading dataset: {e}")

    def _process_item(self, item: dict) -> RichDecoderSample | None:
        """Process a single item."""
        # Get emotion label
        emotion = item.get("emotion", "neutral")
        emotion_id = self.EMOTION_MAP.get(emotion.lower(), 0)

        # Get language
        language = item.get("language", "en")
        language_id = self.LANGUAGE_MAP.get(language.lower(), 0)

        # Get audio - handle multiple formats
        audio_data = item.get("audio")
        if audio_data is None:
            return None

        try:
            # Format 1: HuggingFace Audio dict with array
            if isinstance(audio_data, dict):
                audio_array = audio_data.get("array")
                sample_rate = audio_data.get("sampling_rate", 16000)
            # Format 2: torchcodec AudioDecoder (lazy loading)
            elif hasattr(audio_data, 'get_all_samples'):
                samples = audio_data.get_all_samples()
                audio_array = samples.data.numpy().squeeze()  # (1, N) -> (N,)
                sample_rate = samples.sample_rate
            else:
                return None
        except Exception:
            return None

        if audio_array is None:
            return None

        # Resample if needed
        if sample_rate != 16000:
            ratio = sample_rate / 16000
            new_len = int(len(audio_array) / ratio)
            audio_array = np.interp(
                np.linspace(0, len(audio_array) - 1, new_len),
                np.arange(len(audio_array)),
                audio_array,
            )

        # Check duration
        duration = len(audio_array) / 16000
        if duration > self.max_audio_len:
            return None

        return RichDecoderSample(
            audio_path="__in_memory__",
            audio_array=np.array(audio_array, dtype=np.float32),
            emotion_id=emotion_id,
            language_id=language_id,
        )

    def get_samples(self) -> list[RichDecoderSample]:
        return self.samples


class RichDecoderTrainer:
    """Trainer for RichDecoder."""

    def __init__(
        self,
        config: RichDecoderTrainingConfig,
        whisper_model: WhisperMLX,
        rich_decoder: RichDecoder,
        tokenizer: Any,
    ):
        self.config = config
        self.whisper_model = whisper_model
        self.rich_decoder = rich_decoder
        self.tokenizer = tokenizer

        # Optimizer (only trainable params)
        self.optimizer = optim.AdamW(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Training state
        self.step = 0
        self.epoch = 0
        self.best_loss = float("inf")
        self.best_acc = 0.0

        # Encoder cache
        self.encoder_cache = None
        if config.encoder_cache_dir:
            self.encoder_cache = TrainingEncoderCache(
                cache_dir=config.encoder_cache_dir,
                use_compression=True,
            )

        # Output
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.output_dir / "training.log"

    def log(self, message: str):
        """Log to console and file."""
        print(message)
        with open(self.log_file, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {message}\n")

    def _encode_audio(self, audio: np.ndarray) -> mx.array:
        """Encode audio through frozen Whisper encoder."""
        mel = log_mel_spectrogram(audio)

        # Pad/trim to 30s (3000 frames)
        target_frames = 3000
        if mel.shape[0] < target_frames:
            pad_len = target_frames - mel.shape[0]
            mel = np.pad(mel, ((0, pad_len), (0, 0)), mode='constant')
        else:
            mel = mel[:target_frames]

        mel_mx = mx.array(mel)[None, ...]  # (1, 3000, n_mels)
        return self.whisper_model.embed_audio(mel_mx)


    def _prepare_batch(
        self,
        samples: list[RichDecoderSample],
    ) -> dict[str, mx.array]:
        """Prepare a batch for training."""
        batch_encoder_outputs = []
        batch_tokens = []
        emotion_labels = []
        language_labels = []

        max_samples = int(self.config.max_audio_len * self.config.sample_rate)

        for sample in samples:
            # Load audio
            if sample.audio_array is not None:
                audio = sample.audio_array
            else:
                audio = load_audio(sample.audio_path)

            # Trim to max length
            if len(audio) > max_samples:
                audio = audio[:max_samples]

            # Encode
            encoder_output = self._encode_audio(audio)
            batch_encoder_outputs.append(encoder_output[0])  # Remove batch dim

            # Create decoder input (SOT token only for now)
            # In a full implementation, we'd use actual transcriptions
            sot_token = self.tokenizer.sot
            batch_tokens.append([sot_token])

            # Labels
            emotion_labels.append(sample.emotion_id)
            language_labels.append(sample.language_id)

        # Stack encoder outputs
        encoder_outputs = mx.stack(batch_encoder_outputs)

        # Pad tokens to same length
        max_len = max(len(t) for t in batch_tokens)
        padded_tokens = []
        for tokens in batch_tokens:
            padded = tokens + [self.tokenizer.eot] * (max_len - len(tokens))
            padded_tokens.append(padded)
        token_ids = mx.array(padded_tokens)

        return {
            "encoder_outputs": encoder_outputs,
            "token_ids": token_ids,
            "emotion_labels": mx.array(emotion_labels),
            "language_labels": mx.array(language_labels),
        }

    def _compute_loss(
        self,
        rich_outputs: dict[str, mx.array],
        batch: dict[str, mx.array],
    ) -> tuple[mx.array, dict[str, float]]:
        """Compute multi-task loss."""
        losses = {}

        # Emotion loss (cross-entropy on mean-pooled output)
        emotion_logits = rich_outputs["emotion"]  # (batch, seq, num_emotions)
        emotion_pooled = mx.mean(emotion_logits, axis=1)  # (batch, num_emotions)
        emotion_loss = mx.mean(
            nn.losses.cross_entropy(emotion_pooled, batch["emotion_labels"]),
        )
        losses["emotion"] = float(emotion_loss)

        # Language loss (cross-entropy on mean-pooled output)
        language_logits = rich_outputs["language"]  # (batch, seq, num_languages)
        language_pooled = mx.mean(language_logits, axis=1)  # (batch, num_languages)
        language_loss = mx.mean(
            nn.losses.cross_entropy(language_pooled, batch["language_labels"]),
        )
        losses["language"] = float(language_loss)

        # Combined weighted loss
        total_loss = (
            self.config.emotion_loss_weight * emotion_loss +
            self.config.language_loss_weight * language_loss
        )

        return total_loss, losses

    def train_step(self, batch_samples: list[RichDecoderSample]) -> dict[str, float]:
        """Single training step."""
        batch = self._prepare_batch(batch_samples)

        def loss_fn(decoder):
            # Forward through rich decoder
            outputs = decoder(
                x=batch["token_ids"],
                xa=batch["encoder_outputs"],
            )
            total_loss, _ = self._compute_loss(outputs, batch)
            return total_loss

        # Compute loss and gradients
        loss, grads = mx.value_and_grad(loss_fn)(self.rich_decoder)

        # Update only trainable parameters (LoRA, prosody attention, rich heads)
        trainable_grads = {
            key: grad
            for key, grad in grads.items()
            if any(k in key for k in ["lora", "prosody", "rich"])
        }

        self.optimizer.update(self.rich_decoder, trainable_grads)
        mx.eval(self.rich_decoder.parameters())

        # Get loss breakdown (MLX doesn't track gradients in forward pass)
        outputs = self.rich_decoder(
            x=batch["token_ids"],
            xa=batch["encoder_outputs"],
        )
        _, losses = self._compute_loss(outputs, batch)
        losses["total"] = float(loss)

        return losses

    def train(
        self,
        train_samples: list[RichDecoderSample],
        val_samples: list[RichDecoderSample],
    ):
        """Main training loop."""
        steps_per_epoch = (len(train_samples) + self.config.batch_size - 1) // self.config.batch_size

        self.log("Starting RichDecoder training")
        self.log(f"Epochs: {self.config.epochs}")
        self.log(f"Train samples: {len(train_samples)}")
        self.log(f"Val samples: {len(val_samples)}")
        self.log(f"Steps per epoch: {steps_per_epoch}")
        self.log(f"LoRA rank: {self.config.lora_rank}, alpha: {self.config.lora_alpha}")

        # Print class distribution
        emotion_counts = {}
        for s in train_samples:
            emotion_counts[s.emotion_id] = emotion_counts.get(s.emotion_id, 0) + 1
        self.log(f"Emotion distribution: {len(emotion_counts)} classes")

        for epoch in range(self.config.epochs):
            self.epoch = epoch
            self.log(f"Starting epoch {epoch}")

            # Shuffle
            rng = np.random.default_rng(42 + epoch)
            rng.shuffle(train_samples)

            # Create batches
            batches = [
                train_samples[i:i + self.config.batch_size]
                for i in range(0, len(train_samples), self.config.batch_size)
            ]

            # Training epoch
            epoch_losses = []
            for batch in batches:
                losses = self.train_step(batch)
                epoch_losses.append(losses["total"])
                self.step += 1

                # Logging
                if self.step % self.config.log_interval == 0:
                    avg_loss = np.mean(epoch_losses[-self.config.log_interval:])
                    self.log(
                        f"  Step {self.step}: loss={avg_loss:.4f} "
                        f"(emo={losses.get('emotion', 0):.4f}, lang={losses.get('language', 0):.4f})",
                    )

                # Save checkpoint
                if self.step % self.config.save_interval == 0:
                    self._save_checkpoint(f"step_{self.step}.npz")

                clear_memory()

            # Epoch summary
            if epoch_losses:
                avg_epoch_loss = np.mean(epoch_losses)
                self.log(f"Epoch {epoch + 1}/{self.config.epochs}: loss={avg_epoch_loss:.4f}")

            # Validation
            if val_samples:
                val_loss, val_acc = self._validate(val_samples)
                self.log(f"  Val loss: {val_loss:.4f}, emotion_acc: {val_acc:.2%}")

                if val_acc > self.best_acc:
                    self.best_acc = val_acc
                    self.best_loss = val_loss
                    self._save_checkpoint("best.npz")
                    self.log(f"  New best model saved! (acc={val_acc:.2%})")

        # Save final
        self._save_checkpoint("final.npz")
        self.log("Training complete!")

    def _validate(
        self,
        val_samples: list[RichDecoderSample],
    ) -> tuple[float, float]:
        """Run validation."""
        # Set to eval mode to disable dropout
        self.rich_decoder.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch_start in range(0, len(val_samples), self.config.batch_size):
            batch_end = min(batch_start + self.config.batch_size, len(val_samples))
            batch_samples = val_samples[batch_start:batch_end]

            batch = self._prepare_batch(batch_samples)

            # Forward pass
            outputs = self.rich_decoder(
                x=batch["token_ids"],
                xa=batch["encoder_outputs"],
            )

            # Loss
            loss, _ = self._compute_loss(outputs, batch)
            total_loss += float(loss) * len(batch_samples)

            # Emotion accuracy
            emotion_logits = outputs["emotion"]
            emotion_pooled = mx.mean(emotion_logits, axis=1)
            preds = mx.argmax(emotion_pooled, axis=-1)
            correct += int(mx.sum(preds == batch["emotion_labels"]))
            total += len(batch_samples)

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)

        # Restore training mode
        self.rich_decoder.train()

        return avg_loss, accuracy

    def _save_checkpoint(self, filename: str):
        """Save trainable weights only."""
        save_path = self.output_dir / filename
        self.rich_decoder.save_trainable(str(save_path))
        self.log(f"Saved checkpoint: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="RichDecoder Training")
    parser.add_argument("--emotion-data", type=str, default="data/emotion/unified_emotion",
                        help="Path to unified emotion dataset")
    parser.add_argument("--output-dir", type=str, default="checkpoints/rich_decoder",
                        help="Output directory")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--model", type=str, default="mlx-community/whisper-large-v3-mlx",
                        help="Whisper model")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--encoder-cache-dir", type=str,
                        help="Directory for cached encoder outputs")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")

    args = parser.parse_args()

    config = RichDecoderTrainingConfig(
        emotion_data_dir=args.emotion_data,
        output_dir=args.output_dir,
        whisper_model=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        encoder_cache_dir=args.encoder_cache_dir,
    )

    print("=" * 70)
    print("RichDecoder Training")
    print("=" * 70)
    print(f"Output: {config.output_dir}")
    print(f"Model: {config.whisper_model}")
    print(f"LoRA: rank={config.lora_rank}, alpha={config.lora_alpha}")
    print()

    # Load Whisper model
    print("1. Loading Whisper model...")
    whisper_model = WhisperMLX.from_pretrained(config.whisper_model)
    print(f"   Encoder: {whisper_model.config.n_audio_state}-dim")

    # Get tokenizer
    print("2. Loading tokenizer...")
    tokenizer = get_whisper_tokenizer("large-v3")

    # Create RichDecoder config
    decoder_config = RichDecoderConfig(
        n_vocab=whisper_model.config.n_vocab,
        n_ctx=whisper_model.config.n_text_ctx,
        n_state=whisper_model.config.n_text_state,
        n_head=whisper_model.config.n_text_head,
        n_layer=whisper_model.config.n_text_layer,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout if hasattr(config, 'lora_dropout') else 0.0,
    )

    # Create RichDecoder from Whisper decoder
    print("3. Creating RichDecoder...")
    rich_decoder = RichDecoder.from_whisper_decoder(
        whisper_model.decoder,
        config=decoder_config,
    )

    # Count parameters
    from mlx.utils import tree_flatten
    total_params = sum(p.size for _, p in tree_flatten(rich_decoder.parameters()))
    trainable = rich_decoder.get_trainable_parameters()
    trainable_params = sum(p.size for _, p in tree_flatten(trainable))
    print(f"   Total parameters: {total_params / 1e6:.2f}M")
    print(f"   Trainable parameters: {trainable_params / 1e6:.2f}M")
    print(f"   LoRA adapters: {len(rich_decoder.lora_adapters)}")

    # Load emotion data
    print("4. Loading emotion data...")
    train_dataset = UnifiedEmotionDataset(
        data_dir=config.emotion_data_dir,
        split="train",
        max_audio_len=config.max_audio_len,
    )
    val_dataset = UnifiedEmotionDataset(
        data_dir=config.emotion_data_dir,
        split="validation",
        max_audio_len=config.max_audio_len,
    )

    train_samples = train_dataset.get_samples()
    val_samples = val_dataset.get_samples()

    if not train_samples:
        print("ERROR: No training data found!")
        return

    print(f"   Train: {len(train_samples)}, Val: {len(val_samples)}")

    # Create trainer
    trainer = RichDecoderTrainer(
        config=config,
        whisper_model=whisper_model,
        rich_decoder=rich_decoder,
        tokenizer=tokenizer,
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"5. Resuming from checkpoint: {args.resume}")
        rich_decoder.load_trainable(args.resume)

    # Train
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)
    trainer.train(train_samples, val_samples)


if __name__ == "__main__":
    main()
