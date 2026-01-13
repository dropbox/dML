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
Language Head Training Script.

Trains LanguageHead for per-frame language identification and code-switching detection.

The model outputs language probabilities per frame (50Hz), enabling detection of
mixed-language utterances like "I went to the 商店 yesterday".

Usage:
    python -m tools.whisper_mlx.train_language_head \
        --output-dir checkpoints/language_head_v1 \
        --epochs 5

Datasets:
- CommonVoice: English, Chinese, Japanese, Hindi
- OpenSLR: Korean, Russian, German, Spanish, French

References:
    - CommonVoice: https://commonvoice.mozilla.org/
    - Whisper language detection: https://github.com/openai/whisper
"""

import argparse
import gc
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
import pandas as pd

# Import Whisper components
from .audio import load_audio, log_mel_spectrogram
from .encoder_cache import TrainingEncoderCache
from .model import WhisperMLX
from .rich_ctc_head import LanguageHead


def clear_memory():
    """Clear memory between batches."""
    if hasattr(mx, 'clear_cache'):
        mx.clear_cache()
    gc.collect()


# Language codes to train on (subset of Whisper languages)
# Focus on major languages with good data coverage
TRAIN_LANGUAGES = {
    "en": 0,   # English
    "zh": 1,   # Chinese (Mandarin)
    "ja": 2,   # Japanese
    "hi": 3,   # Hindi
    "ko": 4,   # Korean
    "ru": 5,   # Russian
    "de": 6,   # German
    "es": 7,   # Spanish
    "fr": 8,   # French
}

INV_TRAIN_LANGUAGES = {v: k for k, v in TRAIN_LANGUAGES.items()}


@dataclass
class LanguageTrainingConfig:
    """Configuration for language head training."""

    # Data
    commonvoice_dir: str | None = "data/commonvoice/cv-corpus-24.0-2025-12-05"
    openslr_dir: str | None = "data/openslr"
    multilingual_dir: str | None = "data/multilingual"

    # Output
    output_dir: str = "checkpoints/language_head_v1"

    # Model
    whisper_model: str = "mlx-community/whisper-large-v3-mlx"
    d_model: int = 1280
    hidden_dim: int = 256
    num_languages: int = 9  # Match TRAIN_LANGUAGES

    # Training
    epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500

    # Class balancing
    use_class_weights: bool = True
    max_samples_per_language: int = 5000  # Limit per language for balance

    # Audio
    max_audio_len: float = 15.0  # seconds
    sample_rate: int = 16000

    # Logging
    log_interval: int = 50
    save_interval: int = 500
    eval_interval: int = 500

    # Encoder caching (~3x speedup)
    encoder_cache_dir: str | None = None

    # Length-sorted batching (~1.3x speedup)
    length_sorted_batching: bool = False
    bucket_size_multiplier: int = 100


@dataclass
class LanguageSample:
    """Sample for language training."""

    audio_path: str
    language: str      # Language code (e.g., "en", "zh")
    language_id: int   # Mapped to TRAIN_LANGUAGES
    transcript: str = ""  # Optional transcript
    duration: float = 0.0  # For length-sorted batching


class CommonVoiceDataset:
    """
    CommonVoice dataset loader for language identification.

    Loads audio files with language labels from CommonVoice directories.
    """

    def __init__(
        self,
        data_dir: str,
        languages: list[str],
        max_samples_per_language: int = 5000,
        max_audio_len: float = 15.0,
        split: str = "validated",
    ):
        self.data_dir = Path(data_dir)
        self.max_samples_per_language = max_samples_per_language
        self.max_audio_len = max_audio_len
        self.split = split
        self.samples: list[LanguageSample] = []

        print(f"Loading CommonVoice from: {self.data_dir}")

        for lang in languages:
            self._load_language(lang)

        print(f"Total CommonVoice samples: {len(self.samples)}")

    def _load_language(self, lang: str):
        """Load samples for a specific language."""
        # Map language codes to CommonVoice directory names
        lang_dir_map = {
            "en": "en",
            "zh": "zh-CN",
            "ja": "ja",
            "hi": "hi",
        }

        dir_name = lang_dir_map.get(lang, lang)
        lang_dir = self.data_dir / dir_name

        if not lang_dir.exists():
            print(f"  {lang}: directory not found at {lang_dir}")
            return

        # Try different TSV files in order of quality
        tsv_names = [f"{self.split}.tsv", "train.tsv", "validated.tsv"]
        tsv_path = None
        for name in tsv_names:
            candidate = lang_dir / name
            if candidate.exists():
                tsv_path = candidate
                break

        if tsv_path is None:
            print(f"  {lang}: no TSV file found")
            return

        # Load TSV
        try:
            df = pd.read_csv(tsv_path, sep='\t', low_memory=False)
        except Exception as e:
            print(f"  {lang}: error reading TSV: {e}")
            return

        clips_dir = lang_dir / "clips"
        if not clips_dir.exists():
            print(f"  {lang}: clips directory not found")
            return

        # Get language ID
        if lang not in TRAIN_LANGUAGES:
            print(f"  {lang}: not in TRAIN_LANGUAGES")
            return
        lang_id = TRAIN_LANGUAGES[lang]

        # Sample rows
        loaded = 0
        skipped = 0

        # Shuffle for random sampling
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

        for _, row in df_shuffled.iterrows():
            if loaded >= self.max_samples_per_language:
                break

            audio_file = row.get("path", "")
            if not audio_file:
                skipped += 1
                continue

            audio_path = clips_dir / audio_file
            if not audio_path.exists():
                # Try with .mp3 extension
                if not audio_file.endswith(".mp3"):
                    audio_path = clips_dir / f"{audio_file}.mp3"
                if not audio_path.exists():
                    skipped += 1
                    continue

            # Get transcript if available
            transcript = row.get("sentence", "")

            self.samples.append(LanguageSample(
                audio_path=str(audio_path),
                language=lang,
                language_id=lang_id,
                transcript=transcript if pd.notna(transcript) else "",
            ))
            loaded += 1

        print(f"  {lang}: loaded {loaded} samples ({skipped} skipped)")

    def get_samples(self) -> list[LanguageSample]:
        return self.samples


class OpenSLRDataset:
    """
    OpenSLR dataset loader for language identification.

    Loads audio files from extracted OpenSLR directories.
    """

    def __init__(
        self,
        data_dir: str,
        languages: list[str],
        max_samples_per_language: int = 5000,
        max_audio_len: float = 15.0,
    ):
        self.data_dir = Path(data_dir)
        self.max_samples_per_language = max_samples_per_language
        self.max_audio_len = max_audio_len
        self.samples: list[LanguageSample] = []

        print(f"Loading OpenSLR from: {self.data_dir}")

        for lang in languages:
            self._load_language(lang)

        print(f"Total OpenSLR samples: {len(self.samples)}")

    def _load_language(self, lang: str):
        """Load samples for a specific language from OpenSLR."""
        lang_dir = self.data_dir / lang

        if not lang_dir.exists():
            # Try alternative names
            alt_names = {
                "ko": "ko",
                "ru": "ru",
                "de": "de",
                "es": "es",
                "fr": "fr",
            }
            alt_name = alt_names.get(lang)
            if alt_name:
                lang_dir = self.data_dir / alt_name
            if not lang_dir.exists():
                print(f"  {lang}: directory not found at {lang_dir}")
                return

        # Get language ID
        if lang not in TRAIN_LANGUAGES:
            print(f"  {lang}: not in TRAIN_LANGUAGES")
            return
        lang_id = TRAIN_LANGUAGES[lang]

        # Find all audio files (wav, flac, mp3)
        audio_files = []
        for ext in ["*.wav", "*.flac", "*.mp3"]:
            audio_files.extend(lang_dir.rglob(ext))

        if not audio_files:
            print(f"  {lang}: no audio files found")
            return

        # Shuffle and sample
        rng = np.random.default_rng(42)
        rng.shuffle(audio_files)

        loaded = 0
        for audio_path in audio_files[:self.max_samples_per_language]:
            self.samples.append(LanguageSample(
                audio_path=str(audio_path),
                language=lang,
                language_id=lang_id,
            ))
            loaded += 1

        print(f"  {lang}: loaded {loaded} samples from OpenSLR")

    def get_samples(self) -> list[LanguageSample]:
        return self.samples


class LanguageTrainer:
    """Trainer for LanguageHead."""

    def __init__(
        self,
        config: LanguageTrainingConfig,
        whisper_model: WhisperMLX,
        language_head: LanguageHead,
        class_counts: dict[int, int] | None = None,
    ):
        self.config = config
        self.whisper_model = whisper_model
        self.language_head = language_head

        # Optimizer
        self.optimizer = optim.AdamW(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Class weights for imbalanced data
        if config.use_class_weights and class_counts:
            total = sum(class_counts.values())
            weights = []
            for i in range(config.num_languages):
                count = class_counts.get(i, 1)
                # Inverse frequency weighting with smoothing
                weight = total / (config.num_languages * count)
                weights.append(min(weight, 5.0))  # Cap weight
            self.class_weights = mx.array(weights, dtype=mx.float32)
        else:
            self.class_weights = None

        # Training state
        self.step = 0
        self.epoch = 0
        self.best_loss = float("inf")
        self.best_acc = 0.0

        # Encoder caching
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

    def _encode_audio(self, audio: np.ndarray) -> tuple[mx.array, int]:
        """Encode audio through frozen Whisper encoder."""
        mel = log_mel_spectrogram(audio)
        actual_mel_frames = mel.shape[0]

        # Pad/trim to 30s (3000 frames)
        target_frames = 3000
        if mel.shape[0] < target_frames:
            pad_len = target_frames - mel.shape[0]
            mel = np.pad(mel, ((0, pad_len), (0, 0)), mode='constant')
        else:
            mel = mel[:target_frames]
            actual_mel_frames = target_frames

        mel_mx = mx.array(mel)[None, ...]  # (1, 3000, n_mels)
        encoder_output = self.whisper_model.embed_audio(mel_mx)
        mx.eval(encoder_output)

        # Actual encoder frames = mel frames // 2
        actual_frames = min(actual_mel_frames // 2, 1500)

        return encoder_output[0], actual_frames

    def _encode_audio_cached(
        self, audio: np.ndarray, audio_path: str,
    ) -> tuple[mx.array, int]:
        """Encode with caching support."""
        if self.encoder_cache is not None:
            cached = self.encoder_cache.load(audio_path)
            if cached is not None:
                return cached

        enc_out, actual_frames = self._encode_audio(audio)

        if self.encoder_cache is not None:
            self.encoder_cache.save(audio_path, enc_out, actual_frames)

        return enc_out, actual_frames

    def _prepare_batch(
        self,
        samples: list[LanguageSample],
    ) -> tuple[mx.array, mx.array]:
        """Prepare a batch for training."""
        batch_encoder_outputs = []
        labels = []

        for sample in samples:
            try:
                audio = load_audio(sample.audio_path)
            except Exception:
                # Skip bad audio
                continue

            # Trim to max length
            max_samples = int(self.config.max_audio_len * self.config.sample_rate)
            if len(audio) > max_samples:
                audio = audio[:max_samples]

            # Encode
            if self.encoder_cache is not None:
                enc_out, _ = self._encode_audio_cached(audio, sample.audio_path)
            else:
                enc_out, _ = self._encode_audio(audio)

            batch_encoder_outputs.append(enc_out)
            labels.append(sample.language_id)

        if not batch_encoder_outputs:
            return None, None

        encoder_outputs = mx.stack(batch_encoder_outputs)
        label_array = mx.array(labels)

        return encoder_outputs, label_array

    def train_step(self, batch_samples: list[LanguageSample]) -> float:
        """Single training step."""
        encoder_outputs, labels = self._prepare_batch(batch_samples)

        if encoder_outputs is None:
            return 0.0

        def loss_fn(head):
            # Get per-frame logits
            logits = head(encoder_outputs)  # (batch, T, num_languages)

            # Mean pool over time for utterance-level classification
            logits_pooled = mx.mean(logits, axis=1)  # (batch, num_languages)

            # Cross-entropy loss with optional class weights
            log_probs = logits_pooled - mx.logsumexp(logits_pooled, axis=-1, keepdims=True)

            if self.class_weights is not None:
                # Weighted cross-entropy
                weights = self.class_weights[labels]
                loss = -mx.sum(weights * log_probs[mx.arange(len(labels)), labels])
                loss = loss / mx.sum(weights)
            else:
                loss = -mx.mean(log_probs[mx.arange(len(labels)), labels])

            return loss

        loss, grads = mx.value_and_grad(loss_fn)(self.language_head)
        self.optimizer.update(self.language_head, grads)
        mx.eval(self.language_head.parameters())

        return float(loss)

    def train(
        self,
        train_samples: list[LanguageSample],
        val_samples: list[LanguageSample],
    ):
        """Main training loop."""
        steps_per_epoch = (len(train_samples) + self.config.batch_size - 1) // self.config.batch_size

        self.log("Starting language head training")
        self.log(f"Epochs: {self.config.epochs}")
        self.log(f"Train samples: {len(train_samples)}")
        self.log(f"Val samples: {len(val_samples)}")
        self.log(f"Languages: {self.config.num_languages}")
        self.log(f"Steps per epoch: {steps_per_epoch}")

        # Print class distribution
        train_counts = {}
        for s in train_samples:
            train_counts[s.language] = train_counts.get(s.language, 0) + 1
        self.log(f"Train distribution: {train_counts}")

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
                loss = self.train_step(batch)
                epoch_losses.append(loss)
                self.step += 1

                # Logging
                if self.step % self.config.log_interval == 0:
                    avg_loss = np.mean(epoch_losses[-self.config.log_interval:])
                    self.log(f"  Step {self.step}: loss={avg_loss:.4f}")

                # Save checkpoint
                if self.step % self.config.save_interval == 0:
                    self._save_checkpoint(f"step_{self.step}.npz")

                # Validation
                if self.step % self.config.eval_interval == 0 and val_samples:
                    val_loss, val_acc, per_lang_acc = self._validate(val_samples)
                    self.log(f"  Val loss: {val_loss:.4f}, accuracy: {val_acc:.2%}")
                    self.log(f"  Per-language: {per_lang_acc}")

                    if val_acc > self.best_acc:
                        self.best_acc = val_acc
                        self.best_loss = val_loss
                        self._save_checkpoint("best.npz")
                        self.log(f"  New best model saved! (acc={val_acc:.2%})")

                clear_memory()

            # Epoch summary
            if epoch_losses:
                avg_epoch_loss = np.mean(epoch_losses)
                self.log(f"Epoch {epoch + 1}/{self.config.epochs}: loss={avg_epoch_loss:.4f}")

        # Save final
        self._save_checkpoint("final.npz")
        self.log("Training complete!")

    def _validate(
        self,
        val_samples: list[LanguageSample],
    ) -> tuple[float, float, dict[str, float]]:
        """Run validation."""
        # Set to eval mode to disable dropout
        self.language_head.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        # Per-language tracking
        per_lang_correct = {}
        per_lang_total = {}

        for batch_start in range(0, len(val_samples), self.config.batch_size):
            batch_end = min(batch_start + self.config.batch_size, len(val_samples))
            batch = val_samples[batch_start:batch_end]

            encoder_outputs, labels = self._prepare_batch(batch)

            if encoder_outputs is None:
                continue

            # Forward pass
            logits = self.language_head(encoder_outputs)
            logits_pooled = mx.mean(logits, axis=1)

            # Loss
            log_probs = logits_pooled - mx.logsumexp(logits_pooled, axis=-1, keepdims=True)
            loss = -mx.mean(log_probs[mx.arange(len(labels)), labels])
            total_loss += float(loss) * len(batch)

            # Accuracy
            preds = mx.argmax(logits_pooled, axis=-1)
            mx.eval(preds)

            for i, sample in enumerate(batch):
                pred = int(preds[i])
                true = sample.language_id
                lang = sample.language

                if lang not in per_lang_correct:
                    per_lang_correct[lang] = 0
                    per_lang_total[lang] = 0

                per_lang_total[lang] += 1
                if pred == true:
                    per_lang_correct[lang] += 1
                    correct += 1
                total += 1

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)

        # Per-language accuracy
        per_lang_acc = {}
        for lang in per_lang_total:
            if per_lang_total[lang] > 0:
                per_lang_acc[lang] = per_lang_correct[lang] / per_lang_total[lang]

        # Restore training mode
        self.language_head.train()

        return avg_loss, accuracy, per_lang_acc

    def _save_checkpoint(self, filename: str):
        """Save head weights."""
        from mlx.utils import tree_flatten

        weights = {}
        for k, v in tree_flatten(self.language_head.parameters()):
            weights[f"language.{k}"] = v

        weights["_step"] = mx.array(self.step)
        weights["_epoch"] = mx.array(self.epoch)

        save_path = self.output_dir / filename
        mx.savez(str(save_path), **weights)
        self.log(f"Saved checkpoint: {save_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint and resume training state."""
        self.log(f"Loading checkpoint: {checkpoint_path}")

        data = mx.load(checkpoint_path)

        # Restore step and epoch
        if "_step" in data:
            self.step = int(data["_step"])
            del data["_step"]
        if "_epoch" in data:
            self.epoch = int(data["_epoch"])
            del data["_epoch"]

        # Restore model weights
        head_weights = []
        prefix = "language."
        for k, v in data.items():
            if k.startswith(prefix):
                head_weights.append((k[len(prefix):], v))

        if head_weights:
            self.language_head.load_weights(head_weights)
            mx.eval(self.language_head.parameters())

        self.log(f"Resumed from step {self.step}, epoch {self.epoch}")


def main():
    parser = argparse.ArgumentParser(description="Language Head Training")
    parser.add_argument("--output-dir", type=str, default="checkpoints/language_head_v1",
                        help="Output directory")
    parser.add_argument("--commonvoice-dir", type=str,
                        default="data/commonvoice/cv-corpus-24.0-2025-12-05",
                        help="CommonVoice directory")
    parser.add_argument("--openslr-dir", type=str, default="data/openslr",
                        help="OpenSLR directory")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max-samples", type=int, default=5000,
                        help="Max samples per language")
    parser.add_argument("--model", type=str, default="mlx-community/whisper-large-v3-mlx",
                        help="Whisper model")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--encoder-cache-dir", type=str,
                        help="Directory for cached encoder outputs")

    args = parser.parse_args()

    config = LanguageTrainingConfig(
        commonvoice_dir=args.commonvoice_dir,
        openslr_dir=args.openslr_dir,
        output_dir=args.output_dir,
        whisper_model=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_samples_per_language=args.max_samples,
        encoder_cache_dir=args.encoder_cache_dir,
    )

    print("=" * 70)
    print("Language Head Training for Whisper")
    print("=" * 70)
    print(f"Output: {config.output_dir}")
    print(f"Model: {config.whisper_model}")
    if config.encoder_cache_dir:
        print(f"Encoder cache: {config.encoder_cache_dir}")
    print()

    # Load Whisper model
    print("1. Loading Whisper model...")
    whisper_model = WhisperMLX.from_pretrained(config.whisper_model)
    d_model = whisper_model.config.n_audio_state
    config.d_model = d_model
    print(f"   d_model={d_model}")

    # Create head
    print("2. Creating LanguageHead...")
    language_head = LanguageHead(
        d_model=d_model,
        num_languages=config.num_languages,
        hidden_dim=config.hidden_dim,
    )

    from mlx.utils import tree_flatten
    n_params = sum(p.size for _, p in tree_flatten(language_head.parameters()))
    print(f"   Parameters: {n_params / 1e3:.1f}K")

    # Load datasets
    all_samples = []

    # CommonVoice languages (en, zh, ja, hi)
    print("3. Loading CommonVoice data...")
    cv_languages = ["en", "zh", "ja", "hi"]
    if Path(config.commonvoice_dir).exists():
        cv_dataset = CommonVoiceDataset(
            data_dir=config.commonvoice_dir,
            languages=cv_languages,
            max_samples_per_language=config.max_samples_per_language,
            max_audio_len=config.max_audio_len,
        )
        all_samples.extend(cv_dataset.get_samples())
    else:
        print(f"   CommonVoice not found at {config.commonvoice_dir}")

    # OpenSLR languages (ko, ru, de, es, fr)
    print("4. Loading OpenSLR data...")
    slr_languages = ["ko", "ru", "de", "es", "fr"]
    if Path(config.openslr_dir).exists():
        slr_dataset = OpenSLRDataset(
            data_dir=config.openslr_dir,
            languages=slr_languages,
            max_samples_per_language=config.max_samples_per_language,
            max_audio_len=config.max_audio_len,
        )
        all_samples.extend(slr_dataset.get_samples())
    else:
        print(f"   OpenSLR not found at {config.openslr_dir}")

    if not all_samples:
        print("ERROR: No training data found!")
        return

    # Split train/val (90/10)
    rng = np.random.default_rng(42)
    rng.shuffle(all_samples)
    val_size = int(len(all_samples) * 0.1)
    val_samples = all_samples[:val_size]
    train_samples = all_samples[val_size:]

    print(f"\nTotal samples: {len(all_samples)}")
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    # Compute class counts
    class_counts = {}
    for s in train_samples:
        class_counts[s.language_id] = class_counts.get(s.language_id, 0) + 1
    print(f"Class counts: {class_counts}")

    # Create trainer
    trainer = LanguageTrainer(
        config=config,
        whisper_model=whisper_model,
        language_head=language_head,
        class_counts=class_counts,
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)
    trainer.train(train_samples, val_samples)


if __name__ == "__main__":
    main()
