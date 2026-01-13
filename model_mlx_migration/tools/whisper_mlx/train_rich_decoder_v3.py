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
RichDecoder v3 Training Script - Multi-Task ULTRA Approach.

Key improvements over v1:
1. Multi-task learning: Train emotion, para, language, phoneme, pitch simultaneously
2. Pre-trained initialization: Start from best checkpoints for each head
3. Head freezing: Keep excellent heads frozen (para 96.96%, lang 98.61%)
4. Higher LoRA rank: 32 vs 8 for more adaptation capacity
5. Weighted multi-task loss: Prioritize weak heads (emotion, phoneme)
6. Curriculum learning: Easy samples first, then hard cases

Architecture:
    encoder_out (frozen) -> decoder (frozen + LoRA rank=32) -> rich outputs

    Heads:
    - Emotion: 8 classes (init from v1: 82.34%)
    - Para: 50 classes (init from para_v3: 96.96%, FROZEN)
    - Language: 100 classes (init from lang: 98.61%, FROZEN)
    - Phoneme: 178 Misaki (init from Kokoro, optional)
    - Pitch: F0 regression (new)

Usage:
    python -m tools.whisper_mlx.train_rich_decoder_v3 \
        --output-dir checkpoints/rich_decoder_v3 \
        --init-emotion checkpoints/rich_decoder_v1/best.npz \
        --init-para checkpoints/paralinguistics_v3/best.npz \
        --freeze-para \
        --lora-rank 32 \
        --epochs 20

References:
    - RICHDECODER_V3_ULTRAPLAN.md
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


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class RichDecoderV3Config:
    """Configuration for RichDecoder v3 training."""

    # Data directories
    emotion_data_dir: str | None = None
    para_data_dir: str | None = None

    # Pre-trained checkpoint initialization
    init_emotion: str | None = None  # e.g., checkpoints/rich_decoder_v1/best.npz
    init_para: str | None = None     # e.g., checkpoints/paralinguistics_v3/best.npz
    init_language: str | None = None
    init_phoneme: str | None = None

    # Head freezing
    freeze_para: bool = False
    freeze_language: bool = False
    freeze_phoneme: bool = False

    # Output
    output_dir: str = "checkpoints/rich_decoder_v3"

    # Model
    whisper_model: str = "mlx-community/whisper-large-v3-mlx"

    # LoRA configuration - HIGHER than v1
    lora_rank: int = 32   # v1 was 8
    lora_alpha: int = 64  # v1 was 16
    lora_dropout: float = 0.0

    # Training
    epochs: int = 20
    batch_size: int = 4
    learning_rate: float = 5e-5
    warmup_steps: int = 1000
    weight_decay: float = 0.01

    # Multi-task loss weights
    # Higher weight for tasks we want to improve
    emotion_loss_weight: float = 2.0   # Priority: improve from 82%
    phoneme_loss_weight: float = 1.5   # Priority: improve PER
    pitch_loss_weight: float = 1.0     # New task
    # Lower weight for already-good tasks (regularization)
    para_loss_weight: float = 0.3      # 96.96% already great
    language_loss_weight: float = 0.3  # 98.61% already great
    deviation_loss_weight: float = 0.5

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

    # Curriculum learning
    use_curriculum: bool = False
    curriculum_start_epoch: int = 5  # Switch to hard samples after this


# =============================================================================
# Dataset Classes
# =============================================================================

@dataclass
class MultiTaskSample:
    """Sample for multi-task RichDecoder training."""

    audio_path: str
    audio_array: np.ndarray | None = None
    features_path: str | None = None  # Pre-extracted features (for pseudo-labels)

    # Labels (Optional - not all samples have all labels)
    emotion_id: int | None = None
    language_id: int | None = None
    para_id: int | None = None
    phoneme_seq: list[int] | None = None
    pitch_seq: np.ndarray | None = None

    # Text (for decoder)
    text: str | None = None

    # Source dataset (for curriculum)
    source: str = "unknown"
    difficulty: float = 0.5  # 0=easy, 1=hard
    is_pseudo_label: bool = False  # Whether this is a pseudo-labeled sample


class UnifiedEmotionDatasetV3:
    """Load unified emotion dataset with 8-class mapping."""

    # 8-class emotion mapping (simpler than v1's 34)
    EMOTION_MAP_8 = {
        'neutral': 0, 'happy': 1, 'sad': 2, 'angry': 3,
        'fear': 4, 'disgust': 5, 'surprise': 6, 'other': 7,
    }

    # Map extended emotions to 8-class
    EMOTION_REMAP = {
        'neutral': 'neutral', 'happy': 'happy', 'sad': 'sad', 'angry': 'angry',
        'fear': 'fear', 'disgust': 'disgust', 'surprise': 'surprise',
        'contempt': 'other', 'amused': 'happy', 'frustrated': 'angry',
        'annoyed': 'angry', 'confused': 'other', 'excited': 'happy',
        'bored': 'neutral', 'calm': 'neutral', 'anxious': 'fear',
    }

    LANGUAGE_MAP = {
        'en': 0, 'zh': 1, 'ja': 2, 'ko': 3, 'de': 4,
        'es': 5, 'fr': 6, 'ru': 7, 'hi': 8, 'pt': 9,
        'ar': 10, 'it': 11, 'nl': 12, 'pl': 13, 'tr': 14,
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
        self.samples: list[MultiTaskSample] = []

        self._load_dataset()
        print(f"Loaded {len(self.samples)} {split} samples from {data_dir}")

    def _load_dataset(self):
        """Load dataset from HuggingFace format."""
        from datasets import load_from_disk

        try:
            ds = load_from_disk(str(self.data_dir))
            split_ds = ds[self.split]

            for i in range(len(split_ds)):
                try:
                    item = split_ds[i]
                    sample = self._process_item(item)
                    if sample:
                        self.samples.append(sample)
                except Exception:
                    pass

                if (i + 1) % 1000 == 0:
                    print(f"  Processed {i + 1}/{len(split_ds)} samples")

        except Exception as e:
            print(f"Error loading dataset: {e}")

    def _process_item(self, item: dict) -> MultiTaskSample | None:
        """Process a single item."""
        # Get emotion label and map to 8-class
        emotion = item.get("emotion", "neutral").lower()
        emotion_remapped = self.EMOTION_REMAP.get(emotion, "other")
        emotion_id = self.EMOTION_MAP_8.get(emotion_remapped, 7)

        # Get language
        language = item.get("language", "en").lower()
        language_id = self.LANGUAGE_MAP.get(language, 0)

        # Get audio
        audio_data = item.get("audio")
        if audio_data is None:
            return None

        try:
            if isinstance(audio_data, dict):
                audio_array = audio_data.get("array")
                sample_rate = audio_data.get("sampling_rate", 16000)
            elif hasattr(audio_data, 'get_all_samples'):
                samples = audio_data.get_all_samples()
                audio_array = samples.data.numpy().squeeze()
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

        return MultiTaskSample(
            audio_path="__in_memory__",
            audio_array=np.array(audio_array, dtype=np.float32),
            emotion_id=emotion_id,
            language_id=language_id,
            source="emotion",
            difficulty=0.5,
        )

    def get_samples(self) -> list[MultiTaskSample]:
        return self.samples


class ManifestDataset:
    """Load dataset from JSON manifest files (V4-style format).

    This enables V3 to use the same data format as V4, with encoder cache support.
    """

    EMOTION_MAP_8 = {
        'neutral': 0, 'happy': 1, 'sad': 2, 'angry': 3,
        'fear': 4, 'disgust': 5, 'surprise': 6, 'other': 7,
    }

    PARA_MAP = {
        'speech': 0, 'filler': 1, 'laugh': 2, 'silence': 3,
        'breath': 4, 'yawn': 5, 'sigh': 6, 'cough': 7,
        'sneeze': 8, 'cry': 9, 'sing': 40, 'hum': 41,
    }

    LANGUAGE_MAP = {
        'en': 0, 'zh': 1, 'ja': 2, 'ko': 3, 'de': 4,
        'es': 5, 'fr': 6, 'ru': 7, 'hi': 8, 'pt': 9,
    }

    def __init__(self, manifest_path: str):
        """Load dataset from manifest file.

        Args:
            manifest_path: Path to JSON manifest file
        """
        import json
        self.manifest_path = Path(manifest_path)
        self.samples: list[MultiTaskSample] = []

        print(f"Loading manifest from: {manifest_path}")
        with open(manifest_path) as f:
            manifest = json.load(f)

        for item in manifest:
            sample = self._process_item(item)
            if sample:
                self.samples.append(sample)

        print(f"Loaded {len(self.samples)} samples from manifest")

    def _process_item(self, item: dict) -> MultiTaskSample | None:
        """Process a single manifest item."""
        audio_path = item.get("audio_path")
        features_path = item.get("features_path")  # Pre-extracted features
        is_pseudo_label = item.get("is_pseudo_label", False)

        # Need either audio_path or features_path
        if not audio_path and not features_path:
            return None

        # Map emotion
        emotion = item.get("emotion", "neutral").lower()
        emotion_id = self.EMOTION_MAP_8.get(emotion)
        if emotion_id is None:
            emotion_id = 7  # 'other'

        # Map para
        para = item.get("para", "speech").lower()
        para_id = self.PARA_MAP.get(para, 0)

        # Map language
        language = item.get("language", "en").lower()
        language_id = self.LANGUAGE_MAP.get(language, 0)

        return MultiTaskSample(
            audio_path=audio_path or "",  # Empty string if using features_path
            features_path=features_path,
            emotion_id=emotion_id,
            para_id=para_id,
            language_id=language_id,
            text=item.get("text", ""),
            source=item.get("source", "unknown"),
            is_pseudo_label=is_pseudo_label,
        )

    def get_samples(self) -> list[MultiTaskSample]:
        return self.samples


class CombinedMultiTaskDataset:
    """Combine multiple datasets for multi-task training."""

    # Hard emotions (high arousal, rare) get higher difficulty
    HARD_EMOTIONS = {3, 4, 5, 6}  # angry, fear, disgust, surprise
    # Singing paralinguistics classes (from ULTRAPLAN: classes 40-49)
    SINGING_PARA_CLASSES = set(range(40, 50))

    def __init__(
        self,
        datasets: list[Any],
        shuffle: bool = True,
        seed: int = 42,
        assign_difficulty: bool = True,
    ):
        self.samples: list[MultiTaskSample] = []

        for ds in datasets:
            self.samples.extend(ds.get_samples())

        # Assign difficulty scores for curriculum learning
        if assign_difficulty:
            self._assign_difficulty_scores()

        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(self.samples)

        print(f"Combined dataset: {len(self.samples)} total samples")

        # Count by source
        sources = {}
        for s in self.samples:
            sources[s.source] = sources.get(s.source, 0) + 1
        for src, cnt in sorted(sources.items()):
            print(f"  {src}: {cnt}")

        # Difficulty distribution
        easy = sum(1 for s in self.samples if s.difficulty <= 0.5)
        hard = sum(1 for s in self.samples if s.difficulty > 0.5)
        print(f"Difficulty: {easy} easy (<=0.5), {hard} hard (>0.5)")

    def _assign_difficulty_scores(self):
        """Assign difficulty scores based on sample characteristics.

        Hard cases (difficulty > 0.5):
        - High arousal emotions: angry, fear, disgust, surprise
        - Singing audio (para classes 40-49)
        - Rare classes / minority samples

        Easy cases (difficulty <= 0.5):
        - Common emotions: neutral, happy, sad
        - Speech (not singing)
        - Well-represented classes
        """
        for sample in self.samples:
            difficulty = 0.3  # Base difficulty

            # Emotion-based difficulty
            if sample.emotion_id is not None:
                if sample.emotion_id in self.HARD_EMOTIONS:
                    difficulty += 0.3  # High arousal emotions are harder

            # Para-based difficulty (singing detection)
            if sample.para_id is not None:
                if sample.para_id in self.SINGING_PARA_CLASSES:
                    difficulty += 0.4  # Singing is harder

            # Source-based difficulty
            if sample.source == "vocalset":
                difficulty += 0.3  # Singing dataset
            elif sample.source == "vocalsound":
                difficulty += 0.2  # Non-speech sounds

            # Clamp to [0, 1]
            sample.difficulty = min(1.0, max(0.0, difficulty))

    def get_samples(self) -> list[MultiTaskSample]:
        return self.samples

    def get_samples_by_difficulty(self, max_difficulty: float) -> list[MultiTaskSample]:
        """Get samples up to a difficulty threshold (for curriculum)."""
        return [s for s in self.samples if s.difficulty <= max_difficulty]


# =============================================================================
# Trainer
# =============================================================================

class RichDecoderV3Trainer:
    """Multi-task trainer for RichDecoder v3."""

    def __init__(
        self,
        config: RichDecoderV3Config,
        whisper_model: WhisperMLX,
        rich_decoder: RichDecoder,
        tokenizer: Any,
    ):
        self.config = config
        self.whisper_model = whisper_model
        self.rich_decoder = rich_decoder
        self.tokenizer = tokenizer

        # Optimizer
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

        # Track frozen heads
        self.frozen_heads = set()
        if config.freeze_para:
            self.frozen_heads.add("para")
        if config.freeze_language:
            self.frozen_heads.add("language")
        if config.freeze_phoneme:
            self.frozen_heads.add("phoneme")

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

        mel_mx = mx.array(mel)[None, ...]
        return self.whisper_model.embed_audio(mel_mx)


    def _prepare_batch(
        self,
        samples: list[MultiTaskSample],
    ) -> dict[str, mx.array]:
        """Prepare a batch for multi-task training."""
        batch_encoder_outputs = []
        batch_tokens = []
        emotion_labels = []
        language_labels = []
        para_labels = []

        # Masks for which samples have which labels
        has_emotion = []
        has_language = []
        has_para = []

        max_samples = int(self.config.max_audio_len * self.config.sample_rate)

        for sample in samples:
            encoder_output = None

            # First, try loading pre-extracted features (for pseudo-labels)
            if sample.features_path:
                try:
                    import numpy as np
                    data = np.load(sample.features_path)
                    # LibriSpeech features use 'encoder_features' key
                    features_key = 'encoder_features' if 'encoder_features' in data else 'encoder_output'
                    cached_features = mx.array(data[features_key].astype(np.float32))
                    encoder_output = cached_features[None, ...]  # Add batch dim
                except Exception as e:
                    print(f"Warning: Failed to load features from {sample.features_path}: {e}")
                    continue  # Skip this sample

            # Try encoder cache if available and no pre-extracted features
            if encoder_output is None and self.encoder_cache is not None and sample.audio_path:
                cached_result = self.encoder_cache.load(sample.audio_path)
                if cached_result is not None:
                    cached_features, _ = cached_result  # (features, seq_len)
                    encoder_output = cached_features[None, ...]  # Add batch dim

            # Fall back to computing encoder if not cached
            if encoder_output is None:
                if sample.audio_array is not None:
                    audio = sample.audio_array
                elif sample.audio_path:
                    audio = load_audio(sample.audio_path)
                else:
                    continue  # Skip - no audio source

                if len(audio) > max_samples:
                    audio = audio[:max_samples]

                encoder_output = self._encode_audio(audio)

            batch_encoder_outputs.append(encoder_output[0])

            # Decoder input (SOT token)
            sot_token = self.tokenizer.sot
            batch_tokens.append([sot_token])

            # Labels with masks
            if sample.emotion_id is not None:
                emotion_labels.append(sample.emotion_id)
                has_emotion.append(True)
            else:
                emotion_labels.append(0)
                has_emotion.append(False)

            if sample.language_id is not None:
                language_labels.append(sample.language_id)
                has_language.append(True)
            else:
                language_labels.append(0)
                has_language.append(False)

            if sample.para_id is not None:
                para_labels.append(sample.para_id)
                has_para.append(True)
            else:
                para_labels.append(0)
                has_para.append(False)

        # Stack
        encoder_outputs = mx.stack(batch_encoder_outputs)

        # Pad tokens
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
            "para_labels": mx.array(para_labels),
            "has_emotion": mx.array(has_emotion),
            "has_language": mx.array(has_language),
            "has_para": mx.array(has_para),
        }

    def _compute_loss(
        self,
        rich_outputs: dict[str, mx.array],
        batch: dict[str, mx.array],
    ) -> tuple[mx.array, dict[str, float]]:
        """Compute weighted multi-task loss."""
        losses = {}
        total_loss = mx.array(0.0)

        # Emotion loss (if not frozen and samples have labels)
        if "emotion" not in self.frozen_heads:
            emotion_logits = rich_outputs["emotion"]
            emotion_pooled = mx.mean(emotion_logits, axis=1)
            emotion_loss = mx.mean(
                nn.losses.cross_entropy(emotion_pooled, batch["emotion_labels"])
                * batch["has_emotion"],
            )
            losses["emotion"] = float(emotion_loss)
            total_loss = total_loss + self.config.emotion_loss_weight * emotion_loss

        # Language loss
        if "language" not in self.frozen_heads:
            language_logits = rich_outputs["language"]
            language_pooled = mx.mean(language_logits, axis=1)
            language_loss = mx.mean(
                nn.losses.cross_entropy(language_pooled, batch["language_labels"])
                * batch["has_language"],
            )
            losses["language"] = float(language_loss)
            total_loss = total_loss + self.config.language_loss_weight * language_loss

        # Para loss (often frozen)
        if "para" not in self.frozen_heads and mx.any(batch["has_para"]):
            para_logits = rich_outputs["para"]
            para_pooled = mx.mean(para_logits, axis=1)
            para_loss = mx.mean(
                nn.losses.cross_entropy(para_pooled, batch["para_labels"])
                * batch["has_para"],
            )
            losses["para"] = float(para_loss)
            total_loss = total_loss + self.config.para_loss_weight * para_loss

        return total_loss, losses

    def train_step(self, batch_samples: list[MultiTaskSample]) -> dict[str, float]:
        """Single training step."""
        batch = self._prepare_batch(batch_samples)

        def loss_fn(decoder):
            outputs = decoder(
                x=batch["token_ids"],
                xa=batch["encoder_outputs"],
            )
            total_loss, _ = self._compute_loss(outputs, batch)
            return total_loss

        # Compute loss and gradients
        loss, grads = mx.value_and_grad(loss_fn)(self.rich_decoder)

        # Filter to trainable parameters only
        trainable_grads = {}
        for key, grad in grads.items():
            # LoRA adapters are always trainable
            if "lora" in key:
                trainable_grads[key] = grad
            # Prosody attention if enabled
            elif "prosody" in key:
                trainable_grads[key] = grad
            # Rich heads based on frozen status
            elif "rich" in key:
                # Check if this head is frozen
                head_frozen = False
                if "emotion" in key and "emotion" in self.frozen_heads:
                    head_frozen = True
                elif "para" in key and "para" in self.frozen_heads:
                    head_frozen = True
                elif "lang" in key and "language" in self.frozen_heads:
                    head_frozen = True

                if not head_frozen:
                    trainable_grads[key] = grad

        self.optimizer.update(self.rich_decoder, trainable_grads)
        mx.eval(self.rich_decoder.parameters())

        # Get loss breakdown
        outputs = self.rich_decoder(
            x=batch["token_ids"],
            xa=batch["encoder_outputs"],
        )
        _, losses = self._compute_loss(outputs, batch)
        losses["total"] = float(loss)

        return losses

    def train(
        self,
        train_samples: list[MultiTaskSample],
        val_samples: list[MultiTaskSample],
    ):
        """Main training loop with optional curriculum."""
        # Split samples by difficulty for curriculum learning
        if self.config.use_curriculum:
            easy_samples = [s for s in train_samples if s.difficulty <= 0.5]
            hard_samples = [s for s in train_samples if s.difficulty > 0.5]
            self.log(f"Curriculum mode: {len(easy_samples)} easy, {len(hard_samples)} hard samples")
        else:
            easy_samples = train_samples
            hard_samples = []

        steps_per_epoch = (len(train_samples) + self.config.batch_size - 1) // self.config.batch_size

        self.log("=" * 70)
        self.log("RichDecoder v3 Multi-Task Training")
        self.log("=" * 70)
        self.log(f"Epochs: {self.config.epochs}")
        self.log(f"Train samples: {len(train_samples)}")
        self.log(f"Val samples: {len(val_samples)}")
        self.log(f"Steps per epoch: {steps_per_epoch}")
        self.log(f"LoRA: rank={self.config.lora_rank}, alpha={self.config.lora_alpha}")
        self.log(f"Frozen heads: {self.frozen_heads or 'none'}")
        self.log(f"Loss weights: emo={self.config.emotion_loss_weight}, "
                 f"lang={self.config.language_loss_weight}, para={self.config.para_loss_weight}")
        if self.config.use_curriculum:
            self.log(f"Curriculum: enabled, hard samples after epoch {self.config.curriculum_start_epoch}")

        for epoch in range(self.config.epochs):
            self.epoch = epoch

            # Select samples based on curriculum stage
            if self.config.use_curriculum and epoch < self.config.curriculum_start_epoch:
                # Phase 1: Train on easy samples
                current_samples = easy_samples
                self.log(f"\nStarting epoch {epoch + 1}/{self.config.epochs} [CURRICULUM: easy samples]")
            elif self.config.use_curriculum and hard_samples:
                # Phase 2: Focus on hard samples (mix of hard + some easy for balance)
                # 70% hard, 30% easy for this phase
                n_hard = int(len(hard_samples) * 0.7)
                n_easy = int(len(easy_samples) * 0.3)
                rng = np.random.default_rng(42 + epoch)
                hard_subset = list(rng.choice(len(hard_samples), min(n_hard, len(hard_samples)), replace=False))
                easy_subset = list(rng.choice(len(easy_samples), min(n_easy, len(easy_samples)), replace=False))
                current_samples = [hard_samples[i] for i in hard_subset] + [easy_samples[i] for i in easy_subset]
                self.log(f"\nStarting epoch {epoch + 1}/{self.config.epochs} [CURRICULUM: hard focus ({len(current_samples)} samples)]")
            else:
                # Normal mode or no hard samples
                current_samples = train_samples
                self.log(f"\nStarting epoch {epoch + 1}/{self.config.epochs}")

            # Shuffle
            rng = np.random.default_rng(42 + epoch)
            rng.shuffle(current_samples)

            # Create batches from current samples
            batches = [
                current_samples[i:i + self.config.batch_size]
                for i in range(0, len(current_samples), self.config.batch_size)
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
                    loss_str = f"  Step {self.step}: loss={avg_loss:.4f}"
                    for k, v in sorted(losses.items()):
                        if k != "total":
                            loss_str += f" ({k[:3]}={v:.4f})"
                    self.log(loss_str)

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
                val_loss, val_metrics = self._validate(val_samples)
                self.log(f"  Val loss: {val_loss:.4f}")
                for metric, value in sorted(val_metrics.items()):
                    self.log(f"    {metric}: {value:.2%}")

                # Save best model based on emotion accuracy (primary metric)
                emotion_acc = val_metrics.get("emotion_acc", 0)
                if emotion_acc > self.best_acc:
                    self.best_acc = emotion_acc
                    self.best_loss = val_loss
                    self._save_checkpoint("best.npz")
                    self.log(f"  New best model! (emotion_acc={emotion_acc:.2%})")

        # Save final
        self._save_checkpoint("final.npz")
        self.log("\nTraining complete!")
        self.log(f"Best emotion accuracy: {self.best_acc:.2%}")

    def _validate(
        self,
        val_samples: list[MultiTaskSample],
    ) -> tuple[float, dict[str, float]]:
        """Run validation with per-task metrics."""
        # Set to eval mode to disable dropout
        self.rich_decoder.eval()

        total_loss = 0.0
        metrics = {
            "emotion_correct": 0,
            "emotion_total": 0,
            "language_correct": 0,
            "language_total": 0,
        }

        for batch_start in range(0, len(val_samples), self.config.batch_size):
            batch_end = min(batch_start + self.config.batch_size, len(val_samples))
            batch_samples = val_samples[batch_start:batch_end]

            batch = self._prepare_batch(batch_samples)

            outputs = self.rich_decoder(
                x=batch["token_ids"],
                xa=batch["encoder_outputs"],
            )

            loss, _ = self._compute_loss(outputs, batch)
            total_loss += float(loss) * len(batch_samples)

            # Per-task accuracy
            # Emotion
            emotion_logits = outputs["emotion"]
            emotion_pooled = mx.mean(emotion_logits, axis=1)
            emotion_preds = mx.argmax(emotion_pooled, axis=-1)

            for i, sample in enumerate(batch_samples):
                if sample.emotion_id is not None:
                    metrics["emotion_total"] += 1
                    if int(emotion_preds[i]) == sample.emotion_id:
                        metrics["emotion_correct"] += 1

                if sample.language_id is not None:
                    language_logits = outputs["language"]
                    language_pooled = mx.mean(language_logits, axis=1)
                    language_preds = mx.argmax(language_pooled, axis=-1)
                    metrics["language_total"] += 1
                    if int(language_preds[i]) == sample.language_id:
                        metrics["language_correct"] += 1

        # Compute metrics
        avg_loss = total_loss / max(len(val_samples), 1)
        result_metrics = {}

        if metrics["emotion_total"] > 0:
            result_metrics["emotion_acc"] = metrics["emotion_correct"] / metrics["emotion_total"]
        if metrics["language_total"] > 0:
            result_metrics["language_acc"] = metrics["language_correct"] / metrics["language_total"]

        # Restore training mode
        self.rich_decoder.train()

        return avg_loss, result_metrics

    def _save_checkpoint(self, filename: str):
        """Save trainable weights."""
        save_path = self.output_dir / filename
        self.rich_decoder.save_trainable(str(save_path))
        self.log(f"Saved checkpoint: {save_path}")

    def load_head_from_checkpoint(self, head_name: str, checkpoint_path: str):
        """Load a specific head from a checkpoint file."""
        weights = dict(mx.load(checkpoint_path))
        loaded = 0

        for key, value in weights.items():
            # Map checkpoint keys to rich_heads keys
            if head_name == "para" and key.startswith("paralinguistics."):
                # Map paralinguistics.X to rich_heads.para_X
                suffix = key[len("paralinguistics."):]
                target_key = f"rich_heads.para_{suffix}"
                try:
                    self._set_param(target_key, value)
                    loaded += 1
                except Exception as e:
                    print(f"  Warning: Could not load {key}: {e}")

            elif head_name == "emotion" and "rich_heads.emotion" in key:
                try:
                    self._set_param(key, value)
                    loaded += 1
                except Exception as e:
                    # Dimension mismatch - checkpoint may have different class count
                    self.log(f"  Warning: Could not load {key}: {e}")

        if loaded == 0:
            self.log(f"WARNING: No weights loaded for {head_name} head from {checkpoint_path}")
            self.log("  This likely means dimension mismatch (different class counts)")
            self.log(f"  The {head_name} head will train from random initialization")
        else:
            self.log(f"Loaded {loaded} weights for {head_name} head from {checkpoint_path}")

    def _set_param(self, key: str, value: mx.array):
        """Set a parameter by dotted key path."""
        parts = key.split(".")
        obj = self.rich_decoder
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="RichDecoder v3 Multi-Task Training")

    # Data
    parser.add_argument("--emotion-data", type=str, default="data/emotion/combined_emotion_hf",
                        help="Path to unified emotion dataset")
    parser.add_argument("--output-dir", type=str, default="checkpoints/rich_decoder_v3",
                        help="Output directory")

    # Pre-trained initialization
    parser.add_argument("--init-emotion", type=str,
                        help="Initialize emotion head from checkpoint")
    parser.add_argument("--init-para", type=str,
                        help="Initialize para head from checkpoint")
    parser.add_argument("--init-all", type=str,
                        help="Initialize all heads from a rich decoder checkpoint")

    # Freezing
    parser.add_argument("--freeze-para", action="store_true",
                        help="Freeze paralinguistics head")
    parser.add_argument("--freeze-language", action="store_true",
                        help="Freeze language head")

    # Model
    parser.add_argument("--model", type=str, default="mlx-community/whisper-large-v3-mlx",
                        help="Whisper model")
    parser.add_argument("--lora-rank", type=int, default=32,
                        help="LoRA rank (default: 32, v1 was 8)")
    parser.add_argument("--lora-alpha", type=int, default=64,
                        help="LoRA alpha (default: 64, v1 was 16)")

    # Training
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")

    # Loss weights
    parser.add_argument("--emotion-weight", type=float, default=2.0,
                        help="Emotion loss weight")
    parser.add_argument("--para-weight", type=float, default=0.3,
                        help="Para loss weight")
    parser.add_argument("--language-weight", type=float, default=0.3,
                        help="Language loss weight")

    # Curriculum learning
    parser.add_argument("--curriculum", action="store_true",
                        help="Enable curriculum learning (easy samples first)")
    parser.add_argument("--curriculum-start-epoch", type=int, default=5,
                        help="Epoch to switch from easy to hard samples")

    # Manifest files (alternative to HuggingFace datasets)
    parser.add_argument("--train-manifest", type=str,
                        help="JSON manifest for training data (alternative to --emotion-data)")
    parser.add_argument("--val-manifest", type=str,
                        help="JSON manifest for validation data")

    # Encoder cache
    parser.add_argument("--encoder-cache", type=str,
                        help="Directory with pre-extracted encoder features")

    args = parser.parse_args()

    config = RichDecoderV3Config(
        emotion_data_dir=args.emotion_data,
        output_dir=args.output_dir,
        whisper_model=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        init_emotion=args.init_emotion,
        init_para=args.init_para,
        freeze_para=args.freeze_para,
        freeze_language=args.freeze_language,
        emotion_loss_weight=args.emotion_weight,
        para_loss_weight=args.para_weight,
        language_loss_weight=args.language_weight,
        use_curriculum=args.curriculum,
        curriculum_start_epoch=args.curriculum_start_epoch,
        encoder_cache_dir=args.encoder_cache,
    )

    # Store manifest paths for later use
    train_manifest_path = args.train_manifest
    val_manifest_path = args.val_manifest

    print("=" * 70)
    print("RichDecoder v3 Multi-Task Training")
    print("=" * 70)
    print(f"Output: {config.output_dir}")
    print(f"Model: {config.whisper_model}")
    print(f"LoRA: rank={config.lora_rank}, alpha={config.lora_alpha}")
    print(f"Freeze: para={config.freeze_para}, language={config.freeze_language}")
    if config.encoder_cache_dir:
        print(f"Encoder cache: {config.encoder_cache_dir}")
    if train_manifest_path:
        print("Data source: JSON manifests")
    else:
        print("Data source: HuggingFace datasets")
    print()

    # Load Whisper model
    print("1. Loading Whisper model...")
    whisper_model = WhisperMLX.from_pretrained(config.whisper_model)
    print(f"   Encoder: {whisper_model.config.n_audio_state}-dim")

    # Get tokenizer
    print("2. Loading tokenizer...")
    tokenizer = get_whisper_tokenizer("large-v3")

    # Create RichDecoder config with higher LoRA
    decoder_config = RichDecoderConfig(
        n_vocab=whisper_model.config.n_vocab,
        n_ctx=whisper_model.config.n_text_ctx,
        n_state=whisper_model.config.n_text_state,
        n_head=whisper_model.config.n_text_head,
        n_layer=whisper_model.config.n_text_layer,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        num_emotions=8,  # v3 uses 8-class
    )

    # Create RichDecoder
    print("3. Creating RichDecoder with LoRA rank=32...")
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

    # Use manifests if provided, otherwise use HuggingFace datasets
    if train_manifest_path:
        print("   Using manifest files:")
        print(f"     Train: {train_manifest_path}")
        print(f"     Val: {val_manifest_path}")
        train_dataset = ManifestDataset(train_manifest_path)
        train_samples = train_dataset.get_samples()
        if val_manifest_path:
            val_dataset = ManifestDataset(val_manifest_path)
            val_samples = val_dataset.get_samples()
        else:
            # Use 10% of training for validation
            split_idx = int(len(train_samples) * 0.9)
            val_samples = train_samples[split_idx:]
            train_samples = train_samples[:split_idx]
    else:
        train_dataset = UnifiedEmotionDatasetV3(
            data_dir=config.emotion_data_dir,
            split="train",
            max_audio_len=config.max_audio_len,
        )
        val_dataset = UnifiedEmotionDatasetV3(
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
    trainer = RichDecoderV3Trainer(
        config=config,
        whisper_model=whisper_model,
        rich_decoder=rich_decoder,
        tokenizer=tokenizer,
    )

    # Load pre-trained heads if specified
    if args.init_all:
        print(f"5. Loading all heads from: {args.init_all}")
        rich_decoder.load_trainable(args.init_all)
    else:
        if args.init_emotion:
            print(f"5a. Loading emotion head from: {args.init_emotion}")
            rich_decoder.load_trainable(args.init_emotion)
        if args.init_para:
            print(f"5b. Loading para head from: {args.init_para}")
            trainer.load_head_from_checkpoint("para", args.init_para)

    # Resume from checkpoint if specified
    if args.resume:
        print(f"6. Resuming from checkpoint: {args.resume}")
        rich_decoder.load_trainable(args.resume)

    # Train
    print("\n" + "=" * 70)
    print("Starting Multi-Task Training")
    print("=" * 70)
    trainer.train(train_samples, val_samples)


if __name__ == "__main__":
    main()
