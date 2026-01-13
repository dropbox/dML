# Copyright 2024-2026 Andrew Yates
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
Rich audio dataset loaders for Phase 4 multihead training.

Supports loading labeled datasets for:
- Emotion (CREMA-D, RAVDESS, combined emotion)
- Pitch (VocalSet)
- Phoneme (LibriSpeech with alignments)
- Language (CommonVoice)
- Paralinguistics (VocalSound)
"""

import logging
import queue
import random
import threading
import wave
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)

# E1 fix: Import scipy.signal at module level instead of inside functions
try:
    import scipy.signal as scipy_signal
except ImportError:
    scipy_signal = None

from .dataloader import compute_fbank_features, load_audio_file

# Emotion label mappings
CREMA_D_EMOTIONS = {
    "ANG": 0,  # Anger
    "DIS": 1,  # Disgust
    "FEA": 2,  # Fear
    "HAP": 3,  # Happy
    "NEU": 4,  # Neutral
    "SAD": 5,  # Sad
}

RAVDESS_EMOTIONS = {
    "01": 4,  # Neutral
    "02": 4,  # Calm (map to neutral)
    "03": 3,  # Happy
    "04": 5,  # Sad
    "05": 0,  # Angry
    "06": 2,  # Fearful
    "07": 1,  # Disgust
    "08": 6,  # Surprised
}

# Extended emotion labels (8 classes)
EMOTION_LABELS = [
    "anger",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
    "other",
]


@dataclass
class RichAudioSample:
    """Audio sample with rich labels."""

    audio_path: str
    text: str | None = None
    emotion_label: int | None = None
    pitch_hz: float | None = None
    phonemes: list[int] | None = None
    language: str | None = None
    paralinguistic_label: int | None = None
    duration: float | None = None
    speaker_id: str | None = None


class CREMADDataset:
    """
    CREMA-D emotion recognition dataset.

    File naming: {ActorID}_{Statement}_{Emotion}_{Level}.wav
    Example: 1001_DFA_ANG_XX.wav
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        val_ratio: float = 0.1,
    ):
        """
        Initialize CREMA-D dataset.

        Args:
            root_dir: Path to CREMA-D audio files.
            split: "train" or "val".
            val_ratio: Ratio of samples to use for validation.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.val_ratio = val_ratio
        self.samples: list[RichAudioSample] = []
        self._load_samples()

    def _parse_filename(self, filename: str) -> dict | None:
        """Parse CREMA-D filename to extract metadata."""
        # Format: {ActorID}_{Statement}_{Emotion}_{Level}.wav
        parts = filename.replace(".wav", "").split("_")
        if len(parts) >= 4:
            actor_id, statement, emotion, level = parts[0], parts[1], parts[2], parts[3]
            if emotion in CREMA_D_EMOTIONS:
                return {
                    "actor_id": actor_id,
                    "statement": statement,
                    "emotion": emotion,
                    "level": level,
                    "emotion_label": CREMA_D_EMOTIONS[emotion],
                }
        return None

    def _load_samples(self) -> None:
        """Load all samples from the dataset."""
        all_files = list(self.root_dir.glob("*.wav"))

        # Parse all valid files
        valid_samples = []
        for audio_file in all_files:
            metadata = self._parse_filename(audio_file.name)
            if metadata:
                valid_samples.append(
                    RichAudioSample(
                        audio_path=str(audio_file),
                        emotion_label=metadata["emotion_label"],
                        speaker_id=metadata["actor_id"],
                    ),
                )

        # Split into train/val
        random.seed(42)  # Reproducible split
        random.shuffle(valid_samples)
        split_idx = int(len(valid_samples) * (1 - self.val_ratio))

        if self.split == "train":
            self.samples = valid_samples[:split_idx]
        else:
            self.samples = valid_samples[split_idx:]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, mx.array]:
        """Get a single sample."""
        sample = self.samples[idx]

        # Load audio
        audio = load_audio_file(sample.audio_path)

        # Compute features
        features = compute_fbank_features(audio)

        return {
            "features": mx.array(features),
            "feature_lengths": mx.array([features.shape[0]]),
            "emotion_label": mx.array([sample.emotion_label]),
        }


class VocalSetDataset:
    """
    VocalSet dataset for pitch estimation.

    Contains singing exercises with known pitch patterns.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        val_ratio: float = 0.1,
    ):
        """
        Initialize VocalSet dataset.

        Args:
            root_dir: Path to VocalSet FULL directory.
            split: "train" or "val".
            val_ratio: Ratio of samples for validation.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.val_ratio = val_ratio
        self.samples: list[RichAudioSample] = []
        self._load_samples()

    def _load_samples(self) -> None:
        """Load all samples from VocalSet."""
        # VocalSet structure: FULL/{singer}/.../{technique}/*.wav
        all_files = list(self.root_dir.rglob("*.wav"))

        valid_samples = []
        for audio_file in all_files:
            # Extract singer from path
            relative = audio_file.relative_to(self.root_dir)
            parts = relative.parts
            singer = parts[0] if parts else "unknown"

            valid_samples.append(
                RichAudioSample(
                    audio_path=str(audio_file),
                    speaker_id=singer,
                    # Pitch will be computed from audio during training
                    pitch_hz=None,
                ),
            )

        # Split train/val
        random.seed(42)
        random.shuffle(valid_samples)
        split_idx = int(len(valid_samples) * (1 - self.val_ratio))

        if self.split == "train":
            self.samples = valid_samples[:split_idx]
        else:
            self.samples = valid_samples[split_idx:]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, mx.array]:
        """Get a single sample with pitch estimation."""
        sample = self.samples[idx]

        # Load audio
        audio = load_audio_file(sample.audio_path)

        # Compute features
        features = compute_fbank_features(audio)

        # Estimate pitch using simple autocorrelation
        pitch_hz = self._estimate_pitch(audio)

        return {
            "features": mx.array(features),
            "feature_lengths": mx.array([features.shape[0]]),
            "pitch_hz": mx.array([pitch_hz]),
        }

    def _estimate_pitch(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        min_hz: float = 80.0,
        max_hz: float = 600.0,
    ) -> float:
        """
        Estimate fundamental frequency using autocorrelation.

        Args:
            audio: Audio samples.
            sample_rate: Sample rate in Hz.
            min_hz: Minimum frequency to detect.
            max_hz: Maximum frequency to detect.

        Returns:
            Estimated F0 in Hz, or 0 if unvoiced.
        """
        # Simple autocorrelation-based pitch detection
        # For production, use CREPE or PYIN

        frame_length = 2048
        if len(audio) < frame_length:
            return 0.0

        # Take a frame from the middle
        start = len(audio) // 2 - frame_length // 2
        frame = audio[start:start + frame_length]

        # Autocorrelation
        corr = np.correlate(frame, frame, mode="full")
        corr = corr[len(corr) // 2:]

        # Find peaks
        min_period = int(sample_rate / max_hz)
        max_period = int(sample_rate / min_hz)

        if max_period > len(corr):
            max_period = len(corr) - 1

        # Find the first peak after min_period
        search_range = corr[min_period:max_period]
        if len(search_range) == 0:
            return 0.0

        peak_idx = np.argmax(search_range) + min_period

        # Convert to frequency
        pitch_hz = sample_rate / peak_idx if peak_idx > 0 else 0.0

        return float(pitch_hz)


class CombinedEmotionDataset:
    """
    Combined emotion dataset using HuggingFace arrow format.

    Loads from pre-processed combined emotion dataset with embedded audio.
    Supports 6-class emotion labels: angry, disgust, fear, happy, neutral, sad.
    """

    # Map emotion strings to integer labels (matches CREMA-D ordering)
    EMOTION_TO_INT = {
        "angry": 0,      # Maps to ANG
        "anger": 0,      # Alternative spelling
        "disgust": 1,    # Maps to DIS
        "fear": 2,       # Maps to FEA
        "fearful": 2,    # Alternative spelling
        "happy": 3,      # Maps to HAP
        "happiness": 3,  # Alternative spelling
        "neutral": 4,    # Maps to NEU
        "sad": 5,        # Maps to SAD
        "sadness": 5,    # Alternative spelling
    }

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
    ):
        """
        Initialize from HuggingFace dataset.

        Args:
            root_dir: Path to combined_emotion_hf directory.
            split: "train" or "validation".
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.dataset = None
        self._load_dataset()

    def _load_dataset(self) -> None:
        """Load HuggingFace dataset."""
        from datasets import load_from_disk

        dataset = load_from_disk(str(self.root_dir))
        self.dataset = dataset[self.split] if self.split in dataset else dataset["train"]

    def __len__(self) -> int:
        return len(self.dataset) if self.dataset else 0

    def __getitem__(self, idx: int) -> dict[str, mx.array]:
        """Get a single sample with audio decoding."""
        item = self.dataset[idx]

        # Get audio from HuggingFace audio decoder
        audio_obj = item["audio"]

        # Decode audio using torchcodec
        audio_samples = audio_obj.get_all_samples()
        audio_data = audio_samples.data.numpy()  # [1, num_samples] tensor

        # Convert to mono if needed and flatten
        if audio_data.ndim > 1:
            audio_data = audio_data.squeeze(0)  # Remove channel dim

        # Resample if needed (target is 16kHz)
        if audio_samples.sample_rate != 16000:
            # E1 fix: Use module-level scipy_signal import
            audio_data = scipy_signal.resample(
                audio_data,
                int(len(audio_data) * 16000 / audio_samples.sample_rate),
            )

        # Compute features
        features = compute_fbank_features(audio_data.astype(np.float32))

        # Get emotion label
        emotion_str = item.get("emotion", "").lower()
        emotion_label = self.EMOTION_TO_INT.get(emotion_str, 4)  # Default to neutral

        return {
            "features": mx.array(features),
            "feature_lengths": mx.array([features.shape[0]]),
            "emotion_label": mx.array([emotion_label]),
        }


# VocalSound paralinguistics labels (6 classes)
VOCALSOUND_LABELS = {
    "laughter": 0,
    "sigh": 1,
    "cough": 2,
    "throat_clearing": 3,
    "sneeze": 4,
    "sniff": 5,
}

PARALINGUISTIC_LABELS = [
    "laughter",
    "sigh",
    "cough",
    "throat_clearing",
    "sneeze",
    "sniff",
]


class VocalSoundDataset:
    """
    VocalSound paralinguistics dataset.

    Contains 6 classes: laughter, sigh, cough, throat_clearing, sneeze, sniff.
    Uses manifest.json for labels and splits.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
    ):
        """
        Initialize VocalSound dataset.

        Args:
            root_dir: Path to vocalsound_labeled directory.
            split: "train" or "test".
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.samples: list[RichAudioSample] = []
        self._load_samples()

    def _load_samples(self) -> None:
        """Load samples from manifest.json."""
        import json

        manifest_path = self.root_dir / "manifest.json"
        if not manifest_path.exists():
            raise ValueError(f"Manifest not found: {manifest_path}")

        with open(manifest_path) as f:
            manifest = json.load(f)

        for item in manifest:
            if item.get("split") != self.split:
                continue

            audio_path = item.get("audio_path")
            label = item.get("label", "").lower()

            if audio_path and label in VOCALSOUND_LABELS:
                self.samples.append(
                    RichAudioSample(
                        audio_path=str(audio_path),
                        paralinguistic_label=VOCALSOUND_LABELS[label],
                    ),
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, mx.array]:
        """Get a single sample."""
        sample = self.samples[idx]

        # Load audio
        audio = load_audio_file(sample.audio_path)

        # Compute features
        features = compute_fbank_features(audio)

        return {
            "features": mx.array(features),
            "feature_lengths": mx.array([features.shape[0]]),
            "paralinguistic_label": mx.array([sample.paralinguistic_label]),
        }


# MELD emotion labels (7 classes from Friends TV dataset)
MELD_EMOTION_TO_INT = {
    "neutral": 4,    # Maps to our neutral
    "joy": 3,        # Maps to our happy
    "surprise": 6,   # Maps to our surprise
    "anger": 0,      # Maps to our anger
    "sadness": 5,    # Maps to our sad
    "disgust": 1,    # Maps to our disgust
    "fear": 2,       # Maps to our fear
}


class MELDDataset:
    """
    MELD (Multimodal EmotionLines Dataset) emotion recognition dataset.

    MELD contains audio clips from the Friends TV show with 7 emotion labels:
    neutral, joy, surprise, anger, sadness, disgust, fear.

    Audio files: audio_{split}/dia{dialogue_id}_utt{utterance_id}.wav
    Labels: {split}_sent_emo.csv
    """

    def __init__(
        self,
        root_dir: str = "data/emotion_punctuation/MELD.Raw",
        split: str = "train",
    ):
        """
        Initialize MELD dataset.

        Args:
            root_dir: Path to MELD.Raw directory.
            split: "train", "dev", or "test".
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.samples: list[RichAudioSample] = []
        self._load_samples()

    def _load_samples(self) -> None:
        """Load samples from CSV and match with audio files."""
        import csv

        # Map split names
        csv_name = f"{self.split}_sent_emo.csv"
        audio_dir_name = f"audio_{self.split}"

        csv_path = self.root_dir / csv_name
        audio_dir = self.root_dir / audio_dir_name

        if not csv_path.exists():
            raise ValueError(f"MELD CSV not found: {csv_path}")
        if not audio_dir.exists():
            raise ValueError(f"MELD audio dir not found: {audio_dir}")

        # Read CSV
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dialogue_id = row.get("Dialogue_ID", "")
                utterance_id = row.get("Utterance_ID", "")
                emotion = row.get("Emotion", "").lower()

                if not dialogue_id or not utterance_id:
                    continue

                # Construct audio filename
                audio_filename = f"dia{dialogue_id}_utt{utterance_id}.wav"
                audio_path = audio_dir / audio_filename

                if not audio_path.exists():
                    continue

                # Map emotion to label
                if emotion not in MELD_EMOTION_TO_INT:
                    continue

                emotion_label = MELD_EMOTION_TO_INT[emotion]

                self.samples.append(
                    RichAudioSample(
                        audio_path=str(audio_path),
                        emotion_label=emotion_label,
                        text=row.get("Utterance", ""),
                        speaker_id=row.get("Speaker", ""),
                    ),
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, mx.array]:
        """Get a single sample."""
        sample = self.samples[idx]

        # Load audio
        audio = load_audio_file(sample.audio_path)

        # Compute features
        features = compute_fbank_features(audio)

        return {
            "features": mx.array(features),
            "feature_lengths": mx.array([features.shape[0]]),
            "emotion_label": mx.array([sample.emotion_label]),
        }


class RichAudioDataLoader:
    """
    Data loader for rich audio training.

    Handles batching with variable-length sequences and multiple label types.
    """

    def __init__(
        self,
        dataset: Union[CREMADDataset, VocalSetDataset, CombinedEmotionDataset, "VocalSoundDataset"],
        batch_size: int = 16,
        shuffle: bool = True,
        max_duration: float = 30.0,
        drop_last: bool = True,
        prefetch_batches: int = 2,
    ):
        """
        Initialize rich audio data loader.

        Args:
            dataset: Dataset to load from.
            batch_size: Batch size.
            shuffle: Whether to shuffle.
            max_duration: Maximum audio duration in seconds.
            drop_last: Drop incomplete final batch.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_duration = max_duration
        self.drop_last = drop_last
        self.prefetch_batches = prefetch_batches

        # Frame rate (10ms per frame)
        self.max_frames = int(max_duration * 100)

        # M8 fix: Pre-filter long samples using duration metadata (avoid loading/computing features).
        self._eligible_indices = self._prefilter_indices()

    def _probe_wav_duration_seconds(self, path: str) -> float | None:
        try:
            with wave.open(path, "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
            if rate <= 0:
                return None
            return float(frames) / float(rate)
        except Exception:
            return None

    def _prefilter_indices(self) -> list[int]:
        if not hasattr(self.dataset, "samples"):
            return list(range(len(self.dataset)))

        samples = getattr(self.dataset, "samples", None)
        if not isinstance(samples, list):
            return list(range(len(self.dataset)))

        eligible: list[int] = []
        for i, sample in enumerate(samples):
            duration = getattr(sample, "duration", None)
            audio_path = getattr(sample, "audio_path", None)

            if duration is None and isinstance(audio_path, str):
                duration = self._probe_wav_duration_seconds(audio_path)
                if duration is not None:
                    try:
                        sample.duration = duration
                    except Exception:
                        pass

            if duration is not None and duration > self.max_duration:
                continue

            eligible.append(i)

        return eligible

    def _iter_batches(self) -> Iterator[dict[str, mx.array]]:
        """Iterate over batches."""
        indices = list(self._eligible_indices)

        if self.shuffle:
            random.shuffle(indices)

        batch = []
        skipped_count = 0
        skipped_samples = []  # Track first few for debugging

        for idx in indices:
            try:
                sample = self.dataset[idx]

                # Skip samples that are too long (safety net for datasets without duration metadata)
                if sample["features"].shape[0] > self.max_frames:
                    continue

                batch.append(sample)

                if len(batch) == self.batch_size:
                    yield self._collate_batch(batch)
                    batch = []

            except Exception as e:
                # Skip problematic samples with rate-limited logging
                skipped_count += 1
                if len(skipped_samples) < 5:
                    skipped_samples.append((idx, str(e)))
                continue

        # Handle last batch
        if batch and not self.drop_last:
            yield self._collate_batch(batch)

        # Log summary of skipped samples (once per epoch, not per sample)
        if skipped_count > 0:
            logger.warning(
                f"Skipped {skipped_count} samples this epoch. "
                f"Examples: {skipped_samples[:3]}",
            )

    def __iter__(self) -> Iterator[dict[str, mx.array]]:
        base_iter = self._iter_batches()
        if self.prefetch_batches and self.prefetch_batches > 0:
            return _PrefetchIterator(base_iter, max_prefetch=self.prefetch_batches)
        return base_iter

    def _collate_batch(self, batch: list[dict]) -> dict[str, mx.array]:
        """Collate samples into a batch with padding."""
        # Get max feature length
        max_feat_len = max(s["features"].shape[0] for s in batch)

        # Pad features
        features = []
        feature_lengths = []

        for sample in batch:
            feat = sample["features"]
            pad_len = max_feat_len - feat.shape[0]
            if pad_len > 0:
                padding = mx.zeros((pad_len, feat.shape[1]))
                feat = mx.concatenate([feat, padding], axis=0)
            features.append(feat)
            feature_lengths.append(sample["feature_lengths"])

        result = {
            "features": mx.stack(features),
            "feature_lengths": mx.concatenate(feature_lengths),
        }

        # Add label tensors if present
        if "emotion_label" in batch[0]:
            result["emotion_labels"] = mx.concatenate([s["emotion_label"] for s in batch])

        if "pitch_hz" in batch[0]:
            result["pitch_hz"] = mx.concatenate([s["pitch_hz"] for s in batch])

        if "paralinguistic_label" in batch[0]:
            result["paralinguistic_labels"] = mx.concatenate([s["paralinguistic_label"] for s in batch])

        return result

    def __len__(self) -> int:
        """Approximate number of batches."""
        return len(self._eligible_indices) // self.batch_size


class _PrefetchIterator:
    def __init__(self, iterator: Iterator, max_prefetch: int):
        self._it = iterator
        self._queue: queue.Queue = queue.Queue(maxsize=max_prefetch)
        self._sentinel = object()
        self._exc: BaseException | None = None
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        try:
            for item in self._it:
                self._queue.put(item)
        except BaseException as e:
            self._exc = e
        finally:
            self._queue.put(self._sentinel)

    def __iter__(self) -> "_PrefetchIterator":
        return self

    def __next__(self):
        item = self._queue.get()
        if item is self._sentinel:
            if self._exc is not None:
                raise self._exc
            raise StopIteration
        return item


def create_emotion_loader(
    data_dir: str = "data/emotion/crema-d",
    split: str = "train",
    batch_size: int = 16,
    shuffle: bool = True,
) -> RichAudioDataLoader:
    """
    Create an emotion dataset loader.

    Args:
        data_dir: Path to emotion dataset.
        split: "train" or "val".
        batch_size: Batch size.
        shuffle: Whether to shuffle.

    Returns:
        Configured data loader.
    """
    dataset = CREMADDataset(data_dir, split)
    return RichAudioDataLoader(dataset, batch_size, shuffle)


def create_pitch_loader(
    data_dir: str = "data/vocalset/FULL",
    split: str = "train",
    batch_size: int = 16,
    shuffle: bool = True,
) -> RichAudioDataLoader:
    """
    Create a pitch estimation dataset loader.

    Args:
        data_dir: Path to VocalSet FULL directory.
        split: "train" or "val".
        batch_size: Batch size.
        shuffle: Whether to shuffle.

    Returns:
        Configured data loader.
    """
    dataset = VocalSetDataset(data_dir, split)
    return RichAudioDataLoader(dataset, batch_size, shuffle)


def create_combined_emotion_loader(
    data_dir: str = "data/emotion/combined_emotion_hf",
    split: str = "train",
    batch_size: int = 16,
    shuffle: bool = True,
) -> RichAudioDataLoader:
    """
    Create a combined emotion dataset loader.

    Args:
        data_dir: Path to combined emotion HF dataset.
        split: "train" or "validation".
        batch_size: Batch size.
        shuffle: Whether to shuffle.

    Returns:
        Configured data loader.
    """
    dataset = CombinedEmotionDataset(data_dir, split)
    return RichAudioDataLoader(dataset, batch_size, shuffle)


def create_paralinguistics_loader(
    data_dir: str = "data/paralinguistics/vocalsound_labeled",
    split: str = "train",
    batch_size: int = 16,
    shuffle: bool = True,
) -> RichAudioDataLoader:
    """
    Create a paralinguistics dataset loader using VocalSound.

    Args:
        data_dir: Path to vocalsound_labeled directory.
        split: "train" or "test".
        batch_size: Batch size.
        shuffle: Whether to shuffle.

    Returns:
        Configured data loader.
    """
    dataset = VocalSoundDataset(data_dir, split)
    return RichAudioDataLoader(dataset, batch_size, shuffle)


def create_meld_loader(
    data_dir: str = "data/emotion_punctuation/MELD.Raw",
    split: str = "train",
    batch_size: int = 16,
    shuffle: bool = True,
) -> RichAudioDataLoader:
    """
    Create a MELD emotion dataset loader.

    MELD contains ~10K emotion-labeled clips from the Friends TV show.
    7 classes: neutral, joy, surprise, anger, sadness, disgust, fear.

    Args:
        data_dir: Path to MELD.Raw directory.
        split: "train", "dev", or "test".
        batch_size: Batch size.
        shuffle: Whether to shuffle.

    Returns:
        Configured data loader.
    """
    dataset = MELDDataset(data_dir, split)
    return RichAudioDataLoader(dataset, batch_size, shuffle)
