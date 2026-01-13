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
Speaker embedding training data loaders.

Supports CN-Celeb, LibriSpeech, and VoxCeleb datasets for
speaker verification training with AAM-Softmax.
"""

import logging
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from .dataloader import compute_fbank_features, load_audio_file

logger = logging.getLogger(__name__)


@dataclass
class SpeakerSample:
    """Single speaker audio sample."""

    audio_path: str
    speaker_id: str
    speaker_idx: int  # Numerical index for classification
    duration: float | None = None
    genre: str | None = None  # For CN-Celeb genre info


class SpeakerDataset:
    """
    Base class for speaker verification datasets.

    Provides speaker-indexed samples for AAM-Softmax training.
    """

    def __init__(
        self,
        root_dir: str,
        min_duration: float = 2.0,
        max_duration: float = 10.0,
        sample_rate: int = 16000,
    ):
        """
        Initialize speaker dataset.

        Args:
            root_dir: Root directory containing dataset.
            min_duration: Minimum audio duration in seconds.
            max_duration: Maximum audio duration in seconds.
            sample_rate: Target sample rate.
        """
        self.root_dir = Path(root_dir)
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.sample_rate = sample_rate

        self.samples: list[SpeakerSample] = []
        self.speaker_to_idx: dict[str, int] = {}
        self.idx_to_speaker: dict[int, str] = {}

    @property
    def num_speakers(self) -> int:
        """Number of unique speakers in the dataset."""
        return len(self.speaker_to_idx)

    def _register_speaker(self, speaker_id: str) -> int:
        """Register a speaker and return their index."""
        if speaker_id not in self.speaker_to_idx:
            idx = len(self.speaker_to_idx)
            self.speaker_to_idx[speaker_id] = idx
            self.idx_to_speaker[idx] = speaker_id
        return self.speaker_to_idx[speaker_id]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, mx.array]:
        """Get a single sample."""
        sample = self.samples[idx]

        # Load audio
        audio = load_audio_file(sample.audio_path, target_sr=self.sample_rate)

        # Random crop to max_duration
        max_samples = int(self.max_duration * self.sample_rate)
        if len(audio) > max_samples:
            start = random.randint(0, len(audio) - max_samples)
            audio = audio[start:start + max_samples]

        # Compute features
        features = compute_fbank_features(audio, sample_rate=self.sample_rate)

        return {
            "features": mx.array(features),
            "feature_lengths": mx.array([features.shape[0]]),
            "speaker_idx": mx.array([sample.speaker_idx]),
        }


class CNCelebDataset(SpeakerDataset):
    """
    CN-Celeb dataset loader for Chinese speaker verification.

    Dataset structure:
        data/id***/<genre>-<session>-<utter>.flac

    997 speakers, 126,532 utterances, 271.72 hours.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",  # "train", "dev", "eval"
        min_duration: float = 2.0,
        max_duration: float = 10.0,
        sample_rate: int = 16000,
    ):
        """
        Initialize CN-Celeb dataset.

        Args:
            root_dir: Root directory (containing CN-Celeb_flac/).
            split: Dataset split ("train" uses all data except dev/eval).
            min_duration: Minimum audio duration.
            max_duration: Maximum audio duration.
            sample_rate: Target sample rate.
        """
        super().__init__(root_dir, min_duration, max_duration, sample_rate)
        self.split = split
        self._load_samples()

    def _load_samples(self) -> None:
        """Load all samples from CN-Celeb."""
        # Find the CN-Celeb_flac directory
        cnceleb_dir = self.root_dir
        if (self.root_dir / "CN-Celeb_flac").exists():
            cnceleb_dir = self.root_dir / "CN-Celeb_flac"

        data_dir = cnceleb_dir / "data"
        if not data_dir.exists():
            raise ValueError(f"CN-Celeb data directory not found: {data_dir}")

        # Load speaker lists for dev/eval splits
        dev_speakers = set()
        eval_speakers = set()

        dev_dir = cnceleb_dir / "dev"
        if dev_dir.exists():
            for f in dev_dir.iterdir():
                if f.is_dir():
                    dev_speakers.add(f.name)

        eval_dir = cnceleb_dir / "eval"
        if eval_dir.exists():
            for f in eval_dir.iterdir():
                if f.is_dir():
                    eval_speakers.add(f.name)

        # Iterate over speaker directories
        for speaker_dir in sorted(data_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue

            speaker_id = speaker_dir.name

            # Filter by split
            if self.split == "dev" and speaker_id not in dev_speakers:
                continue
            if self.split == "eval" and speaker_id not in eval_speakers:
                continue
            if self.split == "train":
                if speaker_id in dev_speakers or speaker_id in eval_speakers:
                    continue

            # Register speaker
            speaker_idx = self._register_speaker(speaker_id)

            # Load utterances
            for audio_file in speaker_dir.glob("*.flac"):
                # Parse filename for genre: <genre>-<session>-<utter>.flac
                parts = audio_file.stem.split("-")
                genre = parts[0] if len(parts) >= 3 else None

                self.samples.append(
                    SpeakerSample(
                        audio_path=str(audio_file),
                        speaker_id=speaker_id,
                        speaker_idx=speaker_idx,
                        genre=genre,
                    ),
                )

            # Also check for .wav files
            for audio_file in speaker_dir.glob("*.wav"):
                parts = audio_file.stem.split("-")
                genre = parts[0] if len(parts) >= 3 else None

                self.samples.append(
                    SpeakerSample(
                        audio_path=str(audio_file),
                        speaker_id=speaker_id,
                        speaker_idx=speaker_idx,
                        genre=genre,
                    ),
                )


class LibriSpeechSpeakerDataset(SpeakerDataset):
    """
    LibriSpeech dataset for speaker verification training.

    Uses speaker IDs from LibriSpeech for auxiliary training data.
    """

    def __init__(
        self,
        root_dir: str,
        splits: list[str] = None,
        min_duration: float = 2.0,
        max_duration: float = 10.0,
        sample_rate: int = 16000,
    ):
        """
        Initialize LibriSpeech speaker dataset.

        Args:
            root_dir: Root directory containing LibriSpeech.
            splits: List of splits to use (default: train-clean-100, train-clean-360).
            min_duration: Minimum audio duration.
            max_duration: Maximum audio duration.
            sample_rate: Target sample rate.
        """
        super().__init__(root_dir, min_duration, max_duration, sample_rate)
        self.splits = splits or ["train-clean-100", "train-clean-360"]
        self._load_samples()

    def _load_samples(self) -> None:
        """Load all samples from LibriSpeech."""
        for split in self.splits:
            split_dir = self.root_dir / split

            if not split_dir.exists():
                print(f"Warning: Split directory not found: {split_dir}")
                continue

            # LibriSpeech structure: split/speaker_id/chapter_id/*.flac
            for speaker_dir in split_dir.iterdir():
                if not speaker_dir.is_dir():
                    continue

                speaker_id = f"libri_{speaker_dir.name}"
                speaker_idx = self._register_speaker(speaker_id)

                for chapter_dir in speaker_dir.iterdir():
                    if not chapter_dir.is_dir():
                        continue

                    for audio_file in chapter_dir.glob("*.flac"):
                        self.samples.append(
                            SpeakerSample(
                                audio_path=str(audio_file),
                                speaker_id=speaker_id,
                                speaker_idx=speaker_idx,
                            ),
                        )


class CombinedSpeakerDataset(SpeakerDataset):
    """
    Combined dataset from multiple speaker datasets.

    Merges speaker indices across datasets for joint training.
    """

    def __init__(
        self,
        datasets: list[SpeakerDataset],
    ):
        """
        Initialize combined dataset.

        Args:
            datasets: List of speaker datasets to combine.
        """
        # Use first dataset's parameters
        if not datasets:
            raise ValueError("At least one dataset required")

        first = datasets[0]
        super().__init__(
            str(first.root_dir),
            first.min_duration,
            first.max_duration,
            first.sample_rate,
        )

        self._combine_datasets(datasets)

    def _combine_datasets(self, datasets: list[SpeakerDataset]) -> None:
        """Combine samples from all datasets with unified speaker indices."""
        for dataset_idx, dataset in enumerate(datasets):
            for sample in dataset.samples:
                # Create globally unique speaker ID using dataset index
                global_speaker_id = f"{dataset.__class__.__name__}_{dataset_idx}_{sample.speaker_id}"
                speaker_idx = self._register_speaker(global_speaker_id)

                self.samples.append(
                    SpeakerSample(
                        audio_path=sample.audio_path,
                        speaker_id=global_speaker_id,
                        speaker_idx=speaker_idx,
                        duration=sample.duration,
                        genre=sample.genre,
                    ),
                )


class SpeakerDataLoader:
    """
    Data loader for speaker embedding training.

    Handles batching with speaker-balanced sampling.
    """

    def __init__(
        self,
        dataset: SpeakerDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        samples_per_speaker: int = 2,
        drop_last: bool = True,
    ):
        """
        Initialize speaker data loader.

        Args:
            dataset: Speaker dataset to load from.
            batch_size: Number of samples per batch.
            shuffle: Whether to shuffle data.
            samples_per_speaker: Samples per speaker per batch (for contrastive).
            drop_last: Whether to drop last incomplete batch.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.samples_per_speaker = samples_per_speaker
        self.drop_last = drop_last

        # Build speaker-to-samples index for balanced sampling
        self.speaker_samples: dict[int, list[int]] = {}
        for idx, sample in enumerate(dataset.samples):
            if sample.speaker_idx not in self.speaker_samples:
                self.speaker_samples[sample.speaker_idx] = []
            self.speaker_samples[sample.speaker_idx].append(idx)

    def __iter__(self) -> Iterator[dict[str, mx.array]]:
        """Iterate over batches."""
        indices = list(range(len(self.dataset)))

        if self.shuffle:
            random.shuffle(indices)

        batch = []
        skipped_count = 0
        skipped_samples = []  # Track first few for debugging

        for idx in indices:
            try:
                sample = self.dataset[idx]
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

    def __len__(self) -> int:
        """Approximate number of batches."""
        return len(self.dataset) // self.batch_size

    def _collate_batch(self, batch: list[dict]) -> dict[str, mx.array]:
        """Collate samples into a batch with padding."""
        # Get max feature length
        max_feat_len = max(s["features"].shape[0] for s in batch)

        features = []
        feature_lengths = []
        speaker_indices = []

        for sample in batch:
            # Pad features
            feat = sample["features"]
            pad_len = max_feat_len - feat.shape[0]
            if pad_len > 0:
                padding = mx.zeros((pad_len, feat.shape[1]))
                feat = mx.concatenate([feat, padding], axis=0)
            features.append(feat)
            feature_lengths.append(sample["feature_lengths"])
            speaker_indices.append(sample["speaker_idx"])

        return {
            "features": mx.stack(features),
            "feature_lengths": mx.concatenate(feature_lengths),
            "speaker_indices": mx.concatenate(speaker_indices),
        }


class VerificationTrialLoader:
    """
    Data loader for speaker verification evaluation.

    Loads trial pairs for computing EER.
    """

    def __init__(
        self,
        dataset: SpeakerDataset,
        num_positive_pairs: int = 1000,
        num_negative_pairs: int = 1000,
    ):
        """
        Initialize verification trial loader.

        Args:
            dataset: Speaker dataset.
            num_positive_pairs: Number of same-speaker pairs.
            num_negative_pairs: Number of different-speaker pairs.
        """
        self.dataset = dataset
        self.num_positive_pairs = num_positive_pairs
        self.num_negative_pairs = num_negative_pairs

        # Build speaker-to-samples index
        self.speaker_samples: dict[int, list[int]] = {}
        for idx, sample in enumerate(dataset.samples):
            if sample.speaker_idx not in self.speaker_samples:
                self.speaker_samples[sample.speaker_idx] = []
            self.speaker_samples[sample.speaker_idx].append(idx)

        # Filter speakers with at least 2 samples (for positive pairs)
        self.valid_speakers = [
            spk for spk, samples in self.speaker_samples.items()
            if len(samples) >= 2
        ]

    def generate_trials(self) -> list[tuple[int, int, int]]:
        """
        Generate trial pairs.

        Returns:
            List of (idx1, idx2, label) tuples.
            label=1 for same speaker, label=0 for different.
        """
        trials = []

        # Positive pairs (same speaker)
        for _ in range(self.num_positive_pairs):
            if not self.valid_speakers:
                break

            spk = random.choice(self.valid_speakers)
            samples = self.speaker_samples[spk]

            idx1, idx2 = random.sample(samples, 2)
            trials.append((idx1, idx2, 1))

        # Negative pairs (different speakers)
        speakers = list(self.speaker_samples.keys())
        for _ in range(self.num_negative_pairs):
            if len(speakers) < 2:
                break

            spk1, spk2 = random.sample(speakers, 2)
            idx1 = random.choice(self.speaker_samples[spk1])
            idx2 = random.choice(self.speaker_samples[spk2])
            trials.append((idx1, idx2, 0))

        random.shuffle(trials)
        return trials


def create_cnceleb_loader(
    root_dir: str,
    split: str = "train",
    batch_size: int = 32,
    shuffle: bool = True,
) -> SpeakerDataLoader:
    """
    Create a CN-Celeb speaker data loader.

    Args:
        root_dir: Root directory containing CN-Celeb.
        split: Dataset split ("train", "dev", "eval").
        batch_size: Batch size.
        shuffle: Whether to shuffle.

    Returns:
        Configured data loader.
    """
    dataset = CNCelebDataset(root_dir, split)
    return SpeakerDataLoader(dataset, batch_size, shuffle)


def create_librispeech_speaker_loader(
    root_dir: str,
    splits: list[str] = None,
    batch_size: int = 32,
    shuffle: bool = True,
) -> SpeakerDataLoader:
    """
    Create a LibriSpeech speaker data loader.

    Args:
        root_dir: Root directory containing LibriSpeech.
        splits: List of splits to use.
        batch_size: Batch size.
        shuffle: Whether to shuffle.

    Returns:
        Configured data loader.
    """
    dataset = LibriSpeechSpeakerDataset(root_dir, splits)
    return SpeakerDataLoader(dataset, batch_size, shuffle)
