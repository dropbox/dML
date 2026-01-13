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
Data loading utilities for Zipformer ASR training.

Supports LibriSpeech and other ASR datasets.
"""

import logging
import random
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)

# E1 fix: Import scipy.signal at module level instead of inside functions
try:
    import scipy.signal as scipy_signal
except ImportError:
    scipy_signal = None


@dataclass
class AudioSample:
    """Single audio sample with transcription."""

    audio_path: str
    text: str
    speaker_id: str | None = None
    duration: float | None = None


def load_audio_file(path: str, target_sr: int = 16000) -> np.ndarray:
    """
    Load audio file and resample to target sample rate.

    Args:
        path: Path to audio file.
        target_sr: Target sample rate (default 16kHz).

    Returns:
        Audio samples as numpy array.
    """
    try:
        import soundfile as sf

        audio, sr = sf.read(path)

        # Convert stereo to mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Resample if needed
        if sr != target_sr:
            # E1 fix: Use module-level scipy_signal import
            num_samples = int(len(audio) * target_sr / sr)
            audio = scipy_signal.resample(audio, num_samples)

        return audio.astype(np.float32)

    except ImportError:
        # Fallback for FLAC files without soundfile
        import subprocess

        result = subprocess.run(
            ["sox", path, "-t", "raw", "-r", str(target_sr), "-c", "1", "-e", "float", "-"],
            capture_output=True,
        )
        return np.frombuffer(result.stdout, dtype=np.float32)


def compute_fbank_features(
    audio: np.ndarray,
    sample_rate: int = 16000,
    n_mels: int = 80,
    frame_length_ms: float = 25.0,
    frame_shift_ms: float = 10.0,
) -> np.ndarray:
    """
    Compute log-mel filterbank features.

    Args:
        audio: Audio samples.
        sample_rate: Sample rate in Hz.
        n_mels: Number of mel filterbank channels.
        frame_length_ms: Frame length in milliseconds.
        frame_shift_ms: Frame shift in milliseconds.

    Returns:
        Log-mel features of shape (time, n_mels).
    """
    # E1 fix: Use module-level scipy_signal import
    # Convert to samples
    frame_length = int(sample_rate * frame_length_ms / 1000)
    frame_shift = int(sample_rate * frame_shift_ms / 1000)
    n_fft = 512

    # STFT
    _, _, Zxx = scipy_signal.stft(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=frame_length,
        noverlap=frame_length - frame_shift,
        nfft=n_fft,
    )

    # Power spectrum
    power_spectrum = np.abs(Zxx) ** 2

    # Mel filterbank
    mel_filterbank = _create_mel_filterbank(n_fft, sample_rate, n_mels)

    # Apply filterbank
    mel_features = np.dot(mel_filterbank, power_spectrum)

    # Log
    mel_features = np.log(mel_features + 1e-10)

    # Transpose to (time, n_mels)
    return mel_features.T.astype(np.float32)


def _create_mel_filterbank(
    n_fft: int,
    sample_rate: int,
    n_mels: int,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> np.ndarray:
    """Create mel filterbank matrix."""
    if f_max is None:
        f_max = sample_rate / 2

    # Mel scale conversion
    mel_min = 2595 * np.log10(1 + f_min / 700)
    mel_max = 2595 * np.log10(1 + f_max / 700)

    # Mel points
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)

    # FFT bins
    fft_bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    # Filterbank
    filterbank = np.zeros((n_mels, n_fft // 2 + 1))

    for i in range(n_mels):
        left = fft_bins[i]
        center = fft_bins[i + 1]
        right = fft_bins[i + 2]

        for j in range(left, center):
            if center > left:
                filterbank[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right > center:
                filterbank[i, j] = (right - j) / (right - center)

    return filterbank


class LibriSpeechDataset:
    """
    LibriSpeech dataset loader.

    Supports train-clean-100, train-clean-360, dev-clean, test-clean.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train-clean-100",
        tokenizer: Callable[[str], list[int]] | None = None,
    ):
        """
        Initialize LibriSpeech dataset.

        Args:
            root_dir: Root directory containing LibriSpeech data.
            split: Dataset split (train-clean-100, dev-clean, test-clean).
            tokenizer: Optional tokenizer function (text -> token IDs).
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.tokenizer = tokenizer or self._default_tokenizer

        self.samples: list[AudioSample] = []
        self._load_samples()

    def _default_tokenizer(self, text: str) -> list[int]:
        """Simple character tokenizer as fallback."""
        # Map characters to IDs (0 = blank, 1-26 = a-z, 27 = space, 28 = ')
        char_map = {" ": 27, "'": 28}
        for i, c in enumerate("abcdefghijklmnopqrstuvwxyz"):
            char_map[c] = i + 1

        tokens = []
        for c in text.lower():
            if c in char_map:
                tokens.append(char_map[c])
        return tokens

    def _load_samples(self) -> None:
        """Load all samples from the dataset."""
        split_dir = self.root_dir / self.split

        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")

        # LibriSpeech structure: split/speaker_id/chapter_id/*.flac
        for speaker_dir in split_dir.iterdir():
            if not speaker_dir.is_dir():
                continue

            speaker_id = speaker_dir.name

            for chapter_dir in speaker_dir.iterdir():
                if not chapter_dir.is_dir():
                    continue

                # Load transcripts
                transcript_file = chapter_dir / f"{speaker_id}-{chapter_dir.name}.trans.txt"
                if not transcript_file.exists():
                    continue

                transcripts = {}
                with open(transcript_file) as f:
                    for line in f:
                        parts = line.strip().split(" ", 1)
                        if len(parts) == 2:
                            transcripts[parts[0]] = parts[1]

                # Load audio files
                for audio_file in chapter_dir.glob("*.flac"):
                    utterance_id = audio_file.stem
                    if utterance_id in transcripts:
                        self.samples.append(
                            AudioSample(
                                audio_path=str(audio_file),
                                text=transcripts[utterance_id],
                                speaker_id=speaker_id,
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

        # Tokenize text
        tokens = self.tokenizer(sample.text)

        return {
            "features": mx.array(features),
            "feature_lengths": mx.array([features.shape[0]]),
            "targets": mx.array(tokens),
            "target_lengths": mx.array([len(tokens)]),
            "text": sample.text,
        }


class ASRDataLoader:
    """
    Data loader for ASR training.

    Handles batching, shuffling, and padding.
    """

    def __init__(
        self,
        dataset: LibriSpeechDataset,
        batch_size: int = 8,
        shuffle: bool = True,
        max_duration: float = 30.0,
        drop_last: bool = True,
    ):
        """
        Initialize data loader.

        Args:
            dataset: Dataset to load from.
            batch_size: Batch size.
            shuffle: Whether to shuffle data.
            max_duration: Maximum audio duration in seconds.
            drop_last: Whether to drop the last incomplete batch.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_duration = max_duration
        self.drop_last = drop_last

        # Frame rate (10ms per frame)
        self.max_frames = int(max_duration * 100)

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

                # Skip samples that are too long
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

    def _collate_batch(self, batch: list[dict]) -> dict[str, mx.array]:
        """Collate samples into a batch with padding."""
        # Get max lengths
        max_feat_len = max(s["features"].shape[0] for s in batch)
        max_target_len = max(s["targets"].shape[0] for s in batch)

        # Pad features
        features = []
        feature_lengths = []
        targets = []
        target_lengths = []

        for sample in batch:
            # Pad features
            feat = sample["features"]
            pad_len = max_feat_len - feat.shape[0]
            if pad_len > 0:
                padding = mx.zeros((pad_len, feat.shape[1]))
                feat = mx.concatenate([feat, padding], axis=0)
            features.append(feat)
            feature_lengths.append(sample["feature_lengths"])

            # Pad targets
            tgt = sample["targets"]
            pad_len = max_target_len - tgt.shape[0]
            if pad_len > 0:
                padding = mx.zeros((pad_len,), dtype=tgt.dtype)
                tgt = mx.concatenate([tgt, padding], axis=0)
            targets.append(tgt)
            target_lengths.append(sample["target_lengths"])

        return {
            "features": mx.stack(features),
            "feature_lengths": mx.concatenate(feature_lengths),
            "targets": mx.stack(targets),
            "target_lengths": mx.concatenate(target_lengths),
        }

    def __len__(self) -> int:
        """Approximate number of batches."""
        return len(self.dataset) // self.batch_size


def create_librispeech_loader(
    data_dir: str | None = None,
    root_dir: str | None = None,
    splits: list[str] | None = None,
    split: str = "train-clean-100",
    batch_size: int = 8,
    shuffle: bool = True,
    tokenizer: Callable | None = None,
    include_rich_labels: bool = False,
) -> ASRDataLoader:
    """
    Create a LibriSpeech data loader.

    Args:
        data_dir: Root directory containing LibriSpeech (preferred).
        root_dir: Alias for data_dir (backwards compatibility).
        splits: List of dataset splits to combine.
        split: Single dataset split (used if splits is None).
        batch_size: Batch size.
        shuffle: Whether to shuffle.
        tokenizer: Optional tokenizer function.
        include_rich_labels: Include rich audio labels (emotion, phoneme, etc.).
            Currently a no-op for LibriSpeech which doesn't have these labels.

    Returns:
        Configured data loader.
    """
    # Handle backwards compatibility
    directory = data_dir or root_dir
    if directory is None:
        raise ValueError("Either data_dir or root_dir must be provided")

    # Handle single split or multiple splits
    split_list = splits if splits is not None else [split]

    # Combine datasets from all splits
    all_samples = []
    for s in split_list:
        try:
            dataset = LibriSpeechDataset(directory, s, tokenizer)
            all_samples.extend(dataset.samples)
        except ValueError:
            # Skip missing splits
            pass

    if not all_samples:
        raise ValueError(f"No samples found in splits: {split_list}")

    # Create combined dataset
    combined_dataset = LibriSpeechDataset.__new__(LibriSpeechDataset)
    combined_dataset.root_dir = Path(directory)
    combined_dataset.split = ",".join(split_list)
    combined_dataset.tokenizer = tokenizer or combined_dataset._default_tokenizer
    combined_dataset.samples = all_samples

    return ASRDataLoader(combined_dataset, batch_size, shuffle)
