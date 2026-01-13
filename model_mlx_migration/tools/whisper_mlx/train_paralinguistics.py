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
Paralinguistics Training Script.

Trains ParalinguisticsHead for non-speech vocalization detection:
- laughter, cough, sneeze, sniff, sigh, throat_clear, etc.

Datasets:
- VocalSound (lmms-lab/vocalsound): 5446 samples of vocal sounds
- ESC-50: 240 relevant samples (breathing, coughing, laughing, sneezing, snoring)

Usage:
    python -m tools.whisper_mlx.train_paralinguistics \
        --output-dir checkpoints/paralinguistics \
        --epochs 5

References:
    - VocalSound: https://arxiv.org/abs/2205.03433
    - ESC-50: https://github.com/karolpiczak/ESC-50
"""

import argparse
import gc
import os
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np

# Import Whisper components
from .audio import get_audio_duration, load_audio, log_mel_spectrogram
from .encoder_cache import TrainingEncoderCache
from .model import WhisperMLX
from .multi_head import (
    PARALINGUISTICS_CLASSES,
    MultiHeadConfig,
    ParalinguisticsHead,
    compute_class_weights_from_counts,
    create_paralinguistics_class_weights,
    focal_loss,
    paralinguistics_loss,
)


def clear_memory():
    """Clear memory between batches."""
    if hasattr(mx, 'clear_cache'):
        mx.clear_cache()
    gc.collect()


@dataclass
class ParalinguisticsTrainingConfig:
    """Configuration for paralinguistics training."""

    # Data
    vocalsound_cache: str | None = None  # HuggingFace cache dir
    vocalsound_dir: str | None = None    # Local VocalSound directory with manifest.json
    esc50_dir: str | None = None  # ESC-50 directory

    # Output
    output_dir: str = "checkpoints/paralinguistics"

    # Model
    whisper_model: str = "mlx-community/whisper-large-v3-mlx"
    model_size: str = "large-v3"
    d_model: int = 1280

    # Training
    epochs: int = 5
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01

    # Loss configuration
    use_focal_loss: bool = True  # Use focal loss for hard example mining
    focal_gamma: float = 2.0     # Focal loss gamma parameter

    # Class weights for imbalanced classes
    use_class_weights: bool = True
    use_frequency_weights: bool = True  # Compute weights from actual class counts
    speech_weight: float = 0.5  # Down-weight speech (only if not use_frequency_weights)
    event_weight: float = 2.0   # Up-weight events (only if not use_frequency_weights)

    # Audio
    max_audio_len: float = 10.0  # seconds
    sample_rate: int = 16000

    # Logging
    log_interval: int = 50
    save_interval: int = 500

    # Encoder caching (~3x speedup by caching frozen encoder outputs)
    encoder_cache_dir: str | None = None

    # Length-sorted batching (~1.3x speedup by reducing padding waste)
    length_sorted_batching: bool = False
    bucket_size_multiplier: int = 100  # Bucket size = batch_size * multiplier


@dataclass
class ParalinguisticsSample:
    """Sample for paralinguistics training."""

    audio_path: str
    label: str  # Label string (e.g., "laughter", "cough")
    label_id: int  # Mapped to PARALINGUISTICS_CLASSES

    # In-memory audio (for HuggingFace datasets)
    audio_array: np.ndarray | None = None

    # Duration for length-sorted batching
    duration: float = 0.0


# VocalSound label mapping to PARALINGUISTICS_CLASSES (V2 - 50 classes)
# See multi_head.py for full class list
VOCALSOUND_MAP = {
    "Cough": 2,           # cough (V2 index)
    "Laughter": 1,        # laughter
    "Sigh": 3,            # sigh (V2 index)
    "Sneeze": 8,          # sneeze (V2 index - now its own class)
    "Sniff": 8,           # sniff -> sneeze category (similar sounds)
    "Throat clearing": 7, # throat_clear
}

# Local VocalSound label mapping (lowercase from manifest)
LOCAL_VOCALSOUND_MAP = {
    "cough": 2,           # cough (V2 index)
    "laughter": 1,        # laughter
    "sigh": 3,            # sigh (V2 index)
    "sneeze": 8,          # sneeze (V2 index - now its own class)
    "sniff": 8,           # sniff -> sneeze category
    "throat_clearing": 7, # throat_clear
}

# ESC-50 category mapping (category name -> PARALINGUISTICS_CLASSES V2)
# Now uses proper V2 indices with separate classes for all sounds
ESC50_MAP = {
    "breathing": 4,       # breath (V2 index)
    "coughing": 2,        # cough (V2 index)
    "laughing": 1,        # laughter
    "sneezing": 8,        # sneeze (V2 index - now its own class)
    "snoring": 10,        # groan (closest match - low vocalization)
    "crying": 5,          # cry (V2 index - now its own class!)
    "crying_baby": 5,     # cry (same as adult crying for now)
    "clapping": 0,        # speech (no good match - environmental sound)
}


class VocalSoundDataset:
    """
    VocalSound dataset loader from HuggingFace or local files.

    ~21000 samples: Cough, Laughter, Sigh, Sneeze, Sniff, Throat clearing
    """

    def __init__(
        self,
        cache_dir: str | None = None,
        local_dir: str | None = None,
        max_audio_len: float = 10.0,
        val_split: float = 0.1,
    ):
        self.max_audio_len = max_audio_len
        self.samples: list[ParalinguisticsSample] = []

        # Try local first if provided
        if local_dir:
            print(f"Loading VocalSound from local directory: {local_dir}")
            self._load_local(local_dir)
        else:
            print("Loading VocalSound from HuggingFace...")
            self._load_vocalsound(cache_dir)

        print(f"Total samples: {len(self.samples)}")

    def _load_local(self, local_dir: str):
        """Load VocalSound from local manifest.json file."""
        import json

        import soundfile as sf

        manifest_path = Path(local_dir) / "manifest.json"
        if not manifest_path.exists():
            print(f"ERROR: manifest.json not found in {local_dir}")
            return

        with open(manifest_path) as f:
            manifest = json.load(f)

        print(f"  Found {len(manifest)} entries in manifest")

        loaded = 0
        skipped = 0
        for entry in manifest:
            audio_path = entry.get('audio_path', '')
            label = entry.get('label', '')
            entry.get('split', 'train')

            # Map label to class ID
            if label not in LOCAL_VOCALSOUND_MAP:
                skipped += 1
                continue

            label_id = LOCAL_VOCALSOUND_MAP[label]

            # Check audio file exists
            if not os.path.exists(audio_path):
                skipped += 1
                continue

            # Get duration
            try:
                info = sf.info(audio_path)
                duration = info.duration
                if duration > self.max_audio_len:
                    skipped += 1
                    continue
            except Exception:
                skipped += 1
                continue

            self.samples.append(ParalinguisticsSample(
                audio_path=audio_path,
                label=label,
                label_id=label_id,
                duration=duration,
            ))
            loaded += 1

        print(f"  Loaded {loaded} samples, skipped {skipped}")

    def _load_vocalsound(self, cache_dir: str | None):
        """Load VocalSound from HuggingFace."""
        try:
            from datasets import Audio, load_dataset
        except ImportError:
            print("ERROR: datasets library required. Install: pip install datasets")
            return

        kwargs = {}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir

        try:
            dataset = load_dataset('lmms-lab/vocalsound', **kwargs)
        except Exception as e:
            print(f"ERROR loading VocalSound: {e}")
            return

        # Process both splits
        for split in ['test', 'val']:
            if split not in dataset:
                continue

            split_ds = dataset[split]

            # Force audio decoding with soundfile backend (avoids torchcodec hangs)
            if 'audio' in split_ds.column_names:
                split_ds = split_ds.cast_column("audio", Audio(sampling_rate=16000, decode=True))

            print(f"  Processing {split}: {len(split_ds)} samples")

            for i in range(len(split_ds)):
                try:
                    item = split_ds[i]
                    sample = self._process_item(item)
                    if sample:
                        self.samples.append(sample)
                except Exception:
                    # Skip corrupted audio
                    pass

    def _process_item(self, item: dict) -> ParalinguisticsSample | None:
        """Process a single VocalSound item."""
        # Get label
        label = item.get("answer", "")
        if label not in VOCALSOUND_MAP:
            return None
        label_id = VOCALSOUND_MAP[label]

        # Get audio
        audio_data = item.get("audio", {})
        if not audio_data:
            return None

        try:
            # Handle different audio formats
            if hasattr(audio_data, "get_all_samples"):
                # torchcodec AudioDecoder
                samples = audio_data.get_all_samples()
                audio_array = samples.data.numpy()
                if audio_array.ndim > 1:
                    audio_array = audio_array.mean(axis=0)
                sample_rate = samples.sample_rate
            elif isinstance(audio_data, dict):
                # Decoded audio from cast_column
                audio_array = audio_data.get("array")
                sample_rate = audio_data.get("sampling_rate", 16000)
            else:
                return None
        except Exception:
            return None

        if audio_array is None:
            return None

        # Resample to 16kHz if needed (should already be 16kHz from cast_column)
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

        return ParalinguisticsSample(
            audio_path="__in_memory__",
            label=label,
            label_id=label_id,
            audio_array=np.array(audio_array, dtype=np.float32),
        )

    def get_samples(self) -> list[ParalinguisticsSample]:
        return self.samples


class ESC50Dataset:
    """
    ESC-50 dataset loader for paralinguistics-relevant categories.

    240 relevant samples: breathing (40), coughing (40), laughing (40),
    sneezing (40), snoring (40), crying_baby (40 - excluded)
    """

    def __init__(
        self,
        data_dir: str,
        max_audio_len: float = 10.0,
    ):
        self.data_dir = Path(data_dir)
        self.max_audio_len = max_audio_len
        self.samples: list[ParalinguisticsSample] = []

        print(f"Loading ESC-50 from: {self.data_dir}")
        self._load_esc50()

        print(f"Total samples: {len(self.samples)}")

    def _load_esc50(self):
        """Load paralinguistics-relevant ESC-50 samples."""
        import pandas as pd

        meta_path = self.data_dir / "meta" / "esc50.csv"
        if not meta_path.exists():
            print(f"  Meta file not found: {meta_path}")
            return

        df = pd.read_csv(meta_path)

        # Filter for paralinguistics-relevant categories
        relevant_cats = list(ESC50_MAP.keys())
        df_filtered = df[df['category'].isin(relevant_cats)]

        print(f"  Found {len(df_filtered)} relevant samples")

        for _, row in df_filtered.iterrows():
            filename = row['filename']
            category = row['category']

            audio_path = self.data_dir / "audio" / filename
            if not audio_path.exists():
                continue

            label_id = ESC50_MAP.get(category, 10)  # Default to "other"

            self.samples.append(ParalinguisticsSample(
                audio_path=str(audio_path),
                label=category,
                label_id=label_id,
            ))

    def get_samples(self) -> list[ParalinguisticsSample]:
        return self.samples


class LocalESC50ParaDataset:
    """
    Load wav files from esc50_para directory.

    Files are named like: breathing_0008.wav, laughing_0024.wav
    Categories: breathing, clapping, coughing, crying, laughing, sneezing, snoring
    """

    def __init__(
        self,
        data_dir: str,
        max_audio_len: float = 10.0,
    ):
        self.data_dir = Path(data_dir)
        self.max_audio_len = max_audio_len
        self.samples: list[ParalinguisticsSample] = []

        print(f"Loading local ESC-50 para from: {self.data_dir}")
        self._load_samples()
        print(f"Total samples: {len(self.samples)}")

    def _load_samples(self):
        """Load wav files from directory, extracting category from filename."""
        if not self.data_dir.exists():
            print(f"  Directory not found: {self.data_dir}")
            return

        wav_files = list(self.data_dir.glob("*.wav"))
        print(f"  Found {len(wav_files)} .wav files")

        for wav_path in wav_files:
            # Extract category from filename (e.g., "breathing_0008.wav" -> "breathing")
            filename = wav_path.stem  # "breathing_0008"
            parts = filename.rsplit("_", 1)
            if len(parts) < 2:
                continue
            category = parts[0]

            if category not in ESC50_MAP:
                continue

            label_id = ESC50_MAP[category]

            self.samples.append(ParalinguisticsSample(
                audio_path=str(wav_path),
                label=category,
                label_id=label_id,
            ))

    def get_samples(self) -> list[ParalinguisticsSample]:
        return self.samples


class PodcastFillersDataset:
    """
    PodcastFillers dataset loader for filler detection (um, uh, erm, ah).

    From: ylacombe/podcast_fillers on HuggingFace
    """

    def __init__(
        self,
        data_dir: str | None = None,
        cache_dir: str | None = None,
        max_audio_len: float = 10.0,
    ):
        self.data_dir = Path(data_dir) if data_dir else None
        self.max_audio_len = max_audio_len
        self.samples: list[ParalinguisticsSample] = []

        # Try to load from disk first, then from HuggingFace
        if self.data_dir and self.data_dir.exists():
            self._load_from_disk()
        else:
            self._load_from_huggingface(cache_dir)

        print(f"Total filler samples: {len(self.samples)}")

    def _load_from_disk(self):
        """Load from saved HuggingFace format on disk."""
        from datasets import Audio, load_from_disk

        print(f"Loading PodcastFillers from disk: {self.data_dir}")

        try:
            ds = load_from_disk(str(self.data_dir))

            # Force audio decoding with soundfile backend (avoids torchcodec hangs)
            if 'audio' in ds.column_names:
                ds = ds.cast_column("audio", Audio(sampling_rate=16000, decode=True))

            self._process_dataset(ds)
        except Exception as e:
            print(f"  Error loading from disk: {e}")

    def _load_from_huggingface(self, cache_dir: str | None):
        """Load from HuggingFace Hub."""
        try:
            from datasets import Audio, load_dataset
        except ImportError:
            print("ERROR: datasets library required. Install: pip install datasets")
            return

        print("Loading PodcastFillers from HuggingFace...")

        try:
            kwargs = {}
            if cache_dir:
                kwargs["cache_dir"] = cache_dir

            ds = load_dataset('ylacombe/podcast_fillers', split='train', **kwargs)

            # Force audio decoding with soundfile backend (avoids torchcodec hangs)
            if 'audio' in ds.column_names:
                ds = ds.cast_column("audio", Audio(sampling_rate=16000, decode=True))

            self._process_dataset(ds)
        except Exception as e:
            print(f"  Error loading from HuggingFace: {e}")

    def _process_dataset(self, ds):
        """Process dataset by accessing raw file paths and decoding with soundfile."""
        import soundfile as sf

        # Access underlying arrow table to get file paths directly (bypass torchcodec)
        table = ds.data
        audio_col = table.column('audio')

        skipped = 0
        for i in range(len(audio_col)):
            try:
                item = audio_col[i].as_py()
                audio_path = item.get('path') if isinstance(item, dict) else None

                if not audio_path:
                    skipped += 1
                    continue

                # Load audio directly with soundfile (bypasses torchcodec)
                try:
                    audio_array, sample_rate = sf.read(audio_path)
                except Exception:
                    # Try with librosa for mp3 files
                    try:
                        import librosa
                        audio_array, sample_rate = librosa.load(audio_path, sr=None)
                    except Exception:
                        skipped += 1
                        continue

                # Convert stereo to mono
                if audio_array.ndim > 1:
                    audio_array = audio_array.mean(axis=-1)

                # Resample to 16kHz if needed
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
                    skipped += 1
                    continue

                self.samples.append(ParalinguisticsSample(
                    audio_path="__in_memory__",
                    label="filler",
                    label_id=8,
                    audio_array=np.array(audio_array, dtype=np.float32),
                ))

                if (i + 1) % 50 == 0:
                    print(f"    Processed {i + 1}/{len(audio_col)} samples ({skipped} skipped)")

            except Exception:
                skipped += 1

        if skipped > 0:
            print(f"    Skipped {skipped} samples (too long or decode error)")

    def _process_item(self, item: dict) -> ParalinguisticsSample | None:
        """Process a single filler item."""
        # Get audio
        audio_data = item.get("audio", {})
        if not audio_data:
            return None

        try:
            # Handle different audio formats
            if hasattr(audio_data, "get_all_samples"):
                # torchcodec AudioDecoder
                samples = audio_data.get_all_samples()
                audio_array = samples.data.numpy()
                if audio_array.ndim > 1:
                    audio_array = audio_array.mean(axis=0)
                sample_rate = samples.sample_rate
            elif isinstance(audio_data, dict):
                # Decoded audio from cast_column
                audio_array = audio_data.get("array")
                sample_rate = audio_data.get("sampling_rate", 16000)
            else:
                return None
        except Exception:
            return None

        if audio_array is None:
            return None

        # Resample to 16kHz if needed (should already be 16kHz from cast_column)
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

        return ParalinguisticsSample(
            audio_path="__in_memory__",
            label="filler",
            label_id=8,  # filler class
            audio_array=np.array(audio_array, dtype=np.float32),
        )

    def get_samples(self) -> list[ParalinguisticsSample]:
        return self.samples


class SilenceDataset:
    """
    Silence dataset loader from extracted silence segments.

    Expects .wav files in the specified directory.
    """

    def __init__(
        self,
        data_dir: str = "data/paralinguistics/silence",
        max_audio_len: float = 10.0,
    ):
        self.data_dir = Path(data_dir)
        self.max_audio_len = max_audio_len
        self.samples: list[ParalinguisticsSample] = []

        if self.data_dir.exists():
            self._load_silence()
        else:
            print(f"  Silence directory not found: {self.data_dir}")

        print(f"Total silence samples: {len(self.samples)}")

    def _load_silence(self):
        """Load silence samples from directory."""
        import soundfile as sf

        print(f"Loading silence samples from: {self.data_dir}")

        wav_files = list(self.data_dir.glob("*.wav"))
        print(f"  Found {len(wav_files)} .wav files")

        for audio_path in wav_files:
            try:
                # Check file size and duration
                info = sf.info(audio_path)
                duration = info.duration

                if duration > self.max_audio_len:
                    continue

                self.samples.append(ParalinguisticsSample(
                    audio_path=str(audio_path),
                    label="silence",
                    label_id=9,  # silence class
                ))
            except Exception as e:
                print(f"  Error loading {audio_path}: {e}")

    def get_samples(self) -> list[ParalinguisticsSample]:
        return self.samples


class SEP28kDataset:
    """
    SEP-28k (Stuttering Events in Podcasts) dataset loader.

    Contains ~28,000 3-second clips annotated with:
    - Prolongation (elongated syllables)
    - Block (gasps/stuttered pauses)
    - Sound Repetition (repeated syllables)
    - Word Repetition (repeated words)
    - Interjection (um, uh)

    Reference: https://github.com/apple/ml-stuttering-events-dataset
    """

    # SEP-28k label to PARALINGUISTICS_CLASSES mapping
    LABEL_MAP = {
        "Prolongation": 13,
        "Block": 14,
        "SoundRep": 15,
        "WordRep": 16,
        "Interjection": 8,  # Generic filler
        # Additional annotations (used for filtering)
        "Unsure": None,
        "PoorAudioQuality": None,
        "DifficultToUnderstand": None,
        "NaturalPause": 9,  # Map to silence
        "Music": None,
        "NoSpeech": 9,  # Map to silence
    }

    def __init__(
        self,
        data_dir: str = "data/sep28k",
        max_audio_len: float = 10.0,
        min_confidence: int = 2,  # Minimum annotator agreement (0-3)
    ):
        self.data_dir = Path(data_dir)
        self.max_audio_len = max_audio_len
        self.min_confidence = min_confidence
        self.samples: list[ParalinguisticsSample] = []

        if self.data_dir.exists():
            self._load_sep28k()
        else:
            print(f"  SEP-28k directory not found: {self.data_dir}")
            print("  Download from: https://github.com/apple/ml-stuttering-events-dataset")

        print(f"Total SEP-28k samples: {len(self.samples)}")

    def _load_sep28k(self):
        """Load SEP-28k dataset from CSV and audio files."""

        print(f"Loading SEP-28k from: {self.data_dir}")

        # Look for CSV files
        csv_files = list(self.data_dir.glob("*.csv"))
        if not csv_files:
            csv_files = list(self.data_dir.glob("**/*.csv"))

        if not csv_files:
            print(f"  No CSV files found in {self.data_dir}")
            return

        for csv_path in csv_files:
            self._process_csv(csv_path)

    def _process_csv(self, csv_path: Path):
        """Process a single SEP-28k CSV file."""
        import pandas as pd

        print(f"  Processing {csv_path.name}...")

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"    Error reading CSV: {e}")
            return

        # SEP-28k CSV format: Show, EpId, ClipId, Start, Stop, <labels>
        # Labels are 0-3 indicating annotator agreement
        label_columns = ["Prolongation", "Block", "SoundRep", "WordRep", "Interjection"]

        processed = 0
        for _, row in df.iterrows():
            # Find the label with highest agreement
            best_label = None
            best_score = 0

            for label in label_columns:
                if label in row:
                    score = row[label]
                    if score >= self.min_confidence and score > best_score:
                        best_score = score
                        best_label = label

            if best_label is None:
                continue  # No confident label

            label_id = self.LABEL_MAP.get(best_label)
            if label_id is None:
                continue  # Skip uncertain/bad quality

            # Build audio path
            # SEP-28k uses: {Show}/{EpId}/{ClipId}.wav
            show = row.get("Show", "")
            ep_id = row.get("EpId", "")
            clip_id = row.get("ClipId", "")

            if not all([show, ep_id, clip_id]):
                continue

            audio_path = self.data_dir / "clips" / show / str(ep_id) / f"{clip_id}.wav"

            if not audio_path.exists():
                # Try alternative paths
                alt_path = self.data_dir / show / str(ep_id) / f"{clip_id}.wav"
                if alt_path.exists():
                    audio_path = alt_path
                else:
                    continue

            self.samples.append(ParalinguisticsSample(
                audio_path=str(audio_path),
                label=best_label.lower(),
                label_id=label_id,
            ))
            processed += 1

        print(f"    Loaded {processed} samples from {csv_path.name}")

    def get_samples(self) -> list[ParalinguisticsSample]:
        return self.samples


class FillerDataset:
    """
    Filler dataset loader from extracted filler .wav files.

    Loads from extracted DisfluencySpeech fillers (um, uh, etc.)
    """

    def __init__(
        self,
        data_dir: str = "data/paralinguistics/fillers",
        max_audio_len: float = 10.0,
    ):
        self.data_dir = Path(data_dir)
        self.max_audio_len = max_audio_len
        self.samples: list[ParalinguisticsSample] = []

        if self.data_dir.exists():
            self._load_fillers()
        else:
            print(f"  Filler directory not found: {self.data_dir}")

        print(f"Total extracted filler samples: {len(self.samples)}")

    def _load_fillers(self):
        """Load filler samples from directory."""
        import soundfile as sf

        print(f"Loading extracted filler samples from: {self.data_dir}")

        wav_files = list(self.data_dir.glob("*.wav"))
        print(f"  Found {len(wav_files)} .wav files")

        for audio_path in wav_files:
            try:
                # Check file size and duration
                info = sf.info(audio_path)
                duration = info.duration

                if duration > self.max_audio_len:
                    continue

                self.samples.append(ParalinguisticsSample(
                    audio_path=str(audio_path),
                    label="filler",
                    label_id=8,  # filler class
                ))
            except Exception as e:
                print(f"  Error loading {audio_path}: {e}")

    def get_samples(self) -> list[ParalinguisticsSample]:
        return self.samples


class ParalinguisticsTrainer:
    """Trainer for ParalinguisticsHead."""

    def __init__(
        self,
        config: ParalinguisticsTrainingConfig,
        whisper_model: WhisperMLX,
        paralinguistics_head: ParalinguisticsHead,
        class_counts: dict[int, int] | None = None,
    ):
        self.config = config
        self.whisper_model = whisper_model
        self.paralinguistics_head = paralinguistics_head

        # Optimizer
        self.optimizer = optim.AdamW(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Class weights for imbalanced loss
        if config.use_class_weights:
            if config.use_frequency_weights and class_counts:
                self.class_weights = compute_class_weights_from_counts(
                    class_counts=class_counts,
                    num_classes=len(PARALINGUISTICS_CLASSES),
                )
            else:
                self.class_weights = create_paralinguistics_class_weights(
                    speech_weight=config.speech_weight,
                    event_weight=config.event_weight,
                )
        else:
            self.class_weights = None

        # Training state
        self.step = 0
        self.epoch = 0
        self.best_loss = float("inf")
        self.best_acc = 0.0

        # Encoder caching (~3x speedup by caching frozen encoder outputs)
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

    def _populate_sample_durations(self, samples: list[ParalinguisticsSample]) -> None:
        """
        Populate duration field for samples that have duration=0.

        Uses soundfile.info() for fast metadata-only access.
        For in-memory audio samples, computes duration from array length.
        Modifies samples in-place.
        """
        samples_needing_duration = [s for s in samples if s.duration == 0.0]
        if not samples_needing_duration:
            return

        self.log(f"Populating durations for {len(samples_needing_duration)} samples...")
        start_time = time.time()

        for i, sample in enumerate(samples_needing_duration):
            if sample.audio_array is not None:
                # In-memory audio: compute from array length
                sample.duration = len(sample.audio_array) / self.config.sample_rate
            elif sample.audio_path:
                # File-based audio: use soundfile.info()
                sample.duration = get_audio_duration(sample.audio_path)
            else:
                sample.duration = 0.0

            if (i + 1) % 10000 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                self.log(f"  {i + 1}/{len(samples_needing_duration)} ({rate:.0f} samples/sec)")

        elapsed = time.time() - start_time
        self.log(f"Duration population complete in {elapsed:.1f}s")

    def _create_length_sorted_batches(
        self, samples: list[ParalinguisticsSample],
    ) -> list[list[ParalinguisticsSample]]:
        """
        Create batches of similar-length samples.

        Strategy:
        1. Sort all samples by duration
        2. Divide into buckets of size (batch_size * bucket_size_multiplier)
        3. Shuffle samples within each bucket
        4. Create batches from each bucket
        5. Shuffle the batches

        This reduces padding waste while maintaining some randomness.
        Expected speedup: 1.2-1.3x (less computation wasted on padding).

        Args:
            samples: List of ParalinguisticsSample (must have duration populated)

        Returns:
            List of batches, each batch is a list of ParalinguisticsSample
        """
        # Sort by duration (shortest first reduces peak memory)
        sorted_samples = sorted(samples, key=lambda s: s.duration)

        batch_size = self.config.batch_size
        bucket_size = batch_size * self.config.bucket_size_multiplier

        batches = []
        rng = np.random.default_rng()

        # Process in buckets
        for bucket_start in range(0, len(sorted_samples), bucket_size):
            bucket_end = min(bucket_start + bucket_size, len(sorted_samples))
            bucket = sorted_samples[bucket_start:bucket_end]

            # Shuffle within bucket
            rng.shuffle(bucket)

            # Create batches from bucket
            for i in range(0, len(bucket), batch_size):
                batch = bucket[i:i + batch_size]
                if batch:
                    batches.append(batch)

        # Shuffle batches for training randomness
        rng.shuffle(batches)

        return batches

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


    def _encode_audio_cached(
        self, audio: np.ndarray, audio_path: str | None = None,
    ) -> tuple[mx.array, int]:
        """
        Encode audio through frozen Whisper encoder with caching support.

        Args:
            audio: Audio waveform
            audio_path: Optional path for cache key (None for in-memory audio)

        Returns:
            Tuple of (encoder_output, actual_frames) where encoder_output
            has batch dimension removed.
        """
        # Check cache first (only if audio_path is provided)
        if self.encoder_cache is not None and audio_path is not None:
            cached = self.encoder_cache.load(audio_path)
            if cached is not None:
                enc_out, actual_frames = cached
                return enc_out, actual_frames

        # Cache miss or no caching - compute encoder output
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
        enc_out = encoder_output[0]  # Remove batch dim

        # Actual encoder frames = mel frames // 2 (Whisper encoder downsamples 2x)
        actual_frames = min(actual_mel_frames // 2, 1500)

        # Save to cache (only if audio_path is provided)
        if self.encoder_cache is not None and audio_path is not None:
            self.encoder_cache.save(audio_path, enc_out, actual_frames)

        return enc_out, actual_frames

    def _prepare_batch(
        self,
        samples: list[ParalinguisticsSample],
    ) -> tuple[mx.array, mx.array]:
        """Prepare a batch for training."""
        batch_encoder_outputs = []
        labels = []

        for sample in samples:
            # Load audio
            if sample.audio_array is not None:
                audio = sample.audio_array
                audio_path_for_cache = None  # Can't cache in-memory audio
            else:
                audio = load_audio(sample.audio_path)
                audio_path_for_cache = sample.audio_path

            # Skip empty audio samples
            if len(audio) == 0:
                continue

            # Trim to max length
            max_samples = int(self.config.max_audio_len * self.config.sample_rate)
            if len(audio) > max_samples:
                audio = audio[:max_samples]

            # Encode (with caching if audio_path is available)
            if self.encoder_cache is not None and audio_path_for_cache is not None:
                # Use cached version
                enc_out, _ = self._encode_audio_cached(audio, audio_path_for_cache)
                batch_encoder_outputs.append(enc_out)
            else:
                # Original path - no caching
                encoder_output = self._encode_audio(audio)
                batch_encoder_outputs.append(encoder_output[0])  # Remove batch dim

            # For paralinguistics, we use utterance-level labels
            # The entire clip is labeled as one class
            labels.append(sample.label_id)

        # Handle case where all samples in batch were empty
        if not batch_encoder_outputs:
            return None, None

        encoder_outputs = mx.stack(batch_encoder_outputs)
        label_array = mx.array(labels)

        return encoder_outputs, label_array

    def train_step(self, batch_samples: list[ParalinguisticsSample]) -> float:
        """Single training step."""
        encoder_outputs, labels = self._prepare_batch(batch_samples)

        # Skip if batch was empty after filtering
        if encoder_outputs is None:
            return 0.0

        def loss_fn(head):
            # Get utterance-level logits (mean pooled)
            logits = head(encoder_outputs, return_frame_logits=False)

            if self.config.use_focal_loss:
                return focal_loss(
                    logits, labels,
                    gamma=self.config.focal_gamma,
                    alpha=self.class_weights,
                    reduction="mean",
                )
            return paralinguistics_loss(
                logits, labels,
                class_weights=self.class_weights,
                reduction="mean",
            )

        loss, grads = mx.value_and_grad(loss_fn)(self.paralinguistics_head)
        self.optimizer.update(self.paralinguistics_head, grads)
        mx.eval(self.paralinguistics_head.parameters())

        return float(loss)

    def train(
        self,
        train_samples: list[ParalinguisticsSample],
        val_samples: list[ParalinguisticsSample],
    ):
        """Main training loop."""
        # Calculate steps per epoch for resuming
        steps_per_epoch = (len(train_samples) + self.config.batch_size - 1) // self.config.batch_size

        # Determine starting epoch and step within epoch
        start_epoch = self.epoch if self.step > 0 else 0
        start_step_in_epoch = self.step - (start_epoch * steps_per_epoch) if self.step > 0 else 0

        if self.step > 0:
            self.log(f"Resuming paralinguistics training from step {self.step}")
        else:
            self.log("Starting paralinguistics training")

        self.log(f"Epochs: {self.config.epochs}")
        self.log(f"Train samples: {len(train_samples)}")
        self.log(f"Val samples: {len(val_samples)}")
        self.log(f"Classes: {len(PARALINGUISTICS_CLASSES)}")
        self.log(f"Steps per epoch: {steps_per_epoch}")
        self.log(f"Length-sorted batching: {self.config.length_sorted_batching}")

        # Populate durations if using length-sorted batching
        if self.config.length_sorted_batching:
            self._populate_sample_durations(train_samples)
            self._populate_sample_durations(val_samples)
            # Log duration statistics
            durations = [s.duration for s in train_samples]
            self.log(f"Duration stats: min={min(durations):.1f}s, max={max(durations):.1f}s, mean={np.mean(durations):.1f}s")

        # Print class distribution
        train_counts = {}
        for s in train_samples:
            train_counts[s.label] = train_counts.get(s.label, 0) + 1
        self.log(f"Train distribution: {train_counts}")

        for epoch in range(start_epoch, self.config.epochs):
            self.epoch = epoch
            self.log(f"Starting epoch {epoch}")

            # Track padding efficiency for length-sorted batching
            total_frames = 0
            total_padded_frames = 0

            # Shuffle with fixed seed for reproducibility during resume
            rng = np.random.default_rng(42 + epoch)

            if self.config.length_sorted_batching:
                # Use length-sorted batches
                batches = self._create_length_sorted_batches(train_samples)
            else:
                # Standard random shuffle
                rng.shuffle(train_samples)
                batches = [
                    train_samples[i:i + self.config.batch_size]
                    for i in range(0, len(train_samples), self.config.batch_size)
                ]

            # Training epoch
            epoch_losses = []
            batch_idx = 0
            skipped_count = 0
            for batch in batches:
                batch_idx += 1

                # Skip batches already completed (for resume)
                if epoch == start_epoch and batch_idx <= start_step_in_epoch:
                    skipped_count += 1
                    continue

                # Log first batch after skip
                if skipped_count > 0 and len(epoch_losses) == 0:
                    self.log(f"  Skipped {skipped_count} batches, starting at batch {batch_idx}")
                    skipped_count = 0  # Reset to avoid logging again

                loss = self.train_step(batch)
                epoch_losses.append(loss)
                self.step += 1

                # Track padding efficiency for length-sorted batching
                if self.config.length_sorted_batching and batch:
                    batch_durations = [s.duration for s in batch]
                    max_duration = max(batch_durations)
                    actual_frames = sum(int(d * 100) for d in batch_durations)  # ~100 frames/sec
                    padded_frames = len(batch) * int(max_duration * 100)
                    total_frames += actual_frames
                    total_padded_frames += padded_frames

                # Logging
                if self.step % self.config.log_interval == 0:
                    avg_loss = np.mean(epoch_losses[-self.config.log_interval:])
                    log_msg = f"  Step {self.step}: loss={avg_loss:.4f}"

                    # Add padding efficiency if using length-sorted batching
                    if self.config.length_sorted_batching and total_padded_frames > 0:
                        efficiency = total_frames / total_padded_frames * 100
                        log_msg += f", pad_eff={efficiency:.1f}%"

                    self.log(log_msg)

                # Save checkpoint
                if self.step % self.config.save_interval == 0:
                    self._save_checkpoint(f"step_{self.step}.npz")

                clear_memory()

            # Epoch summary
            if epoch_losses:
                avg_epoch_loss = np.mean(epoch_losses)
                epoch_msg = f"Epoch {epoch + 1}/{self.config.epochs}: loss={avg_epoch_loss:.4f}"

                # Add padding efficiency summary for length-sorted batching
                if self.config.length_sorted_batching and total_padded_frames > 0:
                    efficiency = total_frames / total_padded_frames * 100
                    epoch_msg += f", pad_eff={efficiency:.1f}%"

                self.log(epoch_msg)

            # Validation
            if val_samples:
                val_loss, val_acc = self._validate(val_samples)
                self.log(f"  Val loss: {val_loss:.4f}, accuracy: {val_acc:.2%}")

                # Save best by accuracy (more meaningful than loss for classification)
                if val_acc > self.best_acc:
                    self.best_acc = val_acc
                    self.best_loss = val_loss
                    self._save_checkpoint("best.npz")
                    self.log(f"  New best model saved! (acc={val_acc:.2%})")

            # Reset start_step_in_epoch for subsequent epochs
            start_step_in_epoch = 0

        # Save final
        self._save_checkpoint("final.npz")
        self.log("Training complete!")

    def _validate(
        self,
        val_samples: list[ParalinguisticsSample],
    ) -> tuple[float, float]:
        """Run validation."""
        # Set to eval mode to disable dropout
        self.paralinguistics_head.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch_start in range(0, len(val_samples), self.config.batch_size):
            batch_end = min(batch_start + self.config.batch_size, len(val_samples))
            batch = val_samples[batch_start:batch_end]

            encoder_outputs, labels = self._prepare_batch(batch)

            # Skip if batch was empty after filtering
            if encoder_outputs is None:
                continue

            # Forward pass
            logits = self.paralinguistics_head(encoder_outputs, return_frame_logits=False)

            # Loss (use same loss function as training)
            if self.config.use_focal_loss:
                loss = focal_loss(
                    logits, labels,
                    gamma=self.config.focal_gamma,
                    alpha=self.class_weights,
                    reduction="mean",
                )
            else:
                loss = paralinguistics_loss(
                    logits, labels,
                    class_weights=self.class_weights,
                    reduction="mean",
                )
            actual_batch_size = encoder_outputs.shape[0]
            total_loss += float(loss) * actual_batch_size

            # Accuracy
            preds = mx.argmax(logits, axis=-1)
            correct += int(mx.sum(preds == labels))
            total += actual_batch_size

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)

        # Restore training mode
        self.paralinguistics_head.train()

        return avg_loss, accuracy

    def _save_checkpoint(self, filename: str):
        """Save head weights."""
        from mlx.utils import tree_flatten

        weights = {}
        for k, v in tree_flatten(self.paralinguistics_head.parameters()):
            weights[f"paralinguistics.{k}"] = v

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
        head_weights = {}
        prefix = "paralinguistics."
        for k, v in data.items():
            if k.startswith(prefix):
                head_weights[k[len(prefix):]] = v

        if head_weights:
            # Check for class count mismatch in output layer
            current_params = dict(self.paralinguistics_head.parameters())
            compatible_weights = []
            skipped = []

            for key, value in head_weights.items():
                # Get expected shape from current model
                if key in current_params:
                    expected_shape = current_params[key].shape
                    actual_shape = value.shape

                    if expected_shape != actual_shape:
                        # Shape mismatch - likely class count changed
                        if "fc2" in key:
                            # Output layer - partial load for compatible classes
                            self.log(f"  Class count changed: {actual_shape} -> {expected_shape}")
                            if len(actual_shape) == 2:
                                # Weight matrix: [out_features, in_features]
                                old_classes = actual_shape[0]
                                new_classes = expected_shape[0]
                                # Copy weights for existing classes
                                new_weight = current_params[key]  # Keep random init
                                new_weight[:old_classes, :] = value[:old_classes, :]
                                compatible_weights.append((key, new_weight))
                                self.log(f"  Partial load {key}: copied {old_classes}/{new_classes} classes")
                            elif len(actual_shape) == 1:
                                # Bias: [out_features]
                                old_classes = actual_shape[0]
                                new_classes = expected_shape[0]
                                new_bias = current_params[key]
                                new_bias[:old_classes] = value[:old_classes]
                                compatible_weights.append((key, new_bias))
                                self.log(f"  Partial load {key}: copied {old_classes}/{new_classes} classes")
                        else:
                            skipped.append(key)
                            self.log(f"  Skipped {key}: shape mismatch {actual_shape} vs {expected_shape}")
                    else:
                        compatible_weights.append((key, value))
                else:
                    skipped.append(key)

            if compatible_weights:
                self.paralinguistics_head.load_weights(compatible_weights)
                mx.eval(self.paralinguistics_head.parameters())

            if skipped:
                self.log(f"  Skipped {len(skipped)} incompatible weights")

        self.log(f"Resumed from step {self.step}, epoch {self.epoch}")


def main():
    parser = argparse.ArgumentParser(description="Paralinguistics Training")
    parser.add_argument("--output-dir", type=str, default="checkpoints/paralinguistics",
                        help="Output directory")
    parser.add_argument("--vocalsound-cache", type=str, help="HuggingFace cache dir")
    parser.add_argument("--vocalsound-dir", type=str,
                        default="data/paralinguistics/vocalsound_labeled",
                        help="Local VocalSound directory with manifest.json")
    parser.add_argument("--esc50-dir", type=str, default="data/paralinguistics/esc50",
                        help="ESC-50 directory")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--model", type=str, default="mlx-community/whisper-large-v3-mlx",
                        help="Whisper model")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")

    # Loss configuration
    parser.add_argument("--no-focal-loss", action="store_true",
                        help="Disable focal loss (use regular CE)")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Focal loss gamma (higher=more focus on hard examples)")
    parser.add_argument("--no-frequency-weights", action="store_true",
                        help="Disable frequency-based class weights")
    # Encoder caching (~3x speedup)
    parser.add_argument("--encoder-cache-dir", type=str,
                        help="Directory for cached encoder outputs (~3x speedup)")

    # Length-sorted batching (~1.3x speedup)
    parser.add_argument("--length-sorted-batching", action="store_true",
                        help="Enable length-sorted batching (reduces padding waste ~1.3x speedup)")
    parser.add_argument("--bucket-size-multiplier", type=int, default=100,
                        help="Bucket size = batch_size * multiplier (default: 100)")

    args = parser.parse_args()

    config = ParalinguisticsTrainingConfig(
        vocalsound_cache=args.vocalsound_cache,
        vocalsound_dir=args.vocalsound_dir,
        esc50_dir=args.esc50_dir,
        output_dir=args.output_dir,
        whisper_model=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_focal_loss=not args.no_focal_loss,
        focal_gamma=args.focal_gamma,
        use_frequency_weights=not args.no_frequency_weights,
        encoder_cache_dir=args.encoder_cache_dir,
        length_sorted_batching=args.length_sorted_batching,
        bucket_size_multiplier=args.bucket_size_multiplier,
    )

    print("=" * 70)
    print("Paralinguistics Training for Whisper")
    print("=" * 70)
    print(f"Output: {config.output_dir}")
    print(f"Model: {config.whisper_model}")
    if config.encoder_cache_dir:
        print(f"Encoder cache: {config.encoder_cache_dir} (~3x speedup)")
    print()

    # Load Whisper model
    print("1. Loading Whisper model...")
    whisper_model = WhisperMLX.from_pretrained(config.whisper_model)
    d_model = whisper_model.config.n_audio_state
    config.d_model = d_model
    print(f"   d_model={d_model}")

    # Create head
    print("2. Creating ParalinguisticsHead...")
    head_config = MultiHeadConfig(
        d_model=d_model,
        use_paralinguistics=True,
        num_paralinguistics_classes=11,  # Base paralinguistics classes only
        paralinguistics_hidden_dim=256,
    )
    paralinguistics_head = ParalinguisticsHead(head_config)

    from mlx.utils import tree_flatten
    n_params = sum(p.size for _, p in tree_flatten(paralinguistics_head.parameters()))
    print(f"   Parameters: {n_params / 1e6:.2f}M")

    # Load datasets
    all_samples = []

    print("3. Loading VocalSound...")
    try:
        vocalsound = VocalSoundDataset(
            cache_dir=config.vocalsound_cache,
            local_dir=config.vocalsound_dir,
            max_audio_len=config.max_audio_len,
        )
        all_samples.extend(vocalsound.get_samples())
        print(f"   Loaded {len(vocalsound.get_samples())} samples")
    except Exception as e:
        print(f"   Error loading VocalSound: {e}")

    print("4. Loading ESC-50...")
    if config.esc50_dir and Path(config.esc50_dir).exists():
        esc50 = ESC50Dataset(
            data_dir=config.esc50_dir,
            max_audio_len=config.max_audio_len,
        )
        all_samples.extend(esc50.get_samples())
        print(f"   Loaded {len(esc50.get_samples())} samples")
    else:
        print(f"   ESC-50 not found at {config.esc50_dir}")

    # Load local ESC-50 para wav files (extracted paralinguistic sounds)
    print("4b. Loading local ESC-50 para...")
    esc50_para_dir = Path("data/paralinguistics/esc50_para")
    try:
        esc50_para = LocalESC50ParaDataset(
            data_dir=str(esc50_para_dir),
            max_audio_len=config.max_audio_len,
        )
        all_samples.extend(esc50_para.get_samples())
        print(f"   Loaded {len(esc50_para.get_samples())} local ESC-50 para samples")
    except Exception as e:
        print(f"   Error loading local ESC-50 para: {e}")

    # NOTE: PodcastFillers (ylacombe/podcast_fillers) contains full 57-min podcast episodes,
    # not segmented filler clips. No timestamp annotations. Use extracted fillers instead.
    print("5. PodcastFillers skipped (contains full episodes, not clips)")

    # Load extracted filler .wav files (from DisfluencySpeech)
    print("6. Loading extracted filler .wav files...")
    extracted_fillers_dir = Path("data/paralinguistics/fillers")
    try:
        extracted_fillers = FillerDataset(
            data_dir=str(extracted_fillers_dir),
            max_audio_len=config.max_audio_len,
        )
        all_samples.extend(extracted_fillers.get_samples())
        print(f"   Loaded {len(extracted_fillers.get_samples())} extracted filler samples")
    except Exception as e:
        print(f"   Error loading extracted fillers: {e}")

    # Load Silence samples
    print("7. Loading Silence samples...")
    silence_dir = Path("data/paralinguistics/silence")
    try:
        silence = SilenceDataset(
            data_dir=str(silence_dir),
            max_audio_len=config.max_audio_len,
        )
        all_samples.extend(silence.get_samples())
        print(f"   Loaded {len(silence.get_samples())} silence samples")
    except Exception as e:
        print(f"   Error loading Silence: {e}")

    # Load SEP-28k disfluency dataset
    print("8. Loading SEP-28k disfluency samples...")
    sep28k_dir = Path("data/sep28k")
    try:
        sep28k = SEP28kDataset(
            data_dir=str(sep28k_dir),
            max_audio_len=config.max_audio_len,
            min_confidence=2,  # Require at least 2 annotators to agree
        )
        all_samples.extend(sep28k.get_samples())
        print(f"   Loaded {len(sep28k.get_samples())} SEP-28k disfluency samples")
    except Exception as e:
        print(f"   Error loading SEP-28k: {e}")
        print("   Download from: https://github.com/apple/ml-stuttering-events-dataset")

    # Load Laughterscape dataset (8,170 laughter samples)
    print("9. Loading Laughterscape laughter samples...")
    laughterscape_dir = Path("data/paralinguistics/laughterscape")
    if laughterscape_dir.exists():
        try:
            wav_files = list(laughterscape_dir.glob("*.wav"))
            laugh_count = 0
            for wav_path in wav_files:
                all_samples.append(ParalinguisticsSample(
                    audio_path=str(wav_path),
                    label="laughter",
                    label_id=1,  # laughter class
                ))
                laugh_count += 1
            print(f"   Loaded {laugh_count} Laughterscape samples")
        except Exception as e:
            print(f"   Error loading Laughterscape: {e}")
    else:
        print(f"   Laughterscape not found at {laughterscape_dir}")

    # Load CoughVid dataset (972 cough samples)
    print("10. Loading CoughVid cough samples...")
    coughvid_dir = Path("data/paralinguistics/coughvid")
    if coughvid_dir.exists():
        try:
            wav_files = list(coughvid_dir.glob("*.wav"))
            cough_count = 0
            for wav_path in wav_files:
                all_samples.append(ParalinguisticsSample(
                    audio_path=str(wav_path),
                    label="cough",
                    label_id=4,  # cough class
                ))
                cough_count += 1
            print(f"   Loaded {cough_count} CoughVid samples")
        except Exception as e:
            print(f"   Error loading CoughVid: {e}")
    else:
        print(f"   CoughVid not found at {coughvid_dir}")

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

    # Compute class counts for frequency-based weights
    class_counts = {}
    for s in train_samples:
        class_counts[s.label_id] = class_counts.get(s.label_id, 0) + 1
    print(f"Class counts: {class_counts}")

    # Create trainer
    trainer = ParalinguisticsTrainer(
        config=config,
        whisper_model=whisper_model,
        paralinguistics_head=paralinguistics_head,
        class_counts=class_counts,
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    print("\n" + "=" * 70)
    if args.resume:
        print(f"Resuming Training from {args.resume}")
    else:
        print("Starting Training")
    print("=" * 70)
    trainer.train(train_samples, val_samples)


if __name__ == "__main__":
    main()
