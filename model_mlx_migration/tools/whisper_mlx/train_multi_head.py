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
Multi-Head Training Script.

Trains multiple output heads on a frozen Whisper encoder:
1. CTC Head - Text transcription (LibriSpeech)
2. Emotion Head - 34-class expression (Expresso) or 8-class (RAVDESS)
3. Singing Head - Singing vs speaking detection (RAVDESS, VocalSet)
4. Pitch Head - F0 tracking (extracted from audio)

The encoder is FROZEN - only heads are trained.
Joint training uses weighted loss combination.

Supported Datasets:
    - Expresso: 40h, 34 expressive styles (SOTA) - loads from HuggingFace
    - RAVDESS: 24.8GB, 8 emotions + singing - download from Zenodo
    - VocalSet: 10.1h singing techniques - download from Zenodo
    - LibriSpeech: 1000h text transcription - download from OpenSLR

Usage:
    # RECOMMENDED: Train with Expresso (SOTA 34 styles)
    python -m tools.whisper_mlx.train_multi_head \
        --expresso \
        --output-dir checkpoints/multi_head_expresso \
        --epochs 10

    # Train with RAVDESS + VocalSet (local files)
    python -m tools.whisper_mlx.train_multi_head \
        --ravdess-dir data/ravdess \
        --vocalset-dir data/vocalset \
        --output-dir checkpoints/multi_head_singing \
        --epochs 10

    # Full training (all datasets)
    python -m tools.whisper_mlx.train_multi_head \
        --expresso \
        --ravdess-dir data/ravdess \
        --vocalset-dir data/vocalset \
        --librispeech-dir data/librispeech/train-clean-100 \
        --output-dir checkpoints/multi_head_full \
        --epochs 10

    # Train only emotion head
    python -m tools.whisper_mlx.train_multi_head \
        --expresso \
        --train-ctc false \
        --train-pitch false \
        --epochs 5

References:
    - Expresso: https://huggingface.co/datasets/ylacombe/expresso
    - RAVDESS: https://zenodo.org/record/1188976
    - VocalSet: https://zenodo.org/record/1193957
    - LibriSpeech: https://www.openslr.org/12
"""

import argparse
import gc
import resource
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np

# Import Whisper components
from .audio import get_audio_duration, load_audio, log_mel_spectrogram
from .ctc_head import CTCDraftHead, create_ctc_draft_head
from .encoder_cache import TrainingEncoderCache
from .model import WhisperMLX
from .multi_head import (
    EXPRESSO_STYLE_MAP,
    EXTENDED_EMOTIONS,
    RAVDESS_EMOTIONS,
    WhisperMultiHead,
    create_multi_head,
    crepe_pitch_loss,
    emotion_loss,
    pitch_loss,
    singing_loss,
)
from .tokenizer import get_whisper_tokenizer

# PyTorch CTC loss
try:
    import torch  # noqa: F401 - used for HAS_TORCH detection and samples.data
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss / 1024 / 1024


def clear_memory():
    """Clear memory between batches."""
    if hasattr(mx, 'clear_cache'):
        mx.clear_cache()
    gc.collect()


def log_softmax(x: mx.array, axis: int = -1) -> mx.array:
    """Numerically stable log softmax."""
    max_x = mx.max(x, axis=axis, keepdims=True)
    shifted = x - max_x
    return shifted - mx.log(mx.sum(mx.exp(shifted), axis=axis, keepdims=True) + 1e-10)


@dataclass
class MultiHeadTrainingConfig:
    """Configuration for multi-head training."""

    # Data directories
    librispeech_dir: str | None = None  # For CTC
    ravdess_dir: str | None = None      # For emotion/singing
    prosody_dir: str | None = None      # Consolidated prosody (crema, esd, jvnv, ravdess)
    use_contours_only: bool = False        # Load only contours_train/val.json (with f0_contour)
    expresso_dir: str | None = None     # For 34-style emotion (SOTA)
    vocalset_dir: str | None = None     # For singing techniques
    emov_db_dir: str | None = None      # Additional emotion data

    # Output
    output_dir: str = "checkpoints/multi_head"

    # Model
    whisper_model: str = "mlx-community/whisper-large-v3-mlx"
    model_size: str = "large-v3"
    d_model: int = 1280

    # Training
    epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 1e-4
    warmup_steps: int = 500
    grad_clip: float = 1.0
    weight_decay: float = 0.01
    lr_decay: str = "cosine"  # "cosine", "linear", or "none"
    min_lr_ratio: float = 0.1  # Decay to 10% of max LR

    # Head-specific training flags
    train_ctc: bool = True
    train_emotion: bool = True
    train_singing: bool = True
    train_pitch: bool = True

    # Head architecture options
    use_crepe_pitch: bool = False  # CREPE-style 360-bin pitch (vs MLP)
    use_attention_emotion: bool = False  # Attention-based emotion (vs mean pooling)
    use_extended_singing: bool = False  # Extended singing with style + intensity

    # Loss weights
    ctc_loss_weight: float = 1.0
    emotion_loss_weight: float = 0.5
    singing_loss_weight: float = 0.3
    pitch_loss_weight: float = 0.2

    # Audio
    max_audio_len: float = 15.0  # seconds
    sample_rate: int = 16000

    # Logging
    log_interval: int = 50
    save_interval: int = 500
    eval_interval: int = 200

    # Validation
    val_split: float = 0.1

    # Encoder caching (~3x speedup by caching frozen encoder outputs)
    encoder_cache_dir: str | None = None

    # Length-sorted batching (~1.3x speedup by reducing padding waste)
    length_sorted_batching: bool = False
    bucket_size_multiplier: int = 100  # Bucket size = batch_size * multiplier


@dataclass
class MultiModalSample:
    """Sample with multiple labels for multi-head training."""

    audio_path: str
    duration: float = 0.0

    # CTC labels (text)
    transcript: str | None = None
    language: str = "en"

    # Emotion labels
    emotion_id: int | None = None  # 0-33 for extended taxonomy
    emotion_label: str | None = None

    # Singing labels
    is_singing: bool | None = None  # True = singing, False = speaking
    singing_style: int | None = None  # 0-16 for VocalSet techniques
    singing_intensity: float | None = None  # 0.0-1.0 vocal intensity

    # Pitch labels (extracted from audio)
    pitch_hz: np.ndarray | None = None  # Frame-level F0
    voiced_mask: np.ndarray | None = None

    # In-memory audio data (for HuggingFace datasets)
    audio_array: np.ndarray | None = None


class RAVDESSDataset:
    """
    RAVDESS dataset loader.

    RAVDESS file naming convention:
        Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor.wav

    Modality: 01 = full-AV, 02 = video-only, 03 = audio-only
    Vocal channel: 01 = speech, 02 = song
    Emotion: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
    Intensity: 01 = normal, 02 = strong
    Statement: 01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door"
    Repetition: 01 = 1st repetition, 02 = 2nd repetition
    Actor: 01 to 24

    Song files only have 5 emotions (no neutral or disgust).
    """

    # RAVDESS emotion mapping (1-indexed in filenames)
    EMOTION_MAP = {
        1: 0,  # neutral -> 0
        2: 1,  # calm -> 1
        3: 2,  # happy -> 2
        4: 3,  # sad -> 3
        5: 4,  # angry -> 4
        6: 5,  # fearful -> 5
        7: 6,  # disgust -> 6
        8: 7,  # surprised -> 7
    }

    # Statement transcripts
    STATEMENTS = {
        1: "Kids are talking by the door",
        2: "Dogs are sitting by the door",
    }

    def __init__(
        self,
        data_dir: str,
        max_audio_len: float = 15.0,
        val_split: float = 0.1,
    ):
        self.data_dir = Path(data_dir)
        self.max_audio_len = max_audio_len
        self.samples: list[MultiModalSample] = []

        print(f"Loading RAVDESS from: {self.data_dir}")
        self._load_ravdess()

        # Split into train/val
        rng = np.random.default_rng(42)
        indices = rng.permutation(len(self.samples))
        val_size = int(len(self.samples) * val_split)

        self.val_indices = set(indices[:val_size])
        self.train_indices = set(indices[val_size:])

        # Stats
        singing_count = sum(1 for s in self.samples if s.is_singing)
        emotion_counts = {}
        for s in self.samples:
            if s.emotion_label:
                emotion_counts[s.emotion_label] = emotion_counts.get(s.emotion_label, 0) + 1

        print(f"Total samples: {len(self.samples)}")
        print(f"Train: {len(self.train_indices)}, Val: {len(self.val_indices)}")
        print(f"Singing: {singing_count}, Speaking: {len(self.samples) - singing_count}")
        print(f"Emotions: {emotion_counts}")

    def _load_ravdess(self):
        """Load RAVDESS dataset."""
        # RAVDESS structure: Actor_XX/XX-XX-XX-XX-XX-XX-XX.wav
        audio_files = list(self.data_dir.rglob("*.wav"))

        if not audio_files:
            # Try looking in nested structure
            audio_files = list(self.data_dir.rglob("*.wav"))

        print(f"Found {len(audio_files)} audio files")

        for audio_file in audio_files:
            sample = self._parse_ravdess_filename(audio_file)
            if sample:
                self.samples.append(sample)

    def _parse_ravdess_filename(self, audio_path: Path) -> MultiModalSample | None:
        """Parse RAVDESS filename to extract labels."""
        filename = audio_path.stem  # Remove .wav

        # Parse: Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor
        parts = filename.split("-")
        if len(parts) < 7:
            return None

        try:
            modality = int(parts[0])
            vocal_channel = int(parts[1])
            emotion_raw = int(parts[2])
            int(parts[3])
            statement = int(parts[4])
            int(parts[5])
            int(parts[6])
        except ValueError:
            return None

        # Skip video-only
        if modality == 2:
            return None

        # Vocal channel: 1 = speech, 2 = song
        is_singing = vocal_channel == 2

        # Map emotion
        if emotion_raw not in self.EMOTION_MAP:
            return None
        emotion_id = self.EMOTION_MAP[emotion_raw]
        emotion_label = RAVDESS_EMOTIONS[emotion_id]

        # Get transcript
        transcript = self.STATEMENTS.get(statement, "")

        return MultiModalSample(
            audio_path=str(audio_path),
            transcript=transcript,
            emotion_id=emotion_id,
            emotion_label=emotion_label,
            is_singing=is_singing,
        )

    def get_train_samples(self) -> list[MultiModalSample]:
        return [self.samples[i] for i in self.train_indices]

    def get_val_samples(self) -> list[MultiModalSample]:
        return [self.samples[i] for i in self.val_indices]


class ExpressoDataset:
    """
    Expresso dataset loader - SOTA for expressive speech.

    40 hours of professional studio recordings with 34 expressive styles.
    Loaded from HuggingFace: ylacombe/expresso

    Features:
    - 34 expressive styles (vs RAVDESS's 8)
    - Both read speech and improvised dialogues
    - Text transcriptions included
    - Professional 48kHz audio
    """

    def __init__(
        self,
        cache_dir: str | None = None,
        config: str = "read",  # "read" for read speech, can expand later
        max_audio_len: float = 15.0,
        val_split: float = 0.1,
    ):
        self.max_audio_len = max_audio_len
        self.samples: list[MultiModalSample] = []

        print("Loading Expresso dataset from HuggingFace...")
        self._load_expresso(cache_dir, config)

        # Split
        rng = np.random.default_rng(42)
        indices = rng.permutation(len(self.samples))
        val_size = int(len(self.samples) * val_split)

        self.val_indices = set(indices[:val_size])
        self.train_indices = set(indices[val_size:])

        # Stats
        style_counts = {}
        singing_count = 0
        for s in self.samples:
            if s.emotion_label:
                style_counts[s.emotion_label] = style_counts.get(s.emotion_label, 0) + 1
            if s.is_singing:
                singing_count += 1

        print(f"Total samples: {len(self.samples)}")
        print(f"Train: {len(self.train_indices)}, Val: {len(self.val_indices)}")
        print(f"Singing: {singing_count}, Speaking: {len(self.samples) - singing_count}")
        print(f"Styles: {len(style_counts)} unique")
        for style, count in sorted(style_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  {style}: {count}")

    def _load_expresso(self, cache_dir: str | None, config: str):
        """Load Expresso from HuggingFace."""
        try:
            from datasets import load_dataset
        except ImportError:
            print("ERROR: datasets library required. Install: pip install datasets")
            return

        # Load dataset
        kwargs = {}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir

        try:
            # Load dataset - use index access to handle corrupted files
            dataset = load_dataset("ylacombe/expresso", config, **kwargs)
        except Exception as e:
            print(f"ERROR loading Expresso: {e}")
            print("Try: pip install datasets soundfile")
            return

        # Process samples with index-based access for error handling
        error_count = 0
        success_count = 0
        for split in dataset:
            split_ds = dataset[split]
            n_items = len(split_ds)
            print(f"  Processing {split} split: {n_items} items")
            for i in range(n_items):
                try:
                    item = split_ds[i]
                    sample = self._process_item(item)
                    if sample:
                        self.samples.append(sample)
                        success_count += 1
                        if success_count % 500 == 0:
                            print(f"    Loaded {success_count} samples...")
                except RuntimeError as e:
                    # Skip corrupted audio files (torchcodec buffer errors)
                    error_count += 1
                    if error_count <= 5:
                        print(f"  Skipping corrupted audio {i}: {str(e)[:50]}...")
                    elif error_count == 6:
                        print("  (suppressing further error messages)")
                except Exception as e:
                    error_count += 1
                    if error_count <= 5:
                        print(f"  Error processing item {i}: {str(e)[:50]}...")

        if error_count > 0:
            print(f"  Loaded {success_count} samples, skipped {error_count} corrupted files")

    def _process_item(self, item: dict) -> MultiModalSample | None:
        """Process a single Expresso item."""
        # Extract audio
        audio_data = item.get("audio", {})
        if not audio_data:
            return None

        # Get audio array and sample rate
        # Handle dict, AudioDecoder (torchcodec), and AudioFile formats
        try:
            if hasattr(audio_data, "get_all_samples"):
                # torchcodec AudioDecoder format (newer HuggingFace datasets)
                samples = audio_data.get_all_samples()
                audio_array = samples.data.numpy()
                if audio_array.ndim > 1:
                    audio_array = audio_array.mean(axis=0)  # Convert stereo to mono
                sample_rate = samples.sample_rate
            elif hasattr(audio_data, "get"):
                # Standard dict format
                audio_array = audio_data.get("array")
                sample_rate = audio_data.get("sampling_rate", 48000)
            elif hasattr(audio_data, "__getitem__"):
                # AudioDecoder format with subscript access
                audio_array = audio_data["array"]
                sample_rate = audio_data.get("sampling_rate", 48000)
            else:
                print(f"WARNING: Unknown audio format: {type(audio_data)}")
                return None
        except Exception as e:
            print(f"WARNING: Could not extract audio: {e}")
            return None

        if audio_array is None:
            return None

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            # Simple decimation for 48kHz -> 16kHz
            if sample_rate == 48000:
                audio_array = audio_array[::3]  # Downsample by 3
            else:
                # Use scipy for other rates
                try:
                    from scipy import signal
                    num_samples = int(len(audio_array) * 16000 / sample_rate)
                    audio_array = signal.resample(audio_array, num_samples)
                except ImportError:
                    # Fallback: simple decimation
                    ratio = sample_rate // 16000
                    audio_array = audio_array[::ratio] if ratio > 1 else audio_array

        # Check duration
        duration = len(audio_array) / 16000
        if duration > self.max_audio_len:
            return None

        # Get style and map to emotion ID
        style = item.get("style", "default").lower()
        emotion_id = EXPRESSO_STYLE_MAP.get(style, 15)  # Default to "default" style
        emotion_label = EXTENDED_EMOTIONS[emotion_id] if emotion_id < len(EXTENDED_EMOTIONS) else style

        # Detect singing
        is_singing = style == "singing"

        # Get transcript
        transcript = item.get("text", "")

        # Store audio data directly in sample for in-memory processing
        return MultiModalSample(
            audio_path="__in_memory__",  # Marker for in-memory audio
            transcript=transcript,
            emotion_id=emotion_id,
            emotion_label=emotion_label,
            is_singing=is_singing,
            duration=duration,
            audio_array=np.array(audio_array, dtype=np.float32),  # Store audio
        )

    def get_train_samples(self) -> list[MultiModalSample]:
        return [self.samples[i] for i in self.train_indices]

    def get_val_samples(self) -> list[MultiModalSample]:
        return [self.samples[i] for i in self.val_indices]


class VocalSetDataset:
    """
    VocalSet dataset loader - professional singing techniques.

    10.1 hours from 20 professional singers.
    Download from: https://zenodo.org/record/1193957

    Singing techniques: vibrato, straight, belt, breathy, lip_trill, etc.
    """

    # Singing technique categories (17 total from VocalSet)
    TECHNIQUES = [
        "belt", "breathy", "fast_forte", "fast_piano", "forte",
        "inhaled", "lip_trill", "messa", "pp", "slow_forte",
        "slow_piano", "spoken", "straight", "trill", "trillo",
        "vibrato", "vocal_fry",
    ]
    TECHNIQUE_TO_ID = {t: i for i, t in enumerate(TECHNIQUES)}

    # Intensity mapping based on technique characteristics
    # Higher intensity: belt, forte, fast_forte, slow_forte
    # Lower intensity: breathy, pp, inhaled, spoken
    TECHNIQUE_INTENSITY = {
        "belt": 0.9, "breathy": 0.3, "fast_forte": 0.85, "fast_piano": 0.4,
        "forte": 0.85, "inhaled": 0.2, "lip_trill": 0.5, "messa": 0.6,
        "pp": 0.2, "slow_forte": 0.8, "slow_piano": 0.35, "spoken": 0.4,
        "straight": 0.6, "trill": 0.7, "trillo": 0.7, "vibrato": 0.65,
        "vocal_fry": 0.25,
    }

    def __init__(
        self,
        data_dir: str,
        max_audio_len: float = 15.0,
        val_split: float = 0.1,
    ):
        self.data_dir = Path(data_dir)
        self.max_audio_len = max_audio_len
        self.samples: list[MultiModalSample] = []

        print(f"Loading VocalSet from: {self.data_dir}")
        self._load_vocalset()

        # Split
        rng = np.random.default_rng(42)
        indices = rng.permutation(len(self.samples))
        val_size = int(len(self.samples) * val_split)

        self.val_indices = set(indices[:val_size])
        self.train_indices = set(indices[val_size:])

        print(f"Total samples: {len(self.samples)}")
        print(f"Train: {len(self.train_indices)}, Val: {len(self.val_indices)}")
        print(f"Techniques: {len(self.TECHNIQUES)}, with style and intensity labels")

    def _load_vocalset(self):
        """Load VocalSet audio files with style and intensity labels."""
        # VocalSet structure: FULL/singer/category/technique/...wav
        audio_files = list(self.data_dir.rglob("*.wav"))
        print(f"Found {len(audio_files)} audio files")

        style_counts = {}
        for audio_file in audio_files:
            # Extract technique from path (folder containing audio)
            # Path: .../singer/category/technique/file.wav
            parts = audio_file.parts
            technique = None
            for part in reversed(parts[:-1]):  # Skip filename
                normalized = part.lower().replace("vibrado", "vibrato")  # Fix typo
                if normalized in self.TECHNIQUE_TO_ID:
                    technique = normalized
                    break

            style_id = self.TECHNIQUE_TO_ID.get(technique, -1) if technique else -1
            intensity = self.TECHNIQUE_INTENSITY.get(technique, 0.5) if technique else 0.5

            # Track style distribution
            if technique:
                style_counts[technique] = style_counts.get(technique, 0) + 1

            self.samples.append(MultiModalSample(
                audio_path=str(audio_file),
                transcript="",  # No text for singing samples
                is_singing=True,
                singing_style=style_id,
                singing_intensity=intensity,
                emotion_id=14,  # "singing" in extended taxonomy
                emotion_label="singing",
            ))

        print(f"Style distribution: {style_counts}")

    def get_train_samples(self) -> list[MultiModalSample]:
        return [self.samples[i] for i in self.train_indices]

    def get_val_samples(self) -> list[MultiModalSample]:
        return [self.samples[i] for i in self.val_indices]


class LibriSpeechDataset:
    """LibriSpeech dataset for CTC training (text only)."""

    def __init__(
        self,
        data_dir: str,
        max_audio_len: float = 15.0,
        val_split: float = 0.1,
    ):
        self.data_dir = Path(data_dir)
        self.max_audio_len = max_audio_len
        self.samples: list[MultiModalSample] = []

        print(f"Loading LibriSpeech from: {self.data_dir}")
        self._load_librispeech()

        # Split
        rng = np.random.default_rng(42)
        indices = rng.permutation(len(self.samples))
        val_size = int(len(self.samples) * val_split)

        self.val_indices = set(indices[:val_size])
        self.train_indices = set(indices[val_size:])

        print(f"Total samples: {len(self.samples)}")
        print(f"Train: {len(self.train_indices)}, Val: {len(self.val_indices)}")

    def _load_librispeech(self):
        """Load LibriSpeech format."""
        trans_files = list(self.data_dir.rglob("*.trans.txt"))
        print(f"Found {len(trans_files)} transcript files")

        for trans_file in trans_files:
            chapter_dir = trans_file.parent

            with open(trans_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split(" ", 1)
                    if len(parts) < 2:
                        continue

                    utterance_id = parts[0]
                    transcript = parts[1]

                    # Find audio
                    audio_path = chapter_dir / f"{utterance_id}.flac"
                    if not audio_path.exists():
                        audio_path = chapter_dir / f"{utterance_id}.wav"

                    if audio_path.exists():
                        self.samples.append(MultiModalSample(
                            audio_path=str(audio_path),
                            transcript=transcript,
                            language="en",
                            is_singing=False,  # LibriSpeech is speech only
                        ))

    def get_train_samples(self) -> list[MultiModalSample]:
        return [self.samples[i] for i in self.train_indices]

    def get_val_samples(self) -> list[MultiModalSample]:
        return [self.samples[i] for i in self.val_indices]


# Mapping from prosody_type in JSON manifests to EXTENDED_EMOTIONS indices
# The JSON manifests use different indices than EXTENDED_EMOTIONS
PROSODY_TYPE_TO_EXTENDED = {
    0: 0,   # neutral (also used for calm/surprise in some datasets)
    40: 4,  # angry
    41: 3,  # sad
    42: 2,  # happy
    48: 6,  # disgust
    49: 5,  # fearful
    50: 7,  # surprise
}


class ConsolidatedProsodyDataset:
    """
    Consolidated prosody dataset loader.

    Loads from JSON manifest files containing emotion-annotated samples from
    multiple sources (crema-d, esd, jvnv, ravdess, resd, emns).

    Expected JSON format per entry:
    {
        "text": "transcript",
        "annotated_text": "<emotion type='angry'>transcript</emotion>",
        "prosody_type": 40,  # See PROSODY_TYPE_TO_EXTENDED mapping
        "audio_path": "/absolute/path/to/audio.wav",
        "duration_s": 2.5,
        "source": "crema-d"
    }

    Supports loading from:
    - data/prosody/crema.json (CREMA-D: 7,442 samples, 6 emotions)
    - data/prosody/esd.json (ESD: 17,500 samples, 5 emotions)
    - data/prosody/jvnv.json (JVNV: 1,615 samples, 6 emotions)
    - data/prosody/ravdess.json (RAVDESS: 1,440 samples, 7 emotions)
    """

    # Manifest files to load
    MANIFEST_FILES = [
        "crema.json",
        "esd.json",
        "jvnv.json",
        "ravdess.json",
    ]

    def __init__(
        self,
        data_dir: str,
        max_audio_len: float = 15.0,
        val_split: float = 0.1,
        manifests: list[str] | None = None,
    ):
        """
        Initialize consolidated prosody dataset.

        Args:
            data_dir: Directory containing JSON manifest files (e.g., data/prosody)
            max_audio_len: Maximum audio length in seconds
            val_split: Fraction of data to use for validation
            manifests: Optional list of specific manifest files to load
        """
        self.data_dir = Path(data_dir)
        self.max_audio_len = max_audio_len
        self.samples: list[MultiModalSample] = []

        # Use provided manifests or defaults
        manifest_files = manifests or self.MANIFEST_FILES

        print(f"Loading consolidated prosody from: {self.data_dir}")
        self._load_manifests(manifest_files)

        # Split into train/val
        rng = np.random.default_rng(42)
        indices = rng.permutation(len(self.samples))
        val_size = int(len(self.samples) * val_split)

        self.val_indices = set(indices[:val_size])
        self.train_indices = set(indices[val_size:])

        # Print stats
        self._print_stats()

    def _load_manifests(self, manifest_files: list[str]):
        """Load samples from JSON manifest files."""
        import json

        for manifest_file in manifest_files:
            manifest_path = self.data_dir / manifest_file
            if not manifest_path.exists():
                print(f"  Skipping missing manifest: {manifest_file}")
                continue

            print(f"  Loading {manifest_file}...")
            try:
                with open(manifest_path, encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"  Error loading {manifest_file}: {e}")
                continue

            loaded = 0
            skipped = 0
            for entry in data:
                sample = self._process_entry(entry)
                if sample:
                    self.samples.append(sample)
                    loaded += 1
                else:
                    skipped += 1

            print(f"    Loaded: {loaded}, Skipped: {skipped}")

    def _process_entry(self, entry: dict) -> MultiModalSample | None:
        """Process a single JSON entry into a MultiModalSample."""
        # Required fields
        audio_path = entry.get("audio_path", "")
        if not audio_path or not Path(audio_path).exists():
            return None

        # Check duration
        duration = entry.get("duration_s", 0)
        if duration > self.max_audio_len:
            return None

        # Map prosody_type to emotion ID
        prosody_type = entry.get("prosody_type", -1)
        emotion_id = PROSODY_TYPE_TO_EXTENDED.get(prosody_type, 0)

        # Get emotion label
        if emotion_id < len(EXTENDED_EMOTIONS):
            emotion_label = EXTENDED_EMOTIONS[emotion_id]
        else:
            emotion_label = f"emotion_{emotion_id}"

        # Get transcript
        transcript = entry.get("text", "")

        # Determine if singing (prosody_type 14 in extended taxonomy)
        # Note: None of these datasets have singing, all are speech
        is_singing = False

        # Extract precomputed f0_contour if available (normalized 0-1)
        pitch_hz = None
        voiced_mask = None
        f0_contour = entry.get("f0_contour")
        if f0_contour is not None and len(f0_contour) > 0:
            f0_min = entry.get("f0_min", 50.0)
            f0_max = entry.get("f0_max", 500.0)
            # Convert normalized contour to Hz
            f0_contour = np.array(f0_contour, dtype=np.float32)
            # Denormalize: pitch_hz = f0_min + contour * (f0_max - f0_min)
            pitch_hz = f0_min + f0_contour * (f0_max - f0_min)
            # Values near f0_min (or 0 in contour) are unvoiced
            voiced_mask = (f0_contour > 0.01).astype(np.float32)

        return MultiModalSample(
            audio_path=audio_path,
            transcript=transcript,
            duration=duration,
            emotion_id=emotion_id,
            emotion_label=emotion_label,
            is_singing=is_singing,
            language="en",  # Most prosody datasets are English
            pitch_hz=pitch_hz,
            voiced_mask=voiced_mask,
        )

    def _print_stats(self):
        """Print dataset statistics."""
        print(f"Total samples: {len(self.samples)}")
        print(f"Train: {len(self.train_indices)}, Val: {len(self.val_indices)}")

        # Emotion distribution
        emotion_counts = {}
        for s in self.samples:
            if s.emotion_label:
                emotion_counts[s.emotion_label] = emotion_counts.get(s.emotion_label, 0) + 1

        print("Emotion distribution:")
        for emotion in sorted(emotion_counts.keys()):
            count = emotion_counts[emotion]
            pct = 100 * count / len(self.samples)
            print(f"  {emotion}: {count} ({pct:.1f}%)")

    def get_train_samples(self) -> list[MultiModalSample]:
        return [self.samples[i] for i in self.train_indices]

    def get_val_samples(self) -> list[MultiModalSample]:
        return [self.samples[i] for i in self.val_indices]


class UnifiedEmotionDataset:
    """
    Unified emotion dataset loader.

    Loads from HuggingFace dataset format created by create_unified_emotion_dataset.py
    Expected columns: audio, emotion, emotion_id, source, language
    """

    # Map unified emotion names to EXTENDED_EMOTIONS indices
    EMOTION_MAP = {
        "neutral": 0,
        "happy": 2,
        "sad": 3,
        "angry": 4,
        "fear": 5,
        "disgust": 6,
        "surprise": 7,
        "calm": 1,
        "excited": 9,  # Map to "singing" slot temporarily
        "other": 8,    # Map to "contempt" slot temporarily
    }

    def __init__(
        self,
        data_dir: str,
        max_audio_len: float = 15.0,
    ):
        self.data_dir = Path(data_dir)
        self.max_audio_len = max_audio_len
        self.train_samples: list[MultiModalSample] = []
        self.val_samples: list[MultiModalSample] = []

        print(f"Loading unified emotion dataset from: {self.data_dir}")
        self._load_dataset()

    def _load_dataset(self):
        """Load unified emotion dataset."""
        try:
            from datasets import load_from_disk
        except ImportError:
            print("ERROR: datasets library required")
            return

        try:
            ds = load_from_disk(str(self.data_dir))
        except Exception as e:
            print(f"ERROR loading dataset: {e}")
            return

        for split, samples_list in [("train", self.train_samples), ("validation", self.val_samples)]:
            if split not in ds:
                continue

            split_ds = ds[split]
            print(f"  Loading {split}: {len(split_ds)} samples")

            loaded = 0
            skipped = 0
            for i in range(len(split_ds)):
                try:
                    item = split_ds[i]
                    sample = self._process_item(item)
                    if sample:
                        samples_list.append(sample)
                        loaded += 1
                    else:
                        skipped += 1
                except Exception as e:
                    skipped += 1
                    if skipped <= 5:
                        print(f"    Error at {i}: {str(e)[:50]}")

            print(f"    Loaded: {loaded}, Skipped: {skipped}")

    def _process_item(self, item: dict) -> MultiModalSample | None:
        """Process a single item."""
        audio_data = item.get("audio")
        if audio_data is None:
            return None

        # Get audio array - handle different formats
        audio_array = None
        sample_rate = 16000

        try:
            # New torchcodec AudioDecoder format
            if hasattr(audio_data, 'get_all_samples'):
                samples = audio_data.get_all_samples()
                audio_tensor = samples.data  # torch.Tensor [channels, samples]
                audio_array = audio_tensor.squeeze().numpy()
                sample_rate = samples.sample_rate
            # Old dict format with 'array' key
            elif isinstance(audio_data, dict):
                audio_array = audio_data.get("array")
                sample_rate = audio_data.get("sampling_rate", 16000)
                if audio_array is not None:
                    audio_array = np.array(audio_array)
            else:
                return None
        except Exception:
            return None

        if audio_array is None or len(audio_array) == 0:
            return None

        # Check duration
        duration = len(audio_array) / sample_rate
        if duration > self.max_audio_len:
            return None

        # Get emotion
        emotion = item.get("emotion", "neutral")
        emotion_id = self.EMOTION_MAP.get(emotion, 0)

        return MultiModalSample(
            audio_path="__in_memory__",
            transcript="",
            emotion_id=emotion_id,
            emotion_label=emotion,
            is_singing=False,  # Emotion dataset is speech only
            duration=duration,
            audio_array=np.array(audio_array, dtype=np.float32),
            language=item.get("language", "en"),
        )

    def get_train_samples(self) -> list[MultiModalSample]:
        return self.train_samples

    def get_val_samples(self) -> list[MultiModalSample]:
        return self.val_samples


class UnifiedSingingDataset:
    """
    Unified singing dataset loader.

    Loads from HuggingFace dataset format created by create_unified_singing_dataset.py
    Expected columns: audio, is_singing, transcription, language, source
    """

    def __init__(
        self,
        data_dir: str,
        max_audio_len: float = 15.0,
    ):
        self.data_dir = Path(data_dir)
        self.max_audio_len = max_audio_len
        self.train_samples: list[MultiModalSample] = []
        self.val_samples: list[MultiModalSample] = []

        print(f"Loading unified singing dataset from: {self.data_dir}")
        self._load_dataset()

    def _load_dataset(self):
        """Load unified singing dataset."""
        try:
            from datasets import load_from_disk
        except ImportError:
            print("ERROR: datasets library required")
            return

        try:
            ds = load_from_disk(str(self.data_dir))
        except Exception as e:
            print(f"ERROR loading dataset: {e}")
            return

        for split, samples_list in [("train", self.train_samples), ("validation", self.val_samples)]:
            if split not in ds:
                continue

            split_ds = ds[split]
            print(f"  Loading {split}: {len(split_ds)} samples")

            loaded = 0
            skipped = 0
            for i in range(len(split_ds)):
                try:
                    item = split_ds[i]
                    sample = self._process_item(item)
                    if sample:
                        samples_list.append(sample)
                        loaded += 1
                    else:
                        skipped += 1
                except Exception as e:
                    skipped += 1
                    if skipped <= 5:
                        print(f"    Error at {i}: {str(e)[:50]}")

            print(f"    Loaded: {loaded}, Skipped: {skipped}")

    def _process_item(self, item: dict) -> MultiModalSample | None:
        """Process a single item."""
        audio_data = item.get("audio")
        audio_path = item.get("audio_path")

        # Handle audio_path (file path) format
        if audio_path is not None and audio_data is None:
            # Just store the path, load later
            is_singing = item.get("is_singing", 1) == 1
            return MultiModalSample(
                audio_path=audio_path,
                transcript=item.get("transcription", ""),
                is_singing=is_singing,
            )

        if audio_data is None:
            return None

        # Get audio array - handle different formats
        audio_array = None
        sample_rate = 16000

        try:
            # New torchcodec AudioDecoder format
            if hasattr(audio_data, 'get_all_samples'):
                samples = audio_data.get_all_samples()
                audio_tensor = samples.data  # torch.Tensor [channels, samples]
                audio_array = audio_tensor.squeeze().numpy()
                sample_rate = samples.sample_rate
            # Old dict format with 'array' key
            elif isinstance(audio_data, dict):
                audio_array = audio_data.get("array")
                sample_rate = audio_data.get("sampling_rate", 16000)
                if audio_array is not None:
                    audio_array = np.array(audio_array)
            else:
                return None
        except Exception:
            return None

        if audio_array is None or len(audio_array) == 0:
            return None

        # Check duration
        duration = len(audio_array) / sample_rate
        if duration > self.max_audio_len:
            return None

        is_singing = item.get("is_singing", True)

        return MultiModalSample(
            audio_path="__in_memory__",
            transcript=item.get("transcription", ""),
            emotion_id=14 if is_singing else 0,  # "singing" in extended taxonomy
            emotion_label="singing" if is_singing else "neutral",
            is_singing=is_singing,
            duration=duration,
            audio_array=np.array(audio_array, dtype=np.float32),
            language=item.get("language", "zh"),
        )

    def get_train_samples(self) -> list[MultiModalSample]:
        return self.train_samples

    def get_val_samples(self) -> list[MultiModalSample]:
        return self.val_samples


def extract_pitch_with_librosa(audio: np.ndarray, sample_rate: int = 16000) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract pitch (F0) from audio using librosa.

    Returns:
        pitch_hz: (T,) F0 in Hz (0 for unvoiced)
        voiced_mask: (T,) binary mask for voiced frames
    """
    try:
        import librosa
    except ImportError:
        # Return zeros if librosa not available
        # Whisper uses 50Hz frame rate (30s -> 1500 frames)
        n_frames = int(len(audio) / sample_rate * 50)
        return np.zeros(n_frames), np.zeros(n_frames)

    # Use pyin for pitch tracking
    pitch_hz, voiced_flag, voiced_prob = librosa.pyin(
        audio,
        fmin=50,
        fmax=800,
        sr=sample_rate,
        frame_length=2048,
        hop_length=int(sample_rate / 50),  # Match Whisper frame rate (50Hz)
    )

    # Replace NaN with 0
    pitch_hz = np.nan_to_num(pitch_hz, nan=0.0)
    voiced_mask = voiced_flag.astype(np.float32)

    return pitch_hz, voiced_mask


class MultiHeadTrainer:
    """
    Trainer for multi-head architecture.

    Supports joint training of CTC, emotion, singing, and pitch heads.
    """

    def __init__(
        self,
        config: MultiHeadTrainingConfig,
        whisper_model: WhisperMLX,
        multi_head: WhisperMultiHead,
        ctc_head: CTCDraftHead,
        tokenizer,
    ):
        self.config = config
        self.whisper_model = whisper_model
        self.multi_head = multi_head
        self.ctc_head = ctc_head
        self.tokenizer = tokenizer

        # Separate optimizers per head to avoid state collision
        # (Each head has fc1/fc2 with different shapes; shared optimizer state would collide)
        self.ctc_optimizer = optim.AdamW(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        ) if config.train_ctc else None

        self.emotion_optimizer = optim.AdamW(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        ) if config.train_emotion else None

        self.singing_optimizer = optim.AdamW(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        ) if config.train_singing else None

        self.pitch_optimizer = optim.AdamW(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        ) if config.train_pitch else None

        # Training state
        self.step = 0
        self.epoch = 0
        self.best_loss = float("inf")
        self.total_steps = 0  # Will be set in train()

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

    def _populate_sample_durations(self, samples: list[MultiModalSample]) -> None:
        """
        Populate duration field for samples that have duration=0.

        Uses soundfile.info() for fast metadata-only access.
        Modifies samples in-place.
        """
        samples_needing_duration = [s for s in samples if s.duration == 0.0]
        if not samples_needing_duration:
            return

        self.log(f"Populating durations for {len(samples_needing_duration)} samples...")
        start_time = time.time()

        for i, sample in enumerate(samples_needing_duration):
            sample.duration = get_audio_duration(sample.audio_path)
            if (i + 1) % 10000 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                self.log(f"  {i + 1}/{len(samples_needing_duration)} ({rate:.0f} samples/sec)")

        elapsed = time.time() - start_time
        self.log(f"Duration population complete in {elapsed:.1f}s")

    def _create_length_sorted_batches(
        self, samples: list[MultiModalSample],
    ) -> list[list[MultiModalSample]]:
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
            samples: List of MultiModalSample (must have duration populated)

        Returns:
            List of batches, each batch is a list of MultiModalSample
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

    def _get_learning_rate(self) -> float:
        """Compute current learning rate with warmup and decay."""
        import math

        base_lr = self.config.learning_rate
        warmup = self.config.warmup_steps

        # Warmup phase: linear increase from 0 to base_lr
        if self.step < warmup:
            return base_lr * (self.step / warmup)

        # After warmup: apply decay
        if self.config.lr_decay == "none":
            return base_lr

        # Progress after warmup (0 to 1)
        steps_after_warmup = self.step - warmup
        total_decay_steps = max(self.total_steps - warmup, 1)
        progress = min(steps_after_warmup / total_decay_steps, 1.0)

        if self.config.lr_decay == "cosine":
            # Cosine decay from base_lr to min_lr
            min_lr = base_lr * self.config.min_lr_ratio
            return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
        if self.config.lr_decay == "linear":
            # Linear decay from base_lr to min_lr
            min_lr = base_lr * self.config.min_lr_ratio
            return base_lr - (base_lr - min_lr) * progress
        return base_lr

    def _update_optimizer_lr(self, lr: float):
        """Update learning rate for all optimizers."""
        for opt in [self.ctc_optimizer, self.emotion_optimizer,
                    self.singing_optimizer, self.pitch_optimizer]:
            if opt is not None:
                opt.learning_rate = lr

    def _encode_audio(self, audio: np.ndarray) -> mx.array:
        """Encode audio through frozen Whisper encoder."""
        # Mel spectrogram
        mel = log_mel_spectrogram(audio)

        # Pad/trim to 30s (3000 frames for mel, 1500 for encoder output)
        target_frames = 3000
        if mel.shape[0] < target_frames:
            pad_len = target_frames - mel.shape[0]
            mel = np.pad(mel, ((0, pad_len), (0, 0)), mode='constant')
        else:
            mel = mel[:target_frames]

        # Convert to MLX - shape (n_frames, n_mels) = (3000, n_mels)
        # NOTE: encoder expects (batch, n_frames, n_mels) despite embed_audio docstring
        mel_mx = mx.array(mel)[None, ...]  # (1, 3000, n_mels)

        # Run encoder
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
        # Mel spectrogram
        mel = log_mel_spectrogram(audio)
        actual_mel_frames = mel.shape[0]

        # Pad/trim to 30s (3000 frames for mel, 1500 for encoder output)
        target_frames = 3000
        if mel.shape[0] < target_frames:
            pad_len = target_frames - mel.shape[0]
            mel = np.pad(mel, ((0, pad_len), (0, 0)), mode='constant')
        else:
            mel = mel[:target_frames]
            actual_mel_frames = target_frames

        # Convert to MLX
        mel_mx = mx.array(mel)[None, ...]  # (1, 3000, n_mels)

        # Run encoder
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
        samples: list[MultiModalSample],
    ) -> tuple[mx.array, dict]:
        """
        Prepare a batch for training.

        Returns:
            encoder_outputs: (batch, T, d_model)
            labels: Dict of labels for each head
        """
        batch_encoder_outputs = []
        ctc_targets = []
        ctc_input_lengths = []
        ctc_target_lengths = []
        emotion_targets = []
        singing_targets = []
        singing_style_targets = []
        singing_intensity_targets = []
        pitch_targets = []
        voiced_targets = []

        for sample in samples:
            # Load audio (from file or in-memory)
            if sample.audio_array is not None:
                # In-memory audio (e.g., from HuggingFace Expresso)
                audio = sample.audio_array
                audio_path_for_cache = None  # Can't cache in-memory audio
            else:
                # Load from file
                audio = load_audio(sample.audio_path)
                audio_path_for_cache = sample.audio_path

            # Trim to max length
            max_samples = int(self.config.max_audio_len * self.config.sample_rate)
            if len(audio) > max_samples:
                audio = audio[:max_samples]

            # Encode (with caching if audio_path is available)
            if self.encoder_cache is not None and audio_path_for_cache is not None:
                # Use cached version
                enc_out, actual_frames = self._encode_audio_cached(audio, audio_path_for_cache)
                batch_encoder_outputs.append(enc_out)
            else:
                # Original path - no caching
                # Track actual audio length for CTC
                actual_frames = min(int(len(audio) / self.config.sample_rate * 50), 1500)
                encoder_output = self._encode_audio(audio)
                batch_encoder_outputs.append(encoder_output[0])  # Remove batch dim

            # CTC labels
            if sample.transcript and self.config.train_ctc:
                tokens = self.tokenizer.encode(sample.transcript)
                ctc_targets.append(tokens)
                ctc_input_lengths.append(actual_frames)
                ctc_target_lengths.append(len(tokens))
            else:
                ctc_targets.append([])
                ctc_input_lengths.append(0)
                ctc_target_lengths.append(0)

            # Emotion labels
            if sample.emotion_id is not None:
                emotion_targets.append(sample.emotion_id)
            else:
                emotion_targets.append(-1)  # Invalid

            # Singing labels
            if sample.is_singing is not None:
                singing_targets.append(1 if sample.is_singing else 0)
            else:
                singing_targets.append(-1)  # Invalid

            # Singing style labels (for extended singing head)
            if sample.singing_style is not None and sample.singing_style >= 0:
                singing_style_targets.append(sample.singing_style)
            else:
                singing_style_targets.append(-1)  # Invalid

            # Singing intensity labels (for extended singing head)
            if sample.singing_intensity is not None:
                singing_intensity_targets.append(sample.singing_intensity)
            else:
                singing_intensity_targets.append(-1.0)  # Invalid

            # Pitch labels (use precomputed if available, else extract from audio)
            if self.config.train_pitch:
                if sample.pitch_hz is not None and sample.voiced_mask is not None:
                    # Use precomputed pitch from prosody JSON
                    pitch_hz = sample.pitch_hz
                    voiced_mask = sample.voiced_mask
                else:
                    # Fall back to librosa extraction (slow)
                    pitch_hz, voiced_mask = extract_pitch_with_librosa(audio)
                # Pad to 1500 frames
                if len(pitch_hz) < 1500:
                    pitch_hz = np.pad(pitch_hz, (0, 1500 - len(pitch_hz)))
                    voiced_mask = np.pad(voiced_mask, (0, 1500 - len(voiced_mask)))
                else:
                    pitch_hz = pitch_hz[:1500]
                    voiced_mask = voiced_mask[:1500]
                pitch_targets.append(pitch_hz)
                voiced_targets.append(voiced_mask)

        # Stack encoder outputs
        encoder_outputs = mx.stack(batch_encoder_outputs)

        labels = {
            "ctc_targets": ctc_targets,
            "ctc_input_lengths": ctc_input_lengths,
            "ctc_target_lengths": ctc_target_lengths,
            "emotion_targets": mx.array(emotion_targets),
            "singing_targets": mx.array(singing_targets),
            "singing_style_targets": mx.array(singing_style_targets),
            "singing_intensity_targets": mx.array(singing_intensity_targets),
        }

        if pitch_targets:
            labels["pitch_targets"] = mx.array(np.stack(pitch_targets))
            labels["voiced_targets"] = mx.array(np.stack(voiced_targets))

        return encoder_outputs, labels

    def _compute_loss(
        self,
        encoder_outputs: mx.array,
        labels: dict,
    ) -> tuple[mx.array, dict[str, float]]:
        """
        Compute combined loss for all heads.

        Returns:
            total_loss: Combined weighted loss
            loss_dict: Individual losses for logging
        """
        loss_dict = {}
        total_loss = mx.array(0.0)

        # CTC loss
        if self.config.train_ctc and any(labels["ctc_target_lengths"]):
            ctc_logits = self.ctc_head(encoder_outputs)

            # Use MLX CTC loss (or PyTorch if available)
            ctc_loss_value = self._compute_ctc_loss(
                ctc_logits,
                labels["ctc_targets"],
                labels["ctc_input_lengths"],
                labels["ctc_target_lengths"],
            )
            loss_dict["ctc"] = float(ctc_loss_value)
            total_loss = total_loss + self.config.ctc_loss_weight * ctc_loss_value

        # Emotion loss
        if self.config.train_emotion:
            valid_mask = labels["emotion_targets"] >= 0
            num_valid = mx.sum(valid_mask.astype(mx.float32))
            if num_valid > 0:
                emotion_logits = self.multi_head.emotion_head(encoder_outputs)

                # Clamp invalid targets to 0 for safe indexing (will be masked out)
                safe_targets = mx.maximum(labels["emotion_targets"], 0)

                # Masked loss: compute loss for all, multiply by mask, average over valid
                per_sample_loss = emotion_loss(emotion_logits, safe_targets, reduction="none")
                emotion_loss_value = mx.sum(per_sample_loss * valid_mask.astype(mx.float32)) / num_valid

                loss_dict["emotion"] = float(emotion_loss_value)
                total_loss = total_loss + self.config.emotion_loss_weight * emotion_loss_value

        # Singing loss
        if self.config.train_singing:
            valid_mask = labels["singing_targets"] >= 0
            num_valid = mx.sum(valid_mask.astype(mx.float32))
            if num_valid > 0:
                singing_result = self.multi_head.singing_head(encoder_outputs)

                # ExtendedSingingHead returns (singing_logit, style_logits, intensity)
                # Regular SingingHead returns just logits
                if isinstance(singing_result, tuple):
                    singing_logits, style_logits, intensity_pred = singing_result

                    # 1. Binary singing/speaking loss
                    safe_targets = mx.maximum(labels["singing_targets"], 0)
                    per_sample_loss = singing_loss(singing_logits, safe_targets, reduction="none")
                    per_sample_loss = per_sample_loss.squeeze(-1)
                    singing_loss_value = mx.sum(per_sample_loss * valid_mask.astype(mx.float32)) / num_valid

                    # 2. Style classification loss (only for samples with valid style labels)
                    style_valid_mask = labels["singing_style_targets"] >= 0
                    num_style_valid = mx.sum(style_valid_mask.astype(mx.float32))
                    style_loss_value = mx.array(0.0)
                    if num_style_valid > 0:
                        safe_style_targets = mx.maximum(labels["singing_style_targets"], 0)
                        # Cross-entropy loss for style classification
                        log_probs = log_softmax(style_logits, axis=-1)
                        # Gather the log probability for the correct class
                        batch_size = style_logits.shape[0]
                        per_sample_style_loss = -log_probs[mx.arange(batch_size), safe_style_targets]
                        style_loss_value = mx.sum(per_sample_style_loss * style_valid_mask.astype(mx.float32)) / num_style_valid
                        loss_dict["style"] = float(style_loss_value)

                    # 3. Intensity regression loss (MSE, only for samples with valid intensity)
                    intensity_valid_mask = labels["singing_intensity_targets"] >= 0
                    num_intensity_valid = mx.sum(intensity_valid_mask.astype(mx.float32))
                    intensity_loss_value = mx.array(0.0)
                    if num_intensity_valid > 0:
                        intensity_targets = mx.maximum(labels["singing_intensity_targets"], 0.0)
                        # MSE loss for intensity prediction
                        intensity_pred_squeezed = intensity_pred.squeeze(-1)
                        per_sample_intensity_loss = (intensity_pred_squeezed - intensity_targets) ** 2
                        intensity_loss_value = mx.sum(per_sample_intensity_loss * intensity_valid_mask.astype(mx.float32)) / num_intensity_valid
                        loss_dict["intensity"] = float(intensity_loss_value)

                    # Combined singing loss: binary + style + intensity
                    combined_singing_loss = singing_loss_value + 0.5 * style_loss_value + 0.3 * intensity_loss_value
                    loss_dict["singing"] = float(singing_loss_value)
                    total_loss = total_loss + self.config.singing_loss_weight * combined_singing_loss
                else:
                    # Regular SingingHead - just binary loss
                    singing_logits = singing_result
                    safe_targets = mx.maximum(labels["singing_targets"], 0)
                    per_sample_loss = singing_loss(singing_logits, safe_targets, reduction="none")
                    per_sample_loss = per_sample_loss.squeeze(-1)
                    singing_loss_value = mx.sum(per_sample_loss * valid_mask.astype(mx.float32)) / num_valid

                    loss_dict["singing"] = float(singing_loss_value)
                    total_loss = total_loss + self.config.singing_loss_weight * singing_loss_value

        # Pitch loss
        if self.config.train_pitch and "pitch_targets" in labels:
            if self.config.use_crepe_pitch:
                # CREPE pitch head: get logits for cross-entropy loss
                pitch_hz, voicing_prob, logits = self.multi_head.pitch_head(
                    encoder_outputs, return_bins=True,
                )
                pitch_loss_value = crepe_pitch_loss(
                    logits,
                    labels["pitch_targets"],
                    labels["voiced_targets"],
                )
            else:
                # MLP pitch head: MSE loss
                pitch_hz, voicing_prob = self.multi_head.pitch_head(encoder_outputs)
                pitch_loss_value = pitch_loss(
                    pitch_hz,
                    voicing_prob,
                    labels["pitch_targets"],
                    labels["voiced_targets"],
                )
            loss_dict["pitch"] = float(pitch_loss_value)
            total_loss = total_loss + self.config.pitch_loss_weight * pitch_loss_value

        return total_loss, loss_dict

    def _compute_ctc_loss(
        self,
        logits: mx.array,
        targets: list[list[int]],
        input_lengths: list[int],
        target_lengths: list[int],
    ) -> mx.array:
        """Compute CTC loss using best available method."""
        # Use soft alignment approximation (always differentiable)
        batch_size = logits.shape[0]
        total_loss = mx.array(0.0)
        num_valid = 0

        for b in range(batch_size):
            T = input_lengths[b]
            S = target_lengths[b]
            target_tokens = targets[b]

            if S == 0 or T == 0:
                continue

            sample_logits = logits[b, :T, :]
            log_probs = log_softmax(sample_logits, axis=-1)

            # Soft monotonic alignment
            frame_positions = mx.arange(T, dtype=mx.float32) / T
            token_positions = (mx.arange(S, dtype=mx.float32) + 0.5) / S

            sigma = 1.0 / S
            distances = mx.abs(
                mx.expand_dims(frame_positions, 1) - mx.expand_dims(token_positions, 0),
            )
            alignment_weights = mx.exp(-0.5 * (distances / sigma) ** 2)
            alignment_weights = alignment_weights / (mx.sum(alignment_weights, axis=1, keepdims=True) + 1e-8)

            target_array = mx.array(target_tokens)
            target_log_probs = log_probs[:, target_array]
            sample_loss = -mx.sum(alignment_weights * target_log_probs) / S

            total_loss = total_loss + sample_loss
            num_valid += 1

        return total_loss / max(num_valid, 1)

    def train_step(self, batch_samples: list[MultiModalSample]) -> dict[str, float]:
        """Single training step."""
        # Prepare batch
        encoder_outputs, labels = self._prepare_batch(batch_samples)

        # Define loss function for gradient computation
        def loss_fn(ctc_head, emotion_head, singing_head, pitch_head):
            # Temporarily update heads
            self.ctc_head = ctc_head
            self.multi_head.emotion_head = emotion_head
            self.multi_head.singing_head = singing_head
            self.multi_head.pitch_head = pitch_head

            loss, _ = self._compute_loss(encoder_outputs, labels)
            return loss

        # Compute gradients (need argnums for all 4 modules)
        loss, grads = mx.value_and_grad(loss_fn, argnums=(0, 1, 2, 3))(
            self.ctc_head,
            self.multi_head.emotion_head,
            self.multi_head.singing_head,
            self.multi_head.pitch_head,
        )

        # Update parameters using separate optimizers per head
        # (Avoids state collision when heads have same param names but different shapes)
        if self.config.train_ctc and self.ctc_optimizer:
            self.ctc_optimizer.update(self.ctc_head, grads[0])
        if self.config.train_emotion and self.emotion_optimizer:
            self.emotion_optimizer.update(self.multi_head.emotion_head, grads[1])
        if self.config.train_singing and self.singing_optimizer:
            self.singing_optimizer.update(self.multi_head.singing_head, grads[2])
        if self.config.train_pitch and self.pitch_optimizer:
            self.pitch_optimizer.update(self.multi_head.pitch_head, grads[3])

        mx.eval(self.ctc_head.parameters())
        mx.eval(self.multi_head.emotion_head.parameters())
        mx.eval(self.multi_head.singing_head.parameters())
        mx.eval(self.multi_head.pitch_head.parameters())

        # Get loss breakdown
        _, loss_dict = self._compute_loss(encoder_outputs, labels)

        return loss_dict

    def train(
        self,
        train_samples: list[MultiModalSample],
        val_samples: list[MultiModalSample],
    ):
        """Main training loop."""
        # Calculate total steps for LR scheduling
        steps_per_epoch = (len(train_samples) + self.config.batch_size - 1) // self.config.batch_size
        self.total_steps = steps_per_epoch * self.config.epochs

        self.log("Starting multi-head training")
        self.log(f"Epochs: {self.config.epochs}")
        self.log(f"Train samples: {len(train_samples)}")
        self.log(f"Val samples: {len(val_samples)}")
        self.log(f"Heads: CTC={self.config.train_ctc}, Emotion={self.config.train_emotion}, "
                 f"Singing={self.config.train_singing}, Pitch={self.config.train_pitch}")
        self.log(f"LR schedule: {self.config.lr_decay}, warmup={self.config.warmup_steps}, "
                 f"total_steps={self.total_steps}")
        self.log(f"Length-sorted batching: {self.config.length_sorted_batching}")

        # Populate durations if using length-sorted batching
        if self.config.length_sorted_batching:
            self._populate_sample_durations(train_samples)
            self._populate_sample_durations(val_samples)
            # Log duration statistics
            durations = [s.duration for s in train_samples]
            self.log(f"Duration stats: min={min(durations):.1f}s, max={max(durations):.1f}s, mean={np.mean(durations):.1f}s")

        for epoch in range(self.config.epochs):
            self.epoch = epoch

            # Track padding efficiency for length-sorted batching
            total_frames = 0
            total_padded_frames = 0

            if self.config.length_sorted_batching:
                # Use length-sorted batches
                batches = self._create_length_sorted_batches(train_samples)
            else:
                # Standard random shuffle
                rng = np.random.default_rng()
                rng.shuffle(train_samples)
                batches = [
                    train_samples[i:i + self.config.batch_size]
                    for i in range(0, len(train_samples), self.config.batch_size)
                ]

            # Training epoch
            epoch_losses = []
            for batch in batches:

                # Update learning rate before step
                current_lr = self._get_learning_rate()
                self._update_optimizer_lr(current_lr)

                loss_dict = self.train_step(batch)
                epoch_losses.append(loss_dict)
                self.step += 1

                # Track padding efficiency for length-sorted batching
                if self.config.length_sorted_batching and batch:
                    batch_durations = [s.duration for s in batch]
                    max_duration = max(batch_durations)
                    actual_frames = sum(int(d * 100) for d in batch_durations)  # ~100 frames/sec
                    padded_frames = len(batch) * int(max_duration * 100)
                    total_frames += actual_frames
                    total_padded_frames += padded_frames

                # Early step logging (first N steps) for debugging visibility
                # This ensures training progress is visible before regular log_interval
                early_log_limit = 50
                if self.step <= early_log_limit and self.step % 5 == 0:
                    loss_str = " | ".join([f"{k}={v:.4f}" for k, v in loss_dict.items()])
                    self.log(f"  [early] Step {self.step}: {loss_str}, lr={current_lr:.2e}")

                # Logging
                if self.step % self.config.log_interval == 0:
                    avg_losses = {}
                    for key in loss_dict:
                        avg_losses[key] = np.mean([loss.get(key, 0) for loss in epoch_losses[-self.config.log_interval:]])
                    loss_str = " | ".join([f"{k}={v:.4f}" for k, v in avg_losses.items()])
                    log_msg = f"  Step {self.step}: {loss_str}, lr={current_lr:.2e}"

                    # Add padding efficiency if using length-sorted batching
                    if self.config.length_sorted_batching and total_padded_frames > 0:
                        efficiency = total_frames / total_padded_frames * 100
                        log_msg += f", pad_eff={efficiency:.1f}%"

                    self.log(log_msg)

                # Save checkpoint
                if self.step % self.config.save_interval == 0:
                    self._save_checkpoint(f"step_{self.step}.npz")

                # Clear memory
                clear_memory()

            # Epoch summary
            avg_epoch_losses = {}
            for key in epoch_losses[0]:
                avg_epoch_losses[key] = np.mean([loss.get(key, 0) for loss in epoch_losses])
            loss_str = " | ".join([f"{k}={v:.4f}" for k, v in avg_epoch_losses.items()])
            epoch_msg = f"Epoch {epoch + 1}/{self.config.epochs}: {loss_str}"

            # Add padding efficiency summary for length-sorted batching
            if self.config.length_sorted_batching and total_padded_frames > 0:
                efficiency = total_frames / total_padded_frames * 100
                epoch_msg += f", pad_eff={efficiency:.1f}%"

            self.log(epoch_msg)

            # Validation
            if val_samples:
                val_loss = self._validate(val_samples)
                self.log(f"  Val loss: {val_loss:.4f}")

                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self._save_checkpoint("best.npz")
                    self.log("  New best model saved!")

        self.log("Training complete!")

    def _validate(self, val_samples: list[MultiModalSample]) -> float:
        """Run validation."""
        # Set heads to eval mode to disable dropout
        self.ctc_head.eval()
        self.multi_head.eval()

        total_loss = 0.0
        num_batches = 0

        for batch_start in range(0, len(val_samples), self.config.batch_size):
            batch_end = min(batch_start + self.config.batch_size, len(val_samples))
            batch = val_samples[batch_start:batch_end]

            encoder_outputs, labels = self._prepare_batch(batch)
            loss, _ = self._compute_loss(encoder_outputs, labels)

            total_loss += float(loss)
            num_batches += 1

        # Restore training mode
        self.ctc_head.train()
        self.multi_head.train()

        return total_loss / max(num_batches, 1)

    def _save_checkpoint(self, filename: str):
        """Save all head weights."""
        from mlx.utils import tree_flatten

        weights = {}

        # CTC head - flatten nested params
        for k, v in tree_flatten(self.ctc_head.parameters()):
            weights[f"ctc.{k}"] = v

        # Emotion head
        for k, v in tree_flatten(self.multi_head.emotion_head.parameters()):
            weights[f"emotion.{k}"] = v

        # Singing head
        for k, v in tree_flatten(self.multi_head.singing_head.parameters()):
            weights[f"singing.{k}"] = v

        # Pitch head
        for k, v in tree_flatten(self.multi_head.pitch_head.parameters()):
            weights[f"pitch.{k}"] = v

        # Training state
        weights["_step"] = mx.array(self.step)
        weights["_epoch"] = mx.array(self.epoch)
        # Handle inf specially to avoid mx.array issues
        best_loss_val = self.best_loss if self.best_loss != float("inf") else 1e30
        weights["_best_loss"] = mx.array(best_loss_val)

        save_path = self.output_dir / filename
        mx.savez(str(save_path), **weights)
        self.log(f"Saved checkpoint: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Head Training")
    # Data sources
    parser.add_argument("--librispeech-dir", type=str, help="LibriSpeech directory (for CTC)")
    parser.add_argument("--ravdess-dir", type=str, help="RAVDESS directory (for emotion/singing)")
    parser.add_argument("--prosody-dir", type=str, help="Consolidated prosody dir with JSON manifests (crema, esd, jvnv, ravdess)")
    parser.add_argument("--use-contours-only", action="store_true", help="Load only contours_train/val.json (with f0_contour for pitch training)")
    parser.add_argument("--expresso", action="store_true", help="Load Expresso from HuggingFace (SOTA 34 styles)")
    parser.add_argument("--vocalset-dir", type=str, help="VocalSet directory (for singing)")
    parser.add_argument("--unified-emotion-dir", type=str, help="Unified emotion dataset dir (HuggingFace format)")
    parser.add_argument("--unified-singing-dir", type=str, help="Unified singing dataset dir (HuggingFace format)")
    # Output
    parser.add_argument("--output-dir", type=str, default="checkpoints/multi_head", help="Output directory")
    parser.add_argument("--model", type=str, default="mlx-community/whisper-large-v3-mlx", help="Whisper model")
    # Training
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    # Head flags
    parser.add_argument("--train-ctc", type=str, default="true", help="Train CTC head")
    parser.add_argument("--train-emotion", type=str, default="true", help="Train emotion head")
    parser.add_argument("--train-singing", type=str, default="true", help="Train singing head")
    parser.add_argument("--train-pitch", type=str, default="true", help="Train pitch head")
    # Architecture options
    parser.add_argument("--use-crepe-pitch", action="store_true", help="Use CREPE-style 360-bin pitch head (vs MLP)")
    parser.add_argument("--use-attention-emotion", action="store_true", help="Use attention-based emotion head")
    parser.add_argument("--use-extended-singing", action="store_true", help="Use extended singing head with style + intensity")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--pretrained-singing", type=str, help="Load pretrained singing head weights (for transfer learning)")
    # Encoder caching (~3x speedup)
    parser.add_argument("--encoder-cache-dir", type=str, help="Directory for cached encoder outputs (~3x speedup)")

    # Length-sorted batching (~1.3x speedup)
    parser.add_argument("--length-sorted-batching", action="store_true",
                        help="Enable length-sorted batching (reduces padding waste ~1.3x speedup)")
    parser.add_argument("--bucket-size-multiplier", type=int, default=100,
                        help="Bucket size = batch_size * multiplier (default: 100)")

    args = parser.parse_args()

    # Parse boolean flags
    def parse_bool(s):
        return s.lower() in ("true", "1", "yes")

    # Determine if we have emotion/singing data
    has_emotion_data = (
        (args.ravdess_dir and Path(args.ravdess_dir).exists()) or
        (args.prosody_dir and Path(args.prosody_dir).exists()) or
        args.expresso or
        (args.vocalset_dir and Path(args.vocalset_dir).exists()) or
        (args.unified_emotion_dir and Path(args.unified_emotion_dir).exists()) or
        (args.unified_singing_dir and Path(args.unified_singing_dir).exists())
    )

    config = MultiHeadTrainingConfig(
        librispeech_dir=args.librispeech_dir,
        ravdess_dir=args.ravdess_dir,
        prosody_dir=args.prosody_dir,
        use_contours_only=args.use_contours_only,
        expresso_dir="huggingface" if args.expresso else None,
        vocalset_dir=args.vocalset_dir,
        output_dir=args.output_dir,
        whisper_model=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        train_ctc=parse_bool(args.train_ctc) and args.librispeech_dir is not None,
        train_emotion=parse_bool(args.train_emotion) and has_emotion_data,
        train_singing=parse_bool(args.train_singing) and has_emotion_data,
        train_pitch=parse_bool(args.train_pitch),
        # Architecture options
        use_crepe_pitch=args.use_crepe_pitch,
        use_attention_emotion=args.use_attention_emotion,
        use_extended_singing=args.use_extended_singing,
        # Encoder caching
        encoder_cache_dir=args.encoder_cache_dir,
        # Length-sorted batching
        length_sorted_batching=args.length_sorted_batching,
        bucket_size_multiplier=args.bucket_size_multiplier,
    )

    print("=" * 70)
    print("Multi-Head Training for Whisper")
    print("=" * 70)
    print(f"Output: {config.output_dir}")
    print(f"Model: {config.whisper_model}")
    print(f"Heads: CTC={config.train_ctc}, Emotion={config.train_emotion}, "
          f"Singing={config.train_singing}, Pitch={config.train_pitch}")
    if config.encoder_cache_dir:
        print(f"Encoder cache: {config.encoder_cache_dir} (~3x speedup)")
    if config.use_crepe_pitch or config.use_attention_emotion or config.use_extended_singing:
        print(f"Architecture: CREPE-Pitch={config.use_crepe_pitch}, "
              f"Attention-Emotion={config.use_attention_emotion}, "
              f"Extended-Singing={config.use_extended_singing}")
    print()

    # Load Whisper model
    print("1. Loading Whisper model...")
    whisper_model = WhisperMLX.from_pretrained(config.whisper_model)
    d_model = whisper_model.config.n_audio_state
    config.d_model = d_model
    print(f"   d_model={d_model}")

    # Create heads
    print("2. Creating heads...")
    ctc_head = create_ctc_draft_head(config.model_size)
    multi_head = create_multi_head(
        config.model_size,
        ctc_head=ctc_head,
        use_crepe_pitch=config.use_crepe_pitch,
        use_attention_emotion=config.use_attention_emotion,
        use_extended_singing=config.use_extended_singing,
    )
    # Helper to count params in nested dict from MLX .parameters()
    def count_params(params):
        from mlx.utils import tree_flatten
        flat = tree_flatten(params)
        return sum(p.size for _, p in flat)

    print(f"   CTC: {count_params(ctc_head.parameters()) / 1e6:.1f}M params")
    print(f"   Emotion: {count_params(multi_head.emotion_head.parameters()) / 1e6:.1f}M params")
    print(f"   Singing: {count_params(multi_head.singing_head.parameters()) / 1e6:.1f}M params")
    print(f"   Pitch: {count_params(multi_head.pitch_head.parameters()) / 1e6:.1f}M params")

    # Load pretrained singing weights if specified
    if args.pretrained_singing and Path(args.pretrained_singing).exists():
        print(f"\n   Loading pretrained singing weights from: {args.pretrained_singing}")
        from mlx.utils import tree_flatten
        pretrained = dict(mx.load(args.pretrained_singing))

        # Extract singing weights from pretrained checkpoint
        singing_weights = []
        for key, value in pretrained.items():
            if key.startswith("singing."):
                new_key = key.replace("singing.", "")
                singing_weights.append((new_key, value))

        if singing_weights:
            # Load weights that match (shared layers, singing_fc)
            # Skip style_fc and intensity_fc which have different dimensions in extended head
            current_params = dict(tree_flatten(multi_head.singing_head.parameters()))
            loaded_count = 0
            for key, value in singing_weights:
                if key in current_params and current_params[key].shape == value.shape:
                    print(f"      Loading: {key} {value.shape}")
                    loaded_count += 1
                else:
                    if key in current_params:
                        print(f"      Skipping (shape mismatch): {key} pretrained={value.shape} vs current={current_params[key].shape}")
                    else:
                        print(f"      Skipping (not found): {key}")

            # Manually set matching weights (don't use load_weights which requires all params)
            matching_weights = {
                k: v for k, v in singing_weights
                if k in current_params and current_params[k].shape == v.shape
            }
            if matching_weights:
                # Update parameters in-place
                def set_nested_attr(obj, path, value):
                    """Set a nested attribute like 'shared_fc.weight'"""
                    parts = path.split(".")
                    for part in parts[:-1]:
                        obj = getattr(obj, part)
                    setattr(obj, parts[-1], value)

                for key, value in matching_weights.items():
                    try:
                        set_nested_attr(multi_head.singing_head, key, value)
                    except Exception as e:
                        print(f"      Failed to set {key}: {e}")
                mx.eval(multi_head.singing_head.parameters())
                print(f"      Loaded {loaded_count} pretrained weight tensors")
        else:
            print("      No singing weights found in checkpoint")

    # Load tokenizer
    print("3. Loading tokenizer...")
    tokenizer = get_whisper_tokenizer()

    # Load datasets
    train_samples = []
    val_samples = []
    dataset_num = 4

    # RAVDESS (8 emotions + singing)
    if config.ravdess_dir and Path(config.ravdess_dir).exists():
        print(f"{dataset_num}. Loading RAVDESS from {config.ravdess_dir}...")
        ravdess = RAVDESSDataset(config.ravdess_dir, val_split=config.val_split)
        train_samples.extend(ravdess.get_train_samples())
        val_samples.extend(ravdess.get_val_samples())
        dataset_num += 1

    # Consolidated Prosody (crema-d, esd, jvnv, ravdess from JSON manifests)
    if config.prosody_dir and Path(config.prosody_dir).exists():
        # Use contours files only if training pitch (they have precomputed f0_contour)
        if config.use_contours_only:
            manifests = ["contours_train.json", "contours_val.json"]
            print(f"{dataset_num}. Loading contours data (with f0_contour) from {config.prosody_dir}...")
        else:
            manifests = None  # Use default manifests
            print(f"{dataset_num}. Loading consolidated prosody from {config.prosody_dir}...")
        prosody = ConsolidatedProsodyDataset(
            config.prosody_dir,
            max_audio_len=config.max_audio_len,
            val_split=config.val_split,
            manifests=manifests,
        )
        train_samples.extend(prosody.get_train_samples())
        val_samples.extend(prosody.get_val_samples())
        dataset_num += 1

    # Expresso (34 styles - SOTA)
    if config.expresso_dir:
        print(f"{dataset_num}. Loading Expresso from HuggingFace (34 expressive styles)...")
        expresso = ExpressoDataset(val_split=config.val_split)
        train_samples.extend(expresso.get_train_samples())
        val_samples.extend(expresso.get_val_samples())
        dataset_num += 1

    # VocalSet (singing techniques)
    if config.vocalset_dir and Path(config.vocalset_dir).exists():
        print(f"{dataset_num}. Loading VocalSet from {config.vocalset_dir}...")
        vocalset = VocalSetDataset(config.vocalset_dir, val_split=config.val_split)
        train_samples.extend(vocalset.get_train_samples())
        val_samples.extend(vocalset.get_val_samples())
        dataset_num += 1

    # LibriSpeech (text transcription for CTC)
    if config.librispeech_dir and Path(config.librispeech_dir).exists():
        print(f"{dataset_num}. Loading LibriSpeech from {config.librispeech_dir}...")
        libri = LibriSpeechDataset(config.librispeech_dir, val_split=config.val_split)
        train_samples.extend(libri.get_train_samples())
        val_samples.extend(libri.get_val_samples())
        dataset_num += 1

    # Unified Emotion Dataset (HuggingFace format)
    if args.unified_emotion_dir and Path(args.unified_emotion_dir).exists():
        print(f"{dataset_num}. Loading unified emotion from {args.unified_emotion_dir}...")
        unified_emotion = UnifiedEmotionDataset(args.unified_emotion_dir, max_audio_len=config.max_audio_len)
        train_samples.extend(unified_emotion.get_train_samples())
        val_samples.extend(unified_emotion.get_val_samples())
        dataset_num += 1

    # Unified Singing Dataset (HuggingFace format)
    if args.unified_singing_dir and Path(args.unified_singing_dir).exists():
        print(f"{dataset_num}. Loading unified singing from {args.unified_singing_dir}...")
        unified_singing = UnifiedSingingDataset(args.unified_singing_dir, max_audio_len=config.max_audio_len)
        train_samples.extend(unified_singing.get_train_samples())
        val_samples.extend(unified_singing.get_val_samples())
        dataset_num += 1

    if not train_samples:
        print("ERROR: No training data found!")
        print("Please provide at least one of:")
        print("  --ravdess-dir <path>     (8 emotions + singing)")
        print("  --prosody-dir <path>     (consolidated: crema-d, esd, jvnv, ravdess)")
        print("  --expresso               (34 styles from HuggingFace)")
        print("  --vocalset-dir <path>    (singing techniques)")
        print("  --librispeech-dir <path> (text transcription)")
        return

    print(f"\nTotal training samples: {len(train_samples)}")
    print(f"Total validation samples: {len(val_samples)}")

    # Create trainer
    trainer = MultiHeadTrainer(
        config=config,
        whisper_model=whisper_model,
        multi_head=multi_head,
        ctc_head=ctc_head,
        tokenizer=tokenizer,
    )

    # Train
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)
    trainer.train(train_samples, val_samples)


if __name__ == "__main__":
    main()
