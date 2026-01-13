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
Prosody Training Data Preparation Pipeline

Phase B of Kokoro Prosody Annotation Roadmap:
Prepares training data for prosody embedding fine-tuning.

Data Sources:
1. Synthetic: CosyVoice2 generates emotional variants of base texts
2. Emotional Speech Datasets: ESD, RAVDESS, CREMA-D
3. Auto-annotation: Extract prosody features from existing speech

Output Format:
    {
        "text": "Hello world",
        "annotated_text": "<emotion type='excited'>Hello world</emotion>",
        "prosody_type": 42,  # EMOTION_EXCITED
        "audio_path": "data/prosody/train/0001.wav",
        "f0_mean": 220.5,
        "f0_std": 30.2,
        "duration_s": 1.2,
        "energy_rms": 0.08
    }

Usage:
    # Prepare all data sources (recommended)
    python scripts/prepare_prosody_training_data.py --all

    # Just synthetic data from CosyVoice2
    python scripts/prepare_prosody_training_data.py --synthetic

    # Download emotional speech datasets
    python scripts/prepare_prosody_training_data.py --download-datasets

    # Auto-annotate existing audio files
    python scripts/prepare_prosody_training_data.py --auto-annotate /path/to/audio

    # Verify dataset quality
    python scripts/prepare_prosody_training_data.py --verify

See: reports/main/PROSODY_ROADMAP.md
"""

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Prosody Types (matching C++ prosody_types.h)
# =============================================================================

PROSODY_TYPES = {
    "NEUTRAL": 0,
    "EMPHASIS": 1,
    "STRONG_EMPHASIS": 2,
    "REDUCED_EMPHASIS": 3,
    "RATE_X_SLOW": 10,
    "RATE_SLOW": 11,
    "RATE_FAST": 12,
    "RATE_X_FAST": 13,
    "PITCH_X_LOW": 20,
    "PITCH_LOW": 21,
    "PITCH_HIGH": 22,
    "PITCH_X_HIGH": 23,
    "VOLUME_X_SOFT": 30,
    "VOLUME_SOFT": 31,
    "VOLUME_LOUD": 32,
    "VOLUME_X_LOUD": 33,
    "VOLUME_WHISPER": 34,
    "EMOTION_ANGRY": 40,
    "EMOTION_SAD": 41,
    "EMOTION_EXCITED": 42,
    "EMOTION_WORRIED": 43,
    "EMOTION_ALARMED": 44,
    "EMOTION_CALM": 45,
    "EMOTION_EMPATHETIC": 46,
    "EMOTION_CONFIDENT": 47,
    "EMOTION_FRUSTRATED": 48,
    "EMOTION_NERVOUS": 49,
    "EMOTION_SURPRISED": 50,
    "EMOTION_DISAPPOINTED": 51,
    "QUESTION": 60,
    "WHISPER": 61,
    "LOUD": 62,
}

# Map emotion names to prosody types
EMOTION_TO_PROSODY = {
    "angry": PROSODY_TYPES["EMOTION_ANGRY"],
    "anger": PROSODY_TYPES["EMOTION_ANGRY"],  # JVNV
    "sad": PROSODY_TYPES["EMOTION_SAD"],
    "sadness": PROSODY_TYPES["EMOTION_SAD"],  # RESD
    "excited": PROSODY_TYPES["EMOTION_EXCITED"],
    "happy": PROSODY_TYPES["EMOTION_EXCITED"],  # Map happy to excited
    "happiness": PROSODY_TYPES["EMOTION_EXCITED"],  # RESD
    "enthusiasm": PROSODY_TYPES["EMOTION_EXCITED"],  # RESD
    "worried": PROSODY_TYPES["EMOTION_WORRIED"],
    "calm": PROSODY_TYPES["NEUTRAL"],  # CALM → NEUTRAL (prosodically identical ~145 Hz)
    "neutral": PROSODY_TYPES["NEUTRAL"],
    "surprised": PROSODY_TYPES["EMOTION_SURPRISED"],
    "surprise": PROSODY_TYPES["EMOTION_SURPRISED"],  # JVNV
    "fear": PROSODY_TYPES["EMOTION_NERVOUS"],
    "fearful": PROSODY_TYPES["EMOTION_NERVOUS"],
    "disgust": PROSODY_TYPES["EMOTION_FRUSTRATED"],
}

# CREMA-D sentence codes to text mapping
CREMA_SENTENCES = {
    "IEO": "It's eleven o'clock.",
    "TIE": "That is exactly what happened.",
    "IOM": "I'm on my way to the meeting.",
    "IWW": "I wonder what this is about.",
    "TAI": "The airplane is almost full.",
    "MTI": "Maybe tomorrow it will be cold.",
    "IWL": "I would like a new alarm clock.",
    "ITH": "I think I have a doctor's appointment.",
    "DFA": "Don't forget a jacket.",
    "ITS": "I think I've seen this before.",
    "TSI": "The surface is slick.",
    "WSI": "We'll stop in a couple of minutes.",
}

# CREMA-D emotion codes
CREMA_EMOTIONS = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ProsodySample:
    """A single prosody training sample."""
    text: str
    annotated_text: str
    prosody_type: int
    audio_path: str
    f0_mean: float = 0.0
    f0_std: float = 0.0
    f0_range: float = 0.0
    duration_s: float = 0.0
    energy_rms: float = 0.0
    source: str = "synthetic"  # synthetic, esd, ravdess, crema-d, auto


@dataclass
class DatasetStats:
    """Statistics for a prosody dataset."""
    total_samples: int = 0
    samples_by_type: Dict[int, int] = field(default_factory=dict)
    total_duration_s: float = 0.0
    mean_f0: float = 0.0
    mean_duration: float = 0.0


# =============================================================================
# Feature Extraction
# =============================================================================

def extract_f0(audio: np.ndarray, sr: int = 24000) -> Dict:
    """Extract F0 (pitch) features using librosa's yin (fast) or pyin (accurate).

    Uses yin by default for 10-100x faster processing.
    Set PROSODY_F0_METHOD=pyin env var for higher accuracy (much slower).
    """
    try:
        import librosa
    except ImportError:
        print("librosa required. Install with: pip install librosa")
        return {"f0_mean": 0.0, "f0_std": 0.0, "f0_range": 0.0, "voiced_ratio": 0.0}

    import os
    use_pyin = os.environ.get("PROSODY_F0_METHOD", "yin").lower() == "pyin"

    if use_pyin:
        # Probabilistic YIN - more accurate but 10-100x slower
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=50,
            fmax=500,
            sr=sr,
            frame_length=1024,
            hop_length=256,
        )
        voiced_f0 = f0[~np.isnan(f0)]
    else:
        # Regular YIN - fast and accurate enough for emotion classification
        f0 = librosa.yin(
            audio,
            fmin=50,
            fmax=500,
            sr=sr,
            frame_length=1024,
            hop_length=256,
        )
        # YIN returns valid values (clamped to fmin/fmax), filter out-of-range
        voiced_f0 = f0[(f0 >= 50) & (f0 <= 500)]

    if len(voiced_f0) == 0:
        return {"f0_mean": 0.0, "f0_std": 0.0, "f0_range": 0.0, "voiced_ratio": 0.0}

    return {
        "f0_mean": float(np.mean(voiced_f0)),
        "f0_std": float(np.std(voiced_f0)),
        "f0_range": float(np.max(voiced_f0) - np.min(voiced_f0)),
        "voiced_ratio": float(len(voiced_f0) / len(f0)),
    }


def extract_prosody_features(audio: np.ndarray, sr: int = 24000) -> Dict:
    """Extract all prosody-relevant features from audio."""
    features = extract_f0(audio, sr)

    # Duration
    duration_s = len(audio) / sr

    # Energy (RMS)
    energy_rms = float(np.sqrt(np.mean(audio ** 2)))

    features["duration_s"] = duration_s
    features["energy_rms"] = energy_rms

    return features


def auto_annotate_prosody(features: Dict, text: str) -> Tuple[int, str]:
    """
    Automatically determine prosody type from audio features.

    Uses heuristics based on F0 and energy patterns.
    Returns (prosody_type, annotated_text).
    """
    f0_mean = features.get("f0_mean", 0)
    f0_std = features.get("f0_std", 0)
    energy = features.get("energy_rms", 0)

    # Default to neutral
    prosody_type = PROSODY_TYPES["NEUTRAL"]
    emotion = None

    # Simple heuristics (will be refined with training data)
    if f0_mean > 250 and f0_std > 40:
        # High pitch + high variance = excited
        prosody_type = PROSODY_TYPES["EMOTION_EXCITED"]
        emotion = "excited"
    elif f0_mean < 150 and f0_std < 25:
        # Low pitch + low variance = calm/sad
        prosody_type = PROSODY_TYPES["EMOTION_CALM"]
        emotion = "calm"
    elif f0_std > 50 and energy > 0.1:
        # High variance + high energy = angry
        prosody_type = PROSODY_TYPES["EMOTION_ANGRY"]
        emotion = "angry"
    elif energy < 0.03:
        # Low energy = soft/sad
        prosody_type = PROSODY_TYPES["EMOTION_SAD"]
        emotion = "sad"

    # Build annotated text
    if emotion:
        annotated_text = f"<emotion type='{emotion}'>{text}</emotion>"
    else:
        annotated_text = text

    return prosody_type, annotated_text


# =============================================================================
# Synthetic Data Generation
# =============================================================================

# Base texts for synthetic generation
SYNTHETIC_BASE_TEXTS = [
    "Hello, how are you today?",
    "I understand what you mean.",
    "Let me help you with that.",
    "Thank you for your patience.",
    "This is really important.",
    "I have some news for you.",
    "Can you please repeat that?",
    "Everything will be okay.",
    "We need to talk about this.",
    "I'm here to assist you.",
    "That's a great question.",
    "I see what you're saying.",
    "Let me think about that.",
    "I appreciate your feedback.",
    "The weather is beautiful today.",
    "Please wait a moment.",
    "I'll get back to you soon.",
    "Do you have any questions?",
    "That sounds wonderful.",
    "I completely agree with you.",
]

# Emotion prompts for synthetic generation
EMOTION_PROMPTS = [
    ("angry", "Say this angrily"),
    ("sad", "Say this sadly"),
    ("excited", "Say this excitedly"),
    ("worried", "Say this with worry"),
    ("calm", "Say this calmly"),
    ("surprised", "Say this with surprise"),
    ("neutral", "Say this neutrally"),
]


def generate_synthetic_data(
    output_dir: Path,
    num_samples: int = 1000,
    use_cosyvoice: bool = True,
) -> List[ProsodySample]:
    """
    Generate synthetic prosody training data.

    Uses CosyVoice2 to synthesize emotional variants of base texts.
    If CosyVoice2 unavailable, uses Kokoro with placeholder annotations.
    """
    samples = []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try to load TTS model
    model = None
    model_type = None
    tokenizer = None
    speaker_embedding = None

    if use_cosyvoice:
        try:
            from tools.pytorch_to_mlx.converters.models import CosyVoice2Model
            model_path = Path.home() / ".cache" / "cosyvoice2" / "cosyvoice2-0.5b"
            if model_path.exists():
                print("Loading CosyVoice2 model...")
                model = CosyVoice2Model.from_pretrained(str(model_path))
                model_type = "cosyvoice2"
                tokenizer = model.tokenizer
                speaker_embedding = tokenizer.random_speaker_embedding()
                print("CosyVoice2 loaded successfully")
        except Exception as e:
            print(f"CosyVoice2 not available: {e}")

    if model is None:
        try:
            from mlx_audio.tts.utils import load_model
            print("Loading Kokoro model...")
            model = load_model("prince-canuma/Kokoro-82M")
            model_type = "kokoro"
            print("Kokoro loaded successfully")
        except Exception as e:
            print(f"Kokoro not available: {e}")
            print("No TTS model available. Skipping synthetic generation.")
            return []

    # Generate samples
    sample_idx = 0
    samples_per_text = max(1, num_samples // (len(SYNTHETIC_BASE_TEXTS) * len(EMOTION_PROMPTS)))

    print(f"Generating {num_samples} synthetic samples...")

    try:
        import soundfile as sf
    except ImportError:
        print("soundfile required. Install with: pip install soundfile")
        return []

    for text in SYNTHETIC_BASE_TEXTS:
        for emotion, prompt in EMOTION_PROMPTS:
            for _ in range(samples_per_text):
                if sample_idx >= num_samples:
                    break

                try:
                    # Generate audio
                    if model_type == "cosyvoice2":
                        # For CosyVoice2, use instruction mode (future)
                        # Currently use standard synthesis
                        text_ids = tokenizer.encode(text)
                        import mlx.core as mx
                        text_ids = mx.array([text_ids])
                        audio = model.synthesize(
                            text_ids,
                            speaker_embedding,
                            temperature=0.9 + random.random() * 0.2,
                        )
                        audio = np.array(audio).flatten()
                        sr = 24000
                    else:
                        # Kokoro
                        audio_chunks = []
                        for result in model.generate(
                            text=text,
                            voice="af_bella",
                            speed=1.0,
                            verbose=False,
                        ):
                            audio_chunks.append(result.audio)
                        audio = np.concatenate(audio_chunks, axis=-1).flatten()
                        sr = 24000

                    # Save audio
                    audio_path = output_dir / f"synthetic_{sample_idx:05d}.wav"
                    sf.write(audio_path, audio, sr)

                    # Extract features
                    features = extract_prosody_features(audio, sr)

                    # Create annotated text
                    prosody_type = EMOTION_TO_PROSODY.get(emotion, PROSODY_TYPES["NEUTRAL"])
                    if emotion != "neutral":
                        annotated_text = f"<emotion type='{emotion}'>{text}</emotion>"
                    else:
                        annotated_text = text

                    sample = ProsodySample(
                        text=text,
                        annotated_text=annotated_text,
                        prosody_type=prosody_type,
                        audio_path=str(audio_path),
                        f0_mean=features["f0_mean"],
                        f0_std=features["f0_std"],
                        f0_range=features["f0_range"],
                        duration_s=features["duration_s"],
                        energy_rms=features["energy_rms"],
                        source="synthetic",
                    )
                    samples.append(sample)
                    sample_idx += 1

                    if sample_idx % 50 == 0:
                        print(f"  Generated {sample_idx}/{num_samples} samples")

                except Exception as e:
                    print(f"  Error generating sample {sample_idx}: {e}")
                    continue

            if sample_idx >= num_samples:
                break
        if sample_idx >= num_samples:
            break

    print(f"Generated {len(samples)} synthetic samples")
    return samples


# =============================================================================
# Dataset Download and Processing
# =============================================================================

DATASET_URLS = {
    # Note: These are placeholder URLs - actual datasets require registration
    "ravdess": "https://zenodo.org/records/1188976/files/Audio_Song_Actors_01-24.zip",
    # ESD requires registration at https://hltsingapore.github.io/ESD/
    # CREMA-D available at https://github.com/CheyneyComputerScience/CREMA-D
}

RAVDESS_EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


def download_ravdess(ravdess_dir: Path) -> List[ProsodySample]:
    """
    Process RAVDESS dataset.

    RAVDESS: Ryerson Audio-Visual Database of Emotional Speech and Song
    - 24 actors (12 male, 12 female)
    - 8 emotions
    - ~7,356 audio files

    Note: Dataset should be manually downloaded from https://zenodo.org/records/1188976

    Args:
        ravdess_dir: Path to the RAVDESS directory containing Actor_* subdirectories
    """
    ravdess_dir = Path(ravdess_dir)

    # Check if dataset exists
    audio_files = list(ravdess_dir.glob("**/*.wav"))
    if not audio_files:
        print(f"RAVDESS not found at {ravdess_dir}")
        print("Please download from: https://zenodo.org/records/1188976")
        print(f"Extract to: {ravdess_dir}")
        return []

    print(f"Processing {len(audio_files)} RAVDESS files...")

    samples = []
    try:
        import soundfile as sf
    except ImportError:
        print("soundfile required")
        return []

    for audio_path in audio_files:
        try:
            # Parse filename: {Modality}-{Vocal channel}-{Emotion}-{Intensity}-{Statement}-{Repetition}-{Actor}.wav
            parts = audio_path.stem.split("-")
            if len(parts) < 7:
                continue

            emotion_code = parts[2]
            emotion = RAVDESS_EMOTIONS.get(emotion_code, "neutral")
            prosody_type = EMOTION_TO_PROSODY.get(emotion, PROSODY_TYPES["NEUTRAL"])

            # Statement: "01" = "Kids are talking by the door"
            #           "02" = "Dogs are sitting by the door"
            statement_code = parts[4]
            if statement_code == "01":
                text = "Kids are talking by the door."
            else:
                text = "Dogs are sitting by the door."

            # Load audio
            audio, sr = sf.read(audio_path)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            # Extract features
            features = extract_prosody_features(audio, sr)

            # Annotated text
            if emotion != "neutral":
                annotated_text = f"<emotion type='{emotion}'>{text}</emotion>"
            else:
                annotated_text = text

            sample = ProsodySample(
                text=text,
                annotated_text=annotated_text,
                prosody_type=prosody_type,
                audio_path=str(audio_path),
                f0_mean=features["f0_mean"],
                f0_std=features["f0_std"],
                f0_range=features["f0_range"],
                duration_s=features["duration_s"],
                energy_rms=features["energy_rms"],
                source="ravdess",
            )
            samples.append(sample)

        except Exception as e:
            print(f"  Error processing {audio_path}: {e}")
            continue

    print(f"Processed {len(samples)} RAVDESS samples")
    return samples


def process_esd(esd_dir: Path, english_only: bool = True) -> List[ProsodySample]:
    """
    Process Emotional Speech Dataset (ESD).

    ESD: Emotional Speech Dataset
    - 20 speakers (10 English: 0011-0020, 10 Chinese: 0001-0010)
    - 5 emotions (Angry, Happy, Neutral, Sad, Surprise)
    - 350 parallel utterances per speaker/emotion
    - Total: ~35,000 utterances

    Structure:
    - esd/{speaker}/{emotion}/{utterance_id}.wav
    - esd/{speaker}/{speaker}.txt (transcripts: utterance_id\\ttext\\temotion)

    Note: Dataset requires registration at https://hltsingapore.github.io/ESD/
    """
    esd_dir = Path(esd_dir)

    if not esd_dir.exists():
        print(f"ESD not found at {esd_dir}")
        print("Please download from: https://hltsingapore.github.io/ESD/")
        return []

    samples = []

    try:
        import soundfile as sf
    except ImportError:
        print("soundfile required")
        return []

    # ESD structure: esd/{speaker}/{emotion}/{utterance_id}.wav
    for speaker_dir in sorted(esd_dir.iterdir()):
        if not speaker_dir.is_dir():
            continue

        speaker_id = speaker_dir.name

        # Skip Chinese speakers if english_only
        if english_only:
            try:
                speaker_num = int(speaker_id)
                if speaker_num < 11:  # 0001-0010 are Chinese
                    continue
            except ValueError:
                continue

        # Load transcripts from {speaker}/{speaker}.txt
        transcripts = {}
        transcript_file = speaker_dir / f"{speaker_id}.txt"
        if transcript_file.exists():
            with open(transcript_file, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        transcripts[parts[0]] = parts[1]

        for emotion_dir in speaker_dir.iterdir():
            if not emotion_dir.is_dir():
                continue

            emotion = emotion_dir.name.lower()
            prosody_type = EMOTION_TO_PROSODY.get(emotion, PROSODY_TYPES["NEUTRAL"])

            for audio_path in emotion_dir.glob("*.wav"):
                try:
                    text = transcripts.get(audio_path.stem, "Unknown text")

                    audio, sr = sf.read(audio_path)
                    if len(audio.shape) > 1:
                        audio = audio.mean(axis=1)

                    features = extract_prosody_features(audio, sr)

                    if emotion != "neutral":
                        annotated_text = f"<emotion type='{emotion}'>{text}</emotion>"
                    else:
                        annotated_text = text

                    sample = ProsodySample(
                        text=text,
                        annotated_text=annotated_text,
                        prosody_type=prosody_type,
                        audio_path=str(audio_path),
                        f0_mean=features["f0_mean"],
                        f0_std=features["f0_std"],
                        f0_range=features["f0_range"],
                        duration_s=features["duration_s"],
                        energy_rms=features["energy_rms"],
                        source="esd",
                    )
                    samples.append(sample)

                except Exception:
                    continue

    print(f"Processed {len(samples)} ESD samples")
    return samples


def process_crema(crema_dir: Path) -> List[ProsodySample]:
    """
    Process CREMA-D dataset.

    CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset
    - 91 actors (48 male, 43 female)
    - 6 emotions (Anger, Disgust, Fear, Happy, Neutral, Sad)
    - 12 sentences
    - 4 intensity levels (Low, Medium, High, Unspecified)
    - Total: ~7,442 clips

    Filename format: {ActorID}_{SentenceCode}_{Emotion}_{Intensity}.wav
    Example: 1001_IEO_ANG_HI.wav

    Note: Dataset available at https://github.com/CheyneyComputerScience/CREMA-D
    License: ODbL (Open Database License) - Commercial use OK
    """
    crema_dir = Path(crema_dir)

    if not crema_dir.exists():
        print(f"CREMA-D not found at {crema_dir}")
        print("Please download from: https://github.com/CheyneyComputerScience/CREMA-D")
        return []

    audio_files = list(crema_dir.glob("*.wav"))
    if not audio_files:
        print(f"No WAV files found in {crema_dir}")
        return []

    print(f"Processing {len(audio_files)} CREMA-D files...")

    samples = []
    try:
        import soundfile as sf
    except ImportError:
        print("soundfile required")
        return []

    for audio_path in audio_files:
        try:
            # Parse filename: {ActorID}_{SentenceCode}_{Emotion}_{Intensity}.wav
            parts = audio_path.stem.split("_")
            if len(parts) < 4:
                continue

            _actor_id = parts[0]  # Unused: actor identifier
            sentence_code = parts[1]
            emotion_code = parts[2]
            _intensity = parts[3]  # Unused: emotion intensity level

            # Get text from sentence code
            text = CREMA_SENTENCES.get(sentence_code, "Unknown sentence.")

            # Get emotion from code
            emotion = CREMA_EMOTIONS.get(emotion_code, "neutral")
            prosody_type = EMOTION_TO_PROSODY.get(emotion, PROSODY_TYPES["NEUTRAL"])

            # Load audio
            audio, sr = sf.read(audio_path)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            # Skip very short files (known issues in CREMA-D)
            if len(audio) < sr * 0.3:  # Less than 0.3 seconds
                continue

            # Extract features
            features = extract_prosody_features(audio, sr)

            # Create annotated text
            if emotion != "neutral":
                annotated_text = f"<emotion type='{emotion}'>{text}</emotion>"
            else:
                annotated_text = text

            sample = ProsodySample(
                text=text,
                annotated_text=annotated_text,
                prosody_type=prosody_type,
                audio_path=str(audio_path),
                f0_mean=features["f0_mean"],
                f0_std=features["f0_std"],
                f0_range=features["f0_range"],
                duration_s=features["duration_s"],
                energy_rms=features["energy_rms"],
                source="crema-d",
            )
            samples.append(sample)

        except Exception:
            # Silently skip problematic files (some CREMA-D files have issues)
            continue

    print(f"Processed {len(samples)} CREMA-D samples")
    return samples


def process_jvnv(jvnv_dir: Path) -> List[ProsodySample]:
    """
    Process JVNV (Japanese Voice and Non-verbal) dataset from HuggingFace.

    JVNV: Japanese emotional speech corpus
    - Multiple speakers
    - 6 emotions: surprise, happy, sad, anger, fear, disgust
    - ~1,615 samples

    The dataset should be saved from HuggingFace using:
        ds = load_dataset('asahi417/jvnv-emotional-speech-corpus')
        ds.save_to_disk('data/prosody/jvnv')

    Note: This dataset has no text transcripts, so we use placeholder text.
    """
    jvnv_dir = Path(jvnv_dir)

    if not jvnv_dir.exists():
        print(f"JVNV not found at {jvnv_dir}")
        print("Download with:")
        print("  from datasets import load_dataset")
        print("  ds = load_dataset('asahi417/jvnv-emotional-speech-corpus')")
        print(f"  ds.save_to_disk('{jvnv_dir}')")
        return []

    try:
        import io

        import pyarrow as pa
        import soundfile as sf
    except ImportError as e:
        print(f"Required packages: {e}")
        return []

    print("Loading JVNV dataset from Arrow files...")

    samples = []

    # JVNV has 'test' split
    split_dir = jvnv_dir / 'test'
    if not split_dir.exists():
        # Try without split subdirectory
        split_dir = jvnv_dir

    arrow_files = sorted(split_dir.glob('*.arrow'))
    if not arrow_files:
        print(f"No arrow files found in {split_dir}")
        return []

    # JVNV has no text - use Japanese placeholder based on emotion
    emotion_texts_ja = {
        "surprise": "えっ、本当ですか",  # "Eh, really?"
        "happy": "とても嬉しいです",  # "I'm very happy"
        "sad": "悲しいです",  # "I'm sad"
        "anger": "許せない",  # "Unforgivable"
        "fear": "怖いです",  # "I'm scared"
        "disgust": "気持ち悪い",  # "Disgusting"
    }

    sample_idx = 0
    for arrow_file in arrow_files:
        with open(arrow_file, 'rb') as f:
            reader = pa.ipc.open_stream(f)
            table = reader.read_all()

        # Convert to Python dicts
        for i in range(table.num_rows):
            try:
                audio_struct = table['audio'][i].as_py()
                emotion = table['style'][i].as_py().lower()
                _speaker_id = table['speaker_id'][i].as_py()  # Unused: speaker identifier

                prosody_type = EMOTION_TO_PROSODY.get(emotion, PROSODY_TYPES["NEUTRAL"])
                text = emotion_texts_ja.get(emotion, "日本語のテスト")

                # Decode audio bytes with soundfile
                audio_bytes = audio_struct['bytes']
                audio, sr = sf.read(io.BytesIO(audio_bytes))
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)

                # Extract features
                features = extract_prosody_features(audio, sr)

                # Create annotated text
                if emotion != "neutral":
                    annotated_text = f"<emotion type='{emotion}'>{text}</emotion>"
                else:
                    annotated_text = text

                sample = ProsodySample(
                    text=text,
                    annotated_text=annotated_text,
                    prosody_type=prosody_type,
                    audio_path=f"jvnv:test:{sample_idx}",
                    f0_mean=features["f0_mean"],
                    f0_std=features["f0_std"],
                    f0_range=features["f0_range"],
                    duration_s=features["duration_s"],
                    energy_rms=features["energy_rms"],
                    source="jvnv",
                )
                samples.append(sample)
                sample_idx += 1

                if sample_idx % 200 == 0:
                    print(f"  Processed {sample_idx} samples")

            except Exception as e:
                print(f"  Error processing sample {sample_idx}: {e}")
                continue

    print(f"Processed {len(samples)} JVNV samples")
    return samples


def process_consolidated_english(ce_dir: Path) -> List[ProsodySample]:
    """
    Process Consolidated English/Multilingual dataset from HuggingFace.

    This is actually a multilingual dataset with 66K samples from 17 source datasets:
    - AESDD, CREMA-D, CaFE, EMNS, Emozionalmente, IEMOCAP, JL-Corpus, JVNV, MESD,
      Oréau, PAVOQUE, RAVDESS, RESD, SUBESCO, THAI_SER, eNTERFACE, nEMO

    Languages: Bengali, English, French, German, Italian, Polish, Russian, Spanish,
               english, greek, japanese, thai

    Emotions: anger, disgust, fear, happiness, neutral, sadness, surprise

    The dataset should be saved from HuggingFace using:
        ds = load_dataset('NathanRoll/speech-emotion-dataset-consolidated')
        ds.save_to_disk('data/prosody/consolidated_english')
    """
    ce_dir = Path(ce_dir)

    if not ce_dir.exists():
        print(f"Consolidated dataset not found at {ce_dir}")
        print("Download with:")
        print("  from datasets import load_dataset")
        print("  ds = load_dataset('NathanRoll/speech-emotion-dataset-consolidated')")
        print(f"  ds.save_to_disk('{ce_dir}')")
        return []

    try:
        import io

        import pyarrow as pa
        import soundfile as sf
    except ImportError as e:
        print(f"Required packages: {e}")
        return []

    print("Loading Consolidated Multilingual dataset from Arrow files...")

    samples = []

    # Dataset has 'train' split
    split_dir = ce_dir / 'train'
    if not split_dir.exists():
        split_dir = ce_dir

    arrow_files = sorted(split_dir.glob('*.arrow'))
    if not arrow_files:
        print(f"No arrow files found in {split_dir}")
        return []

    sample_idx = 0
    for arrow_file in arrow_files:
        with open(arrow_file, 'rb') as f:
            reader = pa.ipc.open_stream(f)
            table = reader.read_all()

        print(f"Processing {table.num_rows} samples from {arrow_file.name}...")

        for i in range(table.num_rows):
            try:
                audio_struct = table['audio'][i].as_py()
                emotion = table['emotion'][i].as_py().lower()
                _language = table['language'][i].as_py().lower()  # Unused: audio language
                source_dataset = table['dataset'][i].as_py()

                prosody_type = EMOTION_TO_PROSODY.get(emotion, PROSODY_TYPES["NEUTRAL"])

                # No text transcript available - use emotion-based placeholder
                emotion_texts = {
                    "anger": "Expressing anger",
                    "disgust": "Expressing disgust",
                    "fear": "Expressing fear",
                    "happiness": "Expressing happiness",
                    "neutral": "Speaking neutrally",
                    "sadness": "Expressing sadness",
                    "surprise": "Expressing surprise",
                }
                text = emotion_texts.get(emotion, "Emotional speech")

                # Decode audio bytes with soundfile
                audio_bytes = audio_struct['bytes']
                audio, sr = sf.read(io.BytesIO(audio_bytes))
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)

                # Extract features
                features = extract_prosody_features(audio, sr)

                # Create annotated text
                if emotion != "neutral":
                    annotated_text = f"<emotion type='{emotion}'>{text}</emotion>"
                else:
                    annotated_text = text

                sample = ProsodySample(
                    text=text,
                    annotated_text=annotated_text,
                    prosody_type=prosody_type,
                    audio_path=f"consolidated:{source_dataset}:{sample_idx}",
                    f0_mean=features["f0_mean"],
                    f0_std=features["f0_std"],
                    f0_range=features["f0_range"],
                    duration_s=features["duration_s"],
                    energy_rms=features["energy_rms"],
                    source=f"consolidated-{source_dataset.lower()}",
                )
                samples.append(sample)
                sample_idx += 1

                if sample_idx % 5000 == 0:
                    print(f"  Processed {sample_idx} samples")

            except Exception as e:
                if sample_idx % 1000 == 0:
                    print(f"  Error processing sample {sample_idx}: {e}")
                continue

    print(f"Processed {len(samples)} Consolidated Multilingual samples")
    return samples


def process_resd(resd_dir: Path) -> List[ProsodySample]:
    """
    Process RESD (Russian Emotional Speech Dataset) from HuggingFace.

    RESD: Russian emotional speech corpus (MIT license - commercial OK!)
    - Multiple speakers
    - 7 emotions: happiness, disgust, anger, fear, enthusiasm, neutral, sadness
    - ~1,396 samples (1116 train + 280 test)

    The dataset should be saved from HuggingFace using:
        ds = load_dataset('Aniemore/resd_annotated')
        ds.save_to_disk('data/prosody/resd')
    """
    resd_dir = Path(resd_dir)

    if not resd_dir.exists():
        print(f"RESD not found at {resd_dir}")
        print("Download with:")
        print("  from datasets import load_dataset")
        print("  ds = load_dataset('Aniemore/resd_annotated')")
        print(f"  ds.save_to_disk('{resd_dir}')")
        return []

    try:
        import io

        import pyarrow as pa
        import soundfile as sf
    except ImportError as e:
        print(f"Required packages: {e}")
        return []

    print("Loading RESD dataset from Arrow files...")

    samples = []

    # Process all splits (train, test)
    for split_name in ['train', 'test']:
        split_dir = resd_dir / split_name
        if not split_dir.exists():
            continue

        arrow_files = sorted(split_dir.glob('*.arrow'))
        if not arrow_files:
            continue

        sample_idx = 0
        for arrow_file in arrow_files:
            with open(arrow_file, 'rb') as f:
                reader = pa.ipc.open_stream(f)
                table = reader.read_all()

            print(f"Processing RESD {split_name} ({table.num_rows} samples from {arrow_file.name})...")

            for i in range(table.num_rows):
                try:
                    audio_struct = table['speech'][i].as_py()
                    emotion = table['emotion'][i].as_py().lower()
                    text = table['text'][i].as_py() if 'text' in table.column_names else 'Русский текст'

                    prosody_type = EMOTION_TO_PROSODY.get(emotion, PROSODY_TYPES["NEUTRAL"])

                    # Decode audio bytes with soundfile
                    audio_bytes = audio_struct['bytes']
                    audio, sr = sf.read(io.BytesIO(audio_bytes))
                    if len(audio.shape) > 1:
                        audio = audio.mean(axis=1)

                    # Extract features
                    features = extract_prosody_features(audio, sr)

                    # Create annotated text
                    if emotion != "neutral":
                        annotated_text = f"<emotion type='{emotion}'>{text}</emotion>"
                    else:
                        annotated_text = text

                    sample = ProsodySample(
                        text=text,
                        annotated_text=annotated_text,
                        prosody_type=prosody_type,
                        audio_path=f"resd:{split_name}:{sample_idx}",
                        f0_mean=features["f0_mean"],
                        f0_std=features["f0_std"],
                        f0_range=features["f0_range"],
                        duration_s=features["duration_s"],
                        energy_rms=features["energy_rms"],
                        source="resd",
                    )
                    samples.append(sample)
                    sample_idx += 1

                except Exception as e:
                    print(f"  Error processing {split_name}[{sample_idx}]: {e}")
                    continue

    print(f"Processed {len(samples)} RESD samples")
    return samples


# =============================================================================
# Auto-Annotation
# =============================================================================

def auto_annotate_directory(
    audio_dir: Path,
    transcript_file: Optional[Path] = None,
) -> List[ProsodySample]:
    """
    Auto-annotate audio files in a directory.

    Args:
        audio_dir: Directory containing .wav files
        transcript_file: Optional file with transcripts (format: filename<tab>text)

    Returns:
        List of auto-annotated samples
    """
    audio_dir = Path(audio_dir)
    samples = []

    try:
        import soundfile as sf
    except ImportError:
        print("soundfile required")
        return []

    # Load transcripts
    transcripts = {}
    if transcript_file and transcript_file.exists():
        with open(transcript_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    transcripts[parts[0]] = parts[1]

    audio_files = list(audio_dir.glob("**/*.wav"))
    print(f"Auto-annotating {len(audio_files)} files...")

    for audio_path in audio_files:
        try:
            # Get transcript
            text = transcripts.get(audio_path.stem, "Unknown text")

            # Load audio
            audio, sr = sf.read(audio_path)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            # Extract features
            features = extract_prosody_features(audio, sr)

            # Auto-annotate based on features
            prosody_type, annotated_text = auto_annotate_prosody(features, text)

            sample = ProsodySample(
                text=text,
                annotated_text=annotated_text,
                prosody_type=prosody_type,
                audio_path=str(audio_path),
                f0_mean=features["f0_mean"],
                f0_std=features["f0_std"],
                f0_range=features["f0_range"],
                duration_s=features["duration_s"],
                energy_rms=features["energy_rms"],
                source="auto",
            )
            samples.append(sample)

        except Exception as e:
            print(f"  Error processing {audio_path}: {e}")
            continue

    print(f"Auto-annotated {len(samples)} samples")
    return samples


# =============================================================================
# Dataset Management
# =============================================================================

def save_dataset(samples: List[ProsodySample], output_path: Path):
    """Save samples to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = [asdict(s) for s in samples]
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(samples)} samples to {output_path}")


def load_dataset(input_path: Path) -> List[ProsodySample]:
    """Load samples from JSON file."""
    with open(input_path) as f:
        data = json.load(f)

    samples = [ProsodySample(**d) for d in data]
    return samples


def compute_stats(samples: List[ProsodySample]) -> DatasetStats:
    """Compute dataset statistics."""
    stats = DatasetStats()
    stats.total_samples = len(samples)

    for sample in samples:
        # Count by prosody type
        pt = sample.prosody_type
        stats.samples_by_type[pt] = stats.samples_by_type.get(pt, 0) + 1

        # Accumulate totals
        stats.total_duration_s += sample.duration_s

    if samples:
        stats.mean_f0 = sum(s.f0_mean for s in samples) / len(samples)
        stats.mean_duration = stats.total_duration_s / len(samples)

    return stats


def print_stats(stats: DatasetStats):
    """Print dataset statistics."""
    print("\n=== Dataset Statistics ===")
    print(f"Total samples: {stats.total_samples}")
    print(f"Total duration: {stats.total_duration_s/3600:.1f} hours")
    print(f"Mean duration: {stats.mean_duration:.2f}s")
    print(f"Mean F0: {stats.mean_f0:.1f} Hz")
    print("\nSamples by prosody type:")

    type_names = {v: k for k, v in PROSODY_TYPES.items()}
    for pt, count in sorted(stats.samples_by_type.items()):
        name = type_names.get(pt, f"UNKNOWN_{pt}")
        print(f"  {name}: {count}")


def verify_dataset(samples: List[ProsodySample]) -> bool:
    """Verify dataset quality."""
    print("\n=== Dataset Verification ===")

    issues = []

    # Check for missing audio files
    missing_audio = 0
    for sample in samples:
        if not Path(sample.audio_path).exists():
            missing_audio += 1

    if missing_audio > 0:
        issues.append(f"Missing audio files: {missing_audio}")

    # Check for invalid F0 values
    invalid_f0 = sum(1 for s in samples if s.f0_mean <= 0)
    if invalid_f0 > 0:
        issues.append(f"Samples with invalid F0: {invalid_f0}")

    # Check for very short samples
    short_samples = sum(1 for s in samples if s.duration_s < 0.5)
    if short_samples > 0:
        issues.append(f"Very short samples (<0.5s): {short_samples}")

    # Check class balance
    stats = compute_stats(samples)
    if stats.samples_by_type:
        max_count = max(stats.samples_by_type.values())
        min_count = min(stats.samples_by_type.values())
        if max_count > min_count * 10:
            issues.append(f"Class imbalance: max/min ratio = {max_count/min_count:.1f}")

    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("Dataset verification passed!")
        return True


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare prosody training data for Phase B"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/prosody",
        help="Output directory for training data",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Prepare all data sources",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic data using TTS",
    )
    parser.add_argument(
        "--synthetic-count",
        type=int,
        default=1000,
        help="Number of synthetic samples to generate",
    )
    parser.add_argument(
        "--download-datasets",
        action="store_true",
        help="Process downloaded emotional speech datasets",
    )
    parser.add_argument(
        "--ravdess-dir",
        type=str,
        help="Path to RAVDESS dataset",
    )
    parser.add_argument(
        "--esd-dir",
        type=str,
        help="Path to ESD dataset",
    )
    parser.add_argument(
        "--crema-dir",
        type=str,
        help="Path to CREMA-D AudioWAV directory",
    )
    parser.add_argument(
        "--jvnv-dir",
        type=str,
        help="Path to JVNV HuggingFace dataset directory",
    )
    parser.add_argument(
        "--resd-dir",
        type=str,
        help="Path to RESD HuggingFace dataset directory",
    )
    parser.add_argument(
        "--consolidated-dir",
        type=str,
        help="Path to Consolidated English/Multilingual HuggingFace dataset directory",
    )
    parser.add_argument(
        "--auto-annotate",
        type=str,
        help="Directory of audio files to auto-annotate",
    )
    parser.add_argument(
        "--transcript-file",
        type=str,
        help="Transcript file for auto-annotation (filename<tab>text)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing dataset quality",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print dataset statistics",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_samples = []

    # Generate synthetic data
    if args.all or args.synthetic:
        print("\n=== Generating Synthetic Data ===")
        synthetic_dir = output_dir / "synthetic"
        samples = generate_synthetic_data(
            synthetic_dir,
            num_samples=args.synthetic_count,
        )
        all_samples.extend(samples)
        save_dataset(samples, output_dir / "synthetic.json")

    # Process RAVDESS
    if args.all or args.download_datasets:
        print("\n=== Processing RAVDESS ===")
        ravdess_dir = Path(args.ravdess_dir) if args.ravdess_dir else output_dir / "ravdess"
        samples = download_ravdess(ravdess_dir)
        if samples:
            all_samples.extend(samples)
            save_dataset(samples, output_dir / "ravdess.json")

    # Process ESD
    if args.esd_dir:
        print("\n=== Processing ESD ===")
        samples = process_esd(Path(args.esd_dir))
        if samples:
            all_samples.extend(samples)
            save_dataset(samples, output_dir / "esd.json")

    # Process CREMA-D
    if args.crema_dir:
        print("\n=== Processing CREMA-D ===")
        samples = process_crema(Path(args.crema_dir))
        if samples:
            all_samples.extend(samples)
            save_dataset(samples, output_dir / "crema.json")

    # Process JVNV (Japanese)
    if args.jvnv_dir:
        print("\n=== Processing JVNV (Japanese) ===")
        samples = process_jvnv(Path(args.jvnv_dir))
        if samples:
            all_samples.extend(samples)
            save_dataset(samples, output_dir / "jvnv.json")

    # Process RESD (Russian)
    if args.resd_dir:
        print("\n=== Processing RESD (Russian) ===")
        samples = process_resd(Path(args.resd_dir))
        if samples:
            all_samples.extend(samples)
            save_dataset(samples, output_dir / "resd.json")

    # Process Consolidated Multilingual
    if args.consolidated_dir:
        print("\n=== Processing Consolidated Multilingual ===")
        samples = process_consolidated_english(Path(args.consolidated_dir))
        if samples:
            all_samples.extend(samples)
            save_dataset(samples, output_dir / "consolidated.json")

    # Auto-annotate directory
    if args.auto_annotate:
        print("\n=== Auto-Annotating Audio ===")
        transcript_file = Path(args.transcript_file) if args.transcript_file else None
        samples = auto_annotate_directory(
            Path(args.auto_annotate),
            transcript_file,
        )
        if samples:
            all_samples.extend(samples)
            save_dataset(samples, output_dir / "auto_annotated.json")

    # Save combined dataset
    if all_samples:
        print("\n=== Saving Combined Dataset ===")
        save_dataset(all_samples, output_dir / "train.json")

        # Split into train/val
        random.shuffle(all_samples)
        split_idx = int(len(all_samples) * 0.9)
        train_samples = all_samples[:split_idx]
        val_samples = all_samples[split_idx:]

        save_dataset(train_samples, output_dir / "train_split.json")
        save_dataset(val_samples, output_dir / "val_split.json")

        print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    # Verify and print stats
    if args.verify or args.stats or all_samples:
        if all_samples:
            samples = all_samples
        elif (output_dir / "train.json").exists():
            samples = load_dataset(output_dir / "train.json")
        else:
            print("No dataset found to verify")
            return

        stats = compute_stats(samples)
        print_stats(stats)

        if args.verify:
            verify_dataset(samples)


if __name__ == "__main__":
    main()
