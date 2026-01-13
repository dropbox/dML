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
Create unified emotion dataset from all available sources.

Sources:
- consolidated_66k: 66,291 samples (HuggingFace) - 7 emotions
- iemocap: 10,039 samples (HuggingFace) - 10 emotions
- resd: 1,396 samples (HuggingFace) - 7 emotions
- nemo_polish: ~4,481 samples (TSV + audio) - emotions in Polish data
- jvnv: 1,615 samples (HuggingFace, test only) - style column

Total: ~83K+ emotion samples

Unified emotion taxonomy (10 classes):
0: neutral, 1: happy, 2: sad, 3: angry, 4: fear, 5: disgust,
6: surprise, 7: calm, 8: excited, 9: other

Output: data/emotion/unified_emotion/ (HuggingFace format)
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import Dataset, DatasetDict, Audio, load_from_disk, concatenate_datasets
import pandas as pd

# Unified emotion taxonomy
UNIFIED_EMOTIONS = [
    "neutral",    # 0
    "happy",      # 1
    "sad",        # 2
    "angry",      # 3
    "fear",       # 4
    "disgust",    # 5
    "surprise",   # 6
    "calm",       # 7
    "excited",    # 8
    "other",      # 9
]

EMOTION_TO_ID = {e: i for i, e in enumerate(UNIFIED_EMOTIONS)}

# Mapping from source datasets to unified taxonomy
CONSOLIDATED_EMOTION_MAP = {
    "neutral": "neutral",
    "happiness": "happy",
    "sadness": "sad",
    "anger": "angry",
    "fear": "fear",
    "disgust": "disgust",
    "surprise": "surprise",
}

IEMOCAP_EMOTION_MAP = {
    "neu": "neutral",
    "hap": "happy",
    "sad": "sad",
    "ang": "angry",
    "fea": "fear",
    "dis": "disgust",
    "sur": "surprise",
    "fru": "angry",  # frustrated -> angry
    "exc": "excited",
    "oth": "other",
}

RESD_EMOTION_MAP = {
    "neutral": "neutral",
    "happiness": "happy",
    "sadness": "sad",
    "anger": "angry",
    "fear": "fear",
    "disgust": "disgust",
    "enthusiasm": "excited",
}

JVNV_STYLE_MAP = {
    "angry": "angry",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "whisper": "other",  # style not emotion
}

NEMO_EMOTION_MAP = {
    "neutral": "neutral",
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "fear": "fear",
    "disgust": "disgust",
    "surprised": "surprise",
}


def map_emotion(emotion: str, source_map: Dict[str, str]) -> Tuple[str, int]:
    """Map source emotion to unified taxonomy."""
    emotion_lower = emotion.lower().strip()
    unified = source_map.get(emotion_lower, "other")
    return unified, EMOTION_TO_ID[unified]


def load_consolidated_66k(data_dir: Path) -> Dataset:
    """Load consolidated_66k emotion dataset with error handling for corrupted audio."""
    print("Loading consolidated_66k...")
    ds = load_from_disk(str(data_dir / "consolidated_66k"))
    train_ds = ds["train"]
    total = len(train_ds)

    # Process samples with error handling - access by index
    valid_samples = []
    skipped = 0
    bad_indices = []

    print(f"  Processing {total} samples (with corruption handling)...")

    for i in range(total):
        if i % 10000 == 0:
            print(f"  Progress: {i}/{total} ({100*i/total:.1f}%), skipped: {skipped}, valid: {len(valid_samples)}")

        try:
            # Access sample by index - this triggers audio loading
            example = train_ds[i]
            audio = example["audio"]

            # Check if audio array is valid
            if audio is None:
                skipped += 1
                bad_indices.append(i)
                continue
            if isinstance(audio, dict):
                arr = audio.get("array")
                if arr is None or len(arr) == 0:
                    skipped += 1
                    bad_indices.append(i)
                    continue

            unified, emotion_id = map_emotion(example["emotion"], CONSOLIDATED_EMOTION_MAP)
            valid_samples.append({
                "audio": audio,
                "emotion": unified,
                "emotion_id": emotion_id,
                "source": "consolidated_66k",
                "language": example.get("language", "en"),
            })
        except Exception as e:
            skipped += 1
            bad_indices.append(i)
            if skipped <= 10:  # Only log first 10 errors
                print(f"  Skipping sample {i}: {type(e).__name__}: {str(e)[:100]}")
            elif skipped == 11:
                print("  (suppressing further error messages...)")

    print(f"  Processed {total} samples, skipped {skipped} corrupted, kept {len(valid_samples)}")
    if bad_indices:
        print(f"  Bad indices (first 20): {bad_indices[:20]}")

    if not valid_samples:
        print("  ERROR: No valid samples found!")
        return None

    # Create dataset from valid samples
    processed = Dataset.from_dict({
        "audio": [s["audio"] for s in valid_samples],
        "emotion": [s["emotion"] for s in valid_samples],
        "emotion_id": [s["emotion_id"] for s in valid_samples],
        "source": [s["source"] for s in valid_samples],
        "language": [s["language"] for s in valid_samples],
    })

    print(f"  Loaded {len(processed)} samples")
    return processed


def load_iemocap(data_dir: Path) -> Dataset:
    """Load IEMOCAP emotion dataset."""
    print("Loading iemocap...")
    ds = load_from_disk(str(data_dir / "iemocap"))

    def process(example):
        unified, emotion_id = map_emotion(example["emotion"], IEMOCAP_EMOTION_MAP)
        return {
            "audio": example["audio"],
            "emotion": unified,
            "emotion_id": emotion_id,
            "source": "iemocap",
            "language": "en",
        }

    processed = ds["train"].map(process, remove_columns=ds["train"].column_names)
    print(f"  Loaded {len(processed)} samples")
    return processed


def load_resd(data_dir: Path) -> Dataset:
    """Load RESD emotion dataset."""
    print("Loading resd...")
    ds = load_from_disk(str(data_dir / "resd"))

    def process(example):
        unified, emotion_id = map_emotion(example["emotion"], RESD_EMOTION_MAP)
        return {
            "audio": example["speech"],  # RESD uses 'speech' column
            "emotion": unified,
            "emotion_id": emotion_id,
            "source": "resd",
            "language": "ru",  # Russian dataset
        }

    # Combine train and test
    datasets_to_concat = []
    for split in ds.keys():
        processed = ds[split].map(process, remove_columns=ds[split].column_names)
        datasets_to_concat.append(processed)

    combined = concatenate_datasets(datasets_to_concat)
    print(f"  Loaded {len(combined)} samples")
    return combined


def load_jvnv(data_dir: Path) -> Optional[Dataset]:
    """Load JVNV emotion dataset (Japanese)."""
    print("Loading jvnv...")
    try:
        ds = load_from_disk(str(data_dir / "jvnv"))
    except Exception as e:
        print(f"  Failed to load jvnv: {e}")
        return None

    def process(example):
        style = example.get("style", "neutral")
        unified, emotion_id = map_emotion(style, JVNV_STYLE_MAP)
        return {
            "audio": example["audio"],
            "emotion": unified,
            "emotion_id": emotion_id,
            "source": "jvnv",
            "language": "ja",
        }

    # Only test split available
    processed = ds["test"].map(process, remove_columns=ds["test"].column_names)
    print(f"  Loaded {len(processed)} samples")
    return processed


def load_nemo_polish(data_dir: Path) -> Optional[Dataset]:
    """Load nEMO Polish emotion dataset."""
    print("Loading nemo_polish...")
    nemo_dir = data_dir / "nemo_polish"
    tsv_path = nemo_dir / "data.tsv"
    samples_dir = nemo_dir / "samples"

    if not tsv_path.exists():
        print(f"  TSV file not found: {tsv_path}")
        return None

    df = pd.read_csv(tsv_path, sep="\t")
    print(f"  Found {len(df)} entries in TSV")

    # Build dataset
    audio_paths = []
    emotions = []
    emotion_ids = []

    for _, row in df.iterrows():
        audio_path = samples_dir / row["file_id"]
        if audio_path.exists():
            unified, eid = map_emotion(row["emotion"], NEMO_EMOTION_MAP)
            audio_paths.append(str(audio_path))
            emotions.append(unified)
            emotion_ids.append(eid)

    if not audio_paths:
        print("  No valid audio files found")
        return None

    ds = Dataset.from_dict({
        "audio": audio_paths,
        "emotion": emotions,
        "emotion_id": emotion_ids,
        "source": ["nemo_polish"] * len(audio_paths),
        "language": ["pl"] * len(audio_paths),
    })

    # Cast audio column to Audio feature type
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    print(f"  Loaded {len(ds)} samples")
    return ds


def ensure_audio_feature(ds: Dataset) -> Dataset:
    """Ensure dataset has Audio feature type."""
    from datasets.features import Audio as AudioFeature
    if ds is None:
        return None
    # Cast audio column to ensure consistent type
    try:
        ds = ds.cast_column("audio", AudioFeature(sampling_rate=16000))
    except Exception as e:
        print(f"  Warning: Could not cast audio column: {e}")
    return ds


def create_unified_emotion_dataset(data_dir: str = "data/emotion", output_dir: str = "data/emotion/unified_emotion"):
    """Create unified emotion dataset from all sources."""
    data_path = Path(data_dir)
    output_path = Path(output_dir)

    datasets_to_concat = []

    # Load each source
    loaders = [
        load_consolidated_66k,
        load_iemocap,
        load_resd,
        load_jvnv,
        load_nemo_polish,
    ]

    for loader in loaders:
        try:
            ds = loader(data_path)
            if ds is not None and len(ds) > 0:
                # Ensure consistent Audio feature type
                ds = ensure_audio_feature(ds)
                datasets_to_concat.append(ds)
        except Exception as e:
            print(f"  Error loading {loader.__name__}: {e}")

    if not datasets_to_concat:
        print("ERROR: No datasets loaded!")
        return

    # Combine all datasets
    print(f"\nCombining {len(datasets_to_concat)} datasets...")
    unified = concatenate_datasets(datasets_to_concat)
    print(f"Total samples: {len(unified)}")

    # Print statistics
    print("\nEmotion distribution:")
    emotion_counts = {}
    for example in unified:
        e = example["emotion"]
        emotion_counts[e] = emotion_counts.get(e, 0) + 1

    for emotion, count in sorted(emotion_counts.items(), key=lambda x: -x[1]):
        print(f"  {emotion}: {count} ({100*count/len(unified):.1f}%)")

    print("\nSource distribution:")
    source_counts = {}
    for example in unified:
        s = example["source"]
        source_counts[s] = source_counts.get(s, 0) + 1

    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  {source}: {count}")

    # Shuffle and split
    print("\nShuffling and splitting...")
    unified = unified.shuffle(seed=42)

    # 90% train, 10% validation
    n_val = int(len(unified) * 0.1)
    train_ds = unified.select(range(n_val, len(unified)))
    val_ds = unified.select(range(n_val))

    print(f"Train: {len(train_ds)}, Validation: {len(val_ds)}")

    # Save
    output_path.mkdir(parents=True, exist_ok=True)
    dataset_dict = DatasetDict({
        "train": train_ds,
        "validation": val_ds,
    })

    print(f"\nSaving to {output_path}...")
    dataset_dict.save_to_disk(str(output_path))
    print("Done!")

    return dataset_dict


if __name__ == "__main__":
    create_unified_emotion_dataset()
