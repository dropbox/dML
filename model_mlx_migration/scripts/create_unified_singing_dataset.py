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
Create unified singing dataset from all available sources.

Sources:
- ace_opencpop: 100,510 train samples (HuggingFace) - Chinese singing
- ravdess_song: RAVDESS song files - English emotion + singing
- vocalset: VocalSet singing techniques - English singing
- dynamicsuperb: DynamicSuperb singing data

Total: ~100K+ singing samples (51GB)

Unified columns:
- audio: Audio feature
- is_singing: bool (True for singing, False for speaking)
- transcription: Optional lyrics
- language: str
- source: str

Output: data/singing/unified_singing/ (HuggingFace format)
"""

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import Dataset, DatasetDict, Audio, load_from_disk, concatenate_datasets


def load_ace_opencpop(data_dir: Path) -> Optional[Dataset]:
    """Load ACE OpenCpop singing dataset (Chinese)."""
    print("Loading ace_opencpop...")
    ace_path = data_dir / "ace_opencpop"

    if not ace_path.exists():
        print(f"  Path not found: {ace_path}")
        return None

    try:
        ds = load_from_disk(str(ace_path))
    except Exception as e:
        print(f"  Failed to load: {e}")
        return None

    def process(example):
        return {
            "audio": example["audio"],
            "is_singing": True,  # All ACE samples are singing
            "transcription": example.get("transcription", ""),
            "language": "zh",
            "source": "ace_opencpop",
        }

    # Process train split
    datasets_to_concat = []
    for split in ["train", "validation", "test"]:
        if split in ds:
            processed = ds[split].map(process, remove_columns=ds[split].column_names)
            datasets_to_concat.append(processed)
            print(f"  {split}: {len(ds[split])} samples")

    if not datasets_to_concat:
        return None

    combined = concatenate_datasets(datasets_to_concat)
    print(f"  Total loaded: {len(combined)} samples")
    return combined


def load_ravdess_song(data_dir: Path) -> Optional[Dataset]:
    """Load RAVDESS song files."""
    print("Loading ravdess_song...")
    ravdess_path = data_dir / "ravdess_song"

    if not ravdess_path.exists():
        print(f"  Path not found: {ravdess_path}")
        return None

    # Find all wav files
    audio_files = list(ravdess_path.rglob("*.wav"))
    print(f"  Found {len(audio_files)} audio files")

    if not audio_files:
        return None

    audio_paths = [str(f) for f in audio_files]

    ds = Dataset.from_dict({
        "audio": audio_paths,
        "is_singing": [True] * len(audio_paths),  # All RAVDESS song files are singing
        "transcription": [""] * len(audio_paths),
        "language": ["en"] * len(audio_paths),
        "source": ["ravdess_song"] * len(audio_paths),
    })

    # Cast audio column
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    print(f"  Loaded {len(ds)} samples")
    return ds


def load_vocalset(data_dir: Path) -> Optional[Dataset]:
    """Load VocalSet singing technique dataset."""
    print("Loading vocalset...")

    # Check both possible locations
    vocalset_paths = [
        data_dir / "vocalset",
        data_dir.parent / "vocalset",  # data/vocalset
    ]

    vocalset_path = None
    for p in vocalset_paths:
        if p.exists() and any(p.rglob("*.wav")):
            vocalset_path = p
            break

    if vocalset_path is None:
        print("  VocalSet not found")
        return None

    # Find all wav files
    audio_files = list(vocalset_path.rglob("*.wav"))
    print(f"  Found {len(audio_files)} audio files in {vocalset_path}")

    if not audio_files:
        return None

    audio_paths = [str(f) for f in audio_files]

    ds = Dataset.from_dict({
        "audio": audio_paths,
        "is_singing": [True] * len(audio_paths),
        "transcription": [""] * len(audio_paths),
        "language": ["en"] * len(audio_paths),
        "source": ["vocalset"] * len(audio_paths),
    })

    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    print(f"  Loaded {len(ds)} samples")
    return ds


def load_dynamicsuperb(data_dir: Path) -> Optional[Dataset]:
    """Load DynamicSuperb singing data."""
    print("Loading dynamicsuperb...")
    ds_path = data_dir / "dynamicsuperb"

    if not ds_path.exists():
        print(f"  Path not found: {ds_path}")
        return None

    # Check if it's HuggingFace format
    if (ds_path / "dataset_dict.json").exists() or (ds_path / "data").exists():
        try:
            ds = load_from_disk(str(ds_path))
            def process(example):
                return {
                    "audio": example["audio"],
                    "is_singing": True,
                    "transcription": example.get("transcription", ""),
                    "language": "zh",  # Chinese singing
                    "source": "dynamicsuperb",
                }

            datasets_to_concat = []
            for split in ds.keys():
                processed = ds[split].map(process, remove_columns=ds[split].column_names)
                datasets_to_concat.append(processed)
                print(f"  {split}: {len(ds[split])} samples")

            combined = concatenate_datasets(datasets_to_concat)
            print(f"  Total loaded: {len(combined)} samples")
            return combined
        except Exception as e:
            print(f"  HuggingFace load failed: {e}")

    # Try loading raw audio files
    audio_files = list(ds_path.rglob("*.wav")) + list(ds_path.rglob("*.mp3")) + list(ds_path.rglob("*.flac"))
    print(f"  Found {len(audio_files)} audio files")

    if not audio_files:
        return None

    audio_paths = [str(f) for f in audio_files]

    ds = Dataset.from_dict({
        "audio": audio_paths,
        "is_singing": [True] * len(audio_paths),
        "transcription": [""] * len(audio_paths),
        "language": ["zh"] * len(audio_paths),
        "source": ["dynamicsuperb"] * len(audio_paths),
    })

    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    print(f"  Loaded {len(ds)} samples")
    return ds


def ensure_audio_feature(ds: Dataset) -> Dataset:
    """Ensure dataset has Audio feature type."""
    from datasets.features import Audio as AudioFeature
    if ds is None:
        return None
    try:
        ds = ds.cast_column("audio", AudioFeature(sampling_rate=16000))
    except Exception as e:
        print(f"  Warning: Could not cast audio column: {e}")
    return ds


def create_unified_singing_dataset(data_dir: str = "data/singing", output_dir: str = "data/singing/unified_singing"):
    """Create unified singing dataset from all sources."""
    data_path = Path(data_dir)
    output_path = Path(output_dir)

    datasets_to_concat = []

    loaders = [
        load_ace_opencpop,
        load_ravdess_song,
        load_vocalset,
        load_dynamicsuperb,
    ]

    for loader in loaders:
        try:
            ds = loader(data_path)
            if ds is not None and len(ds) > 0:
                ds = ensure_audio_feature(ds)
                datasets_to_concat.append(ds)
        except Exception as e:
            print(f"  Error loading {loader.__name__}: {e}")

    if not datasets_to_concat:
        print("ERROR: No datasets loaded!")
        return

    print(f"\nCombining {len(datasets_to_concat)} datasets...")
    unified = concatenate_datasets(datasets_to_concat)
    print(f"Total samples: {len(unified)}")

    # Print statistics
    print("\nSource distribution:")
    source_counts = {}
    for example in unified:
        s = example["source"]
        source_counts[s] = source_counts.get(s, 0) + 1

    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  {source}: {count}")

    print("\nLanguage distribution:")
    lang_counts = {}
    for example in unified:
        lang = example["language"]
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

    for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
        print(f"  {lang}: {count}")

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
    create_unified_singing_dataset()
