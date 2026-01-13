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
Prepare DisfluencySpeech data for paralinguistics training.

This script:
1. Reads parquet files from data/disfluency/data/
2. Extracts disfluency markers from annotated transcripts
3. Creates labeled audio samples for paralinguistics training

Disfluency Classes:
- filler_um: "um" filler words
- filler_uh: "uh" filler words
- filler_er: "er" filler words
- repetition: word/phrase repetitions
- false_start: abandoned utterances

Output: data/paralinguistics/disfluency/ with metadata.json
"""

import json
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm


DISFLUENCY_DIR = Path("data/disfluency/data")
OUTPUT_DIR = Path("data/paralinguistics/disfluency")


def extract_audio_bytes(audio_dict):
    """Extract audio bytes from parquet audio column."""
    if isinstance(audio_dict, dict) and 'bytes' in audio_dict:
        return audio_dict['bytes']
    return audio_dict


def parse_disfluency_annotations(annotated_text):
    """
    Parse annotated transcript to extract disfluency types.

    Common annotation patterns in DisfluencySpeech:
    - [um], [uh], [er] - filler words
    - [repeat] or repeated words - repetitions
    - [false_start] or incomplete - false starts
    - {} brackets for disfluent regions
    """
    labels = set()
    text_lower = annotated_text.lower() if annotated_text else ""

    # Check for filler words (case insensitive)
    if re.search(r'\bum+\b', text_lower):
        labels.add('filler_um')
    if re.search(r'\buh+\b', text_lower):
        labels.add('filler_uh')
    if re.search(r'\ber+\b', text_lower):
        labels.add('filler_er')
    if re.search(r'\bhmm+\b', text_lower):
        labels.add('filler_um')  # Map to um
    if re.search(r'\bah+\b', text_lower):
        labels.add('filler_uh')  # Map to uh

    # Check for repetitions (word repeated consecutively)
    words = text_lower.split()
    for i in range(len(words) - 1):
        if words[i] == words[i + 1] and len(words[i]) > 2:
            labels.add('repetition')
            break

    # Check for annotation markers
    if '[' in annotated_text and ']' in annotated_text:
        if re.search(r'\[rep', text_lower) or re.search(r'\[rp', text_lower):
            labels.add('repetition')
        if re.search(r'\[fs', text_lower) or re.search(r'\[false', text_lower):
            labels.add('false_start')

    return list(labels) if labels else None


def save_audio(audio_bytes, output_path):
    """Save audio bytes to WAV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(audio_bytes)


def process_split(split_name, parquet_files, output_dir):
    """Process a data split (train/val/test)."""
    print(f"\nProcessing {split_name} split...")

    samples = []
    class_counts = {}

    for parquet_file in tqdm(parquet_files, desc=f"Reading {split_name} parquet files"):
        df = pd.read_parquet(parquet_file)

        for idx, row in df.iterrows():
            # Extract audio
            audio_bytes = extract_audio_bytes(row['audio'])
            if not audio_bytes:
                continue

            # Parse disfluency annotations from all transcript columns
            labels = set()
            for col in ['transcript_annotated', 'transcript_a', 'transcript_b', 'transcript_c']:
                if col in row and row[col]:
                    parsed = parse_disfluency_annotations(str(row[col]))
                    if parsed:
                        labels.update(parsed)

            if not labels:
                continue

            # Save audio file
            sample_id = f"{split_name}_{parquet_file.stem}_{idx}"
            audio_path = output_dir / split_name / f"{sample_id}.wav"
            save_audio(audio_bytes, audio_path)

            # Track sample
            for label in labels:
                class_counts[label] = class_counts.get(label, 0) + 1

            samples.append({
                'id': sample_id,
                'path': str(audio_path.relative_to(output_dir.parent.parent)),
                'labels': list(labels),
                'transcript': row.get('transcript_annotated', ''),
            })

    return samples, class_counts


def main():
    print("=" * 60)
    print("DisfluencySpeech Data Preparation")
    print("=" * 60)

    if not DISFLUENCY_DIR.exists():
        print(f"ERROR: DisfluencySpeech not found at {DISFLUENCY_DIR}")
        print("Run: huggingface-cli download amaai-lab/DisfluencySpeech --local-dir data/disfluency")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find parquet files by split
    train_files = sorted(DISFLUENCY_DIR.glob("train-*.parquet"))
    val_files = sorted(DISFLUENCY_DIR.glob("validation-*.parquet"))
    test_files = sorted(DISFLUENCY_DIR.glob("test-*.parquet"))

    print("\nFound files:")
    print(f"  Train: {len(train_files)} parquet files")
    print(f"  Validation: {len(val_files)} parquet files")
    print(f"  Test: {len(test_files)} parquet files")

    all_samples = {}
    all_class_counts = {}

    # Process each split
    if train_files:
        samples, counts = process_split("train", train_files, OUTPUT_DIR)
        all_samples['train'] = samples
        for k, v in counts.items():
            all_class_counts[k] = all_class_counts.get(k, 0) + v

    if val_files:
        samples, counts = process_split("val", val_files, OUTPUT_DIR)
        all_samples['val'] = samples
        for k, v in counts.items():
            all_class_counts[k] = all_class_counts.get(k, 0) + v

    if test_files:
        samples, counts = process_split("test", test_files, OUTPUT_DIR)
        all_samples['test'] = samples
        for k, v in counts.items():
            all_class_counts[k] = all_class_counts.get(k, 0) + v

    # Save metadata
    metadata = {
        'source': 'DisfluencySpeech',
        'classes': list(all_class_counts.keys()),
        'class_counts': all_class_counts,
        'splits': {
            'train': len(all_samples.get('train', [])),
            'val': len(all_samples.get('val', [])),
            'test': len(all_samples.get('test', [])),
        },
        'samples': all_samples,
    }

    metadata_path = OUTPUT_DIR / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("\nTotal samples extracted:")
    for split, count in metadata['splits'].items():
        print(f"  {split}: {count}")

    print("\nClass distribution:")
    for cls, count in sorted(all_class_counts.items()):
        print(f"  {cls}: {count}")

    print(f"\nMetadata saved to: {metadata_path}")
    print(f"Audio files saved to: {OUTPUT_DIR}/")

    print("\nNext steps:")
    print("1. Add these classes to paralinguistics training config")
    print("2. Update train_paralinguistics.py to load this data")
    print("3. Retrain with combined dataset")


if __name__ == "__main__":
    main()
