#!/usr/bin/env python3
"""
Create multi-task dataset for RichDecoder v3.

Combines:
- CREMA-D + RAVDESS (emotion)
- VocalSound (paralinguistics)
- LibriSpeech features (text, pseudo-emotion)

Each sample has labels for multiple tasks.
"""
import json
import random
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from typing import Optional, List, Dict

# Task label definitions
EMOTION_CLASSES = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
# Moved PARA_CLASSES to below ESC50_PARA_MAP

# Emotion mappings
CREMA_EMOTION_MAP = {
    'ANG': 'angry', 'DIS': 'disgust', 'FEA': 'fear',
    'HAP': 'happy', 'NEU': 'neutral', 'SAD': 'sad'
}
RAVDESS_EMOTION_MAP = {
    0: 'neutral', 1: 'neutral', 2: 'happy', 3: 'sad',
    4: 'angry', 5: 'fear', 6: 'disgust', 7: 'surprise'
}

# Paralinguistic mappings for all sources
ESC50_PARA_MAP = {
    'clapping': 'clapping',
    'laughing': 'laugh',
    'breathing': 'breath',
    'crying_baby': 'cry',
    'coughing': 'cough',
    'snoring': 'snore',
    'sneezing': 'sneeze',
}

# Para class list (extended)
PARA_CLASSES = [
    'speech', 'laugh', 'cough', 'sigh', 'breath', 'cry', 'yawn',
    'throat_clear', 'sneeze', 'gasp', 'groan', 'snore', 'filler',
    'silence', 'clapping'
]


@dataclass
class MultiTaskSample:
    """A sample with multi-task labels."""
    audio_path: str
    text: str = ""
    emotion: Optional[str] = None
    para: Optional[str] = None
    language: str = "en"
    source: str = "unknown"

    def to_dict(self) -> Dict:
        return {
            'audio_path': self.audio_path,
            'text': self.text,
            'emotion': self.emotion,
            'para': self.para,
            'language': self.language,
            'source': self.source
        }


def load_crema_d() -> List[MultiTaskSample]:
    """Load CREMA-D emotion samples."""
    samples = []
    audio_dir = Path('/Users/ayates/model_mlx_migration/data/emotion/crema-d')

    sentences = {
        'IEO': "It's eleven o'clock",
        'TIE': "That is exactly what happened",
        'IOM': "I'm on my way to the meeting",
        'IWW': "I wonder what this is about",
        'TAI': "The airplane is almost full",
        'MTI': "Maybe tomorrow it will be cold",
        'IWL': "I would like a new alarm clock",
        'ITH': "I think I have a doctor's appointment",
        'DFA': "Don't forget a jacket",
        'ITS': "I think I've seen this before",
        'TSI': "The surface is slick",
        'WSI': "We'll stop in a couple of minutes"
    }

    for wav_file in audio_dir.glob('*.wav'):
        parts = wav_file.stem.split('_')
        if len(parts) != 4:
            continue
        actor_id, sentence_code, emotion_code, level = parts

        emotion = CREMA_EMOTION_MAP.get(emotion_code)
        if emotion is None:
            continue

        samples.append(MultiTaskSample(
            audio_path=str(wav_file),
            text=sentences.get(sentence_code, ''),
            emotion=emotion,
            para='speech',  # Speech, not paralinguistic
            language='en',
            source='crema-d'
        ))

    print(f"CREMA-D: {len(samples)} samples")
    return samples


def load_ravdess() -> List[MultiTaskSample]:
    """Load RAVDESS emotion samples."""
    samples = []
    manifest_path = Path('/Users/ayates/model_mlx_migration/data/singing/ravdess_emotions_hf/manifest.json')
    audio_dir = Path('/Users/ayates/model_mlx_migration/data/singing/ravdess_emotions_hf')

    with open(manifest_path) as f:
        manifest = json.load(f)

    for item in manifest:
        label = item.get('label')
        if label is None or label not in RAVDESS_EMOTION_MAP:
            continue

        emotion = RAVDESS_EMOTION_MAP[label]
        audio_path_str = item.get('path', '')
        if not audio_path_str:
            continue

        audio_path = Path(audio_path_str)
        if not audio_path.exists():
            audio_path = audio_dir / audio_path.name
        if not audio_path.exists():
            continue

        samples.append(MultiTaskSample(
            audio_path=str(audio_path),
            text='',  # RAVDESS has no transcription
            emotion=emotion,
            para='speech',
            language='en',
            source='ravdess'
        ))

    print(f"RAVDESS: {len(samples)} samples")
    return samples


def load_paralinguistic_sources() -> List[MultiTaskSample]:
    """Load all paralinguistic sources with proper labels."""
    samples = []
    base_dir = Path('/Users/ayates/model_mlx_migration/data/paralinguistics')

    # 1. ESC50 para (280 samples with diverse labels)
    esc50_manifest = base_dir / 'esc50_para' / 'manifest.json'
    if esc50_manifest.exists():
        with open(esc50_manifest) as f:
            manifest = json.load(f)
        for item in manifest:
            label = item.get('label', 'unknown')
            para = ESC50_PARA_MAP.get(label, 'speech')
            audio_path = Path(item.get('path', item.get('audio_path', '')))
            if audio_path.exists():
                samples.append(MultiTaskSample(
                    audio_path=str(audio_path),
                    text='', emotion='neutral', para=para,
                    language='en', source='esc50_para'
                ))
        print(f"ESC50 para: {len([s for s in samples if s.source == 'esc50_para'])} samples")

    # 2. Laughterscape (8170 laughter samples)
    laugh_manifest = base_dir / 'laughterscape' / 'manifest.json'
    if laugh_manifest.exists():
        with open(laugh_manifest) as f:
            manifest = json.load(f)
        for item in manifest:
            audio_path = Path(item.get('path', ''))
            if audio_path.exists():
                samples.append(MultiTaskSample(
                    audio_path=str(audio_path),
                    text='', emotion='happy', para='laugh',  # Laughter often happy
                    language='en', source='laughterscape'
                ))
        print(f"Laughterscape: {len([s for s in samples if s.source == 'laughterscape'])} samples")

    # 3. CoughVid (972 cough samples)
    cough_manifest = base_dir / 'coughvid' / 'manifest.json'
    if cough_manifest.exists():
        with open(cough_manifest) as f:
            manifest = json.load(f)
        for item in manifest:
            audio_path = Path(item.get('path', item.get('audio_path', '')))
            if audio_path.exists():
                samples.append(MultiTaskSample(
                    audio_path=str(audio_path),
                    text='', emotion='neutral', para='cough',
                    language='en', source='coughvid'
                ))
        print(f"CoughVid: {len([s for s in samples if s.source == 'coughvid'])} samples")

    # 4. Fillers (from directory - "uh", "um", etc.)
    fillers_dir = base_dir / 'fillers'
    if fillers_dir.exists():
        filler_count = 0
        for wav_file in fillers_dir.glob('*.wav'):
            samples.append(MultiTaskSample(
                audio_path=str(wav_file),
                text='', emotion='neutral', para='filler',
                language='en', source='fillers'
            ))
            filler_count += 1
        print(f"Fillers: {filler_count} samples")

    # 5. Silence samples
    silence_dir = base_dir / 'silence'
    if silence_dir.exists():
        silence_count = 0
        for wav_file in silence_dir.glob('*.wav'):
            samples.append(MultiTaskSample(
                audio_path=str(wav_file),
                text='', emotion='neutral', para='silence',
                language='en', source='silence'
            ))
            silence_count += 1
        print(f"Silence: {silence_count} samples")

    return samples


def load_librispeech_features() -> List[MultiTaskSample]:
    """Load LibriSpeech pre-extracted features as samples."""
    samples = []
    features_dir = Path('/Users/ayates/model_mlx_migration/data/emotion_punctuation/librispeech_features')

    # These are .npz files with pre-extracted encoder features
    # For v3, we'll use them with pseudo-emotion labels (neutral)
    for npz_file in list(features_dir.glob('*.npz'))[:5000]:  # Limit for now
        samples.append(MultiTaskSample(
            audio_path=str(npz_file),
            text='',  # Would need to load from npz
            emotion='neutral',  # Pseudo-label
            para='speech',
            language='en',
            source='librispeech'
        ))

    print(f"LibriSpeech features: {len(samples)} samples")
    return samples


def main():
    print("=" * 70)
    print("Creating Multi-Task Dataset for RichDecoder v3")
    print("=" * 70)

    # Load all sources
    crema_samples = load_crema_d()
    ravdess_samples = load_ravdess()
    para_samples = load_paralinguistic_sources()
    # librispeech_samples = load_librispeech_features()  # Skip for now

    all_samples = crema_samples + ravdess_samples + para_samples
    print(f"\nTotal: {len(all_samples)} samples")

    # Statistics
    print("\n=== Emotion Distribution ===")
    emotion_counts = Counter(s.emotion for s in all_samples if s.emotion)
    for emotion in EMOTION_CLASSES:
        count = emotion_counts.get(emotion, 0)
        print(f"  {emotion}: {count}")

    print("\n=== Paralinguistics Distribution ===")
    para_counts = Counter(s.para for s in all_samples if s.para)
    for para in PARA_CLASSES:
        count = para_counts.get(para, 0)
        print(f"  {para}: {count}")

    print("\n=== Source Distribution ===")
    source_counts = Counter(s.source for s in all_samples)
    for source, count in source_counts.items():
        print(f"  {source}: {count}")

    # Split train/val
    random.seed(42)
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * 0.9)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    print(f"\nTrain: {len(train_samples)}, Val: {len(val_samples)}")

    # Save as JSON manifest
    output_dir = Path('/Users/ayates/model_mlx_migration/data/v3_multitask')
    output_dir.mkdir(parents=True, exist_ok=True)

    train_manifest = [s.to_dict() for s in train_samples]
    val_manifest = [s.to_dict() for s in val_samples]

    with open(output_dir / 'train_manifest.json', 'w') as f:
        json.dump(train_manifest, f, indent=2)

    with open(output_dir / 'val_manifest.json', 'w') as f:
        json.dump(val_manifest, f, indent=2)

    # Save metadata
    metadata = {
        'emotion_classes': EMOTION_CLASSES,
        'para_classes': PARA_CLASSES,
        'total_samples': len(all_samples),
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'sources': dict(source_counts)
    }
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved to: {output_dir}")
    print("Files: train_manifest.json, val_manifest.json, metadata.json")


if __name__ == '__main__':
    main()
