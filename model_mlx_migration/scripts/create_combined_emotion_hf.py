#!/usr/bin/env python3
"""Create combined emotion dataset from CREMA-D + RAVDESS for RichDecoder v2."""
import json
import random
from pathlib import Path
from datasets import Dataset, DatasetDict, Audio
from collections import Counter

# Target 7-class emotion scheme (no contempt data available)
EMOTION_CLASSES = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
EMOTION_TO_ID = {e: i for i, e in enumerate(EMOTION_CLASSES)}

# CREMA-D emotion mapping
CREMA_EMOTION_MAP = {
    'angry': 'angry',
    'disgust': 'disgust',
    'fear': 'fear',
    'happy': 'happy',
    'neutral': 'neutral',
    'sad': 'sad'
}

# RAVDESS emotion mapping (numeric -> name)
RAVDESS_EMOTION_MAP = {
    0: 'neutral',
    1: 'neutral',  # calm -> neutral
    2: 'happy',
    3: 'sad',
    4: 'angry',
    5: 'fear',
    6: 'disgust',
    7: 'surprise'
}

# CREMA-D sentence texts
CREMA_SENTENCES = {
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


def load_crema_d():
    """Load CREMA-D samples."""
    samples = []
    audio_dir = Path('/Users/ayates/model_mlx_migration/data/emotion/crema-d')

    for wav_file in audio_dir.glob('*.wav'):
        parts = wav_file.stem.split('_')
        if len(parts) != 4:
            continue
        actor_id, sentence_code, emotion_code, level = parts

        emotion = CREMA_EMOTION_MAP.get(emotion_code.lower())
        if emotion is None:
            # Try uppercase codes
            code_map = {'ANG': 'angry', 'DIS': 'disgust', 'FEA': 'fear',
                       'HAP': 'happy', 'NEU': 'neutral', 'SAD': 'sad'}
            emotion = code_map.get(emotion_code)

        if emotion is None:
            continue

        samples.append({
            'audio': str(wav_file),
            'text': CREMA_SENTENCES.get(sentence_code, ''),
            'emotion': emotion,
            'language': 'en',
            'source': 'crema-d'
        })

    print(f"CREMA-D: {len(samples)} samples")
    return samples


def load_ravdess():
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

        # Handle different manifest formats
        audio_path_str = item.get('path') or item.get('audio_path', '')
        if not audio_path_str:
            continue

        audio_path = Path(audio_path_str)
        if not audio_path.exists():
            # Try just the filename in audio_dir
            audio_path = audio_dir / audio_path.name

        if not audio_path.exists():
            continue

        samples.append({
            'audio': str(audio_path),
            'text': '',  # RAVDESS has no transcription
            'emotion': emotion,
            'language': 'en',
            'source': 'ravdess'
        })

    print(f"RAVDESS: {len(samples)} samples")
    return samples


def main():
    print("Creating combined emotion dataset for RichDecoder v2")
    print("=" * 60)

    # Load all sources
    crema_samples = load_crema_d()
    ravdess_samples = load_ravdess()

    all_samples = crema_samples + ravdess_samples
    print(f"\nTotal: {len(all_samples)} samples")

    # Count emotions
    emotion_counts = Counter(s['emotion'] for s in all_samples)
    print("\nEmotion distribution:")
    for emotion in EMOTION_CLASSES:
        count = emotion_counts.get(emotion, 0)
        print(f"  {emotion}: {count} ({100*count/len(all_samples):.1f}%)")

    # Filter to only emotions in our scheme
    filtered_samples = [s for s in all_samples if s['emotion'] in EMOTION_TO_ID]
    print(f"\nFiltered to target scheme: {len(filtered_samples)} samples")

    # Split train/val (90/10)
    random.seed(42)
    random.shuffle(filtered_samples)
    split_idx = int(len(filtered_samples) * 0.9)
    train_samples = filtered_samples[:split_idx]
    val_samples = filtered_samples[split_idx:]

    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    # Create datasets
    def samples_to_dict(samples):
        return {
            'audio': [s['audio'] for s in samples],
            'text': [s['text'] for s in samples],
            'emotion': [s['emotion'] for s in samples],
            'language': [s['language'] for s in samples],
        }

    train_ds = Dataset.from_dict(samples_to_dict(train_samples))
    val_ds = Dataset.from_dict(samples_to_dict(val_samples))

    # Cast audio column
    train_ds = train_ds.cast_column('audio', Audio(sampling_rate=16000))
    val_ds = val_ds.cast_column('audio', Audio(sampling_rate=16000))

    # Save
    output_dir = Path('/Users/ayates/model_mlx_migration/data/emotion/combined_emotion_hf')
    output_dir.mkdir(parents=True, exist_ok=True)

    ds_dict = DatasetDict({'train': train_ds, 'validation': val_ds})
    ds_dict.save_to_disk(str(output_dir))

    print(f"\nSaved to: {output_dir}")

    # Final distribution
    print("\nFinal train emotion distribution:")
    train_counts = Counter(s['emotion'] for s in train_samples)
    for emotion in EMOTION_CLASSES:
        count = train_counts.get(emotion, 0)
        pct = 100*count/len(train_samples) if len(train_samples) > 0 else 0
        print(f"  {emotion}: {count} ({pct:.1f}%)")


if __name__ == '__main__':
    main()
