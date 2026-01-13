#!/usr/bin/env python3
"""Convert CREMA-D to HuggingFace Dataset format for RichDecoder training."""
from pathlib import Path
from datasets import Dataset, DatasetDict, Audio

# CREMA-D emotion codes to unified labels
EMOTION_MAP = {
    'ANG': 'angry',
    'DIS': 'disgust',
    'FEA': 'fear',
    'HAP': 'happy',
    'NEU': 'neutral',
    'SAD': 'sad'
}

# Sentence codes (12 sentences spoken by actors)
SENTENCE_TEXT = {
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


def parse_filename(filename):
    """Parse CREMA-D filename: {ActorID}_{Sentence}_{Emotion}_{Level}.wav"""
    parts = filename.replace('.wav', '').split('_')
    if len(parts) != 4:
        return None
    actor_id, sentence_code, emotion_code, level = parts
    if emotion_code not in EMOTION_MAP:
        return None
    return {
        'actor_id': actor_id,
        'sentence_code': sentence_code,
        'sentence': SENTENCE_TEXT.get(sentence_code, ''),
        'emotion': EMOTION_MAP[emotion_code],
        'emotion_code': emotion_code,
        'level': level
    }


def create_dataset():
    """Create HuggingFace dataset from CREMA-D audio files."""
    audio_dir = Path('/Users/ayates/model_mlx_migration/data/emotion/crema-d')
    output_dir = Path('/Users/ayates/model_mlx_migration/data/emotion/crema_d_hf')

    # Collect all samples
    samples = []
    wav_files = sorted(audio_dir.glob('*.wav'))

    print(f"Found {len(wav_files)} WAV files")

    for wav_file in wav_files:
        info = parse_filename(wav_file.name)
        if info is None:
            continue

        samples.append({
            'audio': str(wav_file),
            'text': info['sentence'],
            'emotion': info['emotion'],
            'language': 'en',
            'actor_id': info['actor_id'],
            'sentence_code': info['sentence_code']
        })

    print(f"Processed {len(samples)} valid samples")

    # Split into train/validation (90/10)
    import random
    random.seed(42)
    random.shuffle(samples)

    split_idx = int(len(samples) * 0.9)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    # Create datasets
    train_ds = Dataset.from_dict({
        'audio': [s['audio'] for s in train_samples],
        'text': [s['text'] for s in train_samples],
        'emotion': [s['emotion'] for s in train_samples],
        'language': [s['language'] for s in train_samples],
    })

    val_ds = Dataset.from_dict({
        'audio': [s['audio'] for s in val_samples],
        'text': [s['text'] for s in val_samples],
        'emotion': [s['emotion'] for s in val_samples],
        'language': [s['language'] for s in val_samples],
    })

    # Cast audio column to Audio feature
    train_ds = train_ds.cast_column('audio', Audio(sampling_rate=16000))
    val_ds = val_ds.cast_column('audio', Audio(sampling_rate=16000))

    # Create DatasetDict
    ds_dict = DatasetDict({
        'train': train_ds,
        'validation': val_ds
    })

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    ds_dict.save_to_disk(str(output_dir))

    print(f"\nSaved to {output_dir}")
    print("Emotion distribution in train:")

    from collections import Counter
    train_emotions = Counter(s['emotion'] for s in train_samples)
    for emotion, count in sorted(train_emotions.items()):
        print(f"  {emotion}: {count} ({100*count/len(train_samples):.1f}%)")


if __name__ == '__main__':
    create_dataset()
