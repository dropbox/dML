#!/usr/bin/env python3
"""Prepare CREMA-D dataset for RichDecoder v2 training."""
import json
from pathlib import Path
from collections import Counter

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
SENTENCE_MAP = {
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
        'sentence': SENTENCE_MAP.get(sentence_code, ''),
        'emotion': EMOTION_MAP[emotion_code],
        'emotion_code': emotion_code,
        'level': level  # XX = unspecified, LO = low, MD = medium, HI = high
    }

def main():
    audio_dir = Path('/Users/ayates/model_mlx_migration/data/emotion/crema-d_audio/AudioWAV')
    output_dir = Path('/Users/ayates/model_mlx_migration/data/emotion/crema-d_audio')

    samples = []
    emotion_counts = Counter()

    for wav_file in sorted(audio_dir.glob('*.wav')):
        info = parse_filename(wav_file.name)
        if info is None:
            print(f"Skipping: {wav_file.name}")
            continue

        samples.append({
            'audio_path': str(wav_file),
            'text': info['sentence'],
            'emotion': info['emotion'],
            'actor_id': info['actor_id'],
            'sentence_code': info['sentence_code']
        })
        emotion_counts[info['emotion']] += 1

    # Write manifest
    manifest_path = output_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump({
            'dataset': 'CREMA-D',
            'license': 'ODbL (Open Database License) - Commercial OK',
            'source': 'https://github.com/CheyneyComputerScience/CREMA-D',
            'total_samples': len(samples),
            'emotions': list(EMOTION_MAP.values()),
            'emotion_counts': dict(emotion_counts),
            'samples': samples
        }, f, indent=2)

    print(f"Created manifest: {manifest_path}")
    print(f"Total samples: {len(samples)}")
    print("Emotion distribution:")
    for emotion, count in sorted(emotion_counts.items()):
        print(f"  {emotion}: {count} ({100*count/len(samples):.1f}%)")

if __name__ == '__main__':
    main()
