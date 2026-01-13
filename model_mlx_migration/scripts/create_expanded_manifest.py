#!/usr/bin/env python3
"""
Create expanded manifest v4 with all available labeled data.

Sources:
1. Current v3_multitask manifest: ~22K samples
2. Prosody with emotion annotations: ~66K samples
3. Combined emotion HF: ~7.6K samples
4. Additional paralinguistics with known labels

Total target: 80K+ samples
"""

import json
import os
from pathlib import Path
from collections import Counter
import re

DATA_ROOT = Path("/Users/ayates/model_mlx_migration/data")
OUTPUT_DIR = DATA_ROOT / "v4_expanded"
OUTPUT_DIR.mkdir(exist_ok=True)

# Emotion mapping (from v3)
EMOTION_MAP = {
    'neutral': 'neutral',
    'happy': 'happy', 'happiness': 'happy', 'joy': 'happy',
    'sad': 'sad', 'sadness': 'sad',
    'angry': 'angry', 'anger': 'angry',
    'fear': 'fear', 'fearful': 'fear',
    'disgust': 'disgust',
    'surprise': 'surprise', 'surprised': 'surprise',
    'contempt': 'other',
    'other': 'other',
    # From prosody annotated_text
    'excited': 'happy',
    'frustrated': 'angry',
}

# Paralinguistic mapping
PARA_MAP = {
    'speech': 'speech',
    'laugh': 'laugh', 'laughter': 'laugh',
    'cough': 'cough',
    'sigh': 'sigh',
    'breath': 'breath',
    'cry': 'cry',
    'yawn': 'yawn',
    'throat_clear': 'throat_clear', 'throatclearing': 'throat_clear',
    'sneeze': 'sneeze',
    'gasp': 'gasp',
    'groan': 'groan',
    'snore': 'snore',
    'filler': 'filler',
    'silence': 'silence',
    'clapping': 'clapping',
    'sniff': 'sniff',
}

all_samples = []

def extract_emotion_from_annotation(text):
    """Extract emotion from annotated_text like <emotion type='surprise'>"""
    match = re.search(r"<emotion type='(\w+)'", text)
    if match:
        emotion = match.group(1).lower()
        return EMOTION_MAP.get(emotion, 'neutral')
    return 'neutral'

# 1. Load existing v3 manifest
print("Loading v3_multitask manifest...")
with open(DATA_ROOT / "v3_multitask/train_manifest.json") as f:
    v3_train = json.load(f)
print(f"  v3 train: {len(v3_train)} samples")

for s in v3_train:
    all_samples.append({
        'audio_path': s['audio_path'],
        'text': s.get('text', ''),
        'emotion': s.get('emotion', 'neutral'),
        'para': s.get('para', 'speech'),
        'language': s.get('language', 'en'),
        'source': s.get('source', 'v3'),
    })

# 2. Load prosody data with emotion annotations
print("Loading prosody data...")
prosody_path = DATA_ROOT / "prosody/train.json"
if prosody_path.exists():
    with open(prosody_path) as f:
        prosody_data = json.load(f)

    # Build audio path mapping for prosody
    # Format: "consolidated:JVNV:0" -> need to find actual audio
    prosody_audio_dirs = {
        'jvnv': DATA_ROOT / "prosody/jvnv",
        'crema-d': DATA_ROOT / "prosody/crema-d",
        'ravdess': DATA_ROOT / "prosody/ravdess",
        'esd': DATA_ROOT / "prosody/esd",
        'resd': DATA_ROOT / "prosody/resd",
    }

    added = 0
    for s in prosody_data:
        audio_ref = s.get('audio_path', '')
        annotated = s.get('annotated_text', '')

        # Extract emotion from annotation
        emotion = extract_emotion_from_annotation(annotated)

        # Try to resolve audio path
        # Format: "consolidated:JVNV:0" or direct path
        if ':' in audio_ref:
            parts = audio_ref.split(':')
            if len(parts) >= 3:
                dataset = parts[1].lower()
                idx = parts[2]
                # Try to find audio file
                if dataset in prosody_audio_dirs:
                    audio_dir = prosody_audio_dirs[dataset]
                    # List files and match by index
                    # This is a heuristic - actual mapping might differ
                    continue  # Skip consolidated refs for now
        elif os.path.isabs(audio_ref) and os.path.exists(audio_ref):
            all_samples.append({
                'audio_path': audio_ref,
                'text': s.get('text', ''),
                'emotion': emotion,
                'para': 'speech',
                'language': 'en',
                'source': s.get('source', 'prosody'),
            })
            added += 1

    print(f"  prosody with resolvable paths: {added}")

# 3. Load HF emotion dataset
print("Loading combined_emotion_hf...")
try:
    import datasets
    hf_path = DATA_ROOT / "emotion/combined_emotion_hf"
    if hf_path.exists():
        ds = datasets.load_from_disk(str(hf_path))

        # Save audio to disk and add to manifest
        hf_audio_dir = DATA_ROOT / "emotion/combined_emotion_audio"
        hf_audio_dir.mkdir(exist_ok=True)

        for i, sample in enumerate(ds['train']):
            # Get audio array
            audio = sample['audio']
            emotion = EMOTION_MAP.get(sample.get('emotion', 'neutral').lower(), 'neutral')

            # Save audio file
            audio_path = hf_audio_dir / f"hf_emotion_{i:05d}.wav"
            if not audio_path.exists():
                import soundfile as sf
                sf.write(str(audio_path), audio['array'], audio['sampling_rate'])

            all_samples.append({
                'audio_path': str(audio_path),
                'text': sample.get('text', ''),
                'emotion': emotion,
                'para': 'speech',
                'language': sample.get('language', 'en'),
                'source': 'combined_emotion_hf',
            })

        print(f"  combined_emotion_hf: {len(ds['train'])} samples")
except Exception as e:
    print(f"  HF emotion load failed: {e}")

# 4. Add prosody WAV files - extract emotion from directory/filename
print("Scanning prosody WAV files...")

# ESD - emotion in parent directory name (Happy, Sad, Angry, Neutral, Surprise)
# IMPORTANT: Speakers 0001-0010 are English, 0011-0020 are Chinese!
esd_dir = DATA_ROOT / "prosody/esd"
if esd_dir.exists():
    esd_count = 0
    esd_zh_count = 0
    for wav in esd_dir.rglob("*.wav"):
        # Extract emotion from parent dir
        parent = wav.parent.name.lower()
        emotion = EMOTION_MAP.get(parent, 'neutral')

        # Extract speaker ID to determine language
        # Path like: esd/0015/Happy/0015_000123.wav
        language = 'en'
        for part in wav.parts:
            if len(part) == 4 and part.isdigit():
                speaker_id = int(part)
                if speaker_id >= 11:  # 0011-0020 are Chinese speakers
                    language = 'zh'
                    esd_zh_count += 1
                break

        all_samples.append({
            'audio_path': str(wav),
            'text': '',
            'emotion': emotion,
            'para': 'speech',
            'language': language,
            'source': 'prosody-esd',
        })
        esd_count += 1
    print(f"  prosody/esd: {esd_count} files ({esd_zh_count} Chinese, {esd_count - esd_zh_count} English)")

# RAVDESS - emotion encoded in filename (3rd number: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised)
RAVDESS_EMOTION = {'01': 'neutral', '02': 'neutral', '03': 'happy', '04': 'sad', '05': 'angry', '06': 'fear', '07': 'disgust', '08': 'surprise'}
ravdess_dir = DATA_ROOT / "prosody/ravdess"
if ravdess_dir.exists():
    ravdess_count = 0
    for wav in ravdess_dir.rglob("*.wav"):
        # Filename: 03-01-05-01-01-01-01.wav (emotion is 3rd number)
        parts = wav.stem.split('-')
        if len(parts) >= 3:
            emotion = RAVDESS_EMOTION.get(parts[2], 'neutral')
        else:
            emotion = 'neutral'
        all_samples.append({
            'audio_path': str(wav),
            'text': '',
            'emotion': emotion,
            'para': 'speech',
            'language': 'en',
            'source': 'prosody-ravdess',
        })
        ravdess_count += 1
    print(f"  prosody/ravdess: {ravdess_count} files")

# CREMA-D in prosody/crema-d/AudioWAV - SKIP to avoid duplicates with emotion/crema-d
# The v3 manifest already includes CREMA-D from data/emotion/crema-d
cremad_dir = DATA_ROOT / "prosody/crema-d/AudioWAV"
if cremad_dir.exists():
    cremad_count = len(list(cremad_dir.glob("*.wav")))
    print(f"  prosody/crema-d: SKIPPED {cremad_count} files (duplicates with emotion/crema-d)")

# JVNV - Japanese emotions
jvnv_dir = DATA_ROOT / "prosody/jvnv"
if jvnv_dir.exists():
    jvnv_count = 0
    for wav in jvnv_dir.rglob("*.wav"):
        # Try to extract emotion from path/filename
        path_str = str(wav).lower()
        emotion = 'neutral'
        for emo in ['angry', 'happy', 'sad', 'fear', 'disgust', 'surprise']:
            if emo in path_str:
                emotion = emo
                break
        all_samples.append({
            'audio_path': str(wav),
            'text': '',
            'emotion': emotion,
            'para': 'speech',
            'language': 'ja',
            'source': 'prosody-jvnv',
        })
        jvnv_count += 1
    print(f"  prosody/jvnv: {jvnv_count} files")

# 5. Add VocalSound (21,024 samples with 6 classes)
print("\nAdding VocalSound...")
vocalsound_dir = DATA_ROOT / "paralinguistics/vocalsound_labeled/audio_16k"
if vocalsound_dir.exists():
    VOCALSOUND_LABEL_MAP = {
        'laughter': 'laugh', 'sigh': 'sigh', 'cough': 'cough',
        'throatclearing': 'throat_clear', 'sneeze': 'sneeze', 'sniff': 'sniff',
    }
    VOCALSOUND_EMOTION = {
        'laugh': 'happy', 'sigh': 'sad', 'cough': 'neutral',
        'throat_clear': 'neutral', 'sneeze': 'neutral', 'sniff': 'neutral',
    }
    vs_count = 0
    for wav in vocalsound_dir.glob("*.wav"):
        parts = wav.stem.split('_')
        if len(parts) >= 3:
            raw_label = parts[-1].lower()
            para = VOCALSOUND_LABEL_MAP.get(raw_label, raw_label)
            emotion = VOCALSOUND_EMOTION.get(para, 'neutral')
            all_samples.append({
                'audio_path': str(wav),
                'text': '',
                'emotion': emotion,
                'para': para,
                'language': 'en',
                'source': 'vocalsound',
            })
            vs_count += 1
    print(f"  vocalsound: {vs_count} samples")

# 6. Add paralinguistic datasets with known labels
print("\nAdding paralinguistic datasets...")

# LaughterScape - all laugh samples (8,170)
laughterscape_dir = DATA_ROOT / "paralinguistics/laughterscape"
if laughterscape_dir.exists():
    laugh_count = 0
    for wav in laughterscape_dir.glob("*.wav"):
        all_samples.append({
            'audio_path': str(wav),
            'text': '',
            'emotion': 'happy',  # Laughing = happy emotion
            'para': 'laugh',
            'language': 'en',
            'source': 'laughterscape',
        })
        laugh_count += 1
    print(f"  laughterscape: {laugh_count} laugh samples")

# CoughVID - all cough samples (972)
coughvid_dir = DATA_ROOT / "paralinguistics/coughvid"
if coughvid_dir.exists():
    cough_count = 0
    for wav in coughvid_dir.glob("*.wav"):
        all_samples.append({
            'audio_path': str(wav),
            'text': '',
            'emotion': 'neutral',
            'para': 'cough',
            'language': 'en',
            'source': 'coughvid',
        })
        cough_count += 1
    print(f"  coughvid: {cough_count} cough samples")

# Fillers - filler words (2,051)
fillers_dir = DATA_ROOT / "paralinguistics/fillers"
if fillers_dir.exists():
    filler_count = 0
    for wav in fillers_dir.glob("*.wav"):
        all_samples.append({
            'audio_path': str(wav),
            'text': '',
            'emotion': 'neutral',
            'para': 'filler',
            'language': 'en',
            'source': 'fillers',
        })
        filler_count += 1
    print(f"  fillers: {filler_count} filler samples")

# Silence - non-speech segments (2,000)
silence_dir = DATA_ROOT / "paralinguistics/silence"
if silence_dir.exists():
    silence_count = 0
    for wav in silence_dir.glob("*.wav"):
        all_samples.append({
            'audio_path': str(wav),
            'text': '',
            'emotion': 'neutral',
            'para': 'silence',
            'language': 'en',
            'source': 'silence',
        })
        silence_count += 1
    print(f"  silence: {silence_count} silence samples")

# POST-PROCESSING FIXES
print("\nApplying post-processing fixes...")

# Fix 1: Convert relative paths to absolute
rel_to_abs = 0
for s in all_samples:
    if not os.path.isabs(s['audio_path']):
        s['audio_path'] = str(DATA_ROOT / s['audio_path'])
        rel_to_abs += 1
print(f"  Fixed {rel_to_abs} relative paths -> absolute")

# Fix 2: Fix para/emotion mismatches
cry_fixed = 0
laugh_fixed = 0
for s in all_samples:
    # Cry should be sad, not neutral
    if s.get('para') == 'cry' and s.get('emotion') == 'neutral':
        s['emotion'] = 'sad'
        cry_fixed += 1
    # Laugh labeled neutral should be happy
    if s.get('para') == 'laugh' and s.get('emotion') == 'neutral':
        s['emotion'] = 'happy'
        laugh_fixed += 1
print(f"  Fixed {cry_fixed} cry->sad, {laugh_fixed} laugh->happy")

# DEDUPLICATION - remove samples with same audio file (by basename)
print("\nDeduplicating by audio filename...")
seen_basenames = set()
unique_samples = []
duplicates_removed = 0
for s in all_samples:
    basename = os.path.basename(s['audio_path'])
    if basename not in seen_basenames:
        seen_basenames.add(basename)
        unique_samples.append(s)
    else:
        duplicates_removed += 1

all_samples = unique_samples
print(f"  Removed {duplicates_removed} duplicates")

# Summary stats
print(f"\nTotal samples (after dedup): {len(all_samples)}")
sources = Counter(s['source'] for s in all_samples)
emotions = Counter(s['emotion'] for s in all_samples)
print(f"Sources: {dict(sources)}")
print(f"Emotions: {dict(emotions)}")

# Split train/val (90/10)
import random
random.seed(42)
random.shuffle(all_samples)
split_idx = int(len(all_samples) * 0.9)
train_samples = all_samples[:split_idx]
val_samples = all_samples[split_idx:]

# Save manifests
print(f"\nSaving to {OUTPUT_DIR}...")
with open(OUTPUT_DIR / "train_manifest.json", 'w') as f:
    json.dump(train_samples, f, indent=2)
with open(OUTPUT_DIR / "val_manifest.json", 'w') as f:
    json.dump(val_samples, f, indent=2)

metadata = {
    'total_samples': len(all_samples),
    'train_samples': len(train_samples),
    'val_samples': len(val_samples),
    'sources': dict(sources),
    'emotions': dict(emotions),
}
with open(OUTPUT_DIR / "metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"Done! Train: {len(train_samples)}, Val: {len(val_samples)}")
