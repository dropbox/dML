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
Organize ALL emotions from all datasets for user review.
Creates union of all available emotions.
"""

import os
import random
import shutil
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np

# RAVDESS emotion codes
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

# ESD emotion folders
ESD_EMOTIONS = {
    "Angry": "angry",
    "Sad": "sad",
    "Happy": "happy",
    "Neutral": "neutral",
    "Surprise": "surprised",
}

# JVNV styles
JVNV_STYLES = {
    "anger": "angry",
    "sad": "sad",
    "happy": "happy",
    "surprise": "surprised",
    "fear": "fearful",
    "disgust": "disgust",
}

# RESD emotions
RESD_EMOTIONS = {
    "anger": "angry",
    "sadness": "sad",
    "happiness": "happy",
    "enthusiasm": "enthusiasm",
    "neutral": "neutral",
    "disgust": "disgust",
    "fear": "fearful",
}

def save_wav(audio_array, path: str, sample_rate: int = 16000):
    audio = np.array(audio_array, dtype=np.float32).flatten()
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

def extract_audio_from_decoder(audio_decoder):
    try:
        samples = audio_decoder.get_all_samples()
        data = samples.data.numpy().flatten()
        sr = samples.sample_rate
        return data, sr
    except Exception:
        return None, None

def load_ravdess_samples(base_dir: Path) -> dict:
    ravdess_path = base_dir / "ravdess"
    samples = defaultdict(list)

    for actor_dir in ravdess_path.glob("Actor_*"):
        for wav_file in actor_dir.glob("*.wav"):
            parts = wav_file.stem.split("-")
            if len(parts) >= 3:
                emotion = RAVDESS_EMOTIONS.get(parts[2])
                if emotion:
                    samples[emotion].append({"path": wav_file, "lang": "en"})
    return samples

def load_esd_samples(base_dir: Path) -> dict:
    esd_path = base_dir / "esd"
    samples = defaultdict(list)

    for speaker_id in range(1, 21):
        speaker_dir = esd_path / f"{speaker_id:04d}"
        if not speaker_dir.exists():
            continue
        for emotion_folder, emotion in ESD_EMOTIONS.items():
            emotion_dir = speaker_dir / emotion_folder
            if emotion_dir.exists():
                for wav_file in emotion_dir.glob("*.wav"):
                    samples[emotion].append({"path": wav_file, "lang": "zh"})
    return samples

def load_jvnv_samples(base_dir: Path) -> dict:
    from datasets import load_from_disk
    jvnv_path = base_dir / "jvnv" / "test"
    samples = defaultdict(list)

    if not jvnv_path.exists():
        return samples

    print("Loading JVNV...")
    dataset = load_from_disk(str(jvnv_path))
    for i, item in enumerate(dataset):
        style = item.get("style", "").lower()
        emotion = JVNV_STYLES.get(style)
        if emotion and item.get("audio"):
            samples[emotion].append({"audio": item["audio"], "lang": "ja", "idx": i})
    return samples

def load_resd_samples(base_dir: Path) -> dict:
    from datasets import load_from_disk
    resd_path = base_dir / "resd"
    samples = defaultdict(list)

    for split in ["train", "test"]:
        split_path = resd_path / split
        if not split_path.exists():
            continue
        print(f"Loading RESD {split}...")
        dataset = load_from_disk(str(split_path))
        for i, item in enumerate(dataset):
            emotion_key = item.get("emotion", "").lower()
            emotion = RESD_EMOTIONS.get(emotion_key)
            if emotion and item.get("speech"):
                samples[emotion].append({"audio": item["speech"], "lang": "ru", "idx": i})
    return samples

def main():
    base_dir = Path("/Users/ayates/model_mlx_migration/data/prosody")
    output_dir = Path("/Users/ayates/voice/emotion_training_samples")

    # Clear and recreate output directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    organized = defaultdict(lambda: defaultdict(list))

    # Load all datasets
    print("Loading RAVDESS (English)...")
    for emotion, samples in load_ravdess_samples(base_dir).items():
        print(f"  {emotion}: {len(samples)}")
        organized[emotion]["en"].extend(samples)

    print("\nLoading ESD (Chinese)...")
    for emotion, samples in load_esd_samples(base_dir).items():
        print(f"  {emotion}: {len(samples)}")
        organized[emotion]["zh"].extend(samples)

    print("\nLoading JVNV (Japanese)...")
    for emotion, samples in load_jvnv_samples(base_dir).items():
        print(f"  {emotion}: {len(samples)}")
        organized[emotion]["ja"].extend(samples)

    print("\nLoading RESD (Russian)...")
    for emotion, samples in load_resd_samples(base_dir).items():
        print(f"  {emotion}: {len(samples)}")
        organized[emotion]["ru"].extend(samples)

    # Get all unique emotions
    all_emotions = sorted(organized.keys())
    print(f"\n{'='*60}")
    print(f"ALL EMOTIONS: {all_emotions}")
    print(f"{'='*60}")

    # Copy samples - limit to 200 per language per emotion for review
    MAX_PER_LANG = 200

    for emotion in all_emotions:
        emotion_dir = output_dir / emotion
        emotion_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{emotion.upper()}:")
        copied = {}

        for lang in sorted(organized[emotion].keys()):
            samples = organized[emotion][lang]
            random.shuffle(samples)
            samples = samples[:MAX_PER_LANG]

            for i, sample in enumerate(samples):
                dst = emotion_dir / f"{lang}_{i+1:04d}.wav"
                try:
                    if "path" in sample:
                        shutil.copy2(sample["path"], dst)
                        copied[lang] = copied.get(lang, 0) + 1
                    elif "audio" in sample:
                        audio_array, sr = extract_audio_from_decoder(sample["audio"])
                        if audio_array is not None:
                            save_wav(audio_array, str(dst), sr)
                            copied[lang] = copied.get(lang, 0) + 1
                except Exception:
                    pass

        for lang, count in sorted(copied.items()):
            print(f"  {lang}: {count}")
        print(f"  TOTAL: {sum(copied.values())}")

    print(f"\n{'='*60}")
    print(f"Done! All emotions saved to: {output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
