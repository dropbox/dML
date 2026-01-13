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
Organize real emotional speech samples into training directory.
Uses ONLY high-quality audio sources:
- RAVDESS: English (American actors, professional quality)
- ESD: Chinese (professional studio recordings)
- JVNV: Japanese (high quality emotional speech)
- RESD: Russian (emotional speech dataset)

Target: 1000+ samples per emotion, multi-language
"""

import os
import random
import shutil
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np

# RAVDESS emotion code mapping (filename format: XX-XX-EMOTION-XX-XX-XX-ACTOR.wav)
# Position 3 is emotion: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
RAVDESS_EMOTIONS = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
}

# ESD uses Chinese for all speakers
ESD_EMOTIONS = {
    "Angry": "angry",
    "Sad": "sad",
    "Happy": "happy",
    "Neutral": "neutral",
}

def save_wav(audio_array, path: str, sample_rate: int = 16000):
    """Save numpy audio array as WAV file."""
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
    """Extract audio array and sample rate from datasets AudioDecoder."""
    try:
        samples = audio_decoder.get_all_samples()
        data = samples.data.numpy().flatten()
        sr = samples.sample_rate
        return data, sr
    except Exception:
        return None, None

def load_ravdess_samples(base_dir: Path) -> dict:
    """Load RAVDESS (English, high quality professional actors)."""
    ravdess_path = base_dir / "ravdess"
    samples_by_emotion = defaultdict(list)

    if not ravdess_path.exists():
        print("  RAVDESS path not found!")
        return samples_by_emotion

    # Each actor folder contains speech files
    for actor_dir in ravdess_path.glob("Actor_*"):
        for wav_file in actor_dir.glob("*.wav"):
            # Parse filename: modality-vocChan-emotion-intensity-statement-repetition-actor.wav
            parts = wav_file.stem.split("-")
            if len(parts) >= 3:
                emotion_code = parts[2]
                emotion = RAVDESS_EMOTIONS.get(emotion_code)
                if emotion:
                    samples_by_emotion[emotion].append({
                        "path": wav_file,
                        "source": "ravdess",
                        "lang": "en",
                    })

    return samples_by_emotion

def load_esd_chinese_samples(base_dir: Path) -> dict:
    """Load ESD samples (Chinese only - all 20 speakers are Chinese)."""
    esd_path = base_dir / "esd"
    samples_by_emotion = defaultdict(list)

    # All ESD speakers (0001-0020) are Chinese
    for speaker_id in range(1, 21):
        speaker_dir = esd_path / f"{speaker_id:04d}"
        if not speaker_dir.exists():
            continue

        for emotion_folder, emotion in ESD_EMOTIONS.items():
            emotion_dir = speaker_dir / emotion_folder
            if not emotion_dir.exists():
                continue

            for wav_file in emotion_dir.glob("*.wav"):
                samples_by_emotion[emotion].append({
                    "path": wav_file,
                    "source": "esd",
                    "lang": "zh",
                })

    return samples_by_emotion

def load_jvnv_samples(base_dir: Path) -> dict:
    """Load Japanese JVNV dataset from Arrow format."""
    from datasets import load_from_disk

    jvnv_path = base_dir / "jvnv"
    if not (jvnv_path / "test").exists():
        return {}

    print("Loading JVNV (Japanese) dataset...")
    samples_by_emotion = defaultdict(list)
    jvnv_styles = {"anger": "angry", "sad": "sad", "happy": "happy"}

    try:
        dataset = load_from_disk(str(jvnv_path / "test"))
        print(f"  JVNV samples: {len(dataset)}")

        for i, item in enumerate(dataset):
            style = item.get("style", "").lower()
            audio = item.get("audio")
            emotion = jvnv_styles.get(style)
            if emotion and audio is not None:
                samples_by_emotion[emotion].append({
                    "audio": audio,
                    "lang": "ja",
                    "idx": i,
                })
    except Exception as e:
        print(f"  Error loading JVNV: {e}")

    return samples_by_emotion

def load_resd_samples(base_dir: Path) -> dict:
    """Load Russian RESD dataset from Arrow format."""
    from datasets import load_from_disk

    resd_path = base_dir / "resd"
    samples_by_emotion = defaultdict(list)
    resd_emotions = {
        "anger": "angry",
        "sadness": "sad",
        "happiness": "happy",
        "enthusiasm": "happy",
        "neutral": "neutral",
    }

    for split in ["train", "test"]:
        split_path = resd_path / split
        if not split_path.exists():
            continue

        print(f"Loading RESD (Russian) {split} dataset...")
        try:
            dataset = load_from_disk(str(split_path))
            print(f"  RESD {split} samples: {len(dataset)}")

            for i, item in enumerate(dataset):
                emotion_key = item.get("emotion", "").lower()
                audio = item.get("speech")
                emotion = resd_emotions.get(emotion_key)
                if emotion and audio is not None:
                    samples_by_emotion[emotion].append({
                        "audio": audio,
                        "lang": "ru",
                        "idx": i,
                        "split": split,
                    })
        except Exception as e:
            print(f"  Error loading RESD {split}: {e}")

    return samples_by_emotion

def main():
    base_dir = Path("/Users/ayates/model_mlx_migration/data/prosody")
    output_dir = Path("/Users/ayates/voice/emotion_training_samples")

    organized = defaultdict(lambda: defaultdict(list))

    # Load RAVDESS (English - professional American actors)
    print("Loading RAVDESS (English, High Quality)...")
    ravdess_samples = load_ravdess_samples(base_dir)
    for emotion, samples_list in ravdess_samples.items():
        print(f"  RAVDESS {emotion} en: {len(samples_list)}")
        organized[emotion]["en"].extend(samples_list)

    # Load ESD (Chinese)
    print("Loading ESD (Chinese, High Quality)...")
    esd_samples = load_esd_chinese_samples(base_dir)
    for emotion, samples_list in esd_samples.items():
        print(f"  ESD {emotion} zh: {len(samples_list)}")
        organized[emotion]["zh"].extend(samples_list)

    # Load JVNV (Japanese)
    jvnv_samples = load_jvnv_samples(base_dir)
    for emotion, samples_list in jvnv_samples.items():
        print(f"  JVNV {emotion} ja: {len(samples_list)}")
        organized[emotion]["ja"].extend(samples_list)

    # Load RESD (Russian)
    resd_samples = load_resd_samples(base_dir)
    for emotion, samples_list in resd_samples.items():
        print(f"  RESD {emotion} ru: {len(samples_list)}")
        organized[emotion]["ru"].extend(samples_list)

    # Print statistics
    print("\n" + "=" * 60)
    print("AVAILABLE HIGH-QUALITY SAMPLES")
    print("=" * 60)

    for emotion in ["neutral", "angry", "sad", "happy"]:
        print(f"\n{emotion.upper()}:")
        total = 0
        for lang in sorted(organized[emotion].keys()):
            count = len(organized[emotion][lang])
            total += count
            print(f"  {lang}: {count}")
        print(f"  TOTAL: {total}")

    # Targets - use all available English since RAVDESS is smaller
    TARGET_ENGLISH = 500  # Use all RAVDESS samples
    TARGET_CHINESE = 500
    TARGET_JAPANESE = 250
    TARGET_RUSSIAN = 200

    print("\n" + "=" * 60)
    print("COPYING HIGH-QUALITY FILES")
    print("=" * 60)

    # Clear output directory
    for emotion in ["neutral", "angry", "sad", "happy"]:
        emotion_dir = output_dir / emotion
        emotion_dir.mkdir(parents=True, exist_ok=True)
        for f in emotion_dir.glob("*.wav"):
            f.unlink()

    for emotion in ["neutral", "angry", "sad", "happy"]:
        emotion_dir = output_dir / emotion
        emotion_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{emotion.upper()}:")
        copied = {}

        # Copy English samples (from RAVDESS)
        en_samples = organized[emotion].get("en", [])
        random.shuffle(en_samples)
        en_samples = en_samples[:TARGET_ENGLISH]

        for i, sample in enumerate(en_samples):
            if "path" in sample:
                src = sample["path"]
                dst = emotion_dir / f"en_{i+1:04d}.wav"
                try:
                    shutil.copy2(src, dst)
                    copied["en"] = copied.get("en", 0) + 1
                except Exception:
                    pass

        # Copy Chinese samples (from ESD)
        zh_samples = organized[emotion].get("zh", [])
        random.shuffle(zh_samples)
        zh_samples = zh_samples[:TARGET_CHINESE]

        for i, sample in enumerate(zh_samples):
            if "path" in sample:
                src = sample["path"]
                dst = emotion_dir / f"zh_{i+1:04d}.wav"
                try:
                    shutil.copy2(src, dst)
                    copied["zh"] = copied.get("zh", 0) + 1
                except Exception:
                    pass

        # Copy Japanese samples (from JVNV)
        ja_samples = organized[emotion].get("ja", [])
        random.shuffle(ja_samples)
        ja_samples = ja_samples[:TARGET_JAPANESE]

        for i, sample in enumerate(ja_samples):
            if "audio" in sample:
                try:
                    audio_decoder = sample["audio"]
                    audio_array, sr = extract_audio_from_decoder(audio_decoder)
                    if audio_array is not None and len(audio_array) > 0:
                        dst = emotion_dir / f"ja_{i+1:04d}.wav"
                        save_wav(audio_array, str(dst), sr)
                        copied["ja"] = copied.get("ja", 0) + 1
                except Exception:
                    pass

        # Copy Russian samples (from RESD)
        ru_samples = organized[emotion].get("ru", [])
        random.shuffle(ru_samples)
        ru_samples = ru_samples[:TARGET_RUSSIAN]

        for i, sample in enumerate(ru_samples):
            if "audio" in sample:
                try:
                    audio_decoder = sample["audio"]
                    audio_array, sr = extract_audio_from_decoder(audio_decoder)
                    if audio_array is not None and len(audio_array) > 0:
                        dst = emotion_dir / f"ru_{i+1:04d}.wav"
                        save_wav(audio_array, str(dst), sr)
                        copied["ru"] = copied.get("ru", 0) + 1
                except Exception:
                    pass

        for lang, count in sorted(copied.items()):
            print(f"  {lang}: {count} files")
        print(f"  TOTAL: {sum(copied.values())} files")

    print("\n" + "=" * 60)
    print(f"Done! High-quality files saved to: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
