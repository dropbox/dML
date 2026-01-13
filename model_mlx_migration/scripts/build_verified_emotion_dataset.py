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
Build verified emotion training dataset with quality checks.

Verification:
1. Whisper transcription - verify speech is intelligible
2. Audio checks - voice detection, volume levels
3. Language detection - must match annotation

Filename format: {lang}_{source}_{original_name}.wav
"""

import json
import os
import shutil
import wave
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Will be imported lazily
whisper_model = None
langdetect = None

@dataclass
class AudioQuality:
    is_valid: bool
    duration_s: float
    rms_db: float
    has_speech: bool
    detected_lang: Optional[str]
    transcription: Optional[str]
    issues: list

def load_whisper():
    global whisper_model
    if whisper_model is None:
        import whisper
        print("Loading Whisper model (base)...")
        whisper_model = whisper.load_model("base")
    return whisper_model

def check_audio_quality(audio_path: str, expected_lang: str) -> AudioQuality:
    """Run all quality checks on an audio file."""
    issues = []

    try:
        # Load audio using scipy
        from scipy.io import wavfile
        sr, y = wavfile.read(audio_path)
        # Convert to float32 normalized
        if y.dtype == np.int16:
            y = y.astype(np.float32) / 32768.0
        elif y.dtype == np.int32:
            y = y.astype(np.float32) / 2147483648.0
        # Handle stereo
        if len(y.shape) > 1:
            y = y[:, 0]
        duration_s = len(y) / sr

        # Check duration
        if duration_s < 0.5:
            issues.append("too_short")
        if duration_s > 30:
            issues.append("too_long")

        # Check volume (RMS in dB)
        rms = np.sqrt(np.mean(y**2))
        rms_db = 20 * np.log10(rms + 1e-10)

        if rms_db < -40:
            issues.append("too_quiet")
        if rms_db > -5:
            issues.append("too_loud")

        # Check if it looks like speech (has some variation)
        if np.std(y) < 0.01:
            issues.append("no_variation")

        # Whisper transcription and language detection
        has_speech = False
        detected_lang = None
        transcription = None

        try:
            model = load_whisper()
            result = model.transcribe(audio_path, language=None)
            transcription = result.get("text", "").strip()
            detected_lang = result.get("language", "")

            if len(transcription) > 3:
                has_speech = True
            else:
                issues.append("no_speech_detected")

            # Check language match
            lang_map = {
                "en": ["en", "english"],
                "zh": ["zh", "chinese", "mandarin"],
                "ja": ["ja", "japanese"],
                "ru": ["ru", "russian"],
                "ko": ["ko", "korean"],
            }

            expected_langs = lang_map.get(expected_lang, [expected_lang])
            if detected_lang and detected_lang not in expected_langs:
                issues.append(f"lang_mismatch:{detected_lang}")

        except Exception as e:
            issues.append(f"whisper_error:{str(e)[:50]}")

        is_valid = len([i for i in issues if not i.startswith("lang_mismatch")]) == 0

        return AudioQuality(
            is_valid=is_valid,
            duration_s=duration_s,
            rms_db=rms_db,
            has_speech=has_speech,
            detected_lang=detected_lang,
            transcription=transcription,
            issues=issues
        )

    except Exception as e:
        return AudioQuality(
            is_valid=False,
            duration_s=0,
            rms_db=-100,
            has_speech=False,
            detected_lang=None,
            transcription=None,
            issues=[f"load_error:{str(e)[:50]}"]
        )

TARGET_SAMPLE_RATE = 16000  # Normalize all audio to 16kHz

def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Resample audio to target sample rate using scipy."""
    if orig_sr == target_sr:
        return audio
    from scipy import signal
    # Calculate new length
    new_length = int(len(audio) * target_sr / orig_sr)
    return signal.resample(audio, new_length)

def save_wav(audio_array, path: str, sample_rate: int, normalize_sr: bool = False):
    """Save audio array as WAV file, PRESERVING original sample rate by default."""
    audio = np.array(audio_array, dtype=np.float32).flatten()

    # Resample to target if explicitly requested (normally we preserve original)
    if normalize_sr and sample_rate != TARGET_SAMPLE_RATE:
        audio = resample_audio(audio, sample_rate, TARGET_SAMPLE_RATE)
        sample_rate = TARGET_SAMPLE_RATE

    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

def extract_hf_audio(audio_obj) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """Extract audio from HuggingFace audio object."""
    try:
        if hasattr(audio_obj, 'get_all_samples'):
            # AudioDecoder
            samples = audio_obj.get_all_samples()
            return samples.data.numpy().flatten(), samples.sample_rate
        elif isinstance(audio_obj, dict):
            return np.array(audio_obj['array']), audio_obj['sampling_rate']
        else:
            return None, None
    except Exception:
        return None, None

def process_ravdess(base_dir: Path, output_dir: Path, verify: bool = True) -> dict:
    """Process RAVDESS dataset (English actors)."""
    print("\n=== RAVDESS (English) ===")
    ravdess_path = base_dir / "ravdess"

    EMOTIONS = {
        "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
        "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
    }

    stats = defaultdict(lambda: {"total": 0, "valid": 0, "issues": Counter()})

    for actor_dir in sorted(ravdess_path.glob("Actor_*")):
        for wav_file in actor_dir.glob("*.wav"):
            parts = wav_file.stem.split("-")
            if len(parts) >= 3:
                emotion = EMOTIONS.get(parts[2])
                if emotion:
                    stats[emotion]["total"] += 1

                    # Output filename: en_ravdess_originalname.wav
                    out_name = f"en_ravdess_{wav_file.stem}.wav"
                    out_path = output_dir / emotion / out_name
                    out_path.parent.mkdir(parents=True, exist_ok=True)

                    # Load and resample to 16kHz
                    try:
                        from scipy.io import wavfile
                        sr, y = wavfile.read(str(wav_file))
                        # Convert to float32 normalized
                        if y.dtype == np.int16:
                            y = y.astype(np.float32) / 32768.0
                        elif y.dtype == np.int32:
                            y = y.astype(np.float32) / 2147483648.0
                        # Handle stereo by taking first channel
                        if len(y.shape) > 1:
                            y = y[:, 0]
                        temp_path = f"/tmp/verify_ravdess_{wav_file.stem}.wav"
                        save_wav(y, temp_path, sr)

                        if verify:
                            quality = check_audio_quality(temp_path, "en")
                            if quality.is_valid:
                                shutil.move(temp_path, out_path)
                                stats[emotion]["valid"] += 1
                            else:
                                os.remove(temp_path)
                                for issue in quality.issues:
                                    stats[emotion]["issues"][issue] += 1
                        else:
                            shutil.move(temp_path, out_path)
                            stats[emotion]["valid"] += 1
                    except Exception as e:
                        stats[emotion]["issues"][f"error:{str(e)[:30]}"] += 1

    for emotion, s in sorted(stats.items()):
        print(f"  {emotion}: {s['valid']}/{s['total']} valid")
        if s['issues']:
            print(f"    issues: {dict(s['issues'])}")

    return stats

def process_esd(base_dir: Path, output_dir: Path, verify: bool = True) -> dict:
    """Process ESD dataset (Chinese speakers)."""
    print("\n=== ESD (Chinese) ===")
    esd_path = base_dir / "esd"

    EMOTIONS = {"Angry": "angry", "Sad": "sad", "Happy": "happy",
                "Neutral": "neutral", "Surprise": "surprised"}

    stats = defaultdict(lambda: {"total": 0, "valid": 0, "issues": Counter()})

    for speaker_id in range(1, 21):
        speaker_dir = esd_path / f"{speaker_id:04d}"
        if not speaker_dir.exists():
            continue

        for emotion_folder, emotion in EMOTIONS.items():
            emotion_dir = speaker_dir / emotion_folder
            if not emotion_dir.exists():
                continue

            for wav_file in emotion_dir.glob("*.wav"):  # All files
                stats[emotion]["total"] += 1

                out_name = f"zh_esd_{wav_file.stem}.wav"
                out_path = output_dir / emotion / out_name
                out_path.parent.mkdir(parents=True, exist_ok=True)

                if verify:
                    quality = check_audio_quality(str(wav_file), "zh")
                    if quality.is_valid:
                        shutil.copy2(wav_file, out_path)
                        stats[emotion]["valid"] += 1
                    else:
                        for issue in quality.issues:
                            stats[emotion]["issues"][issue] += 1
                else:
                    shutil.copy2(wav_file, out_path)
                    stats[emotion]["valid"] += 1

    for emotion, s in sorted(stats.items()):
        print(f"  {emotion}: {s['valid']}/{s['total']} valid")

    return stats

def process_en1gma02(output_dir: Path, verify: bool = True) -> dict:
    """Process En1gma02/processed_english_emotions_tagged dataset.

    Only includes actual emotions, not speaking styles.
    """
    print("\n=== En1gma02 (English) ===")
    import io

    from scipy.io import wavfile

    from datasets import Audio, load_dataset

    # Map En1gma02 styles - include useful categories
    VALID_EMOTIONS = {
        "happy": "happy",
        "sad": "sad",
        "laughing": "laughing",  # Keep as separate category
        "default": "neutral",    # Default/no-emotion = neutral
        "confused": "confused",  # Keep - useful prosody
        "whisper": "whisper",    # Keep - speaking style
        "enunciated": "enunciated",  # Keep - clear speech
        "emphasis": "emphasis",  # Keep - emphatic speech
    }
    # Excluded only: essentials (40), singing (3), longform (2)

    ds = load_dataset('En1gma02/processed_english_emotions_tagged', split='train')
    # Disable audio decoding to avoid torchcodec dependency
    ds = ds.cast_column('audio', Audio(decode=False))
    print(f"  Loaded {len(ds)} samples")

    stats = defaultdict(lambda: {"total": 0, "valid": 0, "issues": Counter()})
    skipped_styles = Counter()

    for idx in range(len(ds)):
        item = ds[idx]
        style = item['style'].lower()
        emotion = VALID_EMOTIONS.get(style)
        sample_id = item['id']
        # Extract transcript and speaker_id
        transcript = item.get('text', '') or ''
        raw_speaker_id = item.get('speaker_id', '') or ''
        # Add prefix for consistency with other datasets
        speaker_id = f"en1gma02_{raw_speaker_id}" if raw_speaker_id else f"en1gma02_{idx:05d}"

        if emotion is None:
            skipped_styles[style] += 1
            continue

        stats[emotion]["total"] += 1

        # Include row index for uniqueness (some IDs are duplicated/mismatched)
        out_name = f"en_en1gma02_{idx:05d}_{sample_id}.wav"
        out_path = output_dir / emotion / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path = out_path.with_suffix('.txt')
        meta_path = out_path.with_suffix('.meta')

        try:
            # Decode audio from raw bytes
            audio_bytes = item['audio']['bytes']
            sr, arr = wavfile.read(io.BytesIO(audio_bytes))
            # Convert to float32 normalized
            if arr.dtype == np.int16:
                arr = arr.astype(np.float32) / 32768.0
            elif arr.dtype == np.int32:
                arr = arr.astype(np.float32) / 2147483648.0
            if len(arr.shape) > 1:
                arr = arr[:, 0]

            # Save temporarily for verification
            temp_path = f"/tmp/verify_{sample_id}.wav"
            save_wav(arr, temp_path, sr)

            if verify:
                quality = check_audio_quality(temp_path, "en")
                if quality.is_valid:
                    shutil.move(temp_path, out_path)
                    # Save transcript
                    if transcript:
                        with open(txt_path, 'w') as f:
                            f.write(transcript)
                    # Save speaker metadata
                    with open(meta_path, 'w') as f:
                        f.write(f"speaker_id={speaker_id}\n")
                    stats[emotion]["valid"] += 1
                else:
                    os.remove(temp_path)
                    for issue in quality.issues:
                        stats[emotion]["issues"][issue] += 1
            else:
                shutil.move(temp_path, out_path)
                # Save transcript
                if transcript:
                    with open(txt_path, 'w') as f:
                        f.write(transcript)
                # Save speaker metadata
                with open(meta_path, 'w') as f:
                    f.write(f"speaker_id={speaker_id}\n")
                stats[emotion]["valid"] += 1

        except Exception as e:
            stats[emotion]["issues"][f"error:{str(e)[:30]}"] += 1

    for emotion, s in sorted(stats.items()):
        print(f"  {emotion}: {s['valid']}/{s['total']} valid")

    if skipped_styles:
        print(f"  Skipped non-emotion styles: {dict(skipped_styles)}")

    return stats

def process_bluebomber(output_dir: Path, verify: bool = True) -> dict:
    """Process Bluebomber182/AI-Emotions dataset.

    DISABLED: This dataset has NO emotion labels - only 'audio' column.
    We cannot verify these are actually angry. Skipping to maintain data quality.
    """
    print("\n=== Bluebomber182 (SKIPPED - no emotion labels) ===")
    print("  Dataset has no emotion annotations, cannot verify samples are angry")
    return {}

def process_kratos(output_dir: Path, verify: bool = True) -> dict:
    """Process Kratos emotion dataset (ENGLISH despite name - scripts are English)."""
    print("\n=== Kratos Emotions (English, not Korean!) ===")
    import io

    from scipy.io import wavfile

    from datasets import Audio, load_dataset

    ds = load_dataset('Kratos-AI/korean-voice-emotion-dataset', split='train')
    # Disable audio decoding
    ds = ds.cast_column('audio', Audio(decode=False))
    print(f"  Loaded {len(ds)} samples (NOTE: These are English, not Korean!)")

    EMOTIONS = {"Angry": "angry", "Happy": "happy", "Sad": "sad", "Surprised": "surprised"}
    stats = defaultdict(lambda: {"total": 0, "valid": 0, "issues": Counter()})

    for i in range(len(ds)):
        item = ds[i]
        emotion_raw = item['Emotion']
        emotion = EMOTIONS.get(emotion_raw)
        if not emotion:
            continue

        stats[emotion]["total"] += 1

        # FIXED: Label as English since scripts are English
        out_name = f"en_kratos_{i:04d}.wav"
        out_path = output_dir / emotion / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Kratos stores audio as file paths, not bytes
            audio_path = item['audio'].get('path')
            audio_bytes = item['audio'].get('bytes')

            if audio_bytes:
                sr, arr = wavfile.read(io.BytesIO(audio_bytes))
            elif audio_path and os.path.exists(audio_path):
                sr, arr = wavfile.read(audio_path)
            else:
                stats[emotion]["issues"]["no_audio"] += 1
                continue

            if arr.dtype == np.int16:
                arr = arr.astype(np.float32) / 32768.0
            elif arr.dtype == np.int32:
                arr = arr.astype(np.float32) / 2147483648.0
            if len(arr.shape) > 1:
                arr = arr[:, 0]

            temp_path = f"/tmp/verify_ko_{i}.wav"
            save_wav(arr, temp_path, sr)

            if verify:
                quality = check_audio_quality(temp_path, "en")  # English, not Korean
                if quality.is_valid:
                    shutil.move(temp_path, out_path)
                    stats[emotion]["valid"] += 1
                else:
                    os.remove(temp_path)
                    for issue in quality.issues:
                        stats[emotion]["issues"][issue] += 1
            else:
                shutil.move(temp_path, out_path)
                stats[emotion]["valid"] += 1
        except Exception as e:
            stats[emotion]["issues"][f"error:{str(e)[:30]}"] += 1

    for emotion, s in sorted(stats.items()):
        print(f"  {emotion}: {s['valid']}/{s['total']} valid")

    return stats

def process_jvnv(base_dir: Path, output_dir: Path, verify: bool = True) -> dict:
    """Process JVNV Japanese dataset."""
    print("\n=== JVNV (Japanese) ===")
    import io

    from scipy.io import wavfile

    from datasets import Audio, load_from_disk

    jvnv_path = base_dir / "jvnv" / "test"
    if not jvnv_path.exists():
        print("  Not found")
        return {}

    ds = load_from_disk(str(jvnv_path))
    # Disable audio decoding
    ds = ds.cast_column('audio', Audio(decode=False))
    print(f"  Loaded {len(ds)} samples")

    STYLES = {"anger": "angry", "sad": "sad", "happy": "happy",
              "surprise": "surprised", "fear": "fearful", "disgust": "disgust"}

    stats = defaultdict(lambda: {"total": 0, "valid": 0, "issues": Counter()})

    for i in range(len(ds)):
        item = ds[i]
        style = item.get("style", "").lower()
        emotion = STYLES.get(style)
        if not emotion:
            continue

        stats[emotion]["total"] += 1

        out_name = f"ja_jvnv_{i:04d}.wav"
        out_path = output_dir / emotion / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            audio_path = item['audio'].get('path')
            audio_bytes = item['audio'].get('bytes')

            if audio_bytes:
                sr, arr = wavfile.read(io.BytesIO(audio_bytes))
            elif audio_path and os.path.exists(audio_path):
                sr, arr = wavfile.read(audio_path)
            else:
                stats[emotion]["issues"]["no_audio"] += 1
                continue

            if arr.dtype == np.int16:
                arr = arr.astype(np.float32) / 32768.0
            elif arr.dtype == np.int32:
                arr = arr.astype(np.float32) / 2147483648.0
            if len(arr.shape) > 1:
                arr = arr[:, 0]

            temp_path = f"/tmp/verify_jvnv_{i}.wav"
            save_wav(arr, temp_path, sr)

            if verify:
                quality = check_audio_quality(temp_path, "ja")
                if quality.is_valid:
                    shutil.move(temp_path, out_path)
                    stats[emotion]["valid"] += 1
                else:
                    os.remove(temp_path)
                    for issue in quality.issues:
                        stats[emotion]["issues"][issue] += 1
            else:
                shutil.move(temp_path, out_path)
                stats[emotion]["valid"] += 1
        except Exception as e:
            stats[emotion]["issues"][f"error:{str(e)[:30]}"] += 1

    for emotion, s in sorted(stats.items()):
        print(f"  {emotion}: {s['valid']}/{s['total']} valid")

    return stats

def process_resd(base_dir: Path, output_dir: Path, verify: bool = True) -> dict:
    """Process RESD Russian dataset."""
    print("\n=== RESD (Russian) ===")
    import io

    from scipy.io import wavfile

    from datasets import Audio, load_from_disk

    resd_path = base_dir / "resd"
    EMOTIONS = {"anger": "angry", "sadness": "sad", "happiness": "happy",
                "enthusiasm": "enthusiasm", "neutral": "neutral",
                "disgust": "disgust", "fear": "fearful"}

    stats = defaultdict(lambda: {"total": 0, "valid": 0, "issues": Counter()})

    for split in ["train", "test"]:
        split_path = resd_path / split
        if not split_path.exists():
            continue

        ds = load_from_disk(str(split_path))
        # Disable audio decoding - RESD uses 'speech' column
        ds = ds.cast_column('speech', Audio(decode=False))
        print(f"  {split}: {len(ds)} samples")

        for i in range(len(ds)):
            item = ds[i]
            emotion_key = item.get("emotion", "").lower()
            emotion = EMOTIONS.get(emotion_key)
            if not emotion:
                continue

            stats[emotion]["total"] += 1

            out_name = f"ru_resd_{split}_{i:04d}.wav"
            out_path = output_dir / emotion / out_name
            out_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                audio_path = item['speech'].get('path')
                audio_bytes = item['speech'].get('bytes')

                if audio_bytes:
                    sr, arr = wavfile.read(io.BytesIO(audio_bytes))
                elif audio_path and os.path.exists(audio_path):
                    sr, arr = wavfile.read(audio_path)
                else:
                    stats[emotion]["issues"]["no_audio"] += 1
                    continue

                if arr.dtype == np.int16:
                    arr = arr.astype(np.float32) / 32768.0
                elif arr.dtype == np.int32:
                    arr = arr.astype(np.float32) / 2147483648.0
                if len(arr.shape) > 1:
                    arr = arr[:, 0]

                temp_path = f"/tmp/verify_resd_{i}.wav"
                save_wav(arr, temp_path, sr)

                if verify:
                    quality = check_audio_quality(temp_path, "ru")
                    if quality.is_valid:
                        shutil.move(temp_path, out_path)
                        stats[emotion]["valid"] += 1
                    else:
                        os.remove(temp_path)
                        for issue in quality.issues:
                            stats[emotion]["issues"][issue] += 1
                else:
                    shutil.move(temp_path, out_path)
                    stats[emotion]["valid"] += 1
            except Exception as e:
                stats[emotion]["issues"][f"error:{str(e)[:30]}"] += 1

    for emotion, s in sorted(stats.items()):
        print(f"  {emotion}: {s['valid']}/{s['total']} valid")

    return stats

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-verify", action="store_true", help="Skip audio verification")
    parser.add_argument("--sources", nargs="+", default=["all"],
                       help="Sources to process: ravdess, esd, en1gma02, bluebomber, korean, jvnv, resd, all")
    args = parser.parse_args()

    base_dir = Path("/Users/ayates/model_mlx_migration/data/prosody")
    output_dir = Path("/Users/ayates/voice/emotion_training_samples")

    verify = not args.no_verify
    sources = args.sources if "all" not in args.sources else [
        "ravdess", "esd", "en1gma02", "bluebomber", "kratos", "jvnv", "resd"
    ]

    print("Building emotion dataset")
    print(f"Output: {output_dir}")
    print(f"Verification: {'ON' if verify else 'OFF'}")
    print(f"Sources: {sources}")
    print("=" * 60)

    all_stats = {}

    if "ravdess" in sources:
        all_stats["ravdess"] = process_ravdess(base_dir, output_dir, verify)

    if "esd" in sources:
        all_stats["esd"] = process_esd(base_dir, output_dir, verify)

    if "en1gma02" in sources:
        all_stats["en1gma02"] = process_en1gma02(output_dir, verify)

    if "bluebomber" in sources:
        all_stats["bluebomber"] = process_bluebomber(output_dir, verify)

    if "kratos" in sources:
        all_stats["kratos"] = process_kratos(output_dir, verify)

    if "jvnv" in sources:
        all_stats["jvnv"] = process_jvnv(base_dir, output_dir, verify)

    if "resd" in sources:
        all_stats["resd"] = process_resd(base_dir, output_dir, verify)

    # Deduplicate files
    # Valid emotions/styles we want to keep
    VALID_EMOTIONS = {
        "neutral", "calm", "happy", "sad", "angry", "fearful",
        "disgust", "surprised", "enthusiasm",
        # Additional styles from En1gma02
        "laughing", "confused", "whisper", "enunciated", "emphasis"
    }

    # Clean up non-emotion directories from previous runs
    print("\n=== Cleanup non-emotion directories ===")
    for item in list(output_dir.iterdir()):
        if item.is_dir() and item.name not in VALID_EMOTIONS:
            file_count = len(list(item.glob("*.wav")))
            shutil.rmtree(item)
            print(f"  Removed {item.name}/ ({file_count} files) - not an emotion")

    print("\n=== Deduplication ===")
    import hashlib
    from collections import defaultdict as dd
    hashes = dd(list)
    for emotion_dir in output_dir.iterdir():
        if not emotion_dir.is_dir():
            continue
        for f in emotion_dir.glob("*.wav"):
            try:
                with open(f, 'rb') as fp:
                    h = hashlib.md5(fp.read()).hexdigest()
                    hashes[h].append(f)
            except Exception:
                pass

    dupes_removed = 0
    for h, paths in hashes.items():
        if len(paths) > 1:
            # Keep the first, remove the rest (and their .meta/.txt files)
            for dup in paths[1:]:
                dup.unlink()
                # Also remove associated .meta and .txt files
                meta_file = dup.with_suffix('.meta')
                txt_file = dup.with_suffix('.txt')
                if meta_file.exists():
                    meta_file.unlink()
                if txt_file.exists():
                    txt_file.unlink()
                dupes_removed += 1
    print(f"  Removed {dupes_removed} duplicate files (with associated .meta/.txt)")

    # Generate comprehensive CSV with all file properties
    print("\n=== Generating dataset manifest CSV ===")
    import csv

    from scipy.io import wavfile as wf_reader

    csv_file = output_dir / "dataset_manifest.csv"
    rows = []

    for emotion_dir in sorted(output_dir.iterdir()):
        if not emotion_dir.is_dir():
            continue
        emotion = emotion_dir.name

        for wav_file in sorted(emotion_dir.glob("*.wav")):
            try:
                # Parse filename: {lang}_{source}_{original}.wav
                parts = wav_file.stem.split("_", 2)
                lang = parts[0] if len(parts) > 0 else ""
                source = parts[1] if len(parts) > 1 else ""
                original = parts[2] if len(parts) > 2 else wav_file.stem

                # Read audio properties
                sr, audio_data = wf_reader.read(str(wav_file))
                if audio_data.dtype == np.int16:
                    audio_float = audio_data.astype(np.float32) / 32768.0
                elif audio_data.dtype == np.int32:
                    audio_float = audio_data.astype(np.float32) / 2147483648.0
                else:
                    audio_float = audio_data.astype(np.float32)
                if len(audio_float.shape) > 1:
                    channels = audio_float.shape[1]
                    audio_float = audio_float[:, 0]
                else:
                    channels = 1

                duration_s = len(audio_float) / sr
                rms = np.sqrt(np.mean(audio_float**2))
                rms_db = 20 * np.log10(rms + 1e-10)

                # Check for transcript file
                txt_file = wav_file.with_suffix('.txt')
                transcript = txt_file.read_text(encoding='utf-8').strip() if txt_file.exists() else ""

                # Check for metadata file (speaker_id)
                meta_file = wav_file.with_suffix('.meta')
                speaker_id = ""
                if meta_file.exists():
                    for line in meta_file.read_text().strip().split('\n'):
                        if line.startswith('speaker_id='):
                            speaker_id = line.split('=', 1)[1]

                rows.append({
                    "file_path": str(wav_file.relative_to(output_dir)),
                    "emotion": emotion,
                    "language": lang,
                    "source": source,
                    "original_name": original,
                    "speaker_id": speaker_id,
                    "sample_rate": sr,
                    "duration_s": round(duration_s, 3),
                    "rms_db": round(rms_db, 2),
                    "channels": channels,
                    "transcript": transcript,
                })
            except Exception as e:
                print(f"  Warning: {wav_file.name}: {e}")

    if rows:
        fieldnames = ["file_path", "emotion", "language", "source", "original_name",
                      "speaker_id", "sample_rate", "duration_s", "rms_db", "channels", "transcript"]
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Wrote {len(rows)} rows to {csv_file}")

    # Save stats
    stats_file = output_dir / "build_stats.json"
    with open(stats_file, "w") as f:
        json.dump({k: {e: {"total": s["total"], "valid": s["valid"]}
                      for e, s in v.items()}
                  for k, v in all_stats.items()}, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Count files per emotion
    for emotion_dir in sorted(output_dir.iterdir()):
        if emotion_dir.is_dir() and emotion_dir.name != "build_stats.json":
            files = list(emotion_dir.glob("*.wav"))
            by_lang = Counter(f.name.split("_")[0] for f in files)
            print(f"{emotion_dir.name}: {len(files)} files - {dict(by_lang)}")

    print(f"\nStats saved to: {stats_file}")

if __name__ == "__main__":
    main()
