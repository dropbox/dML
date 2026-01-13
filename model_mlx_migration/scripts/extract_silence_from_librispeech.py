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

"""Extract silence segments from LibriSpeech audio."""

from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

# Try multiple LibriSpeech locations
LIBRISPEECH_PATHS = [
    Path("data/LibriSpeech/dev-clean"),
    Path("data/LibriSpeech"),
    Path("data/LibriSpeech_full"),
]

OUTPUT_DIR = Path("data/paralinguistics/silence")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_SILENCE_SEC = 0.5
MAX_SILENCE_SEC = 3.0
TARGET_SAMPLES = 2000

def find_silence_regions(audio: np.ndarray, sr: int,
                         energy_threshold_percentile: float = 10,
                         min_frames: int = 20) -> list:
    """Find contiguous silence regions using RMS energy."""
    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)

    rms = librosa.feature.rms(y=audio, frame_length=frame_length,
                               hop_length=hop_length)[0]
    threshold = np.percentile(rms, energy_threshold_percentile)
    is_silent = rms < threshold

    regions = []
    start = None

    for i, silent in enumerate(is_silent):
        if silent and start is None:
            start = i
        elif not silent and start is not None:
            if i - start >= min_frames:
                start_sample = start * hop_length
                end_sample = min(i * hop_length, len(audio))
                regions.append((start_sample, end_sample))
            start = None

    if start is not None and len(is_silent) - start >= min_frames:
        start_sample = start * hop_length
        end_sample = len(audio)
        regions.append((start_sample, end_sample))

    return regions

def extract_silence():
    print("Extracting silence from LibriSpeech...")

    # Find audio files
    audio_files = []
    for libri_path in LIBRISPEECH_PATHS:
        if libri_path.exists():
            audio_files.extend(list(libri_path.rglob("*.flac")))
            audio_files.extend(list(libri_path.rglob("*.wav")))

    if not audio_files:
        print("ERROR: No LibriSpeech audio found!")
        return

    print(f"Found {len(audio_files)} audio files")

    saved = 0
    min_samples = int(MIN_SILENCE_SEC * 16000)
    max_samples = int(MAX_SILENCE_SEC * 16000)

    for audio_path in tqdm(audio_files, desc="Processing"):
        if saved >= TARGET_SAMPLES:
            break

        try:
            audio, sr = librosa.load(str(audio_path), sr=16000)
            regions = find_silence_regions(audio, sr)

            for start, end in regions:
                duration = end - start
                if duration < min_samples:
                    continue
                if duration > max_samples:
                    end = start + max_samples

                silence_audio = audio[start:end]
                output_path = OUTPUT_DIR / f"silence_{saved:05d}.wav"
                sf.write(str(output_path), silence_audio, sr)
                saved += 1

                if saved >= TARGET_SAMPLES:
                    break

        except Exception:
            continue

    print(f"\nExtracted {saved} silence samples to {OUTPUT_DIR}")

if __name__ == "__main__":
    extract_silence()
