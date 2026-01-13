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
Create silence dataset by extracting silent segments from LibriSpeech.

Uses energy-based silence detection to find low-energy regions.
"""

import numpy as np
from pathlib import Path
import soundfile as sf
from typing import List, Tuple
import random


def find_silent_regions(audio: np.ndarray, sr: int = 16000,
                        frame_length: int = 512, hop_length: int = 256,
                        percentile: float = 10, min_duration: float = 0.3) -> List[Tuple[int, int]]:
    """Find silent regions in audio using RMS energy.

    Args:
        audio: Audio waveform
        sr: Sample rate
        frame_length: Frame length for RMS calculation
        hop_length: Hop length for RMS calculation
        percentile: Energy percentile threshold for silence
        min_duration: Minimum silence duration in seconds

    Returns:
        List of (start_sample, end_sample) tuples for silent regions
    """
    # Calculate RMS energy
    num_frames = 1 + (len(audio) - frame_length) // hop_length
    rms = np.zeros(num_frames)

    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        frame = audio[start:end]
        rms[i] = np.sqrt(np.mean(frame**2))

    # Find threshold
    threshold = np.percentile(rms, percentile)

    # Find silent frames
    silent_frames = rms < threshold

    # Find contiguous silent regions
    regions = []
    in_silence = False
    start_frame = 0

    for i, is_silent in enumerate(silent_frames):
        if is_silent and not in_silence:
            start_frame = i
            in_silence = True
        elif not is_silent and in_silence:
            # Convert frames to samples
            start_sample = start_frame * hop_length
            end_sample = i * hop_length
            duration = (end_sample - start_sample) / sr

            if duration >= min_duration:
                regions.append((start_sample, end_sample))

            in_silence = False

    # Handle trailing silence
    if in_silence:
        start_sample = start_frame * hop_length
        end_sample = len(audio)
        duration = (end_sample - start_sample) / sr
        if duration >= min_duration:
            regions.append((start_sample, end_sample))

    return regions


def extract_silence_segments(audio_dir: str, output_dir: str,
                            max_samples: int = 2000,
                            segment_duration: float = 1.0,
                            sr: int = 16000):
    """Extract silence segments from audio files.

    Args:
        audio_dir: Directory containing audio files (recursively searched)
        output_dir: Directory to save silence samples
        max_samples: Maximum number of silence samples to extract
        segment_duration: Duration of each silence sample in seconds
        sr: Target sample rate
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    audio_files = list(Path(audio_dir).rglob("*.flac"))
    if not audio_files:
        audio_files = list(Path(audio_dir).rglob("*.wav"))

    print(f"Found {len(audio_files)} audio files in {audio_dir}")

    if not audio_files:
        print("No audio files found!")
        return

    # Shuffle to get diverse sources
    random.shuffle(audio_files)

    samples_extracted = 0
    segment_samples = int(segment_duration * sr)

    for audio_file in audio_files:
        if samples_extracted >= max_samples:
            break

        try:
            # Load audio
            audio, file_sr = sf.read(audio_file)

            # Resample if needed
            if file_sr != sr:
                # Simple resampling via interpolation
                duration = len(audio) / file_sr
                new_length = int(duration * sr)
                audio = np.interp(
                    np.linspace(0, len(audio) - 1, new_length),
                    np.arange(len(audio)),
                    audio
                )

            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            # Find silent regions
            regions = find_silent_regions(audio, sr=sr)

            # Extract segments from silent regions
            for start, end in regions:
                if samples_extracted >= max_samples:
                    break

                # Extract fixed-length segment from the middle of the region
                region_length = end - start
                if region_length >= segment_samples:
                    # Take from middle of silence region
                    offset = (region_length - segment_samples) // 2
                    segment = audio[start + offset : start + offset + segment_samples]

                    # Verify it's actually quiet (double-check)
                    segment_rms = np.sqrt(np.mean(segment**2))
                    if segment_rms < 0.01:  # Very quiet
                        output_file = output_path / f"silence_{samples_extracted:05d}.wav"
                        sf.write(output_file, segment, sr)
                        samples_extracted += 1

                        if samples_extracted % 100 == 0:
                            print(f"  Extracted {samples_extracted} silence samples")

        except Exception as e:
            print(f"  Error processing {audio_file}: {e}")
            continue

    print(f"\nTotal silence samples extracted: {samples_extracted}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract silence segments from audio")
    parser.add_argument("--audio-dir", default="data/LibriSpeech_full/train-clean-100",
                       help="Directory containing audio files")
    parser.add_argument("--output-dir", default="data/paralinguistics/silence",
                       help="Output directory for silence samples")
    parser.add_argument("--max-samples", type=int, default=2000,
                       help="Maximum number of samples to extract")
    parser.add_argument("--segment-duration", type=float, default=1.0,
                       help="Duration of each silence segment in seconds")

    args = parser.parse_args()

    extract_silence_segments(
        args.audio_dir,
        args.output_dir,
        max_samples=args.max_samples,
        segment_duration=args.segment_duration
    )
