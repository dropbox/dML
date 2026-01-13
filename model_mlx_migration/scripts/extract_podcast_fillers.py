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
Extract filler word samples from PodcastFillers dataset.

The ylacombe/podcast_fillers dataset contains full podcast episodes (~30-60 min each)
without timestamp annotations. This script:

1. Loads each podcast episode from the HuggingFace dataset
2. Runs Whisper with word-level timestamps
3. Identifies filler words (um, uh, er, ah, hmm, like, you know)
4. Extracts short audio clips (0.3-1.5s) around each filler
5. Saves labeled samples for paralinguistics training

Output: data/paralinguistics/podcast_fillers_extracted/
    - {filler_type}_{episode_idx}_{filler_idx}.wav
    - manifest.json with metadata

Usage:
    python scripts/extract_podcast_fillers.py --max-episodes 10
    python scripts/extract_podcast_fillers.py --all  # Process all 173 episodes

Estimated yield: ~50-200 fillers per episode = 8,000-35,000 total samples
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Filler words to extract (lowercase)
FILLER_WORDS = {
    # Primary fillers
    "um": "um_en",
    "uh": "uh_en",
    "er": "er_en",
    "ah": "ah_en",
    "hmm": "hmm_en",
    "hm": "hmm_en",
    "mm": "hmm_en",
    "mhm": "hmm_en",
    # Secondary fillers (discourse markers)
    "like": "like_en",
    "so": "so_en",
    # Hesitation sounds
    "erm": "er_en",
    "ehm": "um_en",
}

# Minimum confidence for word detection
MIN_WORD_CONFIDENCE = 0.5

# Audio extraction parameters
PADDING_BEFORE = 0.1  # seconds before filler
PADDING_AFTER = 0.15  # seconds after filler
MIN_DURATION = 0.2    # minimum clip duration
MAX_DURATION = 2.0    # maximum clip duration

# Validation thresholds
MIN_SPEECH_RATIO = 0.3   # Extracted clip must have at least 30% speech
MIN_ENERGY_THRESHOLD = 0.001  # Minimum RMS energy (reject silence)


def load_podcast_fillers_dataset(data_dir: str = "data/paralinguistics/podcast_fillers"):
    """Load the PodcastFillers dataset from disk."""
    from datasets import load_from_disk, Audio

    print(f"Loading PodcastFillers from: {data_dir}")
    ds = load_from_disk(data_dir)

    # Cast to Audio with 16kHz sampling (uses torchcodec for decoding)
    if hasattr(ds, 'cast_column'):
        ds = ds.cast_column("audio", Audio(sampling_rate=16000, decode=True))

    print(f"  Loaded {len(ds)} episodes")
    return ds


def get_audio_array(audio_data) -> Tuple[np.ndarray, int]:
    """
    Extract audio array from various formats (torchcodec, dict, etc.)

    Returns: (audio_array, sample_rate)
    """
    import torch

    # Handle torchcodec AudioDecoder
    if hasattr(audio_data, 'get_all_samples'):
        samples = audio_data.get_all_samples()
        # samples.data is torch.Tensor with shape [channels, samples]
        audio_tensor = samples.data
        sample_rate = samples.sample_rate

        # Convert to numpy and mono
        if isinstance(audio_tensor, torch.Tensor):
            audio_np = audio_tensor.numpy()
        else:
            audio_np = np.array(audio_tensor)

        # Convert stereo to mono by averaging channels
        if len(audio_np.shape) > 1 and audio_np.shape[0] == 2:
            audio_np = audio_np.mean(axis=0)
        elif len(audio_np.shape) > 1:
            audio_np = audio_np[0]  # Take first channel

        return audio_np.astype(np.float32), sample_rate

    # Handle dict format
    elif isinstance(audio_data, dict):
        if 'array' in audio_data:
            audio = np.array(audio_data['array'], dtype=np.float32)
            sample_rate = audio_data.get('sampling_rate', 16000)
            return audio, sample_rate
        elif 'bytes' in audio_data:
            # Load from bytes using soundfile or pydub
            import soundfile as sf
            import io
            try:
                audio, sample_rate = sf.read(io.BytesIO(audio_data['bytes']))
                return audio.astype(np.float32), sample_rate
            except:
                from pydub import AudioSegment
                audio_seg = AudioSegment.from_file(io.BytesIO(audio_data['bytes']))
                samples = np.array(audio_seg.get_array_of_samples(), dtype=np.float32) / 32768.0
                if audio_seg.channels == 2:
                    samples = samples.reshape(-1, 2).mean(axis=1)
                return samples, audio_seg.frame_rate

    # Handle raw numpy array
    elif isinstance(audio_data, np.ndarray):
        return audio_data.astype(np.float32), 16000

    raise ValueError(f"Unknown audio format: {type(audio_data)}")


def validate_audio_clip(
    audio: np.ndarray,
    sample_rate: int = 16000,
    vad_processor=None,
) -> Tuple[bool, str, float]:
    """
    Validate that an extracted audio clip contains actual speech.

    Returns: (is_valid, reason, speech_ratio)
    """
    # Check minimum energy (reject silence/very quiet clips)
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < MIN_ENERGY_THRESHOLD:
        return False, "too_quiet", 0.0

    # Check for clipping/distortion (reject if > 10% samples are clipped)
    clipped_ratio = np.mean(np.abs(audio) > 0.99)
    if clipped_ratio > 0.1:
        return False, "clipped", 0.0

    # Use VAD to check speech content
    if vad_processor is not None:
        try:
            vad_result = vad_processor.process(audio, sample_rate)
            speech_ratio = vad_result.speech_ratio
            if speech_ratio < MIN_SPEECH_RATIO:
                return False, f"low_speech_ratio_{speech_ratio:.2f}", speech_ratio
            return True, "valid", speech_ratio
        except Exception:
            # VAD failed, fall back to energy-only validation
            pass

    return True, "valid_no_vad", 0.5


def get_vad_processor():
    """Get or create VAD processor (lazy loading)."""
    try:
        from tools.whisper_mlx.silero_vad import SileroVADProcessor
        return SileroVADProcessor(aggressiveness=2)
    except ImportError:
        print("WARNING: SileroVAD not available, skipping speech validation")
        return None


def transcribe_with_timestamps(
    audio: np.ndarray,
    sample_rate: int = 16000,
    model_size: str = "large-v3"
) -> List[Dict]:
    """
    Transcribe audio using Whisper with word-level timestamps.

    Returns list of word dicts with: word, start, end, confidence
    """
    try:
        import mlx_whisper
    except ImportError:
        print("ERROR: mlx-whisper required. Install: pip install mlx-whisper")
        return []

    # Transcribe with word timestamps
    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=f"mlx-community/whisper-{model_size}-mlx",
        word_timestamps=True,
    )

    # Extract words from segments
    words = []
    for segment in result.get("segments", []):
        for word_info in segment.get("words", []):
            words.append({
                "word": word_info.get("word", "").strip().lower(),
                "start": word_info.get("start", 0),
                "end": word_info.get("end", 0),
                "confidence": word_info.get("probability", 1.0),
            })

    return words


def find_fillers(words: List[Dict]) -> List[Dict]:
    """
    Find filler words in transcription.

    Returns list of filler dicts with: word, label, start, end, confidence
    """
    fillers = []

    for word_info in words:
        word = word_info["word"].strip().lower()
        # Remove punctuation
        word_clean = "".join(c for c in word if c.isalnum())

        if word_clean in FILLER_WORDS:
            if word_info.get("confidence", 1.0) >= MIN_WORD_CONFIDENCE:
                fillers.append({
                    "word": word_clean,
                    "label": FILLER_WORDS[word_clean],
                    "start": word_info["start"],
                    "end": word_info["end"],
                    "confidence": word_info.get("confidence", 1.0),
                })

    return fillers


def extract_audio_clip(
    audio: np.ndarray,
    start: float,
    end: float,
    sample_rate: int = 16000,
    padding_before: float = PADDING_BEFORE,
    padding_after: float = PADDING_AFTER,
) -> Tuple[np.ndarray, float, float]:
    """
    Extract audio clip around a timestamp with padding.

    Returns: (audio_clip, actual_start, actual_end)
    """
    # Add padding
    clip_start = max(0, start - padding_before)
    clip_end = min(len(audio) / sample_rate, end + padding_after)

    # Ensure minimum duration
    duration = clip_end - clip_start
    if duration < MIN_DURATION:
        # Center the filler in a MIN_DURATION window
        center = (start + end) / 2
        clip_start = max(0, center - MIN_DURATION / 2)
        clip_end = min(len(audio) / sample_rate, center + MIN_DURATION / 2)

    # Ensure maximum duration
    if clip_end - clip_start > MAX_DURATION:
        center = (start + end) / 2
        clip_start = center - MAX_DURATION / 2
        clip_end = center + MAX_DURATION / 2

    # Convert to samples
    start_sample = int(clip_start * sample_rate)
    end_sample = int(clip_end * sample_rate)

    return audio[start_sample:end_sample], clip_start, clip_end


def save_audio_clip(audio: np.ndarray, path: Path, sample_rate: int = 16000):
    """Save audio clip as WAV file."""
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sample_rate)


def process_episode(
    episode_idx: int,
    audio: np.ndarray,
    sample_rate: int,
    output_dir: Path,
    model_size: str = "large-v3",
    vad_processor=None,
) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Process a single podcast episode.

    Returns: (extracted_metadata_list, rejection_stats)
    """
    print(f"\n  Episode {episode_idx}: {len(audio)/sample_rate:.1f}s audio")

    # Transcribe with word timestamps
    print("    Transcribing...")
    start_time = time.time()
    words = transcribe_with_timestamps(audio, sample_rate, model_size)
    transcribe_time = time.time() - start_time
    print(f"    Transcription: {len(words)} words in {transcribe_time:.1f}s")

    # Find fillers
    fillers = find_fillers(words)
    print(f"    Found {len(fillers)} filler candidates")

    # Extract, validate, and save clips
    extracted = []
    rejection_stats = {"too_quiet": 0, "clipped": 0, "low_speech": 0, "accepted": 0}

    for filler_idx, filler in enumerate(fillers):
        clip, clip_start, clip_end = extract_audio_clip(
            audio, filler["start"], filler["end"], sample_rate
        )

        # Validate the clip
        is_valid, reason, speech_ratio = validate_audio_clip(
            clip, sample_rate, vad_processor
        )

        if not is_valid:
            if "too_quiet" in reason:
                rejection_stats["too_quiet"] += 1
            elif "clipped" in reason:
                rejection_stats["clipped"] += 1
            elif "low_speech" in reason:
                rejection_stats["low_speech"] += 1
            continue

        rejection_stats["accepted"] += 1

        # Generate filename
        filename = f"{filler['label']}_{episode_idx:04d}_{filler_idx:04d}.wav"
        clip_path = output_dir / filename

        # Save clip
        save_audio_clip(clip, clip_path, sample_rate)

        # Record metadata
        extracted.append({
            "filename": filename,
            "label": filler["label"],
            "word": filler["word"],
            "episode_idx": episode_idx,
            "filler_idx": filler_idx,
            "original_start": filler["start"],
            "original_end": filler["end"],
            "clip_start": clip_start,
            "clip_end": clip_end,
            "duration": clip_end - clip_start,
            "confidence": filler["confidence"],
            "speech_ratio": speech_ratio,
        })

    # Print rejection stats
    total_candidates = len(fillers)
    if total_candidates > 0:
        print(f"    Validation: {rejection_stats['accepted']}/{total_candidates} accepted")
        if rejection_stats['too_quiet'] > 0:
            print(f"      - {rejection_stats['too_quiet']} rejected (too quiet)")
        if rejection_stats['low_speech'] > 0:
            print(f"      - {rejection_stats['low_speech']} rejected (low speech ratio)")
        if rejection_stats['clipped'] > 0:
            print(f"      - {rejection_stats['clipped']} rejected (clipped)")

    return extracted, rejection_stats


def main():
    parser = argparse.ArgumentParser(description="Extract fillers from PodcastFillers dataset")
    parser.add_argument("--data-dir", default="data/paralinguistics/podcast_fillers",
                        help="Path to PodcastFillers dataset")
    parser.add_argument("--output-dir", default="data/paralinguistics/podcast_fillers_extracted",
                        help="Output directory for extracted clips")
    parser.add_argument("--max-episodes", type=int, default=10,
                        help="Maximum episodes to process (default: 10)")
    parser.add_argument("--all", action="store_true",
                        help="Process all episodes")
    parser.add_argument("--model-size", default="large-v3",
                        help="Whisper model size (default: large-v3)")
    parser.add_argument("--start-episode", type=int, default=0,
                        help="Starting episode index (for resuming)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    ds = load_podcast_fillers_dataset(args.data_dir)

    # Determine episodes to process
    if args.all:
        num_episodes = len(ds)
    else:
        num_episodes = min(args.max_episodes, len(ds))

    print(f"\nProcessing {num_episodes} episodes (starting from {args.start_episode})")
    print(f"Output: {output_dir}")

    # Load existing manifest if resuming
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            all_metadata = json.load(f)
        print(f"  Loaded existing manifest with {len(all_metadata)} samples")
    else:
        all_metadata = []

    # Initialize VAD processor for clip validation
    print("  Initializing VAD processor...")
    vad_processor = get_vad_processor()

    # Process episodes
    total_fillers = 0
    total_rejection_stats = {"too_quiet": 0, "clipped": 0, "low_speech": 0, "accepted": 0}
    start_time = time.time()

    for episode_idx in range(args.start_episode, args.start_episode + num_episodes):
        if episode_idx >= len(ds):
            break

        try:
            # Get audio
            episode = ds[episode_idx]
            audio_data = episode["audio"]

            # Extract audio array using format-agnostic function
            audio, sample_rate = get_audio_array(audio_data)

            # Process episode
            metadata, rejection_stats = process_episode(
                episode_idx, audio, sample_rate, output_dir, args.model_size, vad_processor
            )

            # Aggregate stats
            all_metadata.extend(metadata)
            total_fillers += len(metadata)
            for key in total_rejection_stats:
                total_rejection_stats[key] += rejection_stats.get(key, 0)

            # Save manifest after each episode
            with open(manifest_path, "w") as f:
                json.dump(all_metadata, f, indent=2)

            print(f"    Saved {len(metadata)} clips. Total: {total_fillers}")

        except Exception as e:
            print(f"    ERROR processing episode {episode_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Final summary
    elapsed = time.time() - start_time
    total_candidates = sum(total_rejection_stats.values())

    print(f"\n{'='*60}")
    print("Extraction Complete!")
    print(f"  Episodes processed: {num_episodes}")
    print(f"  Total fillers extracted: {total_fillers}")
    print(f"  Time: {elapsed/60:.1f} minutes")
    print(f"  Output: {output_dir}")
    print(f"  Manifest: {manifest_path}")

    # Print validation stats
    if total_candidates > 0:
        print("\nValidation Statistics:")
        print(f"  Total candidates: {total_candidates}")
        print(f"  Accepted: {total_rejection_stats['accepted']} ({100*total_rejection_stats['accepted']/total_candidates:.1f}%)")
        print(f"  Rejected - too quiet: {total_rejection_stats['too_quiet']}")
        print(f"  Rejected - low speech: {total_rejection_stats['low_speech']}")
        print(f"  Rejected - clipped: {total_rejection_stats['clipped']}")

    # Print label distribution
    label_counts = {}
    for m in all_metadata:
        label = m["label"]
        label_counts[label] = label_counts.get(label, 0) + 1

    print("\nLabel distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
