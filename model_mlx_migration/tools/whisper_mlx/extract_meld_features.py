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
Extract Whisper encoder features from MELD dataset.

MELD (Multimodal EmotionLines Dataset) contains ~14K utterances from Friends TV show
with emotion labels and punctuated transcripts - ideal for training ProsodyConditionedCTC.

Output format (per utterance):
    {
        "encoder_features": (T, 1280),  # Whisper large-v3 encoder output
        "transcript": str,              # With punctuation (from CSV)
        "emotion": str,                 # Emotion label
        "sentiment": str,               # Sentiment label
        "dialogue_id": int,
        "utterance_id": int,
        "start_time": float,
        "end_time": float,
    }

Usage:
    python -m tools.whisper_mlx.extract_meld_features \
        --meld-dir data/emotion_punctuation/MELD.Raw \
        --output-dir data/meld_features \
        --model-size large-v3

References:
    - PHASE2_PROSODY_CTC_PLAN.md
    - MELD paper: https://arxiv.org/abs/1810.02508
"""

import argparse
import csv
import gc
import sys
import time
from pathlib import Path

import numpy as np

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

# Audio loading
try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


def load_audio(audio_path: str, target_sr: int = 16000) -> np.ndarray:
    """Load audio file and resample to target sample rate."""
    if HAS_SOUNDFILE:
        try:
            audio, sr = sf.read(audio_path)
            if sr != target_sr:
                if HAS_LIBROSA:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                else:
                    raise ValueError(f"Need librosa for resampling: {sr} -> {target_sr}")
            return audio.astype(np.float32)
        except Exception:
            pass

    if HAS_LIBROSA:
        audio, _ = librosa.load(audio_path, sr=target_sr)
        return audio.astype(np.float32)

    raise ImportError("Need soundfile or librosa to load audio")


def load_whisper_model(model_name: str):
    """Load Whisper model for encoder inference."""
    # Map short names to HuggingFace repo names
    model_map = {
        "large-v3": "mlx-community/whisper-large-v3-mlx",
        "large-v2": "mlx-community/whisper-large-v2-mlx",
        "medium": "mlx-community/whisper-medium-mlx",
        "small": "mlx-community/whisper-small-mlx",
        "base": "mlx-community/whisper-base-mlx",
        "tiny": "mlx-community/whisper-tiny-mlx",
    }

    repo_name = model_map.get(model_name, model_name)

    try:
        from mlx_whisper import load_models
        return load_models.load_model(repo_name)
    except ImportError:
        from tools.whisper_mlx.model import WhisperMLX
        return WhisperMLX.from_pretrained(model_name)


def get_encoder_features(model, audio: np.ndarray) -> np.ndarray:
    """Run audio through Whisper encoder to get features.

    For mlx_whisper models:
    1. Pad/trim audio to 30s (480000 samples)
    2. Compute mel spectrogram with correct n_mels
    3. Run encoder to get (1500, 1280) features

    Args:
        model: Loaded Whisper model
        audio: Audio waveform as numpy array (16kHz mono)

    Returns:
        Encoder features as numpy array (T, 1280) where T depends on audio length
    """
    # mlx_whisper uses embed_audio method with mel spectrogram input
    if hasattr(model, 'embed_audio') and hasattr(model, 'dims'):
        try:
            from mlx_whisper import audio as mlx_audio
        except ImportError:
            raise ImportError("Need mlx_whisper for audio processing") from None

        # Convert to mx.array for pad_or_trim
        audio_mx = mx.array(audio)

        # Pad or trim to 30 seconds (480000 samples at 16kHz)
        # This is required for the positional embedding to match
        audio_padded = mlx_audio.pad_or_trim(audio_mx)

        # Compute mel spectrogram with model's n_mels (128 for large-v3)
        n_mels = model.dims.n_mels
        mel = mlx_audio.log_mel_spectrogram(np.array(audio_padded), n_mels=n_mels)

        # Add batch dimension: (T, n_mels) -> (1, T, n_mels)
        mel = mel[None, :]

        # Run encoder
        encoder_output = model.embed_audio(mel)
        mx.eval(encoder_output)

        # Output is (batch, T, d_model) = (1, 1500, 1280)
        if encoder_output.ndim == 3:
            encoder_output = encoder_output[0]  # Remove batch

        return np.array(encoder_output)

    # Fallback: manual mel computation and encoder
    if hasattr(model, 'preprocess'):
        mel = model.preprocess(audio)
    elif hasattr(model, 'compute_features'):
        mel = model.compute_features(mx.array(audio))
    else:
        from tools.whisper_mlx.audio import log_mel_spectrogram
        mel = log_mel_spectrogram(audio)

    if isinstance(mel, np.ndarray):
        mel = mx.array(mel)

    if mel.ndim == 2:
        mel = mel[None, :]  # Add batch

    if hasattr(model, 'encode'):
        encoder_output = model.encode(mel)
    elif hasattr(model, 'encoder'):
        encoder_output = model.encoder(mel)
    else:
        raise ValueError("Cannot find encoder method on model")

    mx.eval(encoder_output)

    if encoder_output.ndim == 3:
        encoder_output = encoder_output[0]

    return np.array(encoder_output)


def parse_meld_csv(csv_path: str) -> list[dict]:
    """Parse MELD CSV file to get utterance metadata."""
    samples = []
    with open(csv_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample = {
                'sr_no': int(row['Sr No.']),
                'utterance': row['Utterance'],  # Has punctuation!
                'speaker': row['Speaker'],
                'emotion': row['Emotion'],
                'sentiment': row['Sentiment'],
                'dialogue_id': int(row['Dialogue_ID']),
                'utterance_id': int(row['Utterance_ID']),
                'season': int(row['Season']),
                'episode': int(row['Episode']),
                'start_time': row['StartTime'],
                'end_time': row['EndTime'],
            }
            samples.append(sample)
    return samples


def find_audio_file(audio_dir: Path, dialogue_id: int, utterance_id: int) -> Path | None:
    """Find audio file for a MELD utterance."""
    # MELD uses format: dia{dialogue_id}_utt{utterance_id}.wav
    filename = f"dia{dialogue_id}_utt{utterance_id}.wav"
    audio_path = audio_dir / filename

    if audio_path.exists():
        return audio_path

    # Try alternative naming
    filename_alt = f"{dialogue_id}_{utterance_id}.wav"
    audio_path_alt = audio_dir / filename_alt
    if audio_path_alt.exists():
        return audio_path_alt

    return None


def extract_features_for_split(
    model,
    csv_path: Path,
    audio_dir: Path,
    output_dir: Path,
    max_samples: int | None = None,
) -> tuple[int, int]:
    """Extract features for one split (train/dev/test)."""
    samples = parse_meld_csv(str(csv_path))

    if max_samples:
        samples = samples[:max_samples]

    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0

    for i, sample in enumerate(samples):
        # Find audio file
        audio_path = find_audio_file(audio_dir, sample['dialogue_id'], sample['utterance_id'])

        if audio_path is None:
            skipped += 1
            continue

        # Output path
        output_name = f"dia{sample['dialogue_id']}_utt{sample['utterance_id']}.npz"
        output_path = output_dir / output_name

        # Skip if already exists
        if output_path.exists():
            processed += 1
            continue

        try:
            # Load audio
            audio = load_audio(str(audio_path))

            # Get encoder features
            encoder_features = get_encoder_features(model, audio)

            # Save features
            np.savez_compressed(
                output_path,
                encoder_features=encoder_features,
                transcript=sample['utterance'],
                emotion=sample['emotion'],
                sentiment=sample['sentiment'],
                dialogue_id=sample['dialogue_id'],
                utterance_id=sample['utterance_id'],
                speaker=sample['speaker'],
                start_time=sample['start_time'],
                end_time=sample['end_time'],
            )

            processed += 1

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(samples)} ({processed} saved, {skipped} skipped)")

        except Exception as e:
            print(f"  Error processing {audio_path}: {e}")
            skipped += 1
            continue

    return processed, skipped


def main():
    parser = argparse.ArgumentParser(description="Extract MELD encoder features")
    parser.add_argument("--meld-dir", type=str, default="data/emotion_punctuation/MELD.Raw",
                       help="Path to MELD.Raw directory")
    parser.add_argument("--output-dir", type=str, default="data/meld_features",
                       help="Output directory for features")
    parser.add_argument("--model-size", type=str, default="large-v3",
                       help="Whisper model size")
    parser.add_argument("--split", type=str, default=None,
                       help="Process only this split (train/dev/test)")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum samples per split (for testing)")
    args = parser.parse_args()

    if not HAS_MLX:
        print("ERROR: MLX not available")
        sys.exit(1)

    meld_dir = Path(args.meld_dir)
    output_dir = Path(args.output_dir)

    if not meld_dir.exists():
        print(f"ERROR: MELD directory not found: {meld_dir}")
        sys.exit(1)

    print(f"Loading Whisper {args.model_size}...")
    model = load_whisper_model(args.model_size)
    print("Model loaded.")

    splits = {
        'train': ('train_sent_emo.csv', 'audio_train'),
        'dev': ('dev_sent_emo.csv', 'audio_dev'),
        'test': ('test_sent_emo.csv', 'audio_test'),
    }

    if args.split:
        if args.split not in splits:
            print(f"ERROR: Unknown split: {args.split}")
            sys.exit(1)
        splits = {args.split: splits[args.split]}

    total_processed = 0
    total_skipped = 0

    for split_name, (csv_name, audio_subdir) in splits.items():
        print(f"\nProcessing {split_name}...")

        csv_path = meld_dir / csv_name
        audio_dir = meld_dir / audio_subdir
        split_output = output_dir / split_name

        if not csv_path.exists():
            print(f"  CSV not found: {csv_path}")
            continue

        if not audio_dir.exists():
            print(f"  Audio dir not found: {audio_dir}")
            continue

        start_time = time.time()
        processed, skipped = extract_features_for_split(
            model, csv_path, audio_dir, split_output,
            max_samples=args.max_samples,
        )
        elapsed = time.time() - start_time

        print(f"  {split_name}: {processed} processed, {skipped} skipped ({elapsed:.1f}s)")

        total_processed += processed
        total_skipped += skipped

        # Clear cache between splits
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        gc.collect()

    print(f"\nTotal: {total_processed} processed, {total_skipped} skipped")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
