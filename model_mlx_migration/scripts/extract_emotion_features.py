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
Extract emotion and pitch features from audio for punctuation training.

This script runs audio through frozen Whisper encoder, emotion head, and pitch head
to extract features that will be used for emotion-aware punctuation training.

Usage:
    python scripts/extract_emotion_features.py \
        --data-dir data/LibriSpeech \
        --output-dir data/emotion_punctuation/librispeech_features \
        --emotion-head checkpoints/emotion_unified_v2/best.npz \
        --pitch-head checkpoints/pitch_combined_v4/best.npz \
        --model-size large-v3

Output format (per audio file):
    {audio_id}.npz containing:
        - encoder_features: (T, 1280) Whisper encoder output
        - emotion_probs: (8,) emotion class probabilities
        - pitch_values: (T,) F0 values per frame
        - transcript: str - original transcript with punctuation
        - duration_s: float - audio duration
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("MLX not available")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_whisper_encoder(model_size: str = "large-v3"):
    """Load Whisper model encoder (frozen)."""
    from tools.whisper_mlx.model import WhisperMLX

    model_name = f"mlx-community/whisper-{model_size}-mlx"
    logger.info(f"Loading Whisper encoder: {model_name}")
    model = WhisperMLX.from_pretrained(model_name)
    return model


def load_emotion_head(checkpoint_path: str):
    """Load trained emotion classification head."""
    from tools.whisper_mlx.multi_head import EmotionHead, MultiHeadConfig

    logger.info(f"Loading emotion head: {checkpoint_path}")
    all_weights = mx.load(checkpoint_path)

    # Filter emotion-specific weights and remove prefix
    prefix = "emotion."
    weights = {}
    for k, v in all_weights.items():
        if k.startswith(prefix):
            weights[k[len(prefix):]] = v

    # Determine dimensions from weights
    if "fc2.weight" in weights:
        out_dim, hidden_dim = weights["fc2.weight"].shape
    else:
        out_dim, hidden_dim = 34, 512  # Default

    if "fc1.weight" in weights:
        _, in_dim = weights["fc1.weight"].shape
    else:
        in_dim = 1280  # Default for large-v3

    config = MultiHeadConfig(
        d_model=in_dim,
        num_emotions=out_dim,
        emotion_hidden_dim=hidden_dim,
    )
    head = EmotionHead(config)
    head.load_weights(list(weights.items()))
    return head


def load_pitch_head(checkpoint_path: str):
    """Load trained pitch prediction head (CREPEPitchHead or simple MLP)."""
    from tools.whisper_mlx.multi_head import MultiHeadConfig, CREPEPitchHead, PitchHeadMLP

    logger.info(f"Loading pitch head: {checkpoint_path}")
    all_weights = mx.load(checkpoint_path)

    # Filter pitch-specific weights and remove prefix
    prefix = "pitch."
    weights = {}
    for k, v in all_weights.items():
        if k.startswith(prefix):
            weights[k[len(prefix):]] = v

    # Determine if this is CREPEPitchHead or simple MLP based on weight keys
    is_crepe = "conv_layers.0.conv.conv.weight" in weights

    if is_crepe:
        # CREPEPitchHead
        if "input_proj.weight" in weights:
            hidden_dim, in_dim = weights["input_proj.weight"].shape
        else:
            in_dim, hidden_dim = 1280, 256

        config = MultiHeadConfig(d_model=in_dim, crepe_hidden_dim=hidden_dim)
        head = CREPEPitchHead(config)
    else:
        # Simple MLP PitchHead
        if "fc1.weight" in weights:
            hidden_dim, in_dim = weights["fc1.weight"].shape
        else:
            in_dim, hidden_dim = 1280, 128

        config = MultiHeadConfig(d_model=in_dim, pitch_hidden_dim=hidden_dim)
        head = PitchHeadMLP(config)

    head.load_weights(list(weights.items()))
    return head


def find_librispeech_samples(data_dir: Path) -> List[Dict]:
    """Find all LibriSpeech audio files and their transcripts."""
    samples = []

    # LibriSpeech structure: {split}/{speaker}/{chapter}/{speaker}-{chapter}-{utterance}.flac
    # Transcript in: {split}/{speaker}/{chapter}/{speaker}-{chapter}.trans.txt

    for split_dir in data_dir.iterdir():
        if not split_dir.is_dir():
            continue
        if split_dir.name.startswith('.'):
            continue

        for speaker_dir in split_dir.iterdir():
            if not speaker_dir.is_dir():
                continue

            for chapter_dir in speaker_dir.iterdir():
                if not chapter_dir.is_dir():
                    continue

                # Find transcript file
                trans_file = chapter_dir / f"{speaker_dir.name}-{chapter_dir.name}.trans.txt"
                if not trans_file.exists():
                    continue

                # Parse transcript file
                transcripts = {}
                with open(trans_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split(' ', 1)
                        if len(parts) == 2:
                            utterance_id, text = parts
                            transcripts[utterance_id] = text

                # Find audio files
                for audio_file in chapter_dir.glob("*.flac"):
                    utterance_id = audio_file.stem
                    if utterance_id in transcripts:
                        samples.append({
                            'audio_path': str(audio_file),
                            'transcript': transcripts[utterance_id],
                            'utterance_id': utterance_id,
                            'speaker': speaker_dir.name,
                            'chapter': chapter_dir.name,
                            'split': split_dir.name,
                        })

    return samples


def extract_features_single(
    sample: Dict,
    whisper_model,
    emotion_head,
    pitch_head,
    output_dir: Path,
) -> Optional[str]:
    """Extract features for a single audio file."""
    from tools.whisper_mlx.audio import load_audio, log_mel_spectrogram

    audio_path = sample['audio_path']
    utterance_id = sample['utterance_id']
    output_path = output_dir / f"{utterance_id}.npz"

    # Skip if already processed
    if output_path.exists():
        return None

    try:
        # Load audio
        audio = load_audio(audio_path)
        duration_s = len(audio) / 16000

        # Skip audio > 30 seconds (exceeds Whisper context)
        if duration_s > 30.0:
            logger.debug(f"Skipping {audio_path}: duration {duration_s:.1f}s > 30s")
            return None

        # Compute mel spectrogram
        mel = log_mel_spectrogram(audio, n_mels=whisper_model.config.n_mels)
        mel_mx = mx.expand_dims(mx.array(mel), axis=0)

        # Run through encoder (variable_length=True for non-30s audio)
        encoder_out = whisper_model.encoder(mel_mx, variable_length=True)

        # Run through emotion head
        emotion_logits = emotion_head(encoder_out)
        emotion_probs = mx.softmax(emotion_logits, axis=-1)

        # Run through pitch head (returns tuple: pitch_hz, voicing_prob)
        pitch_result = pitch_head(encoder_out)
        if isinstance(pitch_result, tuple):
            pitch_hz, voicing_prob = pitch_result
        else:
            pitch_hz = pitch_result
            voicing_prob = None

        # Single mx.eval() for all outputs (batched evaluation)
        mx.eval(encoder_out, emotion_probs, pitch_hz)

        # Convert to numpy and save
        np.savez_compressed(
            output_path,
            encoder_features=np.array(encoder_out[0]),  # (T, 1280)
            emotion_probs=np.array(emotion_probs[0]),   # (num_classes,)
            pitch_values=np.array(pitch_hz[0]),  # (T,)
            transcript=sample['transcript'],
            duration_s=duration_s,
            utterance_id=utterance_id,
            speaker=sample['speaker'],
        )

        return utterance_id

    except Exception as e:
        logger.warning(f"Error processing {audio_path}: {e}")
        return None


def extract_features_batch(
    samples: List[Dict],
    whisper_model,
    emotion_head,
    pitch_head,
    output_dir: Path,
    batch_size: int = 8,
) -> int:
    """Extract features for all samples with batched processing."""
    from tools.whisper_mlx.audio import load_audio, log_mel_spectrogram

    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0

    # Filter out already processed samples
    pending_samples = []
    for sample in samples:
        output_path = output_dir / f"{sample['utterance_id']}.npz"
        if not output_path.exists():
            pending_samples.append(sample)
        else:
            skipped += 1

    logger.info(f"Skipping {skipped} already processed, {len(pending_samples)} remaining")

    # Process in batches
    with tqdm(total=len(pending_samples), desc="Extracting features") as pbar:
        for i in range(0, len(pending_samples), batch_size):
            batch_samples = pending_samples[i:i + batch_size]

            # Load and prepare batch
            batch_data = []
            for sample in batch_samples:
                try:
                    audio = load_audio(sample['audio_path'])
                    duration_s = len(audio) / 16000

                    if duration_s > 30.0:
                        continue

                    mel = log_mel_spectrogram(audio, n_mels=whisper_model.config.n_mels)
                    batch_data.append({
                        'sample': sample,
                        'mel': mel,
                        'duration_s': duration_s,
                    })
                except Exception as e:
                    logger.warning(f"Error loading {sample['audio_path']}: {e}")

            if not batch_data:
                pbar.update(len(batch_samples))
                continue

            # Process each sample (can't truly batch due to variable lengths)
            # But we batch the mx.eval() calls
            results = []
            for data in batch_data:
                mel_mx = mx.expand_dims(mx.array(data['mel']), axis=0)
                encoder_out = whisper_model.encoder(mel_mx, variable_length=True)
                emotion_logits = emotion_head(encoder_out)
                emotion_probs = mx.softmax(emotion_logits, axis=-1)
                pitch_result = pitch_head(encoder_out)
                if isinstance(pitch_result, tuple):
                    pitch_hz, _ = pitch_result
                else:
                    pitch_hz = pitch_result

                results.append({
                    'sample': data['sample'],
                    'duration_s': data['duration_s'],
                    'encoder_out': encoder_out,
                    'emotion_probs': emotion_probs,
                    'pitch_hz': pitch_hz,
                })

            # Single batched eval for all samples in batch
            all_tensors = []
            for r in results:
                all_tensors.extend([r['encoder_out'], r['emotion_probs'], r['pitch_hz']])
            mx.eval(*all_tensors)

            # Save results
            for r in results:
                sample = r['sample']
                output_path = output_dir / f"{sample['utterance_id']}.npz"
                try:
                    np.savez_compressed(
                        output_path,
                        encoder_features=np.array(r['encoder_out'][0]),
                        emotion_probs=np.array(r['emotion_probs'][0]),
                        pitch_values=np.array(r['pitch_hz'][0]),
                        transcript=sample['transcript'],
                        duration_s=r['duration_s'],
                        utterance_id=sample['utterance_id'],
                        speaker=sample['speaker'],
                    )
                    processed += 1
                except Exception as e:
                    logger.warning(f"Error saving {output_path}: {e}")

            pbar.update(len(batch_samples))
            pbar.set_postfix(processed=processed, rate=f"{processed/(i+batch_size)*100:.1f}%")

    return processed


def main():
    parser = argparse.ArgumentParser(description="Extract emotion/pitch features for punctuation training")
    parser.add_argument("--data-dir", type=Path, required=True,
                       help="Path to LibriSpeech data directory")
    parser.add_argument("--output-dir", type=Path, required=True,
                       help="Output directory for extracted features")
    parser.add_argument("--emotion-head", type=str, required=True,
                       help="Path to emotion head checkpoint")
    parser.add_argument("--pitch-head", type=str, required=True,
                       help="Path to pitch head checkpoint")
    parser.add_argument("--model-size", type=str, default="large-v3",
                       help="Whisper model size")
    parser.add_argument("--splits", type=str, nargs="+",
                       default=["train-clean-100", "train-clean-360", "train-other-500", "dev-clean"],
                       help="LibriSpeech splits to process")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum samples to process (for testing)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Feature Extraction for Punctuation Training")
    logger.info("=" * 60)
    logger.info(f"Data dir: {args.data_dir}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Emotion head: {args.emotion_head}")
    logger.info(f"Pitch head: {args.pitch_head}")
    logger.info(f"Model size: {args.model_size}")

    # Load models
    whisper_model = load_whisper_encoder(args.model_size)
    emotion_head = load_emotion_head(args.emotion_head)
    pitch_head = load_pitch_head(args.pitch_head)

    # Find samples
    logger.info(f"Finding samples in {args.data_dir}...")
    all_samples = find_librispeech_samples(args.data_dir)
    logger.info(f"Found {len(all_samples)} total samples")

    # Filter by splits if specified
    if args.splits:
        all_samples = [s for s in all_samples if s['split'] in args.splits]
        logger.info(f"Filtered to {len(all_samples)} samples in splits: {args.splits}")

    # Limit samples if specified
    if args.max_samples:
        all_samples = all_samples[:args.max_samples]
        logger.info(f"Limited to {len(all_samples)} samples")

    # Extract features
    processed = extract_features_batch(
        all_samples,
        whisper_model,
        emotion_head,
        pitch_head,
        args.output_dir,
    )

    logger.info("=" * 60)
    logger.info(f"Extraction complete: {processed} files processed")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
