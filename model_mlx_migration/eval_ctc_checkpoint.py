#!/usr/bin/env python3
"""
Quick CTC checkpoint evaluation for multilingual training.

Usage:
    python eval_ctc_checkpoint.py --checkpoint checkpoints/ctc_multilingual_v1/step_12500.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf

from tools.whisper_mlx.ctc_head import CTCDraftHead


def unflatten_params(flat_params: dict) -> dict:
    """Convert flat parameter dict to nested structure."""
    nested = {}
    for key, value in flat_params.items():
        parts = key.split(".")
        current = nested
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return nested


def compute_wer(reference: str, hypothesis: str) -> tuple[float, int, int, int, int]:
    """Compute Word Error Rate."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    if len(ref_words) == 0:
        if len(hyp_words) == 0:
            return 0.0, 0, 0, 0, 0
        return 1.0, 0, len(hyp_words), 0, 0

    # Simple Levenshtein
    m, n = len(ref_words), len(hyp_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1])

    distance = dp[m][n]
    wer = distance / len(ref_words)
    return wer, 0, 0, 0, len(ref_words)


def load_librispeech_samples(data_dir: str, max_samples: int = 0) -> list[tuple[str, str]]:
    """Load LibriSpeech samples (audio_path, transcript)."""
    samples = []
    data_path = Path(data_dir)

    if not data_path.exists():
        return []

    for speaker_dir in sorted(data_path.iterdir()):
        if not speaker_dir.is_dir():
            continue

        for chapter_dir in sorted(speaker_dir.iterdir()):
            if not chapter_dir.is_dir():
                continue

            speaker_id = speaker_dir.name
            chapter_id = chapter_dir.name

            transcript_file = chapter_dir / f"{speaker_id}-{chapter_id}.trans.txt"
            if not transcript_file.exists():
                continue

            with open(transcript_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split(" ", 1)
                    if len(parts) != 2:
                        continue

                    utterance_id, transcript = parts
                    audio_path = chapter_dir / f"{utterance_id}.flac"
                    if not audio_path.exists():
                        continue

                    samples.append((str(audio_path), transcript))

                    if max_samples > 0 and len(samples) >= max_samples:
                        return samples

    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to CTC checkpoint")
    parser.add_argument("--max-samples", type=int, default=50, help="Max samples to evaluate")
    parser.add_argument("--data-dir", default="data/LibriSpeech/dev-clean", help="LibriSpeech data dir")
    parser.add_argument("--whisper-model", default="mlx-community/whisper-large-v3-mlx", help="Whisper model")
    args = parser.parse_args()

    print("=" * 70)
    print("CTC Checkpoint Evaluation")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data: {args.data_dir}")
    print(f"Max samples: {args.max_samples}")
    print()

    # Load Whisper model (using WhisperMLX which supports variable-length encoding)
    print("Loading Whisper model...")
    from tools.whisper_mlx.model import WhisperMLX
    from tools.whisper_mlx.tokenizer import get_whisper_tokenizer
    whisper = WhisperMLX.from_pretrained(args.whisper_model)
    tokenizer = get_whisper_tokenizer(
        multilingual=True,
        language="en",
        task="transcribe",
    )
    n_mels = whisper.config.n_mels

    # Get config for CTC head
    d_model = 1280  # large-v3
    vocab_size = 51865

    # Create and load CTC head
    print(f"Loading CTC checkpoint: {args.checkpoint}")
    ctc_head = CTCDraftHead(d_model=d_model, vocab_size=vocab_size)

    flat_params = mx.load(args.checkpoint)
    nested_params = unflatten_params(flat_params)
    ctc_head.update(nested_params)

    # Count parameters
    def count_params(params):
        total = 0
        if isinstance(params, dict):
            for v in params.values():
                total += count_params(v)
        elif isinstance(params, mx.array):
            total += params.size
        return total
    param_count = count_params(ctc_head.parameters())
    print(f"CTC head parameters: {param_count:,}")

    # Load samples
    print(f"\nLoading samples from {args.data_dir}...")
    samples = load_librispeech_samples(args.data_dir, args.max_samples)
    print(f"Loaded {len(samples)} samples")

    if not samples:
        print("ERROR: No samples found!")
        return

    # Import audio processing
    from tools.whisper_mlx.audio import log_mel_spectrogram

    # Evaluate
    print("\nEvaluating...")
    total_errors = 0
    total_ref_words = 0
    results = []

    for i, (audio_path, transcript) in enumerate(samples):
        try:
            # Load audio
            audio, sr = sf.read(audio_path)
            audio = audio.astype(np.float32)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)  # Convert stereo to mono
            if sr != 16000:
                from scipy import signal
                audio = signal.resample(audio, int(len(audio) * 16000 / sr))

            # Mel spectrogram (variable length - no padding needed)
            mel = log_mel_spectrogram(audio, n_mels=n_mels)
            mel_tensor = mx.expand_dims(mx.array(mel), axis=0)

            # Encoder with variable length support
            encoder_output = whisper.encoder(mel_tensor, variable_length=True)
            mx.eval(encoder_output)

            # CTC forward
            logits = ctc_head(encoder_output)
            mx.eval(logits)

            # Greedy decode
            tokens = ctc_head.decode_greedy(logits)

            # Decode to text
            hypothesis = tokenizer.decode(tokens)

            # Compute WER
            wer, _, _, _, ref_len = compute_wer(transcript, hypothesis)

            # Track errors for proper WER calculation
            ref_words = transcript.lower().split()
            hyp_words = hypothesis.lower().split()

            # Simple edit distance for error count
            m, n = len(ref_words), len(hyp_words)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(m + 1):
                dp[i][0] = i
            for j in range(n + 1):
                dp[0][j] = j
            for ii in range(1, m + 1):
                for jj in range(1, n + 1):
                    if ref_words[ii-1] == hyp_words[jj-1]:
                        dp[ii][jj] = dp[ii-1][jj-1]
                    else:
                        dp[ii][jj] = 1 + min(dp[ii-1][jj-1], dp[ii-1][jj], dp[ii][jj-1])

            errors = dp[m][n]
            total_errors += errors
            total_ref_words += len(ref_words)

            results.append({
                "wer": wer,
                "ref_words": len(ref_words),
                "hyp_words": len(hyp_words),
                "errors": errors,
            })

            if (i + 1) % 10 == 0:
                running_wer = total_errors / total_ref_words if total_ref_words > 0 else 0
                print(f"  [{i+1}/{len(samples)}] Running WER: {running_wer:.2%}")

        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            continue

    # Final results
    overall_wer = total_errors / total_ref_words if total_ref_words > 0 else 0

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Samples evaluated: {len(results)}")
    print(f"Total reference words: {total_ref_words}")
    print(f"Total errors: {total_errors}")
    print(f"Overall WER: {overall_wer:.2%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
