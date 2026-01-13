#!/usr/bin/env python3
"""
Generate Whisper-formatted CTC training targets from LibriSpeech.

Problem:
    CTC trained on LibriSpeech transcripts produces UPPERCASE tokens without punctuation.
    Whisper decoder outputs lowercase tokens with punctuation.
    This tokenization mismatch causes 0% acceptance rate in speculative decoding.

Solution:
    Run Whisper transcription on LibriSpeech audio files to get the exact token IDs
    that Whisper's decoder would produce. Use these as CTC training targets.

Usage:
    python scripts/generate_ctc_whisper_targets.py \
        --data-dir data/LibriSpeech_full \
        --output-dir data/ctc_whisper_targets \
        --model-size large-v3 \
        --num-workers 4

Output format:
    JSON lines file with:
    {
        "audio_path": "/path/to/audio.flac",
        "tokens": [634, 19737, 456, ...],  # Whisper decoder token IDs
        "text": " He hoped there would be...",  # For verification
        "original_transcript": "HE HOPED THERE WOULD BE..."  # Original for comparison
    }

Author: Worker #2090
Date: 2025-12-29
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class AudioSample:
    """Single audio sample to process."""
    audio_path: str
    original_transcript: str


def load_librispeech_samples(data_dir: Path) -> List[AudioSample]:
    """Load all LibriSpeech samples from data directory."""
    import glob as glob_module

    samples = []
    trans_pattern = str(data_dir / "**" / "*.trans.txt")
    trans_files = [Path(p) for p in glob_module.glob(trans_pattern, recursive=True)]

    print(f"Found {len(trans_files)} transcript files")

    for trans_file in trans_files:
        chapter_dir = trans_file.parent

        with open(trans_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(" ", 1)
                if len(parts) < 2:
                    continue

                utterance_id = parts[0]
                transcript = parts[1]

                # Find audio file
                audio_path = chapter_dir / f"{utterance_id}.flac"
                if not audio_path.exists():
                    audio_path = chapter_dir / f"{utterance_id}.wav"

                if audio_path.exists():
                    samples.append(AudioSample(
                        audio_path=str(audio_path),
                        original_transcript=transcript,
                    ))

    return samples


def transcribe_batch(args: Tuple[List[AudioSample], str, int]) -> List[Dict]:
    """
    Transcribe a batch of audio samples.

    This is the worker function for multiprocessing.
    Each worker loads its own model to avoid serialization issues.
    """
    samples, model_size, worker_id = args

    # Import inside worker to avoid serialization issues
    from tools.whisper_mlx import WhisperMLX
    from tools.whisper_mlx.tokenizer import get_whisper_tokenizer

    print(f"Worker {worker_id}: Loading model {model_size}...")
    model = WhisperMLX.from_pretrained(model_size)
    tokenizer = get_whisper_tokenizer(multilingual=True, num_languages=100)

    results = []
    for i, sample in enumerate(samples):
        if i % 100 == 0:
            print(f"Worker {worker_id}: Processing {i}/{len(samples)}")

        try:
            # Transcribe with greedy decoding (temperature=0)
            result = model.transcribe(
                sample.audio_path,
                temperature=0.0,  # Deterministic
                verbose=False,
            )

            text = result.get("text", "")

            # Encode the transcribed text to get the exact token IDs
            # that Whisper's decoder would produce
            tokens = tokenizer.encode(text)

            # Filter special tokens (timestamps, language tokens)
            # Keep only content tokens (< 50257)
            content_tokens = [t for t in tokens if t < 50257]

            results.append({
                "audio_path": sample.audio_path,
                "tokens": content_tokens,
                "text": text.strip(),
                "original_transcript": sample.original_transcript,
                "num_tokens": len(content_tokens),
            })

        except Exception as e:
            print(f"Worker {worker_id}: Error processing {sample.audio_path}: {e}")
            results.append({
                "audio_path": sample.audio_path,
                "tokens": [],
                "text": "",
                "original_transcript": sample.original_transcript,
                "error": str(e),
            })

    return results


def transcribe_sequential(samples: List[AudioSample], model_size: str, output_path: Path) -> int:
    """
    Transcribe samples sequentially with progress saving.

    This approach:
    1. Loads model once
    2. Processes samples one by one
    3. Saves progress after each batch of 100
    4. Can resume from checkpoint
    """
    from tools.whisper_mlx import WhisperMLX
    from tools.whisper_mlx.tokenizer import get_whisper_tokenizer

    # Check for existing progress
    processed = set()
    if output_path.exists():
        with open(output_path, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed.add(data["audio_path"])
                except:
                    pass
        print(f"Resuming from checkpoint: {len(processed)} already processed")

    # Filter out already processed
    remaining = [s for s in samples if s.audio_path not in processed]
    print(f"Remaining samples: {len(remaining)}")

    if not remaining:
        print("All samples already processed!")
        return len(processed)

    # Load model and tokenizer
    print(f"Loading model {model_size}...")
    model = WhisperMLX.from_pretrained(model_size)
    tokenizer = get_whisper_tokenizer(multilingual=True, num_languages=100)

    # Process samples
    start_time = time.time()
    results_buffer = []

    for i, sample in enumerate(remaining):
        try:
            result = model.transcribe(
                sample.audio_path,
                temperature=0.0,
                verbose=False,
            )

            text = result.get("text", "")

            # Encode the transcribed text to get the exact token IDs
            # that Whisper's decoder would produce
            tokens = tokenizer.encode(text)

            # Filter special tokens (timestamps, language, etc.)
            # Keep only content tokens (< 50257 = base Whisper vocab)
            content_tokens = [t for t in tokens if t < 50257]

            results_buffer.append({
                "audio_path": sample.audio_path,
                "tokens": content_tokens,
                "text": text.strip(),
                "original_transcript": sample.original_transcript,
                "num_tokens": len(content_tokens),
            })

        except Exception as e:
            print(f"Error processing {sample.audio_path}: {e}")
            results_buffer.append({
                "audio_path": sample.audio_path,
                "tokens": [],
                "text": "",
                "original_transcript": sample.original_transcript,
                "error": str(e),
            })

        # Progress and checkpoint
        if (i + 1) % 100 == 0 or i == len(remaining) - 1:
            # Save checkpoint
            with open(output_path, "a") as f:
                for r in results_buffer:
                    f.write(json.dumps(r) + "\n")
            results_buffer = []

            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(remaining) - i - 1) / rate if rate > 0 else 0

            print(f"Progress: {i + 1}/{len(remaining)} ({100*(i+1)/len(remaining):.1f}%) "
                  f"Rate: {rate:.1f} samples/sec, ETA: {eta/3600:.1f}h")

    return len(processed) + len(remaining)


def validate_output(output_path: Path, num_samples: int = 5):
    """Validate and show sample outputs."""
    print("\n" + "=" * 60)
    print("VALIDATION - Sample outputs:")
    print("=" * 60)

    with open(output_path, "r") as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break

            data = json.loads(line)
            print(f"\nSample {i + 1}:")
            print(f"  Audio: {Path(data['audio_path']).name}")
            print(f"  Original: {data['original_transcript'][:60]}...")
            print(f"  Whisper:  {data['text'][:60]}...")
            print(f"  Tokens: {data['tokens'][:10]}... ({data['num_tokens']} total)")

    # Count statistics
    total = 0
    empty = 0
    avg_tokens = 0

    with open(output_path, "r") as f:
        for line in f:
            data = json.loads(line)
            total += 1
            if not data["tokens"]:
                empty += 1
            avg_tokens += data.get("num_tokens", 0)

    print("\nStatistics:")
    print(f"  Total samples: {total}")
    print(f"  Empty outputs: {empty} ({100*empty/total:.1f}%)")
    print(f"  Average tokens: {avg_tokens/total:.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Whisper-formatted CTC training targets"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to LibriSpeech data directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for targets"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="large-v3",
        help="Whisper model size (default: large-v3)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1, sequential)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximum samples to process (0 = all)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Specific split to process (e.g., 'train-clean-100')"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing output file"
    )

    args = parser.parse_args()

    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output filename
    if args.split:
        output_file = output_dir / f"{args.split}_whisper_targets.jsonl"
        data_dir = data_dir / args.split
    else:
        output_file = output_dir / "librispeech_whisper_targets.jsonl"

    # Validate only mode
    if args.validate_only:
        if output_file.exists():
            validate_output(output_file)
        else:
            print(f"Output file not found: {output_file}")
        return

    # Load samples
    print(f"Loading LibriSpeech samples from {data_dir}...")
    samples = load_librispeech_samples(data_dir)
    print(f"Found {len(samples)} samples")

    if args.max_samples > 0:
        samples = samples[:args.max_samples]
        print(f"Limited to {len(samples)} samples")

    if not samples:
        print("No samples found!")
        return

    # Process
    print(f"\nProcessing with model {args.model_size}...")
    print(f"Output: {output_file}")

    if args.num_workers <= 1:
        # Sequential processing (recommended for Apple Silicon)
        total = transcribe_sequential(samples, args.model_size, output_file)
    else:
        # Parallel processing (may cause memory issues on Mac)
        print("Warning: Parallel processing may cause memory issues on Mac")
        print("Consider using --num-workers 1 instead")

        # Split samples into chunks for workers
        chunk_size = len(samples) // args.num_workers
        chunks = []
        for i in range(args.num_workers):
            start = i * chunk_size
            end = start + chunk_size if i < args.num_workers - 1 else len(samples)
            chunks.append((samples[start:end], args.model_size, i))

        # Process in parallel
        all_results = []
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(transcribe_batch, chunk) for chunk in chunks]
            for future in as_completed(futures):
                all_results.extend(future.result())

        # Write results
        with open(output_file, "w") as f:
            for r in all_results:
                f.write(json.dumps(r) + "\n")

        total = len(all_results)

    # Validate
    print(f"\nProcessed {total} samples")
    validate_output(output_file)

    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print(f"1. Update train_ctc.py to use targets from: {output_file}")
    print("2. Retrain CTC head with new targets")
    print("3. Test CTC speculative decoding acceptance rate")


if __name__ == "__main__":
    main()
