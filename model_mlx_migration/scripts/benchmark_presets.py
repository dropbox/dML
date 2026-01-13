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
Benchmark WhisperMLX optimization presets.

This script compares the performance of different optimization presets
on a set of test audio files.

Usage:
    python scripts/benchmark_presets.py --audio test_audio.wav
    python scripts/benchmark_presets.py --librispeech 10
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def benchmark_preset(preset, audio_files, warmup=True, verbose=False):
    """Benchmark a single preset on audio files."""
    from tools.whisper_mlx import WhisperMLX, TranscriptionConfig

    print(f"\n{'='*60}")
    print(f"Preset: {preset.name}")
    config = TranscriptionConfig.from_preset(preset)
    print(config.describe())
    print(f"{'='*60}")

    # Load model
    print("\nLoading model...")
    t0 = time.perf_counter()
    model = WhisperMLX.from_preset(preset, warmup=warmup)
    load_time = time.perf_counter() - t0
    print(f"Model loaded in {load_time:.2f}s")

    # Benchmark transcription
    results = []
    total_audio_duration = 0.0
    total_transcribe_time = 0.0

    for i, audio_file in enumerate(audio_files):
        if verbose:
            print(f"\n[{i+1}/{len(audio_files)}] {audio_file.name}")

        # Get audio duration
        from tools.whisper_mlx.audio import load_audio, SAMPLE_RATE
        audio = load_audio(str(audio_file), sample_rate=SAMPLE_RATE)
        audio_duration = len(audio) / SAMPLE_RATE
        total_audio_duration += audio_duration

        # Transcribe
        t0 = time.perf_counter()
        result = model.transcribe_with_config(audio, config)
        transcribe_time = time.perf_counter() - t0
        total_transcribe_time += transcribe_time

        results.append({
            "file": audio_file.name,
            "audio_duration": audio_duration,
            "transcribe_time": transcribe_time,
            "text": result["text"][:100] + "..." if len(result["text"]) > 100 else result["text"],
            "rtf": transcribe_time / audio_duration,
        })

        if verbose:
            print(f"  Duration: {audio_duration:.2f}s, Time: {transcribe_time:.2f}s, RTF: {results[-1]['rtf']:.3f}x")
            print(f"  Text: {results[-1]['text']}")

    # Summary
    avg_rtf = total_transcribe_time / total_audio_duration
    print(f"\n{preset.name} Summary:")
    print(f"  Files: {len(audio_files)}")
    print(f"  Total audio: {total_audio_duration:.1f}s")
    print(f"  Total time: {total_transcribe_time:.2f}s")
    print(f"  Avg RTF: {avg_rtf:.3f}x")

    return {
        "preset": preset.name,
        "model": config.model_variant.name,
        "weight_bits": config.weight_bits,
        "load_time": load_time,
        "total_audio": total_audio_duration,
        "total_time": total_transcribe_time,
        "avg_rtf": avg_rtf,
        "results": results,
    }


def get_librispeech_files(n_files=10):
    """Get n random files from LibriSpeech test-clean."""
    librispeech_dir = Path.home() / ".cache" / "huggingface" / "hub" / "datasets--hf-internal-testing--librispeech_asr_dummy"

    # Try different possible locations
    possible_dirs = [
        librispeech_dir / "refs" / "main" / "data" / "audio",
        Path("/Users/ayates/model_mlx_migration/test_data/librispeech"),
    ]

    audio_files = []
    for base_dir in possible_dirs:
        if base_dir.exists():
            audio_files.extend(list(base_dir.rglob("*.flac")))
            audio_files.extend(list(base_dir.rglob("*.wav")))

    if not audio_files:
        print("LibriSpeech files not found. Please provide audio files with --audio.")
        return []

    # Shuffle and limit
    import random
    random.shuffle(audio_files)
    return audio_files[:n_files]


def main():
    parser = argparse.ArgumentParser(description="Benchmark WhisperMLX presets")
    parser.add_argument("--audio", type=str, nargs="*", help="Audio file(s) to transcribe")
    parser.add_argument("--librispeech", type=int, default=0, help="Number of LibriSpeech files to use")
    parser.add_argument("--presets", type=str, nargs="*",
                       choices=["max_quality", "balanced", "fast", "ultra_fast"],
                       default=["max_quality", "balanced", "fast"],
                       help="Presets to benchmark")
    parser.add_argument("--no-warmup", action="store_true", help="Skip model warmup")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Get audio files
    audio_files = []
    if args.audio:
        audio_files = [Path(f) for f in args.audio if Path(f).exists()]
    elif args.librispeech > 0:
        audio_files = get_librispeech_files(args.librispeech)
    else:
        # Default: try LibriSpeech with 5 files
        audio_files = get_librispeech_files(5)

    if not audio_files:
        print("No audio files found. Use --audio or --librispeech.")
        return 1

    print(f"Benchmarking with {len(audio_files)} audio files")

    # Import preset enum
    from tools.whisper_mlx import OptimizationPreset

    preset_map = {
        "max_quality": OptimizationPreset.MAX_QUALITY,
        "balanced": OptimizationPreset.BALANCED,
        "fast": OptimizationPreset.FAST,
        "ultra_fast": OptimizationPreset.ULTRA_FAST,
    }

    # Run benchmarks
    all_results = []
    for preset_name in args.presets:
        preset = preset_map[preset_name]
        result = benchmark_preset(
            preset, audio_files,
            warmup=not args.no_warmup,
            verbose=args.verbose,
        )
        all_results.append(result)

    # Final comparison
    print("\n" + "="*60)
    print("BENCHMARK COMPARISON")
    print("="*60)
    print(f"{'Preset':<15} {'Model':<20} {'Weights':<8} {'RTF':<10} {'Speedup':<10}")
    print("-"*60)

    baseline_rtf = all_results[0]["avg_rtf"] if all_results else 1.0
    for r in all_results:
        weights = "FP16" if r["weight_bits"] is None else f"INT{r['weight_bits']}"
        speedup = baseline_rtf / r["avg_rtf"]
        print(f"{r['preset']:<15} {r['model']:<20} {weights:<8} {r['avg_rtf']:.3f}x     {speedup:.2f}x")

    return 0


if __name__ == "__main__":
    sys.exit(main())
