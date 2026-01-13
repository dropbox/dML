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
Benchmark WhisperMLX model variants for speed and accuracy.

Compares:
- large-v3 (32 encoder, 32 decoder) - baseline
- large-v3-turbo (32 encoder, 4 decoder) - 4x faster decoder
- distil-large-v3 (32 encoder, 2 decoder) - 6x faster decoder

Usage:
    python scripts/benchmark_whisper_variants.py --num-files 50
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import jiwer


@dataclass
class BenchmarkResult:
    """Results from benchmarking a single model variant."""
    model_name: str
    num_files: int
    total_audio_seconds: float
    total_time_seconds: float
    rtf: float  # Real-time factor (lower is faster)
    wer: float  # Word Error Rate
    exact_match_rate: float
    avg_time_per_file_ms: float
    speedup_vs_baseline: float = 1.0


def load_ground_truth(trans_file: Path) -> Dict[str, str]:
    """Load LibriSpeech ground truth transcripts."""
    transcripts = {}
    with open(trans_file, "r") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                file_id, text = parts
                transcripts[file_id] = text.lower()
    return transcripts


def find_librispeech_files(
    data_dir: Path,
    max_files: int = 100,
) -> List[tuple]:
    """Find LibriSpeech audio files with their transcripts."""
    files = []
    for trans_file in data_dir.glob("**/*.trans.txt"):
        transcripts = load_ground_truth(trans_file)
        parent_dir = trans_file.parent

        for file_id, text in transcripts.items():
            audio_file = parent_dir / f"{file_id}.flac"
            if audio_file.exists():
                files.append((audio_file, text))
                if len(files) >= max_files:
                    return files

    return files


def benchmark_model(
    model_name: str,
    test_files: List[tuple],
    verbose: bool = False,
) -> BenchmarkResult:
    """Benchmark a single model variant."""
    from tools.whisper_mlx import WhisperMLX
    from tools.whisper_mlx.audio import SAMPLE_RATE, load_audio

    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name}")
    print(f"{'='*60}")

    # Map model names to HuggingFace paths
    hf_model_map = {
        "large-v3": "mlx-community/whisper-large-v3-mlx",
        "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
        "distil-large-v3": "mlx-community/distil-whisper-large-v3",
    }

    hf_path = hf_model_map.get(model_name, model_name)

    # Load model (this includes warmup)
    print(f"Loading model from {hf_path}...")
    t0 = time.perf_counter()
    model = WhisperMLX.from_pretrained(hf_path, warmup=True)
    load_time = time.perf_counter() - t0
    print(f"Model loaded in {load_time:.2f}s")

    # Run transcriptions
    predictions = []
    references = []
    times = []
    total_audio_duration = 0.0

    for i, (audio_file, ground_truth) in enumerate(test_files):
        # Load audio to get duration
        audio = load_audio(str(audio_file), sample_rate=SAMPLE_RATE)
        duration = len(audio) / SAMPLE_RATE
        total_audio_duration += duration

        # Transcribe
        t0 = time.perf_counter()
        result = model.transcribe(str(audio_file))
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        # Collect results
        pred_text = result["text"].strip().lower()
        predictions.append(pred_text)
        references.append(ground_truth)

        if verbose or (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(test_files)}] {duration:.1f}s audio -> {elapsed*1000:.0f}ms")

    # Calculate metrics
    total_time = sum(times)
    rtf = total_time / total_audio_duration

    # Calculate WER using jiwer
    wer = jiwer.wer(references, predictions)

    # Calculate exact match rate
    exact_matches = sum(1 for p, r in zip(predictions, references) if p == r)
    exact_match_rate = exact_matches / len(test_files)

    result = BenchmarkResult(
        model_name=model_name,
        num_files=len(test_files),
        total_audio_seconds=total_audio_duration,
        total_time_seconds=total_time,
        rtf=rtf,
        wer=wer,
        exact_match_rate=exact_match_rate,
        avg_time_per_file_ms=total_time / len(test_files) * 1000,
    )

    print(f"\nResults for {model_name}:")
    print(f"  Audio duration: {total_audio_duration:.1f}s")
    print(f"  Processing time: {total_time:.1f}s")
    print(f"  Real-time factor: {rtf:.3f}x")
    print(f"  WER: {wer*100:.2f}%")
    print(f"  Exact match: {exact_match_rate*100:.1f}%")
    print(f"  Avg time/file: {result.avg_time_per_file_ms:.0f}ms")

    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark WhisperMLX model variants")
    parser.add_argument(
        "--num-files",
        type=int,
        default=50,
        help="Number of test files to use (default: 50)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/benchmarks/librispeech/LibriSpeech",
        help="Path to LibriSpeech data",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["large-v3", "large-v3-turbo", "distil-large-v3"],
        help="Models to benchmark",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/main/MODEL_VARIANT_BENCHMARK.md",
        help="Output report path",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output",
    )

    args = parser.parse_args()

    # Find test files
    data_dir = Path(args.data_dir)
    print(f"Finding test files in {data_dir}...")
    test_files = find_librispeech_files(data_dir, max_files=args.num_files)
    print(f"Found {len(test_files)} test files")

    if len(test_files) == 0:
        print("ERROR: No test files found!")
        return 1

    # Benchmark each model
    results = []
    for model_name in args.models:
        try:
            result = benchmark_model(model_name, test_files, verbose=args.verbose)
            results.append(result)
        except Exception as e:
            print(f"ERROR benchmarking {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Calculate speedup vs baseline (first model)
    if not results:
        print("ERROR: No benchmark results collected!")
        return 1

    baseline_rtf = results[0].rtf
    for r in results:
        r.speedup_vs_baseline = baseline_rtf / r.rtf

    # Generate report
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    report = f"""# WhisperMLX Model Variant Benchmark

**Date**: {time.strftime("%Y-%m-%d %H:%M")}
**Test Files**: {len(test_files)} LibriSpeech test-clean samples
**Total Audio**: {results[0].total_audio_seconds:.1f}s

## Results

| Model | Decoder Layers | RTF | WER | Exact Match | Speedup |
|-------|----------------|-----|-----|-------------|---------|
"""

    decoder_layers = {
        "large-v3": 32,
        "large-v3-turbo": 4,
        "distil-large-v3": 2,
    }

    for r in results:
        layers = decoder_layers.get(r.model_name, "?")
        report += (
            f"| {r.model_name} | {layers} | {r.rtf:.3f}x | "
            f"{r.wer*100:.2f}% | {r.exact_match_rate*100:.1f}% | "
            f"{r.speedup_vs_baseline:.2f}x |\n"
        )

        print(f"{r.model_name}: RTF={r.rtf:.3f}x, WER={r.wer*100:.2f}%, Speedup={r.speedup_vs_baseline:.2f}x")

    report += """
## Analysis

"""

    if len(results) >= 2:
        baseline = results[0]
        for r in results[1:]:
            wer_diff = (r.wer - baseline.wer) * 100
            report += f"""### {r.model_name} vs {baseline.model_name}

- **Speedup**: {r.speedup_vs_baseline:.2f}x faster
- **WER change**: {wer_diff:+.2f}% ({'worse' if wer_diff > 0 else 'better' if wer_diff < 0 else 'same'})
- **Decoder layers**: {decoder_layers.get(r.model_name, '?')} vs {decoder_layers.get(baseline.model_name, '?')}

"""

    report += """## Recommendations

1. **For maximum speed with acceptable quality**: Use `large-v3-turbo` or `distil-large-v3`
2. **For maximum quality**: Use `large-v3` (baseline)
3. **For streaming/real-time**: Use `distil-large-v3` (fastest decoder)

## Usage

```python
from tools.whisper_mlx import WhisperMLX

# Fastest option
model = WhisperMLX.from_pretrained("mlx-community/distil-whisper-large-v3")

# Balanced option
model = WhisperMLX.from_pretrained("mlx-community/whisper-large-v3-turbo")

# Maximum quality
model = WhisperMLX.from_pretrained("mlx-community/whisper-large-v3-mlx")
```
"""

    # Write report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(f"\nReport written to {args.output}")

    # Also save JSON results
    json_path = output_path.with_suffix(".json")
    json_results = [
        {
            "model_name": r.model_name,
            "num_files": r.num_files,
            "total_audio_seconds": r.total_audio_seconds,
            "total_time_seconds": r.total_time_seconds,
            "rtf": r.rtf,
            "wer": r.wer,
            "exact_match_rate": r.exact_match_rate,
            "speedup_vs_baseline": r.speedup_vs_baseline,
        }
        for r in results
    ]
    json_path.write_text(json.dumps(json_results, indent=2))
    print(f"JSON results written to {json_path}")

    return 0


if __name__ == "__main__":
    exit(main())
