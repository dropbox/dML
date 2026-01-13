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
Run streaming evaluation on LibriSpeech dev-clean.

Measures streaming-specific metrics:
- WER (quality)
- First partial latency
- Finalization latency
- Edit rate
- RTF (compute)
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.whisper_mlx.streaming_eval import (
    StreamingEvaluator,
)
from tools.whisper_mlx.audio import load_audio


def parse_librispeech_transcripts(trans_file: Path) -> dict:
    """Parse LibriSpeech .trans.txt file."""
    transcripts = {}
    with open(trans_file) as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                transcripts[parts[0]] = parts[1]
    return transcripts


def iter_librispeech_devclean(data_dir: Path, max_samples: int = None):
    """Iterate over LibriSpeech dev-clean samples."""
    dev_clean = data_dir / "LibriSpeech" / "dev-clean"

    count = 0
    for speaker_dir in sorted(dev_clean.iterdir()):
        if not speaker_dir.is_dir():
            continue

        for chapter_dir in sorted(speaker_dir.iterdir()):
            if not chapter_dir.is_dir():
                continue

            # Find transcript file
            trans_files = list(chapter_dir.glob("*.trans.txt"))
            if not trans_files:
                continue

            transcripts = parse_librispeech_transcripts(trans_files[0])

            for audio_file in sorted(chapter_dir.glob("*.flac")):
                sample_id = audio_file.stem
                if sample_id not in transcripts:
                    continue

                audio = load_audio(str(audio_file))
                reference = transcripts[sample_id]

                yield audio, reference, sample_id

                count += 1
                if max_samples and count >= max_samples:
                    return


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Streaming eval on LibriSpeech dev-clean")
    parser.add_argument("--max-samples", type=int, default=50,
                        help="Maximum samples to evaluate")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (auto-selected from latency-mode if not specified)")
    parser.add_argument("--latency-mode", type=str, default="balanced",
                        choices=["fast", "balanced", "quality"])
    parser.add_argument("--preset", type=str, default=None,
                        choices=["low_latency", "realtime", "balanced", "stable", "no_retract", "responsive", "legacy"],
                        help="Streaming preset (overrides other config)")
    parser.add_argument("--speed-factor", type=float, default=100.0,
                        help="Replay speed (100 = max speed)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path")

    args = parser.parse_args()

    # Import models
    from tools.whisper_mlx.model import WhisperMLX
    from tools.whisper_mlx.streaming import StreamingConfig, get_streaming_config

    # Configure streaming
    if args.preset:
        config = get_streaming_config(args.preset)
        latency_mode = config.latency_mode
        print(f"Using preset: {args.preset}")
    else:
        config = StreamingConfig(
            use_local_agreement=True,
            latency_mode=args.latency_mode,
        )
        latency_mode = args.latency_mode

    # Model selection: explicit > preset recommendation > latency_mode recommendation
    if args.model:
        model_name = args.model
    else:
        # Map latency_mode to recommended model
        MODEL_MAP = {
            "fast": "mlx-community/whisper-small-mlx",
            "balanced": "mlx-community/whisper-large-v3-turbo",
            "quality": "mlx-community/whisper-large-v3-mlx",
        }
        model_name = MODEL_MAP.get(latency_mode, MODEL_MAP["balanced"])
        print(f"Auto-selected model for latency_mode={latency_mode}")

    print(f"Loading model: {model_name}")
    model = WhisperMLX.from_pretrained(model_name)

    # Create evaluator
    evaluator = StreamingEvaluator(
        model,
        config,
        harness_config={"speed_factor": args.speed_factor},
    )

    data_dir = Path(__file__).parent.parent / "data"

    config_name = args.preset if args.preset else latency_mode
    print(f"Evaluating LibriSpeech dev-clean (max {args.max_samples} samples)")
    print(f"Config: {config_name}, Model: {model_name.split('/')[-1]}")
    print(f"Partial interval: {config.partial_interval}s")
    print()

    # Evaluate samples
    count = 0
    for audio, reference, sample_id in iter_librispeech_devclean(data_dir, args.max_samples):
        print(f"[{count+1:3d}] {sample_id}...", end=" ", flush=True)

        try:
            metrics = await evaluator.evaluate_sample(audio, reference, sample_id)
            print(f"WER={metrics.wer * 100:5.1f}%  RTF={metrics.rtf:.3f}  "
                  f"lat={metrics.first_partial_latency_ms:.0f}ms")
        except Exception as e:
            print(f"ERROR: {e}")

        count += 1

    if count == 0:
        print("ERROR: No samples found")
        return

    # Print summary
    evaluator.print_summary()

    # Save results
    output_path = args.output or f"reports/streaming_eval_devclean_{config_name}_{args.max_samples}.json"
    evaluator.save_results(output_path)


if __name__ == "__main__":
    asyncio.run(main())
