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

"""Quick test of realtime preset for RTF < 1.0."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.whisper_mlx.streaming_eval import StreamingEvaluator
from tools.whisper_mlx.audio import load_audio


def parse_librispeech_transcripts(trans_file: Path) -> dict:
    transcripts = {}
    with open(trans_file) as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                transcripts[parts[0]] = parts[1]
    return transcripts


async def main():
    from tools.whisper_mlx.model import WhisperMLX
    from tools.whisper_mlx.streaming import get_streaming_config

    print("Loading model...")
    model = WhisperMLX.from_pretrained("mlx-community/whisper-large-v3-mlx")

    # Use realtime preset
    config = get_streaming_config("realtime")
    print(f"Config: emit_partials={config.emit_partials}, partial_interval={config.partial_interval}")

    evaluator = StreamingEvaluator(
        model, config,
        harness_config={"speed_factor": 100.0}
    )

    data_dir = Path(__file__).parent.parent / "data" / "LibriSpeech" / "dev-clean"

    # Evaluate 10 samples
    count = 0
    for speaker_dir in sorted(data_dir.iterdir())[:2]:
        if not speaker_dir.is_dir():
            continue
        for chapter_dir in sorted(speaker_dir.iterdir())[:1]:
            if not chapter_dir.is_dir():
                continue
            trans_files = list(chapter_dir.glob("*.trans.txt"))
            if not trans_files:
                continue
            transcripts = parse_librispeech_transcripts(trans_files[0])

            for audio_file in sorted(chapter_dir.glob("*.flac"))[:5]:
                sample_id = audio_file.stem
                if sample_id not in transcripts:
                    continue

                audio = load_audio(str(audio_file))
                reference = transcripts[sample_id]

                metrics = await evaluator.evaluate_sample(audio, reference, sample_id)
                print(f"{sample_id}: WER={metrics.wer*100:5.1f}% RTF={metrics.rtf:.3f}")

                count += 1
                if count >= 10:
                    break
            if count >= 10:
                break
        if count >= 10:
            break

    evaluator.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
