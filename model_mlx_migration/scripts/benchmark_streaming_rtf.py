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
Benchmark streaming RTF with different presets.

Compares RTF between:
- realtime: emit_partials=False
- balanced: partial_interval=2.0s (new default)
- legacy: partial_interval=0.5s (old default)
"""

import asyncio
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.whisper_mlx.audio import load_audio


async def benchmark_preset(model, audio, preset_name: str, config):
    """Benchmark a single preset."""
    from tools.whisper_mlx.streaming import StreamingWhisper

    streamer = StreamingWhisper(model, config)

    audio_duration = len(audio) / 16000  # 16kHz

    # Simulate streaming by feeding audio in chunks
    chunk_size = int(0.1 * 16000)  # 100ms chunks

    async def audio_generator():
        for i in range(0, len(audio), chunk_size):
            yield audio[i:i+chunk_size]

    start = time.perf_counter()
    results = []
    partials = 0
    finals = 0

    async for result in streamer.transcribe_stream(audio_generator()):
        results.append(result)
        if result.is_final:
            finals += 1
        else:
            partials += 1

    elapsed = time.perf_counter() - start
    rtf = elapsed / audio_duration

    # Get transcription
    final_text = " ".join(r.text for r in results if r.is_final)

    return {
        "preset": preset_name,
        "rtf": rtf,
        "elapsed": elapsed,
        "audio_duration": audio_duration,
        "partials": partials,
        "finals": finals,
        "text_preview": final_text[:80] + "..." if len(final_text) > 80 else final_text,
    }


async def main():
    from tools.whisper_mlx.model import WhisperMLX
    from tools.whisper_mlx.streaming import get_streaming_config

    # Load model
    print("Loading model...")
    model = WhisperMLX.from_pretrained("mlx-community/whisper-large-v3-mlx")

    # Find a test audio file
    data_dir = Path(__file__).parent.parent / "data"
    dev_clean = data_dir / "LibriSpeech" / "dev-clean"

    # Get first audio file
    audio_files = list(dev_clean.glob("*/*/*.flac"))
    if not audio_files:
        print("ERROR: No audio files found in data/LibriSpeech/dev-clean")
        return

    # Find a longer audio file (10-20s) for better benchmark
    test_file = None
    for f in audio_files[:50]:
        audio = load_audio(str(f))
        duration = len(audio) / 16000
        if 8 < duration < 15:  # 8-15 second audio
            test_file = f
            break

    if test_file is None:
        test_file = audio_files[0]

    audio = load_audio(str(test_file))
    duration = len(audio) / 16000

    print(f"Test audio: {test_file.name}")
    print(f"Duration: {duration:.1f}s")
    print()

    # Presets to benchmark
    presets = [
        ("realtime", get_streaming_config("realtime")),
        ("balanced", get_streaming_config("balanced")),  # New default
        ("responsive", get_streaming_config("responsive")),
        ("legacy", get_streaming_config("legacy")),  # Old default
    ]

    print("=" * 70)
    print(f"{'Preset':<12} {'RTF':>8} {'Elapsed':>8} {'Partials':>10} {'Finals':>8}")
    print("=" * 70)

    results = []
    for preset_name, config in presets:
        result = await benchmark_preset(model, audio, preset_name, config)
        results.append(result)

        print(f"{result['preset']:<12} {result['rtf']:>8.2f} {result['elapsed']:>7.2f}s "
              f"{result['partials']:>10} {result['finals']:>8}")

    print("=" * 70)

    # Calculate improvement
    if len(results) >= 4:
        legacy_rtf = results[3]["rtf"]  # legacy
        balanced_rtf = results[1]["rtf"]  # balanced
        realtime_rtf = results[0]["rtf"]  # realtime

        print("\nRTF Improvement:")
        print(f"  balanced vs legacy: {(1 - balanced_rtf/legacy_rtf)*100:.1f}% reduction")
        print(f"  realtime vs legacy: {(1 - realtime_rtf/legacy_rtf)*100:.1f}% reduction")

    # Show transcription quality
    print("\nTranscription preview (realtime):")
    print(f"  {results[0]['text_preview']}")


if __name__ == "__main__":
    asyncio.run(main())
