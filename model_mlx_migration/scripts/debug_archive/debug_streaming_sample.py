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
Debug script to understand why some streaming samples fail catastrophically.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.whisper_mlx.audio import load_audio


async def debug_sample(sample_path: str, reference: str):
    """Debug streaming vs offline transcription for a single sample."""
    from tools.whisper_mlx.model import WhisperMLX
    from tools.whisper_mlx.streaming import StreamingWhisper, StreamingConfig

    print(f"Loading audio: {sample_path}")
    audio = load_audio(sample_path)
    duration_s = len(audio) / 16000
    print(f"Audio duration: {duration_s:.2f}s")
    print(f"Reference: {reference}")
    print()

    # Load model
    print("Loading model...")
    model = WhisperMLX.from_pretrained("mlx-community/whisper-large-v3-mlx")

    # OFFLINE: Direct transcription
    print("=" * 60)
    print("OFFLINE TRANSCRIPTION")
    print("=" * 60)
    result = model.transcribe(audio)
    offline_text = result.get("text", "").strip()
    print(f"Offline result: {offline_text}")
    print()

    # STREAMING: Async transcription
    print("=" * 60)
    print("STREAMING TRANSCRIPTION")
    print("=" * 60)

    config = StreamingConfig(
        use_local_agreement=True,
        latency_mode="balanced",
        emit_partials=True,
        use_vad=True,
    )
    streamer = StreamingWhisper(model, config)

    # Create async audio generator with small chunks
    chunk_duration_ms = 100  # 100ms chunks
    chunk_samples = int(chunk_duration_ms * 16 / 1000 * 1000)

    async def audio_generator():
        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i:i + chunk_samples]
            yield chunk

    # Collect all streaming results
    results = []
    final_text = ""
    partial_count = 0
    confirmed_texts = []

    async for result in streamer.transcribe_stream(audio_generator()):
        results.append(result)
        if result.is_final:
            final_text = result.text
            print(f"[FINAL] {result.text}")
        elif result.is_confirmed:
            confirmed_texts.append(result.confirmed_text)
            print(f"[CONFIRMED] {result.confirmed_text}")
        else:
            partial_count += 1
            if partial_count <= 5 or partial_count % 10 == 0:
                print(f"[PARTIAL #{partial_count}] {result.text[:80]}...")

    print()
    print(f"Total partials: {partial_count}")
    print(f"Final text: {final_text}")
    print()

    # Analysis
    print("=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    # Check if streaming matches offline
    offline_normalized = offline_text.lower().strip()
    streaming_normalized = final_text.lower().strip()

    if offline_normalized == streaming_normalized:
        print("MATCH: Streaming matches offline exactly!")
    else:
        print("MISMATCH:")
        print(f"  Offline:   {offline_text}")
        print(f"  Streaming: {final_text}")

        # Show what's missing
        offline_words = offline_normalized.split()
        streaming_words = streaming_normalized.split()
        print(f"  Offline words: {len(offline_words)}")
        print(f"  Streaming words: {len(streaming_words)}")

        # Find where they diverge
        for i, (o, s) in enumerate(zip(offline_words, streaming_words)):
            if o != s:
                print(f"  Diverges at word {i}: '{o}' vs '{s}'")
                break


if __name__ == "__main__":
    # Test on the worst-performing sample
    data_dir = Path(__file__).parent.parent / "data" / "LibriSpeech" / "dev-clean"
    sample_path = data_dir / "1272" / "135031" / "1272-135031-0001.flac"

    reference = "HE HAS GONE AND GONE FOR GOOD ANSWERED POLYCHROME WHO HAD MANAGED TO SQUEEZE INTO THE ROOM BESIDE THE DRAGON AND HAD WITNESSED THE OCCURRENCES WITH MUCH INTEREST"

    asyncio.run(debug_sample(str(sample_path), reference))
