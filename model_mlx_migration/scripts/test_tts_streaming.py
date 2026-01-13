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
Test streaming TTS to measure TTFA improvement.

This script compares:
1. Non-streaming TTS (full text → full audio)
2. Streaming TTS (text → chunked audio with lower TTFA)
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


def test_non_streaming():
    """Test non-streaming TTS (full text → full audio)."""
    print("=" * 60)
    print("Non-Streaming TTS (baseline)")
    print("=" * 60)

    from mlx_audio.tts.utils import load_model

    # Load model (should be cached)
    model = load_model("prince-canuma/Kokoro-82M")

    # Warmup
    print("Warming up...")
    for _ in model.generate(text="Hello.", voice="af_bella", speed=1.0, verbose=False):
        pass

    # Test with multi-sentence text
    test_text = "Hello, how are you today? I hope you're having a great day. The weather is beautiful outside."

    print(f"\nText: {test_text}")
    print(f"Length: {len(test_text)} chars\n")

    # Non-streaming: must wait for all audio
    start = time.perf_counter()
    result = None
    for r in model.generate(
        text=test_text,
        voice="af_bella",
        speed=1.0,
        verbose=False
    ):
        result = r
    total_time = (time.perf_counter() - start) * 1000

    audio = np.array(result.audio)
    duration = len(audio) / result.sample_rate

    print(f"Total generation time: {total_time:.0f}ms")
    print(f"Audio duration: {duration:.2f}s")
    print(f"TTFA (time to first audio): {total_time:.0f}ms (same as total)")
    print(f"RTF: {(total_time/1000)/duration:.4f}")

    return total_time, duration


def test_streaming():
    """Test streaming TTS (chunked audio with lower TTFA)."""
    print("\n" + "=" * 60)
    print("Streaming TTS (chunked)")
    print("=" * 60)

    from mlx_audio.tts.utils import load_model

    # Load model (should be cached)
    model = load_model("prince-canuma/Kokoro-82M")

    # Warmup
    print("Warming up...")
    for _ in model.generate(text="Hello.", voice="af_bella", speed=1.0, verbose=False):
        pass

    # Test with multi-sentence text
    test_text = "Hello, how are you today? I hope you're having a great day. The weather is beautiful outside."

    print(f"\nText: {test_text}")
    print(f"Length: {len(test_text)} chars\n")

    # Streaming: split on sentences and stream chunks
    start = time.perf_counter()
    chunks = []
    first_chunk_time = None

    for result in model.generate(
        text=test_text,
        voice="af_bella",
        speed=1.0,
        split_pattern=r"[.!?]+\s*",  # Split on sentences
        verbose=False
    ):
        chunk_time = time.perf_counter() - start
        if first_chunk_time is None:
            first_chunk_time = chunk_time * 1000

        audio = np.array(result.audio)
        duration = len(audio) / result.sample_rate

        chunks.append({
            "elapsed_ms": chunk_time * 1000,
            "samples": len(audio),
            "duration_s": duration,
            "text": result.graphemes if hasattr(result, 'graphemes') else "",
        })

    total_time = (time.perf_counter() - start) * 1000
    total_duration = sum(c["duration_s"] for c in chunks)

    print(f"Chunks generated: {len(chunks)}")
    print("-" * 40)
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {chunk['elapsed_ms']:.0f}ms, {chunk['duration_s']:.2f}s audio")
    print("-" * 40)
    print(f"\nTotal generation time: {total_time:.0f}ms")
    print(f"Audio duration: {total_duration:.2f}s")
    print(f"TTFA (time to first audio): {first_chunk_time:.0f}ms")
    print(f"RTF: {(total_time/1000)/total_duration:.4f}")

    return total_time, total_duration, first_chunk_time


def compare_results():
    """Compare streaming vs non-streaming performance."""
    print("\n" + "=" * 60)
    print("Comparison")
    print("=" * 60)

    non_stream_time, non_stream_duration = test_non_streaming()
    stream_time, stream_duration, stream_ttfa = test_streaming()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"\n{'Metric':<25} {'Non-Streaming':<15} {'Streaming':<15}")
    print("-" * 55)
    print(f"{'Total time (ms)':<25} {non_stream_time:<15.0f} {stream_time:<15.0f}")
    print(f"{'TTFA (ms)':<25} {non_stream_time:<15.0f} {stream_ttfa:<15.0f}")
    print(f"{'TTFA reduction':<25} {'-':<15} {((non_stream_time - stream_ttfa) / non_stream_time * 100):.1f}%")
    print(f"{'Perceived improvement':<25} {'-':<15} {non_stream_time / stream_ttfa:.1f}x faster")


def test_various_lengths():
    """Test streaming with various text lengths."""
    print("\n" + "=" * 60)
    print("Streaming TTFA by Text Length")
    print("=" * 60)

    from mlx_audio.tts.utils import load_model

    model = load_model("prince-canuma/Kokoro-82M")

    # Warmup
    for _ in model.generate(text="Hello.", voice="af_bella", speed=1.0, verbose=False):
        pass

    test_cases = [
        ("Short (1 sentence)", "Hello, how are you today?"),
        ("Medium (3 sentences)", "Hello, how are you today? I hope you're doing well. Let me know if you need anything."),
        ("Long (5 sentences)", "Hello, how are you today? I hope you're doing well. The weather is beautiful outside. Let's go for a walk later. It will be fun!"),
    ]

    print(f"\n{'Case':<20} {'Chars':<8} {'TTFA (ms)':<12} {'Total (ms)':<12} {'Chunks':<8}")
    print("-" * 60)

    for name, text in test_cases:
        start = time.perf_counter()
        first_chunk_time = None
        chunks = 0

        for result in model.generate(
            text=text,
            voice="af_bella",
            speed=1.0,
            split_pattern=r"[.!?]+\s*",
            verbose=False
        ):
            if first_chunk_time is None:
                first_chunk_time = (time.perf_counter() - start) * 1000
            chunks += 1

        total_time = (time.perf_counter() - start) * 1000
        print(f"{name:<20} {len(text):<8} {first_chunk_time:<12.0f} {total_time:<12.0f} {chunks:<8}")


if __name__ == "__main__":
    try:
        compare_results()
        test_various_lengths()
        print("\n" + "=" * 60)
        print("All streaming tests completed")
        print("=" * 60)
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure mlx_audio is installed: pip install mlx_audio")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
