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
Test WhisperMLX batch parallel chunks (OPT-W6) for long audio transcription.

This benchmark compares:
1. mlx-whisper sequential processing (baseline)
2. WhisperMLX batch parallel encoding
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    import mlx_whisper

    from tools.whisper_mlx import WhisperMLX

    # Test with long audio file
    test_audio = "tests/fixtures/audio_long/test_speech_95s.wav"

    if not Path(test_audio).exists():
        print(f"Test audio not found: {test_audio}")
        print("Create it with: ffmpeg -y -stream_loop 14 -i tests/fixtures/audio/test_speech.wav -c copy tests/fixtures/audio_long/test_speech_95s.wav")
        return 1

    # Get audio duration
    from tools.whisper_mlx.audio import SAMPLE_RATE, load_audio
    audio = load_audio(test_audio)
    duration = len(audio) / SAMPLE_RATE
    print(f"{'='*60}")
    print("WhisperMLX Batch Parallel Chunks (OPT-W6) Benchmark")
    print(f"{'='*60}")
    print(f"Audio: {test_audio}")
    print(f"Duration: {duration:.2f}s")
    print()

    # Load WhisperMLX model
    print("Loading WhisperMLX model...")
    model = WhisperMLX.from_pretrained("large-v3")
    print("Model loaded")
    print()

    # Benchmark 1: mlx-whisper (baseline)
    print(f"{'-'*60}")
    print("Test 1: mlx-whisper (baseline)")
    print(f"{'-'*60}")

    t0 = time.perf_counter()
    result_mlx = mlx_whisper.transcribe(test_audio, path_or_hf_repo="mlx-community/whisper-large-v3-mlx")
    mlx_time = time.perf_counter() - t0
    print(f"mlx-whisper: {mlx_time:.2f}s")
    print(f"Real-time factor: {mlx_time/duration:.3f}x")
    print(f"Segments: {len(result_mlx.get('segments', []))}")
    print()

    # Benchmark 2: WhisperMLX sequential (for comparison)
    print(f"{'-'*60}")
    print("Test 2: WhisperMLX sequential chunks")
    print(f"{'-'*60}")

    # Process chunks sequentially to get baseline
    t0 = time.perf_counter()
    sequential_time = 0
    chunk_length = 30.0
    n_chunks = int((duration + chunk_length - 1) / chunk_length)

    for i in range(n_chunks):
        chunk_start = int(i * (chunk_length - 1.0) * SAMPLE_RATE)
        chunk_end = min(chunk_start + int(chunk_length * SAMPLE_RATE), len(audio))
        chunk = audio[chunk_start:chunk_end]

        # Transcribe chunk
        t_chunk = time.perf_counter()
        _ = model.transcribe(chunk, verbose=False)
        sequential_time += time.perf_counter() - t_chunk

    print(f"WhisperMLX sequential: {sequential_time:.2f}s")
    print(f"Real-time factor: {sequential_time/duration:.3f}x")
    print()

    # Benchmark 3: WhisperMLX batch parallel (OPT-W6)
    print(f"{'-'*60}")
    print("Test 3: WhisperMLX batch parallel (OPT-W6)")
    print(f"{'-'*60}")

    for batch_size in [1, 2, 4]:
        t0 = time.perf_counter()
        result_batch = model.transcribe_long(
            test_audio,
            batch_size=batch_size,
            verbose=False,
        )
        batch_time = time.perf_counter() - t0

        print(f"  batch_size={batch_size}: {batch_time:.2f}s (RTF={batch_time/duration:.3f}x)")
        if batch_size == 1:
            _batch1_time = batch_time  # Reference time for batch_size=1

    print()

    # Benchmark 4: WhisperMLX batch parallel with verbose
    print(f"{'-'*60}")
    print("Test 4: WhisperMLX batch parallel (verbose)")
    print(f"{'-'*60}")

    t0 = time.perf_counter()
    result_batch = model.transcribe_long(
        test_audio,
        batch_size=4,
        verbose=True,
    )
    batch_time = time.perf_counter() - t0
    print(f"Total: {batch_time:.2f}s")
    print()

    # Results summary
    print(f"{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Audio duration: {duration:.2f}s")
    print(f"mlx-whisper: {mlx_time:.2f}s (RTF={mlx_time/duration:.3f}x)")
    print(f"WhisperMLX sequential: {sequential_time:.2f}s (RTF={sequential_time/duration:.3f}x)")
    print(f"WhisperMLX batch (bs=4): {batch_time:.2f}s (RTF={batch_time/duration:.3f}x)")
    print()
    print(f"Speedup vs mlx-whisper: {mlx_time/batch_time:.2f}x")
    print(f"Speedup vs sequential: {sequential_time/batch_time:.2f}x")
    print()

    # Text comparison
    print(f"{'-'*60}")
    print("Text Comparison (first 500 chars)")
    print(f"{'-'*60}")
    mlx_text = result_mlx.get("text", "")[:500]
    batch_text = result_batch.get("text", "")[:500]
    print(f"mlx-whisper: {mlx_text}...")
    print()
    print(f"WhisperMLX batch: {batch_text}...")
    print()

    # Check if texts are similar (allow for minor differences)
    mlx_words = set(mlx_text.lower().split())
    batch_words = set(batch_text.lower().split())
    overlap = len(mlx_words & batch_words) / max(len(mlx_words), 1)
    print(f"Word overlap: {overlap*100:.1f}%")

    if overlap > 0.8:
        print("✓ PASS: Batch transcription matches baseline")
    else:
        print("✗ FAIL: Batch transcription differs significantly from baseline")

    return 0


if __name__ == "__main__":
    sys.exit(main())
