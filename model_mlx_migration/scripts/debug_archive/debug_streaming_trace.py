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
Debug script to trace streaming transcription behavior.

Identifies root causes of:
1. RTF > 1 (why is streaming slower than real-time?)
2. Content loss (why are some samples missing text?)
3. High latency (where is time being spent?)
"""

import asyncio
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.whisper_mlx.streaming_eval import compute_wer
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


def get_sample(data_dir: Path, sample_idx: int = 0):
    """Get a specific sample from LibriSpeech dev-clean."""
    dev_clean = data_dir / "LibriSpeech" / "dev-clean"

    idx = 0
    for speaker_dir in sorted(dev_clean.iterdir()):
        if not speaker_dir.is_dir():
            continue
        for chapter_dir in sorted(speaker_dir.iterdir()):
            if not chapter_dir.is_dir():
                continue
            trans_files = list(chapter_dir.glob("*.trans.txt"))
            if not trans_files:
                continue
            transcripts = parse_librispeech_transcripts(trans_files[0])
            for audio_file in sorted(chapter_dir.glob("*.flac")):
                sample_id = audio_file.stem
                if sample_id not in transcripts:
                    continue
                if idx == sample_idx:
                    audio = load_audio(str(audio_file))
                    return audio, transcripts[sample_id], sample_id
                idx += 1
    return None, None, None


async def trace_streaming(audio: np.ndarray, reference: str, sample_id: str):
    """Trace streaming transcription with detailed timing."""
    from tools.whisper_mlx.model import WhisperMLX
    from tools.whisper_mlx.streaming import StreamingWhisper, StreamingConfig

    print(f"\n{'='*70}")
    print(f"TRACING: {sample_id}")
    print(f"{'='*70}")
    print(f"Audio duration: {len(audio)/16000:.2f}s")
    print(f"Reference: {reference[:100]}...")
    print()

    # Load model (cached)
    model = WhisperMLX.from_pretrained("mlx-community/whisper-large-v3-mlx")

    config = StreamingConfig(
        use_local_agreement=True,
        latency_mode="balanced",
        emit_partials=True,
        partial_interval=0.5,
        min_chunk_duration=0.5,
        max_chunk_duration=10.0,
        silence_threshold_duration=0.5,
    )

    streamer = StreamingWhisper(model, config)

    # Track events
    events = []
    finals_collected = []
    chunk_count = 0

    # Audio generator with timing
    chunk_ms = 100  # 100ms chunks
    chunk_samples = int(16000 * chunk_ms / 1000)
    num_chunks = (len(audio) + chunk_samples - 1) // chunk_samples

    start_time = time.perf_counter()

    async def audio_generator():
        nonlocal chunk_count
        for i in range(num_chunks):
            chunk_start = i * chunk_samples
            chunk_end = min((i + 1) * chunk_samples, len(audio))
            chunk = audio[chunk_start:chunk_end]
            chunk_count += 1
            yield chunk

    # Process
    result_count = 0
    async for result in streamer.transcribe_stream(audio_generator()):
        now = time.perf_counter()
        elapsed = now - start_time

        event = {
            "idx": result_count,
            "time_s": elapsed,
            "is_final": result.is_final,
            "is_partial": result.is_partial,
            "text_len": len(result.text),
            "text_preview": result.text[:50] if result.text else "",
            "rtf": result.rtf,
        }
        events.append(event)

        event_type = "FINAL" if result.is_final else ("PARTIAL" if result.is_partial else "OTHER")
        print(f"  [{elapsed:6.2f}s] {event_type:8s} RTF={result.rtf:.2f} len={len(result.text):3d} "
              f"'{result.text[:40]}...'")

        if result.is_final:
            finals_collected.append(result.text)

        result_count += 1

    total_time = time.perf_counter() - start_time
    audio_duration = len(audio) / 16000

    print()
    print("--- Summary ---")
    print(f"Chunks fed: {chunk_count}")
    print(f"Results received: {result_count}")
    print(f"Finals: {len(finals_collected)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Audio duration: {audio_duration:.2f}s")
    print(f"Overall RTF: {total_time/audio_duration:.2f}")
    print()

    # Concatenate finals
    final_text = " ".join(finals_collected)
    print("--- Final Output ---")
    print(f"Concatenated ({len(finals_collected)} segments):")
    print(f"  '{final_text}'")
    print()

    # Compare to reference
    wer_result = compute_wer(reference, final_text)
    print("--- Quality ---")
    print(f"WER: {wer_result['wer']*100:.1f}%")
    print(f"Ref words: {wer_result['ref_words']}")
    print(f"Hyp words: {wer_result['hyp_words']}")
    print()

    # Also run offline for comparison
    print("--- Offline Comparison ---")
    start = time.perf_counter()
    offline_result = model.transcribe(audio, language=None, task="transcribe")
    offline_time = time.perf_counter() - start
    offline_text = offline_result.get("text", "").strip()
    offline_wer = compute_wer(reference, offline_text)

    print(f"Offline text: '{offline_text}'")
    print(f"Offline WER: {offline_wer['wer']*100:.1f}%")
    print(f"Offline time: {offline_time:.2f}s (RTF={offline_time/audio_duration:.2f})")

    return {
        "sample_id": sample_id,
        "audio_duration": audio_duration,
        "streaming_time": total_time,
        "streaming_rtf": total_time / audio_duration,
        "streaming_wer": wer_result["wer"],
        "offline_time": offline_time,
        "offline_rtf": offline_time / audio_duration,
        "offline_wer": offline_wer["wer"],
        "num_finals": len(finals_collected),
    }


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0, help="Sample index to trace")
    parser.add_argument("--num-samples", type=int, default=3, help="Number of samples")
    args = parser.parse_args()

    data_dir = Path(__file__).parent.parent / "data"

    results = []
    for i in range(args.sample, args.sample + args.num_samples):
        audio, reference, sample_id = get_sample(data_dir, i)
        if audio is None:
            print(f"Sample {i} not found")
            continue

        result = await trace_streaming(audio, reference, sample_id)
        results.append(result)

    print("\n" + "="*70)
    print("AGGREGATE RESULTS")
    print("="*70)
    for r in results:
        print(f"{r['sample_id']}: streaming RTF={r['streaming_rtf']:.2f} WER={r['streaming_wer']*100:.1f}%  |  "
              f"offline RTF={r['offline_rtf']:.2f} WER={r['offline_wer']*100:.1f}%  |  "
              f"finals={r['num_finals']}")


if __name__ == "__main__":
    asyncio.run(main())
