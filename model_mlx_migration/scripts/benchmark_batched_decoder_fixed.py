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
Benchmark C2: Batched Decoder Throughput (Fixed)

Compares sequential transcription vs batched transcription for
multiple audio files, bypassing VAD to isolate decoder performance.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import numpy as np

from tools.whisper_mlx import WhisperMLX
from tools.whisper_mlx.audio import load_audio, log_mel_spectrogram
from tools.whisper_mlx.tokenizer import get_whisper_tokenizer


def prepare_mel(audio: np.ndarray, n_mels: int, n_audio_ctx: int) -> mx.array:
    """Prepare mel spectrogram for transcription."""
    mel = log_mel_spectrogram(audio, n_mels=n_mels)
    target_len = n_audio_ctx * 2
    if mel.shape[0] < target_len:
        mel = mx.pad(mel, [(0, target_len - mel.shape[0]), (0, 0)])
    elif mel.shape[0] > target_len:
        mel = mel[:target_len]
    return mel


def transcribe_sequential(model: WhisperMLX, audio_list: list, tokenizer) -> list:
    """Transcribe audio sequentially (one at a time)."""
    results = []
    for audio in audio_list:
        mel = prepare_mel(audio, model.config.n_mels, model.config.n_audio_ctx)
        mel = mel[None]  # Add batch dim
        features = model.embed_audio(mel)
        tokens, segments, _, _ = model._decode_with_metrics(features, tokenizer, temperature=0.0)
        text = tokenizer.decode(tokens).strip()
        results.append({"text": text, "segments": segments})
    return results


def transcribe_batched(model: WhisperMLX, audio_list: list, tokenizer) -> list:
    """Transcribe audio in batch."""
    # Prepare all mels
    mels = []
    for audio in audio_list:
        mel = prepare_mel(audio, model.config.n_mels, model.config.n_audio_ctx)
        mels.append(mel)
    mel_batch = mx.stack(mels)

    # Encode all at once
    features = model.embed_audio(mel_batch)
    mx.eval(features)

    # Decode all at once
    batch_results = model._decode_batch(features, tokenizer, temperature=0.0)

    # Format results
    results = []
    for tokens, segments in batch_results:
        text = tokenizer.decode(tokens).strip()
        results.append({"text": text, "segments": segments})
    return results


def benchmark(model, audio_list, tokenizer, mode, n_runs=3):
    """Benchmark transcription mode."""
    func = transcribe_sequential if mode == "sequential" else transcribe_batched
    times = []

    for run in range(n_runs):
        start = time.perf_counter()
        results = func(model, audio_list, tokenizer)
        mx.eval()  # Ensure all computation is done
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Run {run + 1}: {elapsed:.3f}s")

    return {
        "times": times,
        "avg_time": np.mean(times),
        "std_time": np.std(times),
        "results": results,
    }


def main():
    print("=" * 60)
    print("C2: Batched Decoder Throughput Benchmark (Fixed)")
    print("=" * 60)

    # Load model
    print("\nLoading WhisperMLX model (base)...")
    model = WhisperMLX.from_pretrained("base")
    print(f"Model: {model.config.name}, {model.config.n_text_layer} decoder layers")

    # Get tokenizer
    tokenizer = get_whisper_tokenizer(multilingual=True, language="en", task="transcribe")

    # Find LibriSpeech audio files
    librispeech_path = Path("/Users/ayates/model_mlx_migration/data/benchmarks/librispeech/LibriSpeech/test-clean")
    flac_files = sorted(librispeech_path.glob("*/*/*.flac"))

    if len(flac_files) < 8:
        print("Not enough LibriSpeech files found")
        return

    # Test with different batch sizes
    batch_sizes = [2, 4, 8]

    # Warm up
    print("\nWarming up...")
    audio = load_audio(str(flac_files[0]))
    mel = prepare_mel(audio, model.config.n_mels, model.config.n_audio_ctx)[None]
    _ = model.embed_audio(mel)
    _ = model._decode_with_metrics(model.embed_audio(mel), tokenizer, temperature=0.0)
    mx.eval()

    for batch_size in batch_sizes:
        print(f"\n{'=' * 60}")
        print(f"Batch size: {batch_size}")
        print("=" * 60)

        # Load audio files
        audio_list = []
        for i in range(batch_size):
            audio = load_audio(str(flac_files[i]))
            audio_list.append(audio)
            dur = len(audio) / 16000
            print(f"  Audio {i}: {dur:.2f}s")

        # Sequential benchmark
        print("\nSequential transcription:")
        seq_result = benchmark(model, audio_list, tokenizer, "sequential", n_runs=3)

        # Batched benchmark
        print("\nBatched transcription:")
        batch_result = benchmark(model, audio_list, tokenizer, "batched", n_runs=3)

        # Calculate speedup
        speedup = seq_result["avg_time"] / batch_result["avg_time"]
        throughput_seq = batch_size / seq_result["avg_time"]
        throughput_batch = batch_size / batch_result["avg_time"]

        print("\nResults:")
        print(f"  Sequential: {seq_result['avg_time']:.3f}s +/- {seq_result['std_time']:.3f}s")
        print(f"  Batched:    {batch_result['avg_time']:.3f}s +/- {batch_result['std_time']:.3f}s")
        print(f"  Speedup:    {speedup:.2f}x")
        print(f"  Throughput: {throughput_seq:.2f} -> {throughput_batch:.2f} audio/s")

        # Compare transcriptions
        print("\nTranscription comparison:")
        for i, (seq_res, batch_res) in enumerate(zip(seq_result["results"], batch_result["results"])):
            match = "MATCH" if seq_res["text"] == batch_res["text"] else "DIFFER"
            print(f"  [{i}] {match}: {seq_res['text'][:60]}...")

    print("\n" + "=" * 60)
    print("Benchmark complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
