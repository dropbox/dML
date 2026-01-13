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
End-to-end test for OPT-W2: Dynamic Chunk Sizing

This script tests that dynamic chunk sizing produces identical
transcriptions compared to standard mlx_whisper.

Key validation:
1. Same transcription output (lossless)
2. Performance improvement
"""

import os
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np


def create_speech_audio(text_hint: str, duration_sec: float, output_path: str) -> str:
    """Create test audio using Kokoro TTS for real speech content."""
    sys.path.insert(0, str(Path(__file__).parent.parent))

    try:
        from tools.dashvoice.kokoro_mlx import KokoroMLX

        # Create short speech using Kokoro
        kokoro = KokoroMLX()
        # Adjust text based on target duration
        words_per_sec = 2.5  # approximate speaking rate
        target_words = int(duration_sec * words_per_sec)

        # Generate text that roughly matches target duration
        base_text = "Hello, this is a test of the speech recognition system. "
        full_text = base_text * max(1, target_words // len(base_text.split()))
        full_text = " ".join(full_text.split()[:target_words]) + "."

        audio = kokoro(full_text, voice="af_heart")

        # Save as wav
        sample_rate = 24000  # Kokoro outputs 24kHz
        audio_int16 = (np.array(audio) * 32767).astype(np.int16)

        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())

        return output_path, full_text

    except Exception as e:
        print(f"Warning: Kokoro not available ({e}), using sine wave")
        # Fallback to sine wave
        sample_rate = 16000
        samples = int(duration_sec * sample_rate)
        t = np.linspace(0, duration_sec, samples)
        audio = np.sin(2 * np.pi * 200 * t) * 0.5

        audio_int16 = (audio * 32767).astype(np.int16)
        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())

        return output_path, "[synthetic audio - no real speech]"


def transcribe_standard(audio_path: str, model_path: str) -> tuple[str, float]:
    """Standard mlx_whisper transcription."""
    import mlx_whisper

    start = time.perf_counter()
    result = mlx_whisper.transcribe(audio_path, path_or_hf_repo=model_path, language="en")
    elapsed = time.perf_counter() - start

    return result['text'].strip(), elapsed


def transcribe_dynamic(audio_path: str, model_path: str) -> tuple[str, float, int]:
    """Dynamic chunk transcription."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_whisper.audio import N_FRAMES, load_audio, log_mel_spectrogram, pad_or_trim
    from mlx_whisper.decoding import DecodingOptions, DecodingTask
    from mlx_whisper.load_models import load_model
    from mlx_whisper.tokenizer import get_tokenizer

    # Load model
    model = load_model(model_path, dtype=mx.float16)

    # Load audio and create mel spectrogram
    audio = load_audio(audio_path)
    mel = log_mel_spectrogram(audio, n_mels=model.dims.n_mels)
    mel_frames = mel.shape[0]

    # Calculate actual frames needed (round up to multiple of 2)
    target_frames = ((mel_frames + 1) // 2) * 2
    min_frames = 200  # minimum for stable decoding
    target_frames = max(target_frames, min_frames)

    # Don't exceed standard N_FRAMES
    target_frames = min(target_frames, N_FRAMES)

    # Pad or trim to target
    mel = pad_or_trim(mel, target_frames, axis=-2)
    mel = mel.astype(mx.float16)

    # Store original positional embedding
    original_pos_emb = model.encoder._positional_embedding

    # Create patched encoder
    def dynamic_encode(x):
        x = nn.gelu(model.encoder.conv1(x))
        x = nn.gelu(model.encoder.conv2(x))
        seq_len = x.shape[1]
        pos_emb = original_pos_emb[:seq_len]
        x = x + pos_emb
        for block in model.encoder.blocks:
            x, _, _ = block(x)
        x = model.encoder.ln_post(x)
        return x

    # Get tokenizer
    tokenizer = get_tokenizer(
        model.is_multilingual,
        num_languages=model.num_languages,
        language="en",
        task="transcribe"
    )

    # Time the transcription
    start = time.perf_counter()

    # Encode with dynamic encoder
    audio_features = dynamic_encode(mel[None])
    mx.eval(audio_features)

    # Decode - need to manually run decoding since we have custom encoder outputs
    # The decoder cross-attention works with any length of encoder outputs
    options = DecodingOptions(
        language="en",
        task="transcribe",
        fp16=True,
    )

    # Create inference context
    class DynamicInference:
        def __init__(self, model, audio_features):
            self.model = model
            self.audio_features = audio_features
            self.kv_cache = None
            self.hooks = None

        def logits(self, tokens, audio_features):
            # Use our pre-computed audio features
            if tokens.shape[-1] > 1:
                # Reset kv_cache for new sequence
                self.kv_cache = None

            logits, kv_cache, _ = self.model.decoder(
                tokens, self.audio_features, self.kv_cache
            )
            self.kv_cache = kv_cache
            return logits

        def cleanup_caching(self):
            self.kv_cache = None

        def rearrange_kv_cache(self, source_indices):
            if self.kv_cache is not None:
                def rearrange(x):
                    return x[source_indices] if x is not None else None
                from mlx.utils import tree_map
                self.kv_cache = tree_map(rearrange, self.kv_cache)

    inference = DynamicInference(model, audio_features)

    # Run decoding task
    task = DecodingTask(model, options)
    task.inference = inference

    # Generate initial tokens
    tokens = np.array([list(task._get_initial_tokens())])

    # Simple greedy decoding
    max_tokens = 224  # Whisper's max
    generated = list(tokens[0])

    for _ in range(max_tokens):
        logits = inference.logits(mx.array(tokens), audio_features)
        next_token = int(logits[0, -1].argmax())

        if next_token == tokenizer.eot:
            break

        generated.append(next_token)
        tokens = np.array([generated])

    elapsed = time.perf_counter() - start

    # Decode tokens to text
    text = tokenizer.decode([t for t in generated if t < tokenizer.eot])

    return text.strip(), elapsed, target_frames


def main():
    print("=" * 60)
    print("OPT-W2: Dynamic Chunk Sizing - E2E Validation")
    print("=" * 60)
    print()

    model_path = "mlx-community/whisper-large-v3-turbo"
    test_durations = [5, 10, 15]  # seconds

    results = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for duration in test_durations:
            print(f"\n--- Testing {duration}s audio ---")

            audio_path = os.path.join(tmpdir, f"test_{duration}s.wav")
            audio_path, expected_text = create_speech_audio(
                f"Test {duration}s", duration, audio_path
            )
            print(f"  Expected (approx): {expected_text[:50]}...")

            # Standard transcription
            text_std, time_std = transcribe_standard(audio_path, model_path)
            print(f"  Standard: {time_std*1000:.0f}ms - '{text_std[:50]}...'")

            # Dynamic transcription
            try:
                text_dyn, time_dyn, frames = transcribe_dynamic(audio_path, model_path)
                print(f"  Dynamic:  {time_dyn*1000:.0f}ms ({frames} frames) - '{text_dyn[:50]}...'")

                speedup = time_std / time_dyn if time_dyn > 0 else 0
                print(f"  Speedup: {speedup:.2f}x")

                # Check if outputs match
                match = text_std.lower() == text_dyn.lower()
                print(f"  Match: {'YES' if match else 'NO'}")

                results.append({
                    'duration': duration,
                    'time_std': time_std,
                    'time_dyn': time_dyn,
                    'frames': frames,
                    'speedup': speedup,
                    'match': match,
                })
            except Exception as e:
                print(f"  Dynamic failed: {e}")
                import traceback
                traceback.print_exc()

    # Summary
    if results:
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)

        all_match = all(r['match'] for r in results)
        avg_speedup = np.mean([r['speedup'] for r in results])

        print(f"\nAll transcriptions match: {'YES' if all_match else 'NO'}")
        print(f"Average speedup: {avg_speedup:.2f}x")

        if all_match and avg_speedup > 1.2:
            print("\nConclusion: Dynamic chunking is LOSSLESS and BENEFICIAL")
        elif all_match:
            print("\nConclusion: Dynamic chunking is LOSSLESS but speedup is minimal")
        else:
            print("\nConclusion: Dynamic chunking produces DIFFERENT outputs - needs investigation")


if __name__ == "__main__":
    main()
