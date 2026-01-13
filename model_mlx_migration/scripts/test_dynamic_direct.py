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
Direct test for OPT-W2: Dynamic Chunk Sizing

Uses model.decode() directly with variable-length encoder outputs.
"""

import os
import time


def get_audio_duration(audio_path: str) -> float:
    """Get audio duration using ffprobe."""
    import subprocess
    result = subprocess.run(
        ['ffprobe', '-i', audio_path, '-show_entries', 'format=duration',
         '-v', 'quiet', '-of', 'csv=p=0'],
        capture_output=True, text=True
    )
    return float(result.stdout.strip())


def transcribe_dynamic_direct(audio_path: str, model_path: str) -> tuple[str, float, int]:
    """
    Transcribe with dynamic encoder using direct model.decode().
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_whisper.audio import (
        SAMPLE_RATE,
        load_audio,
        log_mel_spectrogram,
        pad_or_trim,
    )
    from mlx_whisper.decoding import DecodingOptions
    from mlx_whisper.load_models import load_model

    # Load model
    dtype = mx.float16
    model = load_model(model_path, dtype=dtype)

    # Load audio
    audio = load_audio(audio_path)
    _audio_duration = audio.shape[0] / SAMPLE_RATE  # Duration in seconds
    mel = log_mel_spectrogram(audio, n_mels=model.dims.n_mels)
    mel_frames = mel.shape[0]

    # Calculate target frames (dynamic chunking)
    target_frames = ((mel_frames + 1) // 2) * 2
    min_frames = 200  # minimum for stable decoding
    target_frames = max(target_frames, min_frames)

    # Pad to target frames
    mel_dynamic = pad_or_trim(mel, target_frames, axis=-2).astype(dtype)

    # Store original positional embedding
    original_pos_emb = model.encoder._positional_embedding

    # Time the transcription
    start = time.perf_counter()

    # Create patched encoder forward
    def encode_dynamic(x):
        x = nn.gelu(model.encoder.conv1(x))
        x = nn.gelu(model.encoder.conv2(x))
        seq_len = x.shape[1]
        pos_emb = original_pos_emb[:seq_len]
        x = x + pos_emb
        for block in model.encoder.blocks:
            x, _, _ = block(x)
        x = model.encoder.ln_post(x)
        return x

    # Encode
    audio_features = encode_dynamic(mel_dynamic[None])
    mx.eval(audio_features)

    # Create decoding options
    options = DecodingOptions(language="en", fp16=True)

    # Call model.decode() with our mel spectrogram
    # But first we need to patch the encoder so it uses our dynamic forward
    original_call = type(model.encoder).__call__

    def patched_encoder_call(self, x):
        x = nn.gelu(self.conv1(x))
        x = nn.gelu(self.conv2(x))
        seq_len = x.shape[1]
        pos_emb = original_pos_emb[:seq_len]
        x = x + pos_emb
        for block in self.blocks:
            x, _, _ = block(x)
        x = self.ln_post(x)
        return x

    try:
        type(model.encoder).__call__ = patched_encoder_call

        # Call model.decode with our dynamic mel
        result = model.decode(mel_dynamic, options)

        elapsed = time.perf_counter() - start

        return result.text.strip(), elapsed, target_frames

    finally:
        type(model.encoder).__call__ = original_call


def main():
    print("=" * 60)
    print("OPT-W2: Dynamic Chunk Sizing - Direct Decode Test")
    print("=" * 60)
    print()

    import mlx_whisper

    model_path = "mlx-community/whisper-large-v3-turbo"

    # Test with real speech samples
    test_files = [
        "tests/prosody/contour_vs_v5_output/neutral_context_baseline.wav",
        "tests/prosody/contour_vs_v5_output/angry_context_baseline.wav",
    ]

    results = []

    for audio_path in test_files:
        if not os.path.exists(audio_path):
            print(f"Skipping {audio_path} - not found")
            continue

        print(f"\n--- {os.path.basename(audio_path)} ---")
        duration = get_audio_duration(audio_path)
        print(f"  Duration: {duration:.2f}s")

        # Standard transcription
        start = time.perf_counter()
        result_std = mlx_whisper.transcribe(
            audio_path, path_or_hf_repo=model_path, language="en"
        )
        time_std = time.perf_counter() - start
        text_std = result_std['text'].strip()
        print(f"  Standard: {time_std*1000:.0f}ms - '{text_std}'")

        # Dynamic transcription
        try:
            text_dyn, time_dyn, frames = transcribe_dynamic_direct(
                audio_path, model_path
            )
            print(f"  Dynamic:  {time_dyn*1000:.0f}ms ({frames} frames) - '{text_dyn}'")

            speedup = time_std / time_dyn if time_dyn > 0 else 0
            print(f"  Speedup: {speedup:.2f}x")

            match = text_std.lower() == text_dyn.lower()
            print(f"  Match: {'YES' if match else 'NO'}")

            results.append({
                'file': os.path.basename(audio_path),
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

        import numpy as np
        all_match = all(r['match'] for r in results)
        avg_speedup = np.mean([r['speedup'] for r in results])

        print(f"\nAll transcriptions match: {'YES' if all_match else 'NO'}")
        print(f"Average speedup: {avg_speedup:.2f}x")

        if all_match and avg_speedup > 1.2:
            print("\nOPT-W2 VALIDATED: Dynamic chunking is LOSSLESS and provides speedup")
        elif all_match:
            print("\nOPT-W2 is lossless but speedup is minimal")
        else:
            print("\nOPT-W2 produces different outputs - investigation needed")


if __name__ == "__main__":
    main()
