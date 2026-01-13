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
Simple test for OPT-W2: Dynamic Chunk Sizing

Uses actual speech audio and patches only the encoder.
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


def transcribe_with_patched_encoder(audio_path: str, model_path: str) -> tuple[str, float, int]:
    """
    Transcribe with patched encoder that handles variable-length input.
    Uses mlx_whisper's decode() but with a patched encoder.
    """
    import mlx.core as mx
    import mlx.nn as nn
    import mlx_whisper
    from mlx_whisper.audio import (
        N_FRAMES,
        SAMPLE_RATE,
        load_audio,
        log_mel_spectrogram,
        pad_or_trim,
    )
    from mlx_whisper.decoding import DecodingOptions
    from mlx_whisper.transcribe import ModelHolder

    # Get model from holder to share with mlx_whisper.transcribe
    dtype = mx.float16
    model = ModelHolder.get_model(model_path, dtype)

    # Load audio and create mel spectrogram
    audio = load_audio(audio_path)
    _audio_duration = audio.shape[0] / SAMPLE_RATE  # Used for documentation/debugging
    mel = log_mel_spectrogram(audio, n_mels=model.dims.n_mels)
    mel_frames = mel.shape[0]

    # Calculate target frames (dynamic chunking)
    target_frames = ((mel_frames + 1) // 2) * 2
    min_frames = 200  # minimum for stable decoding
    target_frames = max(target_frames, min_frames)

    # If audio is close to or longer than 30s, use standard path
    if target_frames >= N_FRAMES * 0.9:
        result = mlx_whisper.transcribe(audio_path, path_or_hf_repo=model_path, language="en")
        return result['text'].strip(), 0, N_FRAMES

    # Pad to target frames (not full N_FRAMES)
    mel_dynamic = pad_or_trim(mel, target_frames, axis=-2).astype(dtype)

    # Store original encoder call
    original_call = type(model.encoder).__call__

    # Create patched encoder that slices positional embeddings
    original_pos_emb = model.encoder._positional_embedding

    def patched_call(self, x):
        x = nn.gelu(self.conv1(x))
        x = nn.gelu(self.conv2(x))
        seq_len = x.shape[1]
        # Slice positional embedding to match actual sequence length
        pos_emb = original_pos_emb[:seq_len]
        x = x + pos_emb
        for block in self.blocks:
            x, _, _ = block(x)
        x = self.ln_post(x)
        return x

    # Time the transcription
    start = time.perf_counter()

    try:
        # Patch the encoder class method
        type(model.encoder).__call__ = patched_call

        # Encode with dynamic length
        audio_features = model.encoder(mel_dynamic[None])
        mx.eval(audio_features)

        # The audio_features shape is now (1, seq_len/2, n_state)
        # For decode to work, we need to pass these features
        # But decode() expects shape (n_audio_ctx, n_audio_state)...

        # Actually decode() checks the shape and calls encoder if not matching
        # We need to call the internal decode directly

        # Let's use the DecodingOptions and run decode manually
        from mlx_whisper.decoding import DecodingTask

        options = DecodingOptions(language="en", fp16=True)
        task = DecodingTask(model, options)

        # Override the audio features getter
        _original_get_audio_features = task._get_audio_features  # Saved for restoration if needed
        task._get_audio_features = lambda mel: audio_features

        # Run the decoding
        result = task.run(mel_dynamic)

        elapsed = time.perf_counter() - start

        text = result[0].text if result else ""
        return text.strip(), elapsed, target_frames

    finally:
        # Restore original encoder
        type(model.encoder).__call__ = original_call


def main():
    print("=" * 60)
    print("OPT-W2: Dynamic Chunk Sizing - Simple Test")
    print("=" * 60)
    print()

    import mlx_whisper

    model_path = "mlx-community/whisper-large-v3-turbo"

    # Test with real speech samples
    test_files = [
        "tests/prosody/contour_vs_v5_output/neutral_context_baseline.wav",
        "tests/prosody/contour_vs_v5_output/angry_context_baseline.wav",
    ]

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
            text_dyn, time_dyn, frames = transcribe_with_patched_encoder(
                audio_path, model_path
            )
            print(f"  Dynamic:  {time_dyn*1000:.0f}ms ({frames} frames) - '{text_dyn}'")

            if time_dyn > 0:
                speedup = time_std / time_dyn
                print(f"  Speedup: {speedup:.2f}x")

                match = text_std.lower() == text_dyn.lower()
                print(f"  Match: {'YES' if match else 'NO'}")
        except Exception as e:
            print(f"  Dynamic failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
