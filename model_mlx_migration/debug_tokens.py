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

"""Debug script to extract tokens step-by-step from Python decoder."""

import sys

sys.path.insert(0, 'tools')
import mlx.core as mx
import numpy as np
from whisper_mlx.audio import N_SAMPLES, load_audio, log_mel_spectrogram, pad_or_trim
from whisper_mlx.model import WhisperMLX, preprocess_audio_with_vad
from whisper_mlx.tokenizer import get_whisper_tokenizer

# Load model
model = WhisperMLX.from_pretrained('large-v3-turbo')
tokenizer = get_whisper_tokenizer(multilingual=True, num_languages=99, language="en", task="transcribe")

# Load audio
audio = load_audio('data/librispeech/dev-clean/1272/128104/1272-128104-0004.flac')
print(f"Audio length: {len(audio)} samples ({len(audio)/16000:.2f}s)")

# Apply VAD preprocessing (same as C++)
print("\n=== APPLYING VAD ===")
vad_audio, vad_result = preprocess_audio_with_vad(audio, aggressiveness=2, padding_ms=50)
print(f"VAD: {vad_result.speech_ratio*100:.1f}% speech, {len(vad_result.segments)} segments")
print(f"VAD audio: {len(vad_audio)} samples ({len(vad_audio)/16000:.2f}s)")
audio = vad_audio  # Use VAD-processed audio

# Pad to 30s like whisper expects
audio_padded = pad_or_trim(audio, N_SAMPLES)
print(f"Audio padded: {len(audio_padded)} samples ({len(audio_padded)/16000:.2f}s)")

# Compute mel spectrogram
mel = log_mel_spectrogram(audio_padded)
print(f"Mel shape: {mel.shape}")

# Add batch dimension
mel = mel[None, ...]  # (1, 3000, 128)
print(f"Mel batched shape: {mel.shape}")

# Get encoder output
encoder_out = model.embed_audio(mel)
print(f"Encoder output shape: {encoder_out.shape}")

# Run decode with debug
initial_tokens = [50258, 50259, 50359, 50364]  # SOT, en, transcribe, <|0.00|>
tokens = mx.array([initial_tokens])
sample_begin = len(initial_tokens)

# Suppress tokens (standard Whisper approach)
suppress_tokens = [220, 50257]  # " " (blank) and EOT at start

# KV cache
kv_cache = None

# Decode step by step
max_tokens = 100
all_tokens = list(initial_tokens)

print("\n=== DECODE STEP BY STEP ===")
print(f"Initial tokens: {initial_tokens}")

for i in range(max_tokens):
    # Forward pass (decoder returns: logits, kv_cache, cross_attentions, hidden_states)
    logits, kv_cache, _, _ = model.decoder(tokens, encoder_out, kv_cache=kv_cache)

    # Suppress blank/EOT at first step
    if i == 0:
        for t in suppress_tokens:
            logits = logits.at[0, -1, t].add(-float('inf'))

    # Get next token (greedy)
    next_token = mx.argmax(logits[0, -1], axis=-1)
    next_token_int = int(next_token)

    # At step 29 (divergence point - after "poem" in Python, produces "."), dump logits
    if i == 29:
        logits_np = np.array(logits[0, -1])
        print("\n=== STEP 29 LOGITS (Python - after poem) ===")
        print(f"Token 13 (period): {logits_np[13]:.4f}")
        print(f"Token 2221 (Mr): {logits_np[2221]:.4f}")
        print(f"Token 7031 (Bur): {logits_np[7031]:.4f}")
        # Top 5 tokens
        top5_idx = np.argsort(logits_np)[-5:][::-1]
        print(f"Top 5: {[(int(t), tokenizer.decode([int(t)]), float(logits_np[t])) for t in top5_idx]}")
        # Save for comparison
        np.save('/tmp/python_logits_step28.npy', logits_np)

    # Decode to text
    try:
        text = tokenizer.decode([next_token_int])
    except Exception:
        text = "<unk>"

    # Print
    print(f"Step {i:3d}: token={next_token_int:6d} -> {repr(text)}")

    # Check for EOT
    if next_token_int == tokenizer.eot:
        print("=== EOT ===")
        break

    all_tokens.append(next_token_int)
    tokens = mx.array([[next_token_int]])

print("\n=== FINAL TOKENS ===")
print(all_tokens)
print("\n=== DECODED TEXT ===")
print(tokenizer.decode(all_tokens[sample_begin:]))

# Save encoder output for comparison
print("\n=== SAVING ENCODER OUTPUT ===")
encoder_out_np = np.array(encoder_out)
np.save('/tmp/python_encoder_output.npy', encoder_out_np)
print("Saved encoder output to /tmp/python_encoder_output.npy")
print(f"Shape: {encoder_out_np.shape}, dtype: {encoder_out_np.dtype}")
print(f"Range: [{encoder_out_np.min():.4f}, {encoder_out_np.max():.4f}]")
print(f"Mean: {encoder_out_np.mean():.4f}, Std: {encoder_out_np.std():.4f}")
