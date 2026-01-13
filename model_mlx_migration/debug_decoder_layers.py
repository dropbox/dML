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

"""Debug script to extract decoder hidden states layer-by-layer."""

import sys

sys.path.insert(0, 'tools')
import mlx.core as mx
import numpy as np
from whisper_mlx.audio import N_SAMPLES, load_audio, log_mel_spectrogram, pad_or_trim
from whisper_mlx.model import WhisperMLX, preprocess_audio_with_vad
from whisper_mlx.tokenizer import get_whisper_tokenizer

# Load model
print("Loading model...")
model = WhisperMLX.from_pretrained('large-v3-turbo')
tokenizer = get_whisper_tokenizer(multilingual=True, num_languages=99, language="en", task="transcribe")

# Load and preprocess audio
audio = load_audio('data/librispeech/dev-clean/1272/128104/1272-128104-0004.flac')
print(f"Audio length: {len(audio)} samples ({len(audio)/16000:.2f}s)")

# Apply VAD
vad_audio, vad_result = preprocess_audio_with_vad(audio, aggressiveness=2, padding_ms=50)
print(f"VAD: {vad_result.speech_ratio*100:.1f}% speech")
audio = vad_audio

# Compute mel spectrogram
audio_padded = pad_or_trim(audio, N_SAMPLES)
mel = log_mel_spectrogram(audio_padded)
mel = mel[None, ...]  # (1, 3000, 128)

# Get encoder output
encoder_out = model.embed_audio(mel)
print(f"Encoder output shape: {encoder_out.shape}, dtype: {encoder_out.dtype}")

# Save encoder output
encoder_np = np.array(encoder_out)
np.save('/tmp/python_encoder.npy', encoder_np)
print("Saved encoder to /tmp/python_encoder.npy")


def debug_decoder_forward(decoder, tokens, xa, kv_cache=None, step_idx=0):
    """Run decoder with layer-by-layer debug output."""

    seq_len = tokens.shape[-1]

    # Get embeddings
    token_emb = decoder.token_embedding.weight[tokens]

    # Get positional embeddings
    offset = kv_cache[0][0][0].shape[1] if kv_cache else 0
    pos_emb = decoder.positional_embedding[offset:offset + seq_len]

    # Combined embedding
    x = token_emb + pos_emb
    mx.eval(x)

    print(f"\n=== STEP {step_idx} DECODER DEBUG ===")
    print(f"Input tokens: {tokens.tolist()}")
    print(f"Offset: {offset}")

    # Save embedding output
    x_np = np.array(x)
    np.save(f'/tmp/python_step{step_idx}_embedding.npy', x_np)
    print(f"After embedding: shape={x.shape}, dtype={x.dtype}")
    print(f"  min={float(mx.min(x)):.6f}, max={float(mx.max(x)):.6f}, mean={float(mx.mean(x)):.6f}")

    # Initialize cache list if needed
    if kv_cache is None:
        kv_cache = [None] * len(decoder.blocks)

    # Get causal mask
    mask = decoder._mask

    # Process through transformer blocks
    for i, block in enumerate(decoder.blocks):
        x_prev = x

        # Self-attention input
        attn_input = mx.fast.layer_norm(x, block.attn_ln.weight, block.attn_ln.bias, eps=1e-5)
        mx.eval(attn_input)

        # Run block
        x, kv_cache[i], _ = block(x_prev, xa, mask=mask, kv_cache=kv_cache[i])
        mx.eval(x)

        # Save layer output
        if i == 0 or i == len(decoder.blocks) - 1:  # First and last layer
            x_np = np.array(x)
            np.save(f'/tmp/python_step{step_idx}_layer{i}.npy', x_np)
            print(f"After layer {i}: min={float(mx.min(x)):.6f}, max={float(mx.max(x)):.6f}")

    # Final layer norm
    hidden_states = mx.fast.layer_norm(x, decoder.ln.weight, decoder.ln.bias, eps=1e-5)
    mx.eval(hidden_states)

    # Save pre-logit hidden states
    hs_np = np.array(hidden_states)
    np.save(f'/tmp/python_step{step_idx}_hidden.npy', hs_np)
    print(f"Final hidden: min={float(mx.min(hidden_states)):.6f}, max={float(mx.max(hidden_states)):.6f}")

    # Project to vocabulary
    logits = decoder.token_embedding.as_linear(hidden_states)

    # Slice back to original vocab size
    if decoder._pad_vocab and decoder._padded_vocab > decoder.n_vocab:
        logits = logits[..., :decoder.n_vocab]

    mx.eval(logits)

    # Save logits
    logits_np = np.array(logits)
    np.save(f'/tmp/python_step{step_idx}_logits.npy', logits_np)
    print(f"Logits: min={float(mx.min(logits)):.6f}, max={float(mx.max(logits)):.6f}")

    return logits, kv_cache


# Run decode step by step
initial_tokens = [50258, 50259, 50359, 50364]  # SOT, en, transcribe, <|0.00|>
tokens = mx.array([initial_tokens])
sample_begin = len(initial_tokens)

kv_cache = None
all_tokens = list(initial_tokens)

print("\n=== INITIAL PROMPT ===")
print(f"Tokens: {initial_tokens}")
print(f"Text: {tokenizer.decode(initial_tokens)}")

# First decode (entire prompt)
logits, kv_cache = debug_decoder_forward(model.decoder, tokens, encoder_out, kv_cache=None, step_idx=0)

# Greedy decode
next_token = int(mx.argmax(logits[0, -1]))
text = tokenizer.decode([next_token])
print(f"\nFirst output token: {next_token} -> {repr(text)}")
all_tokens.append(next_token)

# Continue decoding
for step in range(1, 50):
    tokens = mx.array([[all_tokens[-1]]])

    if step == 29:  # Divergence point
        print(f"\n{'='*60}")
        print(f"DIVERGENCE POINT STEP {step}")
        print(f"{'='*60}")
        logits, kv_cache = debug_decoder_forward(model.decoder, tokens, encoder_out, kv_cache=kv_cache, step_idx=step)

        # Detailed logit analysis
        logits_np = np.array(logits[0, -1])
        print(f"\nTop tokens at step {step}:")
        top10_idx = np.argsort(logits_np)[-10:][::-1]
        for idx in top10_idx:
            print(f"  Token {idx:6d} ({tokenizer.decode([int(idx)]):>15s}): {logits_np[idx]:.4f}")

        print("\nKey tokens:")
        print(f"  Token 13 (period): {logits_np[13]:.4f}")
        print(f"  Token 2221 (Mr): {logits_np[2221]:.4f}")
    else:
        # Normal forward
        logits, kv_cache, _, _ = model.decoder(tokens, encoder_out, kv_cache=kv_cache)

    next_token = int(mx.argmax(logits[0, -1]))

    # Decode
    text = tokenizer.decode([next_token])
    print(f"Step {step:3d}: token={next_token:6d} -> {repr(text)}")

    # Check for EOT
    if next_token == tokenizer.eot:
        print("=== EOT ===")
        break

    all_tokens.append(next_token)

print("\n=== FINAL RESULT ===")
print(f"All tokens: {all_tokens}")
print(f"Text: {tokenizer.decode(all_tokens[sample_begin:])}")
