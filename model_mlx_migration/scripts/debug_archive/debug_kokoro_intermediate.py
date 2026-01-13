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
Debug Kokoro MLX intermediate values.

Traces through the model to identify where values diverge from expected ranges.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "pytorch_to_mlx"))

import mlx.core as mx
import torch


def load_pytorch_weights():
    """Load PyTorch weights for comparison."""
    model_path = Path.home() / "models" / "kokoro" / "kokoro-v1_0.pth"
    if not model_path.exists():
        print(f"ERROR: PyTorch weights not found at {model_path}")
        return None
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    return state_dict


def load_mlx_model():
    """Load the MLX Kokoro model."""
    from converters.kokoro_converter import load_kokoro_weights
    from converters.models.kokoro import KokoroModel
    from converters.models.kokoro_modules import KokoroConfig

    config = KokoroConfig()
    model = KokoroModel(config)

    # Load weights from PyTorch checkpoint
    model_path = Path.home() / "models" / "kokoro" / "kokoro-v1_0.pth"
    Path.home() / "models" / "kokoro" / "voices" / "af_heart.pt"

    load_kokoro_weights(model, str(model_path))

    return model, config


def debug_source_module():
    """Debug the source module with known F0 values."""
    print("\n" + "=" * 60)
    print("SOURCE MODULE DEBUG")
    print("=" * 60)

    from converters.models.kokoro import SourceModule

    source = SourceModule(sample_rate=24000, num_harmonics=9)

    # Test with constant 200 Hz F0
    f0 = mx.full((1, 10), 200.0)
    upp = 600  # Typical upsampling factor

    har, noise, uv = source(f0, upp)
    mx.eval(har, noise, uv)

    print(f"Input F0 shape: {f0.shape}, F0 value: 200 Hz")
    print(f"Output har shape: {har.shape}")
    print(f"har range: [{float(har.min()):.4f}, {float(har.max()):.4f}]")
    print(f"har RMS: {float(mx.sqrt(mx.mean(har**2))):.4f}")
    print(f"uv mean: {float(mx.mean(uv)):.4f} (should be 1.0 for all voiced)")

    # Check if tanh is saturating
    if float(har.max()) > 0.99 or float(har.min()) < -0.99:
        print("WARNING: tanh output near saturation")

    # Test with range of F0 values
    print("\nF0 sweep test:")
    for f0_val in [100, 200, 300, 400, 500]:
        f0_test = mx.full((1, 10), float(f0_val))
        har, _, _ = source(f0_test, upp)
        mx.eval(har)
        print(
            f"  F0={f0_val}Hz: har range [{float(har.min()):.4f}, {float(har.max()):.4f}]"
        )


def debug_predictor_output(model, tokens, style):
    """Debug the predictor output (F0, noise)."""
    print("\n" + "=" * 60)
    print("PREDICTOR OUTPUT DEBUG")
    print("=" * 60)

    # Run BERT encoding
    mx.array([tokens.shape[1]])
    mx.zeros((1, tokens.shape[1]), dtype=mx.bool_)

    bert_enc = model.bert(
        tokens, attention_mask=mx.ones((1, tokens.shape[1]), dtype=mx.int32)
    )
    bert_enc = model.bert_encoder(bert_enc)
    mx.eval(bert_enc)

    print(f"BERT output shape: {bert_enc.shape}")
    print(
        f"BERT output range: [{float(bert_enc.min()):.4f}, {float(bert_enc.max()):.4f}]"
    )

    # Get style components
    speaker = style[:, 128:]  # [batch, 128]
    style_s = style[:, :128]  # [batch, 128]

    # Run predictor to get durations
    duration_feats = model.predictor.text_encoder(bert_enc, speaker)
    mx.eval(duration_feats)

    print(f"\nDuration feats shape: {duration_feats.shape}")
    print(
        f"Duration feats range: [{float(duration_feats.min()):.4f}, {float(duration_feats.max()):.4f}]"
    )

    # Get duration logits
    # Pack/unpack simulation (simplified - just use the lstm directly)
    lstm_out, _ = model.predictor.lstm(duration_feats)
    mx.eval(lstm_out)
    print(f"LSTM out shape: {lstm_out.shape}")
    print(f"LSTM out range: [{float(lstm_out.min()):.4f}, {float(lstm_out.max()):.4f}]")

    duration_logits = model.predictor.duration_proj(lstm_out)
    mx.eval(duration_logits)
    print(f"Duration logits shape: {duration_logits.shape}")
    print(
        f"Duration logits range: [{float(duration_logits.min()):.4f}, {float(duration_logits.max()):.4f}]"
    )

    # Convert to durations
    duration = mx.sigmoid(duration_logits).sum(axis=-1)
    pred_dur = mx.round(mx.clip(duration, 1, 100)).astype(mx.int32).squeeze()
    mx.eval(pred_dur)

    print(f"\nPredicted durations: {pred_dur.tolist()}")
    total_frames = int(mx.sum(pred_dur))
    print(f"Total expanded frames: {total_frames}")

    # Expand features using predicted durations
    indices = []
    dur_list = pred_dur.tolist()
    for i, d in enumerate(dur_list):
        indices.extend([i] * int(d))
    indices = mx.array(indices)

    en_expanded = duration_feats[0, indices, :]  # [total_frames, hidden]
    en_expanded = en_expanded[None, :, :]  # Add batch dim
    mx.eval(en_expanded)

    print(f"Expanded features shape: {en_expanded.shape}")

    # Get F0 and N predictions
    # First pass through text_encoder for F0/N processing
    enc_out = model.predictor.text_encoder(en_expanded, speaker)
    mx.eval(enc_out)

    print(f"F0/N input (text_encoder out) shape: {enc_out.shape}")
    print(f"F0/N input range: [{float(enc_out.min()):.4f}, {float(enc_out.max()):.4f}]")

    # Apply F0 blocks
    x = enc_out
    for i in range(3):
        block = getattr(model.predictor, f"F0_{i}")
        x = block(x, speaker)
        mx.eval(x)
        print(f"  F0_{i} output range: [{float(x.min()):.4f}, {float(x.max()):.4f}]")

    f0 = model.predictor.F0_proj(x).squeeze(-1)
    mx.eval(f0)

    print(f"\nF0 prediction shape: {f0.shape}")
    print(f"F0 prediction range: [{float(f0.min()):.4f}, {float(f0.max()):.4f}] Hz")
    print(f"F0 mean: {float(mx.mean(f0)):.2f} Hz")

    # Check if F0 is in reasonable speech range (80-400 Hz for typical speech)
    f0_in_range = mx.logical_and(f0 > 80, f0 < 400)
    f0_percent_in_range = float(mx.mean(f0_in_range.astype(mx.float32))) * 100
    print(f"F0 in speech range (80-400 Hz): {f0_percent_in_range:.1f}%")

    # Get N prediction
    x = enc_out
    for i in range(3):
        block = getattr(model.predictor, f"N_{i}")
        x = block(x, speaker)

    noise = model.predictor.N_proj(x).squeeze(-1)
    mx.eval(noise)

    print(f"\nNoise prediction shape: {noise.shape}")
    print(
        f"Noise prediction range: [{float(noise.min()):.4f}, {float(noise.max()):.4f}]"
    )

    return f0, noise, en_expanded, style_s, bert_enc, pred_dur


def debug_decoder_input(model, asr, f0, noise, style):
    """Debug the decoder processing."""
    print("\n" + "=" * 60)
    print("DECODER INPUT DEBUG")
    print("=" * 60)

    print(f"ASR features shape: {asr.shape}")
    print(f"ASR features range: [{float(asr.min()):.4f}, {float(asr.max()):.4f}]")
    print(
        f"F0 shape: {f0.shape}, range: [{float(f0.min()):.4f}, {float(f0.max()):.4f}]"
    )
    print(
        f"Noise shape: {noise.shape}, range: [{float(noise.min()):.4f}, {float(noise.max()):.4f}]"
    )
    print(
        f"Style shape: {style.shape}, range: [{float(style.min()):.4f}, {float(style.max()):.4f}]"
    )


def debug_generator_output(model, features, style, f0):
    """Debug the generator output."""
    print("\n" + "=" * 60)
    print("GENERATOR OUTPUT DEBUG")
    print("=" * 60)

    gen = model.decoder.generator

    # Calculate total upsampling
    total_upp = 1
    for r in gen.config.istft_upsample_rates:
        total_upp *= r
    total_upp *= gen.istft_hop_size
    print(f"Total upsampling factor: {total_upp}")

    # Test source module
    har_source, noise, uv = gen.m_source(f0, total_upp)
    mx.eval(har_source, noise, uv)

    print("\nSource module output:")
    print(f"  har_source shape: {har_source.shape}")
    print(
        f"  har_source range: [{float(har_source.min()):.4f}, {float(har_source.max()):.4f}]"
    )
    print(f"  uv mean: {float(mx.mean(uv)):.4f}")

    # Check if most samples are voiced
    voiced_ratio = float(mx.mean(uv)) * 100
    print(f"  Voiced ratio: {voiced_ratio:.1f}%")


def main():
    print("=" * 60)
    print("KOKORO MLX INTERMEDIATE DEBUG")
    print("=" * 60)

    # Debug source module independently
    debug_source_module()

    # Load model
    print("\nLoading MLX model...")
    model, config = load_mlx_model()
    print("Model loaded successfully")

    # Load voice style
    voice_path = Path.home() / "models" / "kokoro" / "voices" / "af_heart.pt"
    voice_pack = torch.load(voice_path, map_location="cpu", weights_only=False)
    style = mx.array(voice_pack[5].numpy())  # Use 6th style (index 5)
    print(f"Style vector loaded, shape: {style.shape}")

    # Test tokens (simple sequence)
    tokens = mx.array([[0, 47, 44, 51, 51, 54, 0]])  # "hello" approximation

    # Debug predictor
    f0, noise, expanded_feats, style_s, bert_enc, pred_dur = debug_predictor_output(
        model, tokens, style
    )

    # Get text encoder output for ASR features
    input_lengths = mx.array([tokens.shape[1]])
    text_mask = mx.zeros((1, tokens.shape[1]), dtype=mx.bool_)
    t_enc = model.text_encoder(tokens, input_lengths, text_mask)
    mx.eval(t_enc)

    # Build alignment matrix
    dur_list = pred_dur.tolist()
    indices = []
    for i, d in enumerate(dur_list):
        indices.extend([i] * int(d))

    # Expand text encoder output
    if len(indices) > 0:
        indices_arr = mx.array(indices)
        asr = t_enc[0, indices_arr, :]
        asr = asr[None, :, :]
    else:
        asr = t_enc
    mx.eval(asr)

    # Debug decoder input
    debug_decoder_input(model, asr, f0, noise, style_s)

    # Debug generator
    debug_generator_output(model, asr, style_s, f0)

    # Run full synthesis and check audio
    print("\n" + "=" * 60)
    print("FULL SYNTHESIS")
    print("=" * 60)

    audio = model.decoder(asr, f0, noise, style_s)
    mx.eval(audio)

    print(f"Audio shape: {audio.shape}")
    print(f"Audio range: [{float(audio.min()):.4f}, {float(audio.max()):.4f}]")
    print(f"Audio RMS: {float(mx.sqrt(mx.mean(audio**2))):.4f}")

    # Check for clipping
    clipped_high = float(mx.mean((audio > 0.99).astype(mx.float32))) * 100
    clipped_low = float(mx.mean((audio < -0.99).astype(mx.float32))) * 100
    print(f"Samples clipped high: {clipped_high:.2f}%")
    print(f"Samples clipped low: {clipped_low:.2f}%")

    print("\n" + "=" * 60)
    print("DEBUG COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
