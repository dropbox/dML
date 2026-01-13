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
Debug Kokoro overflow - trace through Generator to find saturation source.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import mlx.nn as nn


def debug_generator():
    """Debug Generator by tracing intermediate outputs."""
    from tools.pytorch_to_mlx.converters import KokoroConverter

    print("=" * 60)
    print("Kokoro Generator Overflow Debug")
    print("=" * 60)

    # Load model
    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()

    # Load voice
    voice = converter.load_voice("af_heart")
    mx.eval(voice)

    # Test tokens
    tokens = mx.array([[16, 43, 44, 45, 46, 47, 48]])

    # Get intermediate outputs by stepping through the model
    print("\n=== Tracing through model ===\n")

    # 1. BERT encoding
    bert_out = model.bert(tokens)
    mx.eval(bert_out)
    print(
        f"1. BERT output: shape={bert_out.shape}, range=[{float(bert_out.min()):.4f}, {float(bert_out.max()):.4f}]"
    )

    # 2. BERT encoder projection
    bert_enc = model.bert_encoder(bert_out)
    mx.eval(bert_enc)
    print(
        f"2. BERT encoder: shape={bert_enc.shape}, range=[{float(bert_enc.min()):.4f}, {float(bert_enc.max()):.4f}]"
    )

    # 3. Text encoder
    text_enc = model.text_encoder(tokens)
    mx.eval(text_enc)
    print(
        f"3. Text encoder: shape={text_enc.shape}, range=[{float(text_enc.min()):.4f}, {float(text_enc.max()):.4f}]"
    )

    # 4. Combined
    combined = bert_enc + text_enc
    mx.eval(combined)
    print(
        f"4. Combined: shape={combined.shape}, range=[{float(combined.min()):.4f}, {float(combined.max()):.4f}]"
    )

    # 5. Predictor
    # Split voice embedding like model does
    if voice.shape[-1] == 256:
        style = voice[:, :128]
        speaker = voice[:, 128:]
    else:
        style = voice
        speaker = voice

    duration, f0, noise_pred = model.predictor(combined, speaker)
    mx.eval(duration, f0, noise_pred)
    print(
        f"5a. Duration: shape={duration.shape}, range=[{float(duration.min()):.4f}, {float(duration.max()):.4f}]"
    )
    print(
        f"5b. F0: shape={f0.shape}, range=[{float(f0.min()):.4f}, {float(f0.max()):.4f}]"
    )
    print(
        f"5c. Noise: shape={noise_pred.shape}, range=[{float(noise_pred.min()):.4f}, {float(noise_pred.max()):.4f}]"
    )

    # 6. Decoder (step through)
    decoder = model.decoder

    # Process F0 and noise
    f0_orig = f0
    f0_in = f0[:, :, None]
    n_in = noise_pred[:, :, None]

    f0_proc = decoder.f0_conv(f0_in)
    n_proc = decoder.n_conv(n_in)
    mx.eval(f0_proc, n_proc)
    print(
        f"6a. F0 conv: shape={f0_proc.shape}, range=[{float(f0_proc.min()):.4f}, {float(f0_proc.max()):.4f}]"
    )
    print(
        f"6b. N conv: shape={n_proc.shape}, range=[{float(n_proc.min()):.4f}, {float(n_proc.max()):.4f}]"
    )

    # ASR residual
    asr_res = decoder.asr_res(combined)
    mx.eval(asr_res)
    print(
        f"6c. ASR res: shape={asr_res.shape}, range=[{float(asr_res.min()):.4f}, {float(asr_res.max()):.4f}]"
    )

    # Match lengths
    asr_len = combined.shape[1]
    f0_len = f0_proc.shape[1]
    if f0_len > asr_len:
        scale = f0_len // asr_len
        asr_down = mx.repeat(combined, scale, axis=1)[:, :f0_len, :]
        asr_res_down = mx.repeat(asr_res, scale, axis=1)[:, :f0_len, :]
    elif asr_len > f0_len:
        stride = asr_len // f0_len
        asr_down = combined[:, ::stride, :][:, :f0_len, :]
        asr_res_down = asr_res[:, ::stride, :][:, :f0_len, :]
    else:
        asr_down = combined
        asr_res_down = asr_res

    # Encode
    x = mx.concatenate([asr_down, f0_proc, n_proc], axis=-1)
    mx.eval(x)
    print(
        f"6d. Encode input: shape={x.shape}, range=[{float(x.min()):.4f}, {float(x.max()):.4f}]"
    )

    x = decoder.encode(x, style)
    mx.eval(x)
    print(
        f"6e. Encode output: shape={x.shape}, range=[{float(x.min()):.4f}, {float(x.max()):.4f}]"
    )

    # Decode blocks
    for i, block in enumerate(decoder.decode):
        x = mx.concatenate([x, asr_res_down, f0_proc, n_proc], axis=-1)
        x = block(x, style)
        mx.eval(x)
        print(
            f"6f. Decode[{i}]: shape={x.shape}, range=[{float(x.min()):.4f}, {float(x.max()):.4f}]"
        )

        if block.upsample:
            new_len = x.shape[1]
            asr_res_down = mx.repeat(asr_res_down, 2, axis=1)[:, :new_len, :]
            f0_proc = mx.repeat(f0_proc, 2, axis=1)[:, :new_len, :]
            n_proc = mx.repeat(n_proc, 2, axis=1)[:, :new_len, :]

    print("\n=== Generator internals ===\n")

    # Now step through Generator
    gen = decoder.generator

    # Calculate total upsample
    total_upp = 1
    for r in config.istft_upsample_rates:
        total_upp *= r
    total_upp *= gen.istft_hop_size

    print(f"Total upsample factor: {total_upp}")

    # Source module
    har_source, noise_src, uv = gen.m_source(f0_orig, total_upp)
    mx.eval(har_source, noise_src, uv)
    print(
        f"7a. Harmonic source: shape={har_source.shape}, range=[{float(har_source.min()):.4f}, {float(har_source.max()):.4f}]"
    )
    print(
        f"7b. Noise source: shape={noise_src.shape}, range=[{float(noise_src.min()):.4f}, {float(noise_src.max()):.4f}]"
    )

    # STFT on source
    har_1d = har_source.squeeze(-1)
    source = gen._source_stft(har_1d)
    mx.eval(source)
    print(
        f"7c. Source STFT: shape={source.shape}, range=[{float(source.min()):.4f}, {float(source.max()):.4f}]"
    )

    # Upsampling loop
    x_gen = x  # From decoder output
    for i, (up, noise_conv, noise_res) in enumerate(
        zip(gen.ups, gen.noise_convs, gen.noise_res)
    ):
        x_gen = nn.leaky_relu(x_gen, 0.1)

        # Noise conv
        x_source = noise_conv(source)
        mx.eval(x_source)
        print(
            f"8a. noise_conv[{i}] output: shape={x_source.shape}, range=[{float(x_source.min()):.4f}, {float(x_source.max()):.4f}]"
        )

        # Noise res
        x_source = noise_res(x_source, style)
        mx.eval(x_source)
        print(
            f"8b. noise_res[{i}] output: shape={x_source.shape}, range=[{float(x_source.min()):.4f}, {float(x_source.max()):.4f}]"
        )

        # Upsample x
        x_gen = up(x_gen)
        mx.eval(x_gen)
        print(
            f"8c. ups[{i}] output: shape={x_gen.shape}, range=[{float(x_gen.min()):.4f}, {float(x_gen.max()):.4f}]"
        )

        # Reflection pad at last stage
        if i == gen.num_upsamples - 1:
            x_gen = mx.concatenate([x_gen[:, 1:2, :], x_gen], axis=1)

        # Length matching
        if x_source.shape[1] < x_gen.shape[1]:
            pad_len = x_gen.shape[1] - x_source.shape[1]
            x_source = mx.pad(x_source, [(0, 0), (0, pad_len), (0, 0)])
        elif x_source.shape[1] > x_gen.shape[1]:
            x_source = x_source[:, : x_gen.shape[1], :]

        # Add source
        x_gen = x_gen + x_source
        mx.eval(x_gen)
        print(
            f"8d. After add[{i}]: shape={x_gen.shape}, range=[{float(x_gen.min()):.4f}, {float(x_gen.max()):.4f}]"
        )

        # ResBlocks
        xs = None
        for j in range(gen.num_kernels):
            block_idx = i * gen.num_kernels + j
            if block_idx < len(gen.resblocks):
                if xs is None:
                    xs = gen.resblocks[block_idx](x_gen, style)
                else:
                    xs = xs + gen.resblocks[block_idx](x_gen, style)
        if xs is not None:
            x_gen = xs / gen.num_kernels
        mx.eval(x_gen)
        print(
            f"8e. After resblocks[{i}]: shape={x_gen.shape}, range=[{float(x_gen.min()):.4f}, {float(x_gen.max()):.4f}]"
        )

    # Post conv
    x_gen = nn.leaky_relu(x_gen, 0.1)
    x_gen = gen.conv_post(x_gen)
    mx.eval(x_gen)
    print(
        f"\n9a. conv_post output: shape={x_gen.shape}, range=[{float(x_gen.min()):.4f}, {float(x_gen.max()):.4f}]"
    )

    # Magnitude and phase
    n_bins = gen.post_n_fft // 2 + 1
    log_mag = mx.clip(x_gen[..., :n_bins], -10, 10)
    mag = mx.exp(log_mag)
    mx.eval(mag)
    print(
        f"9b. Magnitude: shape={mag.shape}, range=[{float(mag.min()):.4f}, {float(mag.max()):.4f}]"
    )

    phase_logits = x_gen[..., n_bins:]
    phase = mx.sin(phase_logits)
    mx.eval(phase)
    print(
        f"9c. Phase: shape={phase.shape}, range=[{float(phase.min()):.4f}, {float(phase.max()):.4f}]"
    )

    # ISTFT
    audio = gen._istft_synthesis(mag, phase)
    mx.eval(audio)
    print(
        f"\n10. ISTFT output (before clip): shape={audio.shape}, range=[{float(audio.min()):.4f}, {float(audio.max()):.4f}]"
    )

    # Final clip
    audio_clipped = mx.clip(audio, -1.0, 1.0)
    mx.eval(audio_clipped)

    # Check how much was clipped
    clipped_count = int(mx.sum((audio < -1.0) | (audio > 1.0)))
    total_samples = int(audio.size)
    clip_percent = 100.0 * clipped_count / total_samples

    print(
        f"\n11. Clipping stats: {clipped_count}/{total_samples} samples clipped ({clip_percent:.2f}%)"
    )

    return audio_clipped


if __name__ == "__main__":
    audio = debug_generator()
    print("\n" + "=" * 60)
    print("Debug complete")
    print("=" * 60)
