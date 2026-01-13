#!/usr/bin/env python3
"""Debug encoder stages to find where scale mismatch originates.

Compares MLX encoder output at each stage:
1. encoder_embed
2. Stage 0 (Zipformer2Encoder)
3. Stages 1-5 (DownsampledZipformer2Encoder)
4. _get_full_dim_output
5. Final downsampling
"""

import sys
sys.path.insert(0, "src")

import numpy as np
import mlx.core as mx
from pathlib import Path

from models.zipformer.features import FbankExtractor, load_audio
from models.zipformer.asr_model import ASRModelConfig, load_checkpoint

# Paths
CHECKPOINT = Path("checkpoints/zipformer/en-streaming/exp/pretrained.pt")
BPE_MODEL = Path("checkpoints/zipformer/en-streaming/data/lang_bpe_500/bpe.model")
TEST_WAV = Path("checkpoints/zipformer/sherpa-onnx-streaming-zipformer-en-2023-06-26/test_wavs/0.wav")


def main():
    # Load audio and extract features
    print("Loading audio and extracting features...")
    waveform, sr = load_audio(str(TEST_WAV))
    extractor = FbankExtractor()
    features = extractor.extract(waveform, sr)
    print(f"Features shape: {features.shape}")

    # Add batch dimension
    features = mx.expand_dims(features, axis=0)  # (1, T, 80)
    print(f"Features with batch: {features.shape}")

    # Load model
    print("\nLoading MLX model...")
    model_config = ASRModelConfig(
        vocab_size=500,
        decoder_dim=512,
        context_size=2,
        joiner_dim=512,
        blank_id=0,
    )
    model, _ = load_checkpoint(str(CHECKPOINT), model_config)
    mx.eval(model.parameters())

    encoder = model.encoder

    # Trace through stages
    print("\n" + "="*60)
    print("STAGE-BY-STAGE TRACE")
    print("="*60)

    # Stage 0: encoder_embed
    x = encoder.encoder_embed(features)  # (T', batch, 192)
    mx.eval(x)
    print("\n1. encoder_embed output:")
    print(f"   Shape: {x.shape}")
    print(f"   Range: [{float(mx.min(x)):.4f}, {float(mx.max(x)):.4f}]")
    print(f"   Std: {float(mx.std(x)):.4f}")
    print(f"   First 5: {np.array(x[:, 0, :5]).flatten()[:5]}")

    # Track outputs for _get_full_dim_output
    outputs = []

    # Process through encoder stages
    for i, enc in enumerate(encoder.encoders):
        # Convert channels
        x = encoder._convert_num_channels(x, encoder.encoder_dims[i])
        print(f"\n{2+i}. After channel conversion for stage {i}:")
        print(f"   Shape: {x.shape}, d_model={encoder.encoder_dims[i]}")

        # Run encoder stage
        x = enc(x)
        mx.eval(x)
        outputs.append(x)

        print(f"   Stage {i} encoder output:")
        print(f"   Shape: {x.shape}")
        print(f"   Range: [{float(mx.min(x)):.4f}, {float(mx.max(x)):.4f}]")
        print(f"   Std: {float(mx.std(x)):.4f}")
        print(f"   Mean: {float(mx.mean(x)):.4f}")

    # Combine outputs
    print("\n8. _get_full_dim_output:")
    x = encoder._get_full_dim_output(outputs)
    mx.eval(x)
    print(f"   Shape: {x.shape}")
    print(f"   Range: [{float(mx.min(x)):.4f}, {float(mx.max(x)):.4f}]")
    print(f"   Std: {float(mx.std(x)):.4f}")

    # Final downsampling
    print("\n9. Final downsampling:")
    seq_len, batch_size, d_model = x.shape
    ds = 2
    d_seq_len = (seq_len + ds - 1) // ds
    pad = d_seq_len * ds - seq_len
    if pad > 0:
        x = mx.concatenate([x, mx.broadcast_to(x[-1:], (pad, batch_size, d_model))], axis=0)
    x = mx.reshape(x, (d_seq_len, ds, batch_size, d_model))
    weights = mx.softmax(encoder.downsample_output_bias)[:, None, None]
    x = mx.sum(x * weights, axis=1)
    x = mx.transpose(x, (1, 0, 2))  # (batch, T, d_model)
    mx.eval(x)
    print(f"   Shape: {x.shape}")
    print(f"   Range: [{float(mx.min(x)):.4f}, {float(mx.max(x)):.4f}]")
    print(f"   Std: {float(mx.std(x)):.4f}")

    # Compare with reference: ONNX produces std ~0.76
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"MLX encoder final output std: {float(mx.std(x)):.4f}")
    print("Expected ONNX encoder std: ~0.76")
    print(f"Scale ratio: ~{0.76 / float(mx.std(x)):.2f}x")

    # Check bypass scales
    print("\n" + "="*60)
    print("BYPASS SCALES")
    print("="*60)
    for i, enc in enumerate(encoder.encoders):
        if hasattr(enc, 'encoder'):
            # DownsampledZipformer2Encoder
            layers = enc.encoder.layers
        else:
            # Zipformer2Encoder
            layers = enc.layers

        for j, layer in enumerate(layers):
            bypass_scale = float(layer.bypass_scale)
            bypass_mid_bias = float(mx.mean(mx.abs(layer.bypass_mid.bypass_scale)))
            print(f"   Stage {i} Layer {j}: bypass_scale={bypass_scale:.4f}, bypass_mid={bypass_mid_bias:.4f}")


if __name__ == "__main__":
    main()
