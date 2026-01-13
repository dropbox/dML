#!/usr/bin/env python3
"""
Debug encoder stage 1 (DownsampledZipformer2Encoder).

Stage 1 has downsample=2, so it downsamples input, processes, then upsamples.
"""

import sys
import numpy as np
import torch

sys.path.insert(0, '/Users/ayates/model_mlx_migration')

import mlx.core as mx
from src.models.zipformer.features import load_audio, FbankExtractor
from src.models.zipformer.asr_model import ASRModelConfig, load_checkpoint

CHECKPOINT_PATH = 'checkpoints/zipformer/en-streaming/exp/pretrained.pt'
BPE_MODEL_PATH = 'checkpoints/zipformer/en-streaming/data/lang_bpe_500/bpe.model'
AUDIO_PATH = 'checkpoints/zipformer/en-streaming/test_wavs/1089-134686-0001.wav'


def compare_arrays(name: str, a: np.ndarray, b: np.ndarray):
    """Compare arrays and report."""
    if a.shape != b.shape:
        print(f"  {name}: SHAPE MISMATCH {a.shape} vs {b.shape}")
        return False

    max_diff = np.max(np.abs(a - b))
    mean_diff = np.mean(np.abs(a - b))

    if max_diff < 1e-3:
        status = "OK"
    elif max_diff < 1e-1:
        status = "WARN"
    else:
        status = "FAIL"

    print(f"  {name}: {status} max={max_diff:.6f}, mean={mean_diff:.8f}")
    return max_diff < 1e-1


def debug_stage1():
    """Debug encoder stage 1 (DownsampledZipformer2Encoder)."""

    print("=" * 70)
    print("Debugging Encoder Stage 1 (DownsampledZipformer2Encoder)")
    print("=" * 70)

    # Load MLX model
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(BPE_MODEL_PATH)

    model_config = ASRModelConfig(
        vocab_size=sp.GetPieceSize(),
        decoder_dim=512,
        context_size=2,
        joiner_dim=512,
        blank_id=0,
    )
    mlx_model, _ = load_checkpoint(CHECKPOINT_PATH, model_config)
    mx.eval(mlx_model.parameters())

    # Load PyTorch weights
    pt = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)['model']

    # Get encoder and stage 1
    encoder = mlx_model.encoder
    stage1 = encoder.encoders[1]  # DownsampledZipformer2Encoder

    # Load audio and extract features
    waveform, sr = load_audio(AUDIO_PATH, target_sr=16000)
    fbank_extractor = FbankExtractor()
    features = fbank_extractor.extract(waveform, sr)
    features_batch = mx.expand_dims(features, axis=0)

    # Run encoder_embed
    x = encoder.encoder_embed(features_batch)
    mx.eval(x)
    print(f"encoder_embed output: {x.shape}")

    # Run stage 0 to get input to stage 1
    x = encoder._convert_num_channels(x, encoder.encoder_dims[0])
    mx.eval(x)
    x = encoder.encoders[0](x)
    mx.eval(x)
    print(f"Stage 0 output: {x.shape}")

    # Convert channels for stage 1 (192 -> 256)
    x = encoder._convert_num_channels(x, encoder.encoder_dims[1])
    mx.eval(x)
    print(f"After _convert_num_channels: {x.shape}")

    # Save for comparison
    stage1_input = x

    # Stage 1 structure:
    # - downsample (SimpleDownsample with factor=2)
    # - encoder (Zipformer2Encoder)
    # - upsample (SimpleUpsample with factor=2)
    # - out_combiner (BypassModule)

    print("\n--- Stage 1: SimpleDownsample ---")

    # MLX SimpleDownsample
    ds_mlx = stage1.downsample(stage1_input)
    mx.eval(ds_mlx)
    print(f"MLX downsample output: {ds_mlx.shape}")
    print(f"MLX downsample stats: mean={np.mean(np.array(ds_mlx)):.6f}, std={np.std(np.array(ds_mlx)):.6f}")

    # PyTorch SimpleDownsample
    ds_bias = pt['encoder.encoders.1.downsample.bias'].numpy()
    print(f"Downsample bias: {ds_bias}")

    x_pt = torch.from_numpy(np.array(stage1_input)).float()
    seq_len, batch_size, d_model = x_pt.shape
    ds = 2  # downsample factor
    d_seq_len = (seq_len + ds - 1) // ds

    # Pad to multiple
    pad = d_seq_len * ds - seq_len
    if pad > 0:
        x_pt = torch.cat([x_pt, x_pt[-1:].expand(pad, batch_size, d_model)], dim=0)

    # Reshape and apply weighted sum
    x_pt = x_pt.view(d_seq_len, ds, batch_size, d_model)
    weights = torch.softmax(torch.from_numpy(ds_bias), dim=0)[:, None, None]
    ds_pt = (x_pt * weights).sum(dim=1)

    print(f"PT downsample output: {ds_pt.shape}")
    print(f"PT downsample stats: mean={ds_pt.mean():.6f}, std={ds_pt.std():.6f}")

    compare_arrays("downsample_output", ds_pt.numpy(), np.array(ds_mlx))

    # Check stage1 inner encoder
    print("\n--- Stage 1: Inner Encoder ---")

    inner_encoder_mlx = stage1.encoder(ds_mlx)
    mx.eval(inner_encoder_mlx)
    print(f"MLX inner encoder output: {inner_encoder_mlx.shape}")
    print(f"MLX inner encoder stats: mean={np.mean(np.array(inner_encoder_mlx)):.6f}, std={np.std(np.array(inner_encoder_mlx)):.6f}")

    # Run PyTorch inner encoder layer 0
    print("\n--- Checking inner encoder layer 0 ---")
    prefix = "encoder.encoders.1.encoder.layers.0"

    # Get layer 0 from MLX
    inner_layer0 = stage1.encoder.layers[0]

    # Compute pos_emb
    pos_emb = stage1.encoder.pos_encoder(ds_mlx)
    mx.eval(pos_emb)

    # Test feedforward1
    ff1_out_mlx = inner_layer0.feed_forward1(ds_mlx)
    mx.eval(ff1_out_mlx)

    ff1_w = pt[f'{prefix}.feed_forward1.in_proj.weight']
    ff1_b = pt[f'{prefix}.feed_forward1.in_proj.bias']
    ff1_out_w = pt[f'{prefix}.feed_forward1.out_proj.weight']
    ff1_out_b = pt[f'{prefix}.feed_forward1.out_proj.bias']

    ff1_pt = torch.nn.functional.linear(ds_pt, ff1_w, ff1_b)
    ff1_pt = torch.nn.functional.silu(ff1_pt)
    ff1_pt = torch.nn.functional.linear(ff1_pt, ff1_out_w, ff1_out_b)

    compare_arrays("inner_ff1", ff1_pt.numpy(), np.array(ff1_out_mlx))

    # Upsample
    print("\n--- Stage 1: SimpleUpsample ---")

    us_mlx = stage1.upsample(inner_encoder_mlx)
    mx.eval(us_mlx)
    # Trim to original length
    original_len = stage1_input.shape[0]
    us_mlx_trimmed = us_mlx[:original_len]
    mx.eval(us_mlx_trimmed)
    print(f"MLX upsample output (trimmed): {us_mlx_trimmed.shape}")

    # Full stage 1 output
    print("\n--- Stage 1: Full Output ---")

    full_stage1_mlx = stage1(stage1_input)
    mx.eval(full_stage1_mlx)
    print(f"MLX stage 1 full output: {full_stage1_mlx.shape}")
    print(f"MLX stage 1 stats: mean={np.mean(np.array(full_stage1_mlx)):.6f}, std={np.std(np.array(full_stage1_mlx)):.6f}")


if __name__ == "__main__":
    debug_stage1()
