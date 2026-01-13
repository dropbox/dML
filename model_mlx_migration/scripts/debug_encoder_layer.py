#!/usr/bin/env python3
"""
Debug individual encoder layer components to find divergence.

Traces: feedforward, self_attn, nonlin_attn, conv_module, norm, bypass
"""

import sys
import numpy as np
import torch

sys.path.insert(0, '/Users/ayates/model_mlx_migration')

import mlx.core as mx
from src.models.zipformer.features import load_audio, FbankExtractor
from src.models.zipformer.asr_model import ASRModelConfig, load_checkpoint

# Paths
CHECKPOINT_PATH = 'checkpoints/zipformer/en-streaming/exp/pretrained.pt'
BPE_MODEL_PATH = 'checkpoints/zipformer/en-streaming/data/lang_bpe_500/bpe.model'
AUDIO_PATH = 'checkpoints/zipformer/en-streaming/test_wavs/1089-134686-0001.wav'


def load_pytorch_weights():
    """Load the PyTorch checkpoint weights."""
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    return checkpoint['model']


def compare_arrays(name: str, pt: np.ndarray, mlx: np.ndarray):
    """Compare two arrays and report differences."""
    if pt.shape != mlx.shape:
        print(f"  {name}: SHAPE MISMATCH pt={pt.shape} vs mlx={mlx.shape}")
        return False

    max_diff = np.max(np.abs(pt - mlx))
    mean_diff = np.mean(np.abs(pt - mlx))

    if max_diff < 1e-4:
        status = "OK"
    elif max_diff < 1e-2:
        status = "WARN"
    else:
        status = "FAIL"

    print(f"  {name}: {status} max={max_diff:.6f}, mean={mean_diff:.8f}")
    return max_diff < 1e-2


def debug_encoder_layer():
    """Debug each component of the first encoder layer."""

    print("=" * 70)
    print("Debugging Encoder Layer Components")
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
    pt_weights = load_pytorch_weights()

    # Load audio and extract features
    waveform, sr = load_audio(AUDIO_PATH, target_sr=16000)
    fbank_extractor = FbankExtractor()
    features = fbank_extractor.extract(waveform, sr)
    features_batch = mx.expand_dims(features, axis=0)

    # Run encoder_embed to get input for stage 0
    mlx_embed_out = mlx_model.encoder.encoder_embed(features_batch)
    mx.eval(mlx_embed_out)

    # Get PyTorch encoder_embed output (already verified to match)
    features_pt = torch.from_numpy(np.array(features)).unsqueeze(0).float()
    x_pt = run_pytorch_encoder_embed(features_pt, pt_weights)

    print(f"\nEncoder embed output: MLX={mlx_embed_out.shape}, PT={x_pt.shape}")
    compare_arrays("embed_out", x_pt.detach().numpy(), np.array(mlx_embed_out))

    # Now trace through encoder layer 0
    mlx_encoder0 = mlx_model.encoder.encoders[0]
    layer0 = mlx_encoder0.layers[0]
    prefix = "encoder.encoders.0.layers.0"

    src_mlx = mlx_embed_out
    src_pt = x_pt.clone()
    src_orig_pt = src_pt.clone()

    seq_len, batch_size, d_model = src_pt.shape
    num_heads = 4
    query_head_dim = 32
    pos_head_dim = 4
    value_head_dim = 12

    # Compute pos_emb
    pos_emb_mlx = mlx_encoder0.pos_encoder(src_mlx)
    mx.eval(pos_emb_mlx)
    pos_emb_pt = compute_pytorch_pos_emb(seq_len, batch_size, 48)

    # Compute attention weights
    attn_weights_mlx = layer0.self_attn_weights(src_mlx, pos_emb_mlx)
    mx.eval(attn_weights_mlx)
    attn_weights_pt = compute_pytorch_attn_weights(
        src_pt, pos_emb_pt, pt_weights, prefix,
        num_heads, query_head_dim, pos_head_dim, seq_len
    )

    print("\n--- Attention weights ---")
    compare_arrays("attn_weights", attn_weights_pt.detach().numpy(), np.array(attn_weights_mlx))

    # Component 1: First feedforward
    print("\n--- Component 1: feed_forward1 ---")
    ff1_out_mlx = layer0.feed_forward1(src_mlx)
    mx.eval(ff1_out_mlx)

    ff1_w = pt_weights[f'{prefix}.feed_forward1.in_proj.weight']
    ff1_b = pt_weights[f'{prefix}.feed_forward1.in_proj.bias']
    ff1_out_w = pt_weights[f'{prefix}.feed_forward1.out_proj.weight']
    ff1_out_b = pt_weights[f'{prefix}.feed_forward1.out_proj.bias']

    ff1_pt = torch.nn.functional.linear(src_pt, ff1_w, ff1_b)
    ff1_pt = torch.nn.functional.silu(ff1_pt)
    ff1_pt = torch.nn.functional.linear(ff1_pt, ff1_out_w, ff1_out_b)

    compare_arrays("ff1_out", ff1_pt.detach().numpy(), np.array(ff1_out_mlx))

    src_mlx = src_mlx + ff1_out_mlx
    src_pt = src_pt + ff1_pt
    mx.eval(src_mlx)

    # Component 2: First self-attention
    print("\n--- Component 2: self_attn1 ---")
    attn1_out_mlx = layer0.self_attn1(src_mlx, attn_weights_mlx)
    mx.eval(attn1_out_mlx)

    attn1_in_w = pt_weights[f'{prefix}.self_attn1.in_proj.weight']
    attn1_in_b = pt_weights[f'{prefix}.self_attn1.in_proj.bias']
    attn1_out_w = pt_weights[f'{prefix}.self_attn1.out_proj.weight']
    attn1_out_b = pt_weights[f'{prefix}.self_attn1.out_proj.bias']

    v = torch.nn.functional.linear(src_pt, attn1_in_w, attn1_in_b)
    v = v.view(seq_len, batch_size, num_heads, value_head_dim)
    v = v.permute(1, 2, 0, 3)  # (batch, heads, seq, head_dim)
    v = v.reshape(batch_size * num_heads, seq_len, value_head_dim)

    attn1_out_pt = torch.matmul(attn_weights_pt, v)  # (batch*heads, seq, head_dim)
    attn1_out_pt = attn1_out_pt.view(batch_size, num_heads, seq_len, value_head_dim)
    attn1_out_pt = attn1_out_pt.permute(2, 0, 1, 3)  # (seq, batch, heads, head_dim)
    attn1_out_pt = attn1_out_pt.reshape(seq_len, batch_size, num_heads * value_head_dim)
    attn1_out_pt = torch.nn.functional.linear(attn1_out_pt, attn1_out_w, attn1_out_b)

    compare_arrays("attn1_out", attn1_out_pt.detach().numpy(), np.array(attn1_out_mlx))

    src_mlx = src_mlx + attn1_out_mlx
    src_pt = src_pt + attn1_out_pt
    mx.eval(src_mlx)

    # Component 3: Non-linear attention
    print("\n--- Component 3: nonlin_attention ---")
    nonlin_out_mlx = layer0.nonlin_attention(src_mlx, attn_weights_mlx)
    mx.eval(nonlin_out_mlx)

    nonlin_in_w = pt_weights[f'{prefix}.nonlin_attention.in_proj.weight']
    nonlin_in_b = pt_weights[f'{prefix}.nonlin_attention.in_proj.bias']
    nonlin_out_w = pt_weights[f'{prefix}.nonlin_attention.out_proj.weight']
    nonlin_out_b = pt_weights[f'{prefix}.nonlin_attention.out_proj.bias']

    hidden_channels = nonlin_in_w.shape[0] // 3
    proj = torch.nn.functional.linear(src_pt, nonlin_in_w, nonlin_in_b)
    s, v, y = torch.chunk(proj, 3, dim=-1)
    s = torch.tanh(s)
    v = v * s

    head_dim = hidden_channels // num_heads
    v = v.view(seq_len, batch_size, num_heads, head_dim)
    v = v.permute(1, 2, 0, 3)  # (batch, heads, seq, head_dim)
    v = v.reshape(batch_size * num_heads, seq_len, head_dim)

    nonlin_out_pt = torch.matmul(attn_weights_pt, v)
    nonlin_out_pt = nonlin_out_pt.view(batch_size, num_heads, seq_len, head_dim)
    nonlin_out_pt = nonlin_out_pt.permute(2, 0, 1, 3)
    nonlin_out_pt = nonlin_out_pt.reshape(seq_len, batch_size, hidden_channels)
    nonlin_out_pt = nonlin_out_pt * y
    nonlin_out_pt = torch.nn.functional.linear(nonlin_out_pt, nonlin_out_w, nonlin_out_b)

    compare_arrays("nonlin_out", nonlin_out_pt.detach().numpy(), np.array(nonlin_out_mlx))

    src_mlx = src_mlx + nonlin_out_mlx
    src_pt = src_pt + nonlin_out_pt
    mx.eval(src_mlx)

    # Component 4: First convolution module
    print("\n--- Component 4: conv_module1 ---")
    conv1_out_mlx = layer0.conv_module1(src_mlx)
    mx.eval(conv1_out_mlx)

    conv1_out_pt = run_pytorch_conv_module(src_pt, pt_weights, f'{prefix}.conv_module1', d_model)

    compare_arrays("conv1_out", conv1_out_pt.detach().numpy(), np.array(conv1_out_mlx))

    src_mlx = src_mlx + conv1_out_mlx
    src_pt = src_pt + conv1_out_pt
    mx.eval(src_mlx)

    # Component 5: Second feedforward
    print("\n--- Component 5: feed_forward2 ---")
    ff2_out_mlx = layer0.feed_forward2(src_mlx)
    mx.eval(ff2_out_mlx)

    ff2_w = pt_weights[f'{prefix}.feed_forward2.in_proj.weight']
    ff2_b = pt_weights[f'{prefix}.feed_forward2.in_proj.bias']
    ff2_out_w = pt_weights[f'{prefix}.feed_forward2.out_proj.weight']
    ff2_out_b = pt_weights[f'{prefix}.feed_forward2.out_proj.bias']

    ff2_pt = torch.nn.functional.linear(src_pt, ff2_w, ff2_b)
    ff2_pt = torch.nn.functional.silu(ff2_pt)
    ff2_pt = torch.nn.functional.linear(ff2_pt, ff2_out_w, ff2_out_b)

    compare_arrays("ff2_out", ff2_pt.detach().numpy(), np.array(ff2_out_mlx))

    src_mlx = src_mlx + ff2_out_mlx
    src_pt = src_pt + ff2_pt
    mx.eval(src_mlx)

    print("\n--- After first half of layer ---")
    compare_arrays("src_mid", src_pt.detach().numpy(), np.array(src_mlx))

    # Mid bypass
    print("\n--- Component 6: bypass_mid ---")
    bypass_mid_scale_mlx = layer0.bypass_mid.bypass_scale
    bypass_mid_scale_pt = pt_weights.get(f'{prefix}.bypass_mid.bypass_scale', torch.full((d_model,), 0.5))

    # Print stats for per-channel scales
    print(f"  bypass_mid_scale: MLX mean={np.mean(np.array(bypass_mid_scale_mlx)):.6f}, PT mean={bypass_mid_scale_pt.mean().item():.6f}")

    src_orig_mlx = mlx_embed_out  # Original input
    scale_mlx = mx.clip(bypass_mid_scale_mlx, 0.0, 1.0)
    bypass_mid_out_mlx = src_orig_mlx + scale_mlx * (src_mlx - src_orig_mlx)
    mx.eval(bypass_mid_out_mlx)

    scale_pt = torch.clamp(bypass_mid_scale_pt, 0.0, 1.0)
    bypass_mid_out_pt = src_orig_pt + scale_pt * (src_pt - src_orig_pt)

    compare_arrays("bypass_mid_out", bypass_mid_out_pt.detach().numpy(), np.array(bypass_mid_out_mlx))

    # Continue with bypass_mid output
    src_mlx = bypass_mid_out_mlx
    src_pt = bypass_mid_out_pt

    # Component 7: Second self-attention
    print("\n--- Component 7: self_attn2 ---")
    attn2_out_mlx = layer0.self_attn2(src_mlx, attn_weights_mlx)
    mx.eval(attn2_out_mlx)

    attn2_in_w = pt_weights[f'{prefix}.self_attn2.in_proj.weight']
    attn2_in_b = pt_weights[f'{prefix}.self_attn2.in_proj.bias']
    attn2_out_w = pt_weights[f'{prefix}.self_attn2.out_proj.weight']
    attn2_out_b = pt_weights[f'{prefix}.self_attn2.out_proj.bias']

    v = torch.nn.functional.linear(src_pt, attn2_in_w, attn2_in_b)
    v = v.view(seq_len, batch_size, num_heads, value_head_dim)
    v = v.permute(1, 2, 0, 3)
    v = v.reshape(batch_size * num_heads, seq_len, value_head_dim)

    attn2_out_pt = torch.matmul(attn_weights_pt, v)
    attn2_out_pt = attn2_out_pt.view(batch_size, num_heads, seq_len, value_head_dim)
    attn2_out_pt = attn2_out_pt.permute(2, 0, 1, 3)
    attn2_out_pt = attn2_out_pt.reshape(seq_len, batch_size, num_heads * value_head_dim)
    attn2_out_pt = torch.nn.functional.linear(attn2_out_pt, attn2_out_w, attn2_out_b)

    compare_arrays("attn2_out", attn2_out_pt.detach().numpy(), np.array(attn2_out_mlx))

    src_mlx = src_mlx + attn2_out_mlx
    src_pt = src_pt + attn2_out_pt
    mx.eval(src_mlx)

    # Component 8: Second convolution
    print("\n--- Component 8: conv_module2 ---")
    conv2_out_mlx = layer0.conv_module2(src_mlx)
    mx.eval(conv2_out_mlx)

    conv2_out_pt = run_pytorch_conv_module(src_pt, pt_weights, f'{prefix}.conv_module2', d_model)

    compare_arrays("conv2_out", conv2_out_pt.detach().numpy(), np.array(conv2_out_mlx))

    src_mlx = src_mlx + conv2_out_mlx
    src_pt = src_pt + conv2_out_pt
    mx.eval(src_mlx)

    # Component 9: Third feedforward
    print("\n--- Component 9: feed_forward3 ---")
    ff3_out_mlx = layer0.feed_forward3(src_mlx)
    mx.eval(ff3_out_mlx)

    ff3_w = pt_weights[f'{prefix}.feed_forward3.in_proj.weight']
    ff3_b = pt_weights[f'{prefix}.feed_forward3.in_proj.bias']
    ff3_out_w = pt_weights[f'{prefix}.feed_forward3.out_proj.weight']
    ff3_out_b = pt_weights[f'{prefix}.feed_forward3.out_proj.bias']

    ff3_pt = torch.nn.functional.linear(src_pt, ff3_w, ff3_b)
    ff3_pt = torch.nn.functional.silu(ff3_pt)
    ff3_pt = torch.nn.functional.linear(ff3_pt, ff3_out_w, ff3_out_b)

    compare_arrays("ff3_out", ff3_pt.detach().numpy(), np.array(ff3_out_mlx))

    src_mlx = src_mlx + ff3_out_mlx
    src_pt = src_pt + ff3_pt
    mx.eval(src_mlx)

    # Component 10: BiasNorm
    print("\n--- Component 10: norm (BiasNorm) ---")
    norm_out_mlx = layer0.norm(src_mlx)
    mx.eval(norm_out_mlx)

    norm_bias = pt_weights[f'{prefix}.norm.bias']
    norm_log_scale = pt_weights[f'{prefix}.norm.log_scale']

    diff = src_pt - norm_bias
    var = (diff ** 2).mean(dim=-1, keepdim=True)
    scales = (var ** -0.5) * torch.exp(norm_log_scale)
    norm_out_pt = src_pt * scales

    compare_arrays("norm_out", norm_out_pt.detach().numpy(), np.array(norm_out_mlx))

    src_mlx = norm_out_mlx
    src_pt = norm_out_pt

    # Component 11: Final bypass
    print("\n--- Component 11: bypass ---")
    bypass_scale_mlx = layer0.bypass.bypass_scale
    bypass_scale_pt = pt_weights.get(f'{prefix}.bypass.bypass_scale', torch.full((d_model,), 0.5))

    print(f"  bypass_scale: MLX mean={np.mean(np.array(bypass_scale_mlx)):.6f}, PT mean={bypass_scale_pt.mean().item():.6f}")

    scale_mlx = mx.clip(bypass_scale_mlx, 0.0, 1.0)
    final_out_mlx = src_orig_mlx + scale_mlx * (src_mlx - src_orig_mlx)
    mx.eval(final_out_mlx)

    scale_pt = torch.clamp(bypass_scale_pt, 0.0, 1.0)
    final_out_pt = src_orig_pt + scale_pt * (src_pt - src_orig_pt)

    compare_arrays("final_out", final_out_pt.detach().numpy(), np.array(final_out_mlx))

    # Compare with actual MLX layer output
    print("\n--- Full layer comparison ---")
    full_layer_out_mlx = layer0(mlx_embed_out, pos_emb_mlx)
    mx.eval(full_layer_out_mlx)

    compare_arrays("full_layer_vs_manual", final_out_pt.detach().numpy(), np.array(full_layer_out_mlx))


def run_pytorch_encoder_embed(features_pt, pt_weights):
    """Run PyTorch encoder_embed forward pass."""
    x_pt = features_pt.unsqueeze(1)  # (batch, 1, T, 80)

    # Conv0
    conv0_w = pt_weights['encoder_embed.conv.0.weight']
    conv0_b = pt_weights['encoder_embed.conv.0.bias']
    x_pt = torch.nn.functional.conv2d(x_pt, conv0_w, conv0_b, stride=1, padding=(0, 1))
    z = x_pt - 1.0
    x_pt = torch.where(z > 20, z, torch.log1p(torch.exp(z))) - 0.08 * x_pt - 0.31326168

    # Conv4
    conv4_w = pt_weights['encoder_embed.conv.4.weight']
    conv4_b = pt_weights['encoder_embed.conv.4.bias']
    x_pt = torch.nn.functional.conv2d(x_pt, conv4_w, conv4_b, stride=2)
    z = x_pt - 1.0
    x_pt = torch.where(z > 20, z, torch.log1p(torch.exp(z))) - 0.08 * x_pt - 0.31326168

    # Conv7
    conv7_w = pt_weights['encoder_embed.conv.7.weight']
    conv7_b = pt_weights['encoder_embed.conv.7.bias']
    x_pt = torch.nn.functional.conv2d(x_pt, conv7_w, conv7_b, stride=(1, 2))
    z = x_pt - 1.0
    x_pt = torch.where(z > 20, z, torch.log1p(torch.exp(z))) - 0.08 * x_pt - 0.31326168

    # ConvNext
    residual = x_pt
    dw_w = pt_weights['encoder_embed.convnext.depthwise_conv.weight']
    dw_b = pt_weights['encoder_embed.convnext.depthwise_conv.bias']
    x_pt = torch.nn.functional.conv2d(x_pt, dw_w, dw_b, padding=3, groups=128)
    pw1_w = pt_weights['encoder_embed.convnext.pointwise_conv1.weight']
    pw1_b = pt_weights['encoder_embed.convnext.pointwise_conv1.bias']
    x_pt = torch.nn.functional.conv2d(x_pt, pw1_w, pw1_b)
    z = x_pt - 4.0
    x_pt = torch.where(z > 20, z, torch.log1p(torch.exp(z))) - 0.08 * x_pt - 0.035
    pw2_w = pt_weights['encoder_embed.convnext.pointwise_conv2.weight']
    pw2_b = pt_weights['encoder_embed.convnext.pointwise_conv2.bias']
    x_pt = torch.nn.functional.conv2d(x_pt, pw2_w, pw2_b)
    x_pt = x_pt + residual

    # Flatten
    batch, c, t, f = x_pt.shape
    x_pt = x_pt.transpose(1, 2).reshape(batch, t, c * f)

    # Output projection
    out_w = pt_weights['encoder_embed.out.weight']
    out_b = pt_weights['encoder_embed.out.bias']
    x_pt = torch.nn.functional.linear(x_pt, out_w, out_b)

    # BiasNorm
    norm_bias = pt_weights['encoder_embed.out_norm.bias']
    norm_log_scale = pt_weights['encoder_embed.out_norm.log_scale']
    diff = x_pt - norm_bias
    var = (diff ** 2).mean(dim=-1, keepdim=True)
    x_pt = x_pt * (var ** -0.5) * torch.exp(norm_log_scale)

    return x_pt.transpose(0, 1)  # (T, batch, d_model)


def compute_pytorch_pos_emb(seq_len, batch_size, pos_dim):
    """Compute PyTorch positional encoding."""
    positions = torch.arange(-(seq_len - 1), seq_len, dtype=torch.float32)[:, None]
    div_term = torch.exp(
        torch.arange(0, pos_dim, 2, dtype=torch.float32) *
        (-np.log(10000.0) / pos_dim)
    )
    pe = torch.zeros((2 * seq_len - 1, pos_dim))
    pe[:, 0::2] = torch.sin(positions * div_term)
    pe[:, 1::2] = torch.cos(positions * div_term)
    return pe.unsqueeze(0).expand(batch_size, -1, -1)


def compute_pytorch_attn_weights(src, pos_emb, pt_weights, prefix, num_heads, query_head_dim, pos_head_dim, seq_len):
    """Compute PyTorch attention weights."""
    batch_size = src.shape[1]

    in_proj_w = pt_weights[f'{prefix}.self_attn_weights.in_proj.weight']
    in_proj_b = pt_weights[f'{prefix}.self_attn_weights.in_proj.bias']
    linear_pos_w = pt_weights[f'{prefix}.self_attn_weights.linear_pos.weight']

    proj = torch.nn.functional.linear(src, in_proj_w, in_proj_b)
    query_dim = num_heads * query_head_dim
    q = proj[:, :, :query_dim]
    k = proj[:, :, query_dim:2*query_dim]
    p = proj[:, :, 2*query_dim:]

    q = q.view(seq_len, batch_size, num_heads, query_head_dim).permute(1, 2, 0, 3)
    k = k.view(seq_len, batch_size, num_heads, query_head_dim).permute(1, 2, 0, 3)
    p = p.view(seq_len, batch_size, num_heads, pos_head_dim).permute(1, 2, 0, 3)

    pos_proj = torch.nn.functional.linear(pos_emb, linear_pos_w)
    pos_proj = pos_proj.view(batch_size, -1, num_heads, pos_head_dim).permute(0, 2, 1, 3)

    scale = query_head_dim ** -0.5
    content_score = torch.matmul(q, k.transpose(-1, -2)) * scale

    pos_score = torch.matmul(p, pos_proj.transpose(-1, -2))
    pos_score_flat = pos_score.view(batch_size * num_heads, seq_len, -1)

    # rel_shift
    pos_score_flat = torch.nn.functional.pad(pos_score_flat, (0, 1))
    pos_score_flat = pos_score_flat.view(batch_size * num_heads, -1, seq_len)
    pos_score_flat = pos_score_flat[:, 1:, :]
    pos_score_flat = pos_score_flat.view(batch_size * num_heads, seq_len, -1)
    pos_score_flat = pos_score_flat[:, :, :seq_len]

    pos_score_shifted = pos_score_flat.view(batch_size, num_heads, seq_len, seq_len)
    attn_score = content_score + pos_score_shifted
    attn_weights = torch.softmax(attn_score, dim=-1)
    return attn_weights.view(batch_size * num_heads, seq_len, seq_len)


def run_pytorch_conv_module(src, pt_weights, prefix, d_model):
    """Run PyTorch convolution module (causal)."""
    seq_len, batch_size, _ = src.shape

    # Input projection
    in_proj_w = pt_weights[f'{prefix}.in_proj.weight']
    in_proj_b = pt_weights[f'{prefix}.in_proj.bias']
    x = torch.nn.functional.linear(src, in_proj_w, in_proj_b)
    x, gate = x.chunk(2, dim=-1)
    x = x * torch.sigmoid(gate)

    # Transpose for conv
    x = x.permute(1, 0, 2)  # (batch, seq, d_model)

    # Causal conv
    kernel_size = 31
    half_kernel = (kernel_size + 1) // 2  # 16
    left_pad = kernel_size // 2  # 15

    x_padded = torch.nn.functional.pad(x, (0, 0, left_pad, 0))

    # Causal component
    causal_w = pt_weights[f'{prefix}.depthwise_conv.causal_conv.weight']
    causal_b = pt_weights[f'{prefix}.depthwise_conv.causal_conv.bias']
    x_causal = torch.nn.functional.conv1d(
        x_padded.transpose(1, 2), causal_w, causal_b, groups=d_model
    ).transpose(1, 2)

    # Chunkwise component
    chunk_w = pt_weights[f'{prefix}.depthwise_conv.chunkwise_conv.weight']
    chunk_b = pt_weights[f'{prefix}.depthwise_conv.chunkwise_conv.bias']
    x_chunk_padded = torch.nn.functional.pad(x, (0, 0, left_pad, left_pad))
    x_chunk = torch.nn.functional.conv1d(
        x_chunk_padded.transpose(1, 2), chunk_w, chunk_b, groups=d_model
    ).transpose(1, 2)

    x = x_causal + x_chunk

    # Transpose back
    x = x.permute(1, 0, 2)  # (seq, batch, d_model)

    # SiLU and output projection
    x = torch.nn.functional.silu(x)
    out_proj_w = pt_weights[f'{prefix}.out_proj.weight']
    out_proj_b = pt_weights[f'{prefix}.out_proj.bias']
    x = torch.nn.functional.linear(x, out_proj_w, out_proj_b)

    return x


if __name__ == "__main__":
    debug_encoder_layer()
