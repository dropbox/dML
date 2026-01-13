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
Debug Decoder Blocks - Export intermediate tensors for C++ comparison.

This script runs the full Kokoro pipeline and exports decode block outputs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "pytorch_to_mlx"))

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

MODEL_PATH = Path.home() / "model_mlx_migration" / "kokoro_cpp_export"


def load_weights():
    """Load safetensors weights."""
    weights = {}
    with safe_open(MODEL_PATH / "weights.safetensors", framework="pt") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)
    return weights


def print_tensor_stats(name: str, t: torch.Tensor):
    """Print tensor statistics for comparison."""
    t_np = t.detach().cpu().numpy().flatten()
    print(f"{name}:")
    print(f"  shape: {list(t.shape)}")
    print(f"  min: {t_np.min():.6f}, max: {t_np.max():.6f}, mean: {t_np.mean():.6f}")
    print(f"  first_5: {t_np[:5]}")
    print()


def adain(x, style, fc_weight, fc_bias):
    """Adaptive Instance Normalization.

    x: [batch, time, channels]
    style: [batch, style_dim]
    Returns: [batch, time, channels]
    """
    # Get gamma/beta from style
    fc_out = F.linear(style, fc_weight, fc_bias)  # [batch, 2*channels]
    channels = x.shape[-1]
    gamma = fc_out[:, :channels].unsqueeze(1)  # [batch, 1, channels]
    beta = fc_out[:, channels:].unsqueeze(1)  # [batch, 1, channels]

    # Instance norm (normalize over time dimension)
    mean = x.mean(dim=1, keepdim=True)
    var = x.var(dim=1, keepdim=True, unbiased=False)
    x_norm = (x - mean) / torch.sqrt(var + 1e-5)

    return (1 + gamma) * x_norm + beta


def conv1d_wn(x, weight, bias, stride=1, padding=0):
    """Conv1d with pre-folded weight norm.

    x: [batch, time, channels] -> conv -> [batch, time, channels]
    weight: [out_ch, in_ch, kernel]
    """
    x_t = x.transpose(1, 2)  # [batch, channels, time]
    out = F.conv1d(x_t, weight, bias, stride=stride, padding=padding)
    return out.transpose(1, 2)


def adain_resblk1d(x, style, weights, prefix, upsample=False, debug=False):
    """AdaIN ResBlock 1D.

    x: [batch, time, channels_in]
    style: [batch, style_dim]
    Returns: [batch, time (*2 if upsample), channels_out]
    """
    conv1_w = weights[f"{prefix}.conv1.weight"]
    conv1_b = weights[f"{prefix}.conv1.bias"]
    conv1x1_w = weights[f"{prefix}.conv1x1.weight"]
    conv1x1_b = weights[f"{prefix}.conv1x1.bias"]
    conv2_w = weights[f"{prefix}.conv2.weight"]
    conv2_b = weights[f"{prefix}.conv2.bias"]
    norm1_fc_w = weights[f"{prefix}.norm1.fc.weight"]
    norm1_fc_b = weights[f"{prefix}.norm1.fc.bias"]
    norm2_fc_w = weights[f"{prefix}.norm2.fc.weight"]
    norm2_fc_b = weights[f"{prefix}.norm2.fc.bias"]

    if debug:
        print(f"\n  --- {prefix} debug ---")
        print(f"  input x: [{x.min():.4f}, {x.max():.4f}], mean={x.mean():.4f}")

    # Correct order: norm1 -> actv -> pool -> conv1 -> norm2 -> actv -> conv2
    h = adain(x, style, norm1_fc_w, norm1_fc_b)
    if debug:
        print(f"  after adain1: [{h.min():.4f}, {h.max():.4f}], mean={h.mean():.4f}")
    h = F.leaky_relu(h, 0.2)
    if debug:
        print(
            f"  after leaky_relu1: [{h.min():.4f}, {h.max():.4f}], mean={h.mean():.4f}"
        )

    # Pool (upsample) in residual path - BEFORE conv1
    if upsample:
        h = h.repeat_interleave(2, dim=1)

    h = conv1d_wn(h, conv1_w, conv1_b, padding=1)
    if debug:
        print(f"  after conv1: [{h.min():.4f}, {h.max():.4f}], mean={h.mean():.4f}")

    # norm2 -> actv -> conv2
    h = adain(h, style, norm2_fc_w, norm2_fc_b)
    if debug:
        print(f"  after adain2: [{h.min():.4f}, {h.max():.4f}], mean={h.mean():.4f}")
    h = F.leaky_relu(h, 0.2)
    if debug:
        print(
            f"  after leaky_relu2: [{h.min():.4f}, {h.max():.4f}], mean={h.mean():.4f}"
        )
    h = conv1d_wn(h, conv2_w, conv2_b, padding=1)
    if debug:
        print(f"  after conv2: [{h.min():.4f}, {h.max():.4f}], mean={h.mean():.4f}")

    # Skip connection
    skip = x
    if upsample:
        skip = skip.repeat_interleave(2, dim=1)
    skip = conv1d_wn(skip, conv1x1_w, conv1x1_b, padding=0)
    if debug:
        print(f"  skip: [{skip.min():.4f}, {skip.max():.4f}], mean={skip.mean():.4f}")

    # Residual with rsqrt(2) scaling per StyleTTS2 upstream
    out = (h + skip) * (2**-0.5)
    if debug:
        print(f"  output: [{out.min():.4f}, {out.max():.4f}], mean={out.mean():.4f}")

    return out


def run_decoder(asr_features, f0, noise, style, weights, cpp_outputs=None):
    """Run decoder with intermediate outputs.

    cpp_outputs: dict with keys 'encode', 'decode_0', etc. for comparison
    """
    print("=== PYTHON Decoder Debug ===")
    print_tensor_stats("decoder_input_asr_features", asr_features)
    print_tensor_stats("decoder_input_f0", f0)
    print_tensor_stats("decoder_input_noise", noise)
    print_tensor_stats("decoder_input_style", style)

    py_outputs = {}  # Store Python outputs for comparison

    asr_len = asr_features.shape[1]

    # Step 1: F0/N conv (stride 2)
    f0_in = f0.unsqueeze(-1)  # [batch, len, 1]
    n_in = noise.unsqueeze(-1)

    f0_conv_w = weights["decoder.f0_conv.weight"]
    f0_conv_b = weights["decoder.f0_conv.bias"]
    n_conv_w = weights["decoder.n_conv.weight"]
    n_conv_b = weights["decoder.n_conv.bias"]

    f0_proc = conv1d_wn(f0_in, f0_conv_w, f0_conv_b, stride=2, padding=1)
    n_proc = conv1d_wn(n_in, n_conv_w, n_conv_b, stride=2, padding=1)

    # Step 2: ASR res
    asr_weight = weights["decoder.asr_res.weight"]
    asr_bias = weights["decoder.asr_res.bias"]
    asr_res = conv1d_wn(asr_features, asr_weight, asr_bias, padding=0)

    # Step 3: Match lengths
    f0_len = f0_proc.shape[1]
    if asr_len != f0_len:
        # Match to F0 length
        if f0_len > asr_len:
            scale = f0_len // asr_len
            asr_down = asr_features.repeat_interleave(scale, dim=1)[:, :f0_len]
            asr_res_down = asr_res.repeat_interleave(scale, dim=1)[:, :f0_len]
        else:
            stride = asr_len // f0_len
            indices = torch.arange(0, f0_len) * stride
            indices = indices.clamp(max=asr_len - 1)
            asr_down = asr_features[:, indices]
            asr_res_down = asr_res[:, indices]
    else:
        asr_down = asr_features
        asr_res_down = asr_res

    # Step 4: Encode input
    x = torch.cat([asr_down, f0_proc, n_proc], dim=-1)

    # Step 5: Encode block
    x = adain_resblk1d(x, style, weights, "decoder.encode", debug=True)
    print_tensor_stats("after_encode", x)
    py_outputs["encode"] = x.clone()
    np.save("/tmp/py_after_encode.npy", x.numpy())

    if cpp_outputs and "encode" in cpp_outputs:
        diff = (x - cpp_outputs["encode"]).abs()
        print(
            f"  C++ vs Python max_diff: {diff.max():.6f}, mean_diff: {diff.mean():.6f}"
        )

    # Step 6: Decode blocks
    for i in range(4):
        prefix = f"decoder.decode_{i}"
        is_upsample = i == 3

        # Concatenate residuals
        x = torch.cat([x, asr_res_down, f0_proc, n_proc], dim=-1)
        x = adain_resblk1d(x, style, weights, prefix, upsample=is_upsample)
        print_tensor_stats(f"after_decode_{i}", x)
        py_outputs[f"decode_{i}"] = x.clone()
        np.save(f"/tmp/py_after_decode_{i}.npy", x.numpy())

        if cpp_outputs and f"decode_{i}" in cpp_outputs:
            diff = (x - cpp_outputs[f"decode_{i}"]).abs()
            print(
                f"  C++ vs Python max_diff: {diff.max():.6f}, mean_diff: {diff.mean():.6f}"
            )

        if is_upsample:
            new_len = x.shape[1]
            asr_res_down = asr_res_down.repeat_interleave(2, dim=1)[:, :new_len]
            f0_proc = f0_proc.repeat_interleave(2, dim=1)[:, :new_len]
            n_proc = n_proc.repeat_interleave(2, dim=1)[:, :new_len]

    print_tensor_stats("decoder_output_to_generator", x)
    return x, py_outputs


def main():
    print("=" * 60)
    print("Kokoro Decoder Debug - Intermediate Values")
    print("=" * 60)

    # Load weights
    print("\nLoading weights...")
    weights = load_weights()
    print(f"Loaded {len(weights)} weight tensors")

    # Load voice style
    with safe_open(MODEL_PATH / "voices" / "af_bella.safetensors", framework="pt") as f:
        voice_embed = f.get_tensor("embedding")  # [1, 256]

    # Style is first 128 dims, speaker is last 128
    style = voice_embed[:, :128]  # [1, 128]
    print_tensor_stats("style", style)

    # Load saved tensors from C++ run
    try:
        asr_features = torch.from_numpy(np.load("/tmp/cpp_asr_features.npy"))
        f0 = torch.from_numpy(np.load("/tmp/cpp_f0.npy"))
        noise = torch.from_numpy(np.load("/tmp/cpp_noise.npy"))
        cpp_style = torch.from_numpy(np.load("/tmp/cpp_style.npy"))
        print("Loaded C++ saved tensors")
        print(f"  asr_features shape: {asr_features.shape}")
        print(f"  f0 shape: {f0.shape}")
        print(f"  noise shape: {noise.shape}")
        print(f"  cpp_style shape: {cpp_style.shape}")

        # Also load and compare encode/decode outputs
        cpp_encode = torch.from_numpy(np.load("/tmp/cpp_after_encode.npy"))
        cpp_decode_0 = torch.from_numpy(np.load("/tmp/cpp_after_decode_0.npy"))
        cpp_decode_1 = torch.from_numpy(np.load("/tmp/cpp_after_decode_1.npy"))
        cpp_decode_2 = torch.from_numpy(np.load("/tmp/cpp_after_decode_2.npy"))
        cpp_decode_3 = torch.from_numpy(np.load("/tmp/cpp_after_decode_3.npy"))

        print("\nC++ decode block outputs loaded for comparison")
    except Exception as e:
        print(f"Error loading C++ tensors: {e}")
        print("Using placeholder tensors (run kokoro_full_debug.py for real values)")
        asr_features = torch.randn(1, 62, 512) * 0.5
        f0 = torch.zeros(1, 124)
        f0[0, 5:60] = torch.linspace(100, 250, 55)  # Voiced region
        noise = torch.randn(1, 124) * 5 - 5
        cpp_encode = None
        cpp_decode_0 = cpp_decode_1 = cpp_decode_2 = cpp_decode_3 = None

    # Prepare cpp_outputs dict for comparison
    cpp_outputs = {}
    if cpp_encode is not None:
        cpp_outputs["encode"] = cpp_encode
        cpp_outputs["decode_0"] = cpp_decode_0
        cpp_outputs["decode_1"] = cpp_decode_1
        cpp_outputs["decode_2"] = cpp_decode_2
        cpp_outputs["decode_3"] = cpp_decode_3

    print("\n" + "=" * 60)
    print("Running Decoder...")
    print("=" * 60)

    with torch.no_grad():
        decoder_out, py_outputs = run_decoder(
            asr_features, f0, noise, style, weights, cpp_outputs
        )

    # Summary of differences
    print("\n" + "=" * 60)
    print("SUMMARY: C++ vs Python Differences")
    print("=" * 60)
    if cpp_outputs:
        for stage in ["encode", "decode_0", "decode_1", "decode_2", "decode_3"]:
            if stage in cpp_outputs and stage in py_outputs:
                cpp = cpp_outputs[stage]
                py = py_outputs[stage]
                diff = (py - cpp).abs()
                rel_err = diff / (cpp.abs() + 1e-8)
                print(f"{stage}:")
                print(f"  max_abs_diff: {diff.max():.6f}")
                print(f"  mean_abs_diff: {diff.mean():.6f}")
                print(f"  max_rel_err: {rel_err.max():.6f}")
                print(f"  cpp range: [{cpp.min():.4f}, {cpp.max():.4f}]")
                print(f"  py  range: [{py.min():.4f}, {py.max():.4f}]")
                print()
    else:
        print("No C++ outputs available for comparison")

    print("=" * 60)
    print("Debug complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
