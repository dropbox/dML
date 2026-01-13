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
Export PyTorch encode block intermediate tensors for comparison.
Must be run with: /tmp/kokoro_env/bin/python scripts/export_encode_intermediates.py
"""

from pathlib import Path

import numpy as np
import torch


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().float().numpy()


def main():
    # Load existing reference
    ref_dir = Path("/tmp/kokoro_ref_decoder")
    ref = np.load(ref_dir / "tensors.npz")

    # Load model
    from kokoro import KModel
    model = KModel().eval()

    # Get inputs
    asr_nlc = torch.from_numpy(ref["asr_nlc"].astype(np.float32))  # [1, 63, 512]
    f0_proc = torch.from_numpy(ref["F0_proc"].astype(np.float32))  # [1, 63, 1]
    n_proc = torch.from_numpy(ref["N_proc"].astype(np.float32))    # [1, 63, 1]
    style_128 = torch.from_numpy(ref["style_128"].astype(np.float32))  # [1, 128]

    decoder = model.decoder

    print("=" * 72)
    print("PyTorch Encode Block Intermediate Export")
    print("=" * 72)

    # Initial concat - input to encode
    x = torch.cat([asr_nlc, f0_proc, n_proc], dim=-1)  # NLC format [1, 63, 514]
    print(f"Input x shape: {x.shape}")

    intermediates = {}
    intermediates["encode_input"] = _to_numpy(x)

    # Get encode block (it's an AdainResBlk1d)
    encode = decoder.encode

    # The AdainResBlk1d structure in PyTorch:
    # norm1 -> actv -> conv1 -> norm2 -> actv -> conv2 -> residual + skip
    # BUT in PyTorch, the conv layers work in NCL format

    # Convert to NCL for PyTorch operations
    x_ncl = x.transpose(1, 2)  # [1, 514, 63]
    s = style_128  # [1, 128]

    # === Shortcut path ===
    # shortcut uses conv1x1 directly in NCL format
    skip_ncl = encode.conv1x1(x_ncl) if encode.conv1x1 is not None else x_ncl
    intermediates["encode_skip_ncl"] = _to_numpy(skip_ncl)
    print(f"Skip NCL shape: {skip_ncl.shape}")

    # === Residual path ===
    # norm1(x, s) - operates on NCL
    h_norm1 = encode.norm1(x_ncl, s)
    intermediates["encode_norm1_out_ncl"] = _to_numpy(h_norm1)
    print(f"norm1 out NCL: {h_norm1.shape}, range=[{h_norm1.min():.4f}, {h_norm1.max():.4f}]")

    # actv (LeakyReLU)
    h_actv1 = encode.actv(h_norm1)
    intermediates["encode_actv1_out_ncl"] = _to_numpy(h_actv1)
    print(f"actv1 out NCL: {h_actv1.shape}, range=[{h_actv1.min():.4f}, {h_actv1.max():.4f}]")

    # conv1
    h_conv1 = encode.conv1(h_actv1)
    intermediates["encode_conv1_out_ncl"] = _to_numpy(h_conv1)
    print(f"conv1 out NCL: {h_conv1.shape}, range=[{h_conv1.min():.4f}, {h_conv1.max():.4f}]")

    # norm2(h, s)
    h_norm2 = encode.norm2(h_conv1, s)
    intermediates["encode_norm2_out_ncl"] = _to_numpy(h_norm2)
    print(f"norm2 out NCL: {h_norm2.shape}, range=[{h_norm2.min():.4f}, {h_norm2.max():.4f}]")

    # actv2
    h_actv2 = encode.actv(h_norm2)
    intermediates["encode_actv2_out_ncl"] = _to_numpy(h_actv2)
    print(f"actv2 out NCL: {h_actv2.shape}, range=[{h_actv2.min():.4f}, {h_actv2.max():.4f}]")

    # conv2
    h_conv2 = encode.conv2(h_actv2)
    intermediates["encode_conv2_out_ncl"] = _to_numpy(h_conv2)
    print(f"conv2 out NCL: {h_conv2.shape}, range=[{h_conv2.min():.4f}, {h_conv2.max():.4f}]")

    # Combine: (h + skip) / sqrt(2)
    out_ncl = (h_conv2 + skip_ncl) / np.sqrt(2)
    intermediates["encode_output_ncl"] = _to_numpy(out_ncl)
    print(f"encode out NCL: {out_ncl.shape}, range=[{out_ncl.min():.4f}, {out_ncl.max():.4f}]")

    # Convert to NLC for comparison
    out_nlc = out_ncl.transpose(1, 2)
    intermediates["encode_output_nlc"] = _to_numpy(out_nlc)
    print(f"encode out NLC: {out_nlc.shape}")

    # Also trace norm1 internals (AdaIN)
    print("\n--- norm1 (AdaIN) internals ---")
    norm1 = encode.norm1

    # norm1.fc(s) -> gamma, beta
    h_fc = norm1.fc(s)
    gamma_ncl, beta_ncl = h_fc.chunk(2, dim=1)
    print(f"norm1 fc output: {h_fc.shape}")
    print(f"gamma_ncl: {gamma_ncl.shape}, range=[{gamma_ncl.min():.4f}, {gamma_ncl.max():.4f}]")
    print(f"beta_ncl: {beta_ncl.shape}, range=[{beta_ncl.min():.4f}, {beta_ncl.max():.4f}]")

    intermediates["norm1_gamma_ncl"] = _to_numpy(gamma_ncl)
    intermediates["norm1_beta_ncl"] = _to_numpy(beta_ncl)

    # Instance norm part
    # PyTorch InstanceNorm1d expects NCL format
    # It normalizes over L dimension for each (N, C)
    mean_ncl = x_ncl.mean(dim=2, keepdim=True)
    var_ncl = x_ncl.var(dim=2, keepdim=True, unbiased=True)  # ddof=1
    x_norm_ncl = (x_ncl - mean_ncl) / torch.sqrt(var_ncl + 1e-5)

    print(f"mean_ncl: {mean_ncl.shape}")
    print(f"var_ncl: {var_ncl.shape}")
    print(f"x_norm_ncl: {x_norm_ncl.shape}, range=[{x_norm_ncl.min():.4f}, {x_norm_ncl.max():.4f}]")

    intermediates["norm1_mean_ncl"] = _to_numpy(mean_ncl)
    intermediates["norm1_var_ncl"] = _to_numpy(var_ncl)
    intermediates["norm1_x_norm_ncl"] = _to_numpy(x_norm_ncl)

    # Apply scale and shift
    # gamma, beta are [batch, channels], need to add L dimension
    gamma_ncl_expanded = gamma_ncl[:, :, None]  # [batch, channels, 1]
    beta_ncl_expanded = beta_ncl[:, :, None]
    out_adain = (1 + gamma_ncl_expanded) * x_norm_ncl + beta_ncl_expanded
    print(f"AdaIN output: {out_adain.shape}, range=[{out_adain.min():.4f}, {out_adain.max():.4f}]")

    intermediates["norm1_out_manual_ncl"] = _to_numpy(out_adain)

    # Save all
    out_path = ref_dir / "encode_intermediates.npz"
    np.savez(out_path, **intermediates)
    print(f"\nSaved to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
