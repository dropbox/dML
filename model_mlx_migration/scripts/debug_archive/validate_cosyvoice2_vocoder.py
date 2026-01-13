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
Validate CosyVoice2 HiFi-GAN Vocoder MLX vs PyTorch

Validates that MLX vocoder loads weights correctly and produces reasonable output.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import numpy as np
import torch

MODEL_PATH = "/Users/ayates/.cache/cosyvoice2/cosyvoice2-0.5b/hift.pt"


def validate_weight_loading():
    """Validate that weights are loaded correctly."""
    from tools.pytorch_to_mlx.converters.models import HiFiGANVocoder

    print("=" * 60)
    print("WEIGHT LOADING VALIDATION")
    print("=" * 60)

    # Load MLX model
    vocoder = HiFiGANVocoder.from_pretrained(MODEL_PATH)

    # Load PyTorch weights
    pt_state = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

    results = []

    # Validate conv_pre weights
    print("\n1. conv_pre weights")
    g_pt = pt_state["conv_pre.parametrizations.weight.original0"].numpy()
    v_pt = pt_state["conv_pre.parametrizations.weight.original1"].numpy()
    g_mlx = np.array(vocoder.conv_pre.weight_g)
    v_mlx = np.array(vocoder.conv_pre.weight_v)

    g_match = np.allclose(g_pt, g_mlx, atol=1e-6)
    v_match = np.allclose(v_pt, v_mlx, atol=1e-6)
    results.append(("conv_pre.weight_g", g_match))
    results.append(("conv_pre.weight_v", v_match))
    print(f"   weight_g: {'PASS' if g_match else 'FAIL'}")
    print(f"   weight_v: {'PASS' if v_match else 'FAIL'}")

    # Validate conv_pre bias
    b_pt = pt_state["conv_pre.bias"].numpy()
    b_mlx = np.array(vocoder.conv_pre.bias)
    b_match = np.allclose(b_pt, b_mlx, atol=1e-6)
    results.append(("conv_pre.bias", b_match))
    print(f"   bias: {'PASS' if b_match else 'FAIL'}")

    # Validate conv_post weights
    print("\n2. conv_post weights")
    g_pt = pt_state["conv_post.parametrizations.weight.original0"].numpy()
    v_pt = pt_state["conv_post.parametrizations.weight.original1"].numpy()
    g_mlx = np.array(vocoder.conv_post.weight_g)
    v_mlx = np.array(vocoder.conv_post.weight_v)

    g_match = np.allclose(g_pt, g_mlx, atol=1e-6)
    v_match = np.allclose(v_pt, v_mlx, atol=1e-6)
    results.append(("conv_post.weight_g", g_match))
    results.append(("conv_post.weight_v", v_match))
    print(f"   weight_g: {'PASS' if g_match else 'FAIL'}")
    print(f"   weight_v: {'PASS' if v_match else 'FAIL'}")

    # Validate source module
    print("\n3. m_source weights")
    w_pt = pt_state["m_source.l_linear.weight"].numpy()
    w_mlx = np.array(vocoder.m_source.l_linear.weight)
    w_match = np.allclose(w_pt, w_mlx, atol=1e-6)
    results.append(("m_source.weight", w_match))
    print(f"   weight: {'PASS' if w_match else 'FAIL'}")

    # Summary
    print("\n" + "=" * 60)
    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"WEIGHT LOADING: {passed}/{total} checks passed")
    print("=" * 60)

    return all(r for _, r in results)


def validate_forward_pass():
    """Validate forward pass produces reasonable output."""
    from tools.pytorch_to_mlx.converters.models import HiFiGANVocoder

    print("\n" + "=" * 60)
    print("FORWARD PASS VALIDATION")
    print("=" * 60)

    # Load model
    vocoder = HiFiGANVocoder.from_pretrained(MODEL_PATH)

    # Create test input (random mel spectrogram)
    np.random.seed(42)
    mel_np = np.random.randn(1, 50, 80).astype(np.float32) * 0.1
    mel = mx.array(mel_np)

    # Forward pass
    audio = vocoder(mel)
    audio_np = np.array(audio)

    print("\n1. Shape validation")
    print(f"   Input mel: {mel.shape}")
    print(f"   Output audio: {audio.shape}")
    # Expected: 50 * 128 * 18 ~ 115200
    expected_min = 50 * 100 * 18
    expected_max = 50 * 150 * 18
    shape_ok = expected_min < audio_np.shape[1] < expected_max
    print(f"   Shape OK: {'PASS' if shape_ok else 'FAIL'}")

    print("\n2. Value range validation")
    print(f"   Audio min: {audio_np.min():.4f}")
    print(f"   Audio max: {audio_np.max():.4f}")
    print(f"   Audio mean: {audio_np.mean():.4f}")
    print(f"   Audio std: {audio_np.std():.4f}")
    # After tanh, should be in [-1, 1]
    range_ok = -1.0 <= audio_np.min() and audio_np.max() <= 1.0
    print(f"   Range [-1, 1]: {'PASS' if range_ok else 'FAIL'}")

    print("\n3. Non-zero output")
    nonzero = audio_np.std() > 0.01
    print(f"   Std > 0.01: {'PASS' if nonzero else 'FAIL'}")

    print("\n" + "=" * 60)
    all_pass = shape_ok and range_ok and nonzero
    print(f"FORWARD PASS: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 60)

    return all_pass


def main():
    """Run all validations."""
    print("\nCosyVoice2 HiFi-GAN Vocoder Validation")
    print(f"Model: {MODEL_PATH}")
    print()

    weight_ok = validate_weight_loading()
    forward_ok = validate_forward_pass()

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Weight loading: {'PASS' if weight_ok else 'FAIL'}")
    print(f"Forward pass: {'PASS' if forward_ok else 'FAIL'}")
    print(f"Overall: {'PASS' if weight_ok and forward_ok else 'FAIL'}")
    print("=" * 60)

    return 0 if weight_ok and forward_ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
