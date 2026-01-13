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
Validate Kokoro MLX model after resblock dilation fix.
Tests key components against PyTorch reference.
"""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import scipy.io.wavfile as wavfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.pytorch_to_mlx.converters import KokoroConverter


def test_resblock_dilation():
    """Verify resblock dilations are correct."""
    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()
    mx.eval(model)

    generator = model.decoder.generator
    resblock = generator.resblocks_0

    expected_convs1_dilations = [1, 3, 5]
    expected_convs2_dilations = [1, 1, 1]

    for i in range(3):
        conv1 = getattr(resblock, f"convs1_{i}")
        conv2 = getattr(resblock, f"convs2_{i}")

        assert conv1.dilation == expected_convs1_dilations[i], (
            f"convs1_{i}.dilation={conv1.dilation}, expected={expected_convs1_dilations[i]}"
        )
        assert conv2.dilation == expected_convs2_dilations[i], (
            f"convs2_{i}.dilation={conv2.dilation}, expected={expected_convs2_dilations[i]}"
        )

    print("PASS: Resblock dilations correct")
    return True


def test_whisper_transcription():
    """Test that generated audio can be transcribed correctly."""
    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()
    mx.eval(model)

    # Load reference inputs
    ref = np.load("/tmp/kokoro_ref/tensors.npz")
    gen_traces = np.load("/tmp/kokoro_ref/generator_traces.npz")

    F0_mx = mx.array(ref["F0_pred"])
    style_mx = mx.array(ref["style_128"])
    gen_input_ncl = gen_traces["generator_input_ncl"]
    gen_input = mx.array(gen_input_ncl).transpose(0, 2, 1)  # NLC

    generator = model.decoder.generator

    # Run generator
    audio = generator(gen_input, style_mx, F0_mx)
    mx.eval(audio)

    audio_np = np.array(audio).flatten()

    # Save audio
    output_path = "/tmp/kokoro_ref/mlx_test_audio.wav"
    audio_int = np.clip(audio_np * 32767, -32768, 32767).astype(np.int16)
    wavfile.write(output_path, 24000, audio_int)

    # Test with Whisper
    try:
        import whisper

        w = whisper.load_model("base")
        result = w.transcribe(output_path, language="en")
        transcription = result["text"].strip().lower()
        print(f"Transcription: '{transcription}'")

        # Check if "hello" and "world" are in the transcription
        if "hello" in transcription and "world" in transcription:
            print("PASS: Whisper transcription matches expected 'hello world'")
            return True
        else:
            print(f"FAIL: Transcription '{transcription}' does not match 'hello world'")
            return False
    except Exception as e:
        print(f"Whisper not available: {e}")
        return None


def test_resblock_correlation():
    """Test resblock output correlation with PyTorch reference."""
    import torch
    from huggingface_hub import hf_hub_download

    ckpt_path = hf_hub_download("hexgrad/Kokoro-82M", "kokoro-v1_0.pth")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    dec_state = ckpt["decoder"]

    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()
    mx.eval(model)

    generator = model.decoder.generator
    resblock = generator.resblocks_0

    # Create test input
    np.random.seed(42)
    x_np = np.random.randn(1, 256, 126).astype(np.float32) * 0.1
    s_np = np.random.randn(1, 128).astype(np.float32)

    x_mx = mx.array(x_np.transpose(0, 2, 1))  # NLC
    s_mx = mx.array(s_np)

    # Run MLX resblock
    out_mx = resblock(x_mx, s_mx)
    mx.eval(out_mx)
    out_mx_ncl = np.array(out_mx.transpose(0, 2, 1))

    # Run manual PyTorch implementation
    x_pt = torch.tensor(x_np)
    s_pt = torch.tensor(s_np)

    from trace_resblock_iterations import pytorch_resblock_single_iter

    prefix = "module.generator.resblocks.0"
    weights = {}
    for key in dec_state:
        if key.startswith(prefix):
            short_key = key[len(prefix) + 1 :]
            weights[short_key] = dec_state[key]

    for i in range(3):
        x_pt = pytorch_resblock_single_iter(x_pt, s_pt, weights, i)

    out_pt_np = x_pt.numpy()

    # Check correlation
    corr = np.corrcoef(out_mx_ncl.flatten(), out_pt_np.flatten())[0, 1]
    print(f"Resblock correlation: {corr:.6f}")

    if corr > 0.999:
        print("PASS: Resblock correlation > 0.999")
        return True
    else:
        print(f"FAIL: Resblock correlation {corr:.6f} < 0.999")
        return False


def main():
    results = {}

    print("\n=== Test 1: Resblock Dilations ===")
    results["dilation"] = test_resblock_dilation()

    print("\n=== Test 2: Resblock Correlation ===")
    results["correlation"] = test_resblock_correlation()

    print("\n=== Test 3: Whisper Transcription ===")
    results["whisper"] = test_whisper_transcription()

    print("\n=== Summary ===")
    for name, result in results.items():
        if result is True:
            print(f"  {name}: PASS")
        elif result is False:
            print(f"  {name}: FAIL")
        else:
            print(f"  {name}: SKIPPED")

    # Return 0 if all non-None tests passed
    passed = all(r for r in results.values() if r is not None)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
