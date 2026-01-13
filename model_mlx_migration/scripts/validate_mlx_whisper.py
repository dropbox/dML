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
Validate MLX Kokoro output with Whisper transcription.
"""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import scipy.io.wavfile as wavfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.pytorch_to_mlx.converters import KokoroConverter


def main():
    # Load MLX model
    print("=== Loading MLX Kokoro Model ===")
    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()
    mx.eval(model)

    # Load reference tensors
    ref_path = Path("/tmp/kokoro_ref/tensors.npz")
    if not ref_path.exists():
        print("Reference not found")
        return 1
    ref = np.load(ref_path)

    print("\n=== Reference Info ===")
    meta_path = Path("/tmp/kokoro_ref/metadata.json")
    if meta_path.exists():
        import json

        meta = json.loads(meta_path.read_text())
        print(f"Text: {meta.get('text', 'unknown')}")
        print(f"Voice: {meta.get('voice', 'unknown')}")

    # Get inputs
    asr_ncl = ref["asr_ncl"]  # [1, 512, 63]
    F0_pred = ref["F0_pred"]  # [1, 126]
    N_pred = ref["N_pred"]  # [1, 126]
    style_128 = ref["style_128"]  # [1, 128]

    # Convert to MLX
    asr_nlc = mx.array(asr_ncl).transpose(0, 2, 1)
    F0_mx = mx.array(F0_pred)
    N_mx = mx.array(N_pred)
    style_mx = mx.array(style_128)

    print("\nInput shapes:")
    print(f"  asr_nlc: {asr_nlc.shape}")
    print(
        f"  F0: {F0_mx.shape}, range: [{float(mx.min(F0_mx)):.2f}, {float(mx.max(F0_mx)):.2f}]"
    )
    print(f"  N: {N_mx.shape}")
    print(f"  style: {style_mx.shape}")

    # Run decoder
    print("\n=== Running MLX Decoder ===")
    audio = model.decoder(asr_nlc, F0_mx, N_mx, style_mx)
    mx.eval(audio)

    audio_np = np.array(audio).flatten()
    print(f"Audio shape: {audio_np.shape}")
    print(f"Audio range: [{audio_np.min():.4f}, {audio_np.max():.4f}]")
    print(f"Audio std: {audio_np.std():.4f}")
    print(f"Audio duration: {len(audio_np) / 24000:.2f}s")

    # Save audio
    output_path = Path("/tmp/kokoro_ref/mlx_audio.wav")
    audio_int = np.clip(audio_np * 32767, -32768, 32767).astype(np.int16)
    wavfile.write(output_path, 24000, audio_int)
    print(f"\nSaved: {output_path}")

    # Also save reference audio for comparison
    ref_audio = ref["audio"].flatten()
    ref_output_path = Path("/tmp/kokoro_ref/ref_audio.wav")
    ref_int = np.clip(ref_audio * 32767, -32768, 32767).astype(np.int16)
    wavfile.write(ref_output_path, 24000, ref_int)
    print(f"Saved reference: {ref_output_path}")

    # Compare statistics
    print("\n=== Audio Comparison ===")
    print(f"MLX std: {audio_np.std():.4f}")
    print(f"Ref std: {ref_audio.std():.4f}")
    print(f"Ratio: {audio_np.std() / ref_audio.std():.2f}x")

    min_len = min(len(audio_np), len(ref_audio))
    corr = np.corrcoef(audio_np[:min_len], ref_audio[:min_len])[0, 1]
    print(f"Correlation: {corr:.6f}")

    # Try Whisper transcription
    print("\n=== Whisper Transcription ===")
    try:
        import whisper

        print("Loading Whisper model...")
        whisper_model = whisper.load_model("base")

        print("\nTranscribing MLX audio...")
        result_mlx = whisper_model.transcribe(str(output_path), language="en")
        print(f"MLX transcription: {result_mlx['text']}")

        print("\nTranscribing reference audio...")
        result_ref = whisper_model.transcribe(str(ref_output_path), language="en")
        print(f"Reference transcription: {result_ref['text']}")

    except ImportError:
        print("Whisper not installed - skipping transcription")
        print("Install with: pip install openai-whisper")
    except Exception as e:
        print(f"Whisper error: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
