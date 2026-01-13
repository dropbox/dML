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
Generate Python MLX reference for C++ comparison.

This script uses the MLX Kokoro model to:
1. Tokenize a text string using Kokoro phonemizer
2. Generate audio in deterministic mode
3. Save the tokens and audio for C++ comparison

Usage:
  python scripts/generate_cpp_reference.py [--text "Hello world"] [--voice af_bella]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Generate C++ reference from Python MLX")
    parser.add_argument("--text", default="Hello world", help="Text to synthesize")
    parser.add_argument("--voice", default="af_bella", help="Voice to use")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/kokoro_cpp_ref"),
        help="Output directory",
    )
    args = parser.parse_args()

    import sys
    sys.path.insert(0, "/Users/ayates/model_mlx_migration")

    import mlx.core as mx

    from tools.pytorch_to_mlx.converters import KokoroConverter
    from tools.pytorch_to_mlx.converters.models.kokoro_phonemizer import phonemize_text

    print("Loading MLX Kokoro model...")
    converter = KokoroConverter()
    model, config, state_dict = converter.load_from_hf("hexgrad/Kokoro-82M")
    print(f"Model loaded: {type(model)}")

    # Phonemize text to get token IDs
    print(f"Text: {args.text!r}")
    phonemes, tokens = phonemize_text(args.text)
    print(f"Phonemes: {phonemes!r}")
    print(f"Tokens ({len(tokens)}): {tokens}")

    # Load voice pack and select embedding based on phoneme length
    print(f"Loading voice: {args.voice}")
    voice_pack = converter.load_voice_pack(args.voice)
    voice_emb = converter.select_voice_embedding(voice_pack, len(phonemes))
    print(f"Voice embedding shape: {voice_emb.shape}")

    # Set deterministic mode (zeros instead of random noise)
    model.set_deterministic(True)

    # Enable debug tensor saving if requested
    import os
    if os.environ.get("SAVE_DEBUG_TENSORS"):
        model.save_debug_tensors = True
        # Pass to BERT module for encoder debug saves
        if hasattr(model, 'bert'):
            model.bert.save_debug_tensors = True
        # Pass to decoder->generator chain
        if hasattr(model, 'decoder') and hasattr(model.decoder, 'generator'):
            model.decoder.generator.save_debug_tensors = True

    # Convert tokens to MLX array [batch, seq_len]
    input_ids = mx.array([tokens])  # [1, seq_len]

    # Generate audio
    print("Generating audio in deterministic mode...")
    audio = model(input_ids, voice_emb)
    mx.eval(audio)
    audio_np = np.array(audio).flatten()

    # Save reference
    args.output_dir.mkdir(parents=True, exist_ok=True)

    np.save(args.output_dir / "audio.npy", audio_np)

    metadata = {
        "tokens": tokens,
        "voice": args.voice,
        "text": args.text,
        "phonemes": phonemes,
        "deterministic": True,
        "num_samples": len(audio_np),
    }
    with open(args.output_dir / "tokens.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Calculate stats
    rms = float(np.sqrt(np.mean(audio_np**2)))
    max_amp = float(np.max(np.abs(audio_np)))

    print(f"\nSaved to {args.output_dir}")
    print(f"  Audio shape: {audio_np.shape}")
    print(f"  RMS: {rms:.6f}")
    print(f"  Max amplitude: {max_amp:.6f}")
    print(f"  Duration: {len(audio_np)/24000:.3f}s")

    # Print command for running C++
    token_str = " ".join(map(str, tokens))
    print("\nTo run C++ with same tokens:")
    print(f"  cd src/kokoro && ./test_token_input ../../kokoro_cpp_export {args.voice} {token_str}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
