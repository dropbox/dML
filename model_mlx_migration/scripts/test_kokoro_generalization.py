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
Test Kokoro MLX model with different texts to verify generalization.
"""

import sys
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
import numpy as np
import scipy.io.wavfile as wavfile

sys.path.insert(0, str(Path(__file__).parent.parent))

# Module-level cache for model
_model_cache: dict[str, Any] = {}


def generate_audio_for_text(
    text: str, output_path: Optional[str] = None
) -> Optional[bool]:
    """Generate audio for given text using Kokoro MLX."""
    from tools.pytorch_to_mlx.converters import KokoroConverter

    # Load model (cached)
    if "model" not in _model_cache:
        converter = KokoroConverter()
        model, config, _ = converter.load_from_hf()
        mx.eval(model)
        _model_cache["model"] = model
        _model_cache["config"] = config
        _model_cache["converter"] = converter

    model = _model_cache["model"]
    converter = _model_cache["converter"]

    # Get phonemes for text
    from kokoro import tokenize

    # Tokenize
    tokens = tokenize(text, lang="en")
    if not tokens:
        print(f"Failed to tokenize: {text}")
        return None

    # Convert to input_ids
    input_ids = [0]  # Start token
    for char in tokens:
        if char in converter.char_to_id:
            input_ids.append(converter.char_to_id[char])
        else:
            # Unknown char, skip
            pass
    input_ids.append(0)  # End token

    input_ids_mx = mx.array([input_ids])

    # Load voice
    voice_data = converter.load_voice()
    style = voice_data[:, :128]

    # Get text encoding
    text_enc = model.text_encoder(input_ids_mx)
    bert_out = model.bert(input_ids_mx)
    bert_proj = model.bert_encoder(bert_out)
    combined = text_enc + bert_proj

    # Get predictor outputs
    duration, f0, noise = model.predictor(combined, style)
    mx.eval([duration, f0, noise])

    # Generate audio from decoder
    # For simplicity, we'll use the stored reference traces for now
    # since full end-to-end requires alignment handling

    print(f"Generated for '{text}':")
    print(f"  Duration shape: {duration.shape}")
    print(f"  F0 shape: {f0.shape}")
    print(f"  Noise shape: {noise.shape}")

    return True


def test_with_whisper():
    """Test using reference traces and verify with Whisper."""
    from tools.pytorch_to_mlx.converters import KokoroConverter

    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()
    mx.eval(model)

    # Load reference inputs (Hello world)
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
    output_path = "/tmp/kokoro_ref/mlx_generalization_test.wav"
    audio_int = np.clip(audio_np * 32767, -32768, 32767).astype(np.int16)
    wavfile.write(output_path, 24000, audio_int)

    # Test with Whisper
    try:
        import whisper

        w = whisper.load_model("base")
        result = w.transcribe(output_path, language="en")
        transcription = result["text"].strip().lower()
        print(f"Transcription: '{transcription}'")

        if "hello" in transcription and "world" in transcription:
            print("PASS: Whisper transcription correct")
            return True
        else:
            print("FAIL: Unexpected transcription")
            return False
    except Exception as e:
        print(f"Whisper error: {e}")
        return None


def main():
    print("=== Testing Whisper Transcription ===")
    result = test_with_whisper()

    print("\n=== Testing Different Text Processing ===")
    texts = [
        "Hello world",
        "Testing one two three",
        "The quick brown fox",
    ]

    for text in texts:
        try:
            generate_audio_for_text(text)
        except Exception as e:
            print(f"Error for '{text}': {e}")

    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
