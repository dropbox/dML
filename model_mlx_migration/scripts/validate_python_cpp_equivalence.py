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
Validate Python MLX and C++ Kokoro produce correlated audio when given SAME tokens.

This test proves the C++ model implementation is correct by:
1. Phonemizing text with misaki (Python) to get token_ids
2. Running Python MLX model with those tokens
3. Running C++ model with those same tokens via synthesize_tokens()
4. Computing Pearson correlation between the two audio outputs

Expected: correlation > 0.9 (same tokens → same audio)

Usage:
    python scripts/validate_python_cpp_equivalence.py
    python scripts/validate_python_cpp_equivalence.py --text "Hello world"
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Constants
MODEL_PATH = Path(__file__).parent.parent / "kokoro_cpp_export"
CPP_BINARY = Path(__file__).parent.parent / "src" / "kokoro" / "test_token_input"
VOICE = "af_bella"
SAMPLE_RATE = 24000


def get_tokens_from_text(text: str) -> tuple[str, list[int]]:
    """Get phonemes and token IDs from text using misaki."""
    try:
        from tools.pytorch_to_mlx.converters.models.kokoro_phonemizer import (
            phonemize_text,
        )

        phonemes, token_ids = phonemize_text(text)
        return phonemes, token_ids
    except ImportError:
        raise ImportError("misaki phonemizer not available")


def run_python_mlx(token_ids: list[int], voice: str) -> np.ndarray:
    """Run Python MLX Kokoro with given token IDs."""
    import mlx.core as mx

    from tools.pytorch_to_mlx.converters import KokoroConverter

    # Load model
    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()

    # Prepare tokens
    tokens_mx = mx.array([token_ids])

    # Load voice with correct phoneme length
    # token_ids includes BOS and EOS, so phoneme_length = len(token_ids) - 2
    phoneme_length = len(token_ids) - 2
    voice_data = converter.load_voice(voice, phoneme_length=phoneme_length)
    mx.eval(voice_data)

    # Make deterministic
    model.set_deterministic(True)

    # Synthesize
    audio = model.synthesize(tokens_mx, voice_data)
    mx.eval(audio)

    return np.array(audio).flatten()


def run_cpp_synthesize_tokens(
    token_ids: list[int], voice: str, model_path: Path
) -> np.ndarray:
    """Run C++ Kokoro with given token IDs via synthesize_tokens()."""
    if not CPP_BINARY.exists():
        raise FileNotFoundError(f"C++ binary not found: {CPP_BINARY}")

    # Build command: ./test_token_input <model_path> <voice> <tokens...>
    cmd = [str(CPP_BINARY), str(model_path), voice] + [str(t) for t in token_ids]

    # Run in temp directory to get output.wav
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            cmd,
            cwd=tmpdir,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"C++ stderr:\n{result.stderr}", file=sys.stderr)
            raise RuntimeError(f"C++ synthesis failed: {result.stderr}")

        # Parse JSON output
        try:
            output = json.loads(result.stdout)
            if output.get("status") != "success":
                raise RuntimeError(f"C++ synthesis error: {output.get('error')}")
        except json.JSONDecodeError:
            print(f"C++ stdout:\n{result.stdout}", file=sys.stderr)
            raise RuntimeError("Failed to parse C++ JSON output")

        # Load the generated WAV
        wav_path = Path(tmpdir) / output["wav_file"]
        audio, sr = sf.read(wav_path)

        if sr != SAMPLE_RATE:
            print(f"Warning: C++ sample rate {sr} != expected {SAMPLE_RATE}")

        return np.array(audio).flatten()


def compute_correlation(audio1: np.ndarray, audio2: np.ndarray) -> float:
    """Compute Pearson correlation between two audio arrays."""
    # Align lengths (use shorter)
    min_len = min(len(audio1), len(audio2))
    a1 = audio1[:min_len]
    a2 = audio2[:min_len]

    # Handle edge case of constant signals
    if np.std(a1) == 0 or np.std(a2) == 0:
        return 0.0

    # Pearson correlation
    corr = np.corrcoef(a1, a2)[0, 1]
    return float(corr)


def compute_rmse(audio1: np.ndarray, audio2: np.ndarray) -> float:
    """Compute RMSE between two audio arrays."""
    min_len = min(len(audio1), len(audio2))
    a1 = audio1[:min_len]
    a2 = audio2[:min_len]
    return float(np.sqrt(np.mean((a1 - a2) ** 2)))


def main():
    parser = argparse.ArgumentParser(description="Validate Python-C++ equivalence")
    parser.add_argument(
        "--text",
        type=str,
        default="Hello world, this is a test.",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=VOICE,
        help="Voice to use",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(MODEL_PATH),
        help="Path to C++ model export",
    )
    parser.add_argument(
        "--save-audio",
        action="store_true",
        help="Save audio files for manual inspection",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Python MLX vs C++ Equivalence Test")
    print("=" * 70)
    print()
    print(f'Text: "{args.text}"')
    print(f"Voice: {args.voice}")
    print(f"Model: {args.model_path}")
    print()

    # Step 1: Get tokens from misaki
    print("Step 1: Phonemizing with misaki...")
    try:
        phonemes, token_ids = get_tokens_from_text(args.text)
    except ImportError as e:
        print(f"SKIP: {e}")
        return 0

    print(f"  Phonemes: {phonemes}")
    print(
        f"  Tokens ({len(token_ids)}): {token_ids[:20]}{'...' if len(token_ids) > 20 else ''}"
    )
    print()

    # Step 2: Run Python MLX
    print("Step 2: Running Python MLX with tokens...")
    try:
        audio_python = run_python_mlx(token_ids, args.voice)
        print(f"  Samples: {len(audio_python)}")
        print(f"  Duration: {len(audio_python) / SAMPLE_RATE:.3f}s")
        print(f"  RMS: {np.sqrt(np.mean(audio_python**2)):.6f}")
    except Exception as e:
        print(f"FAIL: Python MLX error: {e}")
        return 1
    print()

    # Step 3: Run C++
    print("Step 3: Running C++ with same tokens...")
    try:
        audio_cpp = run_cpp_synthesize_tokens(
            token_ids, args.voice, Path(args.model_path)
        )
        print(f"  Samples: {len(audio_cpp)}")
        print(f"  Duration: {len(audio_cpp) / SAMPLE_RATE:.3f}s")
        print(f"  RMS: {np.sqrt(np.mean(audio_cpp**2)):.6f}")
    except FileNotFoundError as e:
        print(f"SKIP: {e}")
        print("Build C++ binary with:")
        print(
            "  cd src/kokoro && clang++ -std=c++17 -O2 -I/opt/homebrew/include -L/opt/homebrew/lib \\"
        )
        print(
            "      kokoro.cpp model.cpp g2p.cpp tokenizer.cpp test_token_input.cpp \\"
        )
        print("      -lmlx -lespeak-ng -o test_token_input")
        return 0
    except Exception as e:
        print(f"FAIL: C++ error: {e}")
        return 1
    print()

    # Step 4: Compare
    print("Step 4: Comparing outputs...")
    correlation = compute_correlation(audio_python, audio_cpp)
    rmse = compute_rmse(audio_python, audio_cpp)
    len_diff = abs(len(audio_python) - len(audio_cpp))

    print(
        f"  Length difference: {len_diff} samples ({len_diff / SAMPLE_RATE * 1000:.1f}ms)"
    )
    print(f"  Correlation: {correlation:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print()

    # Save audio if requested
    if args.save_audio:
        sf.write("/tmp/python_mlx_tokens.wav", audio_python, SAMPLE_RATE)
        sf.write("/tmp/cpp_tokens.wav", audio_cpp, SAMPLE_RATE)
        print("Saved:")
        print("  /tmp/python_mlx_tokens.wav")
        print("  /tmp/cpp_tokens.wav")
        print()

    # Verdict
    print("=" * 70)
    if correlation > 0.9:
        print(f"PASS: Correlation {correlation:.4f} > 0.9")
        print("Python MLX and C++ produce equivalent audio when given same tokens.")
        return 0
    elif correlation > 0.7:
        print(f"WARN: Correlation {correlation:.4f} is moderate (0.7-0.9)")
        print(
            "Audio is similar but not equivalent. Check for implementation differences."
        )
        return 1
    else:
        print(f"FAIL: Correlation {correlation:.4f} < 0.7")
        print("Python MLX and C++ produce different audio. Debug needed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
