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
CosyVoice2 End-to-End Validation Script

Tests the full text-to-speech pipeline:
1. Text tokenization
2. LLM speech token generation
3. Flow model mel generation
4. Vocoder audio generation

Usage:
    python scripts/validate_cosyvoice2_e2e.py
    python scripts/validate_cosyvoice2_e2e.py --text "Hello, world!"
    python scripts/validate_cosyvoice2_e2e.py --output output.wav
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="CosyVoice2 E2E Validation")
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(Path.home() / ".cache" / "cosyvoice2" / "cosyvoice2-0.5b"),
        help="Path to CosyVoice2 model directory",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Hello, this is a test of the CosyVoice2 text to speech system.",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output WAV file path (optional)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Maximum speech tokens to generate",
    )
    parser.add_argument(
        "--flow-steps",
        type=int,
        default=10,
        help="Number of flow ODE integration steps",
    )
    parser.add_argument(
        "--speaker-seed",
        type=int,
        default=42,
        help="Random seed for speaker embedding",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("CosyVoice2 End-to-End Validation")
    print("=" * 60)
    print()

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"ERROR: Model path not found: {model_path}")
        print("Run: python scripts/download_cosyvoice2.py")
        return 1

    # Check for required files
    required_files = ["llm.pt", "flow.pt", "hift.pt"]
    missing = [f for f in required_files if not (model_path / f).exists()]
    if missing:
        print(f"ERROR: Missing model files: {missing}")
        return 1

    print(f"Model path: {model_path}")
    print(f"Text: {args.text}")
    print()

    # Import here to allow help without importing
    from tools.pytorch_to_mlx.converters.models import CosyVoice2Model

    # Step 1: Load model
    print("[1/5] Loading CosyVoice2 model...")
    start_time = time.time()
    model = CosyVoice2Model.from_pretrained(model_path)
    load_time = time.time() - start_time
    print(f"      Loaded in {load_time:.2f}s")
    print()

    # Check tokenizer
    if model.tokenizer is None:
        print("ERROR: Tokenizer not loaded")
        return 1

    print(f"      Tokenizer vocab size: {model.tokenizer.vocab_size}")

    # Step 2: Tokenize text
    print("[2/5] Tokenizing text...")
    start_time = time.time()
    text_ids = model.tokenizer.encode(args.text)
    text_ids = text_ids[None, :]  # Add batch dimension
    tokenize_time = time.time() - start_time
    print(f"      Input tokens: {text_ids.shape[1]}")
    print(f"      Tokenized in {tokenize_time * 1000:.2f}ms")
    print()

    # Step 3: Generate speaker embedding
    print("[3/5] Generating speaker embedding...")
    speaker_embedding = model.tokenizer.random_speaker_embedding(seed=args.speaker_seed)
    speaker_embedding = speaker_embedding[None, :]  # Add batch dimension
    print(f"      Speaker embedding shape: {speaker_embedding.shape}")
    print(f"      Using random seed: {args.speaker_seed}")
    print()

    # Step 4: Generate speech tokens
    print("[4/5] Generating speech tokens...")
    start_time = time.time()
    speech_tokens = model.generate_speech_tokens(
        text_ids,
        max_length=args.max_tokens,
        temperature=1.0,
        top_k=25,
        top_p=0.8,
    )
    mx.eval(speech_tokens)
    token_gen_time = time.time() - start_time
    print(f"      Generated tokens: {speech_tokens.shape[1]}")
    print(f"      Token generation: {token_gen_time:.2f}s")
    print(f"      Tokens/second: {speech_tokens.shape[1] / token_gen_time:.1f}")
    print()

    # Step 5: Convert to mel and audio
    print("[5/5] Converting tokens to audio...")
    start_time = time.time()

    # Flow: tokens -> mel
    print("      Running flow model...")
    mel = model.tokens_to_mel(
        speech_tokens,
        speaker_embedding,
        num_steps=args.flow_steps,
    )
    mx.eval(mel)
    flow_time = time.time() - start_time
    print(f"      Mel shape: {mel.shape}")
    print(f"      Flow model: {flow_time:.2f}s")

    # Vocoder: mel -> audio
    print("      Running vocoder...")
    start_time = time.time()
    audio = model.mel_to_audio(mel)
    mx.eval(audio)
    vocoder_time = time.time() - start_time
    print(f"      Audio shape: {audio.shape}")
    print(f"      Vocoder: {vocoder_time:.2f}s")
    print()

    # Summary
    audio_samples = audio.shape[-1]
    audio_duration = audio_samples / model.config.sample_rate
    load_time + token_gen_time + flow_time + vocoder_time

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Input text: {len(args.text)} characters")
    print(f"Text tokens: {text_ids.shape[1]}")
    print(f"Speech tokens: {speech_tokens.shape[1]}")
    print(f"Audio duration: {audio_duration:.2f}s")
    print(f"Audio samples: {audio_samples}")
    print(f"Sample rate: {model.config.sample_rate} Hz")
    print()
    print("Timing breakdown:")
    print(f"  Model loading: {load_time:.2f}s")
    print(f"  Token generation: {token_gen_time:.2f}s")
    print(f"  Flow model: {flow_time:.2f}s")
    print(f"  Vocoder: {vocoder_time:.2f}s")
    print(f"  Total (excl. load): {token_gen_time + flow_time + vocoder_time:.2f}s")
    print()

    rtf = (
        (token_gen_time + flow_time + vocoder_time) / audio_duration
        if audio_duration > 0
        else float("inf")
    )
    print(f"Real-time factor: {rtf:.2f}x (< 1.0 means faster than real-time)")
    print()

    # Save audio if requested
    if args.output:
        try:
            import soundfile as sf

            audio_np = np.array(audio[0])
            # Normalize to [-1, 1]
            audio_np = audio_np / max(abs(audio_np.max()), abs(audio_np.min()), 1e-8)
            sf.write(args.output, audio_np, model.config.sample_rate)
            print(f"Audio saved to: {args.output}")
        except ImportError:
            print("WARNING: soundfile not installed. Cannot save audio.")
            print("Install with: pip install soundfile")
        except Exception as e:
            print(f"ERROR saving audio: {e}")

    print()
    print("Validation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
