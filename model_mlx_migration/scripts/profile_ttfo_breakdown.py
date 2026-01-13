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
TTFO (Time-to-First-Output) Breakdown Profiler
==============================================

Measures the contribution of each phase to TTFO:
1. Mel spectrogram computation
2. Encoder pass
3. Decoder first token

This helps identify the bottleneck for streaming latency optimization.

Usage:
    python scripts/profile_ttfo_breakdown.py
    python scripts/profile_ttfo_breakdown.py --model tiny  # Test smaller model
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class TTFOBreakdown:
    """Breakdown of TTFO into component phases."""
    model_name: str
    audio_duration_s: float

    # Phase timings (ms)
    mel_time_ms: float = 0.0
    encoder_time_ms: float = 0.0
    decoder_first_token_ms: float = 0.0

    # Total
    total_ttfo_ms: float = 0.0

    # Additional info
    warmup_time_ms: float = 0.0
    num_encoder_layers: int = 0
    num_decoder_layers: int = 0

    def __str__(self) -> str:
        total = self.mel_time_ms + self.encoder_time_ms + self.decoder_first_token_ms
        return f"""
TTFO Breakdown for {self.model_name}
{'=' * 60}
Audio Duration: {self.audio_duration_s:.2f}s

Phase Timings:
  1. Mel Spectrogram:     {self.mel_time_ms:8.1f} ms ({100*self.mel_time_ms/total:5.1f}%)
  2. Encoder:             {self.encoder_time_ms:8.1f} ms ({100*self.encoder_time_ms/total:5.1f}%)
  3. Decoder (1st token): {self.decoder_first_token_ms:8.1f} ms ({100*self.decoder_first_token_ms/total:5.1f}%)
  ----------------------------------------------------------------
  TOTAL TTFO:             {total:8.1f} ms

Model Architecture:
  Encoder layers: {self.num_encoder_layers}
  Decoder layers: {self.num_decoder_layers}

P3 Gate: TTFO < 500ms -> {'PASS' if total < 500 else 'FAIL'}
"""


def load_test_audio(audio_path: Optional[str] = None, duration: float = 5.0) -> np.ndarray:
    """Load or generate test audio."""
    import mlx.core as mx

    if audio_path:
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        from tools.whisper_mlx.audio import load_audio
        audio = load_audio(str(path))
        return np.array(audio) if isinstance(audio, mx.array) else audio
    else:
        # Use JFK test audio if available
        test_audio_path = Path(__file__).parent.parent / "data" / "test" / "jfk.wav"
        if test_audio_path.exists():
            from tools.whisper_mlx.audio import load_audio
            audio = load_audio(str(test_audio_path))
            return np.array(audio) if isinstance(audio, mx.array) else audio

        # Generate synthetic audio
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        audio = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        return audio


def profile_ttfo_breakdown(
    model_name: str = "large-v3",
    audio: Optional[np.ndarray] = None,
    warmup: bool = True,
) -> TTFOBreakdown:
    """
    Profile the TTFO breakdown for a Whisper model.

    Args:
        model_name: Whisper model name
        audio: Audio array (uses test audio if None)
        warmup: Whether to do a warmup run first

    Returns:
        TTFOBreakdown with timing details
    """
    import mlx.core as mx
    from tools.whisper_mlx import WhisperMLX
    from tools.whisper_mlx.audio import log_mel_spectrogram

    # Load model
    print(f"Loading model: {model_name}")
    model = WhisperMLX.from_pretrained(model_name)

    # Get model info
    num_encoder_layers = model.config.n_audio_layer
    num_decoder_layers = model.config.n_text_layer

    # Load audio
    if audio is None:
        audio = load_test_audio()

    audio_duration = len(audio) / 16000
    print(f"Audio duration: {audio_duration:.2f}s ({len(audio)} samples)")

    # Create breakdown tracker
    breakdown = TTFOBreakdown(
        model_name=model_name,
        audio_duration_s=audio_duration,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
    )

    # Warmup run (important for MLX compilation)
    warmup_time = 0.0
    if warmup:
        print("Warmup run...")
        warmup_start = time.perf_counter()
        short_audio = audio[:int(16000 * 2)]  # 2 seconds
        mel = log_mel_spectrogram(short_audio, n_mels=model.config.n_mels)
        mel = mx.pad(mel, [(0, model.config.n_audio_ctx * 2 - mel.shape[0]), (0, 0)])
        mel = mel[None]
        audio_features = model.embed_audio(mel)
        mx.eval(audio_features)
        warmup_time = (time.perf_counter() - warmup_start) * 1000
        breakdown.warmup_time_ms = warmup_time
        print(f"Warmup complete: {warmup_time:.1f}ms")

    # Profile each phase
    print("\nProfiling TTFO breakdown...")

    # Phase 1: Mel spectrogram
    print("  Phase 1: Mel spectrogram...")
    mel_start = time.perf_counter()
    mel = log_mel_spectrogram(audio, n_mels=model.config.n_mels)
    # Pad to standard 30s length
    target_len = model.config.n_audio_ctx * 2  # 3000 frames
    if mel.shape[0] < target_len:
        mel = mx.pad(mel, [(0, target_len - mel.shape[0]), (0, 0)])
    elif mel.shape[0] > target_len:
        mel = mel[:target_len, :]
    mel = mel[None]  # Add batch dimension
    mx.eval(mel)  # Ensure computation complete
    mel_end = time.perf_counter()
    breakdown.mel_time_ms = (mel_end - mel_start) * 1000
    print(f"    Mel: {breakdown.mel_time_ms:.1f}ms")

    # Phase 2: Encoder
    print("  Phase 2: Encoder...")
    encoder_start = time.perf_counter()
    audio_features = model.embed_audio(mel)
    mx.eval(audio_features)  # Ensure computation complete
    encoder_end = time.perf_counter()
    breakdown.encoder_time_ms = (encoder_end - encoder_start) * 1000
    print(f"    Encoder: {breakdown.encoder_time_ms:.1f}ms")

    # Phase 3: Decoder first token
    print("  Phase 3: Decoder (first token)...")
    from tools.whisper_mlx.tokenizer import get_whisper_tokenizer

    is_multilingual = model.config.n_vocab >= 51865
    num_langs = model.config.n_vocab - 51765 - int(is_multilingual)
    tokenizer = get_whisper_tokenizer(
        multilingual=is_multilingual,
        num_languages=num_langs,
        language="en" if is_multilingual else None,
        task="transcribe",
    )

    # Initial tokens for decoder (different for multilingual vs non-multilingual)
    if is_multilingual:
        initial_tokens = mx.array([[
            tokenizer.sot,
            tokenizer.language_token,
            tokenizer.transcribe,
            tokenizer.no_timestamps,
        ]])
    else:
        # Non-multilingual models don't have language/task tokens
        initial_tokens = mx.array([[
            tokenizer.sot,
            tokenizer.no_timestamps,
        ]])

    decoder_start = time.perf_counter()
    # Get first token prediction
    logits, _, _, _ = model.decoder(initial_tokens, audio_features, kv_cache=None)
    mx.eval(logits)  # Ensure computation complete
    decoder_end = time.perf_counter()
    breakdown.decoder_first_token_ms = (decoder_end - decoder_start) * 1000
    print(f"    Decoder: {breakdown.decoder_first_token_ms:.1f}ms")

    # Calculate total
    breakdown.total_ttfo_ms = (
        breakdown.mel_time_ms +
        breakdown.encoder_time_ms +
        breakdown.decoder_first_token_ms
    )

    return breakdown


def compare_models(
    models: list[str],
    audio: Optional[np.ndarray] = None,
) -> dict[str, TTFOBreakdown]:
    """Compare TTFO breakdown across multiple models."""
    results = {}

    for model_name in models:
        print(f"\n{'=' * 60}")
        print(f"Profiling: {model_name}")
        print('=' * 60)

        try:
            breakdown = profile_ttfo_breakdown(
                model_name=model_name,
                audio=audio,
                warmup=True,
            )
            results[model_name] = breakdown
        except Exception as e:
            print(f"Error profiling {model_name}: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Profile TTFO breakdown for Whisper models"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="large-v3",
        help="Model name (default: large-v3)"
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to audio file (uses test audio if not specified)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare multiple models (tiny, base, small, medium, large-v3)"
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip warmup run"
    )

    args = parser.parse_args()

    # Load audio once
    audio = load_test_audio(args.audio)

    if args.compare:
        # Compare multiple models (use multilingual .en variants where available)
        # tiny.en/base.en are not multilingual, so use tiny/base which may work
        models = ["tiny.en", "base.en", "small.en", "medium.en", "large-v3"]
        results = compare_models(models, audio)

        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)
        print(f"{'Model':<12} {'Mel (ms)':<10} {'Encoder (ms)':<14} {'Decoder (ms)':<14} {'Total (ms)':<12} {'P3 Gate':<8}")
        print("-" * 80)

        for model_name, breakdown in results.items():
            total = breakdown.mel_time_ms + breakdown.encoder_time_ms + breakdown.decoder_first_token_ms
            gate = "PASS" if total < 500 else "FAIL"
            print(f"{model_name:<12} {breakdown.mel_time_ms:<10.1f} {breakdown.encoder_time_ms:<14.1f} {breakdown.decoder_first_token_ms:<14.1f} {total:<12.1f} {gate:<8}")

        print("\nConclusion: Models meeting P3 <500ms TTFO gate:")
        passing = [m for m, b in results.items()
                   if b.mel_time_ms + b.encoder_time_ms + b.decoder_first_token_ms < 500]
        if passing:
            print(f"  {', '.join(passing)}")
        else:
            print("  None - need optimization or smaller model for first-pass")
    else:
        # Single model profile
        breakdown = profile_ttfo_breakdown(
            model_name=args.model,
            audio=audio,
            warmup=not args.no_warmup,
        )

        print(breakdown)

        # Recommendations
        print("\nAnalysis:")
        total = breakdown.mel_time_ms + breakdown.encoder_time_ms + breakdown.decoder_first_token_ms
        encoder_pct = 100 * breakdown.encoder_time_ms / total

        if encoder_pct > 70:
            print(f"  - Encoder dominates TTFO ({encoder_pct:.0f}%)")
            print("  - Consider: smaller model for first pass, encoder streaming, or speculative first output")
        elif encoder_pct > 50:
            print(f"  - Encoder is major contributor ({encoder_pct:.0f}%)")
            print("  - Consider: encoder optimization or parallel processing")
        else:
            print(f"  - Encoder is not the bottleneck ({encoder_pct:.0f}%)")
            print("  - Focus on decoder or pipeline optimization")

        if breakdown.encoder_time_ms > 1000:
            print(f"\n  Encoder takes {breakdown.encoder_time_ms:.0f}ms ({breakdown.num_encoder_layers} layers)")
            print(f"  That's ~{breakdown.encoder_time_ms/breakdown.num_encoder_layers:.1f}ms per layer")


if __name__ == "__main__":
    main()
