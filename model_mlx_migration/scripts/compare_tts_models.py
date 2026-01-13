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
TTS Model Comparison Benchmark

Compare Kokoro, CosyVoice2, and F5-TTS on:
- Generation speed (real-time factor)
- Whisper transcription accuracy
- Audio quality metrics (RMS, peak amplitude)

Recommendations:
- Voice cloning: Use CosyVoice2 (35x RTF) - 18x faster than F5-TTS with equal/better quality
- Fast synthesis: Use Kokoro (65x RTF) - fastest for non-cloning use cases
- F5-TTS: DEPRECATED - kept for comparison only

Usage:
    python scripts/compare_tts_models.py
    python scripts/compare_tts_models.py --output-dir /tmp/tts_comparison
"""

import argparse
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class TTSResult:
    """Result from a single TTS generation."""

    model: str
    text: str
    status: str  # PASS, FAIL, SKIP
    error: Optional[str] = None
    generation_time_s: float = 0.0
    audio_duration_s: float = 0.0
    rtf: float = 0.0  # Real-time factor (lower is faster)
    audio_rms: float = 0.0
    audio_peak: float = 0.0
    transcription: str = ""
    wer: float = 1.0  # Word error rate


@dataclass
class ModelCapabilities:
    """Track which models are available."""

    kokoro: bool = False
    cosyvoice2: bool = False
    f5tts: bool = False
    whisper: bool = False


# Test sentences that exercise different phoneme patterns
TEST_SENTENCES = [
    "Hello world.",
    "The quick brown fox jumps over the lazy dog.",
    "How are you today?",
    "Testing one two three four five.",
    "Good morning everyone.",
]


def check_capabilities() -> ModelCapabilities:
    """Check which TTS models are available."""
    caps = ModelCapabilities()

    # Check Kokoro
    try:
        from tools.pytorch_to_mlx.converters import KokoroConverter  # noqa: F401

        caps.kokoro = True
    except ImportError:
        pass

    # Check CosyVoice2
    try:
        model_path = Path.home() / ".cache" / "cosyvoice2" / "cosyvoice2-0.5b"
        if model_path.exists() and (model_path / "llm.pt").exists():
            from tools.pytorch_to_mlx.converters.models import (
                CosyVoice2Model,  # noqa: F401
            )

            caps.cosyvoice2 = True
    except ImportError:
        pass

    # Check F5-TTS
    try:
        from f5_tts_mlx.generate import generate  # noqa: F401

        caps.f5tts = True
    except ImportError:
        pass

    # Check Whisper
    try:
        import mlx_whisper  # noqa: F401

        caps.whisper = True
    except ImportError:
        pass

    return caps


def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio file with Whisper."""
    try:
        import mlx_whisper

        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo="mlx-community/whisper-tiny",
        )
        text: str = result.get("text", "")
        return text.strip()
    except Exception as e:
        return f"Error: {e}"


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute word error rate between reference and hypothesis."""
    ref_words = reference.lower().strip().split()
    hyp_words = hypothesis.lower().strip().split()

    if not ref_words:
        return 1.0 if hyp_words else 0.0

    # Simple Levenshtein distance at word level
    m, n = len(ref_words), len(hyp_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n] / max(m, 1)


def generate_kokoro(text: str, output_path: Path) -> TTSResult:
    """Generate audio with Kokoro MLX."""
    result = TTSResult(model="Kokoro", text=text, status="SKIP")

    try:
        import mlx.core as mx

        from tools.pytorch_to_mlx.converters import KokoroConverter
        from tools.pytorch_to_mlx.converters.models.kokoro_phonemizer import (
            phonemize_text,
        )

        # Load model (cached after first call)
        converter = KokoroConverter()
        model, config, _ = converter.load_from_hf()

        # Convert text to tokens - phonemize_text returns (phoneme_str, token_ids)
        phonemes, token_ids = phonemize_text(text)
        tokens_mx = mx.array([token_ids])

        # Load voice with correct phoneme length (critical for duration prediction)
        # Voice packs are indexed by phoneme length: voice_pack[len(phonemes)-1]
        voice = converter.load_voice("af_heart", phoneme_length=len(phonemes))
        mx.eval(voice)

        # Generate
        start_time = time.time()
        audio = model.synthesize(tokens_mx, voice)
        mx.eval(audio)
        generation_time = time.time() - start_time

        # Convert to numpy
        audio_np = np.array(audio).flatten()

        # Save to WAV (24kHz)
        sf.write(str(output_path), audio_np, 24000)

        # Calculate metrics
        result.generation_time_s = generation_time
        result.audio_duration_s = len(audio_np) / 24000
        result.rtf = generation_time / max(result.audio_duration_s, 0.001)
        result.audio_rms = float(np.sqrt(np.mean(audio_np**2)))
        result.audio_peak = float(np.max(np.abs(audio_np)))
        result.status = "PASS"

    except Exception as e:
        result.status = "FAIL"
        result.error = str(e)

    return result


def generate_cosyvoice2(text: str, output_path: Path) -> TTSResult:
    """Generate audio with CosyVoice2 MLX."""
    result = TTSResult(model="CosyVoice2", text=text, status="SKIP")

    try:
        from tools.pytorch_to_mlx.converters.models import CosyVoice2Model

        model_path = Path.home() / ".cache" / "cosyvoice2" / "cosyvoice2-0.5b"
        model = CosyVoice2Model.from_pretrained(model_path)

        # Estimate max_tokens from text length
        # CosyVoice2 generates ~0.21s per token, aim for ~1.5-2 words/second
        # So ~0.5-0.7 seconds per word -> ~3-4 tokens per word
        word_count = len(text.split())
        max_tokens = max(15, min(100, word_count * 4))  # 15-100 range

        # Generate using synthesize_text (convenience method that handles tokenization)
        start_time = time.time()
        audio = model.synthesize_text(
            text=text,
            max_tokens=max_tokens,
            num_flow_steps=10,
        )
        generation_time = time.time() - start_time
        audio_np = np.array(audio)

        # Ensure 1D
        audio_np = np.array(audio_np).flatten()

        # Normalize to prevent clipping (CosyVoice2 outputs very loud audio)
        peak = np.max(np.abs(audio_np))
        if peak > 0.5:
            audio_np = audio_np * (0.9 / peak)

        # Save to WAV (22050Hz for CosyVoice2)
        sample_rate = 22050
        sf.write(str(output_path), audio_np, sample_rate)

        # Calculate metrics
        result.generation_time_s = generation_time
        result.audio_duration_s = len(audio_np) / sample_rate
        result.rtf = generation_time / max(result.audio_duration_s, 0.001)
        result.audio_rms = float(np.sqrt(np.mean(audio_np**2)))
        result.audio_peak = float(np.max(np.abs(audio_np)))
        result.status = "PASS"

    except Exception as e:
        result.status = "FAIL"
        result.error = str(e)

    return result


def generate_f5tts(text: str, output_path: Path) -> TTSResult:
    """Generate audio with F5-TTS MLX."""
    result = TTSResult(model="F5-TTS", text=text, status="SKIP")

    try:
        from f5_tts_mlx.generate import generate

        # Generate
        start_time = time.time()
        generate(
            generation_text=text,
            output_path=str(output_path),
            steps=6,  # Balanced speed/quality (6 steps ~19% faster than 8)
        )
        generation_time = time.time() - start_time

        # Load generated audio to get metrics
        audio_np, sample_rate = sf.read(str(output_path))
        audio_np = np.array(audio_np).flatten()

        # Calculate metrics
        result.generation_time_s = generation_time
        result.audio_duration_s = len(audio_np) / sample_rate
        result.rtf = generation_time / max(result.audio_duration_s, 0.001)
        result.audio_rms = float(np.sqrt(np.mean(audio_np**2)))
        result.audio_peak = float(np.max(np.abs(audio_np)))
        result.status = "PASS"

    except Exception as e:
        result.status = "FAIL"
        result.error = str(e)

    return result


def run_benchmark(output_dir: Path, use_whisper: bool = True) -> list[TTSResult]:
    """Run the full TTS benchmark."""
    caps = check_capabilities()

    print("=" * 70)
    print("TTS Model Comparison Benchmark")
    print("=" * 70)
    print()
    print("Available models:")
    print(f"  Kokoro:     {'YES' if caps.kokoro else 'NO'}")
    print(f"  CosyVoice2: {'YES' if caps.cosyvoice2 else 'NO'}")
    print(f"  F5-TTS:     {'YES' if caps.f5tts else 'NO'}")
    print(f"  Whisper:    {'YES' if caps.whisper else 'NO'}")
    print()

    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for i, text in enumerate(TEST_SENTENCES, 1):
        print(f"--- Test {i}/{len(TEST_SENTENCES)}: '{text[:40]}...' ---")
        print()

        # Test each model
        for model_name, available, generate_fn in [
            ("Kokoro", caps.kokoro, generate_kokoro),
            ("CosyVoice2", caps.cosyvoice2, generate_cosyvoice2),
            ("F5-TTS", caps.f5tts, generate_f5tts),
        ]:
            if not available:
                result = TTSResult(model=model_name, text=text, status="SKIP")
                result.error = "Model not available"
                results.append(result)
                print(f"  {model_name:12}: SKIP (not available)")
                continue

            # Generate audio
            fname = f"{model_name.lower().replace('-', '')}_test{i}.wav"
            output_path = output_dir / fname
            result = generate_fn(text, output_path)

            # Run Whisper transcription if available and generation succeeded
            if use_whisper and caps.whisper and result.status == "PASS":
                result.transcription = transcribe_audio(str(output_path))
                result.wer = compute_wer(text, result.transcription)

            results.append(result)

            # Print result
            if result.status == "PASS":
                print(
                    f"  {model_name:12}: RTF={result.rtf:.3f} "
                    f"Duration={result.audio_duration_s:.2f}s "
                    f"RMS={result.audio_rms:.4f}"
                )
                if result.transcription:
                    wer_str = f"WER={result.wer:.2%}"
                    trans_preview = result.transcription[:50]
                    if len(result.transcription) > 50:
                        trans_preview += "..."
                    print(f"               Whisper: '{trans_preview}' ({wer_str})")
            else:
                print(f"  {model_name:12}: {result.status} - {result.error}")

        print()

    return results


def print_summary(results: list[TTSResult]):
    """Print summary statistics."""
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()

    # Group by model
    models = ["Kokoro", "CosyVoice2", "F5-TTS"]
    for model in models:
        model_results = [r for r in results if r.model == model and r.status == "PASS"]
        if not model_results:
            print(f"{model:12}: No successful generations")
            continue

        avg_rtf = np.mean([r.rtf for r in model_results])
        avg_wer = np.mean([r.wer for r in model_results if r.transcription])
        avg_rms = np.mean([r.audio_rms for r in model_results])

        print(f"{model:12}:")
        print(f"  Avg RTF:   {avg_rtf:.3f}x (lower is faster)")
        if any(r.transcription for r in model_results):
            print(f"  Avg WER:   {avg_wer:.2%}")
        print(f"  Avg RMS:   {avg_rms:.4f}")
        print(f"  Tests:     {len(model_results)} passed")
        print()

    # Comparison table
    print("-" * 70)
    hdr = f"{'Model':12} | {'Avg RTF':>8} | {'Avg WER':>8}"
    hdr += f" | {'Avg RMS':>8} | {'Passed':>6}"
    print(hdr)
    print("-" * 70)

    for model in models:
        model_results = [r for r in results if r.model == model and r.status == "PASS"]
        if not model_results:
            print(f"{model:12} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8} | {'0':>6}")
            continue

        avg_rtf = np.mean([r.rtf for r in model_results])
        avg_wer = np.mean([r.wer for r in model_results if r.transcription])
        avg_rms = np.mean([r.audio_rms for r in model_results])
        has_trans = any(r.transcription for r in model_results)
        wer_str = f"{avg_wer:.2%}" if has_trans else "N/A"
        n_pass = len(model_results)
        row = f"{model:12} | {avg_rtf:>8.3f} | {wer_str:>8}"
        row += f" | {avg_rms:>8.4f} | {n_pass:>6}"
        print(row)

    print("-" * 70)


def main():
    parser = argparse.ArgumentParser(description="TTS Model Comparison Benchmark")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save generated audio files (default: temp dir)",
    )
    parser.add_argument(
        "--no-whisper",
        action="store_true",
        help="Skip Whisper transcription",
    )
    args = parser.parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(tempfile.mkdtemp(prefix="tts_comparison_"))
        print(f"Output directory: {output_dir}")
        print()

    results = run_benchmark(output_dir, use_whisper=not args.no_whisper)
    print_summary(results)

    # Count successes
    passed = sum(1 for r in results if r.status == "PASS")
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    print(f"Audio files saved to: {output_dir}")

    return 0 if passed > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
