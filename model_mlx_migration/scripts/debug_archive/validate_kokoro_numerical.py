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
Kokoro Numerical Validation - Comprehensive Audio Quality Metrics

This script computes quantitative metrics comparing MLX Kokoro outputs:
1. Audio quality metrics (RMS, dynamic range, spectral content)
2. Mel spectrogram analysis
3. Cross-run consistency (deterministic behavior)
4. Comparison with PyTorch reference outputs if available

Usage:
    python scripts/validate_kokoro_numerical.py
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("ERROR: MLX not available")
    sys.exit(1)

try:
    import librosa

    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("WARNING: librosa not available, mel spectrogram metrics disabled")


MODEL_PATH = Path.home() / "models" / "kokoro"
VOICE_NAME = "af_heart"
SAMPLE_RATE = 24000


def compute_mel_spectrogram(
    audio: np.ndarray, sr: int = SAMPLE_RATE
) -> np.ndarray | None:
    """Compute mel spectrogram of audio."""
    if not HAS_LIBROSA:
        return None

    mel = librosa.feature.melspectrogram(
        y=audio.astype(np.float32), sr=sr, n_fft=1024, hop_length=256, n_mels=80
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return cast(np.ndarray, mel_db)


def compute_audio_metrics(audio: np.ndarray) -> dict:
    """Compute comprehensive audio quality metrics."""
    # Flatten if needed
    if audio.ndim > 1:
        audio = audio.flatten()

    # Basic stats
    rms = float(np.sqrt(np.mean(audio**2)))
    max_amp = float(np.max(np.abs(audio)))
    min_amp = float(np.min(audio))
    max_val = float(np.max(audio))
    dynamic_range = max_val - min_amp

    # Check for issues
    has_nan = bool(np.any(np.isnan(audio)))
    has_inf = bool(np.any(np.isinf(audio)))
    is_saturated = max_amp > 0.99
    is_silent = rms < 0.01

    # Spectral metrics if librosa available
    spectral_metrics = {}
    if HAS_LIBROSA:
        # Spectral centroid (brightness)
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio.astype(np.float32), sr=SAMPLE_RATE
        )
        spectral_metrics["spectral_centroid_mean"] = float(np.mean(spectral_centroid))

        # Zero crossing rate (noisiness indicator)
        zcr = librosa.feature.zero_crossing_rate(audio)
        spectral_metrics["zero_crossing_rate_mean"] = float(np.mean(zcr))

    return {
        "rms": rms,
        "max_amplitude": max_amp,
        "dynamic_range": dynamic_range,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "is_saturated": is_saturated,
        "is_silent": is_silent,
        "sample_count": len(audio),
        "duration_sec": len(audio) / SAMPLE_RATE,
        **spectral_metrics,
    }


def compute_comparison_metrics(audio1: np.ndarray, audio2: np.ndarray) -> dict:
    """Compute metrics comparing two audio signals."""
    # Flatten
    if audio1.ndim > 1:
        audio1 = audio1.flatten()
    if audio2.ndim > 1:
        audio2 = audio2.flatten()

    # Handle length differences
    min_len = min(len(audio1), len(audio2))
    if len(audio1) != len(audio2):
        length_diff = abs(len(audio1) - len(audio2))
        audio1 = audio1[:min_len]
        audio2 = audio2[:min_len]
    else:
        length_diff = 0

    # Waveform correlation
    if np.std(audio1) > 0 and np.std(audio2) > 0:
        correlation = float(np.corrcoef(audio1, audio2)[0, 1])
    else:
        correlation = 0.0

    # RMS difference
    rms1 = np.sqrt(np.mean(audio1**2))
    rms2 = np.sqrt(np.mean(audio2**2))
    rms_diff = float(abs(rms1 - rms2))

    # Max absolute difference
    max_diff = float(np.max(np.abs(audio1 - audio2)))
    mean_diff = float(np.mean(np.abs(audio1 - audio2)))

    # Mel spectrogram MSE
    mel_mse = None
    if HAS_LIBROSA:
        mel1 = compute_mel_spectrogram(audio1)
        mel2 = compute_mel_spectrogram(audio2)
        if mel1 is not None and mel2 is not None:
            # Align frames
            min_frames = min(mel1.shape[1], mel2.shape[1])
            mel1 = mel1[:, :min_frames]
            mel2 = mel2[:, :min_frames]
            mel_mse = float(np.mean((mel1 - mel2) ** 2))

    return {
        "correlation": correlation,
        "rms_diff": rms_diff,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "mel_mse": mel_mse,
        "length_diff_samples": length_diff,
    }


def load_mlx_model():
    """Load MLX Kokoro model."""
    from tools.pytorch_to_mlx.converters import KokoroConverter

    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()

    return converter, model, config


def load_voice(converter):
    """Load voice embedding."""
    voice = converter.load_voice(VOICE_NAME)
    mx.eval(voice)
    return voice


def synthesize_test_cases(converter, model, voice):
    """Generate audio for standard test cases."""

    test_cases = [
        {
            "name": "short_7tok",
            "tokens": [16, 43, 44, 45, 46, 47, 48],  # " abcdef"
        },
        {
            "name": "medium_15tok",
            "tokens": [16, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 16],
        },
        {
            "name": "longer_30tok",
            "tokens": list(range(16, 46)),  # 30 tokens
        },
    ]

    results = []
    for tc in test_cases:
        name = tc["name"]
        tokens = tc["tokens"]

        # Run inference
        tokens_mx = mx.array([tokens])
        audio = model.synthesize(tokens_mx, voice)
        mx.eval(audio)
        audio_np = np.array(audio)

        # Compute metrics
        metrics = compute_audio_metrics(audio_np)

        results.append(
            {
                "name": name,
                "token_count": len(tokens),
                "audio": audio_np,
                "metrics": metrics,
            }
        )

        print(f"\n{name} ({len(tokens)} tokens):")
        print(f"  Duration: {metrics['duration_sec']:.3f}s")
        print(f"  RMS: {metrics['rms']:.4f}")
        print(f"  Max Amp: {metrics['max_amplitude']:.4f}")
        print(f"  Saturated: {metrics['is_saturated']}")
        print(f"  Silent: {metrics['is_silent']}")

    return results


def cross_run_consistency_test(converter, model, voice, num_runs: int = 3):
    """Test deterministic behavior across multiple runs."""
    print("\n=== Cross-Run Consistency Test ===")

    # Fixed tokens
    tokens = mx.array([[16, 43, 44, 45, 46, 47, 48]])

    audios = []
    for i in range(num_runs):
        audio = model.synthesize(tokens, voice)
        mx.eval(audio)
        audios.append(np.array(audio))

    # Compare consecutive runs
    comparisons = []
    for i in range(len(audios) - 1):
        comp = compute_comparison_metrics(audios[i], audios[i + 1])
        comparisons.append(comp)
        print(f"\nRun {i} vs Run {i + 1}:")
        print(f"  Correlation: {comp['correlation']:.6f}")
        print(f"  RMS Diff: {comp['rms_diff']:.6f}")
        print(f"  Max Diff: {comp['max_diff']:.6f}")

    # Note: Due to random phase in SineGen, runs may differ
    # The important thing is the output is structurally similar

    return comparisons


def validate_against_pytorch_reference():
    """Compare with saved PyTorch reference if available."""
    print("\n=== PyTorch Reference Comparison ===")

    # Check for saved reference
    ref_path = (
        Path(__file__).parent.parent
        / "tests"
        / "fixtures"
        / "audio"
        / "kokoro_pytorch_ref.npz"
    )

    if not ref_path.exists():
        print("No PyTorch reference found. Skipping comparison.")
        print(f"Expected: {ref_path}")
        return None

    # Load reference
    ref_data = np.load(ref_path)
    ref_audio = ref_data["audio"]
    ref_tokens = ref_data.get("tokens", None)

    print(f"Reference audio shape: {ref_audio.shape}")

    # Generate MLX output with same tokens
    converter, model, config = load_mlx_model()
    voice = load_voice(converter)

    if ref_tokens is not None:
        tokens_mx = mx.array([ref_tokens.tolist()])
    else:
        tokens_mx = mx.array([[16, 43, 44, 45, 46, 47, 48]])

    mlx_audio = model.synthesize(tokens_mx, voice)
    mx.eval(mlx_audio)
    mlx_audio_np = np.array(mlx_audio)

    # Compare
    comp = compute_comparison_metrics(ref_audio, mlx_audio_np)

    print("\nPyTorch vs MLX Comparison:")
    print(f"  Correlation: {comp['correlation']:.4f}")
    print(f"  RMS Diff: {comp['rms_diff']:.4f}")
    print(f"  Mel MSE: {comp['mel_mse']:.4f}" if comp["mel_mse"] else "  Mel MSE: N/A")

    return comp


def run_full_validation():
    """Run comprehensive validation."""
    print("=" * 60)
    print("Kokoro Numerical Validation")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Voice: {VOICE_NAME}")

    # Load model
    print("\nLoading MLX model...")
    converter, model, config = load_mlx_model()

    print("Loading voice...")
    voice = load_voice(converter)

    # Test cases
    print("\n=== Standard Test Cases ===")
    test_results = synthesize_test_cases(converter, model, voice)

    # Cross-run consistency
    consistency_results = cross_run_consistency_test(converter, model, voice)

    # PyTorch comparison (if available)
    pytorch_comparison = validate_against_pytorch_reference()

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    # Check all test cases pass quality thresholds
    all_pass = True
    for tc in test_results:
        m = tc["metrics"]
        passed = (
            not m["has_nan"]
            and not m["has_inf"]
            and not m["is_saturated"]
            and not m["is_silent"]
            and 0.05 < m["rms"] < 0.5
        )
        status = "PASS" if passed else "FAIL"
        print(
            f"  [{status}] {tc['name']}: RMS={m['rms']:.4f}, MaxAmp={m['max_amplitude']:.4f}"
        )
        all_pass = all_pass and passed

    # Consistency check
    if consistency_results:
        avg_corr = np.mean([c["correlation"] for c in consistency_results])
        print(f"\n  Consistency (avg correlation): {avg_corr:.4f}")

    # PyTorch comparison
    if pytorch_comparison:
        corr = pytorch_comparison["correlation"]
        threshold = 0.95
        pt_pass = corr > threshold
        print(f"\n  PyTorch Correlation: {corr:.4f} (threshold: {threshold})")
        print(f"  PyTorch Comparison: {'PASS' if pt_pass else 'FAIL'}")
        all_pass = all_pass and pt_pass

    print(f"\nOverall: {'PASS' if all_pass else 'FAIL'}")

    return all_pass, {
        "test_results": test_results,
        "consistency_results": consistency_results,
        "pytorch_comparison": pytorch_comparison,
    }


if __name__ == "__main__":
    success, results = run_full_validation()
    sys.exit(0 if success else 1)
