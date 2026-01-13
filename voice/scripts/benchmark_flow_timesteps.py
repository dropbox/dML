#!/usr/bin/env python3
"""
Benchmark Flow model with different n_timesteps values.

This script evaluates quality vs speed tradeoff for n_timesteps parameter
in CosyVoice2's CFM (Conditional Flow Matching) decoder.

n_timesteps=10 is the default. Lower values reduce quality but improve speed.
"""

import sys
import time
from pathlib import Path
import torch
import numpy as np

# Add cosyvoice_repo to path
sys.path.insert(0, str(Path(__file__).parent.parent / "cosyvoice_repo"))

# Disable CUDA
torch.cuda.is_available = lambda: False


def load_model():
    """Load CosyVoice2 model."""
    from cosyvoice.cli.cosyvoice import CosyVoice2
    model_dir = Path(__file__).parent.parent / "models/cosyvoice/CosyVoice2-0.5B"
    print(f"Loading model from {model_dir}...")
    return CosyVoice2(str(model_dir), load_jit=False, load_trt=False, fp16=False)


def benchmark_synthesis(model, text: str, instruction: str, n_timesteps: int, device: str):
    """Run synthesis with specified n_timesteps and measure timing."""
    import torch

    # Monkey-patch the flow decoder's forward method to use custom n_timesteps
    flow = model.model.flow
    original_forward = flow.decoder.forward.__func__

    def patched_forward(self, mu, mask, n_timesteps_ignored, *args, **kwargs):
        # Override n_timesteps with our test value
        return original_forward(self, mu, mask, n_timesteps, *args, **kwargs)

    flow.decoder.forward = patched_forward.__get__(flow.decoder, type(flow.decoder))

    try:
        # Run synthesis using inference_sft (no voice prompt needed)
        flow.to(device)
        start = time.perf_counter()

        # Use inference_sft for basic synthesis
        for result in model.inference_sft(
            text,
            spk_id='中文女',  # Default speaker
            stream=False,
            speed=1.0
        ):
            audio = result["tts_speech"]
            break

        elapsed = time.perf_counter() - start

        # Calculate duration
        sample_rate = 22050
        duration = len(audio.numpy().flatten()) / sample_rate
        rtf = elapsed / duration if duration > 0 else float('inf')

        return {
            "n_timesteps": n_timesteps,
            "elapsed_ms": elapsed * 1000,
            "duration_ms": duration * 1000,
            "rtf": rtf,
            "audio": audio.numpy().flatten()
        }
    finally:
        # Restore original forward
        flow.decoder.forward = original_forward.__get__(flow.decoder, type(flow.decoder))


def compute_quality_metrics(audio_ref, audio_test):
    """Compute audio quality metrics between reference and test."""
    # RMS energy
    rms_ref = np.sqrt(np.mean(audio_ref**2))
    rms_test = np.sqrt(np.mean(audio_test**2))

    # Cross-correlation (similarity)
    min_len = min(len(audio_ref), len(audio_test))
    corr = np.corrcoef(audio_ref[:min_len], audio_test[:min_len])[0, 1]

    return {
        "rms_ref": rms_ref,
        "rms_test": rms_test,
        "correlation": corr,
        "length_ratio": len(audio_test) / len(audio_ref)
    }


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    model = load_model()

    text = "你好世界，今天天气真不错。"
    instruction = "用普通话说"

    # Test different n_timesteps values
    timesteps_to_test = [10, 8, 6, 5, 4, 3, 2, 1]

    print("\n" + "=" * 80)
    print("Flow n_timesteps Benchmark")
    print("=" * 80)
    print(f"Text: {text}")
    print(f"Instruction: {instruction}")
    print(f"Device: {device}")
    print()

    results = []
    reference_audio = None

    for n_ts in timesteps_to_test:
        print(f"Testing n_timesteps={n_ts}...")

        try:
            result = benchmark_synthesis(model, text, instruction, n_ts, device)

            if reference_audio is None:
                reference_audio = result["audio"]
                quality = {"correlation": 1.0}
            else:
                quality = compute_quality_metrics(reference_audio, result["audio"])

            result.update(quality)
            results.append(result)

            print(f"  n_timesteps={n_ts:2d}: RTF={result['rtf']:.3f}, "
                  f"Time={result['elapsed_ms']:.0f}ms, "
                  f"Correlation={quality['correlation']:.3f}")
        except Exception as e:
            print(f"  n_timesteps={n_ts:2d}: FAILED - {e}")

    # Summary table
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"{'n_ts':>4} {'RTF':>6} {'Time(ms)':>10} {'Corr':>6} {'Quality':>10}")
    print("-" * 40)

    for r in results:
        quality_rating = "EXCELLENT" if r["correlation"] >= 0.98 else \
                        "GOOD" if r["correlation"] >= 0.95 else \
                        "FAIR" if r["correlation"] >= 0.90 else "POOR"
        print(f"{r['n_timesteps']:4d} {r['rtf']:6.3f} {r['elapsed_ms']:10.0f} "
              f"{r['correlation']:6.3f} {quality_rating:>10}")

    # Recommendation
    print("\n" + "=" * 80)
    print("Recommendation")
    print("=" * 80)

    # Find best quality at RTF < 1.0
    realtime_results = [r for r in results if r["rtf"] < 1.0 and r["correlation"] >= 0.95]
    if realtime_results:
        best = min(realtime_results, key=lambda x: x["n_timesteps"])
        print(f"For real-time with good quality: n_timesteps={best['n_timesteps']} "
              f"(RTF={best['rtf']:.2f}, Corr={best['correlation']:.3f})")
    else:
        print("No configuration achieves both real-time and good quality on this device.")


if __name__ == "__main__":
    main()
