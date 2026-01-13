#!/usr/bin/env python3
"""
SOTA Audio Understanding Models MLX Benchmark

Benchmarks all 6 SOTA audio understanding models converted to MLX:
1. AST (Audio Spectrogram Transformer) - Audio classification
2. emotion2vec - Emotion recognition
3. wav2vec2-xlsr - Phoneme/speech features
4. BEATs - Audio pre-training features
5. wav2vec2-xlsr-ser - Speech emotion recognition
6. WavLM-large - General speech features

Usage:
    python scripts/benchmark_sota_mlx.py
    python scripts/benchmark_sota_mlx.py --models ast,beats
    python scripts/benchmark_sota_mlx.py --runs 10 --warmup 3
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import numpy as np


# Model configurations
MODEL_CONFIGS = {
    "ast": {
        "name": "AST (Audio Spectrogram Transformer)",
        "path": "models/sota/ast-mlx",
        "input_type": "mel",  # mel spectrogram
        "output_type": "logits",  # classification logits
        "num_classes": 527,
    },
    "emotion2vec": {
        "name": "Emotion2vec",
        "path": "models/sota/emotion2vec-mlx",
        "input_type": "audio",  # raw audio
        "output_type": "features",  # hidden states
        "hidden_size": 768,
    },
    "wav2vec2-xlsr": {
        "name": "Wav2Vec2-XLSR (Phoneme)",
        "path": "models/sota/wav2vec2-xlsr-mlx",
        "input_type": "audio",
        "output_type": "features",
        "hidden_size": 1024,
    },
    "beats": {
        "name": "BEATs (Audio Pre-Training)",
        "path": "models/sota/beats-mlx",
        "input_type": "fbank",  # filterbank features
        "output_type": "features",
        "hidden_size": 768,
    },
    "wav2vec2-xlsr-ser": {
        "name": "Wav2Vec2-XLSR-SER (Emotion)",
        "path": "models/sota/wav2vec2-xlsr-ser-mlx",
        "input_type": "audio",
        "output_type": "logits",
        "num_classes": 8,
    },
    "wavlm-large": {
        "name": "WavLM-large (General)",
        "path": "models/sota/wavlm-large-mlx",
        "input_type": "audio",
        "output_type": "features",
        "hidden_size": 1024,
    },
}

# Emotion labels for wav2vec2-xlsr-ser
EMOTION_LABELS = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]


def load_ast_model(model_path: str):
    """Load AST MLX model using from_pretrained."""
    from tools.whisper_mlx.sota.ast_mlx import ASTForAudioClassification

    model = ASTForAudioClassification.from_pretrained(model_path)
    mx.eval(model.parameters())
    return model, None


def load_emotion2vec_model(model_path: str):
    """Load emotion2vec MLX model using from_pretrained."""
    from tools.whisper_mlx.sota.emotion2vec_mlx import Emotion2vecModel

    model = Emotion2vecModel.from_pretrained(model_path)
    mx.eval(model.parameters())
    return model, None


def load_wav2vec2_xlsr_model(model_path: str):
    """Load wav2vec2-xlsr MLX model using from_pretrained."""
    from tools.whisper_mlx.sota.wav2vec2_xlsr_mlx import Wav2Vec2Model

    model = Wav2Vec2Model.from_pretrained(model_path)
    mx.eval(model.parameters())
    return model, None


def load_beats_model(model_path: str):
    """Load BEATs MLX model using from_pretrained."""
    from tools.whisper_mlx.sota.beats_mlx import BEATsModel

    model = BEATsModel.from_pretrained(model_path)
    mx.eval(model.parameters())
    return model, None


def load_wav2vec2_xlsr_ser_model(model_path: str):
    """Load wav2vec2-xlsr-ser MLX model using from_pretrained."""
    from tools.whisper_mlx.sota.wav2vec2_xlsr_mlx import Wav2Vec2ForSequenceClassification

    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
    mx.eval(model.parameters())
    return model, None


def load_wavlm_model(model_path: str):
    """Load WavLM MLX model using from_pretrained."""
    from tools.whisper_mlx.sota.wavlm_mlx import WavLMModel

    model = WavLMModel.from_pretrained(model_path)
    mx.eval(model.parameters())
    return model, None


# Model loader mapping
MODEL_LOADERS = {
    "ast": load_ast_model,
    "emotion2vec": load_emotion2vec_model,
    "wav2vec2-xlsr": load_wav2vec2_xlsr_model,
    "beats": load_beats_model,
    "wav2vec2-xlsr-ser": load_wav2vec2_xlsr_ser_model,
    "wavlm-large": load_wavlm_model,
}


def create_test_input(model_name: str, config: Any, duration_sec: float = 1.0) -> mx.array:
    """Create test input for a model."""
    sample_rate = 16000
    num_samples = int(duration_sec * sample_rate)

    input_type = MODEL_CONFIGS[model_name]["input_type"]

    if input_type == "audio":
        # Raw audio waveform (batch, samples)
        return mx.random.normal((1, num_samples))

    elif input_type == "mel":
        # Mel spectrogram for AST (batch, n_mels, time)
        n_mels = getattr(config, 'num_mel_bins', 128)
        time_frames = 1024  # Fixed for AST
        return mx.random.normal((1, n_mels, time_frames))

    elif input_type == "fbank":
        # Filterbank features for BEATs (batch, time, freq)
        time_frames = 98  # ~1s at 16kHz with 10ms hop
        n_freq = 128
        return mx.random.normal((1, time_frames, n_freq))

    else:
        raise ValueError(f"Unknown input type: {input_type}")


def benchmark_model(
    model_name: str,
    model_path: str,
    num_runs: int = 5,
    warmup_runs: int = 2,
) -> dict:
    """Benchmark a single model."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {MODEL_CONFIGS[model_name]['name']}")
    print(f"{'='*60}")

    # Load model
    print("Loading model...")
    load_start = time.perf_counter()
    try:
        model, config = MODEL_LOADERS[model_name](model_path)
    except Exception as e:
        print(f"  ERROR loading model: {e}")
        return {
            "model": model_name,
            "name": MODEL_CONFIGS[model_name]["name"],
            "status": "FAILED",
            "error": str(e),
        }
    load_time = time.perf_counter() - load_start
    print(f"  Model loaded in {load_time*1000:.1f}ms")

    # Create test input
    test_input = create_test_input(model_name, config)
    mx.eval(test_input)
    print(f"  Input shape: {test_input.shape}")

    # Warmup runs
    print(f"Warmup ({warmup_runs} runs)...")
    for _ in range(warmup_runs):
        output = model(test_input)
        mx.eval(output)

    # Timed runs
    print(f"Benchmarking ({num_runs} runs)...")
    times = []
    for i in range(num_runs):
        start = time.perf_counter()
        output = model(test_input)
        mx.eval(output)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms
        print(f"  Run {i+1}: {elapsed*1000:.2f}ms")

    # Calculate stats
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)

    # Get output shape
    if isinstance(output, tuple):
        output_shape = output[0].shape
    else:
        output_shape = output.shape

    print("\nResults:")
    print(f"  Output shape: {output_shape}")
    print(f"  Mean time: {mean_time:.2f}ms (+/- {std_time:.2f}ms)")
    print(f"  Min/Max: {min_time:.2f}ms / {max_time:.2f}ms")

    # Clean up
    del model, config, test_input, output
    gc.collect()

    return {
        "model": model_name,
        "name": MODEL_CONFIGS[model_name]["name"],
        "status": "PASS",
        "load_time_ms": load_time * 1000,
        "mean_time_ms": mean_time,
        "std_time_ms": std_time,
        "min_time_ms": min_time,
        "max_time_ms": max_time,
        "output_shape": list(output_shape),
    }


def print_summary(results: list):
    """Print summary table."""
    print("\n" + "="*80)
    print("SOTA MLX Models Benchmark Summary")
    print("="*80)

    # Header
    print(f"{'Model':<30} {'Status':<8} {'Load (ms)':<12} {'Inference (ms)':<15}")
    print("-"*80)

    # Results
    for r in results:
        status = r.get("status", "N/A")
        load_time = r.get("load_time_ms", 0)
        mean_time = r.get("mean_time_ms", 0)
        std_time = r.get("std_time_ms", 0)

        if status == "PASS":
            inference_str = f"{mean_time:.2f} +/- {std_time:.2f}"
            load_str = f"{load_time:.1f}"
        else:
            inference_str = r.get("error", "FAILED")[:15]
            load_str = "-"

        print(f"{r['name']:<30} {status:<8} {load_str:<12} {inference_str:<15}")

    print("-"*80)

    # Count passes
    passes = sum(1 for r in results if r.get("status") == "PASS")
    print(f"\nTotal: {passes}/{len(results)} models passed")

    # Summary stats
    if passes > 0:
        avg_inference = np.mean([r["mean_time_ms"] for r in results if r.get("status") == "PASS"])
        print(f"Average inference time: {avg_inference:.2f}ms")


def main():
    parser = argparse.ArgumentParser(description="Benchmark SOTA MLX models")
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of models to benchmark (default: all)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of benchmark runs (default: 5)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of warmup runs (default: 2)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )
    args = parser.parse_args()

    # Select models
    if args.models:
        model_names = [m.strip() for m in args.models.split(",")]
        # Validate
        for name in model_names:
            if name not in MODEL_CONFIGS:
                print(f"ERROR: Unknown model '{name}'")
                print(f"Available: {', '.join(MODEL_CONFIGS.keys())}")
                return 1
    else:
        model_names = list(MODEL_CONFIGS.keys())

    print("="*80)
    print("SOTA MLX Models Benchmark")
    print("="*80)
    print(f"Models: {', '.join(model_names)}")
    print(f"Runs: {args.runs} (warmup: {args.warmup})")

    # Run benchmarks
    results = []
    for model_name in model_names:
        model_path = MODEL_CONFIGS[model_name]["path"]

        # Check if model exists
        if not Path(model_path).exists():
            print(f"\nSkipping {model_name}: path not found ({model_path})")
            results.append({
                "model": model_name,
                "name": MODEL_CONFIGS[model_name]["name"],
                "status": "SKIPPED",
                "error": "Model path not found",
            })
            continue

        result = benchmark_model(
            model_name,
            model_path,
            num_runs=args.runs,
            warmup_runs=args.warmup,
        )
        results.append(result)

    # Print summary
    print_summary(results)

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
