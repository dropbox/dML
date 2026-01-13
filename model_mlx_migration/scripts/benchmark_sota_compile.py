#!/usr/bin/env python3
"""
SOTA Audio Models: mx.compile() Benchmark

Benchmarks all 6 SOTA audio models comparing:
- Eager (uncompiled) execution
- mx.compile() execution

Measures speedup from mx.compile() for each model size.

Usage:
    python scripts/benchmark_sota_compile.py
    python scripts/benchmark_sota_compile.py --models ecapa-tdnn,ast
    python scripts/benchmark_sota_compile.py --runs 20
"""

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx


@dataclass
class BenchResult:
    model_name: str
    params_millions: float
    eager_ms: float
    eager_std: float
    compiled_ms: float
    compiled_std: float
    speedup: float


MODEL_INFO = {
    "ecapa-tdnn": {
        "display": "ECAPA-TDNN",
        "path": "models/sota/ecapa-tdnn-mlx",
        "params_m": 14.0,
        "input_shape": (1, 60, 300),  # (batch, n_mels, time)
        "type": "mel",
    },
    "ast": {
        "display": "AST",
        "path": "models/sota/ast-mlx",
        "params_m": 87.0,
        "input_shape": (1, 128, 1024),  # (batch, n_mels, time)
        "type": "mel",
    },
    "emotion2vec": {
        "display": "Emotion2vec",
        "path": "models/sota/emotion2vec-mlx",
        "params_m": 94.0,
        "input_shape": (1, 16000),  # (batch, samples) - 1s audio
        "type": "audio",
    },
    "wav2vec2-xlsr": {
        "display": "Wav2Vec2-XLSR",
        "path": "models/sota/wav2vec2-xlsr-mlx",
        "params_m": 315.0,
        "input_shape": (1, 16000),  # (batch, samples) - 1s audio
        "type": "audio",
    },
    "beats": {
        "display": "BEATs",
        "path": "models/sota/beats-mlx",
        "params_m": 90.0,
        "input_shape": (1, 98, 128),  # (batch, time, freq)
        "type": "fbank",
    },
    "wavlm-large": {
        "display": "WavLM-large",
        "path": "models/sota/wavlm-large-mlx",
        "params_m": 316.0,
        "input_shape": (1, 16000),  # (batch, samples) - 1s audio
        "type": "audio",
    },
}


def load_model(model_name: str):
    """Load an MLX SOTA model."""
    info = MODEL_INFO[model_name]
    path = info["path"]

    if model_name == "ecapa-tdnn":
        from tools.whisper_mlx.sota.ecapa_tdnn import ECAPATDNNForLanguageID
        model = ECAPATDNNForLanguageID.from_pretrained(path)
    elif model_name == "ast":
        from tools.whisper_mlx.sota.ast_mlx import ASTForAudioClassification
        model = ASTForAudioClassification.from_pretrained(path)
    elif model_name == "emotion2vec":
        from tools.whisper_mlx.sota.emotion2vec_mlx import Emotion2vecModel
        model = Emotion2vecModel.from_pretrained(path)
    elif model_name == "wav2vec2-xlsr":
        from tools.whisper_mlx.sota.wav2vec2_xlsr_mlx import Wav2Vec2Model
        model = Wav2Vec2Model.from_pretrained(path)
    elif model_name == "beats":
        from tools.whisper_mlx.sota.beats_mlx import BEATsModel
        model = BEATsModel.from_pretrained(path)
    elif model_name == "wavlm-large":
        from tools.whisper_mlx.sota.wavlm_mlx import WavLMModel
        model = WavLMModel.from_pretrained(path)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    mx.eval(model.parameters())
    return model


def create_input(model_name: str) -> mx.array:
    """Create test input for a model."""
    info = MODEL_INFO[model_name]
    shape = info["input_shape"]
    return mx.random.normal(shape)


def benchmark_model(
    model_name: str,
    num_runs: int = 10,
    warmup_runs: int = 3,
) -> Optional[BenchResult]:
    """Benchmark a model in eager and compiled modes."""
    info = MODEL_INFO[model_name]

    # Check model exists
    if not Path(info["path"]).exists():
        print(f"  SKIP: {info['display']} - model not found")
        return None

    print(f"\n{'='*60}")
    print(f"Benchmarking: {info['display']} ({info['params_m']:.0f}M params)")
    print(f"{'='*60}")

    # Load model
    print("Loading model...")
    try:
        model = load_model(model_name)
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

    # Create input
    test_input = create_input(model_name)
    mx.eval(test_input)
    print(f"Input shape: {test_input.shape}")

    # Benchmark eager mode
    print(f"\nEager mode ({num_runs} runs)...")
    model.use_compile = False

    # Warmup
    for _ in range(warmup_runs):
        out = model(test_input)
        mx.eval(out)

    # Timed runs
    eager_times = []
    for i in range(num_runs):
        start = time.perf_counter()
        out = model(test_input)
        mx.eval(out)
        elapsed = (time.perf_counter() - start) * 1000
        eager_times.append(elapsed)

    eager_mean = np.mean(eager_times)
    eager_std = np.std(eager_times)
    print(f"  Eager: {eager_mean:.2f} +/- {eager_std:.2f} ms")

    # Benchmark compiled mode
    print(f"\nCompiled mode ({num_runs} runs)...")
    model.use_compile = True
    model._compiled_forward = None  # Reset compiled function

    # Warmup (includes JIT compilation)
    for _ in range(warmup_runs):
        out = model(test_input)
        mx.eval(out)

    # Timed runs
    compiled_times = []
    for i in range(num_runs):
        start = time.perf_counter()
        out = model(test_input)
        mx.eval(out)
        elapsed = (time.perf_counter() - start) * 1000
        compiled_times.append(elapsed)

    compiled_mean = np.mean(compiled_times)
    compiled_std = np.std(compiled_times)
    speedup = eager_mean / compiled_mean
    print(f"  Compiled: {compiled_mean:.2f} +/- {compiled_std:.2f} ms")
    print(f"  Speedup: {speedup:.2f}x")

    # Cleanup
    del model, test_input, out
    gc.collect()

    return BenchResult(
        model_name=info["display"],
        params_millions=info["params_m"],
        eager_ms=eager_mean,
        eager_std=eager_std,
        compiled_ms=compiled_mean,
        compiled_std=compiled_std,
        speedup=speedup,
    )


def print_summary(results: list[BenchResult]):
    """Print summary table."""
    print("\n" + "="*80)
    print("mx.compile() Benchmark Summary - SOTA Audio Models")
    print("="*80)

    # Sort by params
    results = sorted(results, key=lambda r: r.params_millions)

    # Header
    print(f"\n{'Model':<20} {'Params':<10} {'Eager (ms)':<14} {'Compiled (ms)':<14} {'Speedup':<10}")
    print("-"*80)

    for r in results:
        eager_str = f"{r.eager_ms:.2f} +/- {r.eager_std:.2f}"
        compiled_str = f"{r.compiled_ms:.2f} +/- {r.compiled_std:.2f}"
        print(f"{r.model_name:<20} {r.params_millions:<10.0f}M {eager_str:<14} {compiled_str:<14} {r.speedup:.2f}x")

    print("-"*80)

    # Stats
    avg_speedup = np.mean([r.speedup for r in results])
    print(f"\nAverage speedup: {avg_speedup:.2f}x")

    # Analysis
    print("\nKey Findings:")
    small_models = [r for r in results if r.params_millions < 50]
    large_models = [r for r in results if r.params_millions >= 50]

    if small_models:
        small_avg = np.mean([r.speedup for r in small_models])
        print(f"  - Small models (<50M): {small_avg:.2f}x average speedup")
    if large_models:
        large_avg = np.mean([r.speedup for r in large_models])
        print(f"  - Large models (>=50M): {large_avg:.2f}x average speedup")


def main():
    parser = argparse.ArgumentParser(description="Benchmark mx.compile() on SOTA models")
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of models (default: all)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of benchmark runs (default: 10)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup runs (default: 3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file",
    )
    args = parser.parse_args()

    # Select models
    if args.models:
        model_names = [m.strip() for m in args.models.split(",")]
        for name in model_names:
            if name not in MODEL_INFO:
                print(f"ERROR: Unknown model '{name}'")
                print(f"Available: {', '.join(MODEL_INFO.keys())}")
                return 1
    else:
        model_names = list(MODEL_INFO.keys())

    print("="*80)
    print("SOTA Audio Models: mx.compile() Benchmark")
    print("="*80)
    print(f"Models: {', '.join(model_names)}")
    print(f"Runs: {args.runs} (warmup: {args.warmup})")

    # Run benchmarks
    results = []
    for model_name in model_names:
        result = benchmark_model(model_name, args.runs, args.warmup)
        if result:
            results.append(result)

    if not results:
        print("\nNo models benchmarked successfully!")
        return 1

    # Print summary
    print_summary(results)

    # Save results
    if args.output:
        data = [
            {
                "model": r.model_name,
                "params_millions": r.params_millions,
                "eager_ms": r.eager_ms,
                "eager_std": r.eager_std,
                "compiled_ms": r.compiled_ms,
                "compiled_std": r.compiled_std,
                "speedup": r.speedup,
            }
            for r in results
        ]
        with open(args.output, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
