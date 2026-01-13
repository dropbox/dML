#!/usr/bin/env python3
"""Validate AST MLX implementation against PyTorch reference.

This script compares outputs from PyTorch and MLX AST models
to verify numerical equivalence.

Usage:
    python scripts/validate_ast_mlx.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress transformers warnings
import warnings
warnings.filterwarnings("ignore")


def load_pytorch_model():
    """Load HuggingFace PyTorch AST model."""
    from transformers import ASTForAudioClassification

    model_path = "models/sota/ast"
    model = ASTForAudioClassification.from_pretrained(model_path)
    model.eval()
    return model


def load_mlx_model():
    """Load our MLX AST model."""
    from tools.whisper_mlx.sota.ast_mlx import ASTForAudioClassification

    model_path = Path("models/sota/ast-mlx")
    model = ASTForAudioClassification.from_pretrained(str(model_path))
    return model


def validate_outputs():
    """Compare PyTorch and MLX model outputs."""
    import mlx.core as mx

    print("=" * 60)
    print("AST MLX Validation")
    print("=" * 60)

    # Load models
    print("\nLoading PyTorch model...")
    pt_model = load_pytorch_model()

    print("Loading MLX model...")
    mlx_model = load_mlx_model()

    # Create test input
    # AST expects input shape: (batch, num_mel_bins, time_frames)
    # For AudioSet: (batch, 128, 1024)
    np.random.seed(42)
    test_input = np.random.randn(1, 128, 1024).astype(np.float32)

    print(f"\nTest input shape: {test_input.shape}")

    # PyTorch inference
    print("\nRunning PyTorch inference...")
    with torch.no_grad():
        pt_input = torch.from_numpy(test_input)
        pt_output = pt_model(pt_input)
        pt_logits = pt_output.logits.numpy()

    print(f"PyTorch output shape: {pt_logits.shape}")
    print(f"PyTorch logits range: [{pt_logits.min():.4f}, {pt_logits.max():.4f}]")

    # MLX inference
    print("\nRunning MLX inference...")
    mlx_input = mx.array(test_input)
    mlx_logits = mlx_model(mlx_input, use_compile=False)
    mx.eval(mlx_logits)
    mlx_logits_np = np.array(mlx_logits)

    print(f"MLX output shape: {mlx_logits_np.shape}")
    print(f"MLX logits range: [{mlx_logits_np.min():.4f}, {mlx_logits_np.max():.4f}]")

    # Compare outputs
    print("\n" + "=" * 60)
    print("Validation Results")
    print("=" * 60)

    diff = np.abs(pt_logits - mlx_logits_np)
    max_diff = diff.max()
    mean_diff = diff.mean()
    median_diff = np.median(diff)

    print(f"\nMax absolute difference:    {max_diff:.2e}")
    print(f"Mean absolute difference:   {mean_diff:.2e}")
    print(f"Median absolute difference: {median_diff:.2e}")

    # Check predictions match
    pt_pred = np.argmax(pt_logits, axis=-1)
    mlx_pred = np.argmax(mlx_logits_np, axis=-1)
    predictions_match = np.array_equal(pt_pred, mlx_pred)

    print(f"\nPyTorch prediction: {pt_pred[0]}")
    print(f"MLX prediction:     {mlx_pred[0]}")
    print(f"Predictions match:  {predictions_match}")

    # Determine pass/fail
    # Note: For transformer models, 1e-3 is more realistic due to numerical precision
    tolerance = 1e-3
    passed = max_diff < tolerance and predictions_match

    print("\n" + "=" * 60)
    if passed:
        print(f"PASS: Max diff {max_diff:.2e} < tolerance {tolerance}")
    else:
        print(f"FAIL: Max diff {max_diff:.2e} >= tolerance {tolerance}")
    print("=" * 60)

    # Save validation results
    results = {
        "model": "AST (Audio Spectrogram Transformer)",
        "pytorch_path": "models/sota/ast",
        "mlx_path": "models/sota/ast-mlx",
        "input_shape": list(test_input.shape),
        "output_shape": list(pt_logits.shape),
        "max_absolute_diff": float(max_diff),
        "mean_absolute_diff": float(mean_diff),
        "median_absolute_diff": float(median_diff),
        "predictions_match": predictions_match,
        "pytorch_prediction": int(pt_pred[0]),
        "mlx_prediction": int(mlx_pred[0]),
        "tolerance": float(tolerance),
        "passed": bool(passed),
    }

    results_path = Path("models/sota/ast-mlx/validation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved validation results to {results_path}")

    return passed


if __name__ == "__main__":
    success = validate_outputs()
    sys.exit(0 if success else 1)
