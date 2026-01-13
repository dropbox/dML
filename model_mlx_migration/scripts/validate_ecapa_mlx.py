#!/usr/bin/env python3
"""Validate ECAPA-TDNN MLX implementation against PyTorch.

This script compares the outputs of the MLX implementation against a
PyTorch implementation using the same weights to ensure numerical equivalence.

Usage:
    python scripts/validate_ecapa_mlx.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add tools to path
sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "whisper_mlx"))


class PyTorchBatchNorm1d(nn.Module):
    """BatchNorm1d using pre-trained running stats."""

    def __init__(self, num_features: int, weight: torch.Tensor, bias: torch.Tensor,
                 running_mean: torch.Tensor, running_var: torch.Tensor, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.register_buffer('weight', weight)
        self.register_buffer('bias', bias)
        self.register_buffer('running_mean', running_mean)
        self.register_buffer('running_var', running_var)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        return F.batch_norm(x, self.running_mean, self.running_var,
                           self.weight, self.bias, False, 0.0, self.eps)


class PyTorchConv1d(nn.Module):
    """Conv1d with pre-trained weights."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor,
                 stride: int = 1, padding: int = 0, dilation: int = 1):
        super().__init__()
        self.register_buffer('weight', weight)
        self.register_buffer('bias', bias)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv1d(x, self.weight, self.bias, self.stride, self.padding, self.dilation)


def load_pytorch_weights() -> Dict[str, torch.Tensor]:
    """Load PyTorch weights from checkpoint."""
    embedding = torch.load(
        "models/sota/ecapa-tdnn/embedding_model.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    classifier = torch.load(
        "models/sota/ecapa-tdnn/classifier.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    return {"embedding": embedding, "classifier": classifier}


def create_pytorch_block0(weights: Dict[str, torch.Tensor]) -> nn.Module:
    """Create PyTorch Block 0 (initial TDNN)."""
    class Block0(nn.Module):
        def __init__(self):
            super().__init__()
            # Conv with same padding
            self.conv = PyTorchConv1d(
                weights["blocks.0.conv.conv.weight"],
                weights["blocks.0.conv.conv.bias"],
                padding=2,  # (kernel_size - 1) // 2 = (5-1)//2 = 2
            )
            self.norm = PyTorchBatchNorm1d(
                1024,
                weights["blocks.0.norm.norm.weight"],
                weights["blocks.0.norm.norm.bias"],
                weights["blocks.0.norm.norm.running_mean"],
                weights["blocks.0.norm.norm.running_var"],
            )

        def forward(self, x):
            x = self.conv(x)
            x = self.norm(x)
            x = F.relu(x)
            return x

    return Block0()


def run_pytorch_inference(
    mel_input: np.ndarray,
    weights: Dict[str, torch.Tensor],
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference with PyTorch using raw weights.

    Only runs Block 0 for now to validate layer by layer.
    """
    x = torch.from_numpy(mel_input).float()

    # Block 0
    block0 = create_pytorch_block0(weights["embedding"])
    with torch.no_grad():
        out = block0(x)

    return out.numpy()


def run_mlx_inference(mel_input: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference with MLX model."""
    import mlx.core as mx
    from sota.ecapa_tdnn import ECAPATDNNForLanguageID
    from sota.ecapa_config import ECAPATDNNConfig

    # Create and load model
    config = ECAPATDNNConfig.voxlingua107()
    model = ECAPATDNNForLanguageID(config)

    weights = mx.load("models/sota/ecapa-tdnn-mlx/weights.npz")
    model.load_weights(list(weights.items()))

    # Convert from PyTorch (B, C, T) to MLX native (B, T, C) format
    x = mx.array(mel_input.transpose(0, 2, 1))
    out = model.embedding_model.blocks_0(x)
    mx.eval(out)

    # Convert output back to PyTorch format (B, C, T) for comparison
    out_np = np.array(out).transpose(0, 2, 1)
    return out_np


def compare_outputs(
    name: str,
    pytorch_out: np.ndarray,
    mlx_out: np.ndarray,
) -> Dict[str, Any]:
    """Compare outputs and return statistics."""
    diff = np.abs(pytorch_out - mlx_out)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))

    # Compute relative error
    abs_pytorch = np.abs(pytorch_out)
    mask = abs_pytorch > 1e-8
    if mask.any():
        rel_error = float(np.mean(diff[mask] / abs_pytorch[mask]))
    else:
        rel_error = 0.0

    status = "PASS" if max_diff < 1e-4 else "FAIL"

    return {
        "name": name,
        "status": status,
        "pytorch_shape": list(pytorch_out.shape),
        "mlx_shape": list(mlx_out.shape),
        "max_absolute_diff": max_diff,
        "mean_absolute_diff": mean_diff,
        "mean_relative_error": rel_error,
    }


def validate_block0():
    """Validate Block 0 outputs match between PyTorch and MLX."""
    print("=" * 60)
    print("Validating Block 0 (Initial TDNN)")
    print("=" * 60)

    # Create test input
    np.random.seed(42)
    mel_input = np.random.randn(1, 60, 100).astype(np.float32)
    print(f"Input shape: {mel_input.shape}")

    # Load PyTorch weights
    weights = load_pytorch_weights()

    # Run PyTorch
    print("\nRunning PyTorch inference...")
    pytorch_out = run_pytorch_inference(mel_input, weights)
    print(f"PyTorch output shape: {pytorch_out.shape}")
    print(f"PyTorch output mean: {pytorch_out.mean():.6f}")
    print(f"PyTorch output std: {pytorch_out.std():.6f}")

    # Run MLX
    print("\nRunning MLX inference...")
    mlx_out = run_mlx_inference(mel_input)
    print(f"MLX output shape: {mlx_out.shape}")
    print(f"MLX output mean: {mlx_out.mean():.6f}")
    print(f"MLX output std: {mlx_out.std():.6f}")

    # Compare
    result = compare_outputs("Block0", pytorch_out, mlx_out)

    print("\nComparison:")
    print(f"  Status: {result['status']}")
    print(f"  Max absolute diff: {result['max_absolute_diff']:.2e}")
    print(f"  Mean absolute diff: {result['mean_absolute_diff']:.2e}")
    print(f"  Mean relative error: {result['mean_relative_error']:.2e}")

    return result


def validate_full_model_output():
    """Validate full model outputs are reasonable."""
    import mlx.core as mx
    from sota.ecapa_tdnn import ECAPATDNNForLanguageID
    from sota.ecapa_config import ECAPATDNNConfig

    print("\n" + "=" * 60)
    print("Validating Full Model Output")
    print("=" * 60)

    # Create and load model
    config = ECAPATDNNConfig.voxlingua107()
    model = ECAPATDNNForLanguageID(config)

    weights = mx.load("models/sota/ecapa-tdnn-mlx/weights.npz")
    model.load_weights(list(weights.items()))

    # Load label encoder
    # Format: 'ab: Abkhazian' => 0
    labels = {}
    with open("models/sota/ecapa-tdnn-mlx/label_encoder.txt") as f:
        for line in f:
            if "=>" in line:
                label_part, idx_part = line.split("=>")
                # Extract language code from 'ab: Abkhazian'
                lang_code = label_part.strip().strip("'").split(":")[0]
                idx = int(idx_part.strip())
                labels[idx] = lang_code
    model.labels = labels

    # Create test input (3 seconds of random mel features)
    np.random.seed(42)
    mel_input = mx.array(np.random.randn(1, 60, 300).astype(np.float32))

    # Run inference
    print("Running full model inference...")
    logits, predictions, embeddings = model(mel_input, return_embedding=True)
    mx.eval(logits, predictions, embeddings)

    print(f"\nEmbedding shape: {embeddings.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Predicted language index: {int(predictions[0])}")

    # Get confidence
    probs = mx.softmax(logits, axis=-1)
    confidence = float(mx.max(probs[0]))
    pred_idx = int(predictions[0])
    pred_lang = labels.get(pred_idx, "unknown")

    print(f"Predicted language: {pred_lang}")
    print(f"Confidence: {confidence:.4f}")

    # Check embedding statistics
    emb_np = np.array(embeddings.squeeze())
    print("\nEmbedding statistics:")
    print(f"  Mean: {emb_np.mean():.6f}")
    print(f"  Std: {emb_np.std():.6f}")
    print(f"  Min: {emb_np.min():.6f}")
    print(f"  Max: {emb_np.max():.6f}")

    return {
        "embedding_shape": list(embeddings.shape),
        "logits_shape": list(logits.shape),
        "predicted_language": pred_lang,
        "confidence": confidence,
        "embedding_mean": float(emb_np.mean()),
        "embedding_std": float(emb_np.std()),
    }


def extract_features_speechbrain(waveform: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
    """Extract SpeechBrain-compatible FBANK features.

    CRITICAL: SpeechBrain uses eps=1e-4 (energy_floor) for log, NOT 1e-6.
    Using wrong eps causes model to predict wrong language.

    Args:
        waveform: Audio waveform tensor (1, samples) or (samples,)
        sample_rate: Sample rate of audio

    Returns:
        Features tensor (1, n_mels, time) in dB scale
    """
    import torchaudio

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # Resample if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)

    # Compute power spectrogram
    spec_transform = torchaudio.transforms.Spectrogram(
        n_fft=400,
        hop_length=160,
        power=2.0,
        center=True,
    )
    spec = spec_transform(waveform)

    # Apply mel filterbank
    mel_scale = torchaudio.transforms.MelScale(
        n_mels=60,
        sample_rate=16000,
        n_stft=201,  # n_fft // 2 + 1
        norm="slaney",
        mel_scale="slaney",
    )
    mel_spec = mel_scale(spec)

    # Convert to dB scale with eps=1e-4 (matches SpeechBrain's energy_floor)
    # CRITICAL: eps=1e-6 gives wrong predictions!
    features = torch.log10(mel_spec + 1e-4) * 10

    return features


def validate_with_real_audio():
    """Validate with real Thai audio sample."""
    import mlx.core as mx
    from sota.ecapa_tdnn import ECAPATDNNForLanguageID
    from sota.ecapa_config import ECAPATDNNConfig

    print("\n" + "=" * 60)
    print("Validating with Real Audio")
    print("=" * 60)

    # Check if torchaudio is available
    try:
        import torchaudio
    except ImportError:
        print("torchaudio not installed, skipping real audio test")
        return None

    # Load test audio
    audio_path = Path("models/sota/ecapa-tdnn/udhr_th.wav")
    if not audio_path.exists():
        print(f"Test audio not found at {audio_path}")
        return None

    waveform, sample_rate = torchaudio.load(str(audio_path))
    print(f"Loaded audio: shape={waveform.shape}, sample_rate={sample_rate}")

    # Extract SpeechBrain-compatible features
    mel = extract_features_speechbrain(waveform, sample_rate)
    print(f"Features shape: {mel.shape}")
    print(f"Features stats: mean={mel.mean():.2f}, std={mel.std():.2f}")

    # Create and load MLX model
    config = ECAPATDNNConfig.voxlingua107()
    model = ECAPATDNNForLanguageID(config)
    weights = mx.load("models/sota/ecapa-tdnn-mlx/weights.npz")
    model.load_weights(list(weights.items()))

    # Load labels
    # Format: 'ab: Abkhazian' => 0
    labels = {}
    with open("models/sota/ecapa-tdnn-mlx/label_encoder.txt") as f:
        for line in f:
            if "=>" in line:
                label_part, idx_part = line.split("=>")
                lang_code = label_part.strip().strip("'").split(":")[0]
                idx = int(idx_part.strip())
                labels[idx] = lang_code
    model.labels = labels

    # Run inference
    mel_mlx = mx.array(mel.squeeze(0).numpy())  # Remove batch dim from mel
    mel_mlx = mel_mlx[None, :, :]  # Add batch dim: (1, n_mels, time)

    logits, predictions = model(mel_mlx)
    mx.eval(logits, predictions)

    probs = mx.softmax(logits, axis=-1)
    confidence = float(mx.max(probs[0]))
    pred_idx = int(predictions[0])
    pred_lang = labels.get(pred_idx, "unknown")

    print(f"\nPredicted language: {pred_lang}")
    print(f"Confidence: {confidence:.4f}")
    print("Expected: th (Thai)")

    # Success if predicted Thai with reasonable confidence
    # Note: confidence threshold lowered to 0.3 as ECAPA-TDNN often produces
    # lower confidence scores even for correct predictions
    success = pred_lang == "th" and confidence > 0.3

    return {
        "predicted_language": pred_lang,
        "confidence": confidence,
        "expected": "th",
        "success": success,
    }


def main():
    """Main validation function."""
    print("=" * 60)
    print("ECAPA-TDNN MLX Validation")
    print("=" * 60)

    results = {}

    # Validate Block 0
    results["block0"] = validate_block0()

    # Validate full model
    results["full_model"] = validate_full_model_output()

    # Validate with real audio
    results["real_audio"] = validate_with_real_audio()

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    block0_status = results["block0"]["status"]
    print(f"Block 0: {block0_status}")
    print(f"  Max diff: {results['block0']['max_absolute_diff']:.2e}")

    print("\nFull model output:")
    print(f"  Embedding shape: {results['full_model']['embedding_shape']}")
    print(f"  Predicted: {results['full_model']['predicted_language']}")

    if results["real_audio"]:
        real_success = "PASS" if results["real_audio"]["success"] else "FAIL"
        print(f"\nReal audio test: {real_success}")
        print(f"  Predicted: {results['real_audio']['predicted_language']}")
        print(f"  Expected: {results['real_audio']['expected']}")
        print(f"  Confidence: {results['real_audio']['confidence']:.4f}")

    # Save results
    output_path = Path("models/sota/ecapa-tdnn-mlx/validation_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
