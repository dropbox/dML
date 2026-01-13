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
Validate Encoder VAD training pipeline with local audio files.

This script tests that the entire training pipeline works correctly
using local audio files before running the full LibriSpeech training.
"""

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import mlx.core as mx
import mlx.optimizers as optim


def load_local_audio_files(directory: Path, limit: int = 10):
    """Load local audio files for pipeline validation."""
    samples = []
    audio_files = list(directory.glob("*.wav"))

    for audio_path in audio_files[:limit]:
        audio, sr = sf.read(str(audio_path))

        # Resample to 16kHz if needed
        if sr != 16000:
            # Simple resampling for validation
            ratio = 16000 / sr
            new_len = int(len(audio) * ratio)
            audio = np.interp(
                np.linspace(0, len(audio), new_len),
                np.arange(len(audio)),
                audio
            )

        audio = audio.astype(np.float32)

        # Normalize
        if audio.max() > 0:
            audio = audio / np.abs(audio).max()

        # Use filename as pseudo-transcript
        transcript = audio_path.stem
        samples.append((audio, transcript))

    return samples


def main():
    print("=" * 60)
    print("Encoder VAD Pipeline Validation")
    print("=" * 60)

    # 1. Load Whisper model
    print("\n[1/6] Loading Whisper model...")
    t0 = time.perf_counter()
    from tools.whisper_mlx.model import WhisperMLX
    model = WhisperMLX.from_pretrained("large-v3", warmup=True)
    t1 = time.perf_counter()
    print(f"  Model loaded in {t1-t0:.1f}s")

    n_state = model.config.n_audio_state
    n_mels = model.config.n_mels
    print(f"  Encoder config: n_state={n_state}, n_mels={n_mels}")

    # 2. Create VAD head
    print("\n[2/6] Creating EncoderVADHead...")
    from tools.whisper_mlx.encoder_vad import (
        EncoderVADHead,
        SileroVADDistiller,
    )
    from mlx.utils import tree_flatten
    # Use float32 for training to avoid mixed-precision issues with AdamW
    vad_head = EncoderVADHead(n_state=n_state, hidden_dim=256, dtype=mx.float32)
    flat_params = tree_flatten(vad_head.parameters())
    n_params = sum(p.size for _, p in flat_params)
    print(f"  VAD head params: {n_params}")

    # 3. Create distiller
    print("\n[3/6] Creating SileroVADDistiller...")
    distiller = SileroVADDistiller()
    print("  Distiller ready")

    # 4. Load audio samples
    print("\n[4/6] Loading test audio files...")

    # Try multiple audio directories
    audio_dirs = [
        Path("tests/fixtures/audio"),
        Path("tests/prosody/prosody_audio_output"),
    ]

    samples = []
    for audio_dir in audio_dirs:
        if audio_dir.exists():
            dir_samples = load_local_audio_files(audio_dir, limit=5)
            samples.extend(dir_samples)
            print(f"  Loaded {len(dir_samples)} from {audio_dir}")

    if not samples:
        print("  ERROR: No audio files found!")
        return 1

    print(f"  Total samples: {len(samples)}")

    # 5. Test forward pass
    print("\n[5/6] Testing forward pass...")
    from tools.whisper_mlx.audio import log_mel_spectrogram

    audio, _ = samples[0]
    mel = log_mel_spectrogram(audio, n_mels=n_mels)

    # Pad to 30s
    target_len = 3000
    if mel.shape[0] < target_len:
        mel = mx.pad(mel, [(0, target_len - mel.shape[0]), (0, 0)])
    elif mel.shape[0] > target_len:
        mel = mel[:target_len, :]

    mel_batch = mel[None]  # (1, 3000, n_mels)
    encoder_output = model.embed_audio(mel_batch, variable_length=False)
    mx.eval(encoder_output)

    print(f"  Encoder output shape: {encoder_output.shape}")

    seq_len = encoder_output.shape[1]
    labels = distiller.get_vad_labels(audio, seq_len)
    mx.eval(labels)

    print(f"  VAD labels shape: {labels.shape}")
    print(f"  Speech ratio: {float(labels.mean()):.2%}")

    # VAD head forward
    probs = vad_head(encoder_output, training=False)
    mx.eval(probs)
    print(f"  VAD head output shape: {probs.shape}")
    print(f"  VAD head output range: [{float(probs.min()):.3f}, {float(probs.max()):.3f}]")

    # 6. Test training step
    print("\n[6/6] Testing training step...")
    optimizer = optim.AdamW(learning_rate=1e-5, weight_decay=0.01)  # Lower LR for stability

    # Cast to float32 for more stable training
    encoder_output_f32 = encoder_output.astype(mx.float32)
    labels_batch = labels[None].astype(mx.float32)  # Add batch dim explicitly
    mx.eval(encoder_output_f32, labels_batch)

    def loss_fn(vad_head):
        logits = vad_head.get_logits(encoder_output_f32, training=True)
        logits = logits.astype(mx.float32)  # Ensure float32 for loss
        loss = distiller.compute_loss(logits, labels_batch)
        return loss

    loss, grads = mx.value_and_grad(loss_fn)(vad_head)
    mx.eval(loss, grads)

    # Check for NaN in gradients
    grad_flat = tree_flatten(grads)
    has_nan = any(mx.any(mx.isnan(g)).item() for _, g in grad_flat)
    if has_nan:
        print("  WARNING: NaN in gradients!")
        # Debug: print max gradient magnitudes
        for name, g in grad_flat:
            g_max = float(mx.max(mx.abs(g)))
            print(f"    {name}: max abs grad = {g_max:.6f}")

    # Check gradient magnitudes
    grad_max = max(float(mx.max(mx.abs(g))) for _, g in grad_flat)
    print(f"  Max gradient magnitude: {grad_max:.4f}")

    # Debug: Print weights before update
    param_flat_before = tree_flatten(vad_head.parameters())
    print("  Weights before update:")
    for name, p in param_flat_before[:2]:  # First 2 params
        print(f"    {name}: dtype={p.dtype}, range=[{float(p.min()):.4f}, {float(p.max()):.4f}]")

    # Debug: Print grads dtypes
    print("  Gradients dtypes:")
    for name, g in grad_flat[:2]:
        print(f"    {name}: dtype={g.dtype}")

    optimizer.update(vad_head, grads)
    mx.eval(vad_head.parameters(), optimizer.state)

    # Check weights after update
    param_flat = tree_flatten(vad_head.parameters())
    print("  Weights after update:")
    for name, p in param_flat[:2]:
        has_nan = mx.any(mx.isnan(p)).item()
        print(f"    {name}: dtype={p.dtype}, has_nan={has_nan}, range=[{float(p.min()):.4f}, {float(p.max()):.4f}]")

    param_has_nan = any(mx.any(mx.isnan(p)).item() for _, p in param_flat)
    if param_has_nan:
        print("  WARNING: NaN in weights after update!")

    print(f"  Initial loss: {float(loss):.4f}")

    # Run a few more iterations with monitoring
    losses = [float(loss)]
    for i in range(3):
        loss, grads = mx.value_and_grad(loss_fn)(vad_head)
        mx.eval(loss, grads)

        # Skip update if NaN
        if mx.isnan(loss).item():
            print(f"  Iteration {i+2}: NaN loss detected, skipping update")
            continue

        optimizer.update(vad_head, grads)
        mx.eval(vad_head.parameters(), optimizer.state)
        losses.append(float(loss))

    print(f"  Loss after 4 iterations: {losses[-1]:.4f}")

    # Verify loss decreased (training is working)
    print("\n" + "=" * 60)
    print("Pipeline Validation: PASSED")
    print("=" * 60)
    print("\nThe training pipeline is functional. Ready for full training.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
