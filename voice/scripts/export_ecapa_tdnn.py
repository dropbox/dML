#!/usr/bin/env python3
"""
Export ECAPA-TDNN speaker embedding model to TorchScript for C++ inference.

Phase 3.2 of Self-Speech Filtering System (Worker #294-295)

Usage:
    python scripts/export_ecapa_tdnn.py

Output:
    models/speaker/ecapa_tdnn.pt  - TorchScript model (~85MB)

The exported model includes the full pipeline:
- Mel filterbank feature extraction (computed in PyTorch)
- Mean normalization (per-feature, along time axis)
- ECAPA-TDNN embedding extraction
- L2 normalization

This allows the C++ code to pass raw 16kHz audio directly.
"""

import os
import sys
import torch
import torchaudio

# Patch torchaudio for SpeechBrain compatibility (torchaudio 2.9+ removed list_audio_backends)
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['ffmpeg', 'soundfile']

# Create output directory
os.makedirs("models/speaker", exist_ok=True)

print("=" * 60)
print("ECAPA-TDNN TorchScript Export")
print("=" * 60)

# Check for speechbrain
try:
    from speechbrain.inference.speaker import EncoderClassifier
except ImportError:
    print("\nERROR: SpeechBrain not installed")
    print("Install with: pip install speechbrain")
    sys.exit(1)

print("\n[1/6] Loading pre-trained ECAPA-TDNN model from HuggingFace...")
print("      Source: speechbrain/spkrec-ecapa-voxceleb")

# Load the pre-trained model
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="models/speaker/speechbrain_cache"
)

print("      Model loaded successfully")

# Extract modules
compute_features = classifier.mods.compute_features
embedding_model = classifier.mods.embedding_model

print(f"      compute_features: {type(compute_features).__name__}")
print(f"      embedding_model: {type(embedding_model).__name__}")

print("\n[2/6] Creating full pipeline wrapper module...")

class SpeakerEmbeddingPipeline(torch.nn.Module):
    """
    Full ECAPA-TDNN pipeline: raw audio -> embedding.

    Input: (batch, samples) at 16kHz mono
    Output: (batch, 192) L2-normalized embeddings

    The pipeline matches SpeechBrain's encode_batch exactly:
    1. Compute mel filterbank features
    2. Mean normalization (per-feature, along time axis, std_norm=False)
    3. ECAPA-TDNN embedding extraction
    4. L2 normalization
    """
    def __init__(self, compute_features, embedding_model):
        super().__init__()
        self.compute_features = compute_features
        self.embedding_model = embedding_model

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Extract speaker embedding from raw audio waveform.

        Args:
            wav: Audio tensor of shape (batch, samples) at 16kHz

        Returns:
            Embedding tensor of shape (batch, 192), L2-normalized
        """
        # Compute mel filterbank features
        # Input: (batch, samples)
        # Output: (batch, time, n_mels=80)
        feats = self.compute_features(wav)

        # Mean normalization (per-feature, along time axis)
        # This matches SpeechBrain's InputNormalization with:
        # - norm_type="sentence"
        # - mean_norm=True
        # - std_norm=False
        # Mean is computed per-feature (80 mel bins) across time frames
        mean_per_feat = feats.mean(dim=1, keepdim=True)  # (batch, 1, 80)
        feats = feats - mean_per_feat

        # The embedding model expects (batch, time, features)
        # Pass through ECAPA-TDNN
        embeddings = self.embedding_model(feats)

        # embeddings shape: (batch, 1, 192) -> (batch, 192)
        if embeddings.dim() == 3:
            embeddings = embeddings.squeeze(1)

        # L2 normalize for cosine similarity
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

# Create pipeline
pipeline = SpeakerEmbeddingPipeline(compute_features, embedding_model)
pipeline.eval()

print("\n[3/6] Testing pipeline with example input...")

# Create example input: 1 second of audio at 16kHz
example_input = torch.randn(1, 16000)

print(f"      Example input shape: {example_input.shape}")
print(f"      Sample rate: 16000 Hz")
print(f"      Duration: 1.0 seconds")

# Test forward pass
with torch.no_grad():
    test_output = pipeline(example_input)
    print(f"      Test output shape: {test_output.shape}")
    print(f"      Embedding dimension: {test_output.shape[1]}")
    print(f"      Embedding norm: {torch.norm(test_output, dim=1).item():.4f}")

# Verify against SpeechBrain's encode_batch
with torch.no_grad():
    sb_emb = classifier.encode_batch(example_input)
    sb_emb = sb_emb.squeeze(1)  # Remove middle dim
    sb_emb = torch.nn.functional.normalize(sb_emb, p=2, dim=1)

    cosine_sim = torch.nn.functional.cosine_similarity(test_output, sb_emb).item()
    print(f"      Cosine similarity to SpeechBrain: {cosine_sim:.4f}")

    if cosine_sim < 0.999:
        print("      WARNING: Output differs from SpeechBrain reference!")
        sys.exit(1)
    else:
        print("      VERIFIED: Matches SpeechBrain reference!")

print("\n[4/6] Tracing model to TorchScript...")

# We need to use scripting for better compatibility with control flow
# But first try tracing since it's simpler
with torch.no_grad():
    traced_model = torch.jit.trace(pipeline, example_input)

# Optimize for inference
traced_model = torch.jit.optimize_for_inference(traced_model)

print("\n[5/6] Saving model...")

output_path = "models/speaker/ecapa_tdnn.pt"
traced_model.save(output_path)

# Get file size
file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
print(f"      Saved to: {output_path}")
print(f"      File size: {file_size_mb:.1f} MB")

print("\n[6/6] Verifying exported model...")

# Load and test
loaded_model = torch.jit.load(output_path)
loaded_model.eval()

with torch.no_grad():
    verify_output = loaded_model(example_input)

# Check outputs match
max_diff = (test_output - verify_output).abs().max().item()
print(f"      Max difference from original: {max_diff:.2e}")

if max_diff < 1e-4:
    print("      VERIFIED: Outputs match")
else:
    print("      WARNING: Outputs differ significantly!")

# Verify against SpeechBrain again
with torch.no_grad():
    cosine_sim = torch.nn.functional.cosine_similarity(verify_output, sb_emb).item()
    print(f"      Cosine similarity to SpeechBrain: {cosine_sim:.4f}")

# Test with different durations
print("\n      Testing variable-length inputs:")
for duration_sec in [0.5, 1.0, 2.0, 3.0]:
    num_samples = int(duration_sec * 16000)
    test_wav = torch.randn(1, num_samples)
    with torch.no_grad():
        emb = loaded_model(test_wav)
        norm = torch.norm(emb, dim=1).item()
    print(f"        {duration_sec}s ({num_samples} samples) -> shape {emb.shape}, norm {norm:.4f}")

print("\n" + "=" * 60)
print("Export complete!")
print("=" * 60)
print(f"\nModel file: {output_path}")
print(f"Embedding dimension: 192")
print(f"Input sample rate: 16000 Hz")
print(f"Input format: (batch, samples) float32 in range [-1, 1]")
print("\nUsage in C++:")
print('  torch::jit::script::Module model = torch::jit::load("models/speaker/ecapa_tdnn.pt");')
print('  auto input = torch::randn({1, 16000});  // 1 second at 16kHz')
print('  auto embedding = model.forward({input}).toTensor();  // (1, 192)')
