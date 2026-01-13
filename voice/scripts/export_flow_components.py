#!/usr/bin/env python3
"""
Export CosyVoice2 Flow components for C++ inference.

This script exports the components needed to convert speech tokens to mel spectrogram:
- flow_input_embedding: Embedding(vocab_size, 512) - token → embedding
- flow_encoder: The causal encoder that processes token embeddings
- flow_encoder_proj: Linear(encoder_dim, 80) - project to mel dimension
- flow_spk_embed: Linear(spk_dim, 80) - speaker embedding projection
- flow_decoder: CFM decoder for mel generation

Usage:
    source .venv/bin/activate
    python scripts/export_flow_components.py
"""

import sys
import os
import torch
import torch.nn as nn
from pathlib import Path

# Add cosyvoice_repo to path
sys.path.insert(0, str(Path(__file__).parent.parent / "cosyvoice_repo"))

# Disable CUDA
torch.cuda.is_available = lambda: False


def load_cosyvoice_model(model_dir: str):
    """Load CosyVoice2 model."""
    from cosyvoice.cli.cosyvoice import CosyVoice2
    print(f"Loading CosyVoice2 from {model_dir}...")
    model = CosyVoice2(model_dir, load_jit=False, load_trt=False, fp16=False)
    return model


class FlowInputEmbeddingWrapper(nn.Module):
    """Wrapper for flow.input_embedding."""
    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.embedding = embedding

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert speech tokens to embeddings.

        Args:
            token_ids: (batch, seq_len) token IDs

        Returns:
            embeddings: (batch, seq_len, embed_dim)
        """
        # Clamp to valid range (negative tokens are padding)
        return self.embedding(torch.clamp(token_ids, min=0))


class FlowEncoderProjWrapper(nn.Module):
    """Wrapper for flow.encoder_proj."""
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project encoder output to mel dimension."""
        return self.linear(hidden)


class SpkEmbedWrapper(nn.Module):
    """Wrapper for speaker embedding projection."""
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Project and normalize speaker embedding.

        Args:
            embedding: (batch, spk_dim) raw speaker embedding

        Returns:
            projected: (batch, 80) projected embedding
        """
        # Normalize and project (matches CosyVoice2 inference)
        embedding = torch.nn.functional.normalize(embedding, dim=1)
        return self.linear(embedding)


class SimpleFlowInference(nn.Module):
    """
    Simplified flow inference without streaming/caching.

    Converts speech tokens directly to mel spectrogram using:
    1. Token embedding
    2. Encoder
    3. Encoder projection
    4. CFM decoder
    """
    def __init__(self, flow, max_seq_len: int = 500):
        super().__init__()
        self.input_embedding = flow.input_embedding
        self.encoder = flow.encoder
        self.encoder_proj = flow.encoder_proj
        self.decoder = flow.decoder
        self.spk_embed_affine_layer = flow.spk_embed_affine_layer
        self.output_size = flow.output_size
        self.token_mel_ratio = getattr(flow, 'token_mel_ratio', 2)

        # Pre-register sequence range as buffer - moves with model.to(device)
        # This avoids torch.arange() device capture during tracing
        self.register_buffer('_seq_range', torch.arange(max_seq_len).unsqueeze(0))

    def forward(self,
                token: torch.Tensor,
                token_len: torch.Tensor,
                spk_embedding: torch.Tensor) -> torch.Tensor:
        """
        Generate mel spectrogram from speech tokens.

        Args:
            token: (1, seq_len) speech tokens
            token_len: (1,) sequence length
            spk_embedding: (1, 192) speaker embedding

        Returns:
            mel: (1, 80, mel_len) mel spectrogram
        """
        # Project speaker embedding
        spk_embedding = torch.nn.functional.normalize(spk_embedding, dim=1)
        spk_embedding = self.spk_embed_affine_layer(spk_embedding)

        # Create mask from token length
        batch_size = token.shape[0]
        max_len = token.shape[1]
        # mask: (batch, seq, 1) where True = valid
        # Use pre-registered buffer to avoid torch.arange() device capture during tracing
        # The buffer moves with the model to MPS, so it's always on the correct device
        seq_range = self._seq_range[:, :max_len]
        mask = (seq_range < token_len.unsqueeze(1)).unsqueeze(-1).float()

        # Embed tokens
        token_emb = self.input_embedding(torch.clamp(token, min=0)) * mask

        # Encode (non-streaming mode)
        h, h_lengths = self.encoder(token_emb, token_len, streaming=False)

        # Project to mel dimension
        h = self.encoder_proj(h)

        # Calculate mel length (2x token length for CosyVoice2)
        mel_len = h.shape[1]

        # Create conditions (zeros - no prompt)
        # Use new_zeros which inherits device from h tensor
        conds = h.new_zeros([1, mel_len, self.output_size])
        conds = conds.transpose(1, 2)

        # Create mel mask using new_ones which inherits device
        mel_mask = token.new_ones([1, 1, mel_len], dtype=torch.bool)

        # Run CFM decoder
        mel, _ = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mel_mask,
            spks=spk_embedding,
            cond=conds,
            n_timesteps=10,
            streaming=False
        )

        return mel.float()


def export_flow_components(model, output_dir: Path, device: str = "cpu"):
    """Export individual flow components."""
    flow = model.model.flow
    flow.eval()

    print("\n=== Analyzing Flow Model ===")
    print(f"  Type: {type(flow).__name__}")
    print(f"  input_size: {flow.input_size}")
    print(f"  output_size: {flow.output_size}")
    print(f"  vocab_size: {flow.vocab_size}")
    if hasattr(flow, 'token_mel_ratio'):
        print(f"  token_mel_ratio: {flow.token_mel_ratio}")
    if hasattr(flow, 'pre_lookahead_len'):
        print(f"  pre_lookahead_len: {flow.pre_lookahead_len}")

    # Export input embedding
    print("\n=== Exporting flow_input_embedding ===")
    input_emb = flow.input_embedding
    print(f"  Embedding: {input_emb.num_embeddings} x {input_emb.embedding_dim}")

    wrapper = FlowInputEmbeddingWrapper(input_emb).to(device)
    wrapper.eval()

    example_tokens = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long, device=device)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example_tokens)
        output = traced(example_tokens)
        print(f"  Output shape: {output.shape}")

    path = output_dir / "flow_input_embedding.pt"
    traced.save(str(path))
    print(f"  Saved: {path} ({path.stat().st_size / 1024 / 1024:.2f} MB)")

    # Export encoder_proj
    print("\n=== Exporting flow_encoder_proj ===")
    encoder_proj = flow.encoder_proj
    print(f"  Linear: {encoder_proj.in_features} -> {encoder_proj.out_features}")

    wrapper = FlowEncoderProjWrapper(encoder_proj).to(device)
    wrapper.eval()

    example_hidden = torch.randn(1, 10, encoder_proj.in_features, device=device)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example_hidden)
        output = traced(example_hidden)
        print(f"  Output shape: {output.shape}")

    path = output_dir / "flow_encoder_proj.pt"
    traced.save(str(path))
    print(f"  Saved: {path} ({path.stat().st_size / 1024:.2f} KB)")

    # Export speaker embedding layer
    print("\n=== Exporting flow_spk_embed ===")
    spk_layer = flow.spk_embed_affine_layer
    print(f"  Linear: {spk_layer.in_features} -> {spk_layer.out_features}")

    wrapper = SpkEmbedWrapper(spk_layer).to(device)
    wrapper.eval()

    example_spk = torch.randn(1, spk_layer.in_features, device=device)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example_spk)
        output = traced(example_spk)
        print(f"  Output shape: {output.shape}")

    path = output_dir / "flow_spk_embed.pt"
    traced.save(str(path))
    print(f"  Saved: {path} ({path.stat().st_size / 1024:.2f} KB)")

    # Write flow parameters
    print("\n=== Writing flow_info.txt ===")
    info_path = output_dir / "flow_info.txt"
    with open(info_path, "w") as f:
        f.write(f"input_size={flow.input_size}\n")
        f.write(f"output_size={flow.output_size}\n")
        f.write(f"vocab_size={flow.vocab_size}\n")
        f.write(f"encoder_dim={encoder_proj.in_features}\n")
        f.write(f"mel_dim={encoder_proj.out_features}\n")
        f.write(f"spk_dim={spk_layer.in_features}\n")
        if hasattr(flow, 'token_mel_ratio'):
            f.write(f"token_mel_ratio={flow.token_mel_ratio}\n")
        if hasattr(flow, 'pre_lookahead_len'):
            f.write(f"pre_lookahead_len={flow.pre_lookahead_len}\n")
    print(f"  Saved: {info_path}")


def try_export_simple_flow(model, output_dir: Path, device: str = "cpu", max_tokens: int = 200):
    """Try to export simplified full flow inference."""
    print(f"\n=== Attempting Full Flow Export (max_tokens={max_tokens}) ===")

    flow = model.model.flow
    flow.eval()
    flow.to(device)

    try:
        wrapper = SimpleFlowInference(flow)
        wrapper.eval()
        wrapper.to(device)

        # Example inputs - use larger size for better tracing generalization
        token = torch.randint(0, 6000, (1, max_tokens), dtype=torch.long, device=device)
        token_len = torch.tensor([max_tokens], dtype=torch.long, device=device)
        spk_emb = torch.randn(1, 192, device=device)

        print("  Tracing full flow...")
        with torch.no_grad():
            traced = torch.jit.trace(wrapper, (token, token_len, spk_emb), strict=False)

        # Test
        output = traced(token, token_len, spk_emb)
        print(f"  Output shape: {output.shape}")

        path = output_dir / "flow_full.pt"
        traced.save(str(path))
        print(f"  Saved: {path} ({path.stat().st_size / 1024 / 1024:.2f} MB)")
        return True

    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    project_root = Path(__file__).parent.parent
    model_dir = project_root / "models/cosyvoice/CosyVoice2-0.5B"
    output_dir = project_root / "models/cosyvoice/torchscript"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CosyVoice2 Flow Component Export")
    print("=" * 60)

    model = load_cosyvoice_model(str(model_dir))

    # Export individual components with MPS for Metal GPU acceleration
    # Worker #460: Fixed EspnetRelPositionalEncoding trace divergence issue
    # Now using MPS - traced device ops will use MPS device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Exporting with device: {device}")
    export_flow_components(model, output_dir, device)

    # Try full flow export
    try_export_simple_flow(model, output_dir, device)

    # Summary
    print("\n" + "=" * 60)
    print("Export Summary")
    print("=" * 60)
    for f in sorted(output_dir.glob("*.pt")) + sorted(output_dir.glob("*.txt")):
        if f.is_file():
            size = f.stat().st_size
            if size > 1024 * 1024:
                print(f"  {f.name}: {size / 1024 / 1024:.2f} MB")
            else:
                print(f"  {f.name}: {size / 1024:.2f} KB")


if __name__ == "__main__":
    main()
