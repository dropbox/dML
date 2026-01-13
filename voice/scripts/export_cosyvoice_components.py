#!/usr/bin/env python3
"""
Export CosyVoice2 components to TorchScript for C++ integration.

This script exports the components needed for llama.cpp + libtorch hybrid architecture:
- llm_decoder: Linear layer (896→6147) - converts hidden states to speech token logits
- speech_embedding: Embedding table (6147×896) - converts speech tokens to embeddings

The Qwen2 LLM is handled separately by llama.cpp (GGUF format).

Usage:
    python scripts/export_cosyvoice_components.py

Output:
    models/cosyvoice/torchscript/llm_decoder.pt
    models/cosyvoice/torchscript/speech_embedding.pt
    models/cosyvoice/torchscript/llm_embedding.pt (sos/eos/task embeddings)
"""

import sys
import os
import torch
import argparse
from pathlib import Path

# Add cosyvoice_repo to path
sys.path.insert(0, str(Path(__file__).parent.parent / "cosyvoice_repo"))

# Disable CUDA to avoid MPS issues during export
torch.cuda.is_available = lambda: False


def load_cosyvoice_model(model_dir: str):
    """Load CosyVoice2 model."""
    from cosyvoice.cli.cosyvoice import CosyVoice2

    print(f"Loading CosyVoice2 from {model_dir}...")
    model = CosyVoice2(model_dir, load_jit=False, load_trt=False, fp16=False)
    return model


class LLMDecoderWrapper(torch.nn.Module):
    """Wrapper for llm_decoder that can be traced."""
    def __init__(self, linear: torch.nn.Linear):
        super().__init__()
        self.linear = linear

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Convert hidden states to speech token logits.

        Args:
            hidden_states: (batch, seq_len, 896) or (batch, 896)

        Returns:
            logits: (batch, seq_len, 6147) or (batch, 6147)
        """
        return self.linear(hidden_states)


class SpeechEmbeddingWrapper(torch.nn.Module):
    """Wrapper for speech_embedding that can be traced."""
    def __init__(self, embedding: torch.nn.Embedding):
        super().__init__()
        self.embedding = embedding

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert speech token IDs to embeddings.

        Args:
            token_ids: (batch, seq_len) or (batch,) - token IDs in [0, 6146]

        Returns:
            embeddings: (batch, seq_len, 896) or (batch, 896)
        """
        return self.embedding(token_ids)


class LLMEmbeddingWrapper(torch.nn.Module):
    """Wrapper for special embeddings (sos_eos, task_id)."""
    def __init__(self, embedding: torch.nn.Embedding):
        super().__init__()
        self.embedding = embedding

    def forward(self, index: torch.Tensor) -> torch.Tensor:
        """
        Get special embedding by index.

        Args:
            index: scalar or (batch,) - 0 for sos_eos, 1 for task_id

        Returns:
            embedding: (896,) or (batch, 896)
        """
        return self.embedding(index)


def export_llm_decoder(model, output_dir: Path, device: str = "cpu"):
    """Export llm_decoder (Linear 896→6147) to TorchScript."""
    print("\n=== Exporting llm_decoder ===")

    # Get the llm_decoder from Qwen2LM
    llm = model.model.llm
    llm_decoder = llm.llm_decoder

    print(f"  Input size: {llm_decoder.in_features}")
    print(f"  Output size: {llm_decoder.out_features}")

    # Create wrapper
    wrapper = LLMDecoderWrapper(llm_decoder).to(device)
    wrapper.eval()

    # Create example input for tracing
    # Shape: (batch=1, seq_len=1, hidden_size=896)
    example_input = torch.randn(1, 1, llm_decoder.in_features, device=device)

    # Trace the model
    print("  Tracing...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example_input)

    # Verify output shape
    with torch.no_grad():
        output = traced(example_input)
        print(f"  Output shape: {output.shape}")
        assert output.shape == (1, 1, llm_decoder.out_features)

    # Save
    output_path = output_dir / "llm_decoder.pt"
    traced.save(str(output_path))
    print(f"  Saved to: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    return output_path


def export_speech_embedding(model, output_dir: Path, device: str = "cpu"):
    """Export speech_embedding (Embedding 6147×896) to TorchScript."""
    print("\n=== Exporting speech_embedding ===")

    # Get speech_embedding from Qwen2LM
    llm = model.model.llm
    speech_embedding = llm.speech_embedding

    print(f"  Vocab size: {speech_embedding.num_embeddings}")
    print(f"  Embedding dim: {speech_embedding.embedding_dim}")

    # Create wrapper
    wrapper = SpeechEmbeddingWrapper(speech_embedding).to(device)
    wrapper.eval()

    # Create example input for tracing
    # Shape: (batch=1, seq_len=1) - single token
    example_input = torch.tensor([[0]], dtype=torch.long, device=device)

    # Trace the model
    print("  Tracing...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example_input)

    # Verify output shape
    with torch.no_grad():
        output = traced(example_input)
        print(f"  Output shape: {output.shape}")
        assert output.shape == (1, 1, speech_embedding.embedding_dim)

    # Save
    output_path = output_dir / "speech_embedding.pt"
    traced.save(str(output_path))
    print(f"  Saved to: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    return output_path


def export_llm_embedding(model, output_dir: Path, device: str = "cpu"):
    """Export llm_embedding (special tokens: sos_eos, task_id) to TorchScript."""
    print("\n=== Exporting llm_embedding (special tokens) ===")

    # Get llm_embedding from Qwen2LM
    llm = model.model.llm
    llm_embedding = llm.llm_embedding

    print(f"  Vocab size: {llm_embedding.num_embeddings}")
    print(f"  Embedding dim: {llm_embedding.embedding_dim}")
    print(f"  Index 0 = sos_eos, Index 1 = task_id")

    # Create wrapper
    wrapper = LLMEmbeddingWrapper(llm_embedding).to(device)
    wrapper.eval()

    # Create example input for tracing
    example_input = torch.tensor([0], dtype=torch.long, device=device)

    # Trace the model
    print("  Tracing...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example_input)

    # Verify outputs
    with torch.no_grad():
        sos_eos = traced(torch.tensor([0], device=device))
        task_id = traced(torch.tensor([1], device=device))
        print(f"  sos_eos shape: {sos_eos.shape}")
        print(f"  task_id shape: {task_id.shape}")

    # Save
    output_path = output_dir / "llm_embedding.pt"
    traced.save(str(output_path))
    print(f"  Saved to: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024:.2f} KB")

    return output_path


def verify_existing_exports(output_dir: Path):
    """Check and report on existing exported models."""
    print("\n=== Checking Existing Exports ===")

    exported_dir = output_dir.parent / "exported"

    flow_path = exported_dir / "flow_encoder_traced.pt"
    hift_path = exported_dir / "hift_traced.pt"

    for path, name in [(flow_path, "Flow"), (hift_path, "HiFT")]:
        if path.exists():
            size_mb = path.stat().st_size / 1024 / 1024
            print(f"  {name}: {path} ({size_mb:.2f} MB)")

            # Try to load and verify
            try:
                loaded = torch.jit.load(str(path))
                print(f"    Status: VALID TorchScript")
            except Exception as e:
                print(f"    Status: INVALID - {e}")
        else:
            print(f"  {name}: NOT FOUND at {path}")


def export_model_info(model, output_dir: Path):
    """Export model architecture info for C++ reference."""
    print("\n=== Exporting Model Info ===")

    llm = model.model.llm

    info = {
        "llm_input_size": llm.llm_input_size,
        "llm_output_size": llm.llm_output_size,
        "speech_token_size": llm.speech_token_size,
        "sos_eos_id": llm.sos_eos,
        "task_id": llm.task_id,
        "fill_token_id": llm.fill_token,
        "stop_token_ids": llm.stop_token_ids,
        "sample_rate": model.sample_rate,
    }

    # Write as simple text file for easy C++ parsing
    info_path = output_dir / "model_info.txt"
    with open(info_path, "w") as f:
        for key, value in info.items():
            f.write(f"{key}={value}\n")

    print(f"  Model info:")
    for key, value in info.items():
        print(f"    {key}: {value}")
    print(f"  Saved to: {info_path}")

    return info_path


def main():
    parser = argparse.ArgumentParser(description="Export CosyVoice2 components to TorchScript")
    parser.add_argument("--model-dir", type=str,
                        default="models/cosyvoice/CosyVoice2-0.5B",
                        help="Path to CosyVoice2 model directory")
    parser.add_argument("--output-dir", type=str,
                        default="models/cosyvoice/torchscript",
                        help="Output directory for TorchScript models")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "mps"],
                        help="Device for export (use cpu for portable models)")
    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    model_dir = project_root / args.model_dir
    output_dir = project_root / args.output_dir

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CosyVoice2 Component Export")
    print("=" * 60)
    print(f"Model dir: {model_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Device: {args.device}")

    # Load model
    model = load_cosyvoice_model(str(model_dir))

    # Export components
    llm_decoder_path = export_llm_decoder(model, output_dir, args.device)
    speech_embedding_path = export_speech_embedding(model, output_dir, args.device)
    llm_embedding_path = export_llm_embedding(model, output_dir, args.device)
    info_path = export_model_info(model, output_dir)

    # Check existing exports
    verify_existing_exports(output_dir)

    # Summary
    print("\n" + "=" * 60)
    print("Export Summary")
    print("=" * 60)
    print(f"  llm_decoder.pt: {llm_decoder_path}")
    print(f"  speech_embedding.pt: {speech_embedding_path}")
    print(f"  llm_embedding.pt: {llm_embedding_path}")
    print(f"  model_info.txt: {info_path}")

    # Verify all can be loaded
    print("\n=== Verification ===")
    try:
        torch.jit.load(str(llm_decoder_path))
        print("  llm_decoder.pt: OK")
    except Exception as e:
        print(f"  llm_decoder.pt: FAILED - {e}")

    try:
        torch.jit.load(str(speech_embedding_path))
        print("  speech_embedding.pt: OK")
    except Exception as e:
        print(f"  speech_embedding.pt: FAILED - {e}")

    try:
        torch.jit.load(str(llm_embedding_path))
        print("  llm_embedding.pt: OK")
    except Exception as e:
        print(f"  llm_embedding.pt: FAILED - {e}")

    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
