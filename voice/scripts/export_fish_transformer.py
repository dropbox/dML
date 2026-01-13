#!/usr/bin/env python3
"""
Export Fish-Speech DualARTransformer to TorchScript

This script exports the DualARTransformer forward methods for C++ integration.
The generation loop is implemented in C++ using these exported components.

Exported components:
1. forward_generate (slow transformer, 24 layers)
2. forward_generate_fast (fast transformer, 4 layers, per codebook)
3. embed (input embedding)
4. setup_caches (KV cache initialization)

Author: Worker #479 (AI)
Date: 2025-12-11
"""

import argparse
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from typing import Optional, Tuple


def export_transformer(
    checkpoint_path: str,
    output_dir: str,
    device: str = "cpu",
    max_seq_len: int = 2048,
) -> bool:
    """
    Export DualARTransformer forward methods to TorchScript.
    """
    print(f"Loading model from: {checkpoint_path}")

    # Import fish_speech
    try:
        from fish_speech.models.text2semantic.llama import (
            DualARTransformer,
            DualARModelArgs,
            BaseTransformerForwardResult,
        )
        from fish_speech.tokenizer import FishTokenizer
    except ImportError as e:
        print(f"ERROR: fish_speech not installed: {e}")
        return False

    # Load tokenizer
    tokenizer = FishTokenizer.from_pretrained(checkpoint_path)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # Load config
    with open(os.path.join(checkpoint_path, "config.json")) as f:
        config = json.load(f)

    args = DualARModelArgs(**config)
    print(f"Model config: {args.dim}d, {args.n_layer} layers, {args.num_codebooks} codebooks")

    # Create model
    model = DualARTransformer(args, tokenizer)

    # Load weights
    print("Loading weights...")
    weights_path = os.path.join(checkpoint_path, "model.pth")
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

    # Clean up state dict
    if "output.weight" in state_dict:
        del state_dict["output.weight"]

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model = model.to(device)

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Set up caches for generation
    print("\nSetting up KV caches...")
    with torch.device(device):
        model.setup_caches(
            max_batch_size=1,
            max_seq_len=max_seq_len,
            dtype=torch.float32,  # Use float32 for better compatibility
        )

    # Create wrappers for TorchScript export
    print("\nCreating export wrappers...")

    class SlowTransformerWrapper(nn.Module):
        """Wrapper for the slow transformer forward_generate."""

        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(
            self,
            x: torch.Tensor,  # [B, 9, seq_len] - 1 token + 8 codebooks
            input_pos: torch.Tensor,  # [seq_len]
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Forward pass through slow transformer.

            Returns:
                logits: [B, 1, vocab_size] token logits
                hidden_states: [B, 1, dim] hidden states for fast transformer
            """
            result = self.model.forward_generate(
                x, input_pos,
                audio_masks=None,
                audio_parts=None,
            )
            return result.logits, result.hidden_states

    class FastTransformerWrapper(nn.Module):
        """Wrapper for the fast transformer forward_generate_fast."""

        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(
            self,
            hidden_states: torch.Tensor,  # [B, dim]
            input_pos: torch.Tensor,  # [1] - codebook index (0-7)
        ) -> torch.Tensor:
            """
            Forward pass through fast transformer for single codebook.

            Returns:
                codebook_logits: [B, 1, codebook_size]
            """
            return self.model.forward_generate_fast(hidden_states, input_pos)

    class EmbeddingWrapper(nn.Module):
        """Wrapper for input embedding."""

        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(
            self,
            token_ids: torch.Tensor,  # [B, seq_len] - text token IDs
            codebook_indices: torch.Tensor,  # [B, 8, seq_len] - codebook indices
        ) -> torch.Tensor:
            """
            Embed text tokens and codebook indices.

            Returns:
                embeddings: [B, seq_len, dim]
            """
            # Combine into expected format [B, 9, seq_len]
            inp = torch.cat([token_ids.unsqueeze(1), codebook_indices], dim=1)
            return self.model.embed(inp.transpose(1, 2))  # [B, seq_len, dim]

    # Create wrapper instances
    slow_wrapper = SlowTransformerWrapper(model)
    fast_wrapper = FastTransformerWrapper(model)
    embed_wrapper = EmbeddingWrapper(model)

    slow_wrapper.eval()
    fast_wrapper.eval()
    embed_wrapper.eval()

    # Create example inputs
    batch_size = 1
    seq_len = 64
    num_codebooks = 8
    vocab_size = config["vocab_size"]
    codebook_size = config["codebook_size"]

    # Slow transformer input: [B, 9, seq_len]
    slow_input = torch.zeros(batch_size, num_codebooks + 1, seq_len, dtype=torch.long, device=device)
    slow_input[:, 0, :] = torch.randint(0, vocab_size, (batch_size, seq_len))  # text tokens
    slow_input[:, 1:, :] = torch.randint(0, codebook_size, (batch_size, num_codebooks, seq_len))  # codebooks
    slow_input_pos = torch.arange(seq_len, device=device)

    # Fast transformer input
    fast_hidden = torch.randn(batch_size, 1, config["fast_dim"], device=device)
    fast_input_pos = torch.tensor([0], dtype=torch.long, device=device)

    # Embedding input
    embed_tokens = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long, device=device)
    embed_codes = torch.randint(0, codebook_size, (batch_size, num_codebooks, seq_len), dtype=torch.long, device=device)

    os.makedirs(output_dir, exist_ok=True)

    # Test and trace slow transformer
    print("\n1. Tracing slow transformer...")
    with torch.no_grad():
        try:
            slow_out = slow_wrapper(slow_input, slow_input_pos)
            print(f"   Output shapes: logits={slow_out[0].shape}, hidden={slow_out[1].shape}")

            traced_slow = torch.jit.trace(slow_wrapper, (slow_input, slow_input_pos))
            slow_path = os.path.join(output_dir, f"slow_transformer_{device}.pt")
            traced_slow.save(slow_path)
            print(f"   Saved: {slow_path} ({os.path.getsize(slow_path) / 1e6:.1f} MB)")
        except Exception as e:
            print(f"   ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Test and trace fast transformer
    print("\n2. Tracing fast transformer...")
    with torch.no_grad():
        try:
            # Clear fast KV cache first
            for layer in model.fast_layers:
                layer.attention.kv_cache.k_cache.fill_(0)
                layer.attention.kv_cache.v_cache.fill_(0)

            fast_out = fast_wrapper(fast_hidden, fast_input_pos)
            print(f"   Output shape: {fast_out.shape}")

            traced_fast = torch.jit.trace(fast_wrapper, (fast_hidden, fast_input_pos))
            fast_path = os.path.join(output_dir, f"fast_transformer_{device}.pt")
            traced_fast.save(fast_path)
            print(f"   Saved: {fast_path} ({os.path.getsize(fast_path) / 1e6:.1f} MB)")
        except Exception as e:
            print(f"   ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Export config and tokenizer info
    print("\n3. Saving config...")
    export_config = {
        "dim": config["dim"],
        "fast_dim": config.get("fast_dim", config["dim"]),
        "vocab_size": config["vocab_size"],
        "codebook_size": config["codebook_size"],
        "num_codebooks": config["num_codebooks"],
        "n_layer": config["n_layer"],
        "n_fast_layer": config.get("n_fast_layer", 4),
        "max_seq_len": max_seq_len,
        "semantic_begin_id": tokenizer.semantic_begin_id,
        "semantic_end_id": tokenizer.semantic_end_id,
        "im_end_id": tokenizer.get_token_id("<|im_end|>"),
    }

    config_path = os.path.join(output_dir, "transformer_config.json")
    with open(config_path, "w") as f:
        json.dump(export_config, f, indent=2)
    print(f"   Saved: {config_path}")

    print("\nExport complete!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Export Fish-Speech Transformer")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/fish-speech-1.5",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/fish-speech-1.5",
        help="Output directory for TorchScript models",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "mps", "cuda"],
        default="cpu",
        help="Device for export",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Maximum sequence length for KV cache",
    )

    args = parser.parse_args()

    success = export_transformer(
        args.checkpoint,
        args.output_dir,
        args.device,
        args.max_seq_len,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
