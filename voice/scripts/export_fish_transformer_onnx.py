#!/usr/bin/env python3
"""
Export Fish-Speech DualARTransformer to ONNX

ONNX export handles dynamic shapes better than TorchScript.
The KV cache is passed as external inputs/outputs rather than embedded in the model.

Strategy:
1. Export embedding layer
2. Export transformer blocks with explicit KV cache I/O
3. Export output projection

Author: Worker #481 (AI)
Date: 2025-12-11
"""

import argparse
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from typing import Optional, Tuple, List


def export_transformer_onnx(
    checkpoint_path: str,
    output_dir: str,
    max_seq_len: int = 2048,
    opset_version: int = 17,
) -> bool:
    """
    Export DualARTransformer to ONNX format.
    """
    print(f"Loading model from: {checkpoint_path}")
    print(f"ONNX opset version: {opset_version}")

    # Import fish_speech
    try:
        from fish_speech.models.text2semantic.llama import (
            DualARTransformer,
            DualARModelArgs,
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
    print(f"Model config: {args.dim}d, {args.n_layer} slow + {args.n_fast_layer} fast layers")

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

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    os.makedirs(output_dir, exist_ok=True)

    # ==========================================================================
    # 1. Export Embedding Layer
    # ==========================================================================
    print("\n1. Exporting embedding layer...")

    class EmbeddingONNX(nn.Module):
        """Text token + codebook embedding"""
        def __init__(self, model):
            super().__init__()
            self.embeddings = model.embeddings  # Text token embedding
            self.codebook_embeddings = model.codebook_embeddings  # Codebook embedding (combined)

        def forward(self, token_ids: torch.Tensor, codebook_indices: torch.Tensor) -> torch.Tensor:
            """
            Args:
                token_ids: [B, seq_len] - text token IDs
                codebook_indices: [B, 8, seq_len] - 8 codebook indices (summed with offset)

            Returns:
                embeddings: [B, seq_len, dim]
            """
            # Embed text tokens
            text_emb = self.embeddings(token_ids)  # [B, seq_len, dim]

            # Embed codebooks - fish-speech uses a single embedding with offset
            # Each codebook has codebook_size codes, so codebook i uses range [i*1024, (i+1)*1024)
            codebook_emb = torch.zeros_like(text_emb)
            for i in range(8):
                offset = i * 1024  # codebook_size
                codebook_emb = codebook_emb + self.codebook_embeddings(codebook_indices[:, i, :] + offset)

            return text_emb + codebook_emb

    emb_model = EmbeddingONNX(model)
    emb_model.eval()

    # Example inputs
    batch_size = 1
    seq_len = 64
    dummy_tokens = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
    dummy_codes = torch.randint(0, config["codebook_size"], (batch_size, 8, seq_len))

    try:
        emb_path = os.path.join(output_dir, "embedding.onnx")
        with torch.no_grad():
            torch.onnx.export(
                emb_model,
                (dummy_tokens, dummy_codes),
                emb_path,
                input_names=["token_ids", "codebook_indices"],
                output_names=["embeddings"],
                dynamic_axes={
                    "token_ids": {0: "batch", 1: "seq_len"},
                    "codebook_indices": {0: "batch", 2: "seq_len"},
                    "embeddings": {0: "batch", 1: "seq_len"},
                },
                opset_version=opset_version,
                do_constant_folding=True,
            )
        size_mb = os.path.getsize(emb_path) / 1e6
        print(f"   Saved: {emb_path} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"   ERROR exporting embedding: {e}")
        import traceback
        traceback.print_exc()

    # ==========================================================================
    # 2. Export Single Slow Transformer Block (for C++ loop)
    # ==========================================================================
    print("\n2. Exporting slow transformer block...")

    class SlowBlockONNX(nn.Module):
        """Single slow transformer block with explicit KV cache I/O"""
        def __init__(self, block, layer_idx: int, dim: int, head_dim: int, n_head: int):
            super().__init__()
            self.attention = block.attention
            self.feed_forward = block.feed_forward
            self.ffn_norm = block.ffn_norm
            self.attention_norm = block.attention_norm

            self.layer_idx = layer_idx
            self.dim = dim
            self.head_dim = head_dim
            self.n_head = n_head

        def forward(
            self,
            x: torch.Tensor,              # [B, seq_len, dim]
            freqs_cis: torch.Tensor,      # [seq_len, head_dim/2, 2] rotary embeddings
            k_cache: torch.Tensor,        # [B, max_seq_len, n_head, head_dim]
            v_cache: torch.Tensor,        # [B, max_seq_len, n_head, head_dim]
            cache_position: torch.Tensor, # [1] - current position
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Returns:
                output: [B, seq_len, dim]
                k_cache_updated: [B, max_seq_len, n_head, head_dim]
                v_cache_updated: [B, max_seq_len, n_head, head_dim]
            """
            # Pre-norm attention
            h = self.attention_norm(x)

            # Compute Q, K, V
            bsz, seqlen, _ = h.shape
            xq = self.attention.wq(h).view(bsz, seqlen, self.n_head, self.head_dim)
            xk = self.attention.wk(h).view(bsz, seqlen, self.n_head, self.head_dim)
            xv = self.attention.wv(h).view(bsz, seqlen, self.n_head, self.head_dim)

            # Apply rotary embeddings
            # Note: Simplified version, full rotary embedding may need separate export

            # Update KV cache
            pos = cache_position.item()
            k_cache[:, pos:pos+seqlen] = xk
            v_cache[:, pos:pos+seqlen] = xv

            # Compute attention
            # Full attention logic here (simplified for export exploration)

            # FFN
            out = self.feed_forward(self.ffn_norm(x))

            return out + x, k_cache, v_cache

    # Note: Full block export is complex due to attention. Start with simpler approach.

    # ==========================================================================
    # 3. Export Output Head
    # ==========================================================================
    print("\n3. Exporting output head...")

    class OutputHeadONNX(nn.Module):
        """Output projection for token logits"""
        def __init__(self, model):
            super().__init__()
            self.norm = model.norm
            self.output = model.output

        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            """
            Args:
                hidden: [B, seq_len, dim]
            Returns:
                logits: [B, seq_len, vocab_size]
            """
            return self.output(self.norm(hidden))

    head_model = OutputHeadONNX(model)
    head_model.eval()

    dummy_hidden = torch.randn(batch_size, seq_len, config["dim"])

    try:
        head_path = os.path.join(output_dir, "output_head.onnx")
        with torch.no_grad():
            torch.onnx.export(
                head_model,
                (dummy_hidden,),
                head_path,
                input_names=["hidden_states"],
                output_names=["logits"],
                dynamic_axes={
                    "hidden_states": {0: "batch", 1: "seq_len"},
                    "logits": {0: "batch", 1: "seq_len"},
                },
                opset_version=opset_version,
                do_constant_folding=True,
            )
        size_mb = os.path.getsize(head_path) / 1e6
        print(f"   Saved: {head_path} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"   ERROR exporting output head: {e}")
        import traceback
        traceback.print_exc()

    # ==========================================================================
    # 4. Export Fast Transformer Block (for codebook generation)
    # ==========================================================================
    print("\n4. Exporting fast transformer head...")

    class FastHeadONNX(nn.Module):
        """Fast transformer codebook projection"""
        def __init__(self, model):
            super().__init__()
            self.fast_norm = model.fast_norm
            self.fast_output = model.fast_output  # Linear layer for all codebooks

        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            """
            Args:
                hidden: [B, 1, fast_dim] - hidden state from fast transformer
            Returns:
                logits: [B, 1, num_codebooks * codebook_size]
            """
            # Norm and project
            h = self.fast_norm(hidden)
            return self.fast_output(h)

    fast_model = FastHeadONNX(model)
    fast_model.eval()

    fast_dim = config.get("fast_dim", config["dim"])
    dummy_fast_hidden = torch.randn(batch_size, 1, fast_dim)
    dummy_codebook_idx = torch.tensor([0], dtype=torch.long)

    try:
        # Export single fast output head (covers all 8 codebooks)
        fast_path = os.path.join(output_dir, "fast_output.onnx")

        with torch.no_grad():
            torch.onnx.export(
                fast_model,
                (dummy_fast_hidden,),
                fast_path,
                input_names=["hidden_states"],
                output_names=["logits"],
                dynamic_axes={
                    "hidden_states": {0: "batch"},
                    "logits": {0: "batch"},
                },
                opset_version=opset_version,
                do_constant_folding=True,
            )
        size_mb = os.path.getsize(fast_path) / 1e6
        print(f"   Saved: {fast_path} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"   ERROR exporting fast output: {e}")
        import traceback
        traceback.print_exc()

    # ==========================================================================
    # 5. Save Config
    # ==========================================================================
    print("\n5. Saving config...")
    export_config = {
        "dim": config["dim"],
        "fast_dim": config.get("fast_dim", config["dim"]),
        "vocab_size": config["vocab_size"],
        "codebook_size": config["codebook_size"],
        "num_codebooks": config["num_codebooks"],
        "n_layer": config["n_layer"],
        "n_fast_layer": config.get("n_fast_layer", 4),
        "n_head": config["n_head"],
        "head_dim": config["head_dim"],
        "max_seq_len": max_seq_len,
        "semantic_begin_id": tokenizer.semantic_begin_id,
        "semantic_end_id": tokenizer.semantic_end_id,
        "im_end_id": tokenizer.get_token_id("<|im_end|>"),
        "export_format": "onnx",
        "opset_version": opset_version,
    }

    config_path = os.path.join(output_dir, "transformer_onnx_config.json")
    with open(config_path, "w") as f:
        json.dump(export_config, f, indent=2)
    print(f"   Saved: {config_path}")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 60)
    print("ONNX Export Summary")
    print("=" * 60)
    print("Exported components:")
    print("  - embedding.onnx: Text + codebook embedding")
    print("  - output_head.onnx: Token logits projection")
    print("  - fast_output.onnx: Fast transformer codebook logits")
    print()
    print("Note: Full transformer block export requires:")
    print("  - Explicit KV cache handling")
    print("  - Rotary embedding export")
    print("  - Attention mechanism export")
    print()
    print("Recommended approach for full model:")
    print("  - Use ONNX for embedding + output heads (small models)")
    print("  - Implement transformer blocks in C++ with libtorch primitives")
    print("  - OR use llama.cpp with GGUF quantized model (if available)")
    print("=" * 60)

    return True


def main():
    parser = argparse.ArgumentParser(description="Export Fish-Speech Transformer to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/fish-speech-1.5",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/fish-speech-1.5/onnx",
        help="Output directory for ONNX models",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version",
    )

    args = parser.parse_args()

    success = export_transformer_onnx(
        args.checkpoint,
        args.output_dir,
        args.max_seq_len,
        args.opset,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
