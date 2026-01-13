#!/usr/bin/env python3
"""
Export Fish-Speech 1.5 to TorchScript for C++ integration.

This script exports the Fish-Speech model components for native C++ inference:
1. DualARTransformer - Text to semantic tokens (autoregressive)
2. Firefly-GAN Vocoder - Codebook indices to audio waveform

IMPORTANT: Use Python 3.11 venv for this script!
    source venv311/bin/activate
    python scripts/export_fish_speech.py --device cpu --output models/fish-speech

Architecture Notes:
- DualARTransformer: 533M params, 24+4 layers, 8 codebooks
- Firefly-GAN: ~180MB, ConvNeXt encoder + HiFi-GAN decoder
- Inference is autoregressive with KV caching (complex for TorchScript)

Status: WORK IN PROGRESS
- Model loading: COMPLETE
- Tokenizer: COMPLETE
- TorchScript export: TODO (requires wrapper for static shapes)

Copyright 2025 Andrew Yates. All rights reserved.
"""

import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")


def load_dual_ar_transformer(model_dir: str, device: str = "cpu"):
    """Load the DualARTransformer for text-to-semantic generation.

    Returns:
        model: DualARTransformer instance
        tokenizer: FishTokenizer instance
        config: Model configuration dict
    """
    import torch
    from fish_speech.tokenizer import FishTokenizer
    from fish_speech.models.text2semantic.llama import DualARTransformer, DualARModelArgs

    # Load tokenizer
    tokenizer = FishTokenizer.from_pretrained(model_dir)
    print(f"[export] Loaded tokenizer (vocab_size={tokenizer.vocab_size})")

    # Load config
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path) as f:
        config = json.load(f)
    print(f"[export] Loaded config from {config_path}")

    # Create model args
    args = DualARModelArgs(
        dim=config['dim'],
        n_layer=config['n_layer'],
        n_head=config['n_head'],
        n_local_heads=config['n_local_heads'],
        head_dim=config['head_dim'],
        intermediate_size=config['intermediate_size'],
        vocab_size=config['vocab_size'],
        max_seq_len=config['max_seq_len'],
        num_codebooks=config['num_codebooks'],
        codebook_size=config['codebook_size'],
        n_fast_layer=config['n_fast_layer'],
        fast_dim=config['fast_dim'],
        fast_n_head=config['fast_n_head'],
        fast_n_local_heads=config['fast_n_local_heads'],
        fast_head_dim=config['fast_head_dim'],
        fast_intermediate_size=config['fast_intermediate_size'],
    )

    # Create model
    model = DualARTransformer(args, tokenizer)
    print(f"[export] Created model ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")

    # Load weights
    weights_path = os.path.join(model_dir, 'model.pth')
    print(f"[export] Loading weights from {weights_path}...")
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)

    # Remove unexpected key if present
    if 'output.weight' in state_dict:
        del state_dict['output.weight']

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"[export] Model loaded on {device}")

    return model, tokenizer, config


def export_tokenizer_vocab(tokenizer, output_dir: str):
    """Export tokenizer vocabulary for C++ usage.

    Exports:
    - vocab.json: Token to ID mapping
    - special_tokens.json: Special token IDs
    """
    vocab_path = os.path.join(output_dir, 'vocab.json')

    # Export mergeable ranks as vocab
    vocab = {}
    for token_bytes, rank in tokenizer.tkt_model._mergeable_ranks.items():
        try:
            token_str = token_bytes.decode('utf-8')
            vocab[token_str] = rank
        except UnicodeDecodeError:
            # Use hex encoding for non-UTF8 tokens
            vocab[f"<0x{token_bytes.hex()}>"] = rank

    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"[export] Saved vocab ({len(vocab)} tokens) to {vocab_path}")

    # Export special tokens
    special_tokens_path = os.path.join(output_dir, 'special_tokens.json')
    with open(special_tokens_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer.all_special_tokens_with_ids, f, ensure_ascii=False, indent=2)
    print(f"[export] Saved special tokens to {special_tokens_path}")

    return vocab_path, special_tokens_path


def main():
    parser = argparse.ArgumentParser(description='Export Fish-Speech to TorchScript')
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu',
                       help='Device for export (default: cpu)')
    parser.add_argument('--input', default='models/fish-speech-1.5',
                       help='Input model directory')
    parser.add_argument('--output', default='models/fish-speech',
                       help='Output directory for exported files')
    parser.add_argument('--vocab-only', action='store_true',
                       help='Only export vocabulary (no model)')
    args = parser.parse_args()

    # Check Python version
    if sys.version_info < (3, 11):
        print(f"WARNING: Python 3.11+ recommended. Current: {sys.version_info.major}.{sys.version_info.minor}")
        print("Use: source venv311/bin/activate")

    os.makedirs(args.output, exist_ok=True)

    try:
        import torch
        print(f"[export] PyTorch version: {torch.__version__}")
    except ImportError:
        print("ERROR: PyTorch not found. Install with: pip install torch")
        return 1

    # Load model and tokenizer
    print(f"\n[export] Loading Fish-Speech from {args.input}...")
    model, tokenizer, config = load_dual_ar_transformer(args.input, args.device)

    # Export vocabulary
    print(f"\n[export] Exporting vocabulary...")
    export_tokenizer_vocab(tokenizer, args.output)

    if args.vocab_only:
        print("\n[export] Vocabulary export complete (--vocab-only)")
        return 0

    # TODO: TorchScript export
    # The DualARTransformer uses autoregressive generation with dynamic loops,
    # which makes TorchScript export challenging. Options:
    #
    # 1. Create a wrapper with static shapes for fixed-length generation
    # 2. Export only the non-autoregressive components
    # 3. Use ONNX export instead (may handle dynamic shapes better)
    # 4. Use torch.compile with MPS backend (Python runtime but fast)
    #
    # For now, we export the vocabulary and config for C++ tokenization.

    print("\n[export] TorchScript export not yet implemented")
    print("  See reports/main/PHASE5_FISH_SPEECH_EXPORT_2025-12-11.md for details")
    print("  The model uses autoregressive generation requiring wrapper for export")

    # Save config for C++ reference
    config_out = os.path.join(args.output, 'config.json')
    with open(config_out, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"[export] Saved config to {config_out}")

    print("\n[export] Export complete!")
    print(f"  Output directory: {args.output}")
    print(f"  Files: vocab.json, special_tokens.json, config.json")

    return 0


if __name__ == '__main__':
    sys.exit(main())
