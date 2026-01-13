#!/usr/bin/env python3
"""
Export StyleTTS2 modules to TorchScript for C++ loading.

This script loads the Python StyleTTS2 model and exports each component
to TorchScript format (.pt files) that can be loaded by libtorch in C++.

Usage:
    python scripts/export_styletts2.py --checkpoint /path/to/styletts2 --output /path/to/output

The script exports these modules:
    - bert.pt         - BERT encoder for text features
    - text_encoder.pt - Text-to-hidden encoder
    - style_encoder.pt - Style extraction
    - prosody.pt      - Prosody predictor (F0, duration)
    - decoder.pt      - Acoustic decoder
    - hifigan.pt      - HiFi-GAN vocoder

Requirements:
    pip install torch transformers phonemizer librosa
    # Plus StyleTTS2 dependencies
"""

import argparse
import sys
import torch
from pathlib import Path


def export_module(module, example_input, output_path: Path, method: str = "trace"):
    """
    Export a PyTorch module to TorchScript.

    Args:
        module: PyTorch module to export
        example_input: Example input(s) for tracing
        output_path: Path to save .pt file
        method: "trace" or "script"

    Returns:
        True if successful
    """
    module.eval()

    try:
        if method == "trace":
            if isinstance(example_input, (list, tuple)):
                traced = torch.jit.trace(module, example_input)
            else:
                traced = torch.jit.trace(module, (example_input,))
        else:
            traced = torch.jit.script(module)

        traced.save(str(output_path))
        print(f"  ✓ Exported: {output_path.name} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
        return True

    except Exception as e:
        print(f"  ✗ Failed to export {output_path.name}: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Export StyleTTS2 to TorchScript")
    parser.add_argument("--checkpoint", required=True, help="Path to StyleTTS2 checkpoint")
    parser.add_argument("--output", required=True, help="Output directory for .pt files")
    parser.add_argument("--device", default="cpu", help="Device for export (cpu recommended)")
    parser.add_argument("--config", help="Path to config.yml (auto-detected if not specified)")

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect config
    config_path = args.config
    if not config_path:
        for candidate in ["config.yml", "configs/config.yml"]:
            candidate_path = checkpoint_path / candidate
            if candidate_path.exists():
                config_path = str(candidate_path)
                break

    if not config_path:
        print("ERROR: Could not find config.yml", file=sys.stderr)
        sys.exit(1)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")
    print(f"Device: {args.device}")
    print()

    # Load StyleTTS2
    print("Loading StyleTTS2 model...")

    try:
        # Import StyleTTS2 - adjust path as needed
        sys.path.insert(0, str(checkpoint_path.parent / "StyleTTS2"))

        import yaml
        import importlib
        models_mod = importlib.import_module("models")
        utils_mod = importlib.import_module("utils")
        build_model = models_mod.build_model

        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Build model
        model = build_model(config)

        # Load checkpoint
        ckpt_file = checkpoint_path / "epochs_2nd_00020.pth"  # Adjust as needed
        if not ckpt_file.exists():
            # Try to find any .pth file
            pth_files = list(checkpoint_path.glob("*.pth"))
            if pth_files:
                ckpt_file = pth_files[0]
            else:
                print(f"ERROR: No .pth checkpoint found in {checkpoint_path}", file=sys.stderr)
                sys.exit(1)

        print(f"Loading checkpoint: {ckpt_file}")
        state_dict = torch.load(ckpt_file, map_location=args.device)

        # Load weights
        for key, module in model.items():
            if key in state_dict:
                module.load_state_dict(state_dict[key])
                module.to(args.device)
                module.eval()

        print("Model loaded successfully")
        print()

    except ImportError as e:
        print(f"ERROR: Failed to import StyleTTS2: {e}", file=sys.stderr)
        print("\nMake sure StyleTTS2 repository is cloned and in PYTHONPATH", file=sys.stderr)
        print("Try: git clone https://github.com/yl4579/StyleTTS2.git", file=sys.stderr)
        sys.exit(1)

    except Exception as e:
        print(f"ERROR: Failed to load model: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Export each module
    print("Exporting modules to TorchScript...")

    device = torch.device(args.device)
    success_count = 0
    total_count = 0

    # 1. BERT - text feature extraction
    if "bert" in model:
        total_count += 1
        # BERT input: token_ids [batch, seq_len]
        example_tokens = torch.randint(0, 30000, (1, 50), device=device)
        if export_module(model["bert"], example_tokens, output_dir / "bert.pt", "trace"):
            success_count += 1

    # 2. Text Encoder
    if "text_encoder" in model:
        total_count += 1
        # Input: text features from BERT [batch, seq_len, hidden]
        example_text = torch.randn(1, 50, 768, device=device)
        if export_module(model["text_encoder"], example_text, output_dir / "text_encoder.pt", "trace"):
            success_count += 1

    # 3. Style Encoder
    if "style_encoder" in model:
        total_count += 1
        # Input: mel spectrogram [batch, mel_dim, time]
        example_mel = torch.randn(1, 80, 100, device=device)
        if export_module(model["style_encoder"], example_mel, output_dir / "style_encoder.pt", "trace"):
            success_count += 1

    # 4. Prosody Predictor (F0 + Duration)
    if "predictor" in model:
        total_count += 1
        # Input: encoded text, style
        example_enc = torch.randn(1, 50, 512, device=device)
        example_style = torch.randn(1, 128, device=device)
        if export_module(model["predictor"], (example_enc, example_style), output_dir / "prosody.pt", "trace"):
            success_count += 1

    # 5. Decoder
    if "decoder" in model:
        total_count += 1
        # Input: encoded + F0 + style
        example_enc = torch.randn(1, 512, 50, device=device)
        example_f0 = torch.randn(1, 1, 200, device=device)
        example_style = torch.randn(1, 128, device=device)
        if export_module(model["decoder"], (example_enc, example_f0, example_style), output_dir / "decoder.pt", "trace"):
            success_count += 1

    # 6. HiFi-GAN Vocoder
    if "hifigan" in model or "generator" in model:
        total_count += 1
        vocoder = model.get("hifigan") or model.get("generator")
        # Input: mel spectrogram [batch, mel_dim, time]
        example_mel = torch.randn(1, 80, 100, device=device)
        if export_module(vocoder, example_mel, output_dir / "hifigan.pt", "trace"):
            success_count += 1

    print()
    print(f"Export complete: {success_count}/{total_count} modules")

    if success_count < total_count:
        print("\nSome modules failed to export. This may be due to:")
        print("  - Dynamic control flow (variable-length loops)")
        print("  - Data-dependent conditionals")
        print("  - Unsupported operations")
        print("\nConsider using torch.jit.script() with manual annotations,")
        print("or rewriting modules to be more trace-friendly.")
        sys.exit(1)

    print(f"\nTorchScript modules saved to: {output_dir}")
    print("\nTo use in C++:")
    print(f'  torch::jit::load("{output_dir}/bert.pt");')


if __name__ == "__main__":
    main()
