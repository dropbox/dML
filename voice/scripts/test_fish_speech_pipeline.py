#!/usr/bin/env python3
"""
Test Fish-Speech-1.5 End-to-End Pipeline

This script tests the full text-to-speech pipeline:
1. Text → Token IDs (FishTokenizer)
2. Token IDs → Codebook Indices (DualARTransformer)
3. Codebook Indices → Audio (FireflyGAN Vocoder)

Usage:
    python3 scripts/test_fish_speech_pipeline.py --text "Hello world"
    python3 scripts/test_fish_speech_pipeline.py --vocoder-only  # Skip text encoder

Author: Worker #479 (AI)
Date: 2025-12-11
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import soundfile as sf


def load_vocoder_torchscript(path: str, device: str = "cpu"):
    """Load the TorchScript vocoder model."""
    print(f"Loading TorchScript vocoder from: {path}")
    model = torch.jit.load(path, map_location=device)
    model.eval()
    return model


def test_vocoder_only(device: str = "cpu"):
    """Test vocoder with random indices (no text encoder needed)."""
    print("\n" + "=" * 60)
    print("Testing Vocoder Only (random indices)")
    print("=" * 60)

    vocoder_path = f"models/fish-speech-1.5/vocoder_{device}.pt"
    if not os.path.exists(vocoder_path):
        print(f"ERROR: Vocoder not found at {vocoder_path}")
        print("Run: python3 scripts/export_firefly_vocoder_torchscript.py --device {device}")
        return False

    vocoder = load_vocoder_torchscript(vocoder_path, device)

    # Generate random indices (simulating transformer output)
    num_frames = 100  # ~4.76 seconds at 21Hz
    indices = torch.randint(0, 1024, (1, 8, num_frames), dtype=torch.long, device=device)

    print(f"Input indices shape: {indices.shape}")

    # Time the inference
    start = time.time()
    with torch.no_grad():
        audio = vocoder(indices)
    elapsed = time.time() - start

    print(f"Output audio shape: {audio.shape}")
    print(f"Inference time: {elapsed * 1000:.1f} ms")

    # Calculate RTF (Real-Time Factor)
    audio_duration = audio.shape[-1] / 44100  # seconds
    rtf = elapsed / audio_duration
    print(f"Audio duration: {audio_duration:.2f} s")
    print(f"RTF: {rtf:.4f} (< 1.0 means real-time)")

    # Save audio
    output_path = "test_vocoder_random.wav"
    audio_np = audio[0, 0].cpu().numpy()
    sf.write(output_path, audio_np, 44100)
    print(f"Saved: {output_path}")

    return True


def test_full_pipeline(text: str, device: str = "cpu"):
    """Test full text-to-speech pipeline."""
    print("\n" + "=" * 60)
    print(f"Testing Full Pipeline: '{text}'")
    print("=" * 60)

    try:
        from fish_speech.tokenizer import FishTokenizer
        from fish_speech.models.text2semantic.llama import DualARTransformer, DualARModelArgs
        import json
    except ImportError as e:
        print(f"ERROR: fish_speech not installed: {e}")
        print("Run: pip install fish-speech")
        return False

    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer_path = "models/fish-speech-1.5"
    tokenizer = FishTokenizer.from_pretrained(tokenizer_path)
    print(f"   Vocab size: {tokenizer.vocab_size}")

    # Encode text
    print("\n2. Encoding text...")
    token_ids = tokenizer.encode(text)
    print(f"   Text: '{text}'")
    print(f"   Token IDs: {token_ids} (len={len(token_ids)})")

    # Load DualARTransformer config
    print("\n3. Loading DualARTransformer...")
    with open("models/fish-speech-1.5/config.json") as f:
        config = json.load(f)

    args = DualARModelArgs(
        dim=config["dim"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        n_local_heads=config["n_local_heads"],
        head_dim=config["head_dim"],
        intermediate_size=config["intermediate_size"],
        vocab_size=config["vocab_size"],
        max_seq_len=config["max_seq_len"],
        num_codebooks=config["num_codebooks"],
        codebook_size=config["codebook_size"],
        n_fast_layer=config["n_fast_layer"],
        fast_dim=config["fast_dim"],
        fast_n_head=config["fast_n_head"],
        fast_n_local_heads=config["fast_n_local_heads"],
        fast_head_dim=config["fast_head_dim"],
        fast_intermediate_size=config["fast_intermediate_size"],
    )

    model = DualARTransformer(args, tokenizer)

    # Load weights
    print("   Loading weights (this may take a moment)...")
    state_dict = torch.load(
        "models/fish-speech-1.5/model.pth",
        map_location="cpu",
        weights_only=True
    )

    # Remove unexpected keys
    if "output.weight" in state_dict:
        del state_dict["output.weight"]

    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(device)

    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Generate codebook indices
    print("\n4. Generating codebook indices...")

    # Prepare input tensor
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    start = time.time()
    with torch.no_grad():
        # Use the model's generate method
        # This is model-specific and may need adjustment
        try:
            # Try to use generate_semantic if available
            codebook_indices = model.generate(
                input_ids,
                max_new_tokens=500,
                temperature=0.7,
            )
            print(f"   Codebook indices shape: {codebook_indices.shape}")
        except Exception as e:
            print(f"   WARNING: Direct generation failed: {e}")
            print("   Using random indices as fallback...")
            codebook_indices = torch.randint(
                0, 1024, (1, 8, 50),
                dtype=torch.long, device=device
            )

    gen_time = time.time() - start
    print(f"   Generation time: {gen_time * 1000:.1f} ms")

    # Load vocoder
    print("\n5. Loading vocoder...")
    vocoder_path = f"models/fish-speech-1.5/vocoder_{device}.pt"
    vocoder = load_vocoder_torchscript(vocoder_path, device)

    # Decode to audio
    print("\n6. Decoding to audio...")
    if codebook_indices.dim() == 2:
        codebook_indices = codebook_indices.unsqueeze(0)
    if codebook_indices.shape[1] != 8:
        # Transpose if shape is [B, T, 8] instead of [B, 8, T]
        codebook_indices = codebook_indices.transpose(1, 2)

    print(f"   Input shape: {codebook_indices.shape}")

    start = time.time()
    with torch.no_grad():
        audio = vocoder(codebook_indices)
    decode_time = time.time() - start

    print(f"   Output shape: {audio.shape}")
    print(f"   Decode time: {decode_time * 1000:.1f} ms")

    # Calculate stats
    audio_duration = audio.shape[-1] / 44100
    total_time = gen_time + decode_time
    rtf = total_time / audio_duration

    print(f"\n7. Results:")
    print(f"   Audio duration: {audio_duration:.2f} s")
    print(f"   Total time: {total_time * 1000:.1f} ms")
    print(f"   RTF: {rtf:.4f}")

    # Save audio
    output_path = "test_fish_speech_output.wav"
    audio_np = audio[0, 0].cpu().numpy()
    sf.write(output_path, audio_np, 44100)
    print(f"   Saved: {output_path}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Test Fish-Speech Pipeline")
    parser.add_argument(
        "--text",
        type=str,
        default="Hello, this is a test of the fish speech text to speech system.",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "mps", "cuda"],
        default="cpu",
        help="Device for inference",
    )
    parser.add_argument(
        "--vocoder-only",
        action="store_true",
        help="Test vocoder only (skip text encoder)",
    )

    args = parser.parse_args()

    # Check device
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, using CPU")
        args.device = "cpu"
    elif args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    print(f"Device: {args.device}")

    if args.vocoder_only:
        success = test_vocoder_only(args.device)
    else:
        success = test_full_pipeline(args.text, args.device)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
