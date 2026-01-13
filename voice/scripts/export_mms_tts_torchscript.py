#!/usr/bin/env python3
"""
Export MMS-TTS VITS models to TorchScript for C++ integration.

MMS-TTS (Massively Multilingual Speech) models from Meta provide high-quality
TTS for Arabic, Turkish, and Persian. This script exports them to TorchScript
for use in the pure C++ pipeline.

Models:
    - facebook/mms-tts-ara: Arabic (36.3M params)
    - facebook/mms-tts-tur: Turkish (36.3M params)
    - facebook/mms-tts-fas: Persian/Farsi (36.3M params)

Usage:
    # Export all three languages
    python scripts/export_mms_tts_torchscript.py --all

    # Export specific language
    python scripts/export_mms_tts_torchscript.py --lang ar
    python scripts/export_mms_tts_torchscript.py --lang tr
    python scripts/export_mms_tts_torchscript.py --lang fa

    # Export with MPS acceleration
    python scripts/export_mms_tts_torchscript.py --all --device mps

Copyright 2025 Andrew Yates. All rights reserved.
"""

import argparse
import json
import os
import struct
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch

# MMS-TTS model configurations
MODELS = {
    "ar": {
        "model_id": "facebook/mms-tts-ara",
        "name": "Arabic",
        "sample_rate": 16000,
        "test_text": "مرحبا كيف حالك",  # Hello, how are you
    },
    "tr": {
        "model_id": "facebook/mms-tts-tur",
        "name": "Turkish",
        "sample_rate": 16000,
        "test_text": "Merhaba nasılsınız",  # Hello, how are you
    },
    "fa": {
        "model_id": "facebook/mms-tts-fas",
        "name": "Persian",
        "sample_rate": 16000,
        "test_text": "سلام حالتان چطور است",  # Hello, how are you
    },
}


class MMSTTSWrapper(torch.nn.Module):
    """Wrapper for MMS-TTS VITS model with clean TorchScript interface."""

    def __init__(self, model, use_half=False):
        super().__init__()
        self.model = model
        self.use_half = use_half
        if use_half:
            self.model = self.model.half()
        self.model.eval()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Generate audio from tokenized text.

        Args:
            input_ids: [batch, seq_len] int64 tensor of token IDs

        Returns:
            waveform: [batch, num_samples] float32 audio at 16kHz
        """
        # Create attention mask (all ones for valid tokens)
        attention_mask = torch.ones_like(input_ids)

        # Run inference
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        return output.waveform


def save_wav(audio: np.ndarray, path: str, sample_rate: int = 16000):
    """Save audio as WAV file."""
    audio = (audio * 32767).astype(np.int16)

    with open(path, 'wb') as f:
        # WAV header
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + len(audio) * 2))
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))  # Chunk size
        f.write(struct.pack('<H', 1))   # PCM
        f.write(struct.pack('<H', 1))   # Mono
        f.write(struct.pack('<I', sample_rate))
        f.write(struct.pack('<I', sample_rate * 2))
        f.write(struct.pack('<H', 2))   # Block align
        f.write(struct.pack('<H', 16))  # Bits per sample
        f.write(b'data')
        f.write(struct.pack('<I', len(audio) * 2))
        f.write(audio.tobytes())


def export_mms_tts(lang: str, output_dir: str, device: str = "cpu", use_half: bool = False):
    """
    Export MMS-TTS model to TorchScript.

    Args:
        lang: Language code (ar, tr, fa)
        output_dir: Output directory
        device: Target device (cpu, mps, cuda)
        use_half: Use FP16 precision (GPU only)

    Returns:
        Paths to exported model, vocab, and test audio
    """
    from transformers import VitsModel, AutoTokenizer

    config = MODELS[lang]
    precision = "FP16" if use_half else "FP32"
    print(f"\n{'='*60}")
    print(f"Exporting MMS-TTS {config['name']} ({lang})")
    print(f"Model: {config['model_id']}")
    print(f"Device: {device}, Precision: {precision}")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    # Load model and tokenizer
    print(f"\n[1/5] Loading model from HuggingFace...")
    model = VitsModel.from_pretrained(config["model_id"])
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"])

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model parameters: {num_params:.1f}M")
    print(f"  Sample rate: {config['sample_rate']} Hz")
    print(f"  Vocab size: {model.config.vocab_size}")

    # Move to device
    if device != "cpu":
        model = model.to(device)
        print(f"  Moved to {device}")

    # Export tokenizer vocab
    print(f"\n[2/5] Exporting tokenizer vocabulary...")
    vocab_path = os.path.join(output_dir, f"mms_tts_{lang}_vocab.json")

    # Get vocab from tokenizer
    vocab = tokenizer.get_vocab()
    vocab_config = {
        "lang": lang,
        "name": config["name"],
        "model_id": config["model_id"],
        "sample_rate": config["sample_rate"],
        "vocab_size": len(vocab),
        "vocab": vocab,
    }

    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_config, f, ensure_ascii=False, indent=2)
    print(f"  Saved vocab ({len(vocab)} tokens) to {vocab_path}")

    # Create wrapper and prepare for tracing
    print(f"\n[3/5] Creating TorchScript wrapper...")
    wrapper = MMSTTSWrapper(model, use_half=use_half)

    # Prepare example input
    test_text = config["test_text"]
    inputs = tokenizer(test_text, return_tensors="pt")
    input_ids = inputs["input_ids"]

    if device != "cpu":
        input_ids = input_ids.to(device)

    print(f"  Test text: {test_text}")
    print(f"  Input shape: {input_ids.shape}")

    # Trace the model
    print(f"\n[4/5] Tracing model (this may take a moment)...")
    start = time.time()

    with torch.no_grad():
        try:
            traced = torch.jit.trace(wrapper, (input_ids,))
            trace_time = time.time() - start
            print(f"  Traced successfully in {trace_time:.1f}s")
        except Exception as e:
            print(f"  WARNING: Tracing failed: {e}")
            print("  Attempting script export...")
            try:
                traced = torch.jit.script(wrapper)
                print("  Scripted successfully")
            except Exception as e2:
                print(f"  ERROR: Script export also failed: {e2}")
                print("  Falling back to state_dict export...")

                # Save just the state dict for manual loading
                state_path = os.path.join(output_dir, f"mms_tts_{lang}_state.pt")
                torch.save({
                    "state_dict": model.state_dict(),
                    "config": model.config.to_dict(),
                }, state_path)
                print(f"  Saved state dict to {state_path}")
                return state_path, vocab_path, None

    # Save TorchScript model
    suffix = f"_{device}_fp16" if use_half else f"_{device}"
    model_path = os.path.join(output_dir, f"mms_tts_{lang}{suffix}.pt")
    traced.save(model_path)
    model_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"  Saved model ({model_size:.1f} MB) to {model_path}")

    # Test inference and save audio
    print(f"\n[5/5] Testing inference...")
    with torch.no_grad():
        start = time.time()
        output = traced(input_ids)
        elapsed = (time.time() - start) * 1000

    audio = output.squeeze().cpu().numpy()
    duration = len(audio) / config["sample_rate"]

    print(f"  Output shape: {output.shape}")
    print(f"  Inference time: {elapsed:.0f}ms")
    print(f"  Audio duration: {duration:.2f}s")
    print(f"  RTF: {elapsed/1000/duration:.2f}")

    # Save test audio
    test_audio_path = os.path.join(output_dir, f"mms_tts_{lang}_test.wav")
    save_wav(audio, test_audio_path, config["sample_rate"])
    print(f"  Saved test audio to {test_audio_path}")

    # Verify the exported model loads correctly
    print(f"\n  Verifying exported model...")
    loaded = torch.jit.load(model_path)
    with torch.no_grad():
        verify_output = loaded(input_ids.cpu() if device != "cpu" else input_ids)

    # Note: VITS models have stochastic duration prediction, so output lengths
    # may vary between runs. We verify by checking:
    # 1. Model loads successfully
    # 2. Output is non-empty
    # 3. Output is non-silent (has audio content)
    verify_audio = verify_output.squeeze().cpu().numpy()
    verify_rms = np.sqrt(np.mean(verify_audio**2))

    print(f"  Loaded model output shape: {verify_output.shape}")
    print(f"  Loaded model audio RMS: {verify_rms:.4f}")

    if verify_output.shape[-1] > 0 and verify_rms > 0.01:
        print(f"  Verification PASSED (non-empty, non-silent)")
    else:
        print(f"  WARNING: Verification failed (empty or silent output)")

    print(f"\n{'='*60}")
    print(f"Export complete for {config['name']} ({lang})")
    print(f"  Model: {model_path}")
    print(f"  Vocab: {vocab_path}")
    print(f"  Test:  {test_audio_path}")
    print(f"{'='*60}")

    return model_path, vocab_path, test_audio_path


def main():
    parser = argparse.ArgumentParser(
        description='Export MMS-TTS VITS models to TorchScript',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Export all three languages (Arabic, Turkish, Persian)
    python scripts/export_mms_tts_torchscript.py --all

    # Export Arabic only with MPS acceleration
    python scripts/export_mms_tts_torchscript.py --lang ar --device mps

    # Export all with FP16 for smaller models
    python scripts/export_mms_tts_torchscript.py --all --device mps --half
"""
    )
    parser.add_argument('--lang', choices=['ar', 'tr', 'fa'],
                       help='Language to export (ar=Arabic, tr=Turkish, fa=Persian)')
    parser.add_argument('--all', action='store_true',
                       help='Export all three languages')
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu',
                       help='Device for export (default: cpu)')
    parser.add_argument('--output', default='models/mms-tts',
                       help='Output directory (default: models/mms-tts)')
    parser.add_argument('--half', action='store_true',
                       help='Export in FP16 (half precision) for smaller models')
    args = parser.parse_args()

    # Validate arguments
    if not args.lang and not args.all:
        parser.error("Either --lang or --all must be specified")

    # Check device availability
    if args.device == 'mps' and not torch.backends.mps.is_available():
        print("Warning: MPS not available, falling back to CPU")
        args.device = 'cpu'
    elif args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = 'cpu'

    # FP16 requires GPU
    if args.half and args.device == 'cpu':
        print("Warning: FP16 not supported on CPU, ignoring --half flag")
        args.half = False

    # Determine which languages to export
    if args.all:
        languages = ['ar', 'tr', 'fa']
    else:
        languages = [args.lang]

    # Export each language
    results = {}
    for lang in languages:
        try:
            model_path, vocab_path, audio_path = export_mms_tts(
                lang, args.output, args.device, args.half
            )
            results[lang] = {
                "status": "success",
                "model": model_path,
                "vocab": vocab_path,
                "test_audio": audio_path,
            }
        except Exception as e:
            print(f"\nERROR exporting {lang}: {e}")
            results[lang] = {"status": "failed", "error": str(e)}

    # Print summary
    print("\n" + "=" * 60)
    print("EXPORT SUMMARY")
    print("=" * 60)

    for lang, result in results.items():
        name = MODELS[lang]["name"]
        if result["status"] == "success":
            print(f"  {name} ({lang}): SUCCESS")
            print(f"    Model: {result['model']}")
        else:
            print(f"  {name} ({lang}): FAILED - {result['error']}")

    print("=" * 60)

    # Return success if all exports succeeded
    all_success = all(r["status"] == "success" for r in results.values())
    return 0 if all_success else 1


if __name__ == '__main__':
    exit(main())
