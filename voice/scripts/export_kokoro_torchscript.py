#!/usr/bin/env python3
"""
Export Kokoro TTS to TorchScript for C++ integration.

Exports the Kokoro TorchScript model, vocab, voice pack, and test WAV.

CRITICAL - VOICE PACK FORMAT:
    The voice pack MUST be exported as the FULL tensor [N, 1, 256] where N is
    typically 510. Kokoro uses length-indexed voice selection: pack[len(phonemes)-1]

    Each phoneme length has a specific voice embedding for correct prosody.
    Exporting only the first embedding (voices_data[0]) causes:
    - ~200ms extra trailing silence
    - Shorter speech duration
    - Incorrect prosody for different sentence lengths

    This bug was discovered and fixed on 2025-12-11. See TestAudioQuality tests.

Output path:
- If --output ends with .pt/.pth, it is treated as the exact model path (directory
  is derived from the parent).
- Otherwise, --output is treated as an output directory and the model filename is
  derived from the device/dtype (kokoro_<device>[_fp16].pt).

Usage:
    python scripts/export_kokoro_torchscript.py --device mps --dtype float32 --output models/kokoro
    python scripts/export_kokoro_torchscript.py --device mps --dtype float32 --output models/kokoro/kokoro_mps.pt

dtype:
    - float32 (default): stable on MPS, ~328MB
    - float16: smaller/faster but less stable on MPS

Copyright 2025 Andrew Yates. All rights reserved.
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import torch


class KokoroWrapper(torch.nn.Module):
    """Wrapper to export Kokoro model with clean TorchScript interface."""

    def __init__(self, model, device, use_half=False):
        super().__init__()
        # KModel is a torch.nn.Module directly
        self.model = model.to(device)
        if use_half and device != 'cpu':
            # Convert to FP16 for faster inference on GPU
            self.model = self.model.half()
        self.model.eval()
        self.device = device
        self.use_half = use_half

    def forward(self, ids: torch.Tensor, ref: torch.Tensor, speed: torch.Tensor):
        """
        Generate audio from phoneme IDs.

        Args:
            ids: [1, seq_len] int64 tensor of phoneme token IDs
            ref: [1, 1, 256] float32 voice embedding
            speed: [1] float32 speed factor

        Returns:
            audio: [1, 1, num_samples] float32 audio at 24kHz
            duration: [1] int64 predicted duration
        """
        return self.model.forward_with_tokens(ids, ref, speed[0].item())


def resolve_output_paths(output_arg: str, device: str, use_half: bool):
    """
    Resolve output directory and model path.

    If output_arg ends with a known model suffix, treat it as the explicit model
    path and derive the directory from its parent. Otherwise, treat it as a
    directory and generate the model filename from device/dtype.
    """
    output_path = Path(output_arg)
    explicit_model_path = None

    if output_path.suffix in {".pt", ".pth", ".ts"}:
        explicit_model_path = output_path
        output_dir = output_path.parent
    else:
        output_dir = output_path

    suffix = f"_{device}_fp16" if use_half else f"_{device}"
    model_filename = output_dir / f"kokoro{suffix}.pt" if explicit_model_path is None else explicit_model_path

    return output_dir, model_filename


def export_kokoro(device: str, output_dir: str, voice: str = 'af_heart',
                  use_half: bool = False, disable_complex: bool = False,
                  explicit_model_path: Path | None = None):
    """Export Kokoro model components for C++ integration.

    Args:
        device: Target device ('cpu', 'mps', 'cuda')
        output_dir: Output directory for exported files
        voice: Voice pack to export (default: af_heart)
        use_half: Use FP16 precision (only on GPU)
        disable_complex: Use real-valued STFT instead of complex tensors.
                        NOTE: torch.angle() WORKS on MPS in PyTorch 2.9.1+.
                        Default is False for best quality (matches Python Kokoro).
                        Only use --disable-complex for older PyTorch versions.
    """

    output_dir_path = Path(output_dir)
    precision = "FP16 (half)" if use_half else "FP32 (float)"
    complex_mode = "disabled (MPS compatible)" if disable_complex else "enabled (uses torch.angle)"
    print(f"[export] Exporting Kokoro to TorchScript (device={device}, precision={precision}, complex={complex_mode})")
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Import Kokoro - use KModel directly to enable disable_complex flag
    print("[export] Loading Kokoro model...")
    from kokoro import KPipeline, KModel

    # Create model with disable_complex flag
    # disable_complex=False (default): Uses TorchSTFT with torch.angle - matches Python Kokoro exactly
    # disable_complex=True: Uses CustomSTFT (conv-based, real arithmetic) - only for PyTorch < 2.9.1
    # NOTE: PyTorch 2.9.1+ supports torch.angle() on MPS, so disable_complex is no longer needed
    model = KModel(repo_id='hexgrad/Kokoro-82M', disable_complex=disable_complex)

    # Create pipeline for voice loading (uses the model we created)
    pipe = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
    pipe.model = model  # Replace with our model that has the correct disable_complex setting

    # 1. Export vocabulary
    print("[export] Exporting vocabulary...")
    vocab = pipe.model.vocab
    vocab_path = output_dir_path / 'vocab.json'
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"  Saved vocab ({len(vocab)} chars) to {vocab_path}")

    # 2. Export voice embedding pack
    # CRITICAL: Export the FULL voice pack, not just the first embedding!
    # Python Kokoro's infer uses: pack[len(phonemes)-1]
    # Each phoneme length has a specific voice embedding for correct prosody.
    print(f"[export] Exporting voice pack ({voice})...")
    voices_data = pipe.load_voice(voice)  # [N, 1, 256] tensor of N style vectors
    print(f"  Voice pack shape: {voices_data.shape}")

    # VALIDATION: Ensure we have the full voice pack, not a single embedding
    MIN_VOICE_PACK_SIZE = 100  # Kokoro typically has 510 vectors
    if voices_data.dim() != 3:
        raise ValueError(f"Voice pack must be 3D [N, 1, 256], got {voices_data.dim()}D")
    if voices_data.size(0) < MIN_VOICE_PACK_SIZE:
        raise ValueError(
            f"Voice pack too small! Got {voices_data.size(0)} vectors, expected >= {MIN_VOICE_PACK_SIZE}. "
            f"This will cause trailing silence bugs. See 2025-12-11 bug fix."
        )
    if voices_data.size(2) != 256:
        raise ValueError(f"Voice embedding dim must be 256, got {voices_data.size(2)}")

    # Save the FULL voice pack for phoneme-length-indexed selection
    # Shape: [N, 1, 256] where N is typically 510 (for lengths 1-510)
    voice_path = output_dir_path / f'voice_{voice}.pt'
    torch.save(voices_data, voice_path)
    print(f"  Saved FULL voice pack {voices_data.shape} to {voice_path}")
    print(f"  (C++ will use pack[len(phonemes)-1] for each input)")

    # Verify saved file size (should be ~500KB for 510 vectors, not ~2KB for 1 vector)
    saved_size = voice_path.stat().st_size
    expected_min_size = 100 * 256 * 4  # 100 vectors * 256 dims * 4 bytes (float32)
    if saved_size < expected_min_size:
        raise ValueError(
            f"Voice file too small ({saved_size} bytes)! Expected >= {expected_min_size}. "
            f"Likely exported single embedding instead of full pack."
        )

    # 3. Create wrapper and trace
    print(f"[export] Creating TorchScript model on {device}...")
    wrapper = KokoroWrapper(pipe.model, device, use_half=use_half)

    # Create example inputs
    # "Hello" in IPA tokens from vocab
    example_phonemes = "hɛˈloʊ"
    example_ids = [vocab.get(c, 0) for c in example_phonemes]

    ids = torch.tensor([example_ids], dtype=torch.long, device=device)
    # ref_s expects [batch, 256] shape - squeeze middle dim from voice pack
    # For tracing, use voice embedding at index for phoneme count (len-1)
    phoneme_count = len(example_ids)
    voice_idx = min(phoneme_count - 1, voices_data.size(0) - 1)
    ref_tensor = voices_data[voice_idx].squeeze(0).unsqueeze(0).to(device)  # [1, 256]
    if use_half and device != 'cpu':
        ref_tensor = ref_tensor.half()
    speed = torch.tensor([1.0], dtype=torch.float32, device=device)

    print(f"  Example input: {example_phonemes}")
    print(f"  Token IDs: {example_ids}")

    # Trace the model
    with torch.no_grad():
        print("[export] Tracing model (this may take a moment)...")
        start = time.time()

        try:
            traced = torch.jit.trace(wrapper, (ids, ref_tensor, speed))
            trace_time = time.time() - start
            print(f"  Traced in {trace_time:.1f}s")
        except Exception as e:
            print(f"  ERROR tracing: {e}")
            print("[export] Attempting script export instead...")
            traced = torch.jit.script(wrapper)

    # Save model - include precision suffix for FP16
    suffix = f"_{device}_fp16" if use_half else f"_{device}"
    model_path = explicit_model_path if explicit_model_path else output_dir_path / f'kokoro{suffix}.pt'
    traced.save(model_path)
    model_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"  Saved model ({model_size:.1f} MB) to {model_path}")

    # 4. Test inference
    print("[export] Testing inference...")
    test_phonemes = "ˈtɛstɪŋ"
    test_ids = [vocab.get(c, 0) for c in test_phonemes]
    test_tensor = torch.tensor([test_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        start = time.time()
        output = traced(test_tensor, ref_tensor, speed)
        elapsed = (time.time() - start) * 1000

    if isinstance(output, tuple):
        audio = output[0]
    else:
        audio = output

    print(f"  Output shape: {audio.shape}")
    print(f"  Inference time: {elapsed:.0f}ms")
    print(f"  Audio duration: {audio.shape[-1] / 24000:.2f}s")

    # 5. Save test audio
    test_audio_path = output_dir_path / 'test_output.wav'
    save_wav(audio.cpu().numpy().flatten(), test_audio_path)
    print(f"  Saved test audio to {test_audio_path}")

    print("\n[export] Export complete!")
    print(f"  Model: {model_path}")
    print(f"  Vocab: {vocab_path}")
    print(f"  Voice: {voice_path}")

    return model_path, vocab_path, voice_path


def save_wav(audio, path, sample_rate=24000):
    """Save audio as WAV file."""
    import struct

    audio = (audio * 32767).astype('int16')

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


def main():
    parser = argparse.ArgumentParser(description='Export Kokoro to TorchScript')
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu',
                       help='Device for export (default: cpu)')
    parser.add_argument('--output', default='models/kokoro',
                       help='Output directory or full model path (default: models/kokoro)')
    parser.add_argument('--dtype', choices=['float16', 'float32'], default='float32',
                       help='Model dtype to export (default: float32)')
    parser.add_argument('--voice', default='af_heart',
                       help='Voice to export (default: af_heart)')
    parser.add_argument('--half', action='store_true',
                       help='Export in FP16 (half precision) for faster GPU inference')
    parser.add_argument('--disable-complex', action='store_true',
                       help='Use real-valued STFT (only for PyTorch < 2.9.1). Default uses complex STFT for best quality.')
    args = parser.parse_args()

    # Check device availability
    if args.device == 'mps' and not torch.backends.mps.is_available():
        print("Warning: MPS not available, falling back to CPU")
        args.device = 'cpu'
    elif args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = 'cpu'

    dtype = args.dtype
    if args.half:
        # --half is a legacy alias for float16
        dtype = 'float16'

    # FP16 requires GPU
    use_half = dtype == 'float16'
    if use_half and args.device == 'cpu':
        print("Warning: FP16 not supported on CPU, forcing float32")
        use_half = False
        dtype = 'float32'

    # Default: disable_complex=False for best quality (matches Python Kokoro exactly)
    # --disable-complex is no longer needed for MPS in PyTorch 2.9.1+ (torch.angle now works)
    disable_complex = args.disable_complex

    output_dir_path, explicit_model_path = resolve_output_paths(args.output, args.device, use_half)
    export_kokoro(
        args.device,
        str(output_dir_path),
        args.voice,
        use_half=use_half,
        disable_complex=disable_complex,
        explicit_model_path=explicit_model_path,
    )


if __name__ == '__main__':
    main()
