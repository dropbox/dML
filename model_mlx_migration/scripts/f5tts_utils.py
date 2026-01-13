#!/usr/bin/env python3
# Copyright 2024-2025 Andrew Yates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
F5-TTS Utility Functions

DEPRECATED: F5-TTS is deprecated in favor of CosyVoice2.
CosyVoice2 is 18x faster (35x vs 2x RTF) with equal/better quality.
This module is kept for historical compatibility only.

Provides convenience wrappers for F5-TTS with automatic audio resampling.
F5-TTS requires 24kHz reference audio. This module handles automatic
conversion of audio files at other sample rates.

Usage:
    from scripts.f5tts_utils import generate_with_auto_resample, ensure_24khz

    # Auto-resample and generate
    generate_with_auto_resample(
        generation_text="Hello world",
        ref_audio_path="my_voice.wav",  # Any sample rate
        ref_audio_text="Reference text",
        output_path="output.wav"
    )

    # Or just resample audio for manual use
    resampled_path = ensure_24khz("my_voice.wav")
"""

import tempfile
from pathlib import Path
from typing import Optional

import soundfile as sf

# Target sample rate for F5-TTS
F5TTS_SAMPLE_RATE = 24000


def ensure_24khz(
    audio_path: str, output_path: Optional[str] = None, verbose: bool = True
) -> str:
    """
    Ensure audio file is at 24kHz sample rate.

    If the input audio is already 24kHz, returns the original path.
    Otherwise, resamples to 24kHz and saves to output_path (or temp file).

    Args:
        audio_path: Path to input audio file
        output_path: Optional path for resampled output (uses temp file if None)
        verbose: Print status messages

    Returns:
        Path to 24kHz audio file (original or resampled)
    """
    audio, sr = sf.read(audio_path)

    if sr == F5TTS_SAMPLE_RATE:
        if verbose:
            print(f"Audio already at {F5TTS_SAMPLE_RATE}Hz: {audio_path}")
        return audio_path

    # Need to resample
    if verbose:
        print(f"Resampling {audio_path} from {sr}Hz to {F5TTS_SAMPLE_RATE}Hz")

    # Use librosa for high-quality resampling
    try:
        import librosa

        resampled = librosa.resample(audio, orig_sr=sr, target_sr=F5TTS_SAMPLE_RATE)
    except ImportError:
        # Fall back to scipy if librosa not available
        from scipy import signal

        num_samples = int(len(audio) * F5TTS_SAMPLE_RATE / sr)
        resampled = signal.resample(audio, num_samples)

    # Determine output path
    if output_path is None:
        # Create temp file with same name pattern
        suffix = Path(audio_path).suffix or ".wav"
        fd, output_path = tempfile.mkstemp(suffix=suffix)
        import os

        os.close(fd)

    # Save resampled audio
    sf.write(output_path, resampled, F5TTS_SAMPLE_RATE)

    if verbose:
        duration_orig = len(audio) / sr
        duration_new = len(resampled) / F5TTS_SAMPLE_RATE
        print(f"Resampled: {duration_orig:.2f}s -> {duration_new:.2f}s ({output_path})")

    return output_path


def generate_with_auto_resample(
    generation_text: str,
    ref_audio_path: str,
    ref_audio_text: str,
    output_path: str,
    steps: int = 4,
    verbose: bool = True,
    **kwargs,
) -> str:
    """
    Generate speech with F5-TTS, auto-resampling reference audio if needed.

    This is a convenience wrapper around f5_tts_mlx.generate.generate that
    handles automatic resampling of reference audio to 24kHz.

    Args:
        generation_text: Text to generate as speech
        ref_audio_path: Path to reference audio (any sample rate)
        ref_audio_text: Transcript of reference audio
        output_path: Path for generated output audio
        steps: Number of diffusion steps (default 4, optimal for speed/quality)
        verbose: Print status messages
        **kwargs: Additional arguments passed to f5_tts_mlx.generate.generate

    Returns:
        Path to generated audio file
    """
    try:
        from f5_tts_mlx.generate import generate
    except ImportError:
        raise ImportError(
            "f5_tts_mlx package required. Install with: pip install f5-tts-mlx"
        )

    # Ensure reference audio is at 24kHz
    ref_24khz = ensure_24khz(ref_audio_path, verbose=verbose)

    # Track if we created a temp file for cleanup
    cleanup_temp = ref_24khz != ref_audio_path

    try:
        # Generate with F5-TTS
        generate(
            generation_text=generation_text,
            ref_audio_path=ref_24khz,
            ref_audio_text=ref_audio_text,
            output_path=output_path,
            steps=steps,
            **kwargs,
        )
    finally:
        # Clean up temp file if we created one
        if cleanup_temp:
            try:
                Path(ref_24khz).unlink()
            except (OSError, FileNotFoundError):
                pass

    return output_path


def generate_batch(
    texts: list[str],
    ref_audio_path: str,
    ref_audio_text: str,
    output_dir: str,
    steps: int = 4,
    verbose: bool = True,
    **kwargs,
) -> list[str]:
    """
    Generate multiple speech utterances with same reference voice.

    Batch synthesis with F5-TTS. The model loads once and generates
    multiple outputs efficiently. Due to flow-matching architecture,
    each generation is sequential (~1-2s per utterance at 4 steps).

    Args:
        texts: List of texts to generate as speech
        ref_audio_path: Path to reference audio (any sample rate)
        ref_audio_text: Transcript of reference audio
        output_dir: Directory for generated output audio files
        steps: Number of diffusion steps (default 4, optimal for speed/quality)
        verbose: Print status messages
        **kwargs: Additional arguments passed to f5_tts_mlx.generate.generate

    Returns:
        List of paths to generated audio files

    Example:
        >>> from scripts.f5tts_utils import generate_batch
        >>> texts = ["Hello world", "How are you?", "Nice to meet you"]
        >>> outputs = generate_batch(
        ...     texts=texts,
        ...     ref_audio_path="my_voice.wav",
        ...     ref_audio_text="Reference text",
        ...     output_dir="./outputs"
        ... )
    """
    if not texts:
        return []

    try:
        from f5_tts_mlx.generate import generate
    except ImportError:
        raise ImportError(
            "f5_tts_mlx package required. Install with: pip install f5-tts-mlx"
        )

    # Ensure output directory exists
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Ensure reference audio is at 24kHz
    ref_24khz = ensure_24khz(ref_audio_path, verbose=verbose)
    cleanup_temp = ref_24khz != ref_audio_path

    output_paths = []
    try:
        for i, text in enumerate(texts):
            output_path = str(output_dir_path / f"output_{i:03d}.wav")

            if verbose:
                print(f"Generating {i+1}/{len(texts)}: {text[:50]}...")

            generate(
                generation_text=text,
                ref_audio_path=ref_24khz,
                ref_audio_text=ref_audio_text,
                output_path=output_path,
                steps=steps,
                **kwargs,
            )
            output_paths.append(output_path)

    finally:
        # Clean up temp file if we created one
        if cleanup_temp:
            try:
                Path(ref_24khz).unlink()
            except (OSError, FileNotFoundError):
                pass

    return output_paths


def get_audio_info(audio_path: str) -> dict:
    """
    Get information about an audio file.

    Args:
        audio_path: Path to audio file

    Returns:
        Dictionary with sample_rate, duration, channels, and needs_resample
    """
    info = sf.info(audio_path)
    return {
        "sample_rate": info.samplerate,
        "duration": info.duration,
        "channels": info.channels,
        "frames": info.frames,
        "format": info.format,
        "needs_resample": info.samplerate != F5TTS_SAMPLE_RATE,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python f5tts_utils.py <audio_file>")
        print("\nChecks if audio file needs resampling for F5-TTS")
        sys.exit(1)

    audio_path = sys.argv[1]
    if not Path(audio_path).exists():
        print(f"Error: File not found: {audio_path}")
        sys.exit(1)

    info = get_audio_info(audio_path)
    print(f"Audio file: {audio_path}")
    print(f"  Sample rate: {info['sample_rate']}Hz")
    print(f"  Duration: {info['duration']:.2f}s")
    print(f"  Channels: {info['channels']}")
    print(f"  Format: {info['format']}")
    print(f"  Needs resample for F5-TTS: {info['needs_resample']}")

    if info["needs_resample"]:
        print(f"\nTo resample, use: ensure_24khz('{audio_path}')")
