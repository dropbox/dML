#!/usr/bin/env python3
"""
Fish-Speech TTS Wrapper

SOTA multi-language TTS (#1 on TTS-Arena2).

Usage:
    python scripts/fishspeech_tts.py "Text to speak" -o output.wav
    python scripts/fishspeech_tts.py "こんにちは" -o output.wav -l ja

Supported languages: en, ja, zh, ko, fr, de, ar, es

Performance:
    - Model load: ~5-10s (first time)
    - Synthesis: ~200ms per sentence
    - Quality: SOTA (multi-language)

Note: Requires downloading the model first (gated repo - needs HF login):
    1. Request access at: https://huggingface.co/fishaudio/openaudio-s1-mini
    2. Login: huggingface-cli login
    3. Download: huggingface-cli download fishaudio/openaudio-s1-mini \
           --local-dir self_host/fish_speech/checkpoints/openaudio-s1-mini
"""

import argparse
import sys
import time
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Add project root to path for provider import
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from providers import play_audio

# Paths
FISH_SPEECH_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "self_host", "fish_speech")
CHECKPOINT_PATH = os.path.join(FISH_SPEECH_PATH, "checkpoints", "openaudio-s1-mini")


def check_model_exists():
    """Check if the model is downloaded."""
    codec_path = os.path.join(CHECKPOINT_PATH, "codec.pth")
    llama_path = os.path.join(CHECKPOINT_PATH, "model.pth")

    if not os.path.exists(codec_path):
        print(f"Error: Model not found at {codec_path}", file=sys.stderr)
        print("Download with: huggingface-cli download fishaudio/openaudio-s1-mini "
              f"--local-dir {CHECKPOINT_PATH}", file=sys.stderr)
        return False
    return True


def synthesize(text: str, output_path: str, language: str = "en",
               reference_audio: str = None) -> bool:
    """
    Synthesize speech from text using Fish-Speech.

    This is a simplified wrapper that runs the three-stage Fish-Speech pipeline:
    1. (Optional) Encode reference audio to VQ tokens
    2. Generate semantic tokens from text using LLaMA
    3. Decode semantic tokens to audio using DAC

    Args:
        text: Text to synthesize
        output_path: Path to save WAV file
        language: Language code (en, ja, zh, ko, fr, de, ar, es)
        reference_audio: Optional reference audio for voice cloning

    Returns:
        True if successful
    """
    import subprocess
    import tempfile

    if not check_model_exists():
        return False

    try:
        print(f"Synthesizing: {text[:60]}...", file=sys.stderr)
        start = time.time()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Stage 2: Generate semantic tokens from text
            # Using the fish_speech module's text2semantic inference
            codes_path = os.path.join(tmpdir, "codes_0.npy")

            # Run text2semantic inference
            cmd = [
                sys.executable, "-m", "fish_speech.models.text2semantic.inference",
                "--text", text,
                "--checkpoint-path", os.path.join(CHECKPOINT_PATH, "model.pth"),
                "--output-path", tmpdir,
                "--num-samples", "1",
            ]

            if reference_audio:
                # First encode reference
                ref_tokens = os.path.join(tmpdir, "ref.npy")
                ref_cmd = [
                    sys.executable, "-m", "fish_speech.models.dac.inference",
                    "-i", reference_audio,
                    "--checkpoint-path", os.path.join(CHECKPOINT_PATH, "codec.pth"),
                    "-o", ref_tokens,
                ]
                result = subprocess.run(ref_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"Reference encoding failed: {result.stderr}", file=sys.stderr)
                    return False
                cmd.extend(["--prompt-tokens", ref_tokens])

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Text2Semantic failed: {result.stderr}", file=sys.stderr)
                return False

            # Stage 3: Decode semantic tokens to audio
            decode_cmd = [
                sys.executable, "-m", "fish_speech.models.dac.inference",
                "-i", codes_path,
                "--checkpoint-path", os.path.join(CHECKPOINT_PATH, "codec.pth"),
                "-o", output_path,
            ]

            result = subprocess.run(decode_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"DAC decoding failed: {result.stderr}", file=sys.stderr)
                return False

        elapsed = time.time() - start
        print(f"Synthesized in {elapsed:.2f}s", file=sys.stderr)
        print(f"Saved to: {output_path}", file=sys.stderr)

        return True

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Fish-Speech TTS - SOTA multi-language synthesis")
    parser.add_argument("text", nargs='?', help="Text to synthesize")
    parser.add_argument("-o", "--output", required=True, help="Output WAV file path")
    parser.add_argument("-l", "--language", default="en", help="Language code")
    parser.add_argument("--reference", help="Reference audio for voice cloning")
    parser.add_argument("--play", action="store_true", help="Play audio after synthesis")

    args = parser.parse_args()

    if args.text:
        text = args.text
    else:
        text = sys.stdin.read().strip()

    if not text:
        print("Error: No text provided", file=sys.stderr)
        sys.exit(1)

    success = synthesize(
        text,
        args.output,
        language=args.language,
        reference_audio=args.reference
    )

    if success and args.play:
        with open(args.output, 'rb') as f:
            play_audio(f.read())

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
