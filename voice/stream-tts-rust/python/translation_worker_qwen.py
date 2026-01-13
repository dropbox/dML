#!/usr/bin/env python3
"""
Translation Worker using Qwen2.5-7B via llama.cpp
Provides high-quality English to Japanese translation.
"""

import sys
import time
import subprocess
import os

# Configuration
LLAMA_CPP_PATH = "/Users/ayates/voice/llama.cpp/build/bin/llama-cli"
MODEL_PATH = "/Users/ayates/voice/models/qwen/Qwen2.5-7B-Instruct-GGUF/qwen2.5-7b-instruct-q3_k_m.gguf"

# System prompt for translation
SYSTEM_PROMPT = """You are a professional translator. Translate the following English text to natural Japanese.
Preserve the meaning, tone, and context. Only output the Japanese translation, nothing else."""


def translate_text(text: str) -> str:
    """
    Translate English text to Japanese using Qwen2.5-7B.

    Args:
        text: English text to translate

    Returns:
        Japanese translation
    """
    if not text or not text.strip():
        return ""

    # Create prompt for Qwen
    prompt = f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{text.strip()}<|im_end|>
<|im_start|>assistant
"""

    # Call llama.cpp for inference
    try:
        result = subprocess.run(
            [
                LLAMA_CPP_PATH,
                "-m", MODEL_PATH,
                "-p", prompt,
                "-n", "256",  # Max tokens
                "--temp", "0.3",  # Low temperature for consistent translation
                "--top-p", "0.9",
                "-ngl", "99",  # GPU layers (Metal)
                "--no-display-prompt",
                "--log-disable",
            ],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            print(f"[Translation Worker] Error: {result.stderr}", file=sys.stderr, flush=True)
            return text  # Fall back to original text

        # Extract translation from output
        output = result.stdout.strip()

        # Remove any remaining prompt artifacts
        if "<|im_end|>" in output:
            output = output.split("<|im_end|>")[0]

        return output.strip()

    except subprocess.TimeoutExpired:
        print(f"[Translation Worker] Timeout translating text", file=sys.stderr, flush=True)
        return text
    except Exception as e:
        print(f"[Translation Worker] Exception: {e}", file=sys.stderr, flush=True)
        return text


def main():
    """Main worker loop - read from stdin, translate, write to stdout."""

    print("[Translation Worker] Qwen2.5-7B translation worker started", file=sys.stderr, flush=True)
    print(f"[Translation Worker] Model: {MODEL_PATH}", file=sys.stderr, flush=True)
    print(f"[Translation Worker] Using Metal GPU acceleration", file=sys.stderr, flush=True)

    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"[Translation Worker] ERROR: Model not found at {MODEL_PATH}", file=sys.stderr, flush=True)
        sys.exit(1)

    if not os.path.exists(LLAMA_CPP_PATH):
        print(f"[Translation Worker] ERROR: llama-cli not found at {LLAMA_CPP_PATH}", file=sys.stderr, flush=True)
        sys.exit(1)

    print("[Translation Worker] Ready to translate", file=sys.stderr, flush=True)

    # Process lines from stdin
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        start_time = time.time()

        # Translate
        translation = translate_text(line)

        elapsed_ms = (time.time() - start_time) * 1000

        # Log performance
        preview_en = line[:50] + "..." if len(line) > 50 else line
        preview_ja = translation[:50] + "..." if len(translation) > 50 else translation
        print(f"[Translation] {elapsed_ms:.1f}ms | {preview_en} -> {preview_ja}",
              file=sys.stderr, flush=True)

        # Send translation to stdout
        print(translation, flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[Translation Worker] Stopped", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[Translation Worker] Fatal error: {e}", file=sys.stderr, flush=True)
        sys.exit(1)
