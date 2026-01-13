#!/usr/bin/env python3
"""
Translation worker using Qwen2.5-72B-Instruct via llama.cpp
Provides near GPT-4o quality translation (BLEU 35+)
Expected latency: 100-200ms on M4 Max with Metal GPU
"""

import sys
import subprocess
import time
import os
from pathlib import Path

class QwenTranslationWorker:
    def __init__(self):
        """Initialize Qwen translation worker with llama.cpp"""
        # Find llama.cpp binary (try both old and new CMake locations)
        project_root = Path(__file__).parent.parent.parent
        llama_cli_cmake = project_root / "models" / "qwen" / "llama.cpp" / "build" / "bin" / "llama-cli"
        llama_cli_symlink = project_root / "models" / "qwen" / "llama.cpp" / "llama-cli"

        if llama_cli_cmake.exists():
            self.llama_cli = llama_cli_cmake
        elif llama_cli_symlink.exists():
            self.llama_cli = llama_cli_symlink
        else:
            self.llama_cli = llama_cli_cmake  # Will error below with helpful message

        self.model_path = project_root / "models" / "qwen" / "Qwen2.5-7B-Instruct-GGUF" / "qwen2.5-7b-instruct-q3_k_m.gguf"

        # Verify paths exist
        if not self.llama_cli.exists():
            raise FileNotFoundError(
                f"llama-cli not found at {self.llama_cli}\n"
                f"Please run: ./setup_qwen.sh"
            )

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Qwen model not found at {self.model_path}\n"
                f"Please run: ./setup_qwen.sh"
            )

        # Get optimal thread count
        try:
            self.threads = int(subprocess.check_output(['sysctl', '-n', 'hw.ncpu']).decode().strip())
        except:
            self.threads = 8  # fallback

        self.log(f"Initialized Qwen worker")
        self.log(f"llama-cli: {self.llama_cli}")
        self.log(f"Model: {self.model_path}")
        self.log(f"Threads: {self.threads}")

    def log(self, message):
        """Log to stderr"""
        print(f"[Qwen Worker] {message}", file=sys.stderr, flush=True)

    def translate(self, english_text: str) -> str:
        """
        Translate English to Japanese using Qwen2.5-72B

        Args:
            english_text: English text to translate

        Returns:
            Japanese translation
        """
        # Build prompt using Qwen's instruction format
        prompt = f"""Translate the following English text to natural Japanese. Output ONLY the Japanese translation, nothing else.

English: {english_text}
Japanese:"""

        # Run llama.cpp with optimized parameters
        cmd = [
            str(self.llama_cli),
            '--model', str(self.model_path),
            '--prompt', prompt,
            '--n-gpu-layers', '999',  # Use all GPU layers
            '--temp', '0.3',          # Low temperature for consistent translation
            '--top-p', '0.9',          # Nucleus sampling
            '--threads', str(self.threads),
            '--ctx-size', '512',       # Small context (we only need translation)
            '--n-predict', '128',      # Max tokens for translation
            '--repeat-penalty', '1.1',
            '--no-display-prompt',     # Don't echo the prompt
            '--log-disable',           # Disable verbose logging
        ]

        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10.0  # 10 second timeout
            )

            if result.returncode != 0:
                self.log(f"Error: llama-cli returned {result.returncode}")
                self.log(f"stderr: {result.stderr}")
                return english_text  # Fallback to original

            # Extract translation from output
            translation = result.stdout.strip()

            # Remove any potential prompt echoes or extra text
            # llama.cpp should only output the translation with --no-display-prompt
            if '\n' in translation:
                # Take last non-empty line
                lines = [line.strip() for line in translation.split('\n') if line.strip()]
                translation = lines[-1] if lines else translation

            latency = (time.time() - start_time) * 1000
            self.log(f"Translated in {latency:.0f}ms: {english_text[:50]}... -> {translation[:50]}...")

            return translation

        except subprocess.TimeoutExpired:
            self.log(f"Translation timeout after 10s")
            return english_text  # Fallback
        except Exception as e:
            self.log(f"Translation error: {e}")
            return english_text  # Fallback

    def run(self):
        """Main loop: read from stdin, translate, write to stdout"""
        self.log("Worker ready, waiting for input...")

        for line in sys.stdin:
            text = line.strip()
            if not text:
                continue

            # Translate
            translation = self.translate(text)

            # Output to stdout
            print(translation, flush=True)


def main():
    try:
        worker = QwenTranslationWorker()
        worker.run()
    except KeyboardInterrupt:
        print("\n[Qwen Worker] Shutting down...", file=sys.stderr)
    except Exception as e:
        print(f"\n[Qwen Worker] Fatal error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
