#!/usr/bin/env python3
"""
Translation Worker - Phase 1
Translates English to Japanese using NLLB-200 on Metal GPU.

Interface:
  - Input: English text on stdin (one line per request)
  - Output: Japanese text on stdout (one line per response)
  - Errors: stderr

Usage:
  echo "Hello world" | python3 translation_worker.py
"""
import sys
import time
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class TranslationWorker:
    def __init__(self):
        print("[Translation] Initializing NLLB-200 on Metal GPU...", file=sys.stderr)

        # Use smaller model for initial testing (600M params)
        # Later can upgrade to 3.3B for better quality
        model_name = "facebook/nllb-200-distilled-600M"

        # Check if MPS (Metal Performance Shaders) is available
        if not torch.backends.mps.is_available():
            print("[Translation] WARNING: MPS not available, using CPU", file=sys.stderr)
            self.device = "cpu"
        else:
            self.device = "mps"
            print(f"[Translation] Using Metal GPU (MPS)", file=sys.stderr)

        # Language codes for NLLB
        self.src_lang = "eng_Latn"  # English
        self.tgt_lang = "jpn_Jpan"  # Japanese

        # Load model and tokenizer
        start = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            src_lang=self.src_lang,
            tgt_lang=self.tgt_lang
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            dtype=torch.float32  # MPS works best with float32
        ).to(self.device)

        # Get target language token ID for forced BOS
        self.tgt_lang_id = self.tokenizer.convert_tokens_to_ids(self.tgt_lang)

        elapsed = time.perf_counter() - start
        print(f"[Translation] Model loaded in {elapsed:.2f}s", file=sys.stderr)

        # Set model to eval mode
        self.model.eval()

        # Warm up with a test translation
        print("[Translation] Warming up...", file=sys.stderr)
        self.translate("Hello world")
        print("[Translation] Ready to translate", file=sys.stderr)
        sys.stderr.flush()

    def translate(self, text: str) -> str:
        """Translate English text to Japanese using NLLB-200"""
        if not text.strip():
            return ""

        start = time.perf_counter()

        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)

        # Generate translation with forced target language
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tgt_lang_id,
                max_length=512,
                num_beams=3,  # Reduced from 5 for speed
                early_stopping=True
            )

        # Decode output
        translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        elapsed = (time.perf_counter() - start) * 1000
        print(f"[Translation] {elapsed:.1f}ms | {text[:50]}... -> {translated[:50]}...", file=sys.stderr)
        sys.stderr.flush()

        return translated

    def run(self):
        """Main loop: read from stdin, translate, write to stdout"""
        print("[Translation] Listening for input on stdin...", file=sys.stderr)
        sys.stderr.flush()

        for line in sys.stdin:
            text = line.strip()
            if not text:
                continue

            try:
                translated = self.translate(text)
                print(translated)
                sys.stdout.flush()
            except Exception as e:
                print(f"[Translation] ERROR: {e}", file=sys.stderr)
                sys.stderr.flush()
                # Output empty line on error
                print()
                sys.stdout.flush()

if __name__ == "__main__":
    try:
        worker = TranslationWorker()
        worker.run()
    except KeyboardInterrupt:
        print("\n[Translation] Shutting down...", file=sys.stderr)
    except Exception as e:
        print(f"[Translation] FATAL ERROR: {e}", file=sys.stderr)
        sys.exit(1)
