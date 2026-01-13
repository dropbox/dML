#!/usr/bin/env python3
"""
Translation Worker - Optimized for Speed
Translates English to Japanese using NLLB-200 on Metal GPU with aggressive optimizations.

Target: < 70ms per sentence (down from 282ms)

Optimizations:
1. Greedy decoding (num_beams=1) instead of beam search
2. BFloat16 precision on Metal
3. torch.compile with max-autotune
4. Reduced max length for faster generation
5. Batch processing ready

Interface:
  - Input: English text on stdin (one line per request)
  - Output: Japanese text on stdout (one line per response)
  - Errors: stderr

Usage:
  echo "Hello world" | python3 translation_worker_optimized.py
"""
import sys
import time
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class OptimizedTranslationWorker:
    def __init__(self):
        print("[Translation] Initializing optimized NLLB-200 on Metal GPU...", file=sys.stderr)

        # Use smaller model for speed (600M params)
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

        # Load with bfloat16 for M4 Max (native support)
        print("[Translation] Loading model in bfloat16...", file=sys.stderr)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16  # Native on M4 Max
        ).to(self.device)

        # Get target language token ID for forced BOS
        self.tgt_lang_id = self.tokenizer.convert_tokens_to_ids(self.tgt_lang)

        elapsed = time.perf_counter() - start
        print(f"[Translation] Model loaded in {elapsed:.2f}s", file=sys.stderr)

        # Set model to eval mode
        self.model.eval()

        # OPTIMIZATION: torch.compile for M4 Max
        # This compiles the model into optimized Metal kernels
        print("[Translation] Compiling model with torch.compile...", file=sys.stderr)
        try:
            self.model = torch.compile(
                self.model,
                mode="max-autotune",  # Maximum optimization
                backend="aot_eager"   # Works well with MPS
            )
            print("[Translation] Model compilation successful", file=sys.stderr)
        except Exception as e:
            print(f"[Translation] WARNING: torch.compile failed: {e}", file=sys.stderr)
            print("[Translation] Continuing without compilation", file=sys.stderr)

        # Warm up with multiple test translations to trigger compilation
        print("[Translation] Warming up (this will trigger JIT compilation)...", file=sys.stderr)
        for i in range(3):
            _ = self.translate("Hello world")
            print(f"[Translation] Warmup {i+1}/3 complete", file=sys.stderr)

        print("[Translation] Ready to translate", file=sys.stderr)
        sys.stderr.flush()

    def translate(self, text: str) -> str:
        """
        Translate English text to Japanese using optimized NLLB-200

        Optimizations:
        - Greedy decoding (num_beams=1)
        - Shorter max_length (256 instead of 512)
        - No padding for single inputs
        """
        if not text.strip():
            return ""

        start = time.perf_counter()

        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=256,  # Reduced from 512 for speed
            truncation=True,
            padding=False    # No padding for single inputs
        ).to(self.device)

        # Generate translation with GREEDY DECODING (fastest)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tgt_lang_id,
                max_new_tokens=128,    # Limit output length
                num_beams=1,           # GREEDY: 1 beam instead of 3-5
                do_sample=False,       # Deterministic
                early_stopping=False,  # Not needed with greedy
                use_cache=True         # Enable KV cache
            )

        # Decode output
        translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        elapsed = (time.perf_counter() - start) * 1000

        # Truncate for logging (UTF-8 safe)
        text_preview = ''.join(list(text)[:40])
        trans_preview = ''.join(list(translated)[:40])

        print(f"[Translation] {elapsed:.1f}ms | {text_preview}... -> {trans_preview}...", file=sys.stderr)
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
        worker = OptimizedTranslationWorker()
        worker.run()
    except KeyboardInterrupt:
        print("\n[Translation] Shutting down...", file=sys.stderr)
    except Exception as e:
        print(f"[Translation] FATAL ERROR: {e}", file=sys.stderr)
        sys.exit(1)
