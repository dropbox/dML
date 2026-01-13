#!/usr/bin/env python3
"""
Translation Worker - NLLB-200M (Smaller Model)
Testing if smaller model provides acceptable quality with better speed.

Model: facebook/nllb-200-distilled-200M (200M params vs 600M)
Expected: 2-3x faster translation (154ms → 50-80ms)

Interface:
  - Input: English text on stdin (one line per request)
  - Output: Japanese text on stdout (one line per response)
  - Errors: stderr

Usage:
  echo "Hello world" | python3 translation_worker_200m.py
"""
import sys
import time
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class SmallTranslationWorker:
    def __init__(self):
        print("[Translation-200M] Initializing smaller NLLB-200 (200M params)...", file=sys.stderr)

        # Use SMALLER model for speed (200M params instead of 600M)
        model_name = "facebook/nllb-200-distilled-200M"

        # Check if MPS (Metal Performance Shaders) is available
        if not torch.backends.mps.is_available():
            print("[Translation-200M] WARNING: MPS not available, using CPU", file=sys.stderr)
            self.device = "cpu"
        else:
            self.device = "mps"
            print(f"[Translation-200M] Using Metal GPU (MPS)", file=sys.stderr)

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
        print("[Translation-200M] Loading model in bfloat16...", file=sys.stderr)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16  # Native on M4 Max
        ).to(self.device)

        # Get target language token ID for forced BOS
        self.tgt_lang_id = self.tokenizer.convert_tokens_to_ids(self.tgt_lang)

        elapsed = time.perf_counter() - start
        print(f"[Translation-200M] Model loaded in {elapsed:.2f}s", file=sys.stderr)
        print(f"[Translation-200M] Parameters: ~200M (3x smaller than 600M)", file=sys.stderr)

        # Set model to eval mode
        self.model.eval()

        # OPTIMIZATION: torch.compile for M4 Max
        print("[Translation-200M] Compiling model with torch.compile...", file=sys.stderr)
        try:
            self.model = torch.compile(
                self.model,
                mode="max-autotune",  # Maximum optimization
                backend="aot_eager"   # Works well with MPS
            )
            print("[Translation-200M] Model compilation successful", file=sys.stderr)
        except Exception as e:
            print(f"[Translation-200M] WARNING: torch.compile failed: {e}", file=sys.stderr)
            print("[Translation-200M] Continuing without compilation", file=sys.stderr)

        # Warm up with multiple test translations to trigger compilation
        print("[Translation-200M] Warming up (triggers JIT compilation)...", file=sys.stderr)
        for i in range(3):
            _ = self.translate("Hello world")
            print(f"[Translation-200M] Warmup {i+1}/3 complete", file=sys.stderr)

        print("[Translation-200M] Ready to translate", file=sys.stderr)
        sys.stderr.flush()

    def translate(self, text: str) -> str:
        """
        Translate English text to Japanese using smaller NLLB-200M model

        Expected: 50-80ms per sentence (vs 154ms for 600M model)
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

        print(f"[Translation-200M] {elapsed:.1f}ms | {text_preview}... -> {trans_preview}...", file=sys.stderr)
        sys.stderr.flush()

        return translated

    def run(self):
        """Main loop: read from stdin, translate, write to stdout"""
        print("[Translation-200M] Listening for input on stdin...", file=sys.stderr)
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
                print(f"[Translation-200M] ERROR: {e}", file=sys.stderr)
                sys.stderr.flush()
                # Output empty line on error
                print()
                sys.stdout.flush()

if __name__ == "__main__":
    try:
        worker = SmallTranslationWorker()
        worker.run()
    except KeyboardInterrupt:
        print("\n[Translation-200M] Shutting down...", file=sys.stderr)
    except Exception as e:
        print(f"[Translation-200M] FATAL ERROR: {e}", file=sys.stderr)
        sys.exit(1)
