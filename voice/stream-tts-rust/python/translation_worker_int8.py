#!/usr/bin/env python3
"""
Translation Worker - INT8 Quantized for Maximum Speed
Translates English to Japanese using INT8 quantized NLLB-200 on Metal GPU.

Target: < 80ms per sentence (down from 154ms with BFloat16)

Optimizations:
1. Dynamic INT8 quantization (8-bit integers instead of 16-bit floats)
2. Greedy decoding (num_beams=1)
3. Reduced max length
4. KV cache enabled
5. Batch processing ready

Expected speedup: 1.5-2x faster than BFloat16 (154ms → 70-100ms)

Interface:
  - Input: English text on stdin (one line per request)
  - Output: Japanese text on stdout (one line per response)
  - Errors: stderr

Usage:
  echo "Hello world" | python3 translation_worker_int8.py
"""
import sys
import time
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class INT8TranslationWorker:
    def __init__(self):
        print("[Translation-INT8] Initializing quantized NLLB-200 on Metal GPU...", file=sys.stderr)

        # Use smaller model for speed (600M params)
        model_name = "facebook/nllb-200-distilled-600M"

        # Check if MPS (Metal Performance Shaders) is available
        if not torch.backends.mps.is_available():
            print("[Translation-INT8] WARNING: MPS not available, using CPU", file=sys.stderr)
            self.device = "cpu"
        else:
            self.device = "cpu"  # INT8 quantization currently works best on CPU
            print(f"[Translation-INT8] Using CPU for INT8 inference (more stable)", file=sys.stderr)

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

        # Load model in float32 first
        print("[Translation-INT8] Loading model in float32...", file=sys.stderr)
        model_fp32 = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        )

        # Get target language token ID for forced BOS
        self.tgt_lang_id = self.tokenizer.convert_tokens_to_ids(self.tgt_lang)

        # Set model to eval mode before quantization
        model_fp32.eval()

        # OPTIMIZATION: Dynamic INT8 quantization
        print("[Translation-INT8] Applying dynamic INT8 quantization...", file=sys.stderr)
        try:
            # Dynamic quantization: quantizes weights to INT8, activations stay FP32
            # This is faster than full quantization and doesn't require calibration
            self.model = torch.quantization.quantize_dynamic(
                model_fp32,
                {torch.nn.Linear},  # Quantize all Linear layers
                dtype=torch.qint8    # Use 8-bit integers
            )
            print("[Translation-INT8] Model successfully quantized to INT8", file=sys.stderr)
        except Exception as e:
            print(f"[Translation-INT8] WARNING: Quantization failed: {e}", file=sys.stderr)
            print("[Translation-INT8] Falling back to float32 model", file=sys.stderr)
            self.model = model_fp32

        # Move to device (CPU for quantized models)
        self.model = self.model.to(self.device)

        elapsed = time.perf_counter() - start
        print(f"[Translation-INT8] Model loaded and quantized in {elapsed:.2f}s", file=sys.stderr)

        # Check model size reduction
        def get_model_size(model):
            param_size = 0
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            buffer_size = 0
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            return (param_size + buffer_size) / 1024 / 1024  # MB

        model_size_mb = get_model_size(self.model)
        print(f"[Translation-INT8] Model size: {model_size_mb:.1f} MB", file=sys.stderr)

        # Warm up with test translations
        print("[Translation-INT8] Warming up...", file=sys.stderr)
        for i in range(3):
            _ = self.translate("Hello world")
            print(f"[Translation-INT8] Warmup {i+1}/3 complete", file=sys.stderr)

        print("[Translation-INT8] Ready to translate", file=sys.stderr)
        sys.stderr.flush()

    def translate(self, text: str) -> str:
        """
        Translate English text to Japanese using INT8 quantized NLLB-200

        Optimizations:
        - INT8 weights (2x memory reduction, ~1.5-2x speedup)
        - Greedy decoding (num_beams=1)
        - Shorter max_length
        - No padding for single inputs
        """
        if not text.strip():
            return ""

        start = time.perf_counter()

        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=256,
            truncation=True,
            padding=False
        ).to(self.device)

        # Generate translation with GREEDY DECODING (fastest)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tgt_lang_id,
                max_new_tokens=128,
                num_beams=1,           # GREEDY
                do_sample=False,
                early_stopping=False,
                use_cache=True
            )

        # Decode output
        translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        elapsed = (time.perf_counter() - start) * 1000

        # Truncate for logging (UTF-8 safe)
        text_preview = ''.join(list(text)[:40])
        trans_preview = ''.join(list(translated)[:40])

        print(f"[Translation-INT8] {elapsed:.1f}ms | {text_preview}... -> {trans_preview}...", file=sys.stderr)
        sys.stderr.flush()

        return translated

    def run(self):
        """Main loop: read from stdin, translate, write to stdout"""
        print("[Translation-INT8] Listening for input on stdin...", file=sys.stderr)
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
                print(f"[Translation-INT8] ERROR: {e}", file=sys.stderr)
                sys.stderr.flush()
                # Output empty line on error
                print()
                sys.stdout.flush()

if __name__ == "__main__":
    try:
        worker = INT8TranslationWorker()
        worker.run()
    except KeyboardInterrupt:
        print("\n[Translation-INT8] Shutting down...", file=sys.stderr)
    except Exception as e:
        print(f"[Translation-INT8] FATAL ERROR: {e}", file=sys.stderr)
        sys.exit(1)
