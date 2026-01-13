#!/usr/bin/env python3
"""
Translation Worker - ONNX Runtime with CoreML (Metal GPU)
Translates English to Japanese using NLLB-200 ONNX model with INT8 quantization.

Target: < 80ms per sentence (down from 154ms PyTorch BFloat16)

Optimizations:
1. INT8 quantization (4x smaller model)
2. ONNX Runtime optimizations
3. CoreML execution provider (Metal GPU)
4. Greedy decoding
5. Minimal memory copies

Interface:
  - Input: English text on stdin (one line per request)
  - Output: Japanese text on stdout (one line per response)
  - Errors: stderr

Usage:
  echo "Hello world" | python3 translation_worker_onnx.py
"""
import sys
import time
from pathlib import Path
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer

class ONNXTranslationWorker:
    def __init__(self, use_int8=True):
        print("[Translation ONNX] Initializing NLLB-200 with CoreML...", file=sys.stderr)

        # Choose model path
        if use_int8:
            model_path = "./onnx_models/nllb-200-600m-int8"
            print("[Translation ONNX] Using INT8 quantized model", file=sys.stderr)
        else:
            model_path = "./onnx_models/nllb-200-600m"
            print("[Translation ONNX] Using FP32 model", file=sys.stderr)

        model_path = Path(model_path)

        if not model_path.exists():
            print(f"[Translation ONNX] ERROR: Model not found at {model_path}", file=sys.stderr)
            print(f"[Translation ONNX] Please run: python export_nllb_to_onnx.py", file=sys.stderr)
            if use_int8:
                print(f"[Translation ONNX] Then run: python quantize_onnx_model.py", file=sys.stderr)
            sys.exit(1)

        # Language codes for NLLB
        self.src_lang = "eng_Latn"  # English
        self.tgt_lang = "jpn_Jpan"  # Japanese

        # Load tokenizer
        print(f"[Translation ONNX] Loading model from {model_path}...", file=sys.stderr)
        start = time.perf_counter()

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))

        # Get target language token ID for forced BOS
        self.tgt_lang_id = self.tokenizer.convert_tokens_to_ids(self.tgt_lang)

        # Load ONNX model with CoreML execution provider
        # CoreML will use Metal GPU on M4 Max
        try:
            self.model = ORTModelForSeq2SeqLM.from_pretrained(
                str(model_path),
                provider="CoreMLExecutionProvider",  # Use Metal GPU
                session_options=None  # Let ONNX Runtime optimize
            )
            print("[Translation ONNX] Using CoreMLExecutionProvider (Metal GPU)", file=sys.stderr)
        except Exception as e:
            print(f"[Translation ONNX] WARNING: CoreML failed ({e}), falling back to CPU", file=sys.stderr)
            self.model = ORTModelForSeq2SeqLM.from_pretrained(
                str(model_path),
                provider="CPUExecutionProvider"
            )

        elapsed = time.perf_counter() - start
        print(f"[Translation ONNX] Model loaded in {elapsed:.2f}s", file=sys.stderr)

        # Warm up with test translations
        print("[Translation ONNX] Warming up...", file=sys.stderr)
        for i in range(3):
            _ = self.translate("Hello world")
            print(f"[Translation ONNX] Warmup {i+1}/3 complete", file=sys.stderr)

        print("[Translation ONNX] Ready to translate", file=sys.stderr)
        sys.stderr.flush()

    def translate(self, text: str) -> str:
        """
        Translate English text to Japanese using ONNX + CoreML

        Optimizations:
        - ONNX Runtime graph optimizations
        - CoreML execution on Metal GPU
        - INT8 quantized weights
        - Greedy decoding (num_beams=1)
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
        )

        # Generate translation with GREEDY DECODING
        outputs = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tgt_lang_id,
            max_new_tokens=128,
            num_beams=1,           # Greedy
            do_sample=False,
            use_cache=True
        )

        # Decode output
        translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        elapsed = (time.perf_counter() - start) * 1000

        # Truncate for logging (UTF-8 safe)
        text_preview = ''.join(list(text)[:40])
        trans_preview = ''.join(list(translated)[:40])

        print(f"[Translation ONNX] {elapsed:.1f}ms | {text_preview}... -> {trans_preview}...", file=sys.stderr)
        sys.stderr.flush()

        return translated

    def run(self):
        """Main loop: read from stdin, translate, write to stdout"""
        print("[Translation ONNX] Listening for input on stdin...", file=sys.stderr)
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
                print(f"[Translation ONNX] ERROR: {e}", file=sys.stderr)
                sys.stderr.flush()
                # Output empty line on error
                print()
                sys.stdout.flush()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32", action="store_true", help="Use FP32 model instead of INT8")
    args = parser.parse_args()

    try:
        worker = ONNXTranslationWorker(use_int8=not args.fp32)
        worker.run()
    except KeyboardInterrupt:
        print("\n[Translation ONNX] Shutting down...", file=sys.stderr)
    except Exception as e:
        print(f"[Translation ONNX] FATAL ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
