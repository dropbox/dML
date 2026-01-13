#!/usr/bin/env python3
"""Compare CosyVoice2 quality between different torch versions.

Run this script twice:
1. source cosyvoice_venv/bin/activate && python scripts/compare_torch_versions.py
2. source cosyvoice_fresh_venv/bin/activate && python scripts/compare_torch_versions.py

Then run the LLM judge comparison.
"""

import os
import sys
import time
import torch
import torchaudio
from pathlib import Path

# Add CosyVoice repo to path
COSYVOICE_REPO = os.path.join(os.path.dirname(__file__), '..', 'cosyvoice_repo')
sys.path.insert(0, COSYVOICE_REPO)
sys.path.insert(0, os.path.join(COSYVOICE_REPO, 'third_party', 'Matcha-TTS'))

MODEL_DIR = Path(__file__).parent.parent / "models" / "cosyvoice" / "CosyVoice2-0.5B"
OUTPUT_DIR = Path(__file__).parent.parent / "models" / "cosyvoice" / "test_output"
VOICE_SAMPLE = Path(__file__).parent.parent / "tests" / "golden" / "hello.wav"

# Simple test - same as golden reference
TEXT = "你好，我是四川婆婆。"
INSTRUCTION = "用四川话说"


def main():
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")

    from cosyvoice.cli.cosyvoice import CosyVoice2
    from cosyvoice.utils.file_utils import load_wav

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading model...")
    cosyvoice = CosyVoice2(str(MODEL_DIR), load_jit=False, load_trt=False, fp16=False)
    prompt = load_wav(str(VOICE_SAMPLE), 16000)

    # Version tag for output file
    version_tag = torch.__version__.replace(".", "_")
    output_path = OUTPUT_DIR / f"compare_torch_{version_tag}.wav"

    print(f"\nGenerating with torch {torch.__version__}...")
    print(f"Text: {TEXT}")
    print(f"Instruction: {INSTRUCTION}")

    start = time.time()
    all_speech = []
    for result in cosyvoice.inference_instruct2(TEXT, INSTRUCTION, prompt, stream=False):
        all_speech.append(result['tts_speech'])

    full_speech = torch.cat(all_speech, dim=1)
    torchaudio.save(str(output_path), full_speech.cpu(), cosyvoice.sample_rate)

    duration = full_speech.shape[1] / cosyvoice.sample_rate
    gen_time = time.time() - start

    print(f"Generated: {duration:.1f}s in {gen_time:.1f}s (RTF: {gen_time/duration:.2f}x)")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
