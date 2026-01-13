#!/usr/bin/env python3
"""Test CosyVoice2 with system Python."""

import os
import sys
import time
import base64
from pathlib import Path

# Add CosyVoice repo to path
COSYVOICE_REPO = os.path.join(os.path.dirname(__file__), '..', 'cosyvoice_repo')
sys.path.insert(0, COSYVOICE_REPO)
sys.path.insert(0, os.path.join(COSYVOICE_REPO, 'third_party', 'Matcha-TTS'))

MODEL_DIR = Path(__file__).parent.parent / "models" / "cosyvoice" / "CosyVoice2-0.5B"
OUTPUT_DIR = Path(__file__).parent.parent / "models" / "cosyvoice" / "test_output"
VOICE_SAMPLE = Path(__file__).parent.parent / "tests" / "golden" / "hello.wav"

TEXT = "你好，我是四川婆婆。"
INSTRUCTION = "用四川话说"

def main():
    import torch
    import torchaudio

    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")

    from cosyvoice.cli.cosyvoice import CosyVoice2
    from cosyvoice.utils.file_utils import load_wav

    print("\nLoading model...")
    cosyvoice = CosyVoice2(str(MODEL_DIR), load_jit=False, load_trt=False, fp16=False)
    prompt = load_wav(str(VOICE_SAMPLE), 16000)

    print(f"Device: {cosyvoice.model.device}")

    output_path = OUTPUT_DIR / "system_python_test.wav"

    print("Generating...")
    start = time.time()
    all_speech = []
    for result in cosyvoice.inference_instruct2(TEXT, INSTRUCTION, prompt, stream=False):
        all_speech.append(result['tts_speech'])

    full_speech = torch.cat(all_speech, dim=1)
    torchaudio.save(str(output_path), full_speech, cosyvoice.sample_rate)

    duration = full_speech.shape[1] / cosyvoice.sample_rate
    gen_time = time.time() - start
    print(f"Generated: {duration:.1f}s in {gen_time:.1f}s")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
