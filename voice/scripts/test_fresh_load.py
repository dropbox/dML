#!/usr/bin/env python3
"""Test with fresh model load and GPU cache clear."""

import os
import sys
import gc
import base64
from pathlib import Path

# Clear any cached state
gc.collect()

COSYVOICE_REPO = os.path.join(os.path.dirname(__file__), '..', 'cosyvoice_repo')
sys.path.insert(0, COSYVOICE_REPO)
sys.path.insert(0, os.path.join(COSYVOICE_REPO, 'third_party', 'Matcha-TTS'))

MODEL_DIR = Path(__file__).parent.parent / "models" / "cosyvoice" / "CosyVoice2-0.5B"
OUTPUT_DIR = Path(__file__).parent.parent / "models" / "cosyvoice" / "test_output"
VOICE_SAMPLE = Path(__file__).parent.parent / "tests" / "golden" / "hello.wav"

TEXT = "你好，今天天气真好啊！"
INSTRUCTION = "用四川话说"

from dotenv import load_dotenv
load_dotenv()

def main():
    import torch
    import torchaudio

    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()

    print(f"PyTorch: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")

    from cosyvoice.cli.cosyvoice import CosyVoice2
    from cosyvoice.utils.file_utils import load_wav

    print("\nLoading fresh model...")
    cosyvoice = CosyVoice2(str(MODEL_DIR), load_jit=False, load_trt=False, fp16=False)
    prompt = load_wav(str(VOICE_SAMPLE), 16000)

    print("Generating...")
    output_path = OUTPUT_DIR / "fresh_load_test.wav"

    all_speech = []
    for result in cosyvoice.inference_instruct2(TEXT, INSTRUCTION, prompt, stream=False):
        all_speech.append(result['tts_speech'])

    full_speech = torch.cat(all_speech, dim=1)
    torchaudio.save(str(output_path), full_speech, cosyvoice.sample_rate)

    duration = full_speech.shape[1] / cosyvoice.sample_rate
    print(f"Generated: {duration:.1f}s -> {output_path}")

    # Evaluate
    import openai
    client = openai.OpenAI()
    with open(output_path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")

    resp = client.chat.completions.create(
        model="gpt-audio-2025-08-28",
        modalities=["text"],
        messages=[
            {"role": "system", "content": "Output only JSON."},
            {"role": "user", "content": [
                {"type": "text", "text": 'Rate 1-10. Frog sounds? {"score":X,"frog":bool}'},
                {"type": "input_audio", "input_audio": {"data": data, "format": "wav"}}
            ]}
        ]
    )
    print(f"LLM Judge: {resp.choices[0].message.content}")


if __name__ == "__main__":
    main()
