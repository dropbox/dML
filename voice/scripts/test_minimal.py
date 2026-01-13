#!/usr/bin/env python3
"""Minimal CosyVoice2 test - compare with OLD good file."""

import os
import sys
import time
import base64
from pathlib import Path

COSYVOICE_REPO = os.path.join(os.path.dirname(__file__), '..', 'cosyvoice_repo')
sys.path.insert(0, COSYVOICE_REPO)
sys.path.insert(0, os.path.join(COSYVOICE_REPO, 'third_party', 'Matcha-TTS'))

MODEL_DIR = Path(__file__).parent.parent / "models" / "cosyvoice" / "CosyVoice2-0.5B"
OUTPUT_DIR = Path(__file__).parent.parent / "models" / "cosyvoice" / "test_output"
VOICE_SAMPLE = Path(__file__).parent.parent / "tests" / "golden" / "hello.wav"

# Simple test text (short)
TEXT = "你好，我是一个四川婆婆。"
INSTRUCTION = "用四川话说，像一个老太婆在唠叨"

from dotenv import load_dotenv
load_dotenv()


def eval_audio(path: Path) -> str:
    import openai
    client = openai.OpenAI()
    with open(path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    resp = client.chat.completions.create(
        model="gpt-audio-2025-08-28",
        modalities=["text"],
        messages=[
            {"role": "system", "content": "Output only JSON."},
            {"role": "user", "content": [
                {"type": "text", "text": 'Rate audio 1-10. Any frog/croaking/distortion? {"score":X,"frog":bool,"notes":"brief"}'},
                {"type": "input_audio", "input_audio": {"data": data, "format": "wav"}}
            ]}
        ]
    )
    return resp.choices[0].message.content


def main():
    import torch
    import torchaudio
    from cosyvoice.cli.cosyvoice import CosyVoice2
    from cosyvoice.utils.file_utils import load_wav
    from cosyvoice.utils.common import set_all_random_seed

    print("Loading model...")
    cosyvoice = CosyVoice2(str(MODEL_DIR), load_jit=False, load_trt=False, fp16=False)
    prompt = load_wav(str(VOICE_SAMPLE), 16000)

    print(f"\nDevice: {cosyvoice.model.device}")
    print(f"Model class: {type(cosyvoice.model)}")

    print("\n" + "=" * 50)
    print("MINIMAL TEST: 5 generations, same text")
    print("=" * 50)

    for run in range(1, 6):
        output_path = OUTPUT_DIR / f"minimal_test_{run}.wav"

        # Set seed for reproducibility test
        set_all_random_seed(42)

        print(f"\n--- Run {run}/5 ---")
        start = time.time()
        all_speech = []
        for result in cosyvoice.inference_instruct2(TEXT, INSTRUCTION, prompt, stream=False):
            all_speech.append(result['tts_speech'])

        full_speech = torch.cat(all_speech, dim=1)
        torchaudio.save(str(output_path), full_speech, cosyvoice.sample_rate)

        duration = full_speech.shape[1] / cosyvoice.sample_rate
        gen_time = time.time() - start
        print(f"Generated: {duration:.1f}s in {gen_time:.1f}s (RTF: {gen_time/duration:.2f}x)")

        # Evaluate
        try:
            eval_result = eval_audio(output_path)
            print(f"LLM Judge: {eval_result}")
        except Exception as e:
            print(f"Eval error: {e}")

    # Also evaluate OLD good file for comparison
    print("\n" + "=" * 50)
    print("OLD GOOD FILE (Dec 8)")
    print("=" * 50)
    old_file = OUTPUT_DIR / "cosyvoice_sichuan_grandma.wav"
    if old_file.exists():
        eval_result = eval_audio(old_file)
        print(f"LLM Judge: {eval_result}")


if __name__ == "__main__":
    main()
