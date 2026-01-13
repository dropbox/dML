#!/usr/bin/env python3
"""Test if setting seed improves CosyVoice2 quality."""

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

TEXT = "你好，今天天气真好啊！"
INSTRUCTION = "用四川话说"

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
                {"type": "text", "text": 'Rate audio 1-10. Frog/distortion? {"score":X,"frog":bool}'},
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

    print("\n" + "=" * 50)
    print("SEED QUALITY TEST")
    print("=" * 50)

    # Test with different seeds
    seeds = [0, 42, 1986, 12345]

    for seed in seeds:
        output_path = OUTPUT_DIR / f"seed_test_{seed}.wav"

        print(f"\n--- Seed {seed} ---")
        set_all_random_seed(seed)

        start = time.time()
        all_speech = []
        for result in cosyvoice.inference_instruct2(TEXT, INSTRUCTION, prompt, stream=False):
            all_speech.append(result['tts_speech'])

        full_speech = torch.cat(all_speech, dim=1)
        torchaudio.save(str(output_path), full_speech, cosyvoice.sample_rate)

        duration = full_speech.shape[1] / cosyvoice.sample_rate
        gen_time = time.time() - start
        print(f"Generated: {duration:.1f}s in {gen_time:.1f}s")

        # Evaluate
        try:
            eval_result = eval_audio(output_path)
            print(f"LLM Judge: {eval_result}")
        except Exception as e:
            print(f"Eval error: {e}")


if __name__ == "__main__":
    main()
