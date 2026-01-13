#!/usr/bin/env python3
"""Test simpler emotion instructions to find what works."""

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

# Same story text
STORY = """哎呀！你啷个还在屋头坐起嘛？
快点去超市买点东西回来！
家里头啥子都莫得了，酱油莫得了，盐也莫得了！
你是不是又在耍手机？
快点快点，莫磨蹭了！"""

# Simpler instructions - closer to what worked before
TESTS = [
    # Original working style
    ("用地道的四川话说这段话，要有四川老太婆唠叨的语气", "simple_nagging.wav", "Simple nagging (original style)"),

    # Simple happy - less extreme
    ("用四川话说这段话，语气开心一点", "simple_happy.wav", "Simple happy"),

    # Simple angry - less extreme
    ("用四川话说这段话，语气生气一点", "simple_angry.wav", "Simple angry"),

    # Try singing with simpler instruction
    ("用四川话唱这段话", "simple_singing.wav", "Simple singing"),

    # Formal Mandarin - simple
    ("用标准普通话说这段话", "simple_mandarin.wav", "Simple Mandarin"),
]


def main():
    import torch
    import torchaudio
    from cosyvoice.cli.cosyvoice import CosyVoice2
    from cosyvoice.utils.file_utils import load_wav

    from dotenv import load_dotenv
    load_dotenv()
    import openai

    print("Loading CosyVoice2...")
    cosyvoice = CosyVoice2(str(MODEL_DIR), load_jit=False, load_trt=False, fp16=False)
    prompt_speech_16k = load_wav(str(VOICE_SAMPLE), 16000)

    print("\n" + "=" * 60)
    print("Testing SIMPLER emotion instructions")
    print("=" * 60)

    results = []

    for instruction, filename, description in TESTS:
        print(f"\n--- {description} ---")
        print(f"Instruction: {instruction}")

        output_path = OUTPUT_DIR / filename

        start = time.time()
        all_speech = []
        for result in cosyvoice.inference_instruct2(
            STORY.strip(),
            instruction,
            prompt_speech_16k,
            stream=False
        ):
            all_speech.append(result['tts_speech'])

        full_speech = torch.cat(all_speech, dim=1)
        torchaudio.save(str(output_path), full_speech, cosyvoice.sample_rate)
        gen_time = time.time() - start

        waveform, sr = torchaudio.load(str(output_path))
        duration = waveform.shape[1] / sr

        print(f"Duration: {duration:.1f}s, RTF: {gen_time/duration:.2f}x")

        # Quick GPT evaluation
        try:
            client = openai.OpenAI()
            with open(output_path, "rb") as f:
                audio_data = base64.standard_b64encode(f.read()).decode("utf-8")

            response = client.chat.completions.create(
                model="gpt-audio-2025-08-28",
                modalities=["text"],
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Rate 1-10. DYING_FROG: YES/NO. One line."},
                        {"type": "input_audio", "input_audio": {"data": audio_data, "format": "wav"}}
                    ]
                }]
            )
            eval_result = response.choices[0].message.content
            print(f"GPT-5: {eval_result}")
            results.append((description, eval_result))
        except Exception as e:
            print(f"Eval error: {e}")
            results.append((description, "ERROR"))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for desc, result in results:
        print(f"{desc}: {result[:80]}...")


if __name__ == "__main__":
    main()
