#!/usr/bin/env python3
"""Test CosyVoice2 with proper emotion tags embedded in text."""

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

# Test with PROPER tags embedded in text (per official docs)
TESTS = [
    # 1. With laughter tags - happy grandma
    ("""[laughter] 哎呀！你啷个还在屋头坐起嘛？[laughter]
快点去超市买点东西回来！
家里头啥子都莫得了 [breath] 酱油莫得了，盐也莫得了！
你是不是又在耍手机？[laughter]
快点快点，莫磨蹭了！""",
     "用地道的四川话说这段话，要有四川老太婆唠叨的语气",
     "tags_laughter.wav",
     "With [laughter] tags (happy)"),

    # 2. With emphasis tags - angry emphasis
    ("""哎呀！你啷个还在屋头坐起嘛？
<strong>快点</strong>去超市买点东西回来！
家里头<strong>啥子都莫得了</strong>，酱油莫得了，盐也莫得了！
你是不是又在耍手机？
<strong>快点快点，莫磨蹭了！</strong>""",
     "用地道的四川话说这段话，声音急躁一点",
     "tags_strong.wav",
     "With <strong> emphasis (angry)"),

    # 3. With sighs - tired/exasperated
    ("""[sigh] 哎呀！你啷个还在屋头坐起嘛？
快点去超市买点东西回来 [sigh]
家里头啥子都莫得了，酱油莫得了，盐也莫得了！[breath]
你是不是又在耍手机？[sigh]
快点快点，莫磨蹭了！""",
     "用地道的四川话说这段话",
     "tags_sigh.wav",
     "With [sigh] tags (exasperated)"),

    # 4. Standard Mandarin with emphasis
    ("""哎呀！您为何仍旧端坐于室内？
请<strong>尽速</strong>前往超级市场采购物品！
家中已然<strong>一无所有</strong>，酱油消耗殆尽，食盐亦已罄空！
莫非您又沉溺于移动通讯设备之中？
敬请<strong>加速行动</strong>，切勿磨蹭拖延！""",
     "用标准普通话说这段话",
     "tags_mandarin_strong.wav",
     "Mandarin with <strong> emphasis"),
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
    print("Testing PROPER emotion tags (per official docs)")
    print("=" * 60)

    results = []

    for text, instruction, filename, description in TESTS:
        print(f"\n--- {description} ---")
        print(f"Instruction: {instruction}")
        print(f"Text (first 80 chars): {text[:80]}...")

        output_path = OUTPUT_DIR / filename

        start = time.time()
        all_speech = []
        for result in cosyvoice.inference_instruct2(
            text.strip(),
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

        # Play it
        print("Playing...")
        os.system(f"afplay '{output_path}'")

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
                        {"type": "text", "text": "Rate 1-10. DYING_FROG: YES/NO. One line about quality."},
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
        print(f"{desc}: {result[:100]}...")


if __name__ == "__main__":
    main()
