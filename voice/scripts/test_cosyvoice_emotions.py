#!/usr/bin/env python3
"""Test CosyVoice2 emotional expression and singing capabilities."""

import os
import sys
import time

# Add CosyVoice repo to path
COSYVOICE_REPO = os.path.join(os.path.dirname(__file__), '..', 'cosyvoice_repo')
sys.path.insert(0, COSYVOICE_REPO)
sys.path.insert(0, os.path.join(COSYVOICE_REPO, 'third_party', 'Matcha-TTS'))

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'cosyvoice', 'CosyVoice2-0.5B')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'cosyvoice', 'test_output')
VOICE_SAMPLE = os.path.join(os.path.dirname(__file__), '..', 'tests', 'golden', 'hello.wav')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# The happy popo story
POPO_STORY = """哎呀！你啷个还在屋头坐起嘛？
快点去超市买点东西回来！
家里头啥子都莫得了，酱油莫得了，盐也莫得了！
你是不是又在耍手机？
快点快点，莫磨蹭了！"""

# Same story in extremely formal/pretentious Mandarin
FORMAL_STORY = """哎呀！您为何仍旧端坐于室内？
请尽速前往超级市场采购物品！
家中已然一无所有，酱油消耗殆尽，食盐亦已罄空！
莫非您又沉溺于移动通讯设备之中？
敬请加速行动，切勿磨蹭拖延！"""

# 6 test configurations: (text, instruction, output_filename, description)
TEST_CONFIGS = [
    # 1. Sichuanese - very happy
    (POPO_STORY,
     "用四川话说这段话，要非常开心、高兴、笑嘻嘻的语气，像四川老太婆很开心地唠叨",
     "popo_sichuan_happy.wav",
     "Sichuanese Popo - Very Happy"),

    # 2. Formal Mandarin - very happy
    (FORMAL_STORY,
     "用非常标准的普通话说这段话，要非常开心、高兴的语气，像播音员很开心地朗读",
     "popo_mandarin_happy.wav",
     "Formal Mandarin - Very Happy"),

    # 3. Sichuanese - very angry
    (POPO_STORY,
     "用四川话说这段话，要非常生气、愤怒、大声的语气，像四川老太婆很生气地骂人",
     "popo_sichuan_angry.wav",
     "Sichuanese Popo - Very Angry"),

    # 4. Formal Mandarin - very angry
    (FORMAL_STORY,
     "用非常标准的普通话说这段话，要非常生气、愤怒、严厉的语气",
     "popo_mandarin_angry.wav",
     "Formal Mandarin - Very Angry"),

    # 5. Sichuanese - singing
    (POPO_STORY,
     "用四川话唱这段话，要用唱歌的方式，有旋律地唱出来，像四川民歌",
     "popo_sichuan_singing.wav",
     "Sichuanese Popo - Singing"),

    # 6. Formal Mandarin - singing
    (FORMAL_STORY,
     "用标准普通话唱这段话，要用唱歌的方式，有旋律地唱出来，像京剧念白或歌曲",
     "popo_mandarin_singing.wav",
     "Formal Mandarin - Singing"),
]


def main():
    import torch
    import torchaudio
    from cosyvoice.cli.cosyvoice import CosyVoice2
    from cosyvoice.utils.file_utils import load_wav

    print("=" * 70)
    print("CosyVoice2 Emotional Expression & Singing Test")
    print("=" * 70)
    print(f"Model: {MODEL_DIR}")
    print(f"Voice sample: {VOICE_SAMPLE}")
    print(f"Tests to run: {len(TEST_CONFIGS)}")
    print()

    print("Loading CosyVoice2-0.5B...")
    start = time.time()
    cosyvoice = CosyVoice2(MODEL_DIR, load_jit=False, load_trt=False, fp16=False)
    print(f"Model loaded in {time.time() - start:.2f}s\n")

    # Load voice sample
    prompt_speech_16k = load_wav(VOICE_SAMPLE, 16000)

    results = []

    for idx, (text, instruction, filename, description) in enumerate(TEST_CONFIGS, 1):
        print("=" * 70)
        print(f"Test {idx}/{len(TEST_CONFIGS)}: {description}")
        print("=" * 70)
        print(f"Text preview: {text[:50]}...")
        print(f"Instruction: {instruction}")
        print()

        output_file = os.path.join(OUTPUT_DIR, filename)

        start = time.time()
        all_speech = []

        for result in cosyvoice.inference_instruct2(
            text.strip(),
            instruction,
            prompt_speech_16k,
            stream=False
        ):
            all_speech.append(result['tts_speech'])

        # Concatenate all chunks
        full_speech = torch.cat(all_speech, dim=1)
        torchaudio.save(output_file, full_speech, cosyvoice.sample_rate)

        gen_time = time.time() - start

        # Get audio info
        waveform, sample_rate = torchaudio.load(output_file)
        audio_duration = waveform.shape[1] / sample_rate
        rtf = gen_time / audio_duration

        results.append({
            'description': description,
            'gen_time': gen_time,
            'duration': audio_duration,
            'rtf': rtf,
            'file': output_file
        })

        print(f"Generation time: {gen_time:.2f}s")
        print(f"Audio duration: {audio_duration:.2f}s")
        print(f"RTF: {rtf:.2f}x")
        print(f"Output: {output_file}")
        print()

        # Play the audio
        print(f"Playing: {description}...")
        os.system(f"afplay '{output_file}'")
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in results:
        print(f"  {r['description']}: {r['duration']:.1f}s (RTF: {r['rtf']:.2f}x) -> {os.path.basename(r['file'])}")

    total_audio = sum(r['duration'] for r in results)
    total_gen = sum(r['gen_time'] for r in results)
    print(f"\nTotal audio: {total_audio:.1f}s, Total generation: {total_gen:.1f}s, Avg RTF: {total_gen/total_audio:.2f}x")
    print(f"Output directory: {OUTPUT_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
