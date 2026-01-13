#!/usr/bin/env python3
"""Generate Sichuanese mother-in-law nagging story with CosyVoice2."""

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

def main():
    import torch
    import torchaudio
    from cosyvoice.cli.cosyvoice import CosyVoice2
    from cosyvoice.utils.file_utils import load_wav

    print("Loading CosyVoice2-0.5B...")
    start = time.time()
    cosyvoice = CosyVoice2(MODEL_DIR, load_jit=False, load_trt=False, fp16=False)
    print(f"Model loaded in {time.time() - start:.2f}s")

    # Load voice sample
    prompt_speech_16k = load_wav(VOICE_SAMPLE, 16000)

    # Nagging mother-in-law story in Sichuanese style
    # Using Sichuanese expressions like 啷个 (how), 要得 (yahde = okay/yes),
    # 啥子 (what), 莫得 (don't have)
    story_text = """
哎呀！你啷个还在屋头坐起嘛？要得不嘛！
快点去超市买点东西回来！
家里头啥子都莫得了，酱油莫得了，盐也莫得了！
你是不是又在耍手机？要得要得，我晓得你忙。
但是今天必须去买！听到没有？
快点快点，莫磨蹭了！
"""

    # More explicit Sichuanese instruction
    sichuan_instruction = "用地道的四川话说这段话，要有四川老太婆唠叨的语气，声音要急躁一点"

    print(f"\nStory: {story_text.strip()}")
    print(f"\nInstruction: {sichuan_instruction}")
    print("\nGenerating Sichuanese nagging mother-in-law...")

    output_file = os.path.join(OUTPUT_DIR, "cosyvoice_sichuan_motherlnlaw.wav")

    start = time.time()
    all_speech = []
    for i, result in enumerate(cosyvoice.inference_instruct2(
        story_text.strip(),
        sichuan_instruction,
        prompt_speech_16k,
        stream=False
    )):
        all_speech.append(result['tts_speech'])

    # Concatenate all chunks
    full_speech = torch.cat(all_speech, dim=1)
    torchaudio.save(output_file, full_speech, cosyvoice.sample_rate)

    gen_time = time.time() - start

    # Get audio info
    waveform, sample_rate = torchaudio.load(output_file)
    audio_duration = waveform.shape[1] / sample_rate

    print(f"\n=== Results ===")
    print(f"Generation time: {gen_time:.2f}s")
    print(f"Audio duration: {audio_duration:.2f}s")
    print(f"RTF: {gen_time/audio_duration:.2f}x")
    print(f"Output: {output_file}")

    # Play the audio
    print("\nPlaying...")
    os.system(f"afplay '{output_file}'")

    return 0

if __name__ == "__main__":
    sys.exit(main())
