#!/usr/bin/env python3
"""Test CosyVoice2-0.5B with Sichuanese dialect instruction."""

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
    print("=" * 60)
    print("CosyVoice2-0.5B Sichuanese Test")
    print("=" * 60)
    print(f"Model: {MODEL_DIR}")
    print(f"Voice sample: {VOICE_SAMPLE}")
    print()

    import torch
    import torchaudio
    from cosyvoice.cli.cosyvoice import CosyVoice2
    from cosyvoice.utils.file_utils import load_wav

    # Load model
    print("Loading CosyVoice2-0.5B...")
    start = time.time()
    cosyvoice = CosyVoice2(MODEL_DIR, load_jit=False, load_trt=False, fp16=False)
    print(f"Model loaded in {time.time() - start:.2f}s")

    # Load voice sample for zero-shot cloning
    prompt_speech_16k = load_wav(VOICE_SAMPLE, 16000)

    # Test texts
    test_text = "你好，今天天气真好啊！"
    sichuan_instruction = "用四川话说这句话"  # Say this in Sichuanese dialect

    # Generate Sichuanese speech using instruct mode
    print(f"\nText: {test_text}")
    print(f"Instruction: {sichuan_instruction}")
    print("\nGenerating Sichuanese speech...")

    start = time.time()
    output_file = os.path.join(OUTPUT_DIR, "cosyvoice_sichuan.wav")

    for i, result in enumerate(cosyvoice.inference_instruct2(
        test_text,
        sichuan_instruction,
        prompt_speech_16k,
        stream=False
    )):
        torchaudio.save(output_file, result['tts_speech'], cosyvoice.sample_rate)

    gen_time = time.time() - start

    # Get audio info
    waveform, sample_rate = torchaudio.load(output_file)
    audio_duration = waveform.shape[1] / sample_rate
    rtf = gen_time / audio_duration

    print(f"\n=== Results ===")
    print(f"Generation time: {gen_time:.2f}s")
    print(f"Audio duration: {audio_duration:.2f}s")
    print(f"RTF: {rtf:.2f}x")
    print(f"Output: {output_file}")

    # Play the audio
    print("\nPlaying Sichuanese audio...")
    os.system(f"afplay '{output_file}'")

    return 0

if __name__ == "__main__":
    sys.exit(main())
