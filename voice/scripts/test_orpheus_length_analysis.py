#!/usr/bin/env python3
"""
Orpheus-TTS Length Analysis

Investigate why short text hits EOS quickly but long text never does.
"""

import sys
import time
import wave
import numpy as np
import torch

GGUF_PATH = "models/orpheus-tts-turkish-gguf/orpheus-tts-turkish-pt-5000-q5_k_m.gguf"

# Special tokens
START_TOKEN = 128259
TEXT_END_TOKEN = 128260
AUDIO_START_TOKEN = 128261
AUDIO_END_TOKEN = 128257
EOS_TOKEN = 128009
AUDIO_TOKEN_START = 128262

def main():
    print("Orpheus-TTS Length Analysis")
    print("=" * 60)

    try:
        from llama_cpp import Llama

        print("Loading model...")
        llm = Llama(
            model_path=GGUF_PATH,
            n_ctx=4096,
            n_gpu_layers=-1,
            verbose=False
        )

        # Test different text lengths
        texts = [
            "Merhaba",                                    # 1 word
            "Merhaba, nasılsın?",                         # 2 words
            "Merhaba, bugün nasılsın?",                   # 3 words
            "Merhaba, bugün hava çok güzel.",             # 5 words
            "Merhaba, bugün hava çok güzel. Nasılsınız?", # 6+ words
        ]

        print("\nResults:")
        print("-" * 70)
        print(f"{'Text':<45} {'TokIn':>6} {'Gen':>6} {'Audio':>6} {'EOS':>6}")
        print("-" * 70)

        for text in texts:
            text_tokens = llm.tokenize(text.encode('utf-8'), add_bos=False)

            # Format 3 prompt
            input_tokens = [START_TOKEN] + text_tokens + [EOS_TOKEN, TEXT_END_TOKEN, AUDIO_START_TOKEN, AUDIO_END_TOKEN]

            llm.reset()
            llm.eval(input_tokens)

            generated = []
            eos_at = None
            max_tokens = 1000

            for i in range(max_tokens):
                token = llm.sample(temp=0.5, top_k=50, top_p=0.95)

                if token == AUDIO_END_TOKEN:
                    eos_at = i
                    break
                if token in [128001, 128008, EOS_TOKEN] and i > 30:
                    eos_at = i
                    break

                generated.append(token)
                llm.eval([token])

            audio_tokens = [t for t in generated if t >= AUDIO_TOKEN_START]

            eos_str = str(eos_at) if eos_at else "NO"
            text_short = text[:42] + "..." if len(text) > 45 else text
            print(f"{text_short:<45} {len(text_tokens):>6} {len(generated):>6} {len(audio_tokens):>6} {eos_str:>6}")

            # Save audio for the working ones
            if len(audio_tokens) >= 49 and eos_at:
                decode_and_check(audio_tokens, text)

        print("-" * 70)

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

def decode_and_check(audio_tokens, text):
    """Decode and report duration."""
    try:
        import snac
        snac_model = snac.SNAC.from_pretrained("hubertsiuzdak/snac_24khz")

        raw_codes = [t - AUDIO_TOKEN_START for t in audio_tokens]
        num_frames = len(raw_codes) // 7

        cb0 = [raw_codes[i * 7] % 4096 for i in range(num_frames)]
        cb1 = []
        cb2 = []
        for i in range(num_frames):
            base = i * 7
            cb1.extend([raw_codes[base + 1] % 4096, raw_codes[base + 2] % 4096])
            cb2.extend([raw_codes[base + j] % 4096 for j in range(3, 7)])

        c0 = torch.tensor(cb0).unsqueeze(0)
        c1 = torch.tensor(cb1).unsqueeze(0)
        c2 = torch.tensor(cb2).unsqueeze(0)

        with torch.no_grad():
            audio = snac_model.decode([c0, c1, c2])

        duration = len(audio.squeeze()) / 24000

        # Save
        output_path = f"/tmp/orpheus_{len(text.split('_')[0][:10])}.wav"
        audio_np = audio.cpu().numpy().squeeze()
        audio_norm = audio_np / (max(abs(audio_np.max()), abs(audio_np.min())) + 1e-8) * 0.9
        audio_int16 = (audio_norm * 32767).astype(np.int16)

        with wave.open(output_path, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(audio_int16.tobytes())

        print(f"  -> Saved {output_path} ({duration:.2f}s)")

    except Exception as e:
        print(f"  -> Decode error: {e}")

if __name__ == "__main__":
    main()
