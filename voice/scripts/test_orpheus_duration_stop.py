#!/usr/bin/env python3
"""
Orpheus-TTS with Duration-Based Stop

Use audio token count to estimate duration and stop appropriately.
~7 tokens = 1 frame = ~10.67ms audio (24kHz, 256 samples)
So ~700 tokens = 1 second of audio.
"""

import sys
import time
import wave
import numpy as np
import torch

GGUF_PATH = "models/orpheus-tts-turkish-gguf/orpheus-tts-turkish-pt-5000-q5_k_m.gguf"
OUTPUT_WAV = "/tmp/orpheus_duration.wav"

# Special tokens
START_TOKEN = 128259
TEXT_END_TOKEN = 128260
AUDIO_START_TOKEN = 128261
AUDIO_END_TOKEN = 128257
EOS_TOKEN = 128009
AUDIO_TOKEN_START = 128262

# Heuristic: ~700 audio tokens per second
# Turkish speech: ~4-6 chars/second
# So: target_tokens = (text_len_chars / 5) * 700 = text_len_chars * 140

def estimate_audio_tokens(text):
    """Estimate how many audio tokens for text length."""
    # Adjust for Turkish: ~140 tokens per character
    # Add 20% buffer for pauses
    return int(len(text) * 140 * 1.2)

def main():
    print("Orpheus-TTS Duration-Based Stop")
    print("=" * 60)

    try:
        from llama_cpp import Llama
        import snac

        print("Loading model...")
        llm = Llama(
            model_path=GGUF_PATH,
            n_ctx=4096,
            n_gpu_layers=-1,
            verbose=False
        )

        # Test text
        text = "Merhaba, bugün hava çok güzel."
        text_tokens = llm.tokenize(text.encode('utf-8'), add_bos=False)

        # Estimate target
        target_tokens = estimate_audio_tokens(text)
        print(f"Text: '{text}'")
        print(f"Text length: {len(text)} chars")
        print(f"Target audio tokens: {target_tokens}")

        # Format 3 prompt
        input_tokens = [START_TOKEN] + text_tokens + [EOS_TOKEN, TEXT_END_TOKEN, AUDIO_START_TOKEN, AUDIO_END_TOKEN]

        print("\nGenerating...")
        start_time = time.time()

        llm.reset()
        llm.eval(input_tokens)

        generated = []
        audio_count = 0

        # Generate until target audio tokens reached
        max_tokens = target_tokens + 200  # Buffer
        for i in range(max_tokens):
            token = llm.sample(temp=0.5, top_k=50, top_p=0.95)

            # Early stop on EOS tokens
            if token == AUDIO_END_TOKEN:
                print(f"AUDIO_END at {i}")
                break
            if token in [128001, 128008, EOS_TOKEN] and i > 50:
                print(f"EOG at {i}")
                break

            generated.append(token)
            llm.eval([token])

            if token >= AUDIO_TOKEN_START:
                audio_count += 1
                if audio_count >= target_tokens:
                    print(f"Duration target reached at {i}")
                    break

        gen_time = time.time() - start_time
        audio_tokens = [t for t in generated if t >= AUDIO_TOKEN_START]
        print(f"Generated: {len(generated)} total, {len(audio_tokens)} audio in {gen_time:.2f}s")

        # Decode
        if len(audio_tokens) >= 49:
            print("\nDecoding...")
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

            audio_np = audio.cpu().numpy().squeeze()
            duration = len(audio_np) / 24000

            print(f"Audio duration: {duration:.2f}s")
            print(f"Expected: ~{len(text)/5:.1f}s for {len(text)} chars")

            # Check quality
            rms = np.sqrt(np.mean(audio_np ** 2))
            print(f"RMS: {rms:.4f}")

            # Save
            audio_norm = audio_np / (max(abs(audio_np.max()), abs(audio_np.min())) + 1e-8) * 0.9
            audio_int16 = (audio_norm * 32767).astype(np.int16)

            with wave.open(OUTPUT_WAV, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(24000)
                wav.writeframes(audio_int16.tobytes())

            print(f"\nSaved: {OUTPUT_WAV}")
            print("Play: afplay /tmp/orpheus_duration.wav")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
