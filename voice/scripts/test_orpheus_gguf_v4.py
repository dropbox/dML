#!/usr/bin/env python3
"""
Orpheus-TTS Turkish with GGUF - v4 with correct SNAC mapping.

Issue: Orpheus generates codes 0-28410, but SNAC expects 0-4095 per codebook.
Solution: Map codes using modulo 4096 for each codebook layer.

Orpheus structure (7 codes per frame):
- Code 0: maps to SNAC codebook 0
- Codes 1-2: map to SNAC codebook 1
- Codes 3-6: map to SNAC codebook 2
"""

import sys
import time
import wave
import numpy as np
import torch

GGUF_PATH = "models/orpheus-tts-turkish-gguf/orpheus-tts-turkish-pt-5000-q5_k_m.gguf"
OUTPUT_WAV = "/tmp/orpheus_test.wav"

# Special tokens
START_TOKEN = 128259
TEXT_END_TOKEN = 128260
AUDIO_START_TOKEN = 128261
AUDIO_END_TOKEN = 128257
EOS_TOKEN = 128009
AUDIO_TOKEN_START = 128262

def main():
    print("Orpheus-TTS Turkish GGUF Test v4")
    print("=" * 60)

    try:
        from llama_cpp import Llama

        print(f"Loading model...")
        llm = Llama(
            model_path=GGUF_PATH,
            n_ctx=4096,
            n_gpu_layers=-1,
            verbose=False
        )
        print(f"Model loaded")

        # Test text
        text = "Merhaba, bugün hava çok güzel."

        # Tokenize
        text_tokens = llm.tokenize(text.encode('utf-8'), add_bos=False)
        print(f"Text: '{text}'")

        # Format 3
        input_tokens = [START_TOKEN] + text_tokens + [EOS_TOKEN, TEXT_END_TOKEN, AUDIO_START_TOKEN, AUDIO_END_TOKEN]

        # Generate
        print("\nGenerating...")
        start_time = time.time()

        llm.reset()
        llm.eval(input_tokens)

        generated = []
        for i in range(2048):
            token = llm.sample(top_k=50, top_p=0.95, temp=0.5)

            if token == AUDIO_END_TOKEN:
                print(f"AUDIO_END at {i}")
                break
            if token in [128001, 128008] and i > 100:
                break
            if token == EOS_TOKEN and i > 200:
                print(f"EOS at {i}")
                break

            generated.append(token)
            llm.eval([token])

        gen_time = time.time() - start_time
        print(f"Generated {len(generated)} tokens in {gen_time:.2f}s")

        # Extract audio tokens
        audio_tokens = [t for t in generated if t >= AUDIO_TOKEN_START]
        print(f"Audio tokens: {len(audio_tokens)}")

        if len(audio_tokens) < 50:
            print("Not enough audio tokens")
            return

        decode_with_snac_v2(audio_tokens)

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

def decode_with_snac_v2(audio_tokens):
    """Decode with corrected SNAC mapping."""
    print("\n" + "=" * 60)
    print("Decoding with SNAC (v2 mapping)...")

    try:
        import snac

        snac_model = snac.SNAC.from_pretrained("hubertsiuzdak/snac_24khz")

        # Convert to raw codes
        raw_codes = [t - AUDIO_TOKEN_START for t in audio_tokens]
        print(f"Raw codes: {len(raw_codes)}, range {min(raw_codes)}-{max(raw_codes)}")

        # Try multiple interpretations

        # Interpretation 1: Codes are flat indices into 7-codebook space
        # Each codebook has 4096 entries
        # Code = codebook_index * 4096 + entry
        print("\n--- Interpretation 1: 7-codebook flat mapping ---")
        try:
            cb_sizes = [4096] * 7
            parsed_codes = []
            for c in raw_codes:
                cb_idx = 0
                offset = c
                for i, size in enumerate(cb_sizes):
                    if offset < size:
                        cb_idx = i
                        break
                    offset -= size
                else:
                    # Clip to last codebook
                    cb_idx = 6
                    offset = c % 4096
                parsed_codes.append((cb_idx, offset % 4096))

            # Group into frames of 7
            num_frames = len(parsed_codes) // 7
            print(f"Frames: {num_frames}")

            if num_frames < 5:
                print("Not enough frames")
            else:
                # Extract SNAC 3-codebook format from 7-codebook
                # SNAC uses 3 codebooks: 1 + 2 + 4 codes per frame
                cb0 = []
                cb1 = []
                cb2 = []

                for i in range(num_frames):
                    base = i * 7
                    # Map 7 codes to 3 codebooks (1+2+4)
                    cb0.append(parsed_codes[base][1])
                    cb1.extend([parsed_codes[base + 1][1], parsed_codes[base + 2][1]])
                    cb2.extend([parsed_codes[base + j][1] for j in range(3, 7)])

                # Decode
                c0 = torch.tensor(cb0).unsqueeze(0)
                c1 = torch.tensor(cb1).unsqueeze(0)
                c2 = torch.tensor(cb2).unsqueeze(0)

                print(f"c0: {c0.shape}, c1: {c1.shape}, c2: {c2.shape}")

                with torch.no_grad():
                    audio = snac_model.decode([c0, c1, c2])
                audio_np = audio.cpu().numpy().squeeze()
                print(f"Audio: {audio_np.shape}, {len(audio_np)/24000:.2f}s")
                save_wav(audio_np, OUTPUT_WAV)
                print(f"Saved to {OUTPUT_WAV}")
                return

        except Exception as e:
            print(f"Interpretation 1 failed: {e}")

        # Interpretation 2: Direct modulo mapping
        print("\n--- Interpretation 2: Direct modulo 4096 ---")
        try:
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

            print(f"c0: {c0.shape}, c1: {c1.shape}, c2: {c2.shape}")

            with torch.no_grad():
                audio = snac_model.decode([c0, c1, c2])
            audio_np = audio.cpu().numpy().squeeze()

            rms = np.sqrt(np.mean(audio_np ** 2))
            print(f"Audio: {audio_np.shape}, {len(audio_np)/24000:.2f}s, RMS={rms:.4f}")

            save_wav(audio_np, OUTPUT_WAV)
            print(f"Saved to {OUTPUT_WAV}")
            print("Play: afplay /tmp/orpheus_test.wav")
            return

        except Exception as e:
            print(f"Interpretation 2 failed: {e}")

        print("\nAll interpretations failed")

    except Exception as e:
        print(f"SNAC error: {e}")
        import traceback
        traceback.print_exc()

def save_wav(audio, path, sr=24000):
    audio = audio / (max(abs(audio.max()), abs(audio.min())) + 1e-8) * 0.9
    audio_int16 = (audio * 32767).astype(np.int16)
    with wave.open(path, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes(audio_int16.tobytes())

if __name__ == "__main__":
    main()
