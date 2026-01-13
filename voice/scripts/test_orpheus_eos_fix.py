#!/usr/bin/env python3
"""
Orpheus-TTS EOS Detection Fix

The model generates audio tokens but doesn't stop at end of speech.
This script explores different approaches to fix EOS detection.
"""

import sys
import time
import wave
import numpy as np
import torch

GGUF_PATH = "models/orpheus-tts-turkish-gguf/orpheus-tts-turkish-pt-5000-q5_k_m.gguf"
OUTPUT_WAV = "/tmp/orpheus_fixed.wav"

# Special tokens
START_TOKEN = 128259
TEXT_END_TOKEN = 128260
AUDIO_START_TOKEN = 128261
AUDIO_END_TOKEN = 128257
EOS_TOKEN = 128009
AUDIO_TOKEN_START = 128262

def main():
    print("Orpheus-TTS EOS Fix Experiments")
    print("=" * 60)

    try:
        from llama_cpp import Llama

        print("Loading model...")
        llm = Llama(
            model_path=GGUF_PATH,
            n_ctx=4096,
            n_gpu_layers=-1,
            verbose=False,
            logits_all=True  # Enable logit access
        )

        # Test text (short)
        text = "Merhaba"
        text_tokens = llm.tokenize(text.encode('utf-8'), add_bos=False)
        print(f"Text: '{text}' -> {len(text_tokens)} tokens")

        # Try different approaches
        approaches = [
            ("Low temp (0.1)", {"temp": 0.1, "top_k": 10, "top_p": 0.9}),
            ("Medium temp (0.3)", {"temp": 0.3, "top_k": 30, "top_p": 0.95}),
            ("High temp (0.7)", {"temp": 0.7, "top_k": 50, "top_p": 0.95}),
            ("Greedy", {"temp": 0.0}),
            ("Top-k only", {"temp": 0.5, "top_k": 20}),
        ]

        best_result = None
        best_ratio = float('inf')

        for name, params in approaches:
            print(f"\n--- {name} ---")

            # Format 3 prompt
            input_tokens = [START_TOKEN] + text_tokens + [EOS_TOKEN, TEXT_END_TOKEN, AUDIO_START_TOKEN, AUDIO_END_TOKEN]

            llm.reset()
            llm.eval(input_tokens)

            generated = []
            audio_started = False
            consecutive_same = 0
            last_token = None

            max_tokens = 500  # Limit for testing
            early_stop = False

            for i in range(max_tokens):
                token = llm.sample(**params)

                # Track repetition
                if token == last_token:
                    consecutive_same += 1
                else:
                    consecutive_same = 0
                last_token = token

                # Stop conditions
                if token == AUDIO_END_TOKEN:
                    print(f"  AUDIO_END at {i}")
                    early_stop = True
                    break
                if consecutive_same > 20:  # Repeated token
                    print(f"  Repetition at {i} (token {token} x{consecutive_same})")
                    early_stop = True
                    break
                if token in [128001, 128008, EOS_TOKEN] and i > 50:
                    print(f"  EOG({token}) at {i}")
                    early_stop = True
                    break

                generated.append(token)
                llm.eval([token])

            # Analyze
            audio_tokens = [t for t in generated if t >= AUDIO_TOKEN_START]
            ratio = len(generated) / (len(text_tokens) + 1)  # tokens per input char

            print(f"  Generated: {len(generated)} total, {len(audio_tokens)} audio")
            print(f"  Ratio: {ratio:.1f} tokens per input")
            print(f"  Early stop: {early_stop}")

            # Track best (lowest ratio with enough audio)
            if len(audio_tokens) >= 30 and ratio < best_ratio:
                best_ratio = ratio
                best_result = (name, generated, params)

        if best_result:
            print(f"\n{'='*60}")
            print(f"Best approach: {best_result[0]} (ratio {best_ratio:.1f})")

            # Decode best result
            audio_tokens = [t for t in best_result[1] if t >= AUDIO_TOKEN_START]
            if len(audio_tokens) >= 49:  # Need at least 7 frames
                decode_and_save(audio_tokens)
        else:
            print("\nNo good approach found")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

def decode_and_save(audio_tokens):
    """Decode audio tokens and save."""
    print("\nDecoding...")

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

        audio_np = audio.cpu().numpy().squeeze()
        duration = len(audio_np) / 24000

        print(f"Audio: {len(audio_np)} samples, {duration:.2f}s")

        # Save
        audio_norm = audio_np / (max(abs(audio_np.max()), abs(audio_np.min())) + 1e-8) * 0.9
        audio_int16 = (audio_norm * 32767).astype(np.int16)

        with wave.open(OUTPUT_WAV, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(audio_int16.tobytes())

        print(f"Saved: {OUTPUT_WAV}")
        print(f"Expected ~0.5-1s for 'Merhaba', got {duration:.2f}s")

    except Exception as e:
        print(f"Decode error: {e}")

if __name__ == "__main__":
    main()
