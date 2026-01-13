#!/usr/bin/env python3
"""
Orpheus-TTS Turkish with GGUF - v3 with fixed SNAC decode.

Format 3 works: START+text+EOS+AUDIO_START generates 331 audio tokens
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
    print("Orpheus-TTS Turkish GGUF Test v3")
    print("=" * 60)

    try:
        from llama_cpp import Llama

        # Load model
        print(f"Loading model...")
        llm = Llama(
            model_path=GGUF_PATH,
            n_ctx=4096,
            n_gpu_layers=-1,
            verbose=False
        )
        print(f"Model loaded. Vocab: {llm.n_vocab()}")

        # Test text
        text = "Merhaba, bugün hava çok güzel. Nasılsınız?"

        # Tokenize
        text_tokens = llm.tokenize(text.encode('utf-8'), add_bos=False)
        print(f"Text: '{text}'")
        print(f"Text tokens: {len(text_tokens)}")

        # Format 3: START+text+EOS+TEXT_END+AUDIO_START+AUDIO_END
        input_tokens = [START_TOKEN] + text_tokens + [EOS_TOKEN, TEXT_END_TOKEN, AUDIO_START_TOKEN, AUDIO_END_TOKEN]
        print(f"Input tokens: {len(input_tokens)}")

        # Generate
        print("\nGenerating audio tokens...")
        start_time = time.time()

        llm.reset()
        llm.eval(input_tokens)

        generated = []
        for i in range(2048):  # More tokens for longer audio
            token = llm.sample(
                top_k=50,
                top_p=0.95,
                temp=0.5,
                repeat_penalty=1.0,
            )

            # Stop on specific end markers
            if token == AUDIO_END_TOKEN:
                print(f"AUDIO_END at {i}")
                break
            if token in [128001, 128008] and i > 100:  # Allow minimum
                print(f"EOG at {i}")
                break
            if token == EOS_TOKEN and i > 200:
                print(f"EOS at {i}")
                break

            generated.append(token)
            llm.eval([token])

            if i % 200 == 0 and i > 0:
                print(f"  {i} tokens...")

        gen_time = time.time() - start_time
        print(f"Generation: {len(generated)} tokens in {gen_time:.2f}s")

        # Extract audio tokens
        audio_tokens = [t for t in generated if t >= AUDIO_TOKEN_START]
        print(f"Audio tokens: {len(audio_tokens)}")

        if len(audio_tokens) < 100:
            print("WARNING: Not enough audio tokens")
            return

        # Decode with SNAC
        decode_with_snac(audio_tokens)

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

def decode_with_snac(audio_tokens):
    """Decode audio tokens with SNAC - fixed tensor shapes."""
    print("\n" + "=" * 60)
    print("Decoding with SNAC...")

    try:
        import snac

        snac_model = snac.SNAC.from_pretrained("hubertsiuzdak/snac_24khz")

        # Convert to codes (subtract base)
        codes = [t - AUDIO_TOKEN_START for t in audio_tokens]
        print(f"Total codes: {len(codes)}")
        print(f"Code range: {min(codes)} - {max(codes)}")

        # SNAC 24kHz codebook info:
        # - Has 3 quantizer layers
        # - Each layer has different temporal resolution
        # - Orpheus generates 7 codes per "frame" (interleaved)

        # The codes are interleaved: [c0, c1a, c1b, c2a, c2b, c2c, c2d, ...]
        # where c0 is codebook 0, c1a/c1b are codebook 1, c2a-c2d are codebook 2

        # SNAC expects codes as list of 3 tensors:
        # - codes[0]: [batch, num_frames] - codebook 0 (1 per frame)
        # - codes[1]: [batch, num_frames * 2] - codebook 1 (2 per frame)
        # - codes[2]: [batch, num_frames * 4] - codebook 2 (4 per frame)

        num_frames = len(codes) // 7
        if num_frames < 5:
            print(f"Not enough frames: {num_frames}")
            return

        print(f"Frames: {num_frames}")

        # Parse interleaved codes into layers
        cb0_codes = []  # 1 code per frame
        cb1_codes = []  # 2 codes per frame
        cb2_codes = []  # 4 codes per frame

        for i in range(num_frames):
            base = i * 7
            cb0_codes.append(codes[base])
            cb1_codes.extend(codes[base + 1:base + 3])
            cb2_codes.extend(codes[base + 3:base + 7])

        print(f"Codebook 0: {len(cb0_codes)} codes")
        print(f"Codebook 1: {len(cb1_codes)} codes")
        print(f"Codebook 2: {len(cb2_codes)} codes")

        # Convert to tensors with correct shapes
        # SNAC decode expects list of [batch, codes_per_layer, sequence]
        # But from snac source code, it's actually [batch, num_codes_at_rate]
        c0 = torch.tensor(cb0_codes).unsqueeze(0)  # [1, T]
        c1 = torch.tensor(cb1_codes).unsqueeze(0)  # [1, 2*T]
        c2 = torch.tensor(cb2_codes).unsqueeze(0)  # [1, 4*T]

        print(f"c0 shape: {c0.shape}")
        print(f"c1 shape: {c1.shape}")
        print(f"c2 shape: {c2.shape}")

        # Decode
        with torch.no_grad():
            audio = snac_model.decode([c0, c1, c2])

        audio_np = audio.cpu().numpy().squeeze()
        print(f"Audio shape: {audio_np.shape}")
        print(f"Duration: {len(audio_np) / 24000:.2f}s")

        # Check for valid audio
        rms = np.sqrt(np.mean(audio_np ** 2))
        peak = np.max(np.abs(audio_np))
        print(f"RMS: {rms:.4f}, Peak: {peak:.4f}")

        if rms < 0.001:
            print("WARNING: Audio may be silent/noise")

        # Save
        save_wav(audio_np, OUTPUT_WAV, 24000)
        print(f"\nSaved: {OUTPUT_WAV}")
        print("Play: afplay /tmp/orpheus_test.wav")

    except Exception as e:
        print(f"SNAC error: {e}")
        import traceback
        traceback.print_exc()

def save_wav(audio, path, sr=24000):
    """Save audio as WAV."""
    audio = audio / (max(abs(audio.max()), abs(audio.min())) + 1e-8) * 0.9
    audio_int16 = (audio * 32767).astype(np.int16)

    with wave.open(path, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes(audio_int16.tobytes())

if __name__ == "__main__":
    main()
