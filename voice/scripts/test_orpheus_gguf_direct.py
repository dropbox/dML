#!/usr/bin/env python3
"""
Test Orpheus-TTS Turkish with GGUF using direct token IDs.

This approach uses llama-cpp-python with raw token input.
"""

import sys
import time
import wave
import numpy as np
import torch

# Model paths
GGUF_PATH = "models/orpheus-tts-turkish-gguf/orpheus-tts-turkish-pt-5000-q5_k_m.gguf"
OUTPUT_WAV = "/tmp/orpheus_test.wav"

# Special tokens for Orpheus-TTS (from HF model config)
# These need to match the model's vocabulary
START_TOKEN = 128259
END_TOKENS = [128009, 128260, 128261, 128257]
AUDIO_TOKEN_START = 128262  # Audio codes start from this ID
EOS_TOKEN = 128009

def main():
    print("Orpheus-TTS Turkish GGUF Test (Direct Token IDs)")
    print("=" * 60)

    try:
        from llama_cpp import Llama
        import snac

        # Load GGUF model
        print(f"Loading: {GGUF_PATH}")
        start = time.time()

        llm = Llama(
            model_path=GGUF_PATH,
            n_ctx=4096,
            n_gpu_layers=-1,  # Use Metal
            verbose=True  # Show loading info
        )

        load_time = time.time() - start
        print(f"Model loaded in {load_time:.2f}s")
        print(f"Vocab size: {llm.n_vocab()}")

        # Test text
        text = "Merhaba"

        # Tokenize text (basic)
        # llama.cpp tokenize method
        text_tokens = llm.tokenize(text.encode('utf-8'), add_bos=False)
        print(f"Text '{text}' -> tokens: {text_tokens}")

        # Build input: [START] + text_tokens + END_TOKENS
        input_tokens = [START_TOKEN] + text_tokens + END_TOKENS
        print(f"Input tokens: {input_tokens}")

        # Generate with eval + sample loop
        print("\nGenerating audio tokens...")
        start = time.time()

        # Reset KV cache
        llm.reset()

        # Eval input tokens
        llm.eval(input_tokens)

        # Sample tokens
        generated = []
        max_tokens = 1024
        for i in range(max_tokens):
            # Sample next token
            token = llm.sample(
                top_k=40,
                top_p=0.9,
                temp=0.3,
                repeat_penalty=1.1
            )

            # Check for EOS
            if token == EOS_TOKEN or token == llm.token_eos():
                print(f"EOS at token {i}")
                break

            generated.append(token)

            # Eval the new token
            llm.eval([token])

            # Progress
            if i % 100 == 0 and i > 0:
                print(f"  Generated {i} tokens...")

        gen_time = time.time() - start
        print(f"Generated {len(generated)} tokens in {gen_time:.2f}s")
        print(f"First 30 tokens: {generated[:30]}")

        # Count audio tokens
        audio_tokens = [t for t in generated if t >= AUDIO_TOKEN_START]
        text_tokens_gen = [t for t in generated if t < AUDIO_TOKEN_START]
        print(f"Audio tokens (>={AUDIO_TOKEN_START}): {len(audio_tokens)}")
        print(f"Text tokens (<{AUDIO_TOKEN_START}): {len(text_tokens_gen)}")

        if len(audio_tokens) < 50:
            print("\nWARNING: Few audio tokens generated")
            print("The model may need different prompting or parameters")

            # Decode text tokens to see what was generated
            if text_tokens_gen:
                decoded = llm.detokenize(text_tokens_gen)
                print(f"Generated text: {decoded.decode('utf-8', errors='ignore')}")
            return

        # SNAC decode
        print("\nLoading SNAC decoder...")
        snac_model = snac.SNAC.from_pretrained("hubertsiuzdak/snac_24khz")

        # Convert to codes (subtract base)
        codes = [t - AUDIO_TOKEN_START for t in audio_tokens]
        print(f"Audio codes range: {min(codes)} - {max(codes)}")

        # SNAC expects specific structure
        # Orpheus uses 7 codebooks, SNAC 24kHz has specific requirements
        # This is simplified - may need adjustment
        print("\nAttempting SNAC decode...")

        try:
            # SNAC 24kHz expects codes in 3 layers (from snac docs)
            # Layer 0: 1 code per frame
            # Layer 1: 2 codes per frame
            # Layer 2: 4 codes per frame
            # Total: 7 codes per frame

            # Parse 7-tuple codes into 3 layers
            num_frames = len(codes) // 7
            if num_frames < 5:
                print(f"Not enough codes for {num_frames} frames")
                return

            print(f"Processing {num_frames} frames")

            # Reshape codes into layers
            # This needs to match Orpheus's encoding pattern
            codes_tensor = torch.tensor(codes[:num_frames * 7]).reshape(num_frames, 7)

            # Layer 0: codes[0]
            # Layer 1: codes[1:3]
            # Layer 2: codes[3:7]
            layer0 = codes_tensor[:, 0:1].T.unsqueeze(0)  # [1, 1, T]
            layer1 = codes_tensor[:, 1:3].T.unsqueeze(0)  # [1, 2, T]
            layer2 = codes_tensor[:, 3:7].T.unsqueeze(0)  # [1, 4, T]

            # Decode
            with torch.no_grad():
                audio = snac_model.decode([layer0, layer1, layer2])

            audio_np = audio.cpu().numpy().squeeze()
            print(f"Audio shape: {audio_np.shape}")
            print(f"Duration: {len(audio_np) / 24000:.2f}s")

            # Save
            save_wav(audio_np, OUTPUT_WAV, 24000)
            print(f"\nSaved to: {OUTPUT_WAV}")
            print("Play with: afplay /tmp/orpheus_test.wav")

        except Exception as e:
            print(f"SNAC decode error: {e}")
            print("May need to adjust code redistribution logic")
            import traceback
            traceback.print_exc()

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

def save_wav(audio, path, sr=24000):
    """Save audio as WAV file."""
    # Normalize
    audio = audio / (max(abs(audio.max()), abs(audio.min())) + 1e-8) * 0.9
    audio_int16 = (audio * 32767).astype(np.int16)

    with wave.open(path, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes(audio_int16.tobytes())

if __name__ == "__main__":
    main()
