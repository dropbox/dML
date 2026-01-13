#!/usr/bin/env python3
"""
Test Orpheus-TTS Turkish inference with SNAC decoder.

Correct inference flow:
1. Tokenize text with special start/end tokens
2. Generate audio token codes (IDs in range 128262+)
3. Decode audio codes with SNAC
"""

import sys
import time
import wave
import struct
import numpy as np
import torch

# Model paths
GGUF_PATH = "models/orpheus-tts-turkish-gguf/orpheus-tts-turkish-pt-5000-q5_k_m.gguf"
MODEL_ID = "Karayakar/Orpheus-TTS-Turkish-PT-5000"
OUTPUT_WAV = "/tmp/orpheus_test.wav"

# Special tokens for Orpheus-TTS
START_TOKEN = 128259
END_TOKENS = [128009, 128260, 128261, 128257]
AUDIO_TOKEN_START = 128262  # Audio codes start from this ID

def test_with_transformers():
    """Test with transformers (full precision, slower but correct)."""
    print("=== Testing Orpheus-TTS with Transformers ===")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import snac

        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            cache_dir="models/orpheus-tts-turkish"
        )

        # Load model
        print("Loading model (this may take a moment)...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            cache_dir="models/orpheus-tts-turkish",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print(f"Model loaded on: {model.device}")

        # Load SNAC
        print("Loading SNAC decoder...")
        snac_model = snac.SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
        if torch.backends.mps.is_available():
            snac_model = snac_model.to("mps")
            print("SNAC on MPS")
        else:
            print("SNAC on CPU")

        # Test text
        text = "Merhaba, nasılsınız?"
        print(f"\nGenerating audio for: '{text}'")

        # Tokenize with special format
        text_tokens = tokenizer.encode(text, add_special_tokens=False)
        print(f"Text tokens: {text_tokens}")

        # Build input: [START] + text + END_TOKENS
        input_ids = torch.tensor([[START_TOKEN] + text_tokens + END_TOKENS])
        print(f"Input IDs shape: {input_ids.shape}")
        print(f"Input IDs: {input_ids[0].tolist()}")

        # Move to device
        input_ids = input_ids.to(model.device)

        # Generate audio tokens
        print("\nGenerating audio tokens...")
        start = time.time()

        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                max_new_tokens=2048,
                do_sample=True,
                temperature=0.2,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        gen_time = time.time() - start
        print(f"Generation took {gen_time:.2f}s")

        # Extract generated tokens (after input)
        gen_tokens = generated[0, input_ids.shape[1]:].tolist()
        print(f"Generated {len(gen_tokens)} tokens")
        print(f"First 20 tokens: {gen_tokens[:20]}")
        print(f"Last 20 tokens: {gen_tokens[-20:]}")

        # Count audio tokens
        audio_tokens = [t for t in gen_tokens if t >= AUDIO_TOKEN_START]
        print(f"Audio tokens (>={AUDIO_TOKEN_START}): {len(audio_tokens)}")

        if len(audio_tokens) < 100:
            print("WARNING: Very few audio tokens generated!")
            print("Model may not be generating TTS output correctly")
            return False

        # Parse audio codes for SNAC
        # Orpheus uses 4 codebooks (7 code values each)
        # Need to redistribute codes across codebooks
        print("\nDecoding with SNAC...")

        try:
            # Convert audio token IDs to codes (subtract base)
            codes = [t - AUDIO_TOKEN_START for t in audio_tokens]

            # Redistribute to 4 codebook layers
            # SNAC expects codes in specific format
            num_codes = len(codes)
            codes_per_layer = num_codes // 4

            if codes_per_layer < 10:
                print(f"Not enough codes for SNAC: {codes_per_layer} per layer")
                return False

            # Simple reshape (may need adjustment based on actual format)
            layer_codes = []
            for i in range(4):
                layer = codes[i::4][:codes_per_layer]
                layer_codes.append(torch.tensor([layer]))

            # Stack for SNAC
            snac_input = [l.to(snac_model.device) for l in layer_codes]

            # Decode
            with torch.no_grad():
                audio = snac_model.decode(snac_input)

            audio_np = audio.cpu().numpy().squeeze()
            print(f"Audio shape: {audio_np.shape}")
            print(f"Audio duration: {len(audio_np) / 24000:.2f}s")

            # Save WAV
            save_wav(audio_np, OUTPUT_WAV, 24000)
            print(f"Saved to: {OUTPUT_WAV}")

            return True

        except Exception as e:
            print(f"SNAC decode error: {e}")
            print("This may require adjusting the code redistribution logic")
            return False

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def save_wav(audio, path, sr=24000):
    """Save audio as WAV file."""
    # Normalize to int16
    audio = audio / max(abs(audio.max()), abs(audio.min())) * 0.9
    audio_int16 = (audio * 32767).astype(np.int16)

    with wave.open(path, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes(audio_int16.tobytes())

def test_gguf_model():
    """Test GGUF model (faster but need correct token handling)."""
    print("\n=== Testing GGUF Model ===")

    try:
        from llama_cpp import Llama

        print(f"Loading: {GGUF_PATH}")
        llm = Llama(
            model_path=GGUF_PATH,
            n_ctx=4096,
            n_gpu_layers=-1,
            verbose=False
        )
        print("GGUF model loaded")

        # The GGUF model uses raw token IDs, not text prompts
        # We need to pass token IDs directly
        print("\nNote: GGUF requires direct token ID input for TTS")
        print("This needs custom integration (not simple text prompt)")

        return True
    except Exception as e:
        print(f"GGUF error: {e}")
        return False

def main():
    print("Orpheus-TTS Turkish Test")
    print("=" * 60)

    # Try GGUF first (faster if it works)
    # gguf_ok = test_gguf_model()

    # Then transformers (correct but slower)
    transformers_ok = test_with_transformers()

    print("\n" + "=" * 60)
    print("Results:")
    # print(f"  GGUF: {'OK' if gguf_ok else 'NEEDS WORK'}")
    print(f"  Transformers: {'OK' if transformers_ok else 'NEEDS WORK'}")

    if transformers_ok:
        print(f"\nAudio saved to: {OUTPUT_WAV}")
        print("Play with: afplay /tmp/orpheus_test.wav")

if __name__ == "__main__":
    main()
