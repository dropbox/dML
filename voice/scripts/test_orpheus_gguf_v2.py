#!/usr/bin/env python3
"""
Test Orpheus-TTS Turkish with GGUF - improved version.

Findings from v1:
- Model DOES generate audio tokens (132247+)
- But EOS triggered early
- Need to adjust prompting and sampling
"""

import sys
import time
import wave
import numpy as np
import torch

GGUF_PATH = "models/orpheus-tts-turkish-gguf/orpheus-tts-turkish-pt-5000-q5_k_m.gguf"
OUTPUT_WAV = "/tmp/orpheus_test.wav"

# Special tokens - from model vocab analysis
# These were derived from observing the model behavior
START_TOKEN = 128259       # <|audio_start|> or similar
TEXT_END_TOKEN = 128260    # End of text input marker
AUDIO_START_TOKEN = 128261 # Start of audio output
AUDIO_END_TOKEN = 128257   # End of audio
EOS_TOKEN = 128009         # <|eot_id|>

# Audio tokens start at 128262 and go up to ~156940
AUDIO_TOKEN_START = 128262

def main():
    print("Orpheus-TTS Turkish GGUF Test v2")
    print("=" * 60)

    try:
        from llama_cpp import Llama
        import snac

        # Load model
        print(f"Loading: {GGUF_PATH}")
        llm = Llama(
            model_path=GGUF_PATH,
            n_ctx=4096,
            n_gpu_layers=-1,
            verbose=False
        )
        print(f"Model loaded. Vocab: {llm.n_vocab()}")

        # Test multiple prompt formats
        text = "Merhaba, bugün hava çok güzel."

        # Tokenize text
        text_tokens = llm.tokenize(text.encode('utf-8'), add_bos=False)
        print(f"Text: '{text}'")
        print(f"Text tokens ({len(text_tokens)}): {text_tokens}")

        # Try different prompt formats
        prompts = [
            # Format 1: START + text + TEXT_END + AUDIO_START
            ("Format 1: START+text+TEXT_END+AUDIO_START",
             [START_TOKEN] + text_tokens + [TEXT_END_TOKEN, AUDIO_START_TOKEN]),

            # Format 2: Just START + text (no end markers)
            ("Format 2: START+text only",
             [START_TOKEN] + text_tokens),

            # Format 3: With EOS between text and audio (from HF docs)
            ("Format 3: START+text+EOS+AUDIO_START",
             [START_TOKEN] + text_tokens + [EOS_TOKEN, TEXT_END_TOKEN, AUDIO_START_TOKEN, AUDIO_END_TOKEN]),

            # Format 4: BOS + START + text + markers
            ("Format 4: BOS+START+text+markers",
             [128000, START_TOKEN] + text_tokens + [TEXT_END_TOKEN, AUDIO_START_TOKEN]),
        ]

        best_result = None
        best_count = 0

        for name, input_tokens in prompts:
            print(f"\n{name}")
            print(f"Input ({len(input_tokens)}): {input_tokens[:10]}...{input_tokens[-3:]}")

            # Reset and eval
            llm.reset()
            llm.eval(input_tokens)

            # Generate with EOS suppression
            generated = []
            for i in range(512):  # Shorter for testing
                # Sample with EOS penalty
                token = llm.sample(
                    top_k=50,
                    top_p=0.95,
                    temp=0.5,  # Higher temp for more generation
                    repeat_penalty=1.0,  # No repeat penalty
                    # Note: llama-cpp-python doesn't easily support logit bias
                )

                # Only stop on actual end tokens or specific markers
                if token == AUDIO_END_TOKEN:
                    print(f"  AUDIO_END at {i}")
                    break
                if token == llm.token_eos() and i > 50:  # Allow some tokens before EOS
                    print(f"  EOS at {i} (after min)")
                    break
                if token in [128001, 128008]:  # Other EOG tokens
                    if i > 50:
                        print(f"  EOG({token}) at {i}")
                        break

                generated.append(token)
                llm.eval([token])

            # Count audio tokens
            audio_tokens = [t for t in generated if t >= AUDIO_TOKEN_START]
            print(f"  Generated: {len(generated)} total, {len(audio_tokens)} audio")
            print(f"  First 10: {generated[:10]}")

            if len(audio_tokens) > best_count:
                best_count = len(audio_tokens)
                best_result = (name, audio_tokens)

        if best_result and best_count >= 100:
            print(f"\n{'='*60}")
            print(f"BEST: {best_result[0]} with {best_count} audio tokens")

            # Decode with SNAC
            decode_with_snac(best_result[1])
        else:
            print(f"\n{'='*60}")
            print(f"Best result only had {best_count} audio tokens")
            print("Model may need transformers (full weights) for proper TTS")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

def decode_with_snac(audio_tokens):
    """Decode audio tokens with SNAC."""
    print("\nDecoding with SNAC...")

    try:
        import snac

        snac_model = snac.SNAC.from_pretrained("hubertsiuzdak/snac_24khz")

        # Convert to codes
        codes = [t - AUDIO_TOKEN_START for t in audio_tokens]
        print(f"Audio codes: {len(codes)}")
        print(f"Code range: {min(codes)} - {max(codes)}")

        # SNAC expects 7 codes per frame (3 layers: 1+2+4)
        num_frames = len(codes) // 7
        if num_frames < 5:
            print(f"Not enough frames: {num_frames}")
            return

        print(f"Frames: {num_frames}")

        # Reshape to layers
        codes_tensor = torch.tensor(codes[:num_frames * 7]).reshape(num_frames, 7)
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
        print(f"Saved: {OUTPUT_WAV}")
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
