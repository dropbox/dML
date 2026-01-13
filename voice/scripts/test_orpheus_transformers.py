#!/usr/bin/env python3
"""
Orpheus-TTS with Full Transformers Model

GGUF Q5_K_M produces gibberish (user feedback: "sounds terrible").
This script tests the FULL model via transformers to see if quality is better.

Per MANAGER instructions in WORKER_INSTRUCTIONS.md.
"""

import sys
import time
import wave
import numpy as np
import torch

from orpheus_snac_utils import (
    AUDIO_END_TOKEN,
    extract_audio_codes,
    snac_tensors_from_interleaved,
)

# Special tokens (from investigation)

OUTPUT_WAV = "/tmp/orpheus_transformers.wav"
MMS_WAV = "/tmp/mms_tts_comparison.wav"

START_TOKEN = 128259
EOS_TOKEN = 128009
TEXT_END_TOKEN = 128260
AUDIO_START_TOKEN = 128261

def main():
    print("Orpheus-TTS Transformers Test")
    print("=" * 60)
    print("GGUF Q5_K_M = GIBBERISH. Testing full model...")
    print()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import snac

        print("Loading Orpheus-TTS full model...")
        print("(This may take a while - downloading ~13GB if not cached)")
        start = time.time()

        tokenizer = AutoTokenizer.from_pretrained("Karayakar/Orpheus-TTS-Turkish-PT-5000")

        model = AutoModelForCausalLM.from_pretrained(
            "Karayakar/Orpheus-TTS-Turkish-PT-5000",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="mps"  # Use Metal GPU
        )

        load_time = time.time() - start
        print(f"Model loaded in {load_time:.1f}s")

        # Check special tokens
        print("\nSpecial tokens:")
        print(f"  special_tokens_map: {tokenizer.special_tokens_map}")

        # Check for custom tokens
        added_tokens = list(tokenizer.added_tokens_encoder.keys())[:10]
        print(f"  added_tokens (first 10): {added_tokens}")

        # Test text
        text = "Merhaba, bugün hava çok güzel."
        print(f"\nTest text: '{text}'")

        # Build prompt with official special tokens
        text_tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
        prompt_tokens = [
            START_TOKEN,
            *text_tokens,
            EOS_TOKEN,
            TEXT_END_TOKEN,
            AUDIO_START_TOKEN,
            AUDIO_END_TOKEN,  # AUDIO_END_TOKEN
        ]
        inputs = torch.tensor([prompt_tokens], device=model.device)
        print(f"Input tokens ({len(prompt_tokens)}): {inputs.tolist()}")

        # Try generating with proper EOS handling
        print("\nGenerating audio tokens...")
        start = time.time()

        # Official params from HuggingFace model card
        outputs = model.generate(
            inputs,
            max_new_tokens=800,  # ~3-5s of audio
            do_sample=True,
            temperature=0.2,  # Low temp per official
            top_k=10,
            top_p=0.9,
            repetition_penalty=1.9,  # Critical for EOS
            eos_token_id=AUDIO_END_TOKEN,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

        gen_time = time.time() - start

        # Extract generated tokens (after input)
        generated = outputs[0, inputs.shape[1]:].tolist()
        audio_tokens = extract_audio_codes(generated)
        print(f"Generated: {len(generated)} total, {len(audio_tokens)} audio codes in {gen_time:.2f}s")
        if generated:
            print(f"First 20 tokens: {generated[:20]}")
            print(f"Max token id: {max(generated)}")
        if audio_tokens:
            print(f"First 14 audio codes: {audio_tokens[:14]}")

        # Check for proper stop
        if AUDIO_END_TOKEN in generated:
            idx = generated.index(AUDIO_END_TOKEN)
            print(f"AUDIO_END found at position {idx}")
        else:
            print("No AUDIO_END found (max_new_tokens reached)")

        # Decode with SNAC
        if len(audio_tokens) >= 49:
            print("\nDecoding with SNAC...")
            snac_model = snac.SNAC.from_pretrained("hubertsiuzdak/snac_24khz")

            try:
                snac_device = next(snac_model.parameters()).device
            except StopIteration:
                snac_device = torch.device("cpu")
            snac_model = snac_model.to(snac_device)
            codes = snac_tensors_from_interleaved(audio_tokens, snac_device)

            with torch.no_grad():
                audio = snac_model.decode(codes)

            audio_np = audio.cpu().numpy().squeeze()
            duration = len(audio_np) / 24000

            print(f"Audio: {len(audio_np)} samples, {duration:.2f}s")
            print(f"Expected: ~0.5-1.5s for 'Merhaba'")

            # Quality metrics
            rms = np.sqrt(np.mean(audio_np ** 2))
            peak = max(abs(audio_np.max()), abs(audio_np.min()))

            # Zero crossing rate (speech indicator)
            sign_changes = np.sum(np.diff(np.sign(audio_np)) != 0)
            zcr = sign_changes / (2 * len(audio_np)) * 24000

            print(f"\nQuality metrics:")
            print(f"  RMS: {rms:.4f}")
            print(f"  Peak: {peak:.4f}")
            print(f"  ZCR: {zcr:.0f} Hz (speech: 500-3000)")

            # Save
            audio_norm = audio_np / (peak + 1e-8) * 0.9
            audio_int16 = (audio_norm * 32767).astype(np.int16)

            with wave.open(OUTPUT_WAV, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(24000)
                wav.writeframes(audio_int16.tobytes())

            print(f"\nSaved: {OUTPUT_WAV}")
            print(f"Play: afplay {OUTPUT_WAV}")

            # Verdict
            print("\n" + "=" * 60)
            if 0.3 <= duration <= 2.0 and 500 <= zcr <= 3000:
                print("VERDICT: Looks like valid speech audio!")
                print("Next step: Human listening evaluation")
            else:
                print("VERDICT: May not be valid speech")
                print(f"  Duration: {duration:.2f}s (expected 0.3-2.0)")
                print(f"  ZCR: {zcr:.0f} Hz (expected 500-3000)")
        else:
            print(f"\nNot enough audio tokens ({len(audio_tokens)}) to decode")
            print("May need different prompt format or model issue")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

        if "out of memory" in str(e).lower():
            print("\nSuggestion: Try device_map='cpu' or reduce max_new_tokens")
        elif "connection" in str(e).lower() or "timeout" in str(e).lower():
            print("\nSuggestion: Model download may have stalled. Try again.")

if __name__ == "__main__":
    main()
