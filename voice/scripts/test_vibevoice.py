#!/usr/bin/env python3
"""Test VibeVoice-1.5B model."""

import os
import sys
import time
import torch

# Use relative paths from script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
VOICE_SAMPLE = os.path.join(PROJECT_DIR, "tests/golden/hello.wav")
MODEL_PATH = os.path.join(PROJECT_DIR, "models/vibevoice/VibeVoice-1.5B")
OUTPUT_PATH = os.path.join(PROJECT_DIR, "models/vibevoice/test_output/test_output.wav")

def main():
    print(f"Testing VibeVoice-1.5B")
    print(f"Model path: {MODEL_PATH}")
    print(f"Voice sample: {VOICE_SAMPLE}")
    print(f"Device: MPS (Apple Silicon)")
    print()

    # Import after path setup
    from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

    # Load processor
    print("Loading processor...")
    processor = VibeVoiceProcessor.from_pretrained(MODEL_PATH)

    # Load model
    print("Loading model...")
    start = time.time()
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,  # MPS requires float32
        attn_implementation="sdpa",
        device_map=None,
    )
    model.to("mps")
    model.eval()
    model.set_ddpm_inference_steps(num_steps=10)
    print(f"Model loaded in {time.time() - start:.2f}s")

    # Prepare input
    test_text = "Speaker 1: This is a test of the VibeVoice text to speech system."

    print(f"\nInput text: {test_text}")
    print(f"Voice sample: {VOICE_SAMPLE}")

    # Process inputs
    inputs = processor(
        text=[test_text],
        voice_samples=[[VOICE_SAMPLE]],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    # Move to device
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to("mps")

    # Generate
    print("\nGenerating speech...")
    start = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=None,
        cfg_scale=1.3,
        tokenizer=processor.tokenizer,
        generation_config={'do_sample': False},
        verbose=True,
    )
    gen_time = time.time() - start

    # Save output
    if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        processor.save_audio(outputs.speech_outputs[0], output_path=OUTPUT_PATH)

        # Calculate metrics
        sample_rate = 24000
        audio_samples = outputs.speech_outputs[0].shape[-1]
        audio_duration = audio_samples / sample_rate
        rtf = gen_time / audio_duration if audio_duration > 0 else float('inf')

        print(f"\n=== Results ===")
        print(f"Generation time: {gen_time:.2f}s")
        print(f"Audio duration: {audio_duration:.2f}s")
        print(f"RTF (Real Time Factor): {rtf:.2f}x")
        print(f"Output saved to: {OUTPUT_PATH}")
    else:
        print("ERROR: No audio output generated")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
