#!/usr/bin/env python3
# Copyright 2024-2025 Andrew Yates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
CosyVoice3 Full Pipeline RTF Benchmark

Measures REAL end-to-end RTF: Text -> LLM -> DiT Flow -> Vocoder -> Audio

Target: >67.8x RTF (match Kokoro performance)
"""

import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import numpy as np


def load_models():
    """Load all CosyVoice3 components."""
    print("Loading models...")

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "models/cosyvoice3/CosyVoice-BlankEN",
        trust_remote_code=True
    )

    # Load MLX weights
    weights = mx.load('models/cosyvoice3_mlx/model.safetensors')

    # Load LLM model
    from tools.pytorch_to_mlx.converters.models.cosyvoice3_llm import (
        CosyVoice3LLM, create_cosyvoice3_llm_config
    )
    llm_config = create_cosyvoice3_llm_config()
    llm_model = CosyVoice3LLM(llm_config)
    llm_weights = {k[4:]: v for k, v in weights.items() if k.startswith('llm.')}
    llm_model.load_weights(list(llm_weights.items()))

    # Load Flow model
    from tools.pytorch_to_mlx.converters.models.cosyvoice3_dit import (
        CausalMaskedDiffWithDiT, create_cosyvoice3_flow_config
    )
    flow_config = create_cosyvoice3_flow_config()
    flow_model = CausalMaskedDiffWithDiT(flow_config)
    flow_weights = {k[5:]: v for k, v in weights.items() if k.startswith('flow.')}
    flow_model.load_weights(list(flow_weights.items()))

    # Load Vocoder model
    from tools.pytorch_to_mlx.converters.models.cosyvoice3_vocoder import (
        CausalHiFTGenerator, create_cosyvoice3_vocoder_config
    )
    vocoder_config = create_cosyvoice3_vocoder_config()
    vocoder_model = CausalHiFTGenerator(vocoder_config)
    vocoder_weights = {k[8:]: v for k, v in weights.items() if k.startswith('vocoder.')}
    vocoder_model.load_weights(list(vocoder_weights.items()))

    return tokenizer, llm_model, flow_model, vocoder_model


def synthesize_text(tokenizer, llm_model, flow_model, vocoder_model, text,
                    max_tokens=100, num_steps=10):
    """Run full pipeline: text -> audio."""
    # Tokenize
    tokens = tokenizer(text, return_tensors="pt")
    input_ids = mx.array(tokens["input_ids"].numpy())

    # LLM: text -> speech tokens
    speech_tokens = llm_model.generate_speech_tokens(
        input_ids,
        max_length=max_tokens,
        temperature=0.8,
        top_k=25,
    )
    mx.eval(speech_tokens)

    # Flow: speech tokens -> mel
    spk_emb = mx.zeros((1, 192))  # Use zero embedding for benchmark
    mel = flow_model.inference(speech_tokens, spk_emb, num_steps=num_steps)
    mx.eval(mel)

    # Vocoder: mel -> audio
    mel_for_vocoder = mel.transpose(0, 2, 1)
    audio = vocoder_model(mel_for_vocoder)
    mx.eval(audio)

    return audio


def benchmark():
    """Run comprehensive benchmark."""
    print("=" * 60)
    print("CosyVoice3 Full Pipeline RTF Benchmark")
    print("=" * 60)

    try:
        tokenizer, llm_model, flow_model, vocoder_model = load_models()
    except Exception as e:
        print(f"ERROR loading models: {e}")
        print("\nModels may not be converted yet. Check:")
        print("  - models/cosyvoice3_mlx/model.safetensors")
        print("  - models/cosyvoice3/CosyVoice-BlankEN (tokenizer)")
        return

    print("\nModels loaded successfully!")
    print(f"  LLM: {sum(p.size for p in llm_model.parameters().values()) / 1e6:.1f}M params")
    print(f"  Flow: {sum(p.size for p in flow_model.parameters().values()) / 1e6:.1f}M params")
    print(f"  Vocoder: {sum(p.size for p in vocoder_model.parameters().values()) / 1e6:.1f}M params")

    # Warm up
    print("\nWarming up...")
    _ = synthesize_text(tokenizer, llm_model, flow_model, vocoder_model, "Hello", max_tokens=10, num_steps=3)

    # Benchmark with different text lengths
    test_cases = [
        ("Short", "Hello world"),
        ("Medium", "This is a comprehensive test of the CosyVoice three text to speech synthesis system."),
        ("Long", "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet. Text to speech systems must handle various phonemes and prosodic patterns correctly."),
    ]

    print("\n" + "=" * 60)
    print("Benchmark Results")
    print("=" * 60)

    results = []
    for name, text in test_cases:
        print(f"\n{name}: '{text[:50]}...'")

        # Multiple runs for stability
        rtfs = []
        for i in range(3):
            start = time.perf_counter()
            audio = synthesize_text(tokenizer, llm_model, flow_model, vocoder_model, text, max_tokens=100, num_steps=10)
            elapsed = time.perf_counter() - start

            audio_samples = audio.shape[1] if len(audio.shape) > 1 else len(audio)
            audio_duration = audio_samples / 24000
            rtf = audio_duration / elapsed
            rtfs.append(rtf)

        avg_rtf = np.mean(rtfs)
        std_rtf = np.std(rtfs)

        print(f"  Audio: {audio_duration:.2f}s, Time: {elapsed:.3f}s")
        print(f"  RTF: {avg_rtf:.1f}x +/- {std_rtf:.1f}x (runs: {rtfs})")

        results.append({
            'name': name,
            'text_len': len(text),
            'audio_duration': audio_duration,
            'rtf_mean': avg_rtf,
            'rtf_std': std_rtf,
        })

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    overall_rtf = np.mean([r['rtf_mean'] for r in results])
    print(f"Overall Average RTF: {overall_rtf:.1f}x")
    print("Target RTF: >67.8x (Kokoro benchmark)")

    if overall_rtf >= 67.8:
        print("\nSTATUS: PASS - Meets or exceeds Kokoro performance!")
    elif overall_rtf >= 50:
        print(f"\nSTATUS: CLOSE - {67.8 - overall_rtf:.1f}x short of target")
        print("Recommendation: Apply remaining optimizations")
    else:
        print(f"\nSTATUS: NEEDS OPTIMIZATION - {67.8 - overall_rtf:.1f}x short of target")
        print("Action: Profile with Instruments to find bottleneck")
        print("Components to check:")
        print("  1. LLM token generation (likely bottleneck for short text)")
        print("  2. DiT flow steps (likely bottleneck for long audio)")
        print("  3. Vocoder mel->audio")

    return results


if __name__ == "__main__":
    benchmark()
