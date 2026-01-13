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
CosyVoice3 End-to-End RTF Benchmark

Measures Real-Time Factor (RTF) for the complete pipeline:
Text -> Tokenizer -> LLM -> Speech Tokens -> DiT Flow -> Mel -> Vocoder -> Audio

RTF = audio_duration / generation_time
RTF > 1 means faster than real-time
"""

import sys
from pathlib import Path
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx


def benchmark_cosyvoice3():
    """Run end-to-end CosyVoice3 benchmark."""
    print("=" * 70)
    print("CosyVoice3 End-to-End RTF Benchmark")
    print("=" * 70)

    # =========================================================================
    # 1. Load Models
    # =========================================================================
    print("\n1. Loading models...")
    load_start = time.time()

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "models/cosyvoice3/CosyVoice-BlankEN",
        trust_remote_code=True
    )
    print("   Tokenizer loaded")

    # Load MLX weights
    weights = mx.load('models/cosyvoice3_mlx/model.safetensors')
    print(f"   Weights loaded: {len(weights)} tensors")

    # Create LLM
    from tools.pytorch_to_mlx.converters.models.cosyvoice2_llm import (
        CosyVoice2LLM, Qwen2Config
    )
    llm_config = Qwen2Config(
        hidden_size=896,
        num_hidden_layers=24,
        num_attention_heads=7,
        num_key_value_heads=1,
        head_dim=128,
        intermediate_size=4864,
        vocab_size=151936,
        speech_vocab_size=6564,
        rope_theta=1000000.0,
    )
    llm = CosyVoice2LLM(llm_config)
    llm_weights = {k[4:]: v for k, v in weights.items() if k.startswith('llm.')}
    llm.load_weights(list(llm_weights.items()))
    mx.eval(llm.parameters())
    print(f"   LLM loaded: {len(llm_weights)} tensors")

    # Create Flow
    from tools.pytorch_to_mlx.converters.models.cosyvoice3_dit import (
        CausalMaskedDiffWithDiT, create_cosyvoice3_flow_config
    )
    flow_config = create_cosyvoice3_flow_config()
    flow = CausalMaskedDiffWithDiT(flow_config)
    flow_weights = {k[5:]: v for k, v in weights.items() if k.startswith('flow.')}
    flow.load_weights(list(flow_weights.items()))
    mx.eval(flow.parameters())
    print(f"   Flow loaded: {len(flow_weights)} tensors")

    # Create Vocoder
    from tools.pytorch_to_mlx.converters.models.cosyvoice3_vocoder import (
        CausalHiFTGenerator, create_cosyvoice3_vocoder_config
    )
    vocoder_config = create_cosyvoice3_vocoder_config()
    vocoder = CausalHiFTGenerator(vocoder_config)
    vocoder_weights = {k[8:]: v for k, v in weights.items() if k.startswith('vocoder.')}
    vocoder.load_weights(list(vocoder_weights.items()))
    mx.eval(vocoder.parameters())
    print(f"   Vocoder loaded: {len(vocoder_weights)} tensors")

    load_time = time.time() - load_start
    print(f"   Total load time: {load_time:.2f}s")

    # =========================================================================
    # 2. Warmup
    # =========================================================================
    print("\n2. Warming up...")
    warmup_text = "Hello"
    warmup_tokens = tokenizer(warmup_text, return_tensors="pt")
    warmup_ids = mx.array(warmup_tokens["input_ids"].numpy())

    # Warmup LLM with RAS method
    _ = llm.generate_speech_tokens_ras(warmup_ids, max_length=20, top_k=25, top_p=0.8)
    mx.eval(_)

    # Warmup flow with dummy tokens
    dummy_spk = mx.random.normal((1, 192))
    dummy_tokens = mx.zeros((1, 10), dtype=mx.int32)
    _ = flow.inference(dummy_tokens, dummy_spk, num_steps=3)
    mx.eval(_)

    # Warmup vocoder with dummy mel
    dummy_mel = mx.random.normal((1, 80, 20))
    _ = vocoder(dummy_mel)
    mx.eval(_)

    print("   Warmup complete")

    # =========================================================================
    # 3. Benchmark Test Cases
    # =========================================================================
    print("\n3. Running benchmarks...")

    test_texts = [
        ("Short", "Hello world."),
        ("Medium", "This is a test of the CosyVoice three text to speech synthesis system."),
        ("Long", "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet and is commonly used for testing purposes. Let me add even more text to make this a longer test case."),
    ]

    results = []
    sample_rate = 24000  # CosyVoice3 sample rate

    for name, text in test_texts:
        print(f"\n   Test: {name}")
        print(f"   Text: {text[:50]}..." if len(text) > 50 else f"   Text: {text}")

        # Tokenize
        tokens = tokenizer(text, return_tensors="pt")
        input_ids = mx.array(tokens["input_ids"].numpy())
        text_len = input_ids.shape[1]
        print(f"   Text tokens: {text_len}")

        # Random speaker embedding
        spk_emb = mx.random.normal((1, 192))

        # Full pipeline benchmark
        start_time = time.time()

        # Step 1: LLM - Generate speech tokens using RAS (correct method)
        llm_start = time.time()
        speech_tokens = llm.generate_speech_tokens_ras(
            input_ids,
            max_length=500,  # Allow up to 500 tokens
            top_k=25,
            top_p=0.8,
            win_size=10,
            tau_r=0.1,
            speech_token_size=6561,  # Stop tokens >= 6561
            min_token_text_ratio=2.0,
            max_token_text_ratio=20.0,
        )
        mx.eval(speech_tokens)
        llm_time = time.time() - llm_start
        num_speech_tokens = speech_tokens.shape[1]
        print(f"   Speech tokens: {num_speech_tokens}")
        print(f"   LLM time: {llm_time:.3f}s")

        # Step 2: Flow - Generate mel spectrogram
        flow_start = time.time()
        mel = flow.inference(speech_tokens, spk_emb, num_steps=10)
        mx.eval(mel)
        flow_time = time.time() - flow_start
        mel_frames = mel.shape[1]
        print(f"   Mel frames: {mel_frames}")
        print(f"   Flow time: {flow_time:.3f}s")

        # Step 3: Vocoder - Generate audio
        vocoder_start = time.time()
        mel_for_vocoder = mel.transpose(0, 2, 1)  # [B, L, C] -> [B, C, L]
        audio = vocoder(mel_for_vocoder)
        mx.eval(audio)
        vocoder_time = time.time() - vocoder_start
        audio_samples = audio.shape[1]
        print(f"   Audio samples: {audio_samples}")
        print(f"   Vocoder time: {vocoder_time:.3f}s")

        total_time = time.time() - start_time

        # Calculate metrics
        audio_duration = audio_samples / sample_rate
        rtf = audio_duration / total_time

        print("\n   Results:")
        print(f"   - Audio duration: {audio_duration:.2f}s")
        print(f"   - Generation time: {total_time:.3f}s")
        print(f"   - RTF: {rtf:.1f}x real-time")
        print(f"   - Breakdown: LLM {llm_time:.3f}s ({llm_time/total_time*100:.1f}%), "
              f"Flow {flow_time:.3f}s ({flow_time/total_time*100:.1f}%), "
              f"Vocoder {vocoder_time:.3f}s ({vocoder_time/total_time*100:.1f}%)")

        results.append({
            'name': name,
            'text': text,
            'text_tokens': text_len,
            'speech_tokens': num_speech_tokens,
            'mel_frames': mel_frames,
            'audio_samples': audio_samples,
            'audio_duration': audio_duration,
            'total_time': total_time,
            'llm_time': llm_time,
            'flow_time': flow_time,
            'vocoder_time': vocoder_time,
            'rtf': rtf,
        })

    # =========================================================================
    # 4. Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    avg_rtf = np.mean([r['rtf'] for r in results])
    min_rtf = min(r['rtf'] for r in results)
    max_rtf = max(r['rtf'] for r in results)

    print("\nRTF Summary:")
    print(f"  Average RTF: {avg_rtf:.1f}x")
    print(f"  Min RTF: {min_rtf:.1f}x")
    print(f"  Max RTF: {max_rtf:.1f}x")

    print("\nTime Breakdown (average):")
    avg_llm = np.mean([r['llm_time'] for r in results])
    avg_flow = np.mean([r['flow_time'] for r in results])
    avg_vocoder = np.mean([r['vocoder_time'] for r in results])
    avg_total = np.mean([r['total_time'] for r in results])

    print(f"  LLM: {avg_llm:.3f}s ({avg_llm/avg_total*100:.1f}%)")
    print(f"  Flow: {avg_flow:.3f}s ({avg_flow/avg_total*100:.1f}%)")
    print(f"  Vocoder: {avg_vocoder:.3f}s ({avg_vocoder/avg_total*100:.1f}%)")

    print("\nTarget: >67.8x RTF (to exceed Kokoro)")
    if avg_rtf >= 67.8:
        print("STATUS: PASS - CosyVoice3 exceeds target RTF")
    else:
        print(f"STATUS: NEEDS OPTIMIZATION - {67.8/avg_rtf:.1f}x improvement needed")
        bottleneck = max([(avg_llm, 'LLM'), (avg_flow, 'Flow'), (avg_vocoder, 'Vocoder')], key=lambda x: x[0])
        print(f"BOTTLENECK: {bottleneck[1]} ({bottleneck[0]:.3f}s, {bottleneck[0]/avg_total*100:.1f}% of total)")

    return results


if __name__ == "__main__":
    results = benchmark_cosyvoice3()
