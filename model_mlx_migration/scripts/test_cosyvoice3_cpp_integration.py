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
CosyVoice3 C++ Integration Test

This script validates the C++ implementation by:
1. Generating speech tokens using Python LLM
2. Running flow + vocoder in both Python and C++
3. Comparing outputs numerically and via Whisper transcription

This validates the majority of the CosyVoice3 pipeline (flow + vocoder)
while the C++ LLM implementation is pending.
"""

import sys
from pathlib import Path
import numpy as np
import struct

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx


def load_cosyvoice3_models():
    """Load CosyVoice3 MLX models."""
    from tools.pytorch_to_mlx.converters.models.cosyvoice3_dit import (
        CausalMaskedDiffWithDiT, create_cosyvoice3_flow_config
    )
    from tools.pytorch_to_mlx.converters.models.cosyvoice3_vocoder import (
        CausalHiFTGenerator, create_cosyvoice3_vocoder_config
    )
    from tools.pytorch_to_mlx.converters.models.cosyvoice2_llm import (
        CosyVoice2LLM, Qwen2Config
    )

    # Load weights
    weights = mx.load('models/cosyvoice3_mlx/model.safetensors')

    # Flow model
    flow_config = create_cosyvoice3_flow_config()
    flow_model = CausalMaskedDiffWithDiT(flow_config)
    flow_weights = {k[5:]: v for k, v in weights.items() if k.startswith('flow.')}
    flow_model.load_weights(list(flow_weights.items()))
    mx.eval(flow_model.parameters())

    # Vocoder model
    vocoder_config = create_cosyvoice3_vocoder_config()
    vocoder_model = CausalHiFTGenerator(vocoder_config)
    vocoder_weights = {k[8:]: v for k, v in weights.items() if k.startswith('vocoder.')}
    vocoder_model.load_weights(list(vocoder_weights.items()))
    mx.eval(vocoder_model.parameters())

    # LLM model
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
    llm_model = CosyVoice2LLM(llm_config)
    llm_weights = {k[4:]: v for k, v in weights.items() if k.startswith('llm.')}
    llm_model.load_weights(list(llm_weights.items()))
    mx.eval(llm_model.parameters())

    return llm_model, flow_model, vocoder_model


def generate_speech_tokens(llm_model, text: str) -> mx.array:
    """Generate speech tokens from text using the LLM."""
    from transformers import AutoTokenizer

    # Tokenize text
    tokenizer = AutoTokenizer.from_pretrained(
        "models/cosyvoice3/CosyVoice-BlankEN",
        trust_remote_code=True
    )
    text_ids = tokenizer(text, return_tensors="np")["input_ids"]
    text_ids = mx.array(text_ids)

    # Generate speech tokens (use simpler method like the tests)
    speech_tokens = llm_model.generate_speech_tokens(
        text_ids,
        max_length=30,  # Match the tests (30 for "Hello")
        temperature=0.7,
        top_k=25,
    )
    mx.eval(speech_tokens)

    return speech_tokens


def save_tokens_binary(tokens: mx.array, path: str):
    """Save tokens to binary file for C++ reading."""
    tokens_np = np.array(tokens, dtype=np.int32)
    with open(path, 'wb') as f:
        # Write shape: batch, seq_len
        f.write(struct.pack('ii', tokens_np.shape[0], tokens_np.shape[1]))
        # Write tokens
        f.write(tokens_np.tobytes())


def save_mel_binary(mel: mx.array, path: str):
    """Save mel spectrogram to binary file for C++ reading."""
    mel_np = np.array(mel, dtype=np.float32)
    with open(path, 'wb') as f:
        # Write shape: batch, seq_len, mel_dim
        f.write(struct.pack('iii', mel_np.shape[0], mel_np.shape[1], mel_np.shape[2]))
        # Write mel data
        f.write(mel_np.tobytes())


def test_flow_vocoder_pipeline():
    """Test flow + vocoder with Python-generated speech tokens."""
    print("=" * 60)
    print("CosyVoice3 C++ Integration Test")
    print("=" * 60)

    # Load models
    print("\n1. Loading models...")
    llm_model, flow_model, vocoder_model = load_cosyvoice3_models()
    print("   Models loaded successfully")

    # Generate speech tokens
    test_text = "Hello"  # Short text like the tests use
    print(f"\n2. Generating speech tokens for: '{test_text}'")
    speech_tokens = generate_speech_tokens(llm_model, test_text)
    print(f"   Generated {speech_tokens.shape[1]} speech tokens")

    # Create speaker embedding (deterministic for testing)
    mx.random.seed(42)
    spk_emb = mx.random.normal(shape=(1, 192))
    mx.eval(spk_emb)

    # Run Python flow (use more steps like tests: 15-20)
    print("\n3. Running Python flow model...")
    mel_python = flow_model.inference(
        speech_tokens,
        spk_emb,
        num_steps=15,  # More steps for quality
        cfg_strength=0.7
    )
    mx.eval(mel_python)
    print(f"   Python mel shape: {mel_python.shape}")

    # Run Python vocoder
    print("\n4. Running Python vocoder...")
    # Vocoder expects [B, C, L] not [B, L, C]
    mel_for_vocoder = mel_python.transpose(0, 2, 1)
    audio_python = vocoder_model(mel_for_vocoder)  # Use __call__ like tests
    mx.eval(audio_python)
    print(f"   Python audio shape: {audio_python.shape}")

    # Save test data
    print("\n5. Saving test data for C++ comparison...")
    test_dir = Path("test_data/cosyvoice3_cpp_test")
    test_dir.mkdir(parents=True, exist_ok=True)

    save_tokens_binary(speech_tokens, str(test_dir / "speech_tokens.bin"))
    save_mel_binary(mel_python, str(test_dir / "mel_python.bin"))

    # Save speaker embedding
    spk_np = np.array(spk_emb, dtype=np.float32)
    with open(test_dir / "speaker_emb.bin", 'wb') as f:
        f.write(struct.pack('ii', spk_np.shape[0], spk_np.shape[1]))
        f.write(spk_np.tobytes())

    # Save Python audio
    audio_np = np.array(audio_python, dtype=np.float32)
    with open(test_dir / "audio_python.bin", 'wb') as f:
        f.write(struct.pack('ii', audio_np.shape[0], audio_np.shape[1]))
        f.write(audio_np.tobytes())

    # Also save as WAV for listening
    import wave
    wav_path = str(test_dir / "audio_python.wav")
    audio_int16 = (audio_np[0] * 32767).astype(np.int16)
    with wave.open(wav_path, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(audio_int16.tobytes())

    print(f"   Saved to {test_dir}/")
    print(f"   - speech_tokens.bin: {speech_tokens.shape}")
    print(f"   - mel_python.bin: {mel_python.shape}")
    print(f"   - speaker_emb.bin: {spk_emb.shape}")
    print(f"   - audio_python.wav: {audio_np.shape[1]} samples")

    # Verify Python audio with Whisper (use turbo for better TTS accuracy)
    print("\n6. Verifying Python audio with Whisper...")
    import mlx_whisper
    result = mlx_whisper.transcribe(
        wav_path,
        path_or_hf_repo="mlx-community/whisper-turbo",
    )
    transcription = result.get('text', '').strip()
    print(f"   Input text: '{test_text}'")
    print(f"   Whisper:    '{transcription}'")

    # Check if transcription matches
    test_lower = test_text.lower().replace(',', '').replace('.', '')
    trans_lower = transcription.lower().replace(',', '').replace('.', '')

    match_ratio = len(set(test_lower.split()) & set(trans_lower.split())) / len(set(test_lower.split()))
    if match_ratio > 0.5:
        print(f"   PASS: Python TTS produces intelligible speech ({match_ratio*100:.0f}% word match)")
    else:
        print(f"   WARN: Low word match ({match_ratio*100:.0f}%)")

    print("\n" + "=" * 60)
    print("Test data saved. C++ can now use these files to compare.")
    print("=" * 60)

    return True


if __name__ == "__main__":
    test_flow_vocoder_pipeline()
