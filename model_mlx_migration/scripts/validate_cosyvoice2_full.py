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
Validate full CosyVoice2 model loading and inference.

Tests:
1. Load all components (llm.pt, flow.pt, hift.pt)
2. Count parameters
3. Test speech token generation
4. Test mel generation (if flow model works)
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import mlx.core as mx


def find_model_dir():
    """Find CosyVoice2 model directory."""
    paths = [
        Path.home() / ".cache" / "cosyvoice2" / "cosyvoice2-0.5b",
        Path("./models/cosyvoice2"),
    ]
    for path in paths:
        if path.exists():
            return path
    return None


def count_parameters(model):
    """Count parameters in model."""
    total = 0

    def count_params(params):
        nonlocal total
        if isinstance(params, dict):
            for v in params.values():
                count_params(v)
        elif isinstance(params, mx.array):
            total += params.size

    count_params(model.parameters())
    return total


def validate_llm_loading(model_dir: Path):
    """Validate LLM component loading."""
    print("\n" + "=" * 60)
    print("Validating LLM Component")
    print("=" * 60)

    from tools.pytorch_to_mlx.converters.models.cosyvoice2_llm import (
        CosyVoice2LLM,
        Qwen2Config,
    )

    llm_path = model_dir / "llm.pt"
    if not llm_path.exists():
        print(f"[SKIP] llm.pt not found at {llm_path}")
        return None

    print(f"Loading from: {llm_path}")
    config = Qwen2Config()
    llm = CosyVoice2LLM.from_pretrained(str(llm_path), config)

    params = count_parameters(llm)
    print(f"Parameters: {params:,}")

    # Test forward pass
    input_ids = mx.random.randint(0, 1000, (1, 8))
    text_logits, speech_logits, cache = llm(input_ids)
    mx.eval(text_logits, speech_logits)

    print(f"Text logits shape: {text_logits.shape}")
    print(f"Speech logits shape: {speech_logits.shape}")
    print("[PASS] LLM component loaded and runs")

    return llm


def validate_vocoder_loading(model_dir: Path):
    """Validate vocoder component loading."""
    print("\n" + "=" * 60)
    print("Validating Vocoder Component")
    print("=" * 60)

    from tools.pytorch_to_mlx.converters.models.cosyvoice2_vocoder import (
        HiFiGANConfig,
        HiFiGANVocoder,
    )

    vocoder_path = model_dir / "hift.pt"
    if not vocoder_path.exists():
        print(f"[SKIP] hift.pt not found at {vocoder_path}")
        return None

    print(f"Loading from: {vocoder_path}")
    config = HiFiGANConfig()
    vocoder = HiFiGANVocoder.from_pretrained(str(vocoder_path), config)

    params = count_parameters(vocoder)
    print(f"Parameters: {params:,}")

    # Test forward pass with random mel
    mel = mx.random.normal((1, 50, 80))  # [batch, mel_len, mel_dim]
    audio = vocoder(mel)
    mx.eval(audio)

    print(f"Input mel shape: {mel.shape}")
    print(f"Output audio shape: {audio.shape}")
    print("[PASS] Vocoder component loaded and runs")

    return vocoder


def validate_full_model(model_dir: Path):
    """Validate full model loading."""
    print("\n" + "=" * 60)
    print("Validating Full CosyVoice2 Model")
    print("=" * 60)

    from tools.pytorch_to_mlx.converters.models.cosyvoice2 import (
        CosyVoice2Config,
        CosyVoice2Model,
        count_parameters,
    )

    print(f"Loading from: {model_dir}")

    config = CosyVoice2Config()
    model = CosyVoice2Model.from_pretrained(model_dir, config)

    # Count parameters
    llm_params = count_parameters(model.llm)
    flow_params = count_parameters(model.flow)
    vocoder_params = count_parameters(model.vocoder)
    total_params = llm_params + flow_params + vocoder_params

    print("\nParameter Counts:")
    print(f"  LLM:     {llm_params:>12,}")
    print(f"  Flow:    {flow_params:>12,}")
    print(f"  Vocoder: {vocoder_params:>12,}")
    print(f"  Total:   {total_params:>12,}")

    # Test speech token generation
    print("\n--- Testing Speech Token Generation ---")
    text_ids = mx.random.randint(0, 1000, (1, 10))
    tokens = model.generate_speech_tokens(
        text_ids,
        max_length=5,
        temperature=1.0,
        top_k=25,
    )
    mx.eval(tokens)
    print(f"Generated {tokens.shape[1]} tokens: {tokens[0].tolist()}")

    print("\n[PASS] Full model loaded and runs")
    return model


def main():
    """Main validation entry point."""
    print("CosyVoice2 Full Model Validation")
    print("=" * 60)

    model_dir = find_model_dir()
    if model_dir is None:
        print("ERROR: CosyVoice2 model directory not found")
        print("Please run: python scripts/download_cosyvoice2.py")
        return 1

    print(f"Model directory: {model_dir}")

    # Test individual components
    llm = validate_llm_loading(model_dir)
    vocoder = validate_vocoder_loading(model_dir)

    # Test full model
    model = validate_full_model(model_dir)

    # Summary
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)
    print(f"  LLM:      {'PASS' if llm else 'SKIP'}")
    print(f"  Vocoder:  {'PASS' if vocoder else 'SKIP'}")
    print(f"  Full:     {'PASS' if model else 'FAIL'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
