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
Validate CosyVoice2 LLM (Qwen2) weight loading.

Tests:
1. Load llm.pt and inspect structure
2. Load weights into MLX model
3. Compare shapes and statistics
4. Test forward pass
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import mlx.core as mx
import torch

from tools.pytorch_to_mlx.converters.models.cosyvoice2_llm import (
    CosyVoice2LLM,
    Qwen2Config,
)


def find_llm_pt():
    """Find llm.pt file."""
    paths = [
        os.path.expanduser("~/.cache/cosyvoice2/cosyvoice2-0.5b/llm.pt"),
        "./models/cosyvoice2/llm.pt",
    ]
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def inspect_llm_pt(path: str):
    """Inspect llm.pt structure."""
    print(f"\n{'=' * 60}")
    print(f"Inspecting: {path}")
    print(f"{'=' * 60}")

    state_dict = torch.load(path, map_location="cpu", weights_only=False)

    # Count keys by component
    components: dict[str, list[str]] = {}
    for key in state_dict.keys():
        component = key.split(".")[0]
        if component not in components:
            components[component] = []
        components[component].append(key)

    print(f"\nTotal keys: {len(state_dict)}")
    print("\nComponents:")
    for comp, keys in sorted(components.items()):
        total_params = sum(state_dict[k].numel() for k in keys)
        print(f"  {comp}: {len(keys)} keys, {total_params:,} params")

    # Show sample keys
    print("\nSample keys (first 20):")
    for i, key in enumerate(sorted(state_dict.keys())[:20]):
        shape = list(state_dict[key].shape)
        print(f"  {key}: {shape}")

    return state_dict


def test_weight_loading(state_dict: dict):
    """Test weight loading into MLX model."""
    print(f"\n{'=' * 60}")
    print("Testing weight loading")
    print(f"{'=' * 60}")

    # Create MLX model
    config = Qwen2Config()
    model = CosyVoice2LLM(config)

    # Count parameters before loading
    def count_params(params):
        total = 0
        if isinstance(params, dict):
            for v in params.values():
                total += count_params(v)
        elif isinstance(params, mx.array):
            total += params.size
        return total

    params_before = count_params(model.parameters())
    print(f"\nMLX model parameters: {params_before:,}")

    # Load weights
    try:
        model._load_weights(state_dict)
        print("Weight loading: SUCCESS")
    except Exception as e:
        print(f"Weight loading: FAILED - {e}")
        return None

    return model


def validate_shapes(model: CosyVoice2LLM, state_dict: dict):
    """Validate loaded shapes match expected."""
    print(f"\n{'=' * 60}")
    print("Validating shapes")
    print(f"{'=' * 60}")

    checks = [
        ("llm_embedding.weight", model.llm_embedding.weight, (2, 896)),
        ("speech_embedding.weight", model.speech_embedding.weight, (6564, 896)),
        ("llm.embed_tokens", model.llm.embed_tokens.weight, (151936, 896)),
        (
            "llm.layers[0].self_attn.q_proj",
            model.llm.layers[0].self_attn.q_proj.weight,
            (896, 896),
        ),
        (
            "llm.layers[0].self_attn.k_proj",
            model.llm.layers[0].self_attn.k_proj.weight,
            (128, 896),
        ),
        (
            "llm.layers[0].mlp.gate_proj",
            model.llm.layers[0].mlp.gate_proj.weight,
            (4864, 896),
        ),
        ("lm_head", model.lm_head.weight, (151936, 896)),
        ("llm_decoder", model.llm_decoder.weight, (6564, 896)),
    ]

    passed = 0
    for name, weight, expected in checks:
        actual = tuple(weight.shape)
        if actual == expected:
            print(f"  [PASS] {name}: {actual}")
            passed += 1
        else:
            print(f"  [FAIL] {name}: expected {expected}, got {actual}")

    print(f"\nShape validation: {passed}/{len(checks)} passed")
    return passed == len(checks)


def test_forward_pass(model: CosyVoice2LLM):
    """Test forward pass with small input."""
    print(f"\n{'=' * 60}")
    print("Testing forward pass")
    print(f"{'=' * 60}")

    # Small input for testing
    batch, seq_len = 1, 10

    input_ids = mx.random.randint(0, 1000, (batch, seq_len))

    try:
        text_logits, speech_logits, cache = model(input_ids)
        mx.eval(text_logits, speech_logits)

        print(f"\nInput shape: {input_ids.shape}")
        print(f"Text logits shape: {text_logits.shape}")
        print(f"Speech logits shape: {speech_logits.shape}")
        print(f"Cache layers: {len(cache) if cache is not None else 0}")

        # Check output statistics
        text_mean = text_logits.mean().item()
        text_std = mx.sqrt(mx.var(text_logits)).item()
        speech_mean = speech_logits.mean().item()
        speech_std = mx.sqrt(mx.var(speech_logits)).item()

        print(f"\nText logits - mean: {text_mean:.4f}, std: {text_std:.4f}")
        print(f"Speech logits - mean: {speech_mean:.4f}, std: {speech_std:.4f}")

        print("\nForward pass: SUCCESS")
        return True
    except Exception as e:
        print(f"Forward pass: FAILED - {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main validation entry point."""
    print("CosyVoice2 LLM (Qwen2) Validation")
    print("=" * 60)

    # Find llm.pt
    llm_path = find_llm_pt()
    if llm_path is None:
        print("ERROR: llm.pt not found")
        print("Please download CosyVoice2-0.5B model first")
        return 1

    # Inspect structure
    state_dict = inspect_llm_pt(llm_path)

    # Test weight loading
    model = test_weight_loading(state_dict)
    if model is None:
        return 1

    # Validate shapes
    shapes_ok = validate_shapes(model, state_dict)

    # Test forward pass
    forward_ok = test_forward_pass(model)

    # Summary
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    print(f"  Shape validation: {'PASS' if shapes_ok else 'FAIL'}")
    print(f"  Forward pass: {'PASS' if forward_ok else 'FAIL'}")

    return 0 if (shapes_ok and forward_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
