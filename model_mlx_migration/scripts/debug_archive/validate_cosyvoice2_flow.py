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
CosyVoice2 Flow Model Validation Script

Validates the flow model weights can be loaded and run.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """Main validation function."""
    import mlx.core as mx
    import torch

    from tools.pytorch_to_mlx.converters.models.cosyvoice2_flow import (
        DiTDecoder,
        FlowMatchingConfig,
    )

    # Path to flow.pt
    flow_path = Path.home() / ".cache" / "cosyvoice2" / "cosyvoice2-0.5b" / "flow.pt"

    if not flow_path.exists():
        print(f"flow.pt not found at {flow_path}")
        print("Run: python scripts/download_cosyvoice2.py")
        return 1

    print(f"Loading {flow_path}...")
    state_dict = torch.load(flow_path, map_location="cpu", weights_only=False)

    # Count decoder keys
    decoder_keys = [k for k in state_dict.keys() if k.startswith("decoder.estimator")]
    print(f"\nDecoder keys: {len(decoder_keys)}")

    # Check time MLP
    print("\n=== Time MLP ===")
    time_mlp_keys = [k for k in decoder_keys if "time_mlp" in k]
    for k in sorted(time_mlp_keys):
        print(f"  {k}: {state_dict[k].shape}")

    # Check down_blocks
    print("\n=== Down Blocks ===")
    down_keys = [k for k in decoder_keys if "down_blocks.0" in k]
    print(f"  Total keys: {len(down_keys)}")

    # Check block structure
    conv_keys = [
        k
        for k in down_keys
        if ".0.block1" in k or ".0.block2" in k or ".0.mlp" in k or ".0.res_conv" in k
    ]
    attn_keys = [k for k in down_keys if ".1." in k]
    print(f"  Conv block keys: {len(conv_keys)}")
    print(f"  Attention keys: {len(attn_keys)}")

    # Check mid_blocks
    print("\n=== Mid Blocks ===")
    for i in range(12):
        mid_keys = [k for k in decoder_keys if f"mid_blocks.{i}" in k]
        print(f"  mid_blocks.{i}: {len(mid_keys)} keys")

    # Check final
    print("\n=== Final ===")
    final_keys = [k for k in decoder_keys if "final" in k]
    for k in sorted(final_keys):
        print(f"  {k}: {state_dict[k].shape}")

    # Create DiT decoder and test forward pass
    print("\n=== Testing DiT Decoder (Random Weights) ===")
    config = FlowMatchingConfig()
    decoder = DiTDecoder(config)

    # Test inputs
    x = mx.random.normal((1, 50, 80))
    t = mx.array([0.5])
    cond = mx.random.normal((1, 50, 80))

    # Forward pass
    print("Running forward pass with random weights...")
    y = decoder(x, t, cond)
    mx.eval(y)
    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Output range: [{y.min().item():.4f}, {y.max().item():.4f}]")
    print(f"Output std: {y.std().item():.4f}")

    # Test weight loading
    print("\n=== Testing DiT Decoder Weight Loading ===")
    try:
        decoder_loaded = DiTDecoder.from_pretrained(str(flow_path), config)
        print("Weight loading: SUCCESS")

        # Run forward pass with loaded weights
        y_loaded = decoder_loaded(x, t, cond)
        mx.eval(y_loaded)
        print(f"Output (loaded): {y_loaded.shape}")
        print(
            f"Output range: [{y_loaded.min().item():.4f}, {y_loaded.max().item():.4f}]"
        )
        print(f"Output std: {y_loaded.std().item():.4f}")

        # Check some weights were actually loaded
        print("\n=== Weight Verification ===")
        # Check time MLP
        time_w = decoder_loaded.time_mlp.linear_1.weight
        print(
            f"time_mlp.linear_1.weight: shape={time_w.shape}, mean={time_w.mean().item():.6f}"
        )

        # Check down block conv
        down_w = decoder_loaded.down_blocks[0].conv_block.block1_conv.weight
        print(
            f"down_blocks[0].conv.weight: shape={down_w.shape}, mean={down_w.mean().item():.6f}"
        )

        # Check mid block attention
        mid_q = decoder_loaded.mid_blocks[0].attention_blocks[0].to_q.weight
        print(
            f"mid_blocks[0].attn.to_q: shape={mid_q.shape}, mean={mid_q.mean().item():.6f}"
        )

        # Check final proj
        final_w = decoder_loaded.final_proj.weight
        print(
            f"final_proj.weight: shape={final_w.shape}, mean={final_w.mean().item():.6f}"
        )

        weight_loading_pass = True
    except Exception as e:
        print(f"Weight loading: FAILED - {e}")
        import traceback

        traceback.print_exc()
        weight_loading_pass = False

    print("\n=== Summary ===")
    print("DiT decoder forward pass: PASS")
    print(f"Weight loading: {'PASS' if weight_loading_pass else 'FAIL'}")

    if weight_loading_pass:
        print("\nNext steps:")
        print("  1. Compare output against PyTorch reference")
        print("  2. Integrate DiTDecoder into MaskedDiffWithXvec")
        print("  3. Add encoder weight loading")

    return 0 if weight_loading_pass else 1


if __name__ == "__main__":
    sys.exit(main())
