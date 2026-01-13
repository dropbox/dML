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
Create a simple test model for integration testing.

This script creates a minimal PyTorch model and exports it to TorchScript
for use in testing the pytorch_to_mlx converter.

Usage:
    python tests/fixtures/create_test_model.py
"""

from pathlib import Path

import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    """Simple Linear -> ReLU -> Linear model for testing."""

    def __init__(self, input_size=512, hidden_size=256, output_size=128):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        return self.linear2(x)


class TransformerBlockModel(nn.Module):
    """Model with attention for more thorough testing."""

    def __init__(self, d_model=256, nhead=4, dim_ff=512):
        super().__init__()
        self.embedding = nn.Embedding(1000, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, d_model))
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, 1000)

    def forward(self, x):
        # x: (batch, seq_len) token ids
        x = self.embedding(x)
        x = x + self.pos_encoding[:, : x.size(1), :]
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return self.output(x)


def create_simple_model():
    """Create and save simple model."""
    model = SimpleModel()
    model.eval()

    # Export to TorchScript
    example_input = torch.randn(1, 512)
    traced = torch.jit.trace(model, example_input)

    output_path = Path(__file__).parent / "simple_linear.pt"
    traced.save(str(output_path))
    print(f"Created: {output_path}")

    # Verify it works
    output = traced(example_input)
    print(f"Model output shape: {output.shape}")

    return output_path


def create_transformer_model():
    """Create and save transformer model."""
    model = TransformerBlockModel()
    model.eval()

    # Export to TorchScript - use script mode for control flow
    example_input = torch.randint(0, 1000, (1, 64))

    try:
        scripted = torch.jit.script(model)
    except Exception:
        # Fall back to tracing if scripting fails
        scripted = torch.jit.trace(model, example_input)

    output_path = Path(__file__).parent / "transformer_block.pt"
    scripted.save(str(output_path))
    print(f"Created: {output_path}")

    # Verify
    output = scripted(example_input)
    print(f"Model output shape: {output.shape}")

    return output_path


def main():
    """Create all test models."""
    print("Creating test models for integration testing...")
    print()

    paths = []

    # Simple model
    try:
        path = create_simple_model()
        paths.append(path)
    except Exception as e:
        print(f"Failed to create simple model: {e}")

    print()

    # Transformer model
    try:
        path = create_transformer_model()
        paths.append(path)
    except Exception as e:
        print(f"Failed to create transformer model: {e}")

    print()
    print("Summary:")
    print("-" * 40)
    for path in paths:
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  {path.name}: {size_mb:.2f} MB")

    print()
    print("To run integration test:")
    print("  python -m pytorch_to_mlx analyze --input tests/fixtures/simple_linear.pt")
    print(
        "  python -m pytorch_to_mlx convert --input tests/fixtures/simple_linear.pt --output tests/fixtures/simple_linear_mlx/",
    )


if __name__ == "__main__":
    main()
