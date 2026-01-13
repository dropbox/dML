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
Weight conversion utilities for PyTorch -> MLX Zipformer models.

Converts pretrained icefall Zipformer weights to MLX format.
"""

import argparse
from typing import Any

import mlx.core as mx
import numpy as np
import torch


def analyze_checkpoint(checkpoint_path: str) -> dict[str, Any]:
    """
    Analyze a PyTorch checkpoint and return structure information.

    Args:
        checkpoint_path: Path to the PyTorch checkpoint.

    Returns:
        Dictionary with analysis results.
    """
    state_dict = torch.load(checkpoint_path, map_location='cpu')

    if 'model' in state_dict:
        model_dict = state_dict['model']
    else:
        model_dict = state_dict

    # Group keys by component
    analysis = {
        'total_keys': len(model_dict),
        'encoder_keys': 0,
        'decoder_keys': 0,
        'joiner_keys': 0,
        'other_keys': 0,
        'encoder_stages': {},
    }

    for key in model_dict.keys():
        if key.startswith('encoder.'):
            analysis['encoder_keys'] += 1

            # Analyze encoder stages
            if 'encoders.' in key:
                parts = key.split('.')
                idx = parts.index('encoders') + 1
                stage = parts[idx]
                if stage not in analysis['encoder_stages']:
                    analysis['encoder_stages'][stage] = {'layers': set(), 'keys': 0}
                analysis['encoder_stages'][stage]['keys'] += 1

                # Find layer number
                if 'layers.' in key:
                    layer_idx = parts.index('layers') + 1
                    analysis['encoder_stages'][stage]['layers'].add(int(parts[layer_idx]))

        elif key.startswith('decoder.'):
            analysis['decoder_keys'] += 1
        elif key.startswith('joiner.'):
            analysis['joiner_keys'] += 1
        else:
            analysis['other_keys'] += 1

    # Convert layer sets to lists
    for stage in analysis['encoder_stages']:
        analysis['encoder_stages'][stage]['layers'] = sorted(
            analysis['encoder_stages'][stage]['layers'],
        )

    return analysis


def extract_encoder_config(checkpoint_path: str) -> dict[str, Any]:
    """
    Extract encoder configuration from checkpoint weights.

    Args:
        checkpoint_path: Path to the PyTorch checkpoint.

    Returns:
        Configuration dictionary.
    """
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = state_dict.get('model', state_dict)

    config = {}

    # Get embed dimension from encoder_embed.out
    if 'encoder.encoder_embed.out.weight' in model_dict:
        weight = model_dict['encoder.encoder_embed.out.weight']
        config['d_model'] = weight.shape[0]

    # Get attention dimension from first layer's self_attn
    for key in model_dict:
        if 'encoders.0.layers.0.self_attn.in_proj.weight' in key:
            weight = model_dict[key]
            # in_proj projects to: 2*attention_dim + attention_dim//2 + pos_dim*num_heads
            config['in_proj_dim'] = weight.shape[0]
            break

    # Get kernel size from conv module
    for key in model_dict:
        if 'conv_module1.depthwise_conv.weight' in key:
            weight = model_dict[key]
            config['kernel_size'] = weight.shape[2]
            break

    # Get number of stages and layers
    stages = set()
    stage_layers = {}
    for key in model_dict:
        if 'encoder.encoders.' in key and '.layers.' in key:
            parts = key.split('.')
            stage_idx = int(parts[parts.index('encoders') + 1])
            layer_idx = int(parts[parts.index('layers') + 1])
            stages.add(stage_idx)
            if stage_idx not in stage_layers:
                stage_layers[stage_idx] = set()
            stage_layers[stage_idx].add(layer_idx)

    config['num_stages'] = len(stages)
    config['layers_per_stage'] = {
        s: len(layers) for s, layers in stage_layers.items()
    }

    return config


def convert_conv1d_weight(weight: torch.Tensor) -> mx.array:
    """
    Convert PyTorch conv1d weight to MLX format.

    PyTorch: (C_out, C_in, K)
    MLX: (C_out, K, C_in)
    """
    # Transpose from (C_out, C_in, K) to (C_out, K, C_in)
    weight_np = weight.numpy()
    weight_np = np.transpose(weight_np, (0, 2, 1))
    return mx.array(weight_np)


def convert_conv2d_weight(weight: torch.Tensor) -> mx.array:
    """
    Convert PyTorch conv2d weight to MLX format.

    PyTorch: (C_out, C_in, H, W) - OIHW (NCHW convention)
    MLX: (C_out, H, W, C_in) - OHWI (NHWC convention)
    """
    # Transpose from (C_out, C_in, H, W) to (C_out, H, W, C_in)
    weight_np = weight.numpy()
    weight_np = np.transpose(weight_np, (0, 2, 3, 1))
    return mx.array(weight_np)


# Backward compatibility alias
convert_conv_weight = convert_conv1d_weight


def convert_linear_weight(weight: torch.Tensor) -> mx.array:
    """
    Convert PyTorch linear weight to MLX format.

    PyTorch: (out_features, in_features)
    MLX: (out_features, in_features) - same!
    """
    return mx.array(weight.numpy())


def convert_encoder_weights(
    checkpoint_path: str,
    output_path: str | None = None,
) -> dict[str, mx.array]:
    """
    Convert encoder weights from PyTorch checkpoint to MLX format.

    Args:
        checkpoint_path: Path to the PyTorch checkpoint.
        output_path: Optional path to save converted weights.

    Returns:
        Dictionary of MLX arrays.
    """
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = state_dict.get('model', state_dict)

    mlx_weights = {}
    skipped_keys = []

    for key, tensor in model_dict.items():
        # Skip non-encoder weights for now
        if not key.startswith('encoder.'):
            continue

        # Skip balancer counts (training statistics)
        if 'balancer.count' in key or 'deriv_balancer' in key:
            skipped_keys.append(key)
            continue

        # Skip BasicNorm eps
        if 'norm_final.eps' in key:
            skipped_keys.append(key)
            continue

        # Convert based on weight type
        if '.weight' in key:
            # Check if conv weight (3D with shape (C, C_in, K))
            if tensor.ndim == 3 and 'conv' in key.lower():
                mlx_weights[key] = convert_conv_weight(tensor)
            else:
                mlx_weights[key] = convert_linear_weight(tensor)
        elif '.bias' in key:
            mlx_weights[key] = mx.array(tensor.numpy())
        else:
            # Scalar parameters like bypass_scale
            if tensor.ndim == 0:
                mlx_weights[key] = mx.array(tensor.item())
            else:
                mlx_weights[key] = mx.array(tensor.numpy())

    print(f"Converted {len(mlx_weights)} weights")
    print(f"Skipped {len(skipped_keys)} keys (training stats)")

    if output_path:
        mx.savez(output_path, **mlx_weights)
        print(f"Saved to {output_path}")

    return mlx_weights


def convert_full_model(
    checkpoint_path: str,
    output_path: str | None = None,
) -> dict[str, mx.array]:
    """
    Convert all model weights from PyTorch checkpoint to MLX format.

    Includes encoder, decoder, and joiner weights.

    Args:
        checkpoint_path: Path to the PyTorch checkpoint.
        output_path: Optional path to save converted weights.

    Returns:
        Dictionary of MLX arrays.
    """
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = state_dict.get('model', state_dict)

    mlx_weights = {}
    skipped_keys = []

    for key, tensor in model_dict.items():
        # Skip training statistics
        if '.count' in key or 'deriv_balancer' in key:
            skipped_keys.append(key)
            continue

        if '.eps' in key:
            skipped_keys.append(key)
            continue

        # Convert based on weight type
        if '.weight' in key:
            # Check if conv2d weight (4D with spatial dimensions)
            if tensor.ndim == 4 and ('conv' in key.lower() or 'pointwise' in key.lower()):
                mlx_weights[key] = convert_conv2d_weight(tensor)
            # Check if conv1d weight (3D with kernel dimension)
            elif tensor.ndim == 3 and ('conv' in key.lower() or 'pointwise' in key.lower()):
                mlx_weights[key] = convert_conv1d_weight(tensor)
            else:
                mlx_weights[key] = convert_linear_weight(tensor)
        elif '.bias' in key:
            mlx_weights[key] = mx.array(tensor.numpy())
        else:
            # Scalar or other parameters
            if tensor.ndim == 0:
                mlx_weights[key] = mx.array(tensor.item())
            else:
                mlx_weights[key] = mx.array(tensor.numpy())

    print(f"Converted {len(mlx_weights)} weights")
    print(f"Skipped {len(skipped_keys)} keys")

    if output_path:
        mx.savez(output_path, **mlx_weights)
        print(f"Saved to {output_path}")

    return mlx_weights


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch Zipformer weights to MLX format",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to PyTorch checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for MLX weights (optional)",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Only analyze checkpoint structure",
    )

    args = parser.parse_args()

    if args.analyze:
        analysis = analyze_checkpoint(args.checkpoint)
        print("Checkpoint Analysis:")
        print(f"  Total keys: {analysis['total_keys']}")
        print(f"  Encoder keys: {analysis['encoder_keys']}")
        print(f"  Decoder keys: {analysis['decoder_keys']}")
        print(f"  Joiner keys: {analysis['joiner_keys']}")
        print(f"  Other keys: {analysis['other_keys']}")
        print("\nEncoder stages:")
        for stage, info in analysis['encoder_stages'].items():
            print(f"  Stage {stage}: {len(info['layers'])} layers, {info['keys']} keys")

        config = extract_encoder_config(args.checkpoint)
        print("\nExtracted config:")
        for k, v in config.items():
            print(f"  {k}: {v}")
    else:
        weights = convert_full_model(args.checkpoint, args.output)

        # Print sample weights for verification
        print("\nSample converted weights:")
        for i, (key, arr) in enumerate(weights.items()):
            if i >= 5:
                break
            print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")


if __name__ == "__main__":
    main()
