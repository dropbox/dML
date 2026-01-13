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
Convert Conv-TasNet weights from Asteroid/SpeechBrain PyTorch format to MLX.

Usage:
    python scripts/convert_conv_tasnet_to_mlx.py \\
        --input /path/to/asteroid_checkpoint.pth \\
        --output /path/to/mlx_weights.safetensors \\
        --source asteroid

Sources:
    - asteroid: Asteroid library (https://github.com/asteroid-team/asteroid)
    - speechbrain: SpeechBrain library (https://github.com/speechbrain/speechbrain)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np


def load_pytorch_checkpoint(path: str) -> Dict[str, Any]:
    """Load PyTorch checkpoint."""
    try:
        import torch

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        return checkpoint
    except ImportError:
        raise ImportError("PyTorch required for weight conversion. Install with: pip install torch")


def convert_asteroid_weights(checkpoint: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Convert Asteroid Conv-TasNet weights to MLX format.

    Asteroid HuggingFace parameter naming:
        - encoder.filterbank._filters -> encoder.conv.weight
        - decoder.filterbank._filters -> decoder.deconv.weight
        - masker.bottleneck.0.gamma/beta -> separator.layer_norm.weight/bias (gLN)
        - masker.bottleneck.1.weight/bias -> separator.bottleneck.weight/bias
        - masker.TCN.{i}.shared_block.* -> separator.blocks.{i}.*
        - masker.TCN.{i}.res_conv.* -> separator.blocks.{i}.conv1x1_res.*
        - masker.TCN.{i}.skip_conv.* -> separator.blocks.{i}.conv1x1_skip.*
        - masker.mask_net.0.weight -> separator.mask_prelu.weight
        - masker.mask_net.1.weight/bias -> separator.mask_conv.weight/bias

    Asteroid TCN shared_block structure:
        0: Conv1d (bn_chan -> hid_chan) - first expansion conv
        1: PReLU weight (single param)
        2: GlobalLayerNorm (gamma, beta)
        3: DepthwiseConv1d (dilation varies)
        4: PReLU weight (single param)
        5: GlobalLayerNorm (gamma, beta)

    Args:
        checkpoint: Asteroid checkpoint dictionary.

    Returns:
        Dictionary of MLX-compatible weights as numpy arrays.
    """
    # Get state dict
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Get model args if available
    model_args = checkpoint.get("model_args", {})

    mlx_weights: Dict[str, np.ndarray] = {}

    for pt_name, tensor in state_dict.items():
        # Convert tensor to numpy
        if hasattr(tensor, "numpy"):
            arr = tensor.numpy()
        else:
            arr = np.array(tensor)

        # Encoder filterbank -> encoder.conv.weight
        if pt_name == "encoder.filterbank._filters":
            # Asteroid: (n_filters, 1, kernel_size) -> MLX: same shape
            mlx_weights["encoder.conv.weight"] = arr
            continue

        # Decoder filterbank -> decoder.deconv.weight
        if pt_name == "decoder.filterbank._filters":
            # Asteroid: (n_filters, 1, kernel_size) -> MLX: same shape
            mlx_weights["decoder.deconv.weight"] = arr
            continue

        # Bottleneck GlobalLayerNorm (gLN) -> separator.layer_norm
        if pt_name == "masker.bottleneck.0.gamma":
            mlx_weights["separator.layer_norm.weight"] = arr
            continue
        if pt_name == "masker.bottleneck.0.beta":
            mlx_weights["separator.layer_norm.bias"] = arr
            continue

        # Bottleneck Conv1d -> separator.bottleneck
        if pt_name == "masker.bottleneck.1.weight":
            mlx_weights["separator.bottleneck.weight"] = arr
            continue
        if pt_name == "masker.bottleneck.1.bias":
            mlx_weights["separator.bottleneck.bias"] = arr
            continue

        # TCN blocks: masker.TCN.{i}.* -> separator.blocks.{i}.*
        if pt_name.startswith("masker.TCN."):
            parts = pt_name.split(".")
            block_idx = int(parts[2])

            # shared_block components
            if parts[3] == "shared_block":
                subblock_idx = int(parts[4])
                param_name = parts[5]

                if subblock_idx == 0:
                    # First conv (expansion): bn_chan -> hid_chan
                    mlx_weights[f"separator.blocks.{block_idx}.conv1x1_1.{param_name}"] = arr
                elif subblock_idx == 1:
                    # First PReLU
                    mlx_weights[f"separator.blocks.{block_idx}.prelu1.{param_name}"] = arr
                elif subblock_idx == 2:
                    # First GlobalLayerNorm
                    if param_name == "gamma":
                        mlx_weights[f"separator.blocks.{block_idx}.norm1.weight"] = arr
                    elif param_name == "beta":
                        mlx_weights[f"separator.blocks.{block_idx}.norm1.bias"] = arr
                elif subblock_idx == 3:
                    # Depthwise conv - stored as direct attributes on TCNBlock
                    mlx_weights[f"separator.blocks.{block_idx}.dwconv_{param_name}"] = arr
                elif subblock_idx == 4:
                    # Second PReLU
                    mlx_weights[f"separator.blocks.{block_idx}.prelu2.{param_name}"] = arr
                elif subblock_idx == 5:
                    # Second GlobalLayerNorm
                    if param_name == "gamma":
                        mlx_weights[f"separator.blocks.{block_idx}.norm2.weight"] = arr
                    elif param_name == "beta":
                        mlx_weights[f"separator.blocks.{block_idx}.norm2.bias"] = arr
                continue

            # res_conv
            if parts[3] == "res_conv":
                param_name = parts[4]
                mlx_weights[f"separator.blocks.{block_idx}.conv1x1_res.{param_name}"] = arr
                continue

            # skip_conv
            if parts[3] == "skip_conv":
                param_name = parts[4]
                mlx_weights[f"separator.blocks.{block_idx}.conv1x1_skip.{param_name}"] = arr
                continue

        # Mask network: PReLU + Conv
        if pt_name == "masker.mask_net.0.weight":
            mlx_weights["separator.mask_prelu.weight"] = arr
            continue
        if pt_name == "masker.mask_net.1.weight":
            mlx_weights["separator.mask_conv.weight"] = arr
            continue
        if pt_name == "masker.mask_net.1.bias":
            mlx_weights["separator.mask_conv.bias"] = arr
            continue

        # Skip unknown parameters (log warning)
        print(f"Warning: Skipping unknown parameter: {pt_name}")

    return mlx_weights, model_args


def convert_speechbrain_weights(checkpoint: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Convert SpeechBrain Conv-TasNet weights to MLX format.

    SpeechBrain parameter naming differs from Asteroid.

    Args:
        checkpoint: SpeechBrain checkpoint dictionary.

    Returns:
        Dictionary of MLX-compatible weights as numpy arrays.
    """
    # Get state dict
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    mlx_weights: Dict[str, np.ndarray] = {}

    # SpeechBrain to MLX mapping
    name_mapping = {
        "encoder.enc_conv.weight": "encoder.conv.weight",
        "encoder.enc_conv.bias": "encoder.conv.bias",
        "decoder.dec_conv.weight": "decoder.deconv.weight",
        "decoder.dec_conv.bias": "decoder.deconv.bias",
    }

    for sb_name, tensor in state_dict.items():
        # Convert tensor to numpy
        if hasattr(tensor, "numpy"):
            arr = tensor.numpy()
        else:
            arr = np.array(tensor)

        # Check direct mapping
        if sb_name in name_mapping:
            mlx_name = name_mapping[sb_name]
            mlx_weights[mlx_name] = arr
            continue

        # Handle masker/separator layers
        if sb_name.startswith("masknet."):
            mlx_name = sb_name.replace("masknet.", "separator.")
            mlx_weights[mlx_name] = arr
            continue

        # Skip unknown parameters
        print(f"Warning: Skipping unknown parameter: {sb_name}")

    return mlx_weights, {}


def transpose_conv_weights(weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Transpose convolution weights from PyTorch to MLX format.

    PyTorch Conv1d: (out_channels, in_channels, kernel_size)
    MLX Conv1d: (out_channels, kernel_size, in_channels)

    PyTorch ConvTranspose1d: (in_channels, out_channels, kernel_size)
    MLX ConvTranspose1d: (out_channels, kernel_size, in_channels)

    For Asteroid decoder filterbank (512, 1, 32):
    - This is (in_ch, out_ch, kernel) for ConvTranspose1d
    - MLX expects (out_ch, kernel, in_ch) = (1, 32, 512)

    Args:
        weights: Dictionary of weights.

    Returns:
        Weights with transposed convolution kernels.
    """
    result = {}
    for name, arr in weights.items():
        if "weight" in name and arr.ndim == 3:
            if "deconv" in name.lower() or "decoder" in name.lower():
                # ConvTranspose1d: (in_ch, out_ch, kernel) -> (out_ch, kernel, in_ch)
                # Asteroid decoder: (512, 1, 32) -> (1, 32, 512)
                arr = np.transpose(arr, (1, 2, 0))
            else:
                # All other 3D weights are Conv1d: (O, I, K) -> (O, K, I)
                # This includes: encoder.conv, separator.bottleneck, conv1x1_*, etc.
                arr = np.transpose(arr, (0, 2, 1))
        result[name] = arr
    return result


def save_mlx_weights(
    weights: Dict[str, np.ndarray],
    output_path: str,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save weights in MLX safetensors format.

    Args:
        weights: Dictionary of weights as numpy arrays.
        output_path: Output file path.
        config: Optional config to save alongside.
    """
    import mlx.core as mx

    # Convert to MLX arrays
    mlx_weights = {k: mx.array(v) for k, v in weights.items()}

    # Save weights
    mx.save_safetensors(output_path, mlx_weights)
    print(f"Saved weights to {output_path}")

    # Save config if provided
    if config:
        config_path = Path(output_path).with_suffix(".json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Saved config to {config_path}")


def infer_config_from_weights(weights: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """
    Infer model configuration from weight shapes.

    Args:
        weights: Dictionary of weights.

    Returns:
        Inferred configuration dictionary.
    """
    config = {}

    # Infer n_filters from encoder
    if "encoder.conv.weight" in weights:
        config["n_filters"] = weights["encoder.conv.weight"].shape[0]
        config["kernel_size"] = weights["encoder.conv.weight"].shape[1]

    # Infer hidden_channels from bottleneck
    if "separator.bottleneck.weight" in weights:
        config["hidden_channels"] = weights["separator.bottleneck.weight"].shape[0]

    # Infer n_sources from mask conv
    if "separator.mask_conv.weight" in weights and "n_filters" in config:
        total_out = weights["separator.mask_conv.weight"].shape[0]
        config["n_sources"] = total_out // config["n_filters"]

    # Count TCN blocks to infer n_layers and n_stacks
    block_indices = set()
    for name in weights.keys():
        if "separator.blocks." in name:
            # Extract block index
            parts = name.split(".")
            for i, p in enumerate(parts):
                if p == "blocks" and i + 1 < len(parts):
                    try:
                        block_indices.add(int(parts[i + 1]))
                    except ValueError:
                        pass

    if block_indices:
        total_blocks = len(block_indices)
        # Assume n_stacks=3 and infer n_layers
        # Or use typical configurations
        if total_blocks == 24:  # 8 layers * 3 stacks
            config["n_layers"] = 8
            config["n_stacks"] = 3
        elif total_blocks == 12:  # 6 layers * 2 stacks
            config["n_layers"] = 6
            config["n_stacks"] = 2
        else:
            # Best guess
            config["n_layers"] = 8
            config["n_stacks"] = total_blocks // 8 or 3

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Convert Conv-TasNet weights to MLX format"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Input PyTorch checkpoint path",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output MLX weights path (.safetensors)",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["asteroid", "speechbrain", "auto"],
        default="auto",
        help="Source framework (default: auto-detect)",
    )
    parser.add_argument(
        "--no-transpose",
        action="store_true",
        help="Skip convolution weight transposition",
    )

    args = parser.parse_args()

    # Load PyTorch checkpoint
    print(f"Loading checkpoint from {args.input}")
    checkpoint = load_pytorch_checkpoint(args.input)

    # Detect source framework if auto
    source = args.source
    if source == "auto":
        if "state_dict" in checkpoint or any(
            k.startswith("encoder.conv1d") for k in checkpoint.keys()
        ):
            source = "asteroid"
        elif any(k.startswith("encoder.enc_conv") for k in checkpoint.keys()):
            source = "speechbrain"
        else:
            print("Warning: Could not detect source, assuming Asteroid")
            source = "asteroid"

    print(f"Using {source} converter")

    # Convert weights
    if source == "asteroid":
        mlx_weights, model_args = convert_asteroid_weights(checkpoint)
    else:
        mlx_weights, model_args = convert_speechbrain_weights(checkpoint)

    # Transpose convolution weights
    if not args.no_transpose:
        mlx_weights = transpose_conv_weights(mlx_weights)

    # Infer config from weights and model_args
    config = infer_config_from_weights(mlx_weights)
    # Add model_args to config if available
    if model_args:
        config["model_args"] = model_args
        # Extract relevant parameters
        if "n_filters" in model_args:
            config["n_filters"] = model_args["n_filters"]
        if "kernel_size" in model_args:
            config["kernel_size"] = model_args["kernel_size"]
        if "stride" in model_args:
            config["stride"] = model_args["stride"]
        if "bn_chan" in model_args:
            config["hidden_channels"] = model_args["bn_chan"]
        if "skip_chan" in model_args:
            config["skip_channels"] = model_args["skip_chan"]
        if "n_blocks" in model_args:
            config["n_layers"] = model_args["n_blocks"]
        if "n_repeats" in model_args:
            config["n_stacks"] = model_args["n_repeats"]
        if "n_src" in model_args:
            config["n_sources"] = model_args["n_src"]
        if "sample_rate" in model_args:
            config["sample_rate"] = model_args["sample_rate"]
    print(f"Inferred config: {config}")

    # Save
    save_mlx_weights(mlx_weights, args.output, config)

    print("Conversion complete!")
    print(f"Total parameters: {sum(w.size for w in mlx_weights.values()):,}")


if __name__ == "__main__":
    main()
