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
Utility functions for WhisperMLX.
"""

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


def load_weights_from_safetensors(
    model: nn.Module,
    weights_path: str,
    strict: bool = True,
) -> tuple[int, int]:
    """
    Load weights from safetensors file into model.

    Args:
        model: MLX model to load weights into
        weights_path: Path to .safetensors file
        strict: If True, raise error on missing/unexpected keys

    Returns:
        Tuple of (loaded_count, skipped_count)
    """
    try:
        import safetensors.numpy as sf_np
    except ImportError:
        raise ImportError("safetensors is required: pip install safetensors") from None

    # Load weights
    weights = sf_np.load_file(weights_path)

    # Convert to MLX arrays
    mlx_weights = {k: mx.array(v) for k, v in weights.items()}

    # Get model parameters
    model_params = dict(model.named_parameters())

    loaded = 0
    skipped = 0

    for name, param in mlx_weights.items():
        if name in model_params:
            # Update parameter
            model_params[name] = param
            loaded += 1
        else:
            if strict:
                raise KeyError(f"Unexpected weight: {name}")
            skipped += 1

    if strict:
        missing = set(model_params.keys()) - set(mlx_weights.keys())
        if missing:
            raise KeyError(f"Missing weights: {missing}")

    # Apply weights
    model.load_weights(list(mlx_weights.items()))

    return loaded, skipped


def convert_hf_weights(
    hf_weights: dict[str, mx.array],
) -> dict[str, mx.array]:
    """
    Convert HuggingFace Whisper weights to our format.

    The weight names differ slightly:
    - HF: model.encoder.layers.0.self_attn.q_proj.weight
    - Ours: encoder.blocks.0.attn.query.weight

    Args:
        hf_weights: Dictionary of HuggingFace weights

    Returns:
        Dictionary of converted weights
    """
    converted = {}

    name_map = {
        # Encoder conv layers
        "model.encoder.conv1.weight": "encoder.conv1.weight",
        "model.encoder.conv1.bias": "encoder.conv1.bias",
        "model.encoder.conv2.weight": "encoder.conv2.weight",
        "model.encoder.conv2.bias": "encoder.conv2.bias",

        # Encoder positional embedding
        "model.encoder.embed_positions.weight": "encoder._positional_embedding",

        # Encoder layer norm
        "model.encoder.layer_norm.weight": "encoder.ln_post.weight",
        "model.encoder.layer_norm.bias": "encoder.ln_post.bias",

        # Decoder embeddings
        "model.decoder.embed_tokens.weight": "decoder.token_embedding.weight",
        "model.decoder.embed_positions.weight": "decoder.positional_embedding",

        # Decoder layer norm
        "model.decoder.layer_norm.weight": "decoder.ln.weight",
        "model.decoder.layer_norm.bias": "decoder.ln.bias",

        # Output projection (tied with embeddings)
        "proj_out.weight": "decoder.token_embedding.weight",
    }

    for hf_name, our_name in name_map.items():
        if hf_name in hf_weights:
            converted[our_name] = hf_weights[hf_name]

    # Convert encoder layers
    for hf_name, weight in hf_weights.items():
        if "model.encoder.layers." in hf_name:
            # Extract layer index
            parts = hf_name.split(".")
            layer_idx = parts[3]

            # Map attention weights
            if "self_attn.q_proj" in hf_name:
                suffix = "weight" if "weight" in hf_name else "bias"
                converted[f"encoder.blocks.{layer_idx}.attn.query.{suffix}"] = weight
            elif "self_attn.k_proj" in hf_name:
                suffix = "weight" if "weight" in hf_name else "bias"
                converted[f"encoder.blocks.{layer_idx}.attn.key.{suffix}"] = weight
            elif "self_attn.v_proj" in hf_name:
                suffix = "weight" if "weight" in hf_name else "bias"
                converted[f"encoder.blocks.{layer_idx}.attn.value.{suffix}"] = weight
            elif "self_attn.out_proj" in hf_name:
                suffix = "weight" if "weight" in hf_name else "bias"
                converted[f"encoder.blocks.{layer_idx}.attn.out.{suffix}"] = weight
            # Layer norms
            elif "self_attn_layer_norm" in hf_name:
                suffix = "weight" if "weight" in hf_name else "bias"
                converted[f"encoder.blocks.{layer_idx}.attn_ln.{suffix}"] = weight
            elif "final_layer_norm" in hf_name:
                suffix = "weight" if "weight" in hf_name else "bias"
                converted[f"encoder.blocks.{layer_idx}.mlp_ln.{suffix}"] = weight
            # MLP
            elif "fc1" in hf_name:
                suffix = "weight" if "weight" in hf_name else "bias"
                converted[f"encoder.blocks.{layer_idx}.mlp1.{suffix}"] = weight
            elif "fc2" in hf_name:
                suffix = "weight" if "weight" in hf_name else "bias"
                converted[f"encoder.blocks.{layer_idx}.mlp2.{suffix}"] = weight

    # Convert decoder layers
    for hf_name, weight in hf_weights.items():
        if "model.decoder.layers." in hf_name:
            parts = hf_name.split(".")
            layer_idx = parts[3]

            # Self attention
            if "self_attn.q_proj" in hf_name:
                suffix = "weight" if "weight" in hf_name else "bias"
                converted[f"decoder.blocks.{layer_idx}.attn.query.{suffix}"] = weight
            elif "self_attn.k_proj" in hf_name:
                suffix = "weight" if "weight" in hf_name else "bias"
                converted[f"decoder.blocks.{layer_idx}.attn.key.{suffix}"] = weight
            elif "self_attn.v_proj" in hf_name:
                suffix = "weight" if "weight" in hf_name else "bias"
                converted[f"decoder.blocks.{layer_idx}.attn.value.{suffix}"] = weight
            elif "self_attn.out_proj" in hf_name:
                suffix = "weight" if "weight" in hf_name else "bias"
                converted[f"decoder.blocks.{layer_idx}.attn.out.{suffix}"] = weight
            # Cross attention
            elif "encoder_attn.q_proj" in hf_name:
                suffix = "weight" if "weight" in hf_name else "bias"
                converted[f"decoder.blocks.{layer_idx}.cross_attn.query.{suffix}"] = weight
            elif "encoder_attn.k_proj" in hf_name:
                suffix = "weight" if "weight" in hf_name else "bias"
                converted[f"decoder.blocks.{layer_idx}.cross_attn.key.{suffix}"] = weight
            elif "encoder_attn.v_proj" in hf_name:
                suffix = "weight" if "weight" in hf_name else "bias"
                converted[f"decoder.blocks.{layer_idx}.cross_attn.value.{suffix}"] = weight
            elif "encoder_attn.out_proj" in hf_name:
                suffix = "weight" if "weight" in hf_name else "bias"
                converted[f"decoder.blocks.{layer_idx}.cross_attn.out.{suffix}"] = weight
            # Layer norms
            elif "self_attn_layer_norm" in hf_name:
                suffix = "weight" if "weight" in hf_name else "bias"
                converted[f"decoder.blocks.{layer_idx}.attn_ln.{suffix}"] = weight
            elif "encoder_attn_layer_norm" in hf_name:
                suffix = "weight" if "weight" in hf_name else "bias"
                converted[f"decoder.blocks.{layer_idx}.cross_attn_ln.{suffix}"] = weight
            elif "final_layer_norm" in hf_name:
                suffix = "weight" if "weight" in hf_name else "bias"
                converted[f"decoder.blocks.{layer_idx}.mlp_ln.{suffix}"] = weight
            # MLP
            elif "fc1" in hf_name:
                suffix = "weight" if "weight" in hf_name else "bias"
                converted[f"decoder.blocks.{layer_idx}.mlp1.{suffix}"] = weight
            elif "fc2" in hf_name:
                suffix = "weight" if "weight" in hf_name else "bias"
                converted[f"decoder.blocks.{layer_idx}.mlp2.{suffix}"] = weight

    return converted


def download_model(
    model_name: str,
    cache_dir: str | None = None,
) -> Path:
    """
    Download model from HuggingFace hub.

    Args:
        model_name: Model name (e.g., "large-v3", "mlx-community/whisper-large-v3-mlx")
        cache_dir: Optional cache directory

    Returns:
        Path to model directory
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError("huggingface_hub is required: pip install huggingface_hub") from None

    # Normalize model name
    if "/" not in model_name:
        # Use mlx-community pre-converted models
        # Note: "turbo" models don't have the "-mlx" suffix on HuggingFace
        if "turbo" in model_name:
            model_name = f"mlx-community/whisper-{model_name}"
        else:
            model_name = f"mlx-community/whisper-{model_name}-mlx"

    # Download
    model_path = snapshot_download(
        model_name,
        cache_dir=cache_dir,
        allow_patterns=["*.safetensors", "*.json", "config.json"],
    )

    return Path(model_path)


def format_timestamp(seconds: float, always_include_hours: bool = False) -> str:
    """
    Format timestamp for SRT/VTT output.

    Args:
        seconds: Time in seconds
        always_include_hours: Include hours even if zero

    Returns:
        Formatted timestamp string (HH:MM:SS,mmm or MM:SS,mmm)
    """
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    secs = milliseconds // 1_000
    milliseconds -= secs * 1_000

    if always_include_hours or hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
    return f"{minutes:02d}:{secs:02d},{milliseconds:03d}"
