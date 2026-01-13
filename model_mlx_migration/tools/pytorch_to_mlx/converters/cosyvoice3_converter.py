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
CosyVoice3 Weight Converter

Converts PyTorch weights (.pt files) and SafeTensors to MLX format.

Model files:
- CosyVoice-BlankEN/model.safetensors - LLM (Qwen2) weights
- flow.pt - DiT Flow model weights
- hift.pt - CausalHiFT vocoder weights
- speech_tokenizer_v3.onnx - Speech tokenizer (ONNX, not converted)
- campplus.onnx - Speaker encoder (ONNX, not converted)
"""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

# Import model classes
from .models.cosyvoice3 import (
    CosyVoice3Model,
    create_cosyvoice3_config,
)


def load_pytorch_weights(path: Path) -> dict[str, mx.array]:
    """
    Load PyTorch .pt file weights.

    Note: Requires torch to be installed for loading.
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required to load .pt files. Install with: pip install torch") from None

    # Load PyTorch checkpoint
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Convert to MLX arrays
    mlx_weights = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            # Convert to numpy then MLX
            np_array = value.detach().cpu().numpy()
            mlx_weights[key] = mx.array(np_array)

    return mlx_weights


def load_safetensors_weights(path: Path) -> dict[str, mx.array]:
    """Load SafeTensors weights directly to MLX."""
    return mx.load(str(path))


def convert_llm_weights(
    weights: dict[str, mx.array],
    model: nn.Module,
) -> None:
    """
    Map Qwen2 weights from HuggingFace format to MLX model.

    HF format: model.layers.{i}.{component}.{param}
    MLX format: llm.llm.layers.{i}.{component}.{param}

    Note: model.llm = CosyVoice2LLM, model.llm.llm = Qwen2Model
    """
    # Get the inner Qwen2Model
    qwen_model = model.llm.llm

    # Embedding
    if "model.embed_tokens.weight" in weights:
        qwen_model.embed_tokens.weight = weights["model.embed_tokens.weight"]

    # Layers
    num_layers = len(qwen_model.layers)
    for i in range(num_layers):
        prefix = f"model.layers.{i}"
        layer = qwen_model.layers[i]

        # Self attention - weights
        if f"{prefix}.self_attn.q_proj.weight" in weights:
            layer.self_attn.q_proj.weight = weights[f"{prefix}.self_attn.q_proj.weight"]
        if f"{prefix}.self_attn.k_proj.weight" in weights:
            layer.self_attn.k_proj.weight = weights[f"{prefix}.self_attn.k_proj.weight"]
        if f"{prefix}.self_attn.v_proj.weight" in weights:
            layer.self_attn.v_proj.weight = weights[f"{prefix}.self_attn.v_proj.weight"]
        if f"{prefix}.self_attn.o_proj.weight" in weights:
            layer.self_attn.o_proj.weight = weights[f"{prefix}.self_attn.o_proj.weight"]

        # Self attention - biases (Qwen2 has biases on QKV)
        if f"{prefix}.self_attn.q_proj.bias" in weights:
            layer.self_attn.q_proj.bias = weights[f"{prefix}.self_attn.q_proj.bias"]
        if f"{prefix}.self_attn.k_proj.bias" in weights:
            layer.self_attn.k_proj.bias = weights[f"{prefix}.self_attn.k_proj.bias"]
        if f"{prefix}.self_attn.v_proj.bias" in weights:
            layer.self_attn.v_proj.bias = weights[f"{prefix}.self_attn.v_proj.bias"]

        # MLP
        if f"{prefix}.mlp.gate_proj.weight" in weights:
            layer.mlp.gate_proj.weight = weights[f"{prefix}.mlp.gate_proj.weight"]
        if f"{prefix}.mlp.up_proj.weight" in weights:
            layer.mlp.up_proj.weight = weights[f"{prefix}.mlp.up_proj.weight"]
        if f"{prefix}.mlp.down_proj.weight" in weights:
            layer.mlp.down_proj.weight = weights[f"{prefix}.mlp.down_proj.weight"]

        # Layer norms
        if f"{prefix}.input_layernorm.weight" in weights:
            layer.input_layernorm.weight = weights[f"{prefix}.input_layernorm.weight"]
        if f"{prefix}.post_attention_layernorm.weight" in weights:
            layer.post_attention_layernorm.weight = weights[f"{prefix}.post_attention_layernorm.weight"]

    # Final norm
    if "model.norm.weight" in weights:
        qwen_model.norm.weight = weights["model.norm.weight"]

    # LM head (goes on the CosyVoice2LLM, not Qwen2Model)
    if "lm_head.weight" in weights:
        model.llm.lm_head.weight = weights["lm_head.weight"]


def convert_flow_weights(
    weights: dict[str, mx.array],
    model: nn.Module,
) -> None:
    """
    Map DiT flow weights from PyTorch to MLX model.

    PyTorch structure:
    - decoder.estimator.time_embed.time_mlp.{0,2}.*
    - decoder.estimator.input_embed.{proj, conv_pos_embed}.*
    - decoder.estimator.transformer_blocks.{i}.{attn, attn_norm, ff}.*
    - decoder.estimator.{norm_out, proj_out, rotary_embed}.*
    - {input_embedding, pre_lookahead_layer, spk_embed_affine_layer}.*
    """
    # Logging for debug
    loaded_count = 0

    def try_load(key: str, target_obj, attr: str):
        nonlocal loaded_count
        if key in weights:
            setattr(target_obj, attr, weights[key])
            loaded_count += 1
            return True
        return False

    dit = model.flow.dit

    # Time embedding: decoder.estimator.time_embed.time_mlp.{0,2}.*
    try_load("decoder.estimator.time_embed.time_mlp.0.weight", dit.time_embed.mlp.layers[0], "weight")
    try_load("decoder.estimator.time_embed.time_mlp.0.bias", dit.time_embed.mlp.layers[0], "bias")
    try_load("decoder.estimator.time_embed.time_mlp.2.weight", dit.time_embed.mlp.layers[2], "weight")
    try_load("decoder.estimator.time_embed.time_mlp.2.bias", dit.time_embed.mlp.layers[2], "bias")

    # Input embedding projection
    try_load("decoder.estimator.input_embed.proj.weight", dit.input_embed.proj, "weight")
    try_load("decoder.estimator.input_embed.proj.bias", dit.input_embed.proj, "bias")

    # Conv positional embedding
    # Note: PyTorch Conv1d weight shape is [out, in, kernel], MLX uses [out, kernel, in]
    for conv_name, target in [
        ("conv1.0", dit.input_embed.conv_pos_embed.conv1),
        ("conv2.0", dit.input_embed.conv_pos_embed.conv2),
    ]:
        w_key = f"decoder.estimator.input_embed.conv_pos_embed.{conv_name}.weight"
        b_key = f"decoder.estimator.input_embed.conv_pos_embed.{conv_name}.bias"
        if w_key in weights:
            w = weights[w_key]
            w_transposed = mx.transpose(w, (0, 2, 1))
            target.weight = w_transposed
            loaded_count += 1
        if b_key in weights:
            target.bias = weights[b_key]
            loaded_count += 1

    # Rotary embedding inv_freq
    try_load("decoder.estimator.rotary_embed.inv_freq", dit.rotary_embed, "inv_freq")

    # DiT blocks: decoder.estimator.transformer_blocks.{i}.*
    for i, block in enumerate(dit.blocks):
        prefix = f"decoder.estimator.transformer_blocks.{i}"

        # Attention - separate Q, K, V projections
        try_load(f"{prefix}.attn.to_q.weight", block.attn.to_q, "weight")
        try_load(f"{prefix}.attn.to_q.bias", block.attn.to_q, "bias")
        try_load(f"{prefix}.attn.to_k.weight", block.attn.to_k, "weight")
        try_load(f"{prefix}.attn.to_k.bias", block.attn.to_k, "bias")
        try_load(f"{prefix}.attn.to_v.weight", block.attn.to_v, "weight")
        try_load(f"{prefix}.attn.to_v.bias", block.attn.to_v, "bias")

        # Attention output: to_out.0
        try_load(f"{prefix}.attn.to_out.0.weight", block.attn.to_out, "weight")
        try_load(f"{prefix}.attn.to_out.0.bias", block.attn.to_out, "bias")

        # Adaptive norm: attn_norm.linear
        try_load(f"{prefix}.attn_norm.linear.weight", block.attn_norm.linear, "weight")
        try_load(f"{prefix}.attn_norm.linear.bias", block.attn_norm.linear, "bias")

        # FFN: ff.ff.0.0 (first linear) and ff.ff.2 (second linear)
        try_load(f"{prefix}.ff.ff.0.0.weight", block.ff.layers[0], "weight")
        try_load(f"{prefix}.ff.ff.0.0.bias", block.ff.layers[0], "bias")
        try_load(f"{prefix}.ff.ff.2.weight", block.ff.layers[1], "weight")
        try_load(f"{prefix}.ff.ff.2.bias", block.ff.layers[1], "bias")

    # Output norm: norm_out.linear
    try_load("decoder.estimator.norm_out.linear.weight", dit.norm_out.linear, "weight")
    try_load("decoder.estimator.norm_out.linear.bias", dit.norm_out.linear, "bias")

    # Output projection
    try_load("decoder.estimator.proj_out.weight", dit.proj_out, "weight")
    try_load("decoder.estimator.proj_out.bias", dit.proj_out, "bias")

    # Top-level flow components
    # input_embedding.weight
    try_load("input_embedding.weight", model.flow.input_embedding, "weight")

    # pre_lookahead_layer.{conv1, conv2} with Conv1d transposition
    if hasattr(model.flow, "pre_lookahead_layer"):
        pre_layer = model.flow.pre_lookahead_layer
        # conv1: 80 -> 1024, kernel=4
        w_key = "pre_lookahead_layer.conv1.weight"
        if w_key in weights:
            w = weights[w_key]
            # PyTorch: [out, in, kernel] -> MLX: [out, kernel, in]
            pre_layer.conv1.weight = mx.transpose(w, (0, 2, 1))
            loaded_count += 1
        try_load("pre_lookahead_layer.conv1.bias", pre_layer.conv1, "bias")
        # conv2: 1024 -> 80, kernel=3
        w_key = "pre_lookahead_layer.conv2.weight"
        if w_key in weights:
            w = weights[w_key]
            pre_layer.conv2.weight = mx.transpose(w, (0, 2, 1))
            loaded_count += 1
        try_load("pre_lookahead_layer.conv2.bias", pre_layer.conv2, "bias")

    # spk_embed_affine_layer.*
    if hasattr(model.flow, "spk_embed_affine_layer"):
        try_load("spk_embed_affine_layer.weight", model.flow.spk_embed_affine_layer, "weight")
        try_load("spk_embed_affine_layer.bias", model.flow.spk_embed_affine_layer, "bias")

    print(f"    Flow: loaded {loaded_count} weights")


def convert_vocoder_weights(
    weights: dict[str, mx.array],
    model: nn.Module,
) -> None:
    """
    Map CausalHiFT vocoder weights from PyTorch to MLX model.

    Handles:
    - Weight normalization (parametrizations.weight.original0/1)
    - Snake activations (alpha parameters)
    - Conv1d weight transposition

    Note: For inference-only, weight normalization is precomputed.
    """
    loaded_count = 0

    def compute_weight_norm(orig0: mx.array, orig1: mx.array) -> mx.array:
        """Compute actual weight from weight normalization parameters.

        w = g * (v / ||v||)
        where g = original0 (magnitude), v = original1 (direction)
        """
        # Compute L2 norm per output channel
        # orig1 shape: [out, in, kernel] (PyTorch Conv1d)
        v_norm = mx.sqrt(mx.sum(orig1 * orig1, axis=(1, 2), keepdims=True) + 1e-12)
        w = orig0 * (orig1 / v_norm)
        # Transpose to MLX format: [out, kernel, in]
        return mx.transpose(w, (0, 2, 1))

    def try_load_weight_norm_conv(prefix: str, target):
        """Load weight-normalized conv weight to Conv1d wrapper."""
        nonlocal loaded_count
        orig0_key = f"{prefix}.parametrizations.weight.original0"
        orig1_key = f"{prefix}.parametrizations.weight.original1"
        bias_key = f"{prefix}.bias"

        if orig0_key in weights and orig1_key in weights:
            w = compute_weight_norm(weights[orig0_key], weights[orig1_key])
            target.conv.weight = w
            loaded_count += 1

        if bias_key in weights:
            target.conv.bias = weights[bias_key]
            loaded_count += 1

    def try_load_weight_norm_convtrans(prefix: str, target):
        """Load weight-normalized ConvTranspose1d weight."""
        nonlocal loaded_count
        orig0_key = f"{prefix}.parametrizations.weight.original0"
        orig1_key = f"{prefix}.parametrizations.weight.original1"
        bias_key = f"{prefix}.bias"

        if orig0_key in weights and orig1_key in weights:
            # PyTorch ConvTranspose1d: [C_out, C_in, kernel]
            # MLX ConvTranspose1d expects: [C_out, kernel, C_in]
            orig0, orig1 = weights[orig0_key], weights[orig1_key]
            v_norm = mx.sqrt(mx.sum(orig1 * orig1, axis=(1, 2), keepdims=True) + 1e-12)
            w = orig0 * (orig1 / v_norm)
            # Transpose: [C_out, C_in, kernel] -> [C_out, kernel, C_in]
            w = mx.transpose(w, (0, 2, 1))
            target.conv.weight = w
            loaded_count += 1

        if bias_key in weights:
            target.conv.bias = weights[bias_key]
            loaded_count += 1

    def try_load_regular_conv(prefix: str, target, transpose: bool = True):
        """Load regular conv weight to Conv1d wrapper."""
        nonlocal loaded_count
        w_key = f"{prefix}.weight"
        b_key = f"{prefix}.bias"

        if w_key in weights:
            w = weights[w_key]
            if transpose and len(w.shape) == 3:
                w = mx.transpose(w, (0, 2, 1))
            target.conv.weight = w
            loaded_count += 1

        if b_key in weights:
            target.conv.bias = weights[b_key]
            loaded_count += 1

    def try_load_linear(prefix: str, target):
        """Load Linear weight and bias."""
        nonlocal loaded_count
        w_key = f"{prefix}.weight"
        b_key = f"{prefix}.bias"

        if w_key in weights:
            target.weight = weights[w_key]
            loaded_count += 1

        if b_key in weights:
            target.bias = weights[b_key]
            loaded_count += 1

    def try_load_snake(prefix: str, target):
        """Load Snake activation alpha parameter."""
        nonlocal loaded_count
        alpha_key = f"{prefix}.alpha"

        if alpha_key in weights:
            target.alpha = weights[alpha_key]
            loaded_count += 1

    def try_load_resblock(prefix: str, block):
        """Load weights for a ResBlock with Snake activations."""
        # convs1, convs2, activations1, activations2
        for i, (c1, a1, c2, a2) in enumerate(zip(
            block.convs1, block.activations1, block.convs2, block.activations2, strict=False,
        )):
            try_load_weight_norm_conv(f"{prefix}.convs1.{i}", c1)
            try_load_snake(f"{prefix}.activations1.{i}", a1)
            try_load_weight_norm_conv(f"{prefix}.convs2.{i}", c2)
            try_load_snake(f"{prefix}.activations2.{i}", a2)

    vocoder = model.vocoder

    # conv_pre (weight normalized)
    try_load_weight_norm_conv("conv_pre", vocoder.conv_pre)

    # conv_post (weight normalized)
    try_load_weight_norm_conv("conv_post", vocoder.conv_post)

    # ups (weight normalized ConvTranspose1d)
    for i, up in enumerate(vocoder.ups):
        try_load_weight_norm_convtrans(f"ups.{i}", up)

    # resblocks (9 total)
    for i, rb in enumerate(vocoder.resblocks):
        try_load_resblock(f"resblocks.{i}", rb)

    # source_downs (regular Conv1d)
    for i, sd in enumerate(vocoder.source_downs):
        try_load_regular_conv(f"source_downs.{i}", sd)

    # source_resblocks
    for i, srb in enumerate(vocoder.source_resblocks):
        try_load_resblock(f"source_resblocks.{i}", srb)

    # f0_predictor
    # condnet: layers at indices 0, 2, 4, 6, 8
    for i, conv in enumerate(vocoder.f0_predictor.condnet):
        pt_idx = i * 2  # 0, 2, 4, 6, 8
        try_load_weight_norm_conv(f"f0_predictor.condnet.{pt_idx}", conv)

    # classifier
    try_load_linear("f0_predictor.classifier", vocoder.f0_predictor.classifier)

    # m_source.l_linear
    try_load_linear("m_source.l_linear", vocoder.m_source.l_linear)

    print(f"    Vocoder: loaded {loaded_count} weights")


class CosyVoice3Converter:
    """Converter for CosyVoice3 models."""

    def __init__(self, model_path: str | Path):
        """
        Initialize converter.

        Args:
            model_path: Path to CosyVoice3 model directory
        """
        self.model_path = Path(model_path)

        # File paths
        self.llm_path = self.model_path / "CosyVoice-BlankEN" / "model.safetensors"
        self.flow_path = self.model_path / "flow.pt"
        self.vocoder_path = self.model_path / "hift.pt"

    def convert(
        self,
        output_path: Path | None = None,
        dtype: mx.Dtype = mx.float16,
    ) -> CosyVoice3Model:
        """
        Convert CosyVoice3 to MLX format.

        Args:
            output_path: Optional path to save converted weights
            dtype: Model dtype

        Returns:
            Converted CosyVoice3Model
        """
        print("Creating CosyVoice3 model...")
        config = create_cosyvoice3_config()
        model = CosyVoice3Model(config)

        # Load and convert LLM weights
        if self.llm_path.exists():
            print(f"Loading LLM weights from {self.llm_path}...")
            llm_weights = load_safetensors_weights(self.llm_path)
            convert_llm_weights(llm_weights, model)
            print(f"  Loaded {len(llm_weights)} weight tensors")

        # Load and convert flow weights
        if self.flow_path.exists():
            print(f"Loading flow weights from {self.flow_path}...")
            try:
                flow_weights = load_pytorch_weights(self.flow_path)
                convert_flow_weights(flow_weights, model)
                print(f"  Loaded {len(flow_weights)} weight tensors")
            except ImportError as e:
                print(f"  Warning: {e}")
                print("  Skipping flow weights - install torch to convert")

        # Load and convert vocoder weights
        if self.vocoder_path.exists():
            print(f"Loading vocoder weights from {self.vocoder_path}...")
            try:
                vocoder_weights = load_pytorch_weights(self.vocoder_path)
                convert_vocoder_weights(vocoder_weights, model)
                print(f"  Loaded {len(vocoder_weights)} weight tensors")
            except ImportError as e:
                print(f"  Warning: {e}")
                print("  Skipping vocoder weights - install torch to convert")

        # Convert to target dtype
        print(f"Converting to {dtype}...")
        # Note: MLX doesn't have a simple model.to(dtype) - would need to iterate

        # Save if requested
        if output_path:
            print(f"Saving to {output_path}...")
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)

            # Save as safetensors using tree_flatten for proper nested dict handling
            from mlx.utils import tree_flatten
            flat_weights = dict(tree_flatten(model.parameters()))

            mx.save_safetensors(str(output_path / "model.safetensors"), flat_weights)
            print(f"  Saved {len(flat_weights)} weight tensors")

            # Save config
            with open(output_path / "config.json", "w") as f:
                json.dump({
                    "model_type": "cosyvoice3",
                    "sample_rate": config.sample_rate,
                    "token_frame_rate": config.token_frame_rate,
                    "token_mel_ratio": config.token_mel_ratio,
                    "speech_token_size": config.speech_token_size,
                }, f, indent=2)

        return model


def main():
    """Convert CosyVoice3 model."""
    import argparse

    parser = argparse.ArgumentParser(description="Convert CosyVoice3 to MLX")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/cosyvoice3",
        help="Path to CosyVoice3 model directory",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="models/cosyvoice3_mlx",
        help="Output path for converted model",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Model dtype",
    )

    args = parser.parse_args()

    dtype_map = {
        "float16": mx.float16,
        "float32": mx.float32,
        "bfloat16": mx.bfloat16,
    }

    converter = CosyVoice3Converter(args.model_path)
    converter.convert(
        output_path=Path(args.output_path),
        dtype=dtype_map[args.dtype],
    )

    print("\nConversion complete!")
    print(f"Model saved to: {args.output_path}")


if __name__ == "__main__":
    main()
