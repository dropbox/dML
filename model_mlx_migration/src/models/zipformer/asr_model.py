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
Complete Zipformer ASR Model in MLX.

Combines:
- Encoder (Zipformer)
- Decoder (Stateless Predictor)
- Joiner (RNN-T output)
"""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .decoder import Decoder, DecoderConfig
from .joiner import Joiner, JoinerConfig
from .zipformer import Zipformer, ZipformerConfig


@dataclass
class ASRModelConfig:
    """Configuration for complete ASR model."""
    # Encoder config
    d_model: int = 192
    num_encoder_layers: tuple[int, ...] = (2, 2, 2, 2, 2, 2)
    attention_dims: tuple[int, ...] = (128, 128, 128, 256, 128, 128)
    feedforward_dims: tuple[int, ...] = (384, 384, 384, 512, 384, 384)
    ff1_dims: tuple[int, ...] | None = None
    ff2_dims: tuple[int, ...] | None = None
    ff3_dims: tuple[int, ...] | None = None
    num_heads: tuple[int, ...] = (4, 4, 4, 8, 4, 4)
    encoder_dims: tuple[int, ...] = (192, 256, 256, 256, 256, 192)
    downsampling_factors: tuple[int, ...] = (1, 2, 4, 8, 4, 2)
    cnn_module_kernels: tuple[int, ...] = (31, 31, 15, 15, 15, 31)
    pos_dim: int = 48
    pos_head_dim: int = 4
    value_head_dim: int = 12
    dropout: float = 0.0
    causal: bool = True

    # Decoder config
    vocab_size: int = 500
    decoder_dim: int = 512
    blank_id: int = 0
    context_size: int = 2

    # Joiner config
    joiner_dim: int = 512

    @property
    def encoder_output_dim(self) -> int:
        """Output dimension of encoder.

        Icefall Zipformer uses max(encoder_dims) as output dimension,
        combining outputs from multiple stages via _get_full_dim_output().
        """
        return max(self.encoder_dims)


class ASRModel(nn.Module):
    """
    Complete Zipformer ASR Model.

    Architecture:
        mel_features -> Encoder -> encoder_out
        tokens -> Decoder -> decoder_out
        (encoder_out, decoder_out) -> Joiner -> logits

    For streaming inference, use:
        encoder.forward_chunk() for chunked encoding
        decoder.forward_one_step() for single token decoding
        joiner.forward_streaming() for streaming output
    """

    def __init__(self, config: ASRModelConfig):
        """
        Initialize ASR model.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.config = config

        # Encoder
        encoder_config = ZipformerConfig(
            num_encoder_layers=config.num_encoder_layers,
            attention_dims=config.attention_dims,
            feedforward_dims=config.feedforward_dims,
            ff1_dims=config.ff1_dims,
            ff2_dims=config.ff2_dims,
            ff3_dims=config.ff3_dims,
            num_heads=config.num_heads,
            encoder_dims=config.encoder_dims,
            downsampling_factors=config.downsampling_factors,
            cnn_module_kernels=config.cnn_module_kernels,
            pos_dim=config.pos_dim,
            pos_head_dim=config.pos_head_dim,
            value_head_dim=config.value_head_dim,
            causal=config.causal,
        )
        self.encoder = Zipformer(encoder_config)

        # Decoder
        decoder_config = DecoderConfig(
            vocab_size=config.vocab_size,
            decoder_dim=config.decoder_dim,
            blank_id=config.blank_id,
            context_size=config.context_size,
        )
        self.decoder = Decoder(decoder_config)

        # Joiner
        joiner_config = JoinerConfig(
            encoder_dim=config.encoder_output_dim,
            decoder_dim=config.decoder_dim,
            joiner_dim=config.joiner_dim,
            vocab_size=config.vocab_size,
        )
        self.joiner = Joiner(joiner_config)

    def __call__(
        self,
        x: mx.array,
        x_lens: mx.array | None = None,
        y: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """
        Forward pass for training.

        Args:
            x: Mel-spectrogram features of shape (batch, time, mel_dim).
            x_lens: Feature lengths of shape (batch,).
            y: Target tokens of shape (batch, label_len).

        Returns:
            encoder_out: Encoder output (batch, T, encoder_dim).
            encoder_out_lens: Encoder output lengths (batch,).
        """
        # Encode
        encoder_out, encoder_out_lens = self.encoder(x, x_lens)
        return encoder_out, encoder_out_lens

    def get_logits(
        self,
        encoder_out: mx.array,
        decoder_out: mx.array,
    ) -> mx.array:
        """
        Get logits from encoder and decoder outputs.

        Args:
            encoder_out: Encoder output (batch, T, encoder_dim).
            decoder_out: Decoder output (batch, U, decoder_dim).

        Returns:
            Logits of shape (batch, T, U, vocab_size).
        """
        return self.joiner(encoder_out, decoder_out)

    def decode(
        self,
        encoder_out: mx.array,
        y: mx.array,
    ) -> mx.array:
        """
        Full decoder forward pass.

        Args:
            encoder_out: Encoder output (batch, T, encoder_dim).
            y: Target tokens (batch, U).

        Returns:
            Logits of shape (batch, T, U, vocab_size).
        """
        decoder_out = self.decoder(y)
        return self.joiner(encoder_out, decoder_out)


def _extract_encoder_config(model_dict: dict) -> dict:
    """Extract encoder configuration from checkpoint weights."""
    # Count layers per stage
    layer_counts = {}
    encoder_dims = {}
    ff1_dims = {}
    ff2_dims = {}
    ff3_dims = {}

    for k in model_dict.keys():
        if 'encoder.encoders' not in k or 'layers' not in k:
            continue

        parts = k.split('.')
        try:
            # Determine stage and layer indices
            stage_idx = int(parts[2])

            # Check if DownsampledZipformer2Encoder or direct Zipformer2Encoder
            if 'encoder.layers' in k:
                layer_idx = int(parts[5])
            else:
                layer_idx = int(parts[4])

            if stage_idx not in layer_counts:
                layer_counts[stage_idx] = set()
            layer_counts[stage_idx].add(layer_idx)

            # Extract encoder dim and ff dims from feedforward modules
            if 'feed_forward1.in_proj.weight' in k:
                if stage_idx not in encoder_dims:
                    encoder_dims[stage_idx] = model_dict[k].shape[1]
                if stage_idx not in ff1_dims:
                    ff1_dims[stage_idx] = model_dict[k].shape[0]
            if 'feed_forward2.in_proj.weight' in k and stage_idx not in ff2_dims:
                ff2_dims[stage_idx] = model_dict[k].shape[0]
            if 'feed_forward3.in_proj.weight' in k and stage_idx not in ff3_dims:
                ff3_dims[stage_idx] = model_dict[k].shape[0]

        except (ValueError, IndexError):
            pass

    num_stages = max(layer_counts.keys()) + 1 if layer_counts else 6
    num_encoder_layers = tuple(len(layer_counts.get(i, set())) for i in range(num_stages))
    encoder_dims_tuple = tuple(encoder_dims.get(i, 192) for i in range(num_stages))
    ff1_dims_tuple = tuple(ff1_dims.get(i, 384) for i in range(num_stages))
    ff2_dims_tuple = tuple(ff2_dims.get(i, ff1_dims_tuple[i]) for i in range(num_stages))
    ff3_dims_tuple = tuple(ff3_dims.get(i, ff1_dims_tuple[i]) for i in range(num_stages))

    return {
        'num_encoder_layers': num_encoder_layers,
        'encoder_dims': encoder_dims_tuple,
        # Back-compat: keep old name as "ff1"
        'feedforward_dims': ff1_dims_tuple,
        'ff1_dims': ff1_dims_tuple,
        'ff2_dims': ff2_dims_tuple,
        'ff3_dims': ff3_dims_tuple,
    }


def load_checkpoint(
    checkpoint_path: str,
    config: ASRModelConfig | None = None,
) -> tuple[ASRModel, ASRModelConfig]:
    """
    Load model from PyTorch checkpoint.

    Args:
        checkpoint_path: Path to PyTorch checkpoint file.
        config: Optional config. If None, extract from checkpoint.

    Returns:
        Tuple of (model, config).
    """
    import torch

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model_dict = ckpt.get('model', ckpt)

    # Extract config from weights if not provided or incomplete
    # Get vocab size from decoder embedding
    vocab_size = model_dict['decoder.embedding.weight'].shape[0]
    decoder_dim = model_dict['decoder.embedding.weight'].shape[1]
    context_size = model_dict['decoder.conv.weight'].shape[2]

    # Get joiner dim
    joiner_dim = model_dict['joiner.encoder_proj.weight'].shape[0]

    # Extract encoder config from weights
    encoder_cfg = _extract_encoder_config(model_dict)

    config = ASRModelConfig(
        vocab_size=vocab_size,
        decoder_dim=decoder_dim,
        context_size=context_size,
        joiner_dim=joiner_dim,
        num_encoder_layers=encoder_cfg['num_encoder_layers'],
        encoder_dims=encoder_cfg['encoder_dims'],
        feedforward_dims=encoder_cfg['feedforward_dims'],
        ff1_dims=encoder_cfg.get('ff1_dims'),
        ff2_dims=encoder_cfg.get('ff2_dims'),
        ff3_dims=encoder_cfg.get('ff3_dims'),
    )

    # Create model
    model = ASRModel(config)

    # Load encoder weights (handled separately due to complexity)
    _load_encoder_weights(model.encoder, model_dict)

    # Load decoder weights
    model.decoder.embedding.weight = mx.array(
        model_dict['decoder.embedding.weight'].numpy(),
    )
    conv_weight = model_dict['decoder.conv.weight'].numpy()
    model.decoder.conv.weight = mx.array(np.transpose(conv_weight, (0, 2, 1)))

    # Load joiner weights
    model.joiner.encoder_proj.weight = mx.array(
        model_dict['joiner.encoder_proj.weight'].numpy(),
    )
    model.joiner.encoder_proj.bias = mx.array(
        model_dict['joiner.encoder_proj.bias'].numpy(),
    )
    model.joiner.decoder_proj.weight = mx.array(
        model_dict['joiner.decoder_proj.weight'].numpy(),
    )
    model.joiner.decoder_proj.bias = mx.array(
        model_dict['joiner.decoder_proj.bias'].numpy(),
    )
    model.joiner.output_linear.weight = mx.array(
        model_dict['joiner.output_linear.weight'].numpy(),
    )
    model.joiner.output_linear.bias = mx.array(
        model_dict['joiner.output_linear.bias'].numpy(),
    )

    return model, config


def _load_encoder_weights(encoder: Zipformer, model_dict: dict) -> None:
    """
    Load encoder weights from checkpoint.

    This is a complex operation due to the nested structure of Zipformer.
    """
    from .convert_weights import convert_conv1d_weight, convert_conv2d_weight

    # Helper to get and convert weight
    def get_weight(key: str, conv_type: str | None = None):
        if key not in model_dict:
            return None
        tensor = model_dict[key]
        if conv_type == 'conv1d':
            return convert_conv1d_weight(tensor)
        if conv_type == 'conv2d':
            return convert_conv2d_weight(tensor)
        return mx.array(tensor.numpy())

    # Load encoder_embed weights (note: key prefix is 'encoder_embed.' not 'encoder.encoder_embed.')
    prefix = 'encoder_embed.'

    # Conv stack: conv.0, conv.4, conv.7 (indices in Sequential)
    if f'{prefix}conv.0.weight' in model_dict:
        encoder.encoder_embed.conv0_weight = get_weight(f'{prefix}conv.0.weight', 'conv2d')
        encoder.encoder_embed.conv0_bias = get_weight(f'{prefix}conv.0.bias')
    if f'{prefix}conv.4.weight' in model_dict:
        encoder.encoder_embed.conv4_weight = get_weight(f'{prefix}conv.4.weight', 'conv2d')
        encoder.encoder_embed.conv4_bias = get_weight(f'{prefix}conv.4.bias')
    if f'{prefix}conv.7.weight' in model_dict:
        encoder.encoder_embed.conv7_weight = get_weight(f'{prefix}conv.7.weight', 'conv2d')
        encoder.encoder_embed.conv7_bias = get_weight(f'{prefix}conv.7.bias')

    # ConvNext block
    if f'{prefix}convnext.depthwise_conv.weight' in model_dict:
        encoder.encoder_embed.convnext_dw_weight = get_weight(f'{prefix}convnext.depthwise_conv.weight', 'conv2d')
        encoder.encoder_embed.convnext_dw_bias = get_weight(f'{prefix}convnext.depthwise_conv.bias')
    if f'{prefix}convnext.pointwise_conv1.weight' in model_dict:
        encoder.encoder_embed.convnext_pw1_weight = get_weight(f'{prefix}convnext.pointwise_conv1.weight', 'conv2d')
        encoder.encoder_embed.convnext_pw1_bias = get_weight(f'{prefix}convnext.pointwise_conv1.bias')
    if f'{prefix}convnext.pointwise_conv2.weight' in model_dict:
        encoder.encoder_embed.convnext_pw2_weight = get_weight(f'{prefix}convnext.pointwise_conv2.weight', 'conv2d')
        encoder.encoder_embed.convnext_pw2_bias = get_weight(f'{prefix}convnext.pointwise_conv2.bias')

    # Output projection and norm
    if f'{prefix}out.weight' in model_dict:
        encoder.encoder_embed.out.weight = get_weight(f'{prefix}out.weight')
        encoder.encoder_embed.out.bias = get_weight(f'{prefix}out.bias')
    if f'{prefix}out_norm.bias' in model_dict:
        encoder.encoder_embed.out_norm_bias = get_weight(f'{prefix}out_norm.bias')
        encoder.encoder_embed.out_norm_log_scale = get_weight(f'{prefix}out_norm.log_scale')

    # Load encoder downsample_output bias
    if 'encoder.downsample_output.bias' in model_dict:
        encoder.downsample_output_bias = get_weight('encoder.downsample_output.bias')

    # Load encoder stage weights
    for stage_idx, stage in enumerate(encoder.encoders):
        stage_prefix = f'encoder.encoders.{stage_idx}.'

        # Handle both Zipformer2Encoder (has .layers) and DownsampledZipformer2Encoder (has .encoder.layers)
        if hasattr(stage, 'layers'):
            # Direct Zipformer2Encoder
            encoder_layers = stage.layers
            layer_prefix_base = f'{stage_prefix}layers.'
        elif hasattr(stage, 'encoder') and hasattr(stage.encoder, 'layers'):
            # DownsampledZipformer2Encoder wrapping Zipformer2Encoder
            encoder_layers = stage.encoder.layers
            layer_prefix_base = f'{stage_prefix}encoder.layers.'

            # Load downsample bias for DownsampledZipformer2Encoder
            if f'{stage_prefix}downsample.bias' in model_dict:
                stage.downsample.bias = get_weight(f'{stage_prefix}downsample.bias')

            # Load out_combiner bypass_scale
            if f'{stage_prefix}out_combiner.bypass_scale' in model_dict:
                stage.out_combiner.bypass_scale = get_weight(f'{stage_prefix}out_combiner.bypass_scale')
        else:
            continue

        for layer_idx, layer in enumerate(encoder_layers):
            layer_prefix = f'{layer_prefix_base}{layer_idx}.'

            # Self attention weights (Q, K, P projections for computing attention)
            _load_attn_weights_module(layer.self_attn_weights, model_dict, f'{layer_prefix}self_attn_weights.')

            # Self attention (value projections)
            _load_attention_weights(layer.self_attn1, model_dict, f'{layer_prefix}self_attn1.', stage_idx)
            _load_attention_weights(layer.self_attn2, model_dict, f'{layer_prefix}self_attn2.', stage_idx)

            # Nonlin attention
            if f'{layer_prefix}nonlin_attention.in_proj.weight' in model_dict:
                layer.nonlin_attention.in_proj.weight = get_weight(f'{layer_prefix}nonlin_attention.in_proj.weight')
                layer.nonlin_attention.in_proj.bias = get_weight(f'{layer_prefix}nonlin_attention.in_proj.bias')
            if f'{layer_prefix}nonlin_attention.out_proj.weight' in model_dict:
                layer.nonlin_attention.out_proj.weight = get_weight(f'{layer_prefix}nonlin_attention.out_proj.weight')
                layer.nonlin_attention.out_proj.bias = get_weight(f'{layer_prefix}nonlin_attention.out_proj.bias')

            # Feed forward
            _load_feedforward_weights(layer.feed_forward1, model_dict, f'{layer_prefix}feed_forward1.')
            _load_feedforward_weights(layer.feed_forward2, model_dict, f'{layer_prefix}feed_forward2.')
            _load_feedforward_weights(layer.feed_forward3, model_dict, f'{layer_prefix}feed_forward3.')

            # Conv modules
            _load_conv_module_weights(layer.conv_module1, model_dict, f'{layer_prefix}conv_module1.')
            _load_conv_module_weights(layer.conv_module2, model_dict, f'{layer_prefix}conv_module2.')

            # BiasNorm (norm.bias and norm.log_scale)
            # Note: These are critical for proper normalization
            if f'{layer_prefix}norm.bias' in model_dict:
                layer.norm.bias = get_weight(f'{layer_prefix}norm.bias')
            if f'{layer_prefix}norm.log_scale' in model_dict:
                layer.norm.log_scale = get_weight(f'{layer_prefix}norm.log_scale')

            # Bypass modules - critical for proper skip connections
            if f'{layer_prefix}bypass.bypass_scale' in model_dict:
                layer.bypass.bypass_scale = get_weight(f'{layer_prefix}bypass.bypass_scale')
            if f'{layer_prefix}bypass_mid.bypass_scale' in model_dict:
                layer.bypass_mid.bypass_scale = get_weight(f'{layer_prefix}bypass_mid.bypass_scale')

            # Overall bypass scale
            if f'{layer_prefix}bypass_scale' in model_dict:
                layer.bypass_scale = get_weight(f'{layer_prefix}bypass_scale')


def _load_attn_weights_module(attn_weights, model_dict: dict, prefix: str) -> None:
    """Load attention weight computation module (Q, K, P projections)."""
    if f'{prefix}in_proj.weight' in model_dict:
        attn_weights.in_proj.weight = mx.array(model_dict[f'{prefix}in_proj.weight'].numpy())
    if f'{prefix}in_proj.bias' in model_dict:
        attn_weights.in_proj.bias = mx.array(model_dict[f'{prefix}in_proj.bias'].numpy())
    if f'{prefix}linear_pos.weight' in model_dict:
        attn_weights.linear_pos.weight = mx.array(model_dict[f'{prefix}linear_pos.weight'].numpy())


def _load_attention_weights(attn, model_dict: dict, prefix: str, stage_idx: int) -> None:
    """Load self-attention weights (value projection)."""
    if f'{prefix}in_proj.weight' in model_dict:
        attn.in_proj.weight = mx.array(model_dict[f'{prefix}in_proj.weight'].numpy())
    if f'{prefix}in_proj.bias' in model_dict:
        attn.in_proj.bias = mx.array(model_dict[f'{prefix}in_proj.bias'].numpy())
    if f'{prefix}linear_pos.weight' in model_dict:
        attn.linear_pos.weight = mx.array(model_dict[f'{prefix}linear_pos.weight'].numpy())
    if f'{prefix}out_proj.weight' in model_dict:
        attn.out_proj.weight = mx.array(model_dict[f'{prefix}out_proj.weight'].numpy())
        attn.out_proj.bias = mx.array(model_dict[f'{prefix}out_proj.bias'].numpy())


def _load_feedforward_weights(ff, model_dict: dict, prefix: str) -> None:
    """Load feedforward module weights."""
    if f'{prefix}in_proj.weight' in model_dict:
        ff.in_proj.weight = mx.array(model_dict[f'{prefix}in_proj.weight'].numpy())
        ff.in_proj.bias = mx.array(model_dict[f'{prefix}in_proj.bias'].numpy())
    if f'{prefix}out_proj.weight' in model_dict:
        ff.out_proj.weight = mx.array(model_dict[f'{prefix}out_proj.weight'].numpy())
        ff.out_proj.bias = mx.array(model_dict[f'{prefix}out_proj.bias'].numpy())


def _load_conv_module_weights(conv_mod, model_dict: dict, prefix: str) -> None:
    """Load convolution module weights.

    The conv module has:
    - in_proj: Linear(d_model, 2*d_model) for input projection + gating
    - causal_conv_weight/bias: depthwise causal conv
    - chunkwise_conv_weight/bias/scale: depthwise chunkwise conv
    - out_proj: Linear(d_model, d_model) for output
    """

    # Input projection (not pointwise_conv, just Linear)
    if f'{prefix}in_proj.weight' in model_dict:
        conv_mod.in_proj.weight = mx.array(
            model_dict[f'{prefix}in_proj.weight'].numpy(),
        )
        conv_mod.in_proj.bias = mx.array(
            model_dict[f'{prefix}in_proj.bias'].numpy(),
        )

    # Causal depthwise conv weights
    if f'{prefix}depthwise_conv.causal_conv.weight' in model_dict:
        weight = model_dict[f'{prefix}depthwise_conv.causal_conv.weight']
        # PyTorch: (C_out, C_in/groups, kernel) -> MLX: (C_out, kernel, C_in/groups)
        conv_mod.causal_conv_weight = mx.array(
            weight.numpy().transpose(0, 2, 1),
        )
        conv_mod.causal_conv_bias = mx.array(
            model_dict[f'{prefix}depthwise_conv.causal_conv.bias'].numpy(),
        )

    # Chunkwise depthwise conv weights
    if f'{prefix}depthwise_conv.chunkwise_conv.weight' in model_dict:
        weight = model_dict[f'{prefix}depthwise_conv.chunkwise_conv.weight']
        conv_mod.chunkwise_conv_weight = mx.array(
            weight.numpy().transpose(0, 2, 1),
        )
        conv_mod.chunkwise_conv_bias = mx.array(
            model_dict[f'{prefix}depthwise_conv.chunkwise_conv.bias'].numpy(),
        )

    # Chunkwise conv scale
    if f'{prefix}depthwise_conv.chunkwise_conv_scale' in model_dict:
        conv_mod.chunkwise_conv_scale = mx.array(
            model_dict[f'{prefix}depthwise_conv.chunkwise_conv_scale'].numpy(),
        )

    # Output projection
    if f'{prefix}out_proj.weight' in model_dict:
        conv_mod.out_proj.weight = mx.array(
            model_dict[f'{prefix}out_proj.weight'].numpy(),
        )
        conv_mod.out_proj.bias = mx.array(
            model_dict[f'{prefix}out_proj.bias'].numpy(),
        )
