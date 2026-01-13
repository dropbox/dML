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
MLX Implementation of Wav2Vec2 and HuBERT for CTC-based ASR.

Both models share the same architecture:
- 7-layer CNN feature extractor with group normalization
- 24-layer transformer encoder
- CTC head for character-level transcription

Based on HuggingFace transformers Wav2Vec2/HuBERT implementations.
"""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class Wav2Vec2Config:
    """Configuration for Wav2Vec2/HuBERT models."""

    # Feature extractor
    conv_dim: tuple[int, ...] = (512, 512, 512, 512, 512, 512, 512)
    conv_kernel: tuple[int, ...] = (10, 3, 3, 3, 3, 2, 2)
    conv_stride: tuple[int, ...] = (5, 2, 2, 2, 2, 2, 2)
    conv_bias: bool = True
    feat_extract_norm: str = "layer"  # "group" or "layer"
    feat_extract_activation: str = "gelu"

    # Transformer
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    hidden_act: str = "gelu"
    hidden_dropout: float = 0.1
    attention_dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    do_stable_layer_norm: bool = True  # Pre-LN vs Post-LN

    # Positional encoding
    num_conv_pos_embeddings: int = 128
    num_conv_pos_embedding_groups: int = 16

    # CTC
    vocab_size: int = 32
    pad_token_id: int = 0

    # Model type
    model_type: str = "wav2vec2"  # "wav2vec2" or "hubert"

    @classmethod
    def from_hf(cls, hf_config) -> "Wav2Vec2Config":
        """Create config from HuggingFace config."""
        return cls(
            conv_dim=tuple(hf_config.conv_dim),
            conv_kernel=tuple(hf_config.conv_kernel),
            conv_stride=tuple(hf_config.conv_stride),
            conv_bias=hf_config.conv_bias,
            feat_extract_norm=hf_config.feat_extract_norm,
            feat_extract_activation=hf_config.feat_extract_activation,
            hidden_size=hf_config.hidden_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            intermediate_size=hf_config.intermediate_size,
            hidden_act=hf_config.hidden_act,
            hidden_dropout=hf_config.hidden_dropout,
            attention_dropout=hf_config.attention_dropout,
            layer_norm_eps=hf_config.layer_norm_eps,
            do_stable_layer_norm=hf_config.do_stable_layer_norm,
            num_conv_pos_embeddings=hf_config.num_conv_pos_embeddings,
            num_conv_pos_embedding_groups=hf_config.num_conv_pos_embedding_groups,
            vocab_size=hf_config.vocab_size,
            pad_token_id=hf_config.pad_token_id,
            model_type=hf_config.model_type,
        )


class Wav2Vec2GroupNormConvLayer(nn.Module):
    """Single convolutional layer with group normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        bias: bool = True,
        layer_id: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layer_id = layer_id

        # Conv1d: MLX uses NLC format (batch, length, channels)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
            padding=0,  # No padding, handled separately
        )

        # Group normalization (num_groups=out_channels means 1 channel per group)
        self.layer_norm = nn.GroupNorm(
            num_groups=out_channels, dims=out_channels, affine=True,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: [batch, length, channels]
        Returns:
            [batch, new_length, out_channels]
        """
        x = self.conv(x)
        x = self.layer_norm(x)
        return nn.gelu(x)


class Wav2Vec2LayerNormConvLayer(nn.Module):
    """Single convolutional layer with layer normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        bias: bool = True,
        layer_id: int = 0,
    ):
        super().__init__()
        self.layer_id = layer_id

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
            padding=0,
        )

        self.layer_norm = nn.LayerNorm(out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv(x)
        x = self.layer_norm(x)
        return nn.gelu(x)


class Wav2Vec2NoLayerNormConvLayer(nn.Module):
    """First convolutional layer without normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        bias: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
            padding=0,
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv(x)
        return nn.gelu(x)


class Wav2Vec2FeatureExtractor(nn.Module):
    """
    7-layer CNN that converts raw audio to features.

    Input: [batch, audio_samples]
    Output: [batch, sequence_length, 512]
    """

    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.config = config

        # Build conv layers
        conv_layers = []

        # First layer: no normalization
        conv_layers.append(
            Wav2Vec2NoLayerNormConvLayer(
                in_channels=1,
                out_channels=config.conv_dim[0],
                kernel_size=config.conv_kernel[0],
                stride=config.conv_stride[0],
                bias=config.conv_bias,
            ),
        )

        # Remaining layers with normalization
        for i in range(1, len(config.conv_dim)):
            if config.feat_extract_norm == "group":
                conv_layers.append(
                    Wav2Vec2GroupNormConvLayer(
                        in_channels=config.conv_dim[i - 1],
                        out_channels=config.conv_dim[i],
                        kernel_size=config.conv_kernel[i],
                        stride=config.conv_stride[i],
                        bias=config.conv_bias,
                        layer_id=i,
                    ),
                )
            else:
                conv_layers.append(
                    Wav2Vec2LayerNormConvLayer(
                        in_channels=config.conv_dim[i - 1],
                        out_channels=config.conv_dim[i],
                        kernel_size=config.conv_kernel[i],
                        stride=config.conv_stride[i],
                        bias=config.conv_bias,
                        layer_id=i,
                    ),
                )

        self.conv_layers = conv_layers

    def __call__(self, input_values: mx.array) -> mx.array:
        """
        Args:
            input_values: [batch, audio_samples]
        Returns:
            [batch, sequence_length, conv_dim[-1]]
        """
        # Add channel dimension: [batch, samples] -> [batch, samples, 1]
        hidden_states = input_values[:, :, None]

        for conv_layer in self.conv_layers:
            hidden_states = conv_layer(hidden_states)

        return hidden_states


class Wav2Vec2FeatureProjection(nn.Module):
    """Projects CNN features to transformer hidden size."""

    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.layer_norm_eps)
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.layer_norm(hidden_states)
        return self.projection(hidden_states)
        # Note: dropout disabled during inference


class GroupedConv1dWeight(nn.Module):
    """Stores grouped conv weights as MLX parameters."""

    def __init__(self, out_channels: int, kernel_size: int, in_channels_per_group: int):
        super().__init__()
        # Weight shape: [out_channels, kernel_size, in_channels/groups]
        self.weight = mx.zeros((out_channels, kernel_size, in_channels_per_group))


class GroupedConv1dBias(nn.Module):
    """Stores grouped conv bias as MLX parameter."""

    def __init__(self, out_channels: int):
        super().__init__()
        self.bias = mx.zeros((out_channels,))


class Wav2Vec2PositionalConvEmbedding(nn.Module):
    """
    Convolutional positional embedding.

    Uses grouped convolution to add relative positional information.
    MLX doesn't support grouped Conv1d, so we implement it manually.
    """

    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.num_groups = config.num_conv_pos_embedding_groups
        self.hidden_size = config.hidden_size
        self.kernel_size = config.num_conv_pos_embeddings
        self.group_size = config.hidden_size // self.num_groups
        self.padding = config.num_conv_pos_embeddings // 2

        # Store grouped conv weights using submodules
        self.conv = GroupedConv1dWeight(
            config.hidden_size,
            config.num_conv_pos_embeddings,
            self.group_size,
        )
        self.conv_bias = GroupedConv1dBias(config.hidden_size)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """
        Grouped 1D convolution.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
        Returns:
            [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Pad the sequence
        padded = mx.pad(
            hidden_states,
            [(0, 0), (self.padding, self.padding), (0, 0)],
        )

        # Implement grouped convolution
        # Split into groups and convolve each group separately
        outputs = []
        for g in range(self.num_groups):
            # Get input channels for this group
            start_in = g * self.group_size
            end_in = (g + 1) * self.group_size

            # Get output channels for this group
            start_out = g * self.group_size
            end_out = (g + 1) * self.group_size

            # Input slice: [batch, padded_seq, group_size]
            group_input = padded[:, :, start_in:end_in]

            # Weight slice: [group_size, kernel_size, group_size]
            group_weight = self.conv.weight[start_out:end_out, :, :]

            # Convolve
            group_output = mx.conv1d(
                group_input,
                group_weight,
                stride=1,
                padding=0,
            )
            outputs.append(group_output)

        # Concatenate group outputs: [batch, conv_out_len, hidden_size]
        result = mx.concatenate(outputs, axis=-1)

        # Add bias
        result = result + self.conv_bias.bias

        # Trim to original length (conv output may be slightly different due to kernel size)
        if result.shape[1] > seq_len:
            # Remove extra frames
            remove_start = (result.shape[1] - seq_len) // 2
            result = result[:, remove_start:remove_start + seq_len, :]
        elif result.shape[1] < seq_len:
            # Pad if needed (shouldn't happen with correct padding)
            pad_amount = seq_len - result.shape[1]
            result = mx.pad(result, [(0, 0), (0, pad_amount), (0, 0)])

        return result


class Wav2Vec2Attention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        """
        Args:
            hidden_states: [batch, seq_len, embed_dim]
            attention_mask: Optional mask
        Returns:
            [batch, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Reshape for multi-head: [batch, seq, heads, head_dim]
        query = query.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        value = value.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose to [batch, heads, seq, head_dim]
        query = query.transpose(0, 2, 1, 3)
        key = key.transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        attn_weights = (query @ key.transpose(0, 1, 3, 2)) * self.scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = mx.softmax(attn_weights, axis=-1)

        # Apply attention
        attn_output = attn_weights @ value

        # Reshape back
        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)

        # Output projection
        return self.out_proj(attn_output)


class Wav2Vec2FeedForward(nn.Module):
    """Feed-forward network."""

    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = nn.gelu(hidden_states)
        return self.output_dense(hidden_states)


class Wav2Vec2EncoderLayer(nn.Module):
    """Single transformer encoder layer."""

    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.attention = Wav2Vec2Attention(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = Wav2Vec2FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps,
        )
        self.do_stable_layer_norm = config.do_stable_layer_norm

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        if self.do_stable_layer_norm:
            # Pre-LN (stable)
            attn_residual = hidden_states
            hidden_states = self.layer_norm(hidden_states)
            hidden_states = self.attention(hidden_states, attention_mask)
            hidden_states = attn_residual + hidden_states

            ffn_residual = hidden_states
            hidden_states = self.final_layer_norm(hidden_states)
            hidden_states = self.feed_forward(hidden_states)
            hidden_states = ffn_residual + hidden_states
        else:
            # Post-LN
            hidden_states = hidden_states + self.attention(hidden_states, attention_mask)
            hidden_states = self.layer_norm(hidden_states)
            hidden_states = hidden_states + self.feed_forward(hidden_states)
            hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


class Wav2Vec2Encoder(nn.Module):
    """Stack of transformer encoder layers."""

    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.config = config
        self.pos_conv_embed = Wav2Vec2PositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layers = [
            Wav2Vec2EncoderLayer(config) for _ in range(config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        # Add positional embeddings
        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings

        hidden_states = self.layer_norm(hidden_states)

        # Apply encoder layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        return hidden_states


class Wav2Vec2ForCTC(nn.Module):
    """
    Wav2Vec2/HuBERT model for CTC-based ASR.

    Architecture:
    1. Feature extractor (7 CNN layers)
    2. Feature projection
    3. Transformer encoder (24 layers)
    4. CTC head (linear to vocab)

    Usage:
        config = Wav2Vec2Config()
        model = Wav2Vec2ForCTC(config)

        # Input: raw audio
        audio = mx.array(...)  # [batch, samples]
        logits = model(audio)   # [batch, seq_len, vocab_size]

        # CTC decoding
        predictions = mx.argmax(logits, axis=-1)
    """

    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.config = config

        self.feature_extractor = Wav2Vec2FeatureExtractor(config)
        self.feature_projection = Wav2Vec2FeatureProjection(config)
        self.encoder = Wav2Vec2Encoder(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

    def __call__(
        self,
        input_values: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        """
        Args:
            input_values: Raw audio [batch, samples]
            attention_mask: Optional attention mask

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        # Extract features from audio
        hidden_states = self.feature_extractor(input_values)

        # Project to transformer dimension
        hidden_states = self.feature_projection(hidden_states)

        # Prepare attention mask if provided
        if attention_mask is not None:
            # Compute output lengths after CNN downsampling
            # TODO: Properly compute mask for downsampled sequence
            pass

        # Transformer encoder
        hidden_states = self.encoder(hidden_states, attention_mask)

        # CTC head
        return self.lm_head(hidden_states)


    def transcribe(
        self,
        input_values: mx.array,
        processor=None,
    ) -> str:
        """
        Transcribe audio to text using greedy CTC decoding.

        Args:
            input_values: Raw audio [samples] or [batch, samples]
            processor: Optional HuggingFace processor for decoding

        Returns:
            Transcribed text
        """
        # Add batch dimension if needed
        if input_values.ndim == 1:
            input_values = input_values[None, :]

        # Get logits
        logits = self(input_values)
        mx.eval(logits)

        # Greedy decoding
        predicted_ids = mx.argmax(logits, axis=-1)

        # Decode to text
        if processor is not None:
            # Use HuggingFace processor
            import numpy as np
            return processor.decode(np.array(predicted_ids[0]))
        # Simple CTC decoding (collapse repeats, remove blanks)
        return self._ctc_decode(predicted_ids[0])

    def _ctc_decode(self, predicted_ids: mx.array) -> str:
        """Simple CTC greedy decoding."""
        import numpy as np

        ids = np.array(predicted_ids)

        # Collapse repeated tokens
        prev = None
        collapsed = []
        for idx in ids:
            if idx != prev:
                collapsed.append(idx)
                prev = idx

        # Remove blank tokens (pad_token_id = 0)
        filtered = [idx for idx in collapsed if idx != self.config.pad_token_id]

        # For basic vocab (letters), convert to characters
        # Standard Wav2Vec2 vocab: |, pad, a-z, ', space (32 tokens)
        vocab = ["|", "", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l",
                 "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
                 "'", " "]

        chars = []
        for idx in filtered:
            if 0 <= idx < len(vocab):
                chars.append(vocab[idx])
            else:
                chars.append("?")

        return "".join(chars)

    @staticmethod
    def from_hf(hf_path: str) -> "Wav2Vec2ForCTC":
        """
        Load model with weights from HuggingFace.

        Args:
            hf_path: HuggingFace model path

        Returns:
            Wav2Vec2ForCTC with loaded weights
        """
        try:
            import torch  # noqa: F401 - Required dependency for transformers
            from transformers import AutoConfig, AutoModelForCTC
        except ImportError:
            raise ImportError("transformers and torch required for from_hf()") from None

        # Load HuggingFace config and model
        hf_config = AutoConfig.from_pretrained(hf_path)
        hf_model = AutoModelForCTC.from_pretrained(hf_path)

        # Create MLX config
        config = Wav2Vec2Config.from_hf(hf_config)

        # Create MLX model
        model = Wav2Vec2ForCTC(config)

        # Map weights
        mlx_weights = _map_wav2vec2_weights(hf_model, config)

        # Load weights
        model.load_weights(list(mlx_weights.items()))

        return model


def _map_wav2vec2_weights(hf_model, config: Wav2Vec2Config) -> dict[str, mx.array]:
    """
    Map HuggingFace Wav2Vec2/HuBERT weights to MLX format.

    HuggingFace structure:
    - wav2vec2.feature_extractor.conv_layers.N.{conv,layer_norm}.{weight,bias}
    - wav2vec2.feature_projection.{layer_norm,projection}.{weight,bias}
    - wav2vec2.encoder.pos_conv_embed.conv.{weight,bias}
    - wav2vec2.encoder.layer_norm.{weight,bias}
    - wav2vec2.encoder.layers.N.attention.{q,k,v,out}_proj.{weight,bias}
    - wav2vec2.encoder.layers.N.layer_norm.{weight,bias}
    - wav2vec2.encoder.layers.N.feed_forward.{intermediate_dense,output_dense}.{weight,bias}
    - wav2vec2.encoder.layers.N.final_layer_norm.{weight,bias}
    - lm_head.{weight,bias}
    """
    import numpy as np
    import torch

    mlx_weights = {}
    hf_state = hf_model.state_dict()

    def to_mlx(tensor: torch.Tensor) -> mx.array:
        return mx.array(tensor.detach().cpu().numpy())

    # Determine the base prefix (wav2vec2 or hubert)
    base_prefix = "wav2vec2" if config.model_type == "wav2vec2" else "hubert"

    # Feature extractor conv layers
    for i in range(len(config.conv_dim)):
        hf_prefix = f"{base_prefix}.feature_extractor.conv_layers.{i}"
        mlx_prefix = f"feature_extractor.conv_layers.{i}"

        # Conv weight: HuggingFace [out, in, kernel] -> MLX [out, kernel, in]
        if f"{hf_prefix}.conv.weight" in hf_state:
            w = hf_state[f"{hf_prefix}.conv.weight"].detach().cpu().numpy()
            # PyTorch Conv1d: [out_channels, in_channels, kernel_size]
            # MLX Conv1d: [out_channels, kernel_size, in_channels]
            w = np.transpose(w, (0, 2, 1))
            mlx_weights[f"{mlx_prefix}.conv.weight"] = mx.array(w)

        if f"{hf_prefix}.conv.bias" in hf_state:
            mlx_weights[f"{mlx_prefix}.conv.bias"] = to_mlx(
                hf_state[f"{hf_prefix}.conv.bias"],
            )

        # Layer/group norm (for layers 1+)
        if i > 0:
            if f"{hf_prefix}.layer_norm.weight" in hf_state:
                mlx_weights[f"{mlx_prefix}.layer_norm.weight"] = to_mlx(
                    hf_state[f"{hf_prefix}.layer_norm.weight"],
                )
            if f"{hf_prefix}.layer_norm.bias" in hf_state:
                mlx_weights[f"{mlx_prefix}.layer_norm.bias"] = to_mlx(
                    hf_state[f"{hf_prefix}.layer_norm.bias"],
                )

    # Feature projection
    hf_fp = f"{base_prefix}.feature_projection"
    mlx_weights["feature_projection.layer_norm.weight"] = to_mlx(
        hf_state[f"{hf_fp}.layer_norm.weight"],
    )
    mlx_weights["feature_projection.layer_norm.bias"] = to_mlx(
        hf_state[f"{hf_fp}.layer_norm.bias"],
    )
    mlx_weights["feature_projection.projection.weight"] = to_mlx(
        hf_state[f"{hf_fp}.projection.weight"],
    )
    mlx_weights["feature_projection.projection.bias"] = to_mlx(
        hf_state[f"{hf_fp}.projection.bias"],
    )

    # Encoder positional conv embedding
    # Note: This uses weight normalization with parametrizations
    # We need to get the computed weight from the actual module
    hf_enc = f"{base_prefix}.encoder"

    # Get the actual conv weight through the parametrized module
    pos_conv = hf_model
    for attr in f"{base_prefix}.encoder.pos_conv_embed.conv".split("."):
        pos_conv = getattr(pos_conv, attr)

    # Get the computed weight (after weight normalization)
    if hasattr(pos_conv, 'weight'):
        w = pos_conv.weight.detach().cpu().numpy()
        # PyTorch: [out, in/groups, kernel] with groups=num_groups
        # For grouped conv: weight is [out_channels, in_channels/groups, kernel]
        # Our custom grouped conv expects: [out_channels, kernel_size, in_channels/groups]
        w = np.transpose(w, (0, 2, 1))
        mlx_weights["encoder.pos_conv_embed.conv.weight"] = mx.array(w)

    if f"{hf_enc}.pos_conv_embed.conv.bias" in hf_state:
        mlx_weights["encoder.pos_conv_embed.conv_bias.bias"] = to_mlx(
            hf_state[f"{hf_enc}.pos_conv_embed.conv.bias"],
        )

    # Encoder layer norm
    mlx_weights["encoder.layer_norm.weight"] = to_mlx(
        hf_state[f"{hf_enc}.layer_norm.weight"],
    )
    mlx_weights["encoder.layer_norm.bias"] = to_mlx(
        hf_state[f"{hf_enc}.layer_norm.bias"],
    )

    # Encoder layers
    for i in range(config.num_hidden_layers):
        hf_layer = f"{hf_enc}.layers.{i}"
        mlx_layer = f"encoder.layers.{i}"

        # Attention
        for proj in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            mlx_weights[f"{mlx_layer}.attention.{proj}.weight"] = to_mlx(
                hf_state[f"{hf_layer}.attention.{proj}.weight"],
            )
            mlx_weights[f"{mlx_layer}.attention.{proj}.bias"] = to_mlx(
                hf_state[f"{hf_layer}.attention.{proj}.bias"],
            )

        # Layer norms
        mlx_weights[f"{mlx_layer}.layer_norm.weight"] = to_mlx(
            hf_state[f"{hf_layer}.layer_norm.weight"],
        )
        mlx_weights[f"{mlx_layer}.layer_norm.bias"] = to_mlx(
            hf_state[f"{hf_layer}.layer_norm.bias"],
        )

        # Feed forward
        mlx_weights[f"{mlx_layer}.feed_forward.intermediate_dense.weight"] = to_mlx(
            hf_state[f"{hf_layer}.feed_forward.intermediate_dense.weight"],
        )
        mlx_weights[f"{mlx_layer}.feed_forward.intermediate_dense.bias"] = to_mlx(
            hf_state[f"{hf_layer}.feed_forward.intermediate_dense.bias"],
        )
        mlx_weights[f"{mlx_layer}.feed_forward.output_dense.weight"] = to_mlx(
            hf_state[f"{hf_layer}.feed_forward.output_dense.weight"],
        )
        mlx_weights[f"{mlx_layer}.feed_forward.output_dense.bias"] = to_mlx(
            hf_state[f"{hf_layer}.feed_forward.output_dense.bias"],
        )

        # Final layer norm
        mlx_weights[f"{mlx_layer}.final_layer_norm.weight"] = to_mlx(
            hf_state[f"{hf_layer}.final_layer_norm.weight"],
        )
        mlx_weights[f"{mlx_layer}.final_layer_norm.bias"] = to_mlx(
            hf_state[f"{hf_layer}.final_layer_norm.bias"],
        )

    # LM head
    mlx_weights["lm_head.weight"] = to_mlx(hf_state["lm_head.weight"])
    mlx_weights["lm_head.bias"] = to_mlx(hf_state["lm_head.bias"])

    return mlx_weights


# Alias for HuBERT (same architecture)
HuBERTForCTC = Wav2Vec2ForCTC
