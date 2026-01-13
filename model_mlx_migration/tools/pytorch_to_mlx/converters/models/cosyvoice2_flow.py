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
CosyVoice2 Flow Matching Model

Conditional flow matching model that converts speech tokens to mel spectrograms.
This is the core innovation of CosyVoice2.

Architecture:
- Token embedding (6561 -> 512)
- Pre-lookahead convolution layers
- 6 transformer encoder layers
- UNet-style decoder with time embedding
- Speaker embedding projection (192 -> 80)
- Output: 80-dim mel spectrogram

Input: speech tokens [batch, seq_len] + speaker embedding [batch, 192]
Output: mel spectrogram [batch, mel_len, 80]
"""

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class FlowMatchingConfig:
    """Configuration for CosyVoice2 flow matching model (flow.pt)."""

    # Token embedding
    vocab_size: int = 6561  # Speech token vocabulary
    embed_dim: int = 512  # Embedding dimension

    # Encoder
    num_encoder_layers: int = 6
    encoder_attention_heads: int = 8
    encoder_ffn_dim: int = 2048
    encoder_dropout: float = 0.1

    # Pre-lookahead convolution
    pre_lookahead_kernel1: int = 4
    pre_lookahead_kernel2: int = 3

    # Decoder (UNet-style)
    decoder_channels: int = 256
    time_embed_dim: int = 320
    time_mlp_dim: int = 1024
    num_down_blocks: int = 4
    num_mid_blocks: int = 2
    num_up_blocks: int = 4

    # Output
    mel_dim: int = 80
    speaker_embed_dim: int = 192


def sinusoidal_embedding(
    timesteps: mx.array, dim: int, scale: float = 1000.0,
) -> mx.array:
    """
    Create sinusoidal position embeddings for timesteps.

    Matches the Matcha-TTS SinusoidalPosEmb formula:
    emb = scale * t * exp(-log(10000) * i / (half_dim - 1))
    output = [sin(emb), cos(emb)]

    Args:
        timesteps: [batch] - Diffusion timesteps
        dim: Embedding dimension
        scale: Scaling factor (default 1000.0 to match Matcha)

    Returns:
        embeddings: [batch, dim]
    """
    half_dim = dim // 2
    # Match Matcha formula: log(10000) / (half_dim - 1)
    exponent = (
        -math.log(10000.0) * mx.arange(0, half_dim, dtype=mx.float32) / (half_dim - 1)
    )
    # Apply scale factor (Matcha uses scale=1000)
    emb = scale * timesteps[:, None] * mx.exp(exponent)[None, :]
    return mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)


class PreLookaheadLayer(nn.Module):
    """
    Pre-lookahead convolution layers for encoder.

    Two 1D convolutions that provide local context before transformer.
    """

    def __init__(self, config: FlowMatchingConfig):
        super().__init__()
        dim = config.embed_dim

        # conv1: [512, 512, 4] - first lookahead conv
        self.conv1 = nn.Conv1d(dim, dim, config.pre_lookahead_kernel1)

        # conv2: [512, 512, 3] - second lookahead conv
        self.conv2 = nn.Conv1d(dim, dim, config.pre_lookahead_kernel2)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: [batch, seq_len, embed_dim] - NLC format

        Returns:
            y: [batch, seq_len', embed_dim]
        """
        # Conv1d expects NLC, so we can use directly
        x = self.conv1(x)
        x = nn.relu(x)
        x = self.conv2(x)
        return nn.relu(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention for encoder.
    """

    def __init__(self, config: FlowMatchingConfig):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.num_heads = config.encoder_attention_heads
        self.head_dim = config.embed_dim // config.encoder_attention_heads
        self.scale = self.head_dim**-0.5

        # Projections: q, k, v, out all [512, 512]
        self.q_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.k_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.v_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        """
        Args:
            x: [batch, seq_len, embed_dim]
            mask: Optional attention mask

        Returns:
            y: [batch, seq_len, embed_dim]
        """
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3,
        )
        k = k.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3,
        )
        v = v.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3,
        )

        # Attention scores
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        if mask is not None:
            scores = scores + mask

        attn = mx.softmax(scores, axis=-1)

        # Apply attention to values
        out = attn @ v

        # Reshape back
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.embed_dim)

        # Output projection
        return self.out_proj(out)



class TransformerEncoderLayer(nn.Module):
    """
    Single transformer encoder layer.

    Standard transformer with:
    - Self-attention with residual
    - FFN with residual
    - Pre-norm architecture
    """

    def __init__(self, config: FlowMatchingConfig):
        super().__init__()

        # Self-attention
        self.self_attn = MultiHeadAttention(config)

        # Norms (RMSNorm based on PyTorch norm.weight shape [512])
        self.norm1 = nn.RMSNorm(config.embed_dim)
        self.norm2 = nn.RMSNorm(config.embed_dim)

        # FFN
        self.linear1 = nn.Linear(config.embed_dim, config.encoder_ffn_dim)
        self.linear2 = nn.Linear(config.encoder_ffn_dim, config.embed_dim)

        self.dropout = config.encoder_dropout

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        """
        Args:
            x: [batch, seq_len, embed_dim]
            mask: Optional attention mask

        Returns:
            y: [batch, seq_len, embed_dim]
        """
        # Self-attention with residual
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, mask)
        x = residual + x

        # FFN with residual
        residual = x
        x = self.norm2(x)
        x = self.linear1(x)
        x = nn.relu(x)
        x = self.linear2(x)
        return residual + x



class FlowEncoder(nn.Module):
    """
    Encoder for flow matching model.

    Pre-lookahead convolutions + 6 transformer layers.
    """

    def __init__(self, config: FlowMatchingConfig):
        super().__init__()

        # Pre-lookahead convolutions
        self.pre_lookahead_layer = PreLookaheadLayer(config)

        # Transformer encoder layers
        self.encoder_layer = [
            TransformerEncoderLayer(config) for _ in range(config.num_encoder_layers)
        ]

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        """
        Args:
            x: [batch, seq_len, embed_dim]
            mask: Optional attention mask

        Returns:
            y: [batch, seq_len', embed_dim]
        """
        # Pre-lookahead convolution
        x = self.pre_lookahead_layer(x)

        # Transformer layers
        for layer in self.encoder_layer:
            x = layer(x, mask)

        return x


class TimeEmbedding(nn.Module):
    """
    Time embedding MLP for diffusion timesteps.

    Sinusoidal embedding -> Linear -> SiLU -> Linear
    """

    def __init__(self, config: FlowMatchingConfig):
        super().__init__()

        # Time embedding MLP
        self.linear_1 = nn.Linear(config.time_embed_dim, config.time_mlp_dim)
        self.linear_2 = nn.Linear(config.time_mlp_dim, config.time_mlp_dim)

    def __call__(self, t: mx.array) -> mx.array:
        """
        Args:
            t: [batch] - Diffusion timesteps (0 to 1)

        Returns:
            embedding: [batch, time_mlp_dim]
        """
        # Create sinusoidal embedding
        emb = sinusoidal_embedding(t, 320)  # time_embed_dim

        # MLP
        emb = self.linear_1(emb)
        emb = nn.silu(emb)
        return self.linear_2(emb)



# ====================
# DiT-style Decoder Blocks (matches actual flow.pt architecture)
# ====================


class DiTConvBlock(nn.Module):
    """
    DiT-style convolutional block with time injection.

    Structure from flow.pt:
    - block1: Conv1d -> SiLU -> GroupNorm
    - block2: Conv1d -> SiLU -> GroupNorm
    - mlp: Linear (time projection)
    - res_conv: Conv1d kernel=1 (skip connection)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int = 1024,
    ):
        super().__init__()
        self.res_conv: nn.Conv1d | None = None

        # block1: [out, in, 3]
        self.block1_conv = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.block1_norm = nn.GroupNorm(32, out_channels, pytorch_compatible=True)

        # block2: [out, out, 3]
        self.block2_conv = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.block2_norm = nn.GroupNorm(32, out_channels, pytorch_compatible=True)

        # Time projection: [out, time_channels]
        self.time_mlp = nn.Linear(time_channels, out_channels)

        # Residual connection: [out, in, 1]
        if in_channels != out_channels:
            self.res_conv = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.res_conv = None

    def __call__(self, x: mx.array, time_emb: mx.array) -> mx.array:
        """
        Args:
            x: [batch, seq_len, in_channels] NLC format
            time_emb: [batch, time_channels]

        Returns:
            y: [batch, seq_len, out_channels]

        Note: Matches PyTorch Matcha Block1D: Conv -> GroupNorm -> Mish
        Time embedding: SiLU(time_emb) -> Linear -> broadcast add
        """
        residual = x

        # Block 1: Conv -> GroupNorm -> Mish
        x = self.block1_conv(x)
        x = self.block1_norm(x)
        x = nn.mish(x)

        # Add time embedding: Mish on time_emb, then linear projection
        # (matches Matcha ResnetBlock1D: nn.Sequential(nn.Mish(), nn.Linear(...)))
        time_proj = self.time_mlp(nn.mish(time_emb))[:, None, :]
        x = x + time_proj

        # Block 2: Conv -> GroupNorm -> Mish
        x = self.block2_conv(x)
        x = self.block2_norm(x)
        x = nn.mish(x)

        # Residual connection
        if self.res_conv is not None:
            residual = self.res_conv(residual)

        return x + residual  # type: ignore[no-any-return]


class DiTAttentionBlock(nn.Module):
    """
    DiT-style attention block.

    Structure from flow.pt:
    - norm1 -> attn1 (self-attention with to_q, to_k, to_v, to_out)
    - norm3 -> ff (GEGLU-style feedforward)

    Note: norm2 is skipped (no cross-attention in this model)
    """

    def __init__(
        self,
        channels: int = 256,
        num_heads: int = 8,
        head_dim: int = 64,  # 512/8 = 64
        ffn_dim: int = 1024,
    ):
        super().__init__()
        inner_dim = num_heads * head_dim  # 512

        # Self-attention
        self.norm1 = nn.LayerNorm(channels)
        self.to_q = nn.Linear(channels, inner_dim, bias=False)
        self.to_k = nn.Linear(channels, inner_dim, bias=False)
        self.to_v = nn.Linear(channels, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, channels)

        # FFN with GEGLU
        self.norm3 = nn.LayerNorm(channels)
        self.ff_proj = nn.Linear(channels, ffn_dim)  # GEGLU splits this
        self.ff_out = nn.Linear(ffn_dim, channels)

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: [batch, seq_len, channels]

        Returns:
            y: [batch, seq_len, channels]
        """
        batch, seq_len, _ = x.shape

        # Self-attention
        norm_x = self.norm1(x)
        q = self.to_q(norm_x)
        k = self.to_k(norm_x)
        v = self.to_v(norm_x)

        # Reshape for multi-head attention
        q = q.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3,
        )
        k = k.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3,
        )
        v = v.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3,
        )

        # Attention
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = mx.softmax(scores, axis=-1)
        out = attn @ v

        # Reshape back
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
        out = self.to_out(out)
        x = x + out

        # FFN with GELU (simplified from GEGLU)
        norm_x = self.norm3(x)
        ff = self.ff_proj(norm_x)
        ff = nn.gelu(ff)
        ff = self.ff_out(ff)
        return x + ff



class DiTBlock(nn.Module):
    """
    Complete DiT block: ConvBlock + multiple AttentionBlocks.

    Structure from flow.pt:
    - block.0: DiTConvBlock
    - block.1: List of DiTAttentionBlocks (typically 2)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int = 1024,
        num_attention_blocks: int = 2,
    ):
        super().__init__()

        # Convolutional block with time injection
        self.conv_block = DiTConvBlock(in_channels, out_channels, time_channels)

        # Attention blocks
        self.attention_blocks = [
            DiTAttentionBlock(out_channels) for _ in range(num_attention_blocks)
        ]

    def __call__(self, x: mx.array, time_emb: mx.array) -> mx.array:
        """
        Args:
            x: [batch, seq_len, in_channels]
            time_emb: [batch, time_channels]

        Returns:
            y: [batch, seq_len, out_channels]
        """
        x = self.conv_block(x, time_emb)
        for attn in self.attention_blocks:
            x = attn(x)
        return x


class DiTDecoder(nn.Module):
    """
    DiT-style decoder for flow matching.

    Architecture from flow.pt (71.3M params):
    - time_mlp: Sinusoidal embedding -> MLP
    - down_blocks[0]: Input projection (320 -> 256)
    - mid_blocks[0-11]: 12 transformer blocks
    - up_blocks[0]: Output expansion (256 -> 256, with 512 input from skip)
    - final_block: Conv + GroupNorm
    - final_proj: Conv1d to mel dim
    """

    def __init__(self, config: FlowMatchingConfig):
        super().__init__()
        self.config = config

        channels = config.decoder_channels  # 256
        time_channels = config.time_mlp_dim  # 1024
        input_channels = config.time_embed_dim  # 320

        # Time embedding
        self.time_mlp = TimeEmbedding(config)

        # Down block (1 block): 320 -> 256
        self.down_blocks = [DiTBlock(input_channels, channels, time_channels)]

        # Mid blocks (12 blocks): 256 -> 256
        self.mid_blocks = [
            DiTBlock(channels, channels, time_channels) for _ in range(12)
        ]

        # Up block (1 block): 512 -> 256 (with skip connection)
        self.up_blocks = [DiTBlock(channels * 2, channels, time_channels)]

        # Final block
        self.final_conv = nn.Conv1d(channels, channels, 3, padding=1)
        self.final_norm = nn.GroupNorm(32, channels, pytorch_compatible=True)

        # Output projection
        self.final_proj = nn.Conv1d(channels, config.mel_dim, 1)

    def __call__(
        self,
        x: mx.array,
        t: mx.array,
        cond: mx.array,
    ) -> mx.array:
        """
        Predict velocity for flow matching.

        Args:
            x: [batch, mel_len, mel_dim] - Noisy mel spectrogram
            t: [batch] - Diffusion timestep
            cond: [batch, mel_len, mel_dim] - Conditioning

        Returns:
            pred: [batch, mel_len, mel_dim] - Predicted velocity
        """
        # Time embedding
        time_emb = self.time_mlp(t)

        # Add conditioning and project to decoder channels
        # Input: [batch, mel_len, 80] -> need to project to 320
        x = x + cond
        # Pad channels: 80 -> 320
        batch, seq_len, _ = x.shape
        x = mx.concatenate([x, mx.zeros((batch, seq_len, 320 - 80))], axis=-1)

        # Down block
        skip = None
        for block in self.down_blocks:
            x = block(x, time_emb)
            skip = x

        # Mid blocks
        for block in self.mid_blocks:
            x = block(x, time_emb)

        # Up block (with skip connection)
        if skip is not None:
            x = mx.concatenate([x, skip], axis=-1)
        for block in self.up_blocks:
            x = block(x, time_emb)

        # Final block: Conv -> GroupNorm -> Mish (matches Matcha Block1D)
        x = self.final_conv(x)
        x = self.final_norm(x)
        x = nn.mish(x)

        # Output projection
        return self.final_proj(x)


    @staticmethod
    def from_pretrained(
        weights_path: str,
        config: FlowMatchingConfig | None = None,
    ) -> "DiTDecoder":
        """
        Load DiT decoder from pretrained flow.pt weights.

        Args:
            weights_path: Path to flow.pt file
            config: Optional config override

        Returns:
            Loaded DiTDecoder
        """
        import torch

        if config is None:
            config = FlowMatchingConfig()

        decoder = DiTDecoder(config)

        # Load PyTorch weights
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)

        # Load weights
        decoder._load_weights(state_dict)

        return decoder

    def _load_weights(self, state_dict: dict) -> None:
        """Load weights from PyTorch state dict."""

        def to_mlx(t):
            """Convert PyTorch tensor to MLX array."""
            return mx.array(t.numpy())

        def load_conv1d(conv, prefix):
            """Load Conv1d weights (PyTorch: [out, in, k] -> MLX: [out, k, in])."""
            w_key = f"{prefix}.weight"
            b_key = f"{prefix}.bias"
            if w_key in state_dict:
                w = state_dict[w_key].numpy()
                conv.weight = mx.array(w.transpose(0, 2, 1))
            if b_key in state_dict:
                conv.bias = to_mlx(state_dict[b_key])

        def load_linear(linear, prefix):
            """Load Linear weights (PyTorch: [out, in] -> MLX: [out, in])."""
            w_key = f"{prefix}.weight"
            b_key = f"{prefix}.bias"
            if w_key in state_dict:
                linear.weight = to_mlx(state_dict[w_key])
            if b_key in state_dict:
                linear.bias = to_mlx(state_dict[b_key])

        def load_groupnorm(norm, prefix):
            """Load GroupNorm weights."""
            w_key = f"{prefix}.weight"
            b_key = f"{prefix}.bias"
            if w_key in state_dict:
                norm.weight = to_mlx(state_dict[w_key])
            if b_key in state_dict:
                norm.bias = to_mlx(state_dict[b_key])

        def load_layernorm(norm, prefix):
            """Load LayerNorm weights."""
            w_key = f"{prefix}.weight"
            b_key = f"{prefix}.bias"
            if w_key in state_dict:
                norm.weight = to_mlx(state_dict[w_key])
            if b_key in state_dict:
                norm.bias = to_mlx(state_dict[b_key])

        def load_dit_conv_block(conv_block, prefix):
            """Load DiTConvBlock weights."""
            # block1: Conv1d + GroupNorm (indices 0, 2)
            load_conv1d(conv_block.block1_conv, f"{prefix}.block1.block.0")
            load_groupnorm(conv_block.block1_norm, f"{prefix}.block1.block.2")

            # block2: Conv1d + GroupNorm
            load_conv1d(conv_block.block2_conv, f"{prefix}.block2.block.0")
            load_groupnorm(conv_block.block2_norm, f"{prefix}.block2.block.2")

            # Time MLP
            load_linear(conv_block.time_mlp, f"{prefix}.mlp.1")

            # Residual conv
            if conv_block.res_conv is not None:
                load_conv1d(conv_block.res_conv, f"{prefix}.res_conv")

        def load_dit_attention_block(attn_block, prefix):
            """Load DiTAttentionBlock weights."""
            # Self-attention
            load_layernorm(attn_block.norm1, f"{prefix}.norm1")
            load_linear(attn_block.to_q, f"{prefix}.attn1.to_q")
            load_linear(attn_block.to_k, f"{prefix}.attn1.to_k")
            load_linear(attn_block.to_v, f"{prefix}.attn1.to_v")
            load_linear(attn_block.to_out, f"{prefix}.attn1.to_out.0")

            # FFN
            load_layernorm(attn_block.norm3, f"{prefix}.norm3")
            load_linear(attn_block.ff_proj, f"{prefix}.ff.net.0.proj")
            load_linear(attn_block.ff_out, f"{prefix}.ff.net.2")

        def load_dit_block(dit_block, prefix):
            """Load DiTBlock weights."""
            # ConvBlock
            load_dit_conv_block(dit_block.conv_block, f"{prefix}.0")

            # Attention blocks
            for i, attn in enumerate(dit_block.attention_blocks):
                load_dit_attention_block(attn, f"{prefix}.1.{i}")

        # Time MLP
        load_linear(self.time_mlp.linear_1, "decoder.estimator.time_mlp.linear_1")
        load_linear(self.time_mlp.linear_2, "decoder.estimator.time_mlp.linear_2")

        # Down blocks
        for i, block in enumerate(self.down_blocks):
            load_dit_block(block, f"decoder.estimator.down_blocks.{i}")

        # Mid blocks
        for i, block in enumerate(self.mid_blocks):
            load_dit_block(block, f"decoder.estimator.mid_blocks.{i}")

        # Up blocks
        for i, block in enumerate(self.up_blocks):
            load_dit_block(block, f"decoder.estimator.up_blocks.{i}")

        # Final block (Conv + GroupNorm)
        load_conv1d(self.final_conv, "decoder.estimator.final_block.block.0")
        load_groupnorm(self.final_norm, "decoder.estimator.final_block.block.2")

        # Final projection (Conv1d with kernel=1)
        load_conv1d(self.final_proj, "decoder.estimator.final_proj")


# ====================
# Legacy UNet-style blocks (kept for backwards compatibility)
# ====================


class ResBlock(nn.Module):
    """
    Residual block for UNet decoder (legacy).

    Conv -> Norm -> Activation -> Conv -> Norm + Time embedding injection
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
    ):
        super().__init__()
        self.skip: nn.Conv1d | None = None

        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(32, out_channels, pytorch_compatible=True)

        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels, pytorch_compatible=True)

        # Time embedding projection
        self.time_proj = nn.Linear(time_channels, out_channels)

        # Skip connection if channels differ
        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, 1)

    def __call__(self, x: mx.array, time_emb: mx.array) -> mx.array:
        """
        Args:
            x: [batch, seq_len, in_channels] - NLC format
            time_emb: [batch, time_channels]

        Returns:
            y: [batch, seq_len, out_channels]
        """
        residual = x

        # First conv block
        x = self.conv1(x)
        x = self.norm1(x)
        x = nn.silu(x)

        # Add time embedding
        time_proj = self.time_proj(time_emb)[:, None, :]  # [batch, 1, out_channels]
        x = x + time_proj

        # Second conv block
        x = self.conv2(x)
        x = self.norm2(x)

        # Skip connection
        if self.skip is not None:
            residual = self.skip(residual)

        return nn.silu(x + residual)  # type: ignore[no-any-return]


class DownBlock(nn.Module):
    """
    Downsampling block for UNet decoder.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
    ):
        super().__init__()

        self.res1 = ResBlock(in_channels, out_channels, time_channels)
        self.res2 = ResBlock(out_channels, out_channels, time_channels)

        # Downsample with strided conv
        self.downsample = nn.Conv1d(out_channels, out_channels, 3, stride=2, padding=1)

    def __call__(
        self,
        x: mx.array,
        time_emb: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """
        Args:
            x: [batch, seq_len, in_channels]
            time_emb: [batch, time_channels]

        Returns:
            out: [batch, seq_len//2, out_channels]
            skip: [batch, seq_len, out_channels] - for skip connection
        """
        x = self.res1(x, time_emb)
        x = self.res2(x, time_emb)
        skip = x
        x = self.downsample(x)
        return x, skip


class MidBlock(nn.Module):
    """
    Middle block for UNet decoder.
    """

    def __init__(
        self,
        channels: int,
        time_channels: int,
    ):
        super().__init__()

        self.res1 = ResBlock(channels, channels, time_channels)
        self.res2 = ResBlock(channels, channels, time_channels)

    def __call__(self, x: mx.array, time_emb: mx.array) -> mx.array:
        """
        Args:
            x: [batch, seq_len, channels]
            time_emb: [batch, time_channels]

        Returns:
            y: [batch, seq_len, channels]
        """
        x = self.res1(x, time_emb)
        return self.res2(x, time_emb)


class UpBlock(nn.Module):
    """
    Upsampling block for UNet decoder.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
    ):
        super().__init__()

        # Input includes skip connection
        self.res1 = ResBlock(in_channels + out_channels, out_channels, time_channels)
        self.res2 = ResBlock(out_channels, out_channels, time_channels)

        # Upsample with transposed conv or interpolation
        self.upsample = nn.ConvTranspose1d(
            in_channels, in_channels, 4, stride=2, padding=1,
        )

    def __call__(
        self,
        x: mx.array,
        skip: mx.array,
        time_emb: mx.array,
    ) -> mx.array:
        """
        Args:
            x: [batch, seq_len, in_channels]
            skip: [batch, seq_len*2, out_channels]
            time_emb: [batch, time_channels]

        Returns:
            y: [batch, seq_len*2, out_channels]
        """
        x = self.upsample(x)

        # Match skip length if needed
        if x.shape[1] != skip.shape[1]:
            x = x[:, : skip.shape[1], :]

        # Concatenate with skip connection
        x = mx.concatenate([x, skip], axis=-1)

        x = self.res1(x, time_emb)
        return self.res2(x, time_emb)


class FlowDecoder(nn.Module):
    """
    UNet-style decoder for flow matching.

    Architecture:
    - Time embedding MLP
    - Input projection (80 -> 256)
    - 4 down blocks
    - 2 mid blocks
    - 4 up blocks
    - Output projection (256 -> 80)
    """

    def __init__(self, config: FlowMatchingConfig):
        super().__init__()
        self.config = config

        # Time embedding
        self.time_mlp = TimeEmbedding(config)

        # Input projection
        self.input_projection = nn.Linear(config.mel_dim, config.decoder_channels)

        # Down blocks
        self.down_blocks = [
            DownBlock(
                config.decoder_channels, config.decoder_channels, config.time_mlp_dim,
            )
            for _ in range(config.num_down_blocks)
        ]

        # Mid blocks
        self.mid_blocks = [
            MidBlock(config.decoder_channels, config.time_mlp_dim)
            for _ in range(config.num_mid_blocks)
        ]

        # Up blocks
        self.up_blocks = [
            UpBlock(
                config.decoder_channels, config.decoder_channels, config.time_mlp_dim,
            )
            for _ in range(config.num_up_blocks)
        ]

        # Output projection
        self.output_projection = nn.Linear(config.decoder_channels, config.mel_dim)

    def __call__(
        self,
        x: mx.array,
        t: mx.array,
        cond: mx.array,
    ) -> mx.array:
        """
        Predict noise/velocity for flow matching.

        Args:
            x: [batch, mel_len, mel_dim] - Noisy mel spectrogram
            t: [batch] - Diffusion timestep
            cond: [batch, mel_len, mel_dim] - Conditioning (encoder output + speaker)

        Returns:
            pred: [batch, mel_len, mel_dim] - Predicted velocity
        """
        # Time embedding
        time_emb = self.time_mlp(t)

        # Add conditioning to input
        x = x + cond

        # Input projection
        x = self.input_projection(x)

        # Down blocks (store skips)
        skips = []
        for down_block in self.down_blocks:
            x, skip = down_block(x, time_emb)
            skips.append(skip)

        # Mid blocks
        for mid_block in self.mid_blocks:
            x = mid_block(x, time_emb)

        # Up blocks (use skips in reverse order)
        for up_block, skip in zip(self.up_blocks, reversed(skips), strict=False):
            x = up_block(x, skip, time_emb)

        # Output projection
        return self.output_projection(x)



class MaskedDiffWithXvec(nn.Module):
    """
    Flow matching model with speaker conditioning (X-vector).

    This is the main flow model class that combines:
    - Token embedding
    - Flow encoder (pre-lookahead + transformer)
    - Encoder projection to mel dim
    - Speaker embedding projection
    - Flow decoder (UNet-style)

    The model uses conditional flow matching to generate mel spectrograms
    from speech tokens, conditioned on speaker embeddings.
    """

    def __init__(self, config: FlowMatchingConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.input_embedding = nn.Embedding(config.vocab_size, config.embed_dim)

        # Flow encoder
        self.encoder = FlowEncoder(config)

        # Encoder output projection to mel dimension
        self.encoder_proj = nn.Linear(config.embed_dim, config.mel_dim)

        # Speaker embedding projection
        self.spk_embed_affine_layer = nn.Linear(
            config.speaker_embed_dim, config.mel_dim,
        )

        # Flow decoder
        self.decoder = FlowDecoder(config)

    def __call__(
        self,
        tokens: mx.array,
        speaker_embed: mx.array,
        t: mx.array,
        x_noisy: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        """
        Forward pass for training.

        Args:
            tokens: [batch, seq_len] - Speech tokens
            speaker_embed: [batch, speaker_embed_dim] - Speaker embedding (x-vector)
            t: [batch] - Diffusion timestep (0 to 1)
            x_noisy: [batch, mel_len, mel_dim] - Noisy mel spectrogram
            mask: Optional sequence mask

        Returns:
            velocity: [batch, mel_len, mel_dim] - Predicted velocity
        """
        mel_len = x_noisy.shape[1]

        # Embed tokens
        x = self.input_embedding(tokens)

        # Encode
        x = self.encoder(x, mask)

        # Project to mel dimension
        x = self.encoder_proj(x)

        # Add speaker embedding (broadcast across sequence)
        spk = self.spk_embed_affine_layer(speaker_embed)  # [batch, mel_dim]
        cond = x + spk[:, None, :]  # [batch, seq_len, mel_dim]

        # Interpolate conditioning to match mel length
        cond = self._interpolate_to_length(cond, mel_len)

        # Decode (predict velocity)
        return self.decoder(x_noisy, t, cond)


    def _interpolate_to_length(self, x: mx.array, target_length: int) -> mx.array:
        """
        Interpolate tensor to target length along sequence dimension.

        Args:
            x: [batch, seq_len, dim]
            target_length: Target sequence length

        Returns:
            y: [batch, target_length, dim]
        """
        if x.shape[1] == target_length:
            return x

        # Linear interpolation
        indices = mx.linspace(0, x.shape[1] - 1, target_length)
        idx_floor = mx.floor(indices).astype(mx.int32)
        idx_ceil = mx.minimum(idx_floor + 1, x.shape[1] - 1)
        alpha = (indices - idx_floor.astype(mx.float32))[:, None]
        return x[:, idx_floor, :] * (1 - alpha) + x[:, idx_ceil, :] * alpha

    def generate(
        self,
        tokens: mx.array,
        speaker_embed: mx.array,
        mel_length: int,
        num_steps: int = 10,
        mask: mx.array | None = None,
    ) -> mx.array:
        """
        Generate mel spectrogram from tokens using ODE solver.

        Uses Euler method to solve the flow ODE.

        Args:
            tokens: [batch, seq_len] - Speech tokens
            speaker_embed: [batch, speaker_embed_dim] - Speaker embedding
            mel_length: Target mel spectrogram length
            num_steps: Number of ODE solver steps
            mask: Optional sequence mask

        Returns:
            mel: [batch, mel_length, mel_dim] - Generated mel spectrogram
        """
        batch = tokens.shape[0]

        # Embed and encode tokens
        x = self.input_embedding(tokens)
        x = self.encoder(x, mask)
        x = self.encoder_proj(x)

        # Add speaker embedding
        spk = self.spk_embed_affine_layer(speaker_embed)
        cond = x + spk[:, None, :]

        # Interpolate condition to mel_length if needed
        cond = self._interpolate_to_length(cond, mel_length)

        # Start from noise
        x_t = mx.random.normal((batch, mel_length, self.config.mel_dim))

        # Euler integration from t=1 to t=0
        dt = 1.0 / num_steps
        for step in range(num_steps):
            t = mx.array([1.0 - step * dt] * batch)

            # Predict velocity
            v = self.decoder(x_t, t, cond)

            # Euler step
            x_t = x_t - dt * v

        return x_t

    @staticmethod
    def from_pretrained(
        weights_path: str,
        config: FlowMatchingConfig | None = None,
    ) -> "MaskedDiffWithXvec":
        """
        Load flow model from pretrained weights.

        Args:
            weights_path: Path to flow.pt file
            config: Optional config override

        Returns:
            Loaded MaskedDiffWithXvec model
        """
        import torch

        if config is None:
            config = FlowMatchingConfig()

        model = MaskedDiffWithXvec(config)

        # Load PyTorch weights
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)

        # Map weights
        model._load_weights(state_dict)

        return model

    def _load_weights(self, state_dict: dict) -> None:
        """Load weights from PyTorch state dict."""

        # Token embedding
        if "input_embedding.weight" in state_dict:
            self.input_embedding.weight = mx.array(
                state_dict["input_embedding.weight"].numpy(),
            )

        # Encoder projection
        if "encoder_proj.weight" in state_dict:
            self.encoder_proj.weight = mx.array(
                state_dict["encoder_proj.weight"].numpy(),
            )
            self.encoder_proj.bias = mx.array(state_dict["encoder_proj.bias"].numpy())

        # Speaker embedding projection
        if "spk_embed_affine_layer.weight" in state_dict:
            self.spk_embed_affine_layer.weight = mx.array(
                state_dict["spk_embed_affine_layer.weight"].numpy(),
            )
            self.spk_embed_affine_layer.bias = mx.array(
                state_dict["spk_embed_affine_layer.bias"].numpy(),
            )

        # Pre-lookahead layers
        for _i, conv_name in enumerate(["conv1", "conv2"]):
            key_w = f"encoder.pre_lookahead_layer.{conv_name}.weight"
            key_b = f"encoder.pre_lookahead_layer.{conv_name}.bias"
            if key_w in state_dict:
                conv = getattr(self.encoder.pre_lookahead_layer, conv_name)
                # PyTorch: [out, in, kernel], MLX Conv1d: [out, kernel, in]
                w = state_dict[key_w].numpy()
                conv.weight = mx.array(w.transpose(0, 2, 1))
                conv.bias = mx.array(state_dict[key_b].numpy())

        # Encoder layers
        for i, layer in enumerate(self.encoder.encoder_layer):
            prefix = f"encoder.encoder_layer.{i}"

            # Self attention
            for proj in ["q_proj", "k_proj", "v_proj"]:
                key_w = f"{prefix}.self_attn.{proj}.weight"
                if key_w in state_dict:
                    proj_layer = getattr(layer.self_attn, proj)
                    proj_layer.weight = mx.array(state_dict[key_w].numpy().T)

            # Output projection
            if f"{prefix}.self_attn.out_proj.weight" in state_dict:
                layer.self_attn.out_proj.weight = mx.array(
                    state_dict[f"{prefix}.self_attn.out_proj.weight"].numpy().T,
                )

            # Norms
            if f"{prefix}.norm1.weight" in state_dict:
                layer.norm1.weight = mx.array(
                    state_dict[f"{prefix}.norm1.weight"].numpy(),
                )
            if f"{prefix}.norm2.weight" in state_dict:
                layer.norm2.weight = mx.array(
                    state_dict[f"{prefix}.norm2.weight"].numpy(),
                )

            # FFN
            if f"{prefix}.linear1.weight" in state_dict:
                layer.linear1.weight = mx.array(
                    state_dict[f"{prefix}.linear1.weight"].numpy().T,
                )
                layer.linear1.bias = mx.array(
                    state_dict[f"{prefix}.linear1.bias"].numpy(),
                )
            if f"{prefix}.linear2.weight" in state_dict:
                layer.linear2.weight = mx.array(
                    state_dict[f"{prefix}.linear2.weight"].numpy().T,
                )
                layer.linear2.bias = mx.array(
                    state_dict[f"{prefix}.linear2.bias"].numpy(),
                )

        # Decoder (time MLP, projections, blocks)
        self._load_decoder_weights(state_dict)

    def _load_decoder_weights(self, state_dict: dict) -> None:
        """Load decoder weights from PyTorch state dict."""

        # Time MLP - both PyTorch and MLX Linear use [out_features, in_features]
        # No transpose needed for Linear layers
        prefix = "decoder.estimator.time_mlp"
        if f"{prefix}.linear_1.weight" in state_dict:
            self.decoder.time_mlp.linear_1.weight = mx.array(
                state_dict[f"{prefix}.linear_1.weight"].numpy(),
            )
            self.decoder.time_mlp.linear_1.bias = mx.array(
                state_dict[f"{prefix}.linear_1.bias"].numpy(),
            )
        if f"{prefix}.linear_2.weight" in state_dict:
            self.decoder.time_mlp.linear_2.weight = mx.array(
                state_dict[f"{prefix}.linear_2.weight"].numpy(),
            )
            self.decoder.time_mlp.linear_2.bias = mx.array(
                state_dict[f"{prefix}.linear_2.bias"].numpy(),
            )

        # Input/output projections - also Linear layers, no transpose
        if "decoder.estimator.input_projection.weight" in state_dict:
            self.decoder.input_projection.weight = mx.array(
                state_dict["decoder.estimator.input_projection.weight"].numpy(),
            )
            self.decoder.input_projection.bias = mx.array(
                state_dict["decoder.estimator.input_projection.bias"].numpy(),
            )

        if "decoder.estimator.output_projection.weight" in state_dict:
            self.decoder.output_projection.weight = mx.array(
                state_dict["decoder.estimator.output_projection.weight"].numpy(),
            )
            self.decoder.output_projection.bias = mx.array(
                state_dict["decoder.estimator.output_projection.bias"].numpy(),
            )


@dataclass
class CausalFlowConfig(FlowMatchingConfig):
    """Configuration for causal/streaming flow matching model."""

    # Streaming parameters
    pre_lookahead_len: int = 3  # Number of lookahead tokens for context
    token_mel_ratio: float = 4.0  # Ratio of mel frames per token
    streaming_chunk_size: int = 16  # Chunk size for streaming inference


class CausalFlowEncoder(nn.Module):
    """
    Causal/streaming encoder for flow matching model.

    Extends FlowEncoder with:
    - Causal attention masking
    - Context preservation across chunks
    - Cache management for streaming inference
    """

    def __init__(self, config: CausalFlowConfig):
        super().__init__()
        self.config = config

        # Pre-lookahead convolutions (same as non-causal)
        self.pre_lookahead_layer = PreLookaheadLayer(config)

        # Transformer encoder layers
        self.encoder_layer = [
            TransformerEncoderLayer(config) for _ in range(config.num_encoder_layers)
        ]

        # Cache for streaming
        self._cache: list[mx.array] | None = None
        self._context: mx.array | None = None

    def reset_cache(self) -> None:
        """Reset streaming cache."""
        self._cache = None
        self._context = None

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        streaming: bool = False,
        context: mx.array | None = None,
    ) -> mx.array:
        """
        Forward pass with optional streaming support.

        Args:
            x: [batch, seq_len, embed_dim]
            mask: Optional attention mask
            streaming: Whether to use streaming mode
            context: Previous context for streaming

        Returns:
            y: [batch, seq_len', embed_dim]
        """
        if streaming and context is not None:
            # Concatenate previous context with current input
            x = mx.concatenate([context, x], axis=1)
            self._context = context

        # Pre-lookahead convolution
        x = self.pre_lookahead_layer(x)

        # Create causal mask if streaming
        if streaming and mask is None:
            seq_len = x.shape[1]
            # Lower triangular mask for causal attention
            causal_mask = mx.triu(mx.full((seq_len, seq_len), float("-inf")), k=1)
            mask = causal_mask[None, None, :, :]  # [1, 1, seq, seq]

        # Transformer layers
        for layer in self.encoder_layer:
            x = layer(x, mask)

        return x


class CausalMaskedDiffWithXvec(nn.Module):
    """
    Causal/streaming flow matching model with speaker conditioning (X-vector).

    This is the streaming variant of MaskedDiffWithXvec that supports:
    - Chunk-by-chunk processing
    - Context preservation across chunks
    - Cache management for real-time inference

    The model enables streaming TTS with ~150ms latency by processing
    input tokens incrementally while maintaining context.
    """

    def __init__(self, config: CausalFlowConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.input_embedding = nn.Embedding(config.vocab_size, config.embed_dim)

        # Causal flow encoder
        self.encoder = CausalFlowEncoder(config)

        # Encoder output projection to mel dimension
        self.encoder_proj = nn.Linear(config.embed_dim, config.mel_dim)

        # Speaker embedding projection
        self.spk_embed_affine_layer = nn.Linear(
            config.speaker_embed_dim, config.mel_dim,
        )

        # Flow decoder (DiT-style, same as non-causal)
        self.decoder = DiTDecoder(config)

        # Streaming state
        self._mel_cache: mx.array | None = None
        self._noise_cache: mx.array | None = None

    def reset_cache(self) -> None:
        """Reset all streaming caches."""
        self.encoder.reset_cache()
        self._mel_cache = None
        self._noise_cache = None

    def __call__(
        self,
        tokens: mx.array,
        speaker_embed: mx.array,
        t: mx.array,
        x_noisy: mx.array,
        mask: mx.array | None = None,
        streaming: bool = False,
    ) -> mx.array:
        """
        Forward pass for training.

        In training, randomly selects between streaming and non-streaming mode
        to improve model robustness.

        Args:
            tokens: [batch, seq_len] - Speech tokens
            speaker_embed: [batch, speaker_embed_dim] - Speaker embedding (x-vector)
            t: [batch] - Diffusion timestep (0 to 1)
            x_noisy: [batch, mel_len, mel_dim] - Noisy mel spectrogram
            mask: Optional sequence mask
            streaming: Force streaming mode (if False, randomly selects in training)

        Returns:
            velocity: [batch, mel_len, mel_dim] - Predicted velocity
        """
        mel_len = x_noisy.shape[1]

        # Embed tokens
        x = self.input_embedding(tokens)

        # Encode (with optional streaming)
        x = self.encoder(x, mask, streaming=streaming)

        # Project to mel dimension
        x = self.encoder_proj(x)

        # Add speaker embedding (broadcast across sequence)
        spk = self.spk_embed_affine_layer(speaker_embed)  # [batch, mel_dim]
        cond = x + spk[:, None, :]  # [batch, seq_len, mel_dim]

        # Interpolate conditioning to match mel length
        cond = self._interpolate_to_length(cond, mel_len)

        # Decode (predict velocity)
        return self.decoder(x_noisy, t, cond)


    def _interpolate_to_length(self, x: mx.array, target_length: int) -> mx.array:
        """Interpolate tensor to target length along sequence dimension."""
        if x.shape[1] == target_length:
            return x

        # Linear interpolation
        indices = mx.linspace(0, x.shape[1] - 1, target_length)
        idx_floor = mx.floor(indices).astype(mx.int32)
        idx_ceil = mx.minimum(idx_floor + 1, x.shape[1] - 1)
        alpha = (indices - idx_floor.astype(mx.float32))[:, None]
        return x[:, idx_floor, :] * (1 - alpha) + x[:, idx_ceil, :] * alpha

    def generate_streaming(
        self,
        tokens: mx.array,
        speaker_embed: mx.array,
        num_steps: int = 10,
        finalize: bool = False,
    ) -> mx.array:
        """
        Generate mel spectrogram in streaming mode.

        Processes tokens chunk by chunk, maintaining context across calls.
        Call with finalize=True for the last chunk.

        Args:
            tokens: [batch, chunk_seq_len] - Current chunk of speech tokens
            speaker_embed: [batch, speaker_embed_dim] - Speaker embedding
            num_steps: Number of ODE solver steps
            finalize: Whether this is the final chunk

        Returns:
            mel: [batch, chunk_mel_len, mel_dim] - Generated mel spectrogram chunk
        """
        batch = tokens.shape[0]
        token_len = tokens.shape[1]

        # Calculate expected mel length for this chunk
        mel_length = int(token_len * self.config.token_mel_ratio)

        # Handle context
        if not finalize and token_len > self.config.pre_lookahead_len:
            # Split into current tokens and lookahead context
            current_tokens = tokens[:, : -self.config.pre_lookahead_len]
            context_tokens = tokens[:, -self.config.pre_lookahead_len :]

            # Embed and encode with context
            x = self.input_embedding(current_tokens)
            context = self.input_embedding(context_tokens)
            x = self.encoder(x, streaming=True, context=context)
            mel_length = int(current_tokens.shape[1] * self.config.token_mel_ratio)
        else:
            # Process all tokens (final chunk or short sequence)
            x = self.input_embedding(tokens)
            x = self.encoder(x, streaming=finalize)

        # Project to mel dimension
        x = self.encoder_proj(x)

        # Add speaker embedding
        spk = self.spk_embed_affine_layer(speaker_embed)
        cond = x + spk[:, None, :]

        # Interpolate condition to mel_length
        cond = self._interpolate_to_length(cond, mel_length)

        # Initialize or extend noise cache
        if self._noise_cache is None:
            x_t = mx.random.normal((batch, mel_length, self.config.mel_dim))
            self._noise_cache = x_t
        else:
            # Generate new noise for this chunk
            new_noise = mx.random.normal((batch, mel_length, self.config.mel_dim))
            x_t = new_noise

        # Euler integration from t=1 to t=0
        dt = 1.0 / num_steps
        for step in range(num_steps):
            t = mx.array([1.0 - step * dt] * batch)

            # Predict velocity
            v = self.decoder(x_t, t, cond)

            # Euler step
            x_t = x_t - dt * v

        # Cache mel output if not finalizing
        if not finalize:
            if self._mel_cache is None:
                self._mel_cache = x_t
            else:
                self._mel_cache = mx.concatenate([self._mel_cache, x_t], axis=1)

        return x_t

    def generate(
        self,
        tokens: mx.array,
        speaker_embed: mx.array,
        mel_length: int,
        num_steps: int = 10,
        mask: mx.array | None = None,
        streaming: bool = False,
    ) -> mx.array:
        """
        Generate mel spectrogram from tokens using ODE solver.

        Args:
            tokens: [batch, seq_len] - Speech tokens
            speaker_embed: [batch, speaker_embed_dim] - Speaker embedding
            mel_length: Target mel spectrogram length
            num_steps: Number of ODE solver steps
            mask: Optional sequence mask
            streaming: Whether to use streaming encoder mode

        Returns:
            mel: [batch, mel_length, mel_dim] - Generated mel spectrogram
        """
        batch = tokens.shape[0]

        # Embed and encode tokens
        x = self.input_embedding(tokens)
        x = self.encoder(x, mask, streaming=streaming)
        x = self.encoder_proj(x)

        # Add speaker embedding
        spk = self.spk_embed_affine_layer(speaker_embed)
        cond = x + spk[:, None, :]

        # Interpolate condition to mel_length if needed
        cond = self._interpolate_to_length(cond, mel_length)

        # Start from noise
        x_t = mx.random.normal((batch, mel_length, self.config.mel_dim))

        # Euler integration from t=1 to t=0
        dt = 1.0 / num_steps
        for step in range(num_steps):
            t = mx.array([1.0 - step * dt] * batch)

            # Predict velocity
            v = self.decoder(x_t, t, cond)

            # Euler step
            x_t = x_t - dt * v

        return x_t

    @staticmethod
    def from_pretrained(
        weights_path: str,
        config: CausalFlowConfig | None = None,
    ) -> "CausalMaskedDiffWithXvec":
        """
        Load causal flow model from pretrained weights.

        The causal model uses the same weights as the non-causal model,
        but with streaming-compatible architecture.

        Args:
            weights_path: Path to flow.pt file
            config: Optional config override

        Returns:
            Loaded CausalMaskedDiffWithXvec model
        """
        import torch

        if config is None:
            config = CausalFlowConfig()

        model = CausalMaskedDiffWithXvec(config)

        # Load PyTorch weights
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)

        # Map weights (same as non-causal model)
        model._load_weights(state_dict)

        return model

    def _load_weights(self, state_dict: dict) -> None:
        """Load weights from PyTorch state dict."""

        def to_mlx(t):
            """Convert PyTorch tensor to MLX array."""
            return mx.array(t.numpy())

        # Token embedding
        if "input_embedding.weight" in state_dict:
            self.input_embedding.weight = to_mlx(state_dict["input_embedding.weight"])

        # Encoder projection
        if "encoder_proj.weight" in state_dict:
            self.encoder_proj.weight = to_mlx(state_dict["encoder_proj.weight"])
            self.encoder_proj.bias = to_mlx(state_dict["encoder_proj.bias"])

        # Speaker embedding projection
        if "spk_embed_affine_layer.weight" in state_dict:
            self.spk_embed_affine_layer.weight = to_mlx(
                state_dict["spk_embed_affine_layer.weight"],
            )
            self.spk_embed_affine_layer.bias = to_mlx(
                state_dict["spk_embed_affine_layer.bias"],
            )

        # Pre-lookahead layers (encoder)
        for conv_name in ["conv1", "conv2"]:
            key_w = f"encoder.pre_lookahead_layer.{conv_name}.weight"
            key_b = f"encoder.pre_lookahead_layer.{conv_name}.bias"
            if key_w in state_dict:
                conv = getattr(self.encoder.pre_lookahead_layer, conv_name)
                w = state_dict[key_w].numpy()
                conv.weight = mx.array(w.transpose(0, 2, 1))
                conv.bias = to_mlx(state_dict[key_b])

        # Encoder layers
        for i, layer in enumerate(self.encoder.encoder_layer):
            prefix = f"encoder.encoder_layer.{i}"

            # Self attention
            for proj in ["q_proj", "k_proj", "v_proj"]:
                key_w = f"{prefix}.self_attn.{proj}.weight"
                if key_w in state_dict:
                    proj_layer = getattr(layer.self_attn, proj)
                    proj_layer.weight = to_mlx(state_dict[key_w].T)

            # Output projection
            key_out = f"{prefix}.self_attn.out_proj.weight"
            if key_out in state_dict:
                layer.self_attn.out_proj.weight = to_mlx(state_dict[key_out].T)

            # Norms
            if f"{prefix}.norm1.weight" in state_dict:
                layer.norm1.weight = to_mlx(state_dict[f"{prefix}.norm1.weight"])
            if f"{prefix}.norm2.weight" in state_dict:
                layer.norm2.weight = to_mlx(state_dict[f"{prefix}.norm2.weight"])

            # FFN
            if f"{prefix}.linear1.weight" in state_dict:
                layer.linear1.weight = to_mlx(state_dict[f"{prefix}.linear1.weight"].T)
                layer.linear1.bias = to_mlx(state_dict[f"{prefix}.linear1.bias"])
            if f"{prefix}.linear2.weight" in state_dict:
                layer.linear2.weight = to_mlx(state_dict[f"{prefix}.linear2.weight"].T)
                layer.linear2.bias = to_mlx(state_dict[f"{prefix}.linear2.bias"])

        # Load decoder weights using DiTDecoder's method
        self.decoder._load_weights(state_dict)
