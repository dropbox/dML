"""BEATs Configuration for MLX.

Reference: https://github.com/microsoft/unilm/tree/master/beats
"""

from dataclasses import dataclass


@dataclass
class BEATsConfig:
    """Configuration for BEATs model.

    Architecture:
        - Patch embedding: 16x16 patches from spectrogram
        - Positional conv: Grouped Conv1d like wav2vec2
        - 12 transformer encoder layers with relative position embedding
        - GRU-based relative position interpolation

    Attributes:
        encoder_layers: Number of transformer layers (12)
        encoder_embed_dim: Transformer hidden dimension (768)
        encoder_ffn_embed_dim: FFN intermediate dimension (3072)
        encoder_attention_heads: Number of attention heads (12)
        activation_fn: Activation function (gelu)
        layer_norm_first: Pre-norm vs post-norm (False for BEATs)
        conv_pos: Positional conv kernel size (128)
        conv_pos_groups: Positional conv groups (16)
        relative_position_embedding: Use relative position (True)
        num_buckets: Number of relative position buckets (320)
        max_distance: Max distance for relative position (800)
        gru_rel_pos: Use GRU for relative position (True)
        deep_norm: Use deep norm scaling (True)
        input_patch_size: Patch size for spectrogram (16)
        embed_dim: Patch embedding dimension (512)
    """

    encoder_layers: int = 12
    encoder_embed_dim: int = 768
    encoder_ffn_embed_dim: int = 3072
    encoder_attention_heads: int = 12
    activation_fn: str = "gelu"
    dropout: float = 0.0  # Set to 0 for inference
    attention_dropout: float = 0.0
    activation_dropout: float = 0.0
    encoder_layerdrop: float = 0.0
    layer_norm_first: bool = False
    conv_bias: bool = False
    conv_pos: int = 128
    conv_pos_groups: int = 16
    relative_position_embedding: bool = True
    num_buckets: int = 320
    max_distance: int = 800
    gru_rel_pos: bool = True
    deep_norm: bool = True
    input_patch_size: int = 16
    embed_dim: int = 512
    layer_norm_eps: float = 1e-5

    @property
    def head_dim(self) -> int:
        """Attention head dimension."""
        return self.encoder_embed_dim // self.encoder_attention_heads

    @classmethod
    def from_dict(cls, config_dict: dict) -> "BEATsConfig":
        """Create config from dictionary."""
        return cls(
            encoder_layers=config_dict.get("encoder_layers", 12),
            encoder_embed_dim=config_dict.get("encoder_embed_dim", 768),
            encoder_ffn_embed_dim=config_dict.get("encoder_ffn_embed_dim", 3072),
            encoder_attention_heads=config_dict.get("encoder_attention_heads", 12),
            activation_fn=config_dict.get("activation_fn", "gelu"),
            layer_norm_first=config_dict.get("layer_norm_first", False),
            conv_pos=config_dict.get("conv_pos", 128),
            conv_pos_groups=config_dict.get("conv_pos_groups", 16),
            relative_position_embedding=config_dict.get("relative_position_embedding", True),
            num_buckets=config_dict.get("num_buckets", 320),
            max_distance=config_dict.get("max_distance", 800),
            gru_rel_pos=config_dict.get("gru_rel_pos", True),
            deep_norm=config_dict.get("deep_norm", True),
            input_patch_size=config_dict.get("input_patch_size", 16),
            embed_dim=config_dict.get("embed_dim", 512),
        )
