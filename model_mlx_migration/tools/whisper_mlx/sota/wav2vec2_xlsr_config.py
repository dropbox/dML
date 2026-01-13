"""Wav2Vec2-XLSR Configuration for MLX.

This configuration matches the HuggingFace wav2vec2-large-xlsr-53 checkpoint.
Reference: https://huggingface.co/facebook/wav2vec2-large-xlsr-53
"""

from dataclasses import dataclass, field


@dataclass
class Wav2Vec2XLSRConfig:
    """Configuration for Wav2Vec2-XLSR model.

    Architecture:
        - Feature extractor: 7 Conv1d layers with layer norm (same as emotion2vec)
        - Feature projection: LayerNorm + Linear (512 -> 1024)
        - Positional encoding: Grouped Conv1d with weight normalization
        - 24 transformer encoder layers (pre-norm / stable layer norm)

    Attributes:
        hidden_size: Transformer hidden dimension (1024)
        num_hidden_layers: Number of transformer layers (24)
        num_attention_heads: Number of attention heads (16)
        intermediate_size: MLP intermediate dimension (4096)
        hidden_act: Activation function (gelu)
        layer_norm_eps: LayerNorm epsilon (1e-5)
        conv_dim: Feature extractor conv dimensions
        conv_kernel: Feature extractor kernel sizes
        conv_stride: Feature extractor strides
        num_conv_pos_embeddings: Positional conv kernel size (128)
        num_conv_pos_embedding_groups: Positional conv groups (16)
        do_stable_layer_norm: Use pre-norm (True for XLSR)
    """

    # Transformer architecture
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    hidden_act: str = "gelu"

    # Layer normalization
    layer_norm_eps: float = 1e-5
    do_stable_layer_norm: bool = True

    # Dropout (set to 0 for inference)
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    activation_dropout: float = 0.0
    feat_proj_dropout: float = 0.0
    final_dropout: float = 0.0

    # Feature extractor
    conv_dim: list[int] = field(default_factory=lambda: [512, 512, 512, 512, 512, 512, 512])
    conv_kernel: list[int] = field(default_factory=lambda: [10, 3, 3, 3, 3, 2, 2])
    conv_stride: list[int] = field(default_factory=lambda: [5, 2, 2, 2, 2, 2, 2])
    conv_bias: bool = True
    feat_extract_norm: str = "layer"
    feat_extract_activation: str = "gelu"

    # Positional encoding
    num_conv_pos_embeddings: int = 128
    num_conv_pos_embedding_groups: int = 16

    @property
    def head_dim(self) -> int:
        """Attention head dimension."""
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def xlsr_53(cls) -> "Wav2Vec2XLSRConfig":
        """Return configuration for wav2vec2-large-xlsr-53 checkpoint."""
        return cls()

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Wav2Vec2XLSRConfig":
        """Create config from dictionary."""
        return cls(
            hidden_size=config_dict.get("hidden_size", 1024),
            num_hidden_layers=config_dict.get("num_hidden_layers", 24),
            num_attention_heads=config_dict.get("num_attention_heads", 16),
            intermediate_size=config_dict.get("intermediate_size", 4096),
            hidden_act=config_dict.get("hidden_act", "gelu"),
            layer_norm_eps=config_dict.get("layer_norm_eps", 1e-5),
            do_stable_layer_norm=config_dict.get("do_stable_layer_norm", True),
            conv_dim=config_dict.get("conv_dim", [512, 512, 512, 512, 512, 512, 512]),
            conv_kernel=config_dict.get("conv_kernel", [10, 3, 3, 3, 3, 2, 2]),
            conv_stride=config_dict.get("conv_stride", [5, 2, 2, 2, 2, 2, 2]),
            conv_bias=config_dict.get("conv_bias", True),
            num_conv_pos_embeddings=config_dict.get("num_conv_pos_embeddings", 128),
            num_conv_pos_embedding_groups=config_dict.get("num_conv_pos_embedding_groups", 16),
        )
