"""Emotion2vec Configuration for MLX.

This configuration matches the FunASR emotion2vec_base checkpoint.
Reference: https://modelscope.cn/models/iic/emotion2vec_base
Paper: https://arxiv.org/abs/2312.15185
"""

from dataclasses import dataclass, field


@dataclass
class Emotion2vecConfig:
    """Configuration for Emotion2vec model.

    Architecture:
        - Local encoder: 7 Conv1d layers for raw audio feature extraction
        - Project features: LayerNorm + Linear (512 -> 768)
        - Relative positional encoder: 5 Conv1d layers for position encoding
        - Context encoder: 4 transformer blocks (prenet)
        - Main encoder: 8 transformer blocks
        - ALiBi attention bias with learnable scaling
        - Extra tokens: 10 learnable tokens prepended to sequence

    This is a Data2Vec-style model adapted for emotion recognition.

    Attributes:
        embed_dim: Transformer hidden dimension (768)
        depth: Number of main transformer blocks (8)
        prenet_depth: Number of prenet transformer blocks (4)
        num_heads: Number of attention heads (12)
        mlp_ratio: MLP expansion ratio (4.0)
        num_extra_tokens: Number of extra tokens (10)
        layer_norm_eps: LayerNorm epsilon (1e-5)
        local_encoder_spec: Conv1d layer specifications for local encoder
        conv_pos_width: Width of positional conv kernels (95)
        conv_pos_depth: Number of positional conv layers (5)
        conv_pos_groups: Groups for positional conv (16)
    """

    # Transformer architecture
    embed_dim: int = 768
    depth: int = 8
    prenet_depth: int = 4
    num_heads: int = 12
    mlp_ratio: float = 4.0

    # Dropout (set to 0 for inference)
    encoder_dropout: float = 0.0
    attention_dropout: float = 0.0
    activation_dropout: float = 0.0
    post_mlp_drop: float = 0.0

    # Extra tokens
    num_extra_tokens: int = 10
    init_extra_token_zero: bool = True

    # Layer normalization
    layer_norm_eps: float = 1e-5
    norm_affine: bool = True
    layer_norm_first: bool = False

    # Local encoder (feature extraction)
    # Format: [(out_ch, kernel_size, stride), ...]
    local_encoder_spec: list[tuple[int, int, int]] = field(default_factory=lambda: [
        (512, 10, 5),  # Layer 0
        (512, 3, 2),   # Layer 1
        (512, 3, 2),   # Layer 2
        (512, 3, 2),   # Layer 3
        (512, 3, 2),   # Layer 4
        (512, 2, 2),   # Layer 5
        (512, 2, 2),   # Layer 6
    ])

    # Relative positional encoding
    conv_pos_width: int = 95
    conv_pos_depth: int = 5
    conv_pos_groups: int = 16

    # ALiBi attention
    use_alibi_encoder: bool = True
    alibi_scale: float = 1.0
    learned_alibi_scale: bool = True
    learned_alibi_scale_per_head: bool = True
    num_alibi_heads: int = 12

    # Decoder (for Data2Vec reconstruction, optional for inference)
    decoder_dim: int = 384
    decoder_groups: int = 16
    decoder_kernel: int = 7
    decoder_layers: int = 4

    @property
    def intermediate_size(self) -> int:
        """MLP intermediate dimension."""
        return int(self.embed_dim * self.mlp_ratio)

    @property
    def head_dim(self) -> int:
        """Attention head dimension."""
        return self.embed_dim // self.num_heads

    @classmethod
    def base(cls) -> "Emotion2vecConfig":
        """Return configuration for emotion2vec_base checkpoint."""
        return cls()

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Emotion2vecConfig":
        """Load config from YAML file."""
        import yaml
        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)

        model_conf = config_dict.get("model_conf", {})
        audio_conf = model_conf.get("modalities", {}).get("audio", {})

        # Parse feature_encoder_spec
        spec_str = audio_conf.get("feature_encoder_spec", "")
        local_encoder_spec = cls._parse_encoder_spec(spec_str)

        return cls(
            embed_dim=model_conf.get("embed_dim", 768),
            depth=model_conf.get("depth", 8),
            prenet_depth=audio_conf.get("prenet_depth", 4),
            num_heads=model_conf.get("num_heads", 12),
            mlp_ratio=model_conf.get("mlp_ratio", 4.0),
            num_extra_tokens=audio_conf.get("num_extra_tokens", 10),
            layer_norm_eps=model_conf.get("norm_eps", 1e-5),
            layer_norm_first=model_conf.get("layer_norm_first", False),
            local_encoder_spec=local_encoder_spec,
            conv_pos_width=audio_conf.get("conv_pos_width", 95),
            conv_pos_depth=audio_conf.get("conv_pos_depth", 5),
            conv_pos_groups=audio_conf.get("conv_pos_groups", 16),
            use_alibi_encoder=audio_conf.get("use_alibi_encoder", True),
            learned_alibi_scale=audio_conf.get("learned_alibi_scale", True),
            learned_alibi_scale_per_head=audio_conf.get("learned_alibi_scale_per_head", True),
            num_alibi_heads=audio_conf.get("num_alibi_heads", 12),
        )

    @staticmethod
    def _parse_encoder_spec(spec_str: str) -> list[tuple[int, int, int]]:
        """Parse feature encoder specification string.

        Format: '[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]'
        """
        if not spec_str:
            return [
                (512, 10, 5),
                (512, 3, 2),
                (512, 3, 2),
                (512, 3, 2),
                (512, 3, 2),
                (512, 2, 2),
                (512, 2, 2),
            ]
        # For now, return default spec
        # Full parsing would require eval or manual parsing
        return [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ]
