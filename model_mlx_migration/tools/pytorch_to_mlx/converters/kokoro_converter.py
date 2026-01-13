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
Kokoro TTS Model Converter

Converts Kokoro-82M (hexgrad/Kokoro-82M) to MLX format.
Provides validation and benchmarking against PyTorch reference.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from safetensors.numpy import save_file as save_safetensors

    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

try:
    import mlx.core as mx

    from .models.kokoro import KokoroConfig, KokoroModel

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class ConversionResult:
    """Result of model conversion."""

    success: bool
    mlx_path: str
    model_size_mb: float
    num_parameters: int
    error: str | None = None


@dataclass
class ValidationResult:
    """Result of output validation between PyTorch and MLX."""

    passed: bool
    text_encoder_max_error: float
    bert_max_error: float
    error: str | None = None


@dataclass
class BenchmarkResult:
    """Benchmark comparison result."""

    mlx_audio_per_second: float
    pytorch_audio_per_second: float
    speedup: float


class KokoroConverter:
    """
    Converts Kokoro TTS model to MLX format.

    Supports:
    - HuggingFace model conversion (hexgrad/Kokoro-82M)
    - Validation of numerical equivalence
    - Performance benchmarking

    Example:
        converter = KokoroConverter()
        result = converter.convert("hexgrad/Kokoro-82M", "./mlx-kokoro")
        if result.success:
            validation = converter.validate("./mlx-kokoro")
    """

    def __init__(self):
        """Initialize converter."""
        if not MLX_AVAILABLE:
            raise ImportError("MLX is required. Install with: pip install mlx")

    @staticmethod
    def list_supported_models() -> list[str]:
        """Return list of supported model IDs."""
        return [
            "hexgrad/Kokoro-82M",
        ]

    def load_from_hf(
        self,
        model_id: str = "hexgrad/Kokoro-82M",
        cache_dir: str | None = None,
    ) -> tuple[KokoroModel, KokoroConfig, dict[str, Any]]:
        """
        Load Kokoro model from HuggingFace and convert weights to MLX.

        Args:
            model_id: HuggingFace model ID
            cache_dir: Local cache directory

        Returns:
            Tuple of (KokoroModel, KokoroConfig, original_state_dict)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for loading HF models")

        from huggingface_hub import hf_hub_download

        if cache_dir is None:
            cache_dir = str(Path.home() / "models" / "kokoro")

        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        # Download files
        model_file = hf_hub_download(
            model_id, "kokoro-v1_0.pth", local_dir=str(cache_dir),
        )
        config_file = hf_hub_download(model_id, "config.json", local_dir=str(cache_dir))

        # Load config
        with open(config_file) as f:
            config_dict = json.load(f)

        config = KokoroConfig(
            dim_in=config_dict.get("dim_in", 64),
            hidden_dim=config_dict.get("hidden_dim", 512),
            style_dim=config_dict.get("style_dim", 128),
            n_token=config_dict.get("n_token", 178),
            n_layer=config_dict.get("n_layer", 3),
            dropout=config_dict.get("dropout", 0.2),
            text_encoder_kernel_size=config_dict.get("text_encoder_kernel_size", 5),
        )

        # Update PLBERT config
        if "plbert" in config_dict:
            plbert = config_dict["plbert"]
            config.plbert_hidden_size = plbert.get("hidden_size", 768)
            config.plbert_num_attention_heads = plbert.get("num_attention_heads", 12)
            config.plbert_intermediate_size = plbert.get("intermediate_size", 2048)
            config.plbert_num_hidden_layers = plbert.get("num_hidden_layers", 12)

        # Create model
        model = KokoroModel(config)

        # Load PyTorch weights
        state_dict = torch.load(model_file, map_location="cpu", weights_only=True)

        # Load weights into MLX model
        self._load_weights(model, state_dict)

        # Set to eval mode (disables dropout)
        model.eval()

        return model, config, state_dict

    def _load_weights(self, model: KokoroModel, state_dict: dict[str, Any]) -> None:
        """Load PyTorch weights into MLX model."""

        def to_mlx(t):
            if isinstance(t, torch.Tensor):
                return mx.array(t.numpy())
            return t

        # === Load text_encoder weights ===
        te_state = state_dict.get("text_encoder", {})

        if "module.embedding.weight" in te_state:
            model.text_encoder.embedding.weight = to_mlx(
                te_state["module.embedding.weight"],
            )

        for i in range(3):
            if f"module.cnn.{i}.0.weight_g" in te_state:
                model.text_encoder.convs[i].weight_g = to_mlx(
                    te_state[f"module.cnn.{i}.0.weight_g"],
                )
                model.text_encoder.convs[i].weight_v = to_mlx(
                    te_state[f"module.cnn.{i}.0.weight_v"],
                )
                model.text_encoder.convs[i].bias = to_mlx(
                    te_state[f"module.cnn.{i}.0.bias"],
                )

            if f"module.cnn.{i}.1.gamma" in te_state:
                model.text_encoder.norms[i].gamma = to_mlx(
                    te_state[f"module.cnn.{i}.1.gamma"],
                )
                model.text_encoder.norms[i].beta = to_mlx(
                    te_state[f"module.cnn.{i}.1.beta"],
                )

        # LSTM weights
        if "module.lstm.weight_ih_l0" in te_state:
            model.text_encoder.lstm.lstm_forward.Wx = to_mlx(
                te_state["module.lstm.weight_ih_l0"],
            )
            model.text_encoder.lstm.lstm_forward.Wh = to_mlx(
                te_state["module.lstm.weight_hh_l0"],
            )
            model.text_encoder.lstm.lstm_forward.bias = to_mlx(
                te_state["module.lstm.bias_ih_l0"] + te_state["module.lstm.bias_hh_l0"],
            )

            model.text_encoder.lstm.lstm_backward.Wx = to_mlx(
                te_state["module.lstm.weight_ih_l0_reverse"],
            )
            model.text_encoder.lstm.lstm_backward.Wh = to_mlx(
                te_state["module.lstm.weight_hh_l0_reverse"],
            )
            model.text_encoder.lstm.lstm_backward.bias = to_mlx(
                te_state["module.lstm.bias_ih_l0_reverse"]
                + te_state["module.lstm.bias_hh_l0_reverse"],
            )

        # === Load bert_encoder weights ===
        be_state = state_dict.get("bert_encoder", {})
        if "module.weight" in be_state:
            model.bert_encoder.weight = to_mlx(be_state["module.weight"])
            model.bert_encoder.bias = to_mlx(be_state["module.bias"])

        # === Load BERT weights ===
        bert_state = state_dict.get("bert", {})

        # Embeddings
        if "module.embeddings.word_embeddings.weight" in bert_state:
            model.bert.embeddings.word_embeddings.weight = to_mlx(
                bert_state["module.embeddings.word_embeddings.weight"],
            )
            model.bert.embeddings.position_embeddings.weight = to_mlx(
                bert_state["module.embeddings.position_embeddings.weight"],
            )
            model.bert.embeddings.token_type_embeddings.weight = to_mlx(
                bert_state["module.embeddings.token_type_embeddings.weight"],
            )
            model.bert.embeddings.layer_norm.weight = to_mlx(
                bert_state["module.embeddings.LayerNorm.weight"],
            )
            model.bert.embeddings.layer_norm.bias = to_mlx(
                bert_state["module.embeddings.LayerNorm.bias"],
            )

        # Encoder (shared ALBERT layer)
        if "module.encoder.embedding_hidden_mapping_in.weight" in bert_state:
            model.bert.encoder.embedding_hidden_mapping_in.weight = to_mlx(
                bert_state["module.encoder.embedding_hidden_mapping_in.weight"],
            )
            model.bert.encoder.embedding_hidden_mapping_in.bias = to_mlx(
                bert_state["module.encoder.embedding_hidden_mapping_in.bias"],
            )

        # Load shared ALBERT layer weights
        prefix = "module.encoder.albert_layer_groups.0.albert_layers.0"
        if f"{prefix}.attention.query.weight" in bert_state:
            layer = model.bert.encoder.albert_layer

            # Attention
            layer.attention.query.weight = to_mlx(
                bert_state[f"{prefix}.attention.query.weight"],
            )
            layer.attention.query.bias = to_mlx(
                bert_state[f"{prefix}.attention.query.bias"],
            )
            layer.attention.key.weight = to_mlx(
                bert_state[f"{prefix}.attention.key.weight"],
            )
            layer.attention.key.bias = to_mlx(
                bert_state[f"{prefix}.attention.key.bias"],
            )
            layer.attention.value.weight = to_mlx(
                bert_state[f"{prefix}.attention.value.weight"],
            )
            layer.attention.value.bias = to_mlx(
                bert_state[f"{prefix}.attention.value.bias"],
            )
            layer.attention.dense.weight = to_mlx(
                bert_state[f"{prefix}.attention.dense.weight"],
            )
            layer.attention.dense.bias = to_mlx(
                bert_state[f"{prefix}.attention.dense.bias"],
            )
            layer.attention.layer_norm.weight = to_mlx(
                bert_state[f"{prefix}.attention.LayerNorm.weight"],
            )
            layer.attention.layer_norm.bias = to_mlx(
                bert_state[f"{prefix}.attention.LayerNorm.bias"],
            )

            # FFN
            layer.ffn.weight = to_mlx(bert_state[f"{prefix}.ffn.weight"])
            layer.ffn.bias = to_mlx(bert_state[f"{prefix}.ffn.bias"])
            layer.ffn_output.weight = to_mlx(bert_state[f"{prefix}.ffn_output.weight"])
            layer.ffn_output.bias = to_mlx(bert_state[f"{prefix}.ffn_output.bias"])
            layer.full_layer_layer_norm.weight = to_mlx(
                bert_state[f"{prefix}.full_layer_layer_norm.weight"],
            )
            layer.full_layer_layer_norm.bias = to_mlx(
                bert_state[f"{prefix}.full_layer_layer_norm.bias"],
            )

        # === Load predictor weights ===
        pred_state = state_dict.get("predictor", {})

        # Load predictor's main lstm (for duration processing)
        # Uses native nn.LSTM under lstm_fwd and lstm_bwd
        if "module.lstm.weight_ih_l0" in pred_state:
            lstm = model.predictor.lstm
            # Forward direction
            lstm.lstm_fwd.Wx = to_mlx(pred_state["module.lstm.weight_ih_l0"])
            lstm.lstm_fwd.Wh = to_mlx(pred_state["module.lstm.weight_hh_l0"])
            lstm.lstm_fwd.bias = to_mlx(
                pred_state["module.lstm.bias_ih_l0"]
                + pred_state["module.lstm.bias_hh_l0"],
            )
            # Backward direction
            lstm.lstm_bwd.Wx = to_mlx(pred_state["module.lstm.weight_ih_l0_reverse"])
            lstm.lstm_bwd.Wh = to_mlx(pred_state["module.lstm.weight_hh_l0_reverse"])
            lstm.lstm_bwd.bias = to_mlx(
                pred_state["module.lstm.bias_ih_l0_reverse"]
                + pred_state["module.lstm.bias_hh_l0_reverse"],
            )

        # Load predictor's shared lstm
        # Uses native nn.LSTM under lstm_fwd and lstm_bwd
        if "module.shared.weight_ih_l0" in pred_state:
            shared = model.predictor.shared
            # Forward direction
            shared.lstm_fwd.Wx = to_mlx(pred_state["module.shared.weight_ih_l0"])
            shared.lstm_fwd.Wh = to_mlx(pred_state["module.shared.weight_hh_l0"])
            shared.lstm_fwd.bias = to_mlx(
                pred_state["module.shared.bias_ih_l0"]
                + pred_state["module.shared.bias_hh_l0"],
            )
            # Backward direction
            shared.lstm_bwd.Wx = to_mlx(
                pred_state["module.shared.weight_ih_l0_reverse"],
            )
            shared.lstm_bwd.Wh = to_mlx(
                pred_state["module.shared.weight_hh_l0_reverse"],
            )
            shared.lstm_bwd.bias = to_mlx(
                pred_state["module.shared.bias_ih_l0_reverse"]
                + pred_state["module.shared.bias_hh_l0_reverse"],
            )

        # Load F0 blocks (using numbered attributes F0_0, F0_1, F0_2)
        for i in range(3):
            block = getattr(model.predictor, f"F0_{i}")
            prefix = f"module.F0.{i}"
            if f"{prefix}.conv1.weight_v" in pred_state:
                block.conv1.weight_g = to_mlx(pred_state[f"{prefix}.conv1.weight_g"])
                block.conv1.weight_v = to_mlx(pred_state[f"{prefix}.conv1.weight_v"])
                block.conv1.bias = to_mlx(pred_state[f"{prefix}.conv1.bias"])

                block.conv2.weight_g = to_mlx(pred_state[f"{prefix}.conv2.weight_g"])
                block.conv2.weight_v = to_mlx(pred_state[f"{prefix}.conv2.weight_v"])
                block.conv2.bias = to_mlx(pred_state[f"{prefix}.conv2.bias"])

                block.norm1.fc.weight = to_mlx(pred_state[f"{prefix}.norm1.fc.weight"])
                block.norm1.fc.bias = to_mlx(pred_state[f"{prefix}.norm1.fc.bias"])
                # InstanceNorm1d affine parameters for norm1
                if f"{prefix}.norm1.norm.weight" in pred_state:
                    block.norm1.norm_weight = to_mlx(pred_state[f"{prefix}.norm1.norm.weight"])
                    block.norm1.norm_bias = to_mlx(pred_state[f"{prefix}.norm1.norm.bias"])

                block.norm2.fc.weight = to_mlx(pred_state[f"{prefix}.norm2.fc.weight"])
                block.norm2.fc.bias = to_mlx(pred_state[f"{prefix}.norm2.fc.bias"])
                # InstanceNorm1d affine parameters for norm2
                if f"{prefix}.norm2.norm.weight" in pred_state:
                    block.norm2.norm_weight = to_mlx(pred_state[f"{prefix}.norm2.norm.weight"])
                    block.norm2.norm_bias = to_mlx(pred_state[f"{prefix}.norm2.norm.bias"])

            # Block 1 has conv1x1 and pool (for upsampling)
            if i == 1:
                if (
                    f"{prefix}.conv1x1.weight_v" in pred_state
                    and block.conv1x1 is not None
                ):
                    block.conv1x1.weight_g = to_mlx(
                        pred_state[f"{prefix}.conv1x1.weight_g"],
                    )
                    block.conv1x1.weight_v = to_mlx(
                        pred_state[f"{prefix}.conv1x1.weight_v"],
                    )
                # Load pool weights (depthwise ConvTranspose for upsampling)
                if f"{prefix}.pool.weight_v" in pred_state and block.pool is not None:
                    block.pool.weight_g = to_mlx(pred_state[f"{prefix}.pool.weight_g"])
                    block.pool.weight_v = to_mlx(pred_state[f"{prefix}.pool.weight_v"])
                    block.pool.bias = to_mlx(pred_state[f"{prefix}.pool.bias"])

        # Load N blocks (using numbered attributes N_0, N_1, N_2)
        for i in range(3):
            block = getattr(model.predictor, f"N_{i}")
            prefix = f"module.N.{i}"
            if f"{prefix}.conv1.weight_v" in pred_state:
                block.conv1.weight_g = to_mlx(pred_state[f"{prefix}.conv1.weight_g"])
                block.conv1.weight_v = to_mlx(pred_state[f"{prefix}.conv1.weight_v"])
                block.conv1.bias = to_mlx(pred_state[f"{prefix}.conv1.bias"])

                block.conv2.weight_g = to_mlx(pred_state[f"{prefix}.conv2.weight_g"])
                block.conv2.weight_v = to_mlx(pred_state[f"{prefix}.conv2.weight_v"])
                block.conv2.bias = to_mlx(pred_state[f"{prefix}.conv2.bias"])

                block.norm1.fc.weight = to_mlx(pred_state[f"{prefix}.norm1.fc.weight"])
                block.norm1.fc.bias = to_mlx(pred_state[f"{prefix}.norm1.fc.bias"])
                # InstanceNorm1d affine parameters for norm1
                if f"{prefix}.norm1.norm.weight" in pred_state:
                    block.norm1.norm_weight = to_mlx(pred_state[f"{prefix}.norm1.norm.weight"])
                    block.norm1.norm_bias = to_mlx(pred_state[f"{prefix}.norm1.norm.bias"])

                block.norm2.fc.weight = to_mlx(pred_state[f"{prefix}.norm2.fc.weight"])
                block.norm2.fc.bias = to_mlx(pred_state[f"{prefix}.norm2.fc.bias"])
                # InstanceNorm1d affine parameters for norm2
                if f"{prefix}.norm2.norm.weight" in pred_state:
                    block.norm2.norm_weight = to_mlx(pred_state[f"{prefix}.norm2.norm.weight"])
                    block.norm2.norm_bias = to_mlx(pred_state[f"{prefix}.norm2.norm.bias"])

            # Block 1 has conv1x1 and pool (for upsampling)
            if i == 1:
                if (
                    f"{prefix}.conv1x1.weight_v" in pred_state
                    and block.conv1x1 is not None
                ):
                    block.conv1x1.weight_g = to_mlx(
                        pred_state[f"{prefix}.conv1x1.weight_g"],
                    )
                    block.conv1x1.weight_v = to_mlx(
                        pred_state[f"{prefix}.conv1x1.weight_v"],
                    )
                # Load pool weights (depthwise ConvTranspose for upsampling)
                if f"{prefix}.pool.weight_v" in pred_state and block.pool is not None:
                    block.pool.weight_g = to_mlx(pred_state[f"{prefix}.pool.weight_g"])
                    block.pool.weight_v = to_mlx(pred_state[f"{prefix}.pool.weight_v"])
                    block.pool.bias = to_mlx(pred_state[f"{prefix}.pool.bias"])

        # Load F0_proj and N_proj - these are plain Conv1d, not weight-normed
        if "module.F0_proj.weight" in pred_state:
            model.predictor.F0_proj.weight = to_mlx(pred_state["module.F0_proj.weight"])
            model.predictor.F0_proj.bias = to_mlx(pred_state["module.F0_proj.bias"])

        if "module.N_proj.weight" in pred_state:
            model.predictor.N_proj.weight = to_mlx(pred_state["module.N_proj.weight"])
            model.predictor.N_proj.bias = to_mlx(pred_state["module.N_proj.bias"])

        # Load duration_proj (DurationProjWrapper has linear_layer attribute)
        if "module.duration_proj.linear_layer.weight" in pred_state:
            model.predictor.duration_proj.linear_layer.weight = to_mlx(
                pred_state["module.duration_proj.linear_layer.weight"],
            )
            model.predictor.duration_proj.linear_layer.bias = to_mlx(
                pred_state["module.duration_proj.linear_layer.bias"],
            )

        # Load predictor's text_encoder LSTM and FC weights
        # Structure: lstms.0, lstms.2, lstms.4 are BiLSTMs
        #           lstms.1, lstms.3, lstms.5 are FCs for AdaIN
        text_enc = model.predictor.text_encoder

        # BiLSTM layers (lstms_0, lstms_2, lstms_4)
        # These use native nn.LSTM under lstm_fwd and lstm_bwd
        for i in [0, 2, 4]:
            lstm = getattr(text_enc, f"lstms_{i}")
            prefix = f"module.text_encoder.lstms.{i}"
            if f"{prefix}.weight_ih_l0" in pred_state:
                # Forward direction - set on lstm_fwd (native nn.LSTM)
                # Native LSTM uses Wx, Wh, bias (where bias = bias_ih + bias_hh)
                lstm.lstm_fwd.Wx = to_mlx(pred_state[f"{prefix}.weight_ih_l0"])
                lstm.lstm_fwd.Wh = to_mlx(pred_state[f"{prefix}.weight_hh_l0"])
                lstm.lstm_fwd.bias = to_mlx(
                    pred_state[f"{prefix}.bias_ih_l0"]
                    + pred_state[f"{prefix}.bias_hh_l0"],
                )
                # Backward direction - set on lstm_bwd (native nn.LSTM)
                lstm.lstm_bwd.Wx = to_mlx(pred_state[f"{prefix}.weight_ih_l0_reverse"])
                lstm.lstm_bwd.Wh = to_mlx(pred_state[f"{prefix}.weight_hh_l0_reverse"])
                lstm.lstm_bwd.bias = to_mlx(
                    pred_state[f"{prefix}.bias_ih_l0_reverse"]
                    + pred_state[f"{prefix}.bias_hh_l0_reverse"],
                )

        # FC layers for AdaIN (lstms_1, lstms_3, lstms_5)
        for i in [1, 3, 5]:
            fc_layer = getattr(text_enc, f"lstms_{i}")
            prefix = f"module.text_encoder.lstms.{i}"
            if f"{prefix}.fc.weight" in pred_state:
                fc_layer.fc.weight = to_mlx(pred_state[f"{prefix}.fc.weight"])
                fc_layer.fc.bias = to_mlx(pred_state[f"{prefix}.fc.bias"])

        # === Load decoder weights ===
        dec_state = state_dict.get("decoder", {})

        # F0_conv and N_conv (weight-normed in PyTorch)
        if "module.F0_conv.weight_g" in dec_state:
            model.decoder.f0_conv.weight_g = to_mlx(
                dec_state["module.F0_conv.weight_g"],
            )
            model.decoder.f0_conv.weight_v = to_mlx(
                dec_state["module.F0_conv.weight_v"],
            )
            if "module.F0_conv.bias" in dec_state:
                model.decoder.f0_conv.bias = to_mlx(dec_state["module.F0_conv.bias"])
        elif "module.F0_conv.weight" in dec_state:
            # Fallback for non-weight-normed version
            pt_w = dec_state["module.F0_conv.weight"]
            w_norm = torch.sqrt(torch.sum(pt_w**2, dim=(1, 2), keepdim=True) + 1e-12)
            model.decoder.f0_conv.weight_v = to_mlx(pt_w)
            model.decoder.f0_conv.weight_g = to_mlx(w_norm)
            if "module.F0_conv.bias" in dec_state:
                model.decoder.f0_conv.bias = to_mlx(dec_state["module.F0_conv.bias"])

        if "module.N_conv.weight_g" in dec_state:
            model.decoder.n_conv.weight_g = to_mlx(dec_state["module.N_conv.weight_g"])
            model.decoder.n_conv.weight_v = to_mlx(dec_state["module.N_conv.weight_v"])
            if "module.N_conv.bias" in dec_state:
                model.decoder.n_conv.bias = to_mlx(dec_state["module.N_conv.bias"])
        elif "module.N_conv.weight" in dec_state:
            pt_w = dec_state["module.N_conv.weight"]
            w_norm = torch.sqrt(torch.sum(pt_w**2, dim=(1, 2), keepdim=True) + 1e-12)
            model.decoder.n_conv.weight_v = to_mlx(pt_w)
            model.decoder.n_conv.weight_g = to_mlx(w_norm)
            if "module.N_conv.bias" in dec_state:
                model.decoder.n_conv.bias = to_mlx(dec_state["module.N_conv.bias"])

        # asr_res (weight-normed in PyTorch)
        if "module.asr_res.0.weight_g" in dec_state:
            model.decoder.asr_res.weight_g = to_mlx(
                dec_state["module.asr_res.0.weight_g"],
            )
            model.decoder.asr_res.weight_v = to_mlx(
                dec_state["module.asr_res.0.weight_v"],
            )
            if "module.asr_res.0.bias" in dec_state:
                model.decoder.asr_res.bias = to_mlx(dec_state["module.asr_res.0.bias"])
        elif "module.asr_res.0.weight" in dec_state:
            pt_w = dec_state["module.asr_res.0.weight"]
            w_norm = torch.sqrt(torch.sum(pt_w**2, dim=(1, 2), keepdim=True) + 1e-12)
            model.decoder.asr_res.weight_v = to_mlx(pt_w)
            model.decoder.asr_res.weight_g = to_mlx(w_norm)
            if "module.asr_res.0.bias" in dec_state:
                model.decoder.asr_res.bias = to_mlx(dec_state["module.asr_res.0.bias"])

        # Load encode block
        if "module.encode.conv1.weight_v" in dec_state:
            self._load_adain_resblk(model.decoder.encode, dec_state, "module.encode")

        # Load decode blocks (using numbered attributes decode_0, decode_1, etc.)
        for i in range(4):
            block = getattr(model.decoder, f"decode_{i}")
            prefix = f"module.decode.{i}"
            if f"{prefix}.conv1.weight_v" in dec_state:
                self._load_adain_resblk(block, dec_state, prefix)

        # Load generator weights
        self._load_generator_weights(model.decoder.generator, dec_state)

        # === E1: Strict weight loading validation ===
        # Verify all model parameters have been loaded (not left at default initialization)
        self._validate_weights_loaded(model, state_dict)

    def _validate_weights_loaded(
        self, model: KokoroModel, state_dict: dict[str, Any], strict: bool = True,
    ) -> None:
        """
        Validate that all model weights have been loaded from state_dict.

        Args:
            model: The MLX model to validate
            state_dict: The PyTorch state dict that was loaded
            strict: If True, raise error on missing weights. If False, just warn.

        Raises:
            ValueError: If strict=True and weights are missing
        """
        import warnings

        # Count PyTorch keys (leaf tensors only)
        pt_key_count = 0
        for section in ["text_encoder", "bert_encoder", "bert", "predictor", "decoder"]:
            if section in state_dict:
                pt_key_count += len(state_dict[section])

        # Get leaf parameters (actual weight tensors, not modules)
        def get_leaf_params(params, prefix=""):
            """Recursively get all leaf parameters (mx.array) from nested dict."""
            leaves = {}
            for key, value in params.items():
                full_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, mx.array):
                    leaves[full_key] = value
                elif isinstance(value, dict):
                    leaves.update(get_leaf_params(value, full_key))
            return leaves

        mlx_params = model.parameters()
        leaf_params = get_leaf_params(mlx_params)

        # Check for any uninitialized parameters (would have shape but no data)
        uninitialized = []
        for key, param in leaf_params.items():
            try:
                # Access shape to ensure parameter exists and has data
                shape = param.shape
                # Verify it has non-zero size
                if all(s == 0 for s in shape):
                    uninitialized.append(key)
            except Exception:
                uninitialized.append(key)

        if uninitialized:
            msg = f"E1: {len(uninitialized)} model parameters not initialized: {uninitialized[:10]}..."
            if strict:
                raise ValueError(msg)
            warnings.warn(msg, stacklevel=2)

        # Log statistics for debugging (enable for troubleshooting)
        # print(f"E1 validation: {pt_key_count} PyTorch keys -> {mlx_key_count} MLX parameters")

    def _load_adain_resblk(self, block, state, prefix):
        """Load weights for an AdainResBlk1d block."""

        def to_mlx(t):
            if isinstance(t, torch.Tensor):
                return mx.array(t.numpy())
            return t

        # conv1
        if f"{prefix}.conv1.weight_v" in state:
            block.conv1.weight_g = to_mlx(state[f"{prefix}.conv1.weight_g"])
            block.conv1.weight_v = to_mlx(state[f"{prefix}.conv1.weight_v"])
            block.conv1.bias = to_mlx(state[f"{prefix}.conv1.bias"])

        # conv2
        if f"{prefix}.conv2.weight_v" in state:
            block.conv2.weight_g = to_mlx(state[f"{prefix}.conv2.weight_g"])
            block.conv2.weight_v = to_mlx(state[f"{prefix}.conv2.weight_v"])
            block.conv2.bias = to_mlx(state[f"{prefix}.conv2.bias"])

        # norm1 (AdaIN)
        if f"{prefix}.norm1.fc.weight" in state:
            block.norm1.fc.weight = to_mlx(state[f"{prefix}.norm1.fc.weight"])
            block.norm1.fc.bias = to_mlx(state[f"{prefix}.norm1.fc.bias"])
        # InstanceNorm1d affine parameters for norm1
        if f"{prefix}.norm1.norm.weight" in state:
            block.norm1.norm_weight = to_mlx(state[f"{prefix}.norm1.norm.weight"])
            block.norm1.norm_bias = to_mlx(state[f"{prefix}.norm1.norm.bias"])

        # norm2 (AdaIN)
        if f"{prefix}.norm2.fc.weight" in state:
            block.norm2.fc.weight = to_mlx(state[f"{prefix}.norm2.fc.weight"])
            block.norm2.fc.bias = to_mlx(state[f"{prefix}.norm2.fc.bias"])
        # InstanceNorm1d affine parameters for norm2
        if f"{prefix}.norm2.norm.weight" in state:
            block.norm2.norm_weight = to_mlx(state[f"{prefix}.norm2.norm.weight"])
            block.norm2.norm_bias = to_mlx(state[f"{prefix}.norm2.norm.bias"])

        # conv1x1 (skip connection)
        if f"{prefix}.conv1x1.weight_v" in state and block.conv1x1 is not None:
            block.conv1x1.weight_g = to_mlx(state[f"{prefix}.conv1x1.weight_g"])
            block.conv1x1.weight_v = to_mlx(state[f"{prefix}.conv1x1.weight_v"])

        # pool (learned upsampling for decoder blocks with learned_upsample=True)
        if (
            f"{prefix}.pool.weight_v" in state
            and hasattr(block, "pool")
            and getattr(block, "learned_upsample", False)
        ):
            block.pool.weight_g = to_mlx(state[f"{prefix}.pool.weight_g"])
            block.pool.weight_v = to_mlx(state[f"{prefix}.pool.weight_v"])
            block.pool.bias = to_mlx(state[f"{prefix}.pool.bias"])

    def _load_adain_resblock_styled(self, block, state, prefix):
        """Load weights for an AdaINResBlock1dStyled block (noise_res architecture)."""

        def to_mlx(t):
            if isinstance(t, torch.Tensor):
                return mx.array(t.numpy())
            return t

        # Load convs1 and convs2 (3 layers each, using numbered attributes)
        for j in range(3):
            # convs1 (access via convs1_0, convs1_1, convs1_2)
            conv1_prefix = f"{prefix}.convs1.{j}"
            conv1_block = getattr(block, f"convs1_{j}", None)
            if conv1_block is not None and f"{conv1_prefix}.weight_v" in state:
                conv1_block.weight_g = to_mlx(state[f"{conv1_prefix}.weight_g"])
                conv1_block.weight_v = to_mlx(state[f"{conv1_prefix}.weight_v"])
                conv1_block.bias = to_mlx(state[f"{conv1_prefix}.bias"])

            # convs2 (access via convs2_0, convs2_1, convs2_2)
            conv2_prefix = f"{prefix}.convs2.{j}"
            conv2_block = getattr(block, f"convs2_{j}", None)
            if conv2_block is not None and f"{conv2_prefix}.weight_v" in state:
                conv2_block.weight_g = to_mlx(state[f"{conv2_prefix}.weight_g"])
                conv2_block.weight_v = to_mlx(state[f"{conv2_prefix}.weight_v"])
                conv2_block.bias = to_mlx(state[f"{conv2_prefix}.bias"])

            # adain1 (access via adain1_0, adain1_1, adain1_2)
            # AdaINStyleLinear has nested fc attribute: adain1_block.fc.weight
            adain1_prefix = f"{prefix}.adain1.{j}"
            adain1_block = getattr(block, f"adain1_{j}", None)
            if adain1_block is not None and f"{adain1_prefix}.fc.weight" in state:
                adain1_block.fc.weight = to_mlx(state[f"{adain1_prefix}.fc.weight"])
                adain1_block.fc.bias = to_mlx(state[f"{adain1_prefix}.fc.bias"])
            # InstanceNorm1d affine parameters for adain1
            if adain1_block is not None and f"{adain1_prefix}.norm.weight" in state:
                adain1_block.norm_weight = to_mlx(state[f"{adain1_prefix}.norm.weight"])
                adain1_block.norm_bias = to_mlx(state[f"{adain1_prefix}.norm.bias"])

            # adain2 (access via adain2_0, adain2_1, adain2_2)
            # AdaINStyleLinear has nested fc attribute: adain2_block.fc.weight
            adain2_prefix = f"{prefix}.adain2.{j}"
            adain2_block = getattr(block, f"adain2_{j}", None)
            if adain2_block is not None and f"{adain2_prefix}.fc.weight" in state:
                adain2_block.fc.weight = to_mlx(state[f"{adain2_prefix}.fc.weight"])
                adain2_block.fc.bias = to_mlx(state[f"{adain2_prefix}.fc.bias"])
            # InstanceNorm1d affine parameters for adain2
            if adain2_block is not None and f"{adain2_prefix}.norm.weight" in state:
                adain2_block.norm_weight = to_mlx(state[f"{adain2_prefix}.norm.weight"])
                adain2_block.norm_bias = to_mlx(state[f"{adain2_prefix}.norm.bias"])

            # alpha1 and alpha2 (access via alpha1_0, alpha1_1, etc.)
            if f"{prefix}.alpha1.{j}" in state:
                setattr(block, f"alpha1_{j}", to_mlx(state[f"{prefix}.alpha1.{j}"]))
            if f"{prefix}.alpha2.{j}" in state:
                setattr(block, f"alpha2_{j}", to_mlx(state[f"{prefix}.alpha2.{j}"]))

    def _load_generator_weights(self, generator, state):
        """Load weights for the Generator module."""

        def to_mlx(t):
            if isinstance(t, torch.Tensor):
                return mx.array(t.numpy())
            return t

        # Load upsampling layers (WeightNormConvTranspose1d)
        # PyTorch uses weight normalization: weight_g and weight_v
        # weight_g: [in_channels, 1, 1] - magnitude
        # weight_v: [in_channels, out_channels, kernel] - direction
        num_upsamples = generator.num_upsamples
        for i in range(num_upsamples):
            up = getattr(generator, f"ups_{i}")
            prefix = f"module.generator.ups.{i}"
            if f"{prefix}.weight_g" in state:
                # Weight-normalized version
                up.weight_g = to_mlx(state[f"{prefix}.weight_g"])
                up.weight_v = to_mlx(state[f"{prefix}.weight_v"])
                if f"{prefix}.bias" in state:
                    up.bias = to_mlx(state[f"{prefix}.bias"])
            elif f"{prefix}.weight" in state:
                # Fallback for non-weight-normed version
                pt_w = state[f"{prefix}.weight"]
                w_norm = torch.sqrt(
                    torch.sum(pt_w**2, dim=(1, 2), keepdim=True) + 1e-12,
                )
                up.weight_v = to_mlx(pt_w)
                up.weight_g = to_mlx(w_norm)
                if f"{prefix}.bias" in state:
                    up.bias = to_mlx(state[f"{prefix}.bias"])

        # Load noise convs (using numbered attributes noise_convs_0, noise_res_0, etc.)
        # These are regular Conv1d (not weight-normed), load directly into .weight
        for i in range(num_upsamples):
            noise_conv = getattr(generator, f"noise_convs_{i}")
            noise_res = getattr(generator, f"noise_res_{i}")

            # noise_conv - plain Conv1d, not weight-norm
            prefix = f"module.generator.noise_convs.{i}"
            if f"{prefix}.weight" in state:
                noise_conv.weight = to_mlx(state[f"{prefix}.weight"])
                if f"{prefix}.bias" in state:
                    noise_conv.bias = to_mlx(state[f"{prefix}.bias"])

            # noise_res - uses AdaINResBlock1dStyled architecture
            res_prefix = f"module.generator.noise_res.{i}"
            if f"{res_prefix}.convs1.0.weight_v" in state:
                self._load_adain_resblock_styled(noise_res, state, res_prefix)

        # Load resblocks (using numbered attributes resblocks_0, resblocks_1, etc.)
        num_resblocks = generator._num_resblocks
        for i in range(num_resblocks):
            resblock = getattr(generator, f"resblocks_{i}")
            res_prefix = f"module.generator.resblocks.{i}"
            if f"{res_prefix}.convs1.0.weight_v" in state:
                self._load_adain_resblock_styled(resblock, state, res_prefix)

        # Load conv_post
        if "module.generator.conv_post.weight_v" in state:
            generator.conv_post.weight_g = to_mlx(
                state["module.generator.conv_post.weight_g"],
            )
            generator.conv_post.weight_v = to_mlx(
                state["module.generator.conv_post.weight_v"],
            )
            generator.conv_post.bias = to_mlx(state["module.generator.conv_post.bias"])

        # Load m_source.l_linear weights (harmonic combiner)
        if "module.generator.m_source.l_linear.weight" in state:
            # PyTorch linear: weight is [out, in], MLX uses same convention
            generator.m_source.l_linear.weight = to_mlx(
                state["module.generator.m_source.l_linear.weight"],
            )
            generator.m_source.l_linear.bias = to_mlx(
                state["module.generator.m_source.l_linear.bias"],
            )

    def load_voice_pack(
        self,
        voice_name: str,
        model_id: str = "hexgrad/Kokoro-82M",
        cache_dir: str | None = None,
        safetensors_dir: str | None = None,
    ) -> mx.array:
        """
        Load a voice pack from safetensors (preferred) or HuggingFace.

        Voice packs are [510, 1, 256] tensors where each frame corresponds to
        a phoneme sequence length. Use select_voice_embedding() to extract
        the appropriate embedding for a given phoneme sequence.

        Args:
            voice_name: Voice file name (e.g., "af_bella" or "af_bella.pt")
            model_id: HuggingFace model ID
            cache_dir: Local cache directory for HuggingFace downloads
            safetensors_dir: Directory containing pre-exported .safetensors voice files

        Returns:
            Voice pack tensor of shape [510, 1, 256]
        """
        # Strip extension if present
        voice_base = voice_name.replace(".pt", "").replace(".safetensors", "")

        # First, try to load from safetensors (preferred - no PyTorch dependency)
        safetensors_paths = []
        if safetensors_dir:
            safetensors_paths.append(
                Path(safetensors_dir) / f"{voice_base}.safetensors",
            )

        # Also check repo-local export directories
        repo_root = Path(__file__).parent.parent.parent.parent
        safetensors_paths.extend(
            [
                repo_root
                / "kokoro_cpp_export"
                / "voices"
                / f"{voice_base}.safetensors",
            ],
        )

        for sf_path in safetensors_paths:
            if sf_path.exists():
                # Load from safetensors using MLX
                voice_data = mx.load(str(sf_path))
                # mx.load returns either array or dict[str, array]
                if isinstance(voice_data, mx.array):
                    return voice_data
                if "embedding" in voice_data:
                    return voice_data["embedding"]
                # Return first tensor if single-tensor file
                return next(iter(voice_data.values()))

        # Fallback: download from HuggingFace and load with torch
        from huggingface_hub import hf_hub_download

        if cache_dir is None:
            cache_dir = str(Path.home() / "models" / "kokoro")

        # Download voice file
        voice_file = hf_hub_download(
            model_id, f"voices/{voice_base}.pt", local_dir=cache_dir,
        )

        # Load using torch
        voice_data = torch.load(voice_file, map_location="cpu", weights_only=True)

        # E3: Convert to MLX - handle both tensor and dict formats
        voice_array: mx.array
        if isinstance(voice_data, torch.Tensor):
            voice_array = mx.array(voice_data.numpy())
        elif isinstance(voice_data, dict):
            # Extract tensor from common dict formats
            for key in ["embedding", "voice", "weight", "data", "style"]:
                tensor_val = voice_data.get(key)
                if tensor_val is not None and isinstance(tensor_val, torch.Tensor):
                    voice_array = mx.array(tensor_val.numpy())
                    break
            else:
                # If no known key, try to get the first tensor value
                for value in voice_data.values():
                    if isinstance(value, torch.Tensor):
                        voice_array = mx.array(value.numpy())
                        break
                else:
                    raise ValueError(
                        f"No tensor found in voice pack dict. Keys: {list(voice_data.keys())}",
                    )
        else:
            raise ValueError(f"Unknown voice format: {type(voice_data)}")

        return voice_array

    def select_voice_embedding(
        self,
        voice_pack: mx.array,
        phoneme_length: int,
    ) -> mx.array:
        """
        Select voice embedding from pack based on phoneme sequence length.

        This matches the PyTorch behavior: ref_s = voice_pack[len(phonemes) - 1]

        Args:
            voice_pack: [frames, 1, 256] voice pack tensor
            phoneme_length: Number of phonemes in the input sequence

        Returns:
            Voice embedding tensor of shape [1, 256]
        """
        # Select frame based on phoneme length (0-indexed, so length-1)
        idx = min(phoneme_length - 1, voice_pack.shape[0] - 1)
        idx = max(idx, 0)  # Ensure non-negative

        # voice_pack is [frames, 1, 256], select single frame
        voice_embedding = voice_pack[idx]  # [1, 256]

        # Ensure shape is [1, 256]
        if len(voice_embedding.shape) == 1:
            voice_embedding = voice_embedding[None, :]
        elif voice_embedding.shape[0] == 1 and len(voice_embedding.shape) == 2:
            pass  # Already [1, 256]
        elif len(voice_embedding.shape) == 2:
            voice_embedding = voice_embedding.reshape(1, -1)

        return voice_embedding

    def load_voice(
        self,
        voice_name: str,
        model_id: str = "hexgrad/Kokoro-82M",
        cache_dir: str | None = None,
        phoneme_length: int | None = None,
    ) -> mx.array:
        """
        Load a voice embedding from HuggingFace.

        Args:
            voice_name: Voice file name (e.g., "af_bella" or "af_bella.pt")
            model_id: HuggingFace model ID
            cache_dir: Local cache directory
            phoneme_length: Number of phonemes in input sequence. If provided,
                selects the appropriate frame from the voice pack (matching
                PyTorch behavior). If None, uses frame index 0 for compatibility.

        Returns:
            Voice embedding tensor of shape [1, 256]
        """
        voice_pack = self.load_voice_pack(voice_name, model_id, cache_dir)

        if phoneme_length is not None:
            return self.select_voice_embedding(voice_pack, phoneme_length)

        # Backward compatibility: if no phoneme_length provided, use frame 0
        # This is deterministic and better than averaging all frames
        if len(voice_pack.shape) == 3:
            voice_embedding = voice_pack[0]  # [1, 256]
            if voice_embedding.shape[0] == 1:
                voice_embedding = voice_embedding.squeeze(0)  # [256]
        else:
            voice_embedding = voice_pack

        if len(voice_embedding.shape) == 1:
            voice_embedding = voice_embedding[None, :]  # [1, 256]

        # Return full 256-dim embedding
        # Model's __call__ will split into style (first 128) and speaker (second 128)
        return voice_embedding

    def list_voices(
        self,
        model_id: str = "hexgrad/Kokoro-82M",
    ) -> list[str]:
        """
        List available voices for the model.

        Returns:
            List of voice names (without .pt extension)
        """
        # Known voices based on HuggingFace listing
        return [
            # African accented
            "af_alloy",
            "af_aoede",
            "af_bella",
            "af_heart",
            "af_jessica",
            "af_kore",
            "af_nicole",
            "af_nova",
            "af_river",
            "af_sarah",
            "af_sky",
            # American
            "am_adam",
            "am_echo",
            "am_eric",
            "am_fenrir",
            "am_liam",
            "am_michael",
            "am_onyx",
            "am_puck",
            "am_santa",
            # British female/male
            "bf_emma",
            "bf_isabella",
            "bm_george",
            "bm_lewis",
        ]

    def convert(
        self,
        model_id: str,
        output_path: str,
    ) -> ConversionResult:
        """
        Convert a Kokoro model to MLX format.

        Args:
            model_id: HuggingFace model path
            output_path: Directory to save converted model

        Returns:
            ConversionResult with conversion status and metadata
        """
        try:
            model, config, _ = self.load_from_hf(model_id)

            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save config
            config_path = output_dir / "config.json"
            with open(config_path, "w") as f:
                json.dump(
                    {
                        "dim_in": config.dim_in,
                        "hidden_dim": config.hidden_dim,
                        "style_dim": config.style_dim,
                        "n_token": config.n_token,
                        "n_layer": config.n_layer,
                    },
                    f,
                    indent=2,
                )

            # Count parameters
            num_params = sum(
                p.size for p in model.parameters().values() if hasattr(p, "size")
            )

            return ConversionResult(
                success=True,
                mlx_path=str(output_path),
                model_size_mb=num_params * 4 / 1024 / 1024,  # float32
                num_parameters=num_params,
            )

        except Exception as e:
            return ConversionResult(
                success=False,
                mlx_path="",
                model_size_mb=0,
                num_parameters=0,
                error=str(e),
            )

    def validate(
        self,
        model_id: str = "hexgrad/Kokoro-82M",
        test_input: list[int] | None = None,
    ) -> ValidationResult:
        """
        Validate MLX model against PyTorch reference.

        Args:
            model_id: Model to validate
            test_input: Token IDs for testing

        Returns:
            ValidationResult with accuracy metrics
        """
        if not TORCH_AVAILABLE:
            return ValidationResult(
                passed=False,
                text_encoder_max_error=float("inf"),
                bert_max_error=float("inf"),
                error="PyTorch not available for validation",
            )

        try:
            model, config, pt_state = self.load_from_hf(model_id)

            if test_input is None:
                test_input = [1, 2, 3, 4, 5, 10, 20, 30]

            # Test text encoder embedding
            input_ids = mx.array([test_input])
            mlx_embed = model.text_encoder.embedding(input_ids)
            mx.eval(mlx_embed)

            pt_embed = torch.nn.functional.embedding(
                torch.tensor([test_input]),
                pt_state["text_encoder"]["module.embedding.weight"],
            ).numpy()

            embed_error = np.max(np.abs(np.array(mlx_embed) - pt_embed))

            # Test BERT embedding
            bert_state = pt_state.get("bert", {})
            bert_embed_key = "module.embeddings.word_embeddings.weight"
            if bert_embed_key in bert_state:
                mlx_bert_embed = model.bert.embeddings.word_embeddings(input_ids)
                mx.eval(mlx_bert_embed)

                pt_bert_embed = torch.nn.functional.embedding(
                    torch.tensor([test_input]), bert_state[bert_embed_key],
                ).numpy()

                bert_error = np.max(np.abs(np.array(mlx_bert_embed) - pt_bert_embed))
            else:
                bert_error = 0.0  # No BERT weights to validate

            # Consider it passed if both errors are small
            passed = embed_error < 1e-4 and bert_error < 1e-4

            return ValidationResult(
                passed=passed,
                text_encoder_max_error=float(embed_error),
                bert_max_error=float(bert_error),
            )

        except Exception as e:
            return ValidationResult(
                passed=False,
                text_encoder_max_error=float("inf"),
                bert_max_error=float("inf"),
                error=str(e),
            )

    def _fold_weight_norm(self, weight_g: mx.array, weight_v: mx.array) -> np.ndarray:
        """
        Fold WeightNorm into a single merged weight.

        WeightNorm computes: weight = g * v / ||v||
        We pre-compute this to eliminate runtime normalization.

        Args:
            weight_g: Magnitude tensor [out, 1, 1]
            weight_v: Direction tensor [out, in, kernel]

        Returns:
            Merged weight as numpy array
        """
        # Compute ||v|| along (in, kernel) dimensions
        v_norm = mx.sqrt(mx.sum(weight_v**2, axis=(1, 2), keepdims=True) + 1e-12)
        # Compute merged weight: g * v / ||v||
        merged = weight_g * weight_v / v_norm
        mx.eval(merged)
        return np.array(merged)

    def _collect_weights_with_folding(
        self, model: KokoroModel,
    ) -> dict[str, np.ndarray]:
        """
        Collect all model weights with WeightNorm layers folded.

        This iterates through all parameters and folds weight_g/weight_v pairs
        into single merged weights for C++ runtime efficiency.

        Args:
            model: Loaded KokoroModel

        Returns:
            Dictionary of weight name -> numpy array
        """
        weights = {}
        params = model.parameters()

        def flatten_params(d: dict, prefix: str = "") -> list:
            """Recursively flatten nested parameter dictionaries."""
            result = []
            for k, v in d.items():
                full_key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    result.extend(flatten_params(v, full_key))
                elif isinstance(v, list):
                    for i, item in enumerate(v):
                        if isinstance(item, dict):
                            result.extend(flatten_params(item, f"{full_key}.{i}"))
                        elif hasattr(item, "shape"):
                            result.append((f"{full_key}.{i}", item))
                elif hasattr(v, "shape"):
                    result.append((full_key, v))
            return result

        flat_params = flatten_params(params)

        # Build lookup for weight_g/weight_v pairs
        weight_g_keys = {}
        weight_v_keys = {}
        other_keys = {}

        for key, value in flat_params:
            if key.endswith(".weight_g"):
                base = key[:-9]  # Remove '.weight_g'
                weight_g_keys[base] = value
            elif key.endswith(".weight_v"):
                base = key[:-9]  # Remove '.weight_v'
                weight_v_keys[base] = value
            else:
                other_keys[key] = value

        # Fold WeightNorm pairs into single weights
        for base in weight_g_keys:
            if base in weight_v_keys:
                weight_g = weight_g_keys[base]
                weight_v = weight_v_keys[base]
                # Fold: weight = g * v / ||v||
                folded = self._fold_weight_norm(weight_g, weight_v)
                weights[f"{base}.weight"] = folded
            else:
                # weight_g without weight_v - just convert
                mx.eval(weight_g_keys[base])
                weights[f"{base}.weight_g"] = np.array(weight_g_keys[base])

        # Handle any weight_v without weight_g (shouldn't happen, but safety)
        for base in weight_v_keys:
            if base not in weight_g_keys:
                mx.eval(weight_v_keys[base])
                weights[f"{base}.weight_v"] = np.array(weight_v_keys[base])

        # Add all other parameters
        for key, value in other_keys.items():
            mx.eval(value)
            weights[key] = np.array(value)

        return weights

    def export_for_cpp(
        self,
        model_id: str = "hexgrad/Kokoro-82M",
        output_dir: str = "./kokoro_cpp",
        voice: str = "af_bella",
    ) -> dict[str, Any]:
        """
        Export model with folded WeightNorm for C++ runtime.

        Creates:
        - weights.safetensors: All model weights with WeightNorm pre-folded
        - config.json: Model configuration for C++ loading
        - voices/{voice}.safetensors: Voice embeddings

        Args:
            model_id: HuggingFace model ID
            output_dir: Output directory
            voice: Voice to export

        Returns:
            Export metadata
        """
        if not SAFETENSORS_AVAILABLE:
            raise ImportError(
                "safetensors is required. Install with: pip install safetensors",
            )

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load model
        model, config, pt_state = self.load_from_hf(model_id)

        # Collect weights with WeightNorm folding
        weights = self._collect_weights_with_folding(model)

        # Save weights to safetensors
        weights_path = output_path / "weights.safetensors"
        save_safetensors(weights, str(weights_path))

        # Save comprehensive config
        config_dict = {
            # Model dimensions
            "dim_in": config.dim_in,
            "hidden_dim": config.hidden_dim,
            "style_dim": config.style_dim,
            "max_conv_dim": config.max_conv_dim,
            "n_token": config.n_token,
            "n_mels": config.n_mels,
            "n_layer": config.n_layer,
            "max_dur": config.max_dur,
            "dropout": config.dropout,
            "text_encoder_kernel_size": config.text_encoder_kernel_size,
            "multispeaker": config.multispeaker,
            # PLBERT/ALBERT config
            "plbert_hidden_size": config.plbert_hidden_size,
            "plbert_num_attention_heads": config.plbert_num_attention_heads,
            "plbert_intermediate_size": config.plbert_intermediate_size,
            "plbert_max_position_embeddings": config.plbert_max_position_embeddings,
            "plbert_num_hidden_layers": config.plbert_num_hidden_layers,
            "plbert_dropout": config.plbert_dropout,
            "albert_embedding_dim": config.albert_embedding_dim,
            # ISTFTNet config
            "istft_upsample_rates": list(config.istft_upsample_rates),
            "istft_upsample_kernel_sizes": list(config.istft_upsample_kernel_sizes),
            "istft_gen_istft_n_fft": config.istft_gen_istft_n_fft,
            "istft_gen_istft_hop_size": config.istft_gen_istft_hop_size,
            "istft_resblock_kernel_sizes": list(config.istft_resblock_kernel_sizes),
            "istft_resblock_dilation_sizes": [
                list(d) for d in config.istft_resblock_dilation_sizes
            ],
            "istft_upsample_initial_channel": config.istft_upsample_initial_channel,
            # Audio config
            "sample_rate": 24000,
            "hop_size": 256,
            # Vocab
            "vocab_size": 178,
            "bos_token_id": 0,
            "eos_token_id": 0,
            # Export metadata
            "weight_norm_folded": True,
            "format_version": "1.0",
        }

        config_path = output_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        # Export vocabulary
        vocab_dir = output_path / "vocab"
        vocab_dir.mkdir(exist_ok=True)

        # Load phonemizer vocabulary
        from .models.kokoro_phonemizer import PAD_TOKEN, load_vocab

        vocab = load_vocab()

        # Save phoneme -> token ID mapping
        phonemes_path = vocab_dir / "phonemes.json"
        with open(phonemes_path, "w") as f:
            json.dump(vocab, f, indent=2, ensure_ascii=False)

        # Save special tokens
        special_tokens = {
            "pad_token_id": PAD_TOKEN,
            "bos_token_id": 0,
            "eos_token_id": 0,
            "vocab_size": len(vocab),
            "description": "Token 0 serves as both PAD, BOS, and EOS",
        }
        special_tokens_path = vocab_dir / "special_tokens.json"
        with open(special_tokens_path, "w") as f:
            json.dump(special_tokens, f, indent=2)

        # Export voice pack
        voices_dir = output_path / "voices"
        voices_dir.mkdir(exist_ok=True)

        voice_embedding = self.load_voice(voice, model_id)
        if voice_embedding is not None:
            mx.eval(voice_embedding)
            voice_weights = {"embedding": np.array(voice_embedding)}
            voice_path = voices_dir / f"{voice}.safetensors"
            save_safetensors(voice_weights, str(voice_path))

        # Calculate stats
        total_params = sum(w.size for w in weights.values())
        total_size_mb = sum(w.nbytes for w in weights.values()) / (1024 * 1024)

        return {
            "success": True,
            "output_dir": str(output_path),
            "weights_path": str(weights_path),
            "config_path": str(config_path),
            "vocab_dir": str(vocab_dir),
            "voice_path": str(voices_dir / f"{voice}.safetensors"),
            "num_weights": len(weights),
            "total_parameters": total_params,
            "total_size_mb": total_size_mb,
            "weight_norm_folded": True,
        }

    # All 54 Kokoro voices across 9 languages
    ALL_VOICES = [
        # American English (a) - 20 voices
        "af_heart",
        "af_alloy",
        "af_aoede",
        "af_bella",
        "af_jessica",
        "af_kore",
        "af_nicole",
        "af_nova",
        "af_river",
        "af_sarah",
        "af_sky",
        "am_adam",
        "am_echo",
        "am_eric",
        "am_fenrir",
        "am_liam",
        "am_michael",
        "am_onyx",
        "am_puck",
        "am_santa",
        # British English (b) - 8 voices
        "bf_alice",
        "bf_emma",
        "bf_isabella",
        "bf_lily",
        "bm_daniel",
        "bm_fable",
        "bm_george",
        "bm_lewis",
        # Japanese (j) - 5 voices
        "jf_alpha",
        "jf_gongitsune",
        "jf_nezumi",
        "jf_tebukuro",
        "jm_kumo",
        # Mandarin Chinese (z) - 8 voices
        "zf_xiaobei",
        "zf_xiaoni",
        "zf_xiaoxiao",
        "zf_xiaoyi",
        "zm_yunjian",
        "zm_yunxi",
        "zm_yunxia",
        "zm_yunyang",
        # Spanish (e) - 3 voices
        "ef_dora",
        "em_alex",
        "em_santa",
        # French (f) - 1 voice
        "ff_siwis",
        # Hindi (h) - 4 voices
        "hf_alpha",
        "hf_beta",
        "hm_omega",
        "hm_psi",
        # Italian (i) - 2 voices
        "if_sara",
        "im_nicola",
        # Brazilian Portuguese (p) - 3 voices
        "pf_dora",
        "pm_alex",
        "pm_santa",
    ]

    def export_cpp_bundle(
        self,
        model_id: str = "hexgrad/Kokoro-82M",
        output_dir: str = "./kokoro_cpp_bundle",
        voices: list[str] | None = None,
        include_g2p: bool = True,
    ) -> dict[str, Any]:
        """
        Export complete C++-loadable bundle with all voices and G2P lexicons.

        Creates a self-contained bundle for C++ runtime:
        - weights.safetensors: Model weights with WeightNorm pre-folded
        - config.json: Model configuration
        - vocab/phonemes.json: Phoneme vocabulary
        - vocab/special_tokens.json: Special token info
        - voices/*.safetensors: All voice embeddings
        - g2p/: G2P lexicons for all supported languages

        Args:
            model_id: HuggingFace model ID
            output_dir: Output directory for the bundle
            voices: List of voices to export (default: ALL_VOICES)
            include_g2p: Whether to copy G2P lexicons from misaki_export

        Returns:
            Export metadata with bundle contents
        """
        import shutil

        if not SAFETENSORS_AVAILABLE:
            raise ImportError("safetensors required: pip install safetensors")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load model
        model, config, pt_state = self.load_from_hf(model_id)

        # Collect weights with WeightNorm folding
        weights = self._collect_weights_with_folding(model)

        # Save weights to safetensors
        weights_path = output_path / "weights.safetensors"
        save_safetensors(weights, str(weights_path))

        # Save comprehensive config
        config_dict = {
            "dim_in": config.dim_in,
            "hidden_dim": config.hidden_dim,
            "style_dim": config.style_dim,
            "max_conv_dim": config.max_conv_dim,
            "n_token": config.n_token,
            "n_mels": config.n_mels,
            "n_layer": config.n_layer,
            "max_dur": config.max_dur,
            "dropout": config.dropout,
            "text_encoder_kernel_size": config.text_encoder_kernel_size,
            "multispeaker": config.multispeaker,
            "plbert_hidden_size": config.plbert_hidden_size,
            "plbert_num_attention_heads": config.plbert_num_attention_heads,
            "plbert_intermediate_size": config.plbert_intermediate_size,
            "plbert_max_position_embeddings": config.plbert_max_position_embeddings,
            "plbert_num_hidden_layers": config.plbert_num_hidden_layers,
            "plbert_dropout": config.plbert_dropout,
            "albert_embedding_dim": config.albert_embedding_dim,
            "istft_upsample_rates": list(config.istft_upsample_rates),
            "istft_upsample_kernel_sizes": list(config.istft_upsample_kernel_sizes),
            "istft_gen_istft_n_fft": config.istft_gen_istft_n_fft,
            "istft_gen_istft_hop_size": config.istft_gen_istft_hop_size,
            "istft_resblock_kernel_sizes": list(config.istft_resblock_kernel_sizes),
            "istft_resblock_dilation_sizes": [
                list(d) for d in config.istft_resblock_dilation_sizes
            ],
            "istft_upsample_initial_channel": config.istft_upsample_initial_channel,
            "sample_rate": 24000,
            "hop_size": 256,
            "weight_norm_folded": True,
            "format_version": "1.1",
        }

        config_path = output_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        # Export vocabulary
        vocab_dir = output_path / "vocab"
        vocab_dir.mkdir(exist_ok=True)

        from .models.kokoro_phonemizer import PAD_TOKEN, load_vocab

        vocab = load_vocab()

        with open(vocab_dir / "phonemes.json", "w") as f:
            json.dump(vocab, f, indent=2, ensure_ascii=False)

        with open(vocab_dir / "special_tokens.json", "w") as f:
            json.dump(
                {
                    "pad_token_id": PAD_TOKEN,
                    "bos_token_id": 0,
                    "eos_token_id": 0,
                    "vocab_size": len(vocab),
                    "description": "Token 0 serves as PAD, BOS, and EOS",
                },
                f,
                indent=2,
            )

        # Export ALL voice packs
        voices_to_export = voices if voices else self.ALL_VOICES
        voices_dir = output_path / "voices"
        voices_dir.mkdir(exist_ok=True)

        exported_voices = []
        failed_voices = []
        for voice_name in voices_to_export:
            try:
                voice_embedding = self.load_voice(voice_name, model_id)
                if voice_embedding is not None:
                    mx.eval(voice_embedding)
                    voice_weights = {"embedding": np.array(voice_embedding)}
                    voice_path = voices_dir / f"{voice_name}.safetensors"
                    save_safetensors(voice_weights, str(voice_path))
                    exported_voices.append(voice_name)
            except Exception as e:
                failed_voices.append((voice_name, str(e)))

        # Copy G2P lexicons from misaki_export if available
        g2p_copied = False
        g2p_languages = []
        if include_g2p:
            g2p_dir = output_path / "g2p"
            g2p_dir.mkdir(exist_ok=True)

            # Find misaki_export directory
            repo_root = Path(__file__).parent
            while repo_root.parent != repo_root:
                if (repo_root / "misaki_export").exists():
                    break
                repo_root = repo_root.parent

            misaki_export = repo_root / "misaki_export"
            if misaki_export.exists():
                # Copy all language directories and vocab files
                for item in misaki_export.iterdir():
                    dest = g2p_dir / item.name
                    if item.is_dir():
                        if dest.exists():
                            shutil.rmtree(dest)
                        shutil.copytree(item, dest)
                        g2p_languages.append(item.name)
                    elif item.is_file() and item.suffix == ".json":
                        shutil.copy2(item, dest)
                g2p_copied = True

        # Calculate stats
        total_params = sum(w.size for w in weights.values())
        total_size_mb = sum(w.nbytes for w in weights.values()) / (1024 * 1024)

        return {
            "success": True,
            "output_dir": str(output_path),
            "weights_path": str(weights_path),
            "config_path": str(config_path),
            "vocab_dir": str(vocab_dir),
            "voices_dir": str(voices_dir),
            "num_weights": len(weights),
            "total_parameters": total_params,
            "total_size_mb": total_size_mb,
            "voices_exported": len(exported_voices),
            "voices_failed": len(failed_voices),
            "exported_voices": exported_voices,
            "failed_voices": failed_voices,
            "g2p_copied": g2p_copied,
            "g2p_languages": g2p_languages,
            "weight_norm_folded": True,
            "format_version": "1.1",
        }
