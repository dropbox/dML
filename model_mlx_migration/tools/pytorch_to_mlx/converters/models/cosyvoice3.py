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
CosyVoice3 Full Model - MLX Implementation

Fun-CosyVoice 3.0 (0.5B model) text-to-speech system.

Architecture:
1. Text → LLM (Qwen2) → Speech Tokens
2. Speech Tokens + Speaker → DiT Flow → Mel Spectrogram
3. Mel → CausalHiFT Vocoder → Audio Waveform

Key differences from CosyVoice2:
- DiT (Diffusion Transformer) replaces encoder-decoder flow
- CausalHiFT vocoder for streaming support
- Full causal architecture for real-time generation
- Improved quality and naturalness

Total Parameters: ~775M (similar to v2)
- LLM: ~642M (Qwen2-based)
- Flow (DiT): ~113M
- Vocoder: ~21M
"""

from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

# Reuse LLM from v2 - architecture is similar
from .cosyvoice2_llm import CosyVoice2LLM, Qwen2Config
from .cosyvoice3_dit import (
    CausalMaskedDiffWithDiT,
    DiTConfig,
    create_cosyvoice3_flow_config,
)
from .cosyvoice3_vocoder import (
    CausalHiFTConfig,
    CausalHiFTGenerator,
    create_cosyvoice3_vocoder_config,
)


@dataclass
class CosyVoice3Config:
    """Full CosyVoice3 model configuration."""

    # Sample rate
    sample_rate: int = 24000

    # Token settings
    token_frame_rate: int = 25      # 25 tokens per second
    token_mel_ratio: int = 2        # 2 mel frames per token
    speech_token_size: int = 6561   # Speech token vocabulary

    # Streaming
    chunk_size: int = 25            # Streaming chunk size in tokens

    # Component configs
    llm_config: Qwen2Config | None = None
    flow_config: DiTConfig | None = None
    vocoder_config: CausalHiFTConfig | None = None

    def __post_init__(self):
        if self.llm_config is None:
            self.llm_config = Qwen2Config(
                hidden_size=896,
                num_hidden_layers=24,
                num_attention_heads=14,
                num_key_value_heads=2,
                intermediate_size=4864,
                vocab_size=151936,
                rope_theta=1000000.0,
            )
        if self.flow_config is None:
            self.flow_config = create_cosyvoice3_flow_config()
        if self.vocoder_config is None:
            self.vocoder_config = create_cosyvoice3_vocoder_config()


class CosyVoice3Model(nn.Module):
    """
    Full CosyVoice3 TTS Model.

    Pipeline:
    1. encode_text() - Text to LLM hidden states
    2. generate_speech_tokens() - LLM generates speech tokens autoregressively
    3. tokens_to_mel() - DiT flow converts tokens to mel spectrogram
    4. mel_to_audio() - CausalHiFT vocoder converts mel to waveform

    Use synthesize() for end-to-end synthesis.
    """

    def __init__(self, config: CosyVoice3Config):
        super().__init__()
        self.config = config

        # Initialize components
        assert config.llm_config is not None
        assert config.flow_config is not None
        assert config.vocoder_config is not None

        self.llm = CosyVoice2LLM(config.llm_config)
        self.flow = CausalMaskedDiffWithDiT(config.flow_config)
        self.vocoder = CausalHiFTGenerator(config.vocoder_config)

    def generate_speech_tokens(
        self,
        text_ids: mx.array,
        max_length: int = 1000,
        temperature: float = 1.0,
        top_k: int = 25,
        top_p: float = 0.8,
    ) -> mx.array:
        """
        Generate speech tokens from text.

        Args:
            text_ids: Text token IDs [B, L_text]
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling

        Returns:
            Speech tokens [B, L_speech]
        """
        return self.llm.generate(
            text_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    def tokens_to_mel(
        self,
        tokens: mx.array,
        speaker_embedding: mx.array,
        num_steps: int = 10,
        cfg_strength: float = 0.7,
        streaming: bool = False,
    ) -> mx.array:
        """
        Convert speech tokens to mel spectrogram using DiT flow.

        Args:
            tokens: Speech tokens [B, L_tokens]
            speaker_embedding: Speaker embedding [B, 192]
            num_steps: Number of ODE steps for flow matching
            cfg_strength: Classifier-free guidance strength
            streaming: Enable streaming mode

        Returns:
            Mel spectrogram [B, L_mel, mel_dim]
        """
        return self.flow.inference(
            tokens,
            speaker_embedding,
            num_steps=num_steps,
            cfg_strength=cfg_strength,
            streaming=streaming,
        )

    def mel_to_audio(
        self,
        mel: mx.array,
        finalize: bool = True,
    ) -> mx.array:
        """
        Convert mel spectrogram to audio waveform.

        Args:
            mel: Mel spectrogram [B, mel_dim, L] or [B, L, mel_dim]
            finalize: Whether this is the final chunk

        Returns:
            Audio waveform [B, L_audio]
        """
        # Ensure mel is [B, C, L]
        if mel.shape[-1] == self.config.vocoder_config.in_channels:
            mel = mel.transpose(0, 2, 1)

        return self.vocoder(mel, finalize=finalize)

    def synthesize(
        self,
        text_ids: mx.array,
        speaker_embedding: mx.array,
        max_tokens: int = 1000,
        temperature: float = 1.0,
        top_k: int = 25,
        top_p: float = 0.8,
        flow_steps: int = 10,
        cfg_strength: float = 0.7,
    ) -> mx.array:
        """
        End-to-end text-to-speech synthesis.

        Args:
            text_ids: Text token IDs [B, L_text]
            speaker_embedding: Speaker embedding [B, 192]
            max_tokens: Maximum speech tokens to generate
            temperature: LLM sampling temperature
            top_k: LLM top-k sampling
            top_p: LLM top-p sampling
            flow_steps: Number of flow ODE steps
            cfg_strength: Classifier-free guidance strength

        Returns:
            Audio waveform [B, L_audio]
        """
        # Step 1: Generate speech tokens
        tokens = self.generate_speech_tokens(
            text_ids,
            max_length=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        # Step 2: Convert to mel via DiT flow
        mel = self.tokens_to_mel(
            tokens,
            speaker_embedding,
            num_steps=flow_steps,
            cfg_strength=cfg_strength,
        )

        # Step 3: Convert to audio via vocoder
        return self.mel_to_audio(mel)


    def synthesize_streaming(
        self,
        text_ids: mx.array,
        speaker_embedding: mx.array,
        chunk_size: int | None = None,
        **kwargs,
    ) -> Generator[mx.array, None, None]:
        """
        Streaming text-to-speech synthesis.

        Yields audio chunks as they are generated.

        Args:
            text_ids: Text token IDs [B, L_text]
            speaker_embedding: Speaker embedding [B, 192]
            chunk_size: Tokens per chunk (default from config)
            **kwargs: Additional arguments for synthesize()

        Yields:
            Audio chunks [B, L_chunk]
        """
        chunk_size = chunk_size or self.config.chunk_size

        # Generate all tokens first (could be streaming too)
        tokens = self.generate_speech_tokens(
            text_ids,
            max_length=kwargs.get("max_tokens", 1000),
            temperature=kwargs.get("temperature", 1.0),
            top_k=kwargs.get("top_k", 25),
            top_p=kwargs.get("top_p", 0.8),
        )

        # Process in chunks
        num_chunks = (tokens.shape[1] + chunk_size - 1) // chunk_size

        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, tokens.shape[1])
            chunk_tokens = tokens[:, start:end]

            # Flow matching
            mel = self.flow.inference(
                chunk_tokens,
                speaker_embedding,
                num_steps=kwargs.get("flow_steps", 10),
                cfg_strength=kwargs.get("cfg_strength", 0.7),
                streaming=True,
            )

            # Vocoder
            is_final = (i == num_chunks - 1)
            audio = self.mel_to_audio(mel, finalize=is_final)

            yield audio

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        dtype: mx.Dtype = mx.float16,
    ) -> "CosyVoice3Model":
        """
        Load CosyVoice3 from pretrained weights.

        Args:
            model_path: Path to model directory
            dtype: Model dtype (float16 recommended)

        Returns:
            Loaded CosyVoice3Model
        """
        model_path = Path(model_path)

        # Load config
        config = CosyVoice3Config()

        # Create model
        model = cls(config)

        # Load weights
        # LLM weights from CosyVoice-BlankEN/model.safetensors
        llm_path = model_path / "CosyVoice-BlankEN" / "model.safetensors"
        if llm_path.exists():
            llm_weights = mx.load(str(llm_path))
            # Map weights to LLM
            _load_llm_weights(model.llm, llm_weights)

        # Flow weights from flow.pt
        flow_path = model_path / "flow.pt"
        if flow_path.exists():
            # Note: .pt files need PyTorch for loading
            # This is a placeholder - actual conversion needed
            pass

        # Vocoder weights from hift.pt
        hift_path = model_path / "hift.pt"
        if hift_path.exists():
            # Note: .pt files need PyTorch for loading
            # This is a placeholder - actual conversion needed
            pass

        return model

    @staticmethod
    def get_default_model_path() -> Path:
        """Get default model path."""
        return Path("models/cosyvoice3")


def _load_llm_weights(llm: CosyVoice2LLM, weights: dict):
    """Load LLM weights from safetensors format."""
    # This maps HuggingFace Qwen2 weights to our LLM structure
    # Keys are like: "model.layers.0.self_attn.q_proj.weight"

    for key, value in weights.items():
        parts = key.split(".")

        if parts[0] == "model":
            if parts[1] == "embed_tokens":
                llm.embed_tokens.weight = value
            elif parts[1] == "layers":
                layer_idx = int(parts[2])
                layer = llm.layers[layer_idx]

                if parts[3] == "self_attn":
                    if parts[4] == "q_proj":
                        layer.self_attn.q_proj.weight = value
                    elif parts[4] == "k_proj":
                        layer.self_attn.k_proj.weight = value
                    elif parts[4] == "v_proj":
                        layer.self_attn.v_proj.weight = value
                    elif parts[4] == "o_proj":
                        layer.self_attn.o_proj.weight = value
                elif parts[3] == "mlp":
                    if parts[4] == "gate_proj":
                        layer.mlp.gate_proj.weight = value
                    elif parts[4] == "up_proj":
                        layer.mlp.up_proj.weight = value
                    elif parts[4] == "down_proj":
                        layer.mlp.down_proj.weight = value
                elif parts[3] == "input_layernorm":
                    layer.input_layernorm.weight = value
                elif parts[3] == "post_attention_layernorm":
                    layer.post_attention_layernorm.weight = value
            elif parts[1] == "norm":
                llm.norm.weight = value
        elif parts[0] == "lm_head":
            llm.lm_head.weight = value


def create_cosyvoice3_config() -> CosyVoice3Config:
    """Create default CosyVoice3 config matching the official model."""
    return CosyVoice3Config(
        sample_rate=24000,
        token_frame_rate=25,
        token_mel_ratio=2,
        speech_token_size=6561,
        chunk_size=25,
    )
