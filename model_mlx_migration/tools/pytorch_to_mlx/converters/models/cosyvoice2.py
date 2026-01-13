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
CosyVoice2 Full Model - MLX Implementation

Integrates all CosyVoice2 components:
- LLM (Qwen2-based text encoder and token generator)
- Flow (conditional flow matching for tokens to mel)
- Vocoder (HiFi-GAN for mel to audio)

Architecture:
1. Text → LLM → Speech Tokens
2. Speech Tokens + Speaker Embedding → Flow → Mel Spectrogram
3. Mel Spectrogram → Vocoder → Audio Waveform

Total Parameters: ~775M
- LLM: 642M
- Flow: 113M
- Vocoder: 21M
"""

from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .cosyvoice2_flow import FlowMatchingConfig, MaskedDiffWithXvec
from .cosyvoice2_llm import CosyVoice2LLM, Qwen2Config
from .cosyvoice2_tokenizer import CosyVoice2Tokenizer
from .cosyvoice2_vocoder import HiFiGANConfig, HiFiGANVocoder


@dataclass
class CosyVoice2Config:
    """Full CosyVoice2 model configuration."""

    # Sample rate
    sample_rate: int = 24000  # 24kHz audio output

    # Token settings
    token_frame_rate: int = 25  # 25 tokens per second
    token_mel_ratio: int = 2  # 2 mel frames per token

    # Streaming settings
    chunk_size: int = 25  # Streaming chunk size in tokens

    # Model configs
    llm_config: Qwen2Config | None = None
    flow_config: FlowMatchingConfig | None = None
    vocoder_config: HiFiGANConfig | None = None

    def __post_init__(self):
        if self.llm_config is None:
            self.llm_config = Qwen2Config()
        if self.flow_config is None:
            self.flow_config = FlowMatchingConfig()
        if self.vocoder_config is None:
            self.vocoder_config = HiFiGANConfig()


class CosyVoice2Model(nn.Module):
    """
    Full CosyVoice2 TTS Model.

    Pipeline:
    1. encode_text() - Text to LLM hidden states
    2. generate_speech_tokens() - LLM generates speech tokens
    3. tokens_to_mel() - Flow model converts tokens to mel
    4. mel_to_audio() - Vocoder converts mel to waveform

    Or use synthesize() for end-to-end synthesis.
    Or use synthesize_text() for direct text-to-speech.
    """

    def __init__(self, config: CosyVoice2Config):
        super().__init__()
        self.config = config

        # Initialize components (configs guaranteed non-None by __post_init__)
        assert config.llm_config is not None
        assert config.flow_config is not None
        assert config.vocoder_config is not None
        self.llm = CosyVoice2LLM(config.llm_config)
        self.flow = MaskedDiffWithXvec(config.flow_config)
        self.vocoder = HiFiGANVocoder(config.vocoder_config)

        # Tokenizer (loaded separately via from_pretrained)
        self.tokenizer: CosyVoice2Tokenizer | None = None

    def generate_speech_tokens(
        self,
        text_ids: mx.array,
        max_length: int = 1000,
        temperature: float = 1.0,
        top_k: int = 25,
        top_p: float = 0.8,
        repetition_penalty: float = 1.0,
    ) -> mx.array:
        """
        Generate speech tokens from text token IDs (DEPRECATED).

        NOTE: This method uses standard sampling which can cause loops.
        Use generate_speech_tokens_ras() instead for proper CosyVoice2 behavior.

        Args:
            text_ids: [batch, seq_len] - Tokenized text input
            max_length: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            repetition_penalty: Penalty for repetition

        Returns:
            speech_tokens: [batch, token_len] - Generated speech tokens
        """
        return self.llm.generate_speech_tokens(
            text_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

    def generate_speech_tokens_ras(
        self,
        text_ids: mx.array,
        max_length: int = 1000,
        top_k: int = 25,
        top_p: float = 0.8,
        win_size: int = 10,
        tau_r: float = 0.1,
        speech_token_size: int = 6561,
        min_token_text_ratio: float = 2.0,
        max_token_text_ratio: float = 20.0,
    ) -> mx.array:
        """
        Generate speech tokens using Repetition Aware Sampling (ras_sampling).

        This is the correct sampling method for CosyVoice2 that prevents
        repetitive/looping output. Uses the ras_sampling algorithm from
        CosyVoice2's cosyvoice.utils.common module.

        Termination:
        - Stop tokens are [6561, 6562, 6563] (any token >= speech_token_size)
        - Min/max length is computed from text length * ratio

        Args:
            text_ids: [batch, seq_len] - Tokenized text input
            max_length: Maximum number of tokens to generate
            top_k: Maximum tokens in nucleus (default 25)
            top_p: Nucleus sampling probability threshold (default 0.8)
            win_size: Window size for repetition check (default 10)
            tau_r: Repetition threshold ratio (default 0.1)
            speech_token_size: Size of speech vocabulary (default 6561)
            min_token_text_ratio: Min tokens per text token (default 2.0)
            max_token_text_ratio: Max tokens per text token (default 20.0)

        Returns:
            speech_tokens: [batch, token_len] - Generated speech tokens
        """
        return self.llm.generate_speech_tokens_ras(
            text_ids,
            max_length=max_length,
            top_k=top_k,
            top_p=top_p,
            win_size=win_size,
            tau_r=tau_r,
            speech_token_size=speech_token_size,
            min_token_text_ratio=min_token_text_ratio,
            max_token_text_ratio=max_token_text_ratio,
        )

    def generate_speech_tokens_stream(
        self,
        text_ids: mx.array,
        max_length: int = 1000,
        temperature: float = 1.0,
        top_k: int = 25,
        top_p: float = 0.8,
        repetition_penalty: float = 1.0,
        chunk_size: int = 25,
    ) -> Generator[tuple[mx.array, bool], None, None]:
        """
        Generate speech tokens with streaming.

        Yields:
            (token_chunk, is_final): Tuple of tokens and completion flag
        """
        yield from self.llm.generate_speech_tokens_stream(
            text_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            chunk_size=chunk_size,
        )

    def tokens_to_mel(
        self,
        speech_tokens: mx.array,
        speaker_embedding: mx.array,
        num_steps: int = 10,
        mel_length: int | None = None,
    ) -> mx.array:
        """
        Convert speech tokens to mel spectrogram using flow model.

        Args:
            speech_tokens: [batch, token_len] - Speech token IDs
            speaker_embedding: [batch, 192] - Speaker embedding (x-vector)
            num_steps: Number of ODE integration steps
            mel_length: Target mel length. If None, computed from token_len * token_mel_ratio.

        Returns:
            mel: [batch, mel_len, 80] - Mel spectrogram
        """
        # Compute mel length from tokens if not provided
        if mel_length is None:
            token_len = speech_tokens.shape[1]
            mel_length = token_len * self.config.token_mel_ratio

        return self.flow.generate(
            speech_tokens,
            speaker_embedding,
            mel_length=mel_length,
            num_steps=num_steps,
        )

    def mel_to_audio(self, mel: mx.array) -> mx.array:
        """
        Convert mel spectrogram to audio waveform.

        Args:
            mel: [batch, mel_len, 80] - Mel spectrogram

        Returns:
            audio: [batch, samples] - Audio waveform
        """
        return self.vocoder(mel)

    def synthesize(
        self,
        text_ids: mx.array,
        speaker_embedding: mx.array,
        max_tokens: int = 1000,
        temperature: float = 1.0,
        top_k: int = 25,
        top_p: float = 0.8,
        num_flow_steps: int = 10,
    ) -> mx.array:
        """
        End-to-end text-to-speech synthesis (DEPRECATED - use synthesize_ras).

        NOTE: This method uses standard sampling which can cause loops.
        Use synthesize_ras() instead for proper CosyVoice2 behavior.

        Args:
            text_ids: [batch, seq_len] - Tokenized text input
            speaker_embedding: [batch, 192] - Speaker embedding
            max_tokens: Maximum speech tokens to generate
            temperature: LLM sampling temperature
            top_k: LLM top-k sampling
            top_p: LLM nucleus sampling
            num_flow_steps: Flow ODE integration steps

        Returns:
            audio: [batch, samples] - Generated audio waveform
        """
        # Step 1: Generate speech tokens
        speech_tokens = self.generate_speech_tokens(
            text_ids,
            max_length=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        # Step 2: Convert tokens to mel
        mel = self.tokens_to_mel(
            speech_tokens,
            speaker_embedding,
            num_steps=num_flow_steps,
        )

        # Step 3: Convert mel to audio
        return self.mel_to_audio(mel)


    def synthesize_ras(
        self,
        text_ids: mx.array,
        speaker_embedding: mx.array,
        max_tokens: int = 1000,
        top_k: int = 25,
        top_p: float = 0.8,
        win_size: int = 10,
        tau_r: float = 0.1,
        speech_token_size: int = 6561,
        min_token_text_ratio: float = 2.0,
        max_token_text_ratio: float = 20.0,
        num_flow_steps: int = 10,
    ) -> mx.array:
        """
        End-to-end text-to-speech synthesis using Repetition Aware Sampling.

        This is the correct synthesis method for CosyVoice2 that prevents
        repetitive/looping output.

        Termination:
        - Stop tokens are [6561, 6562, 6563] (any token >= speech_token_size)
        - Min/max length is computed from text length * ratio

        Args:
            text_ids: [batch, seq_len] - Tokenized text input
            speaker_embedding: [batch, 192] - Speaker embedding
            max_tokens: Maximum speech tokens to generate
            top_k: LLM top-k sampling (default 25)
            top_p: LLM nucleus sampling (default 0.8)
            win_size: Repetition window size (default 10)
            tau_r: Repetition threshold ratio (default 0.1)
            speech_token_size: Size of speech vocabulary (default 6561)
            min_token_text_ratio: Min tokens per text token (default 2.0)
            max_token_text_ratio: Max tokens per text token (default 20.0)
            num_flow_steps: Flow ODE integration steps (default 10)

        Returns:
            audio: [batch, samples] - Generated audio waveform
        """
        # Step 1: Generate speech tokens using ras_sampling
        speech_tokens = self.generate_speech_tokens_ras(
            text_ids,
            max_length=max_tokens,
            top_k=top_k,
            top_p=top_p,
            win_size=win_size,
            tau_r=tau_r,
            speech_token_size=speech_token_size,
            min_token_text_ratio=min_token_text_ratio,
            max_token_text_ratio=max_token_text_ratio,
        )

        # Step 2: Convert tokens to mel
        mel = self.tokens_to_mel(
            speech_tokens,
            speaker_embedding,
            num_steps=num_flow_steps,
        )

        # Step 3: Convert mel to audio
        return self.mel_to_audio(mel)


    def synthesize_batch(
        self,
        text_ids_list: list[mx.array],
        speaker_embeddings: mx.array | list[mx.array],
        max_tokens: int = 1000,
        temperature: float = 1.0,
        top_k: int = 25,
        top_p: float = 0.8,
        num_flow_steps: int = 10,
        pad_token_id: int = 0,
    ) -> tuple[mx.array, mx.array]:
        """
        Batch text-to-speech synthesis for higher throughput.

        Processes multiple utterances in a single batch, sharing GPU overhead
        across all inputs. Useful for server workloads processing multiple
        requests simultaneously.

        Args:
            text_ids_list: List of [seq_len_i] arrays - Text token IDs for each utterance.
                          Each can have different lengths.
            speaker_embeddings: Either:
                - [batch, 192] array - same speaker for all utterances
                - List of [192] arrays - per-utterance speaker embeddings
            max_tokens: Maximum speech tokens to generate per utterance
            temperature: LLM sampling temperature
            top_k: LLM top-k sampling
            top_p: LLM nucleus sampling
            num_flow_steps: Flow ODE integration steps
            pad_token_id: Token ID for padding (default 0 = EOS)

        Returns:
            Tuple of:
                - audio: [batch, max_samples] - Generated audio (padded)
                - audio_lengths: [batch] - Actual audio length per utterance

        Example:
            >>> # Synthesize 3 utterances in one batch
            >>> texts = ["Hello world", "Good morning", "How are you doing today?"]
            >>> text_ids = [model.tokenizer.encode(t) for t in texts]
            >>> speaker = model.tokenizer.random_speaker_embedding()
            >>> speakers = mx.stack([speaker] * 3)  # [3, 192]
            >>> audio, lengths = model.synthesize_batch(text_ids, speakers)
            >>> # audio: [3, max_samples], lengths: [3]
            >>> # Extract individual audios:
            >>> audio_1 = audio[0, :lengths[0]]
        """
        batch_size = len(text_ids_list)
        if batch_size == 0:
            raise ValueError("Empty text_ids_list")

        # ====================================================================
        # Step 1: Prepare text inputs - pad to max length
        # ====================================================================

        # Ensure all inputs are 1D arrays
        for i, ids in enumerate(text_ids_list):
            if ids.ndim == 0:
                raise ValueError(f"text_ids_list[{i}] must be at least 1D")
            if ids.ndim == 2:
                if ids.shape[0] != 1:
                    raise ValueError(
                        f"text_ids_list[{i}] has batch dim > 1, expected single sequence",
                    )
                text_ids_list[i] = ids.squeeze(0)

        # Get sequence lengths and pad
        text_lengths = [ids.shape[0] for ids in text_ids_list]
        max_text_len = max(text_lengths)

        padded_text_ids = []
        for ids in text_ids_list:
            seq_len = ids.shape[0]
            if seq_len < max_text_len:
                padding = mx.full((max_text_len - seq_len,), pad_token_id, dtype=ids.dtype)
                padded = mx.concatenate([ids, padding])
            else:
                padded = ids
            padded_text_ids.append(padded)

        text_ids = mx.stack(padded_text_ids)  # [batch, max_text_len]

        # ====================================================================
        # Step 2: Prepare speaker embeddings
        # ====================================================================

        if isinstance(speaker_embeddings, list):
            if len(speaker_embeddings) != batch_size:
                raise ValueError(
                    f"speaker_embeddings list length {len(speaker_embeddings)} "
                    f"must match batch size {batch_size}",
                )
            # Stack into batch
            speaker_arrays = []
            for i, spk in enumerate(speaker_embeddings):
                if spk.ndim == 1:
                    speaker_arrays.append(spk)
                elif spk.ndim == 2 and spk.shape[0] == 1:
                    speaker_arrays.append(spk.squeeze(0))
                else:
                    raise ValueError(f"speaker_embeddings[{i}] has invalid shape {spk.shape}")
            speaker_embedding = mx.stack(speaker_arrays)  # [batch, 192]
        else:
            speaker_embedding = speaker_embeddings
            if speaker_embedding.ndim == 1:
                # Single speaker for all - broadcast
                speaker_embedding = mx.broadcast_to(
                    speaker_embedding[None, :], (batch_size, speaker_embedding.shape[0]),
                )
            elif speaker_embedding.shape[0] == 1 and batch_size > 1:
                # Single speaker, broadcast to batch
                speaker_embedding = mx.broadcast_to(
                    speaker_embedding, (batch_size, speaker_embedding.shape[1]),
                )
            elif speaker_embedding.shape[0] != batch_size:
                raise ValueError(
                    f"speaker_embedding batch dim {speaker_embedding.shape[0]} "
                    f"must match input batch size {batch_size}",
                )

        # ====================================================================
        # Step 3: Generate speech tokens (batched LLM)
        # ====================================================================

        speech_tokens, token_lengths = self.llm.generate_speech_tokens_batch(
            text_ids,
            max_length=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        mx.eval(speech_tokens, token_lengths)

        # ====================================================================
        # Step 4: Convert tokens to mel (batched Flow)
        # ====================================================================

        # Compute max mel length from max token length
        max_token_len = speech_tokens.shape[1]
        max_mel_length = max_token_len * self.config.token_mel_ratio

        mel = self.flow.generate(
            speech_tokens,
            speaker_embedding,
            mel_length=max_mel_length,
            num_steps=num_flow_steps,
        )

        mx.eval(mel)

        # ====================================================================
        # Step 5: Convert mel to audio (batched Vocoder)
        # ====================================================================

        audio = self.vocoder(mel)

        mx.eval(audio)

        # ====================================================================
        # Step 6: Compute actual audio lengths from token lengths
        # ====================================================================

        # Audio samples per token (token_mel_ratio * samples_per_mel_frame)
        # mel at 50Hz (token_mel_ratio=2 per 25Hz tokens), vocoder upsamples to 24kHz
        # => 24000 / 50 = 480 samples per mel frame
        # => samples_per_token = token_mel_ratio * 480 = 2 * 480 = 960
        samples_per_token = self.config.token_mel_ratio * (
            self.config.sample_rate // (self.config.token_frame_rate * self.config.token_mel_ratio)
        )
        audio_lengths = token_lengths * samples_per_token

        # Clip audio to valid samples and validate
        audio = mx.clip(audio, -1.0, 1.0)

        return audio, audio_lengths

    def synthesize_batch_text(
        self,
        texts: list[str],
        speaker_embeddings: mx.array | list[mx.array] | None = None,
        max_tokens: int = 1000,
        temperature: float = 1.0,
        top_k: int = 25,
        top_p: float = 0.8,
        num_flow_steps: int = 10,
    ) -> tuple[mx.array, mx.array]:
        """
        Batch text-to-speech synthesis from text strings.

        Convenience method that handles tokenization internally.
        Requires the tokenizer to be loaded via from_pretrained().

        Args:
            texts: List of input text strings
            speaker_embeddings: Either:
                - [batch, 192] array - same speaker for all utterances
                - List of [192] arrays - per-utterance speaker embeddings
                - None: uses random speaker embedding for all
            max_tokens: Maximum speech tokens to generate per utterance
            temperature: LLM sampling temperature
            top_k: LLM top-k sampling
            top_p: LLM nucleus sampling
            num_flow_steps: Flow ODE integration steps

        Returns:
            Tuple of:
                - audio: [batch, max_samples] - Generated audio (padded)
                - audio_lengths: [batch] - Actual audio length per utterance

        Example:
            >>> texts = ["Hello world", "Good morning", "How are you?"]
            >>> audio, lengths = model.synthesize_batch_text(texts)
            >>> for i in range(len(texts)):
            ...     single_audio = audio[i, :lengths[i]]
        """
        if self.tokenizer is None:
            raise RuntimeError(
                "Tokenizer not loaded. Use CosyVoice2Model.from_pretrained() "
                "to load model with tokenizer.",
            )

        batch_size = len(texts)

        # Tokenize all texts
        text_ids_list = [self.tokenizer.encode(text) for text in texts]

        # Use random speaker embedding if not provided
        if speaker_embeddings is None:
            speaker = self.tokenizer.random_speaker_embedding()
            speaker_embeddings = mx.broadcast_to(
                speaker[None, :], (batch_size, speaker.shape[0]),
            )

        return self.synthesize_batch(
            text_ids_list,
            speaker_embeddings,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_flow_steps=num_flow_steps,
        )

    def synthesize_stream(
        self,
        text_ids: mx.array,
        speaker_embedding: mx.array,
        max_tokens: int = 1000,
        temperature: float = 1.0,
        top_k: int = 25,
        top_p: float = 0.8,
        num_flow_steps: int = 10,
        chunk_size: int = 25,
    ) -> Generator[tuple[mx.array, bool], None, None]:
        """
        Streaming text-to-speech synthesis.

        Processes tokens in chunks for low-latency streaming output.

        Args:
            text_ids: [batch, seq_len] - Tokenized text input
            speaker_embedding: [batch, 192] - Speaker embedding
            max_tokens: Maximum speech tokens to generate
            temperature: LLM sampling temperature
            top_k: LLM top-k sampling
            top_p: LLM nucleus sampling
            num_flow_steps: Flow ODE integration steps
            chunk_size: Number of tokens per chunk

        Yields:
            (audio_chunk, is_final): Audio chunk and completion flag
        """
        for token_chunk, is_final in self.generate_speech_tokens_stream(
            text_ids,
            max_length=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            chunk_size=chunk_size,
        ):
            # Convert chunk to mel
            mel_chunk = self.tokens_to_mel(
                token_chunk,
                speaker_embedding,
                num_steps=num_flow_steps,
            )

            # Convert mel to audio
            audio_chunk = self.mel_to_audio(mel_chunk)

            yield audio_chunk, is_final

    def synthesize_text(
        self,
        text: str,
        speaker_embedding: mx.array | None = None,
        max_tokens: int = 1000,
        top_k: int = 25,
        top_p: float = 0.8,
        win_size: int = 10,
        tau_r: float = 0.1,
        speech_token_size: int = 6561,
        min_token_text_ratio: float = 2.0,
        max_token_text_ratio: float = 20.0,
        num_flow_steps: int = 10,
    ) -> mx.array:
        """
        Synthesize speech from text string using Repetition Aware Sampling.

        This is the recommended method for text-to-speech synthesis in CosyVoice2.
        Uses ras_sampling to prevent repetitive/looping output.

        Termination:
        - Stop tokens are [6561, 6562, 6563] (any token >= speech_token_size)
        - Min/max length is computed from text length * ratio

        Requires the tokenizer to be loaded via from_pretrained().

        Args:
            text: Input text string
            speaker_embedding: [192] Speaker embedding. If None, uses random embedding.
            max_tokens: Maximum speech tokens to generate
            top_k: LLM top-k sampling (default 25)
            top_p: LLM nucleus sampling (default 0.8)
            win_size: Repetition window size (default 10)
            tau_r: Repetition threshold ratio (default 0.1)
            speech_token_size: Size of speech vocabulary (default 6561)
            min_token_text_ratio: Min tokens per text token (default 2.0)
            max_token_text_ratio: Max tokens per text token (default 20.0)
            num_flow_steps: Flow ODE integration steps

        Returns:
            audio: [samples] - Generated audio waveform
        """
        if self.tokenizer is None:
            raise RuntimeError(
                "Tokenizer not loaded. Use CosyVoice2Model.from_pretrained() "
                "to load model with tokenizer.",
            )

        # Tokenize text
        text_ids = self.tokenizer.encode(text)
        # Add batch dimension
        text_ids = text_ids[None, :]  # [1, seq_len]

        # Use random speaker embedding if not provided
        if speaker_embedding is None:
            speaker_embedding = self.tokenizer.random_speaker_embedding()
        # Add batch dimension
        speaker_embedding = speaker_embedding[None, :]  # [1, 192]

        # Synthesize using ras_sampling (prevents repetitive loops)
        audio = self.synthesize_ras(
            text_ids,
            speaker_embedding,
            max_tokens=max_tokens,
            top_k=top_k,
            top_p=top_p,
            win_size=win_size,
            tau_r=tau_r,
            speech_token_size=speech_token_size,
            min_token_text_ratio=min_token_text_ratio,
            max_token_text_ratio=max_token_text_ratio,
            num_flow_steps=num_flow_steps,
        )

        # Remove batch dimension
        return audio[0]

    def synthesize_text_stream(
        self,
        text: str,
        speaker_embedding: mx.array | None = None,
        max_tokens: int = 1000,
        temperature: float = 1.0,
        top_k: int = 25,
        top_p: float = 0.8,
        num_flow_steps: int = 10,
        chunk_size: int = 25,
    ) -> Generator[tuple[mx.array, bool], None, None]:
        """
        Streaming text-to-speech synthesis.

        Args:
            text: Input text string
            speaker_embedding: [192] Speaker embedding. If None, uses random embedding.
            max_tokens: Maximum speech tokens to generate
            temperature: LLM sampling temperature
            top_k: LLM top-k sampling
            top_p: LLM nucleus sampling
            num_flow_steps: Flow ODE integration steps
            chunk_size: Number of tokens per chunk

        Yields:
            (audio_chunk, is_final): Audio chunk and completion flag
        """
        if self.tokenizer is None:
            raise RuntimeError(
                "Tokenizer not loaded. Use CosyVoice2Model.from_pretrained() "
                "to load model with tokenizer.",
            )

        # Tokenize text
        text_ids = self.tokenizer.encode(text)
        text_ids = text_ids[None, :]  # [1, seq_len]

        # Use random speaker embedding if not provided
        if speaker_embedding is None:
            speaker_embedding = self.tokenizer.random_speaker_embedding()
        speaker_embedding = speaker_embedding[None, :]  # [1, 192]

        # Stream synthesis
        for audio_chunk, is_final in self.synthesize_stream(
            text_ids,
            speaker_embedding,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_flow_steps=num_flow_steps,
            chunk_size=chunk_size,
        ):
            yield audio_chunk[0], is_final

    @staticmethod
    def from_pretrained(
        model_path: str | Path,
        config: CosyVoice2Config | None = None,
        device: str | None = None,
    ) -> "CosyVoice2Model":
        """
        Load CosyVoice2 model from pretrained weights.

        Args:
            model_path: Path to model directory containing llm.pt, flow.pt, hift.pt
            config: Optional config override
            device: Optional device (unused in MLX)

        Returns:
            Loaded CosyVoice2Model
        """
        model_path = Path(model_path)

        if config is None:
            config = CosyVoice2Config()

        model = CosyVoice2Model(config)

        # Load LLM weights
        llm_path = model_path / "llm.pt"
        if llm_path.exists():
            model.llm = CosyVoice2LLM.from_pretrained(str(llm_path), config.llm_config)

        # Load Flow weights
        flow_path = model_path / "flow.pt"
        if flow_path.exists():
            model.flow = MaskedDiffWithXvec.from_pretrained(
                str(flow_path), config.flow_config,
            )

        # Load Vocoder weights
        vocoder_path = model_path / "hift.pt"
        if vocoder_path.exists():
            model.vocoder = HiFiGANVocoder.from_pretrained(
                str(vocoder_path), config.vocoder_config,
            )

        # Load tokenizer
        model.tokenizer = CosyVoice2Tokenizer.from_pretrained(model_path)

        return model

    @staticmethod
    def get_default_model_path() -> Path:
        """Get default cache path for CosyVoice2 model."""
        return Path.home() / ".cache" / "cosyvoice2" / "cosyvoice2-0.5b"


def count_parameters(model: nn.Module) -> int:
    """Count total parameters in a model."""
    total = 0

    def count_params(params):
        nonlocal total
        if isinstance(params, dict):
            for v in params.values():
                count_params(v)
        elif isinstance(params, mx.array):
            total += params.size

    count_params(model.parameters())
    return total
