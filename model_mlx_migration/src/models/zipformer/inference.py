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
End-to-end ASR inference pipeline for Zipformer.

Audio file → Features → Encoder → Decoder → Text
"""

from collections.abc import Generator
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

from .asr_model import ASRModel, ASRModelConfig, load_checkpoint
from .decoding import (
    DecodingResult,
    StreamingDecoderState,
    greedy_search,
    greedy_search_streaming,
    modified_beam_search,
)
from .features import FbankConfig, FbankExtractor, load_audio

try:
    import sentencepiece as spm
    HAS_SENTENCEPIECE = True
except ImportError:
    HAS_SENTENCEPIECE = False


@dataclass
class ASRConfig:
    """Configuration for ASR inference."""
    # Checkpoint paths
    checkpoint_path: str
    bpe_model_path: str

    # Feature extraction
    sample_rate: int = 16000
    num_mel_bins: int = 80

    # Decoding
    decoding_method: str = "greedy"  # "greedy" or "beam"
    beam_size: int = 4
    max_sym_per_frame: int = 1

    # Streaming parameters
    # Number of fbank frames per chunk (input to encoder)
    # 45 frames produces ~16 encoder output frames after all downsampling
    fbank_chunk_size: int = 45
    # Left context length for attention (frames at base resolution)
    left_context_len: int = 128


@dataclass
class StreamingState:
    """State for streaming ASR inference."""
    encoder_states: list[mx.array]
    decoder_state: StreamingDecoderState | None = None
    accumulated_tokens: list[int] = field(default_factory=list)
    accumulated_score: float = 0.0
    # Number of fbank frames consumed from the source audio (not including padding).
    processed_fbank_frames: int = 0
    # Number of encoder "base resolution" frames processed for cache masking.
    # This is the time axis after Conv2dSubsampling (encoder_embed).
    processed_encoder_frames: int = 0


class ASRPipeline:
    """
    End-to-end ASR inference pipeline.

    Usage:
        pipeline = ASRPipeline.from_pretrained(
            checkpoint_path="checkpoints/zipformer/en-streaming/exp/pretrained.pt",
            bpe_model_path="checkpoints/zipformer/en-streaming/data/lang_bpe_500/bpe.model"
        )
        text = pipeline.transcribe("audio.wav")
    """

    def __init__(
        self,
        model: ASRModel,
        sp: "spm.SentencePieceProcessor",
        fbank_extractor: FbankExtractor,
        config: ASRConfig,
    ):
        """
        Initialize pipeline with pre-loaded components.

        Use from_pretrained() for standard initialization.
        """
        self.model = model
        self.sp = sp
        self.fbank_extractor = fbank_extractor
        self.config = config

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        bpe_model_path: str,
        decoding_method: str = "greedy",
        beam_size: int = 4,
    ) -> "ASRPipeline":
        """
        Load ASR pipeline from pretrained checkpoint.

        Args:
            checkpoint_path: Path to PyTorch checkpoint (.pt file).
            bpe_model_path: Path to sentencepiece model (.model file).
            decoding_method: "greedy" or "beam".
            beam_size: Beam size for beam search.

        Returns:
            Initialized ASR pipeline.
        """
        if not HAS_SENTENCEPIECE:
            raise ImportError(
                "sentencepiece is required for ASR. "
                "Install with: pip install sentencepiece",
            )

        # Load sentencepiece model
        sp = spm.SentencePieceProcessor()
        sp.Load(bpe_model_path)

        # Create model config
        model_config = ASRModelConfig(
            vocab_size=sp.GetPieceSize(),
            decoder_dim=512,
            context_size=2,
            joiner_dim=512,
            blank_id=0,
        )

        # Load model
        model, _ = load_checkpoint(checkpoint_path, model_config)
        mx.eval(model.parameters())

        # Create feature extractor
        fbank_config = FbankConfig(
            sample_rate=16000,
            num_mel_bins=80,
        )
        fbank_extractor = FbankExtractor(fbank_config)

        # Create pipeline config
        config = ASRConfig(
            checkpoint_path=checkpoint_path,
            bpe_model_path=bpe_model_path,
            decoding_method=decoding_method,
            beam_size=beam_size,
        )

        return cls(model, sp, fbank_extractor, config)

    def transcribe(
        self,
        audio: str | np.ndarray | mx.array,
        sample_rate: int | None = None,
    ) -> str:
        """
        Transcribe audio to text.

        Args:
            audio: Path to audio file, or waveform array.
            sample_rate: Sample rate if audio is array.

        Returns:
            Transcribed text.
        """
        # Load audio if path
        if isinstance(audio, str):
            waveform, sr = load_audio(audio, self.config.sample_rate)
        else:
            waveform = audio
            sr = sample_rate or self.config.sample_rate

        # Extract features
        features = self.fbank_extractor.extract(waveform, sr)

        # Add batch dimension: (1, T, mel_dim)
        features = mx.expand_dims(features, axis=0)
        feature_lens = mx.array([features.shape[1]], dtype=mx.int32)

        # Run encoder
        encoder_out, encoder_out_lens = self.model(features, feature_lens)
        mx.eval(encoder_out, encoder_out_lens)

        # Decode
        encoder_out_single = encoder_out[0]  # Remove batch dim
        encoder_out_len = int(encoder_out_lens[0].item())

        if self.config.decoding_method == "beam":
            result = modified_beam_search(
                decoder=self.model.decoder,
                joiner=self.model.joiner,
                encoder_out=encoder_out_single,
                encoder_out_len=encoder_out_len,
                beam_size=self.config.beam_size,
                max_sym_per_frame=self.config.max_sym_per_frame,
            )
        else:
            result = greedy_search(
                decoder=self.model.decoder,
                joiner=self.model.joiner,
                encoder_out=encoder_out_single,
                encoder_out_len=encoder_out_len,
                max_sym_per_frame=self.config.max_sym_per_frame,
            )

        # Decode tokens to text
        text = self.sp.DecodeIds(result.tokens)
        return text

    def transcribe_batch(
        self,
        audios: list[str | np.ndarray],
        sample_rates: list[int] | None = None,
    ) -> list[str]:
        """
        Transcribe a batch of audio files.

        Args:
            audios: List of audio paths or waveform arrays.
            sample_rates: Optional sample rates for arrays.

        Returns:
            List of transcribed texts.
        """
        if sample_rates is None:
            sample_rates = [None] * len(audios)

        # Load and extract features for each audio
        waveforms = []
        for audio, sr in zip(audios, sample_rates, strict=False):
            if isinstance(audio, str):
                wav, _ = load_audio(audio, self.config.sample_rate)
            else:
                wav = audio
            waveforms.append(wav)

        # Extract features (using batch extraction)
        features, feature_lens = self.fbank_extractor.extract_batch(waveforms)

        # Run encoder
        encoder_out, encoder_out_lens = self.model(features, feature_lens)
        mx.eval(encoder_out, encoder_out_lens)

        # Decode each utterance
        batch_size = encoder_out.shape[0]
        texts = []

        for i in range(batch_size):
            encoder_out_single = encoder_out[i]
            encoder_out_len = int(encoder_out_lens[i].item())

            if self.config.decoding_method == "beam":
                result = modified_beam_search(
                    decoder=self.model.decoder,
                    joiner=self.model.joiner,
                    encoder_out=encoder_out_single,
                    encoder_out_len=encoder_out_len,
                    beam_size=self.config.beam_size,
                    max_sym_per_frame=self.config.max_sym_per_frame,
                )
            else:
                result = greedy_search(
                    decoder=self.model.decoder,
                    joiner=self.model.joiner,
                    encoder_out=encoder_out_single,
                    encoder_out_len=encoder_out_len,
                    max_sym_per_frame=self.config.max_sym_per_frame,
                )

            text = self.sp.DecodeIds(result.tokens)
            texts.append(text)

        return texts

    def transcribe_with_details(
        self,
        audio: str | np.ndarray | mx.array,
        sample_rate: int | None = None,
    ) -> tuple[str, DecodingResult, dict]:
        """
        Transcribe with additional details.

        Args:
            audio: Path to audio file, or waveform array.
            sample_rate: Sample rate if audio is array.

        Returns:
            Tuple of (text, decoding_result, info_dict).
        """
        # Load audio if path
        if isinstance(audio, str):
            waveform, sr = load_audio(audio, self.config.sample_rate)
        else:
            waveform = audio
            sr = sample_rate or self.config.sample_rate

        # Extract features
        features = self.fbank_extractor.extract(waveform, sr)

        # Add batch dimension
        features = mx.expand_dims(features, axis=0)
        feature_lens = mx.array([features.shape[1]], dtype=mx.int32)

        # Run encoder
        encoder_out, encoder_out_lens = self.model(features, feature_lens)
        mx.eval(encoder_out, encoder_out_lens)

        # Decode
        encoder_out_single = encoder_out[0]
        encoder_out_len = int(encoder_out_lens[0].item())

        if self.config.decoding_method == "beam":
            result = modified_beam_search(
                decoder=self.model.decoder,
                joiner=self.model.joiner,
                encoder_out=encoder_out_single,
                encoder_out_len=encoder_out_len,
                beam_size=self.config.beam_size,
                max_sym_per_frame=self.config.max_sym_per_frame,
            )
        else:
            result = greedy_search(
                decoder=self.model.decoder,
                joiner=self.model.joiner,
                encoder_out=encoder_out_single,
                encoder_out_len=encoder_out_len,
                max_sym_per_frame=self.config.max_sym_per_frame,
            )

        # Decode tokens to text
        text = self.sp.DecodeIds(result.tokens)

        # Build info dict
        info = {
            "num_frames": features.shape[1],
            "num_encoder_frames": encoder_out_len,
            "num_tokens": len(result.tokens),
            "decoding_method": self.config.decoding_method,
            "score": result.score,
            "token_ids": result.tokens,
        }

        return text, result, info

    def init_streaming(self, batch_size: int = 1) -> StreamingState:
        """
        Initialize streaming state for chunk-by-chunk transcription.

        Args:
            batch_size: Batch size for streaming (default 1).

        Returns:
            Initial StreamingState for streaming transcription.
        """
        # Initialize encoder states
        encoder_states = self.model.encoder.init_states(
            batch_size=batch_size,
            left_context_len=self.config.left_context_len,
        )

        return StreamingState(
            encoder_states=encoder_states,
            decoder_state=None,
            accumulated_tokens=[],
            accumulated_score=0.0,
            processed_fbank_frames=0,
            processed_encoder_frames=0,
        )

    def transcribe_chunk(
        self,
        features: mx.array,
        state: StreamingState,
        *,
        valid_fbank_frames: int | None = None,
    ) -> tuple[str, StreamingState]:
        """
        Transcribe a chunk of features in streaming mode.

        Args:
            features: Feature chunk of shape (batch, chunk_frames, mel_dim).
                      Typically 45 fbank frames per chunk.
            state: Current streaming state from init_streaming() or previous chunk.
            valid_fbank_frames: If features were padded to a fixed chunk size, this
                is the number of real (unpadded) fbank frames in this chunk. When
                provided, decoding and cache masking use only the valid portion.

        Returns:
            Tuple of (partial_text, updated_state).
            partial_text contains only the new tokens from this chunk.
        """
        # Ensure batch dimension
        if features.ndim == 2:
            features = mx.expand_dims(features, axis=0)

        chunk_fbank_frames = int(features.shape[1])
        if valid_fbank_frames is None:
            valid_fbank_frames = chunk_fbank_frames
        if valid_fbank_frames < 0 or valid_fbank_frames > chunk_fbank_frames:
            raise ValueError(
                f"valid_fbank_frames must be in [0, {chunk_fbank_frames}], got {valid_fbank_frames}",
            )

        # Conv2dSubsampling.streaming_forward() outputs ~floor(chunk_fbank_frames/2) frames.
        # Cache masking expects "base resolution" frames (post-subsampling).
        chunk_encoder_frames = valid_fbank_frames // 2

        # Run streaming encoder
        encoder_out, new_encoder_states = self.model.encoder.streaming_forward(
            features,
            state.encoder_states,
            left_context_len=self.config.left_context_len,
            processed_frames=state.processed_encoder_frames,
        )
        mx.eval(encoder_out)

        # If we padded features, drop any encoder frames attributable to padding so the
        # decoder doesn't consume artificial silence frames.
        # The Zipformer encoder preserves the Conv2dSubsampling time axis and then applies
        # a final ceil(/2) downsampling before returning encoder_out.
        expected_encoder_out_frames = (chunk_encoder_frames + 1) // 2
        if expected_encoder_out_frames < encoder_out.shape[1]:
            encoder_out = encoder_out[:, :expected_encoder_out_frames, :]

        # Run streaming decoder
        chunk_result, new_decoder_state = greedy_search_streaming(
            decoder=self.model.decoder,
            joiner=self.model.joiner,
            encoder_out=encoder_out,
            state=state.decoder_state,
            max_sym_per_frame=self.config.max_sym_per_frame,
        )

        # Decode new tokens to text
        partial_text = self.sp.DecodeIds(chunk_result.tokens) if chunk_result.tokens else ""

        # Update state
        new_state = StreamingState(
            encoder_states=new_encoder_states,
            decoder_state=new_decoder_state,
            accumulated_tokens=state.accumulated_tokens + chunk_result.tokens,
            accumulated_score=state.accumulated_score + chunk_result.score,
            processed_fbank_frames=state.processed_fbank_frames + valid_fbank_frames,
            processed_encoder_frames=state.processed_encoder_frames + chunk_encoder_frames,
        )

        return partial_text, new_state

    def transcribe_streaming(
        self,
        audio: str | np.ndarray | mx.array,
        sample_rate: int | None = None,
        use_lookahead: bool = True,
        exact_match: bool = True,
    ) -> Generator[tuple[str, str], None, None]:
        """
        Streaming transcription that yields partial results.

        Processes audio in chunks and yields (partial_text, full_text) tuples
        as transcription progresses.

        When exact_match=True, uses non-streaming encoder_embed to ensure output
        matches non-streaming transcription exactly. This is suitable for offline
        batch processing where all audio is available upfront.

        Args:
            audio: Path to audio file, or waveform array.
            sample_rate: Sample rate if audio is array.
            use_lookahead: If True and exact_match=False, pre-compute conv stack
                          outputs and provide right context. Ignored if exact_match=True.
            exact_match: If True, use non-streaming encoder_embed for exact frame match.
                        This gives identical output to non-streaming transcription.
                        Default True.

        Yields:
            Tuples of (partial_text, full_text) where:
            - partial_text: New text from this chunk
            - full_text: Complete transcription so far
        """
        # Load audio if path
        if isinstance(audio, str):
            waveform, sr = load_audio(audio, self.config.sample_rate)
        else:
            waveform = audio
            sr = sample_rate or self.config.sample_rate

        # Convert to numpy if needed
        if isinstance(waveform, mx.array):
            waveform = np.array(waveform)

        # Extract full features
        full_features = self.fbank_extractor.extract(waveform, sr)
        total_frames = full_features.shape[0]

        if exact_match:
            # Use non-streaming encoder for exact frame match, then greedy decode
            # This gives identical output to non-streaming transcription
            features_batch = mx.expand_dims(full_features, axis=0)
            encoder_out, _ = self.model(features_batch, mx.array([total_frames], dtype=mx.int32))
            mx.eval(encoder_out)

            # Run greedy decoding in chunks to yield partial results
            decoder_state = None  # Will be initialized by greedy_search_streaming
            accumulated_tokens = []

            # Process encoder output in chunks (arbitrary chunk size for yielding)
            enc_chunk_size = 16  # Encoder frames per yield
            num_enc_frames = encoder_out.shape[1]

            for start_enc in range(0, num_enc_frames, enc_chunk_size):
                end_enc = min(start_enc + enc_chunk_size, num_enc_frames)
                enc_chunk = encoder_out[:, start_enc:end_enc, :]

                chunk_result, decoder_state = greedy_search_streaming(
                    decoder=self.model.decoder,
                    joiner=self.model.joiner,
                    encoder_out=enc_chunk,
                    state=decoder_state,
                    max_sym_per_frame=self.config.max_sym_per_frame,
                )

                accumulated_tokens.extend(chunk_result.tokens)
                partial_text = self.sp.DecodeIds(chunk_result.tokens) if chunk_result.tokens else ""
                full_text = self.sp.DecodeIds(accumulated_tokens)

                yield partial_text, full_text
        else:
            # True streaming mode with chunked encoder
            state = self.init_streaming(batch_size=1)
            chunk_size = self.config.fbank_chunk_size
            CONVNEXT_CONTEXT = 3

            # Collect all chunk boundaries
            chunk_boundaries = []
            start = 0
            while start < total_frames:
                end = min(start + chunk_size, total_frames)
                chunk_boundaries.append((start, end))
                start = end

            # If using lookahead, pre-compute conv stack outputs for all chunks
            conv_stack_outputs = []
            if use_lookahead:
                fbank_cache = mx.zeros((1, 7, full_features.shape[1]), dtype=full_features.dtype)

                for start_frame, end_frame in chunk_boundaries:
                    chunk_features = full_features[start_frame:end_frame]

                    if chunk_features.shape[0] < chunk_size:
                        padding = mx.zeros(
                            (chunk_size - chunk_features.shape[0], chunk_features.shape[1]),
                            dtype=chunk_features.dtype,
                        )
                        chunk_features = mx.concatenate([chunk_features, padding], axis=0)

                    chunk_features = mx.expand_dims(chunk_features, axis=0)
                    conv_out, fbank_cache = self.model.encoder.encoder_embed.get_conv_stack_output_with_cache(
                        chunk_features, fbank_cache,
                    )
                    mx.eval(conv_out, fbank_cache)
                    conv_stack_outputs.append(conv_out)

            # Process each chunk
            for i, (start_frame, end_frame) in enumerate(chunk_boundaries):
                chunk_features = full_features[start_frame:end_frame]
                valid_frames = int(chunk_features.shape[0])

                if chunk_features.shape[0] < chunk_size:
                    padding = mx.zeros(
                        (chunk_size - chunk_features.shape[0], chunk_features.shape[1]),
                        dtype=chunk_features.dtype,
                    )
                    chunk_features = mx.concatenate([chunk_features, padding], axis=0)

                chunk_features = mx.expand_dims(chunk_features, axis=0)

                right_context = None
                if use_lookahead and i + 1 < len(conv_stack_outputs):
                    next_conv_out = conv_stack_outputs[i + 1]
                    right_context = next_conv_out[:, :CONVNEXT_CONTEXT, :, :]

                partial_text, state = self.transcribe_chunk_with_context(
                    chunk_features,
                    state,
                    valid_fbank_frames=valid_frames,
                    right_context=right_context,
                )

                full_text = self.sp.DecodeIds(state.accumulated_tokens)
                yield partial_text, full_text

    def transcribe_chunk_with_context(
        self,
        features: mx.array,
        state: StreamingState,
        *,
        valid_fbank_frames: int | None = None,
        right_context: mx.array | None = None,
    ) -> tuple[str, StreamingState]:
        """
        Transcribe a chunk of features in streaming mode with optional right context.

        Similar to transcribe_chunk but supports explicit right context for improved
        accuracy matching non-streaming output.

        Args:
            features: Feature chunk of shape (batch, chunk_frames, mel_dim).
            state: Current streaming state from init_streaming() or previous chunk.
            valid_fbank_frames: If features were padded, the number of real frames.
            right_context: Optional right context from next chunk's conv stack output.
                          Shape (batch, 3, out_width, 128) in NHWC format.

        Returns:
            Tuple of (partial_text, updated_state).
        """
        # Ensure batch dimension
        if features.ndim == 2:
            features = mx.expand_dims(features, axis=0)

        chunk_fbank_frames = int(features.shape[1])
        if valid_fbank_frames is None:
            valid_fbank_frames = chunk_fbank_frames
        if valid_fbank_frames < 0 or valid_fbank_frames > chunk_fbank_frames:
            raise ValueError(
                f"valid_fbank_frames must be in [0, {chunk_fbank_frames}], got {valid_fbank_frames}",
            )

        # With fbank caching: (7 cache + fbank_frames - 7) // 2 = fbank_frames // 2 conv-stack frames
        # For chunks after first: effectively (prev_cache + current - kernel) // stride
        # Using simplified calculation that matches the actual output
        chunk_encoder_frames = (valid_fbank_frames + 7 - 7) // 2  # Simplified: valid_fbank_frames // 2
        # But first chunk produces fewer due to zero cache: (7 zeros + frames - 7) // 2 = frames // 2
        # This is approximate; actual frames determined by encoder output

        # Run streaming encoder with right context
        encoder_out, new_encoder_states = self.model.encoder.streaming_forward(
            features,
            state.encoder_states,
            left_context_len=self.config.left_context_len,
            processed_frames=state.processed_encoder_frames,
            right_context=right_context,
        )
        mx.eval(encoder_out)

        # Compute expected output frames after final downsampling
        # encoder_embed outputs chunk_encoder_frames, then final /2 downsampling
        expected_encoder_out_frames = (chunk_encoder_frames + 1) // 2
        if expected_encoder_out_frames < encoder_out.shape[1]:
            encoder_out = encoder_out[:, :expected_encoder_out_frames, :]

        # Run streaming decoder
        chunk_result, new_decoder_state = greedy_search_streaming(
            decoder=self.model.decoder,
            joiner=self.model.joiner,
            encoder_out=encoder_out,
            state=state.decoder_state,
            max_sym_per_frame=self.config.max_sym_per_frame,
        )

        # Decode new tokens to text
        partial_text = self.sp.DecodeIds(chunk_result.tokens) if chunk_result.tokens else ""

        # Update state
        new_state = StreamingState(
            encoder_states=new_encoder_states,
            decoder_state=new_decoder_state,
            accumulated_tokens=state.accumulated_tokens + chunk_result.tokens,
            accumulated_score=state.accumulated_score + chunk_result.score,
            processed_fbank_frames=state.processed_fbank_frames + valid_fbank_frames,
            processed_encoder_frames=state.processed_encoder_frames + chunk_encoder_frames,
        )

        return partial_text, new_state

    def transcribe_streaming_complete(
        self,
        audio: str | np.ndarray | mx.array,
        sample_rate: int | None = None,
    ) -> tuple[str, dict]:
        """
        Streaming transcription that returns final result.

        Uses streaming encoder for chunk-by-chunk processing but returns
        only the final transcription result.

        Args:
            audio: Path to audio file, or waveform array.
            sample_rate: Sample rate if audio is array.

        Returns:
            Tuple of (text, info_dict).
        """
        # Process all chunks
        final_text = ""
        for partial_text, full_text in self.transcribe_streaming(audio, sample_rate):
            final_text = full_text

        # Build info
        info = {
            "decoding_method": "streaming_greedy",
        }

        return final_text, info


def transcribe_file(
    audio_path: str,
    checkpoint_path: str,
    bpe_model_path: str,
    decoding_method: str = "greedy",
    beam_size: int = 4,
) -> str:
    """
    Convenience function to transcribe a single file.

    Args:
        audio_path: Path to audio file.
        checkpoint_path: Path to model checkpoint.
        bpe_model_path: Path to BPE model.
        decoding_method: "greedy" or "beam".
        beam_size: Beam size for beam search.

    Returns:
        Transcribed text.
    """
    pipeline = ASRPipeline.from_pretrained(
        checkpoint_path=checkpoint_path,
        bpe_model_path=bpe_model_path,
        decoding_method=decoding_method,
        beam_size=beam_size,
    )
    return pipeline.transcribe(audio_path)
