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
WhisperMLX - Custom MLX Whisper implementation.

Main model class that combines:
- Variable-length audio encoder
- Decoder with adjustable timestamp precision
- Optional preallocated KV-cache
- Support for speculative decoding (future)

Target: 8-10x speedup over mlx-whisper with 100% lossless quality.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy

    from .presets import OptimizationPreset, TranscriptionConfig

from collections.abc import Callable
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .audio import (
    SAMPLE_RATE,
    AsyncMelPreparer,
    compute_mel_for_duration,
    load_audio,
    log_mel_spectrogram,
)
from .config import WhisperConfig, get_config
from .decoder import TextDecoder
from .decoding import (
    DecodingOptions,
    apply_filters,
    build_logit_filters,
    compute_logprobs,
    greedy_decode,
    sample_with_temperature,
)
from .encoder import AudioEncoder
from .encoder_cache import EncoderCache
from .encoder_vad import EncoderVADHead, load_encoder_vad_head
from .kv_cache import KVCacheManager
from .medusa import MedusaModule, MedusaTreeVerifier, create_medusa_module
from .silero_vad import VADResult, preprocess_audio_with_vad
from .speculative import SpeculativeDecoder
from .utils import download_model

# Audio cleaning integration (Phase 9)
_AUDIO_CLEANING_AVAILABLE = False
_adaptive_router = None


def _get_audio_router():
    """Lazy-load audio cleaning router."""
    global _AUDIO_CLEANING_AVAILABLE, _adaptive_router

    if _adaptive_router is not None:
        return _adaptive_router

    try:
        from tools.audio_cleaning import AdaptiveRouter

        _adaptive_router = AdaptiveRouter(sample_rate=SAMPLE_RATE)
        _adaptive_router.warmup()
        _AUDIO_CLEANING_AVAILABLE = True
        return _adaptive_router
    except ImportError:
        _AUDIO_CLEANING_AVAILABLE = False
        return None


@dataclass
class TranscriptionCallbacks:
    """
    Callback functions for transcription progress and control.

    Modeled after whisper.cpp callback system (GAP 44 - Phase 6 stretch goal).

    Callbacks:
        new_segment: Called when a new transcription segment is complete.
                     Signature: (segment: dict) -> None

        progress: Called periodically to report transcription progress.
                  Signature: (progress_pct: float) -> None
                  progress_pct is 0.0 to 100.0

        encoder_begin: Called when encoder processing starts.
                       Signature: () -> bool
                       Return False to abort processing.

        abort: Called periodically to check if processing should abort.
               Signature: () -> bool
               Return True to abort processing.

        logits_filter: Called before token sampling to filter logits.
                       Signature: (logits: mx.array, tokens: List[int]) -> mx.array
                       Return filtered logits array.

    Example:
        def on_segment(seg):
            print(f"[{seg['start']:.2f}-{seg['end']:.2f}] {seg['text']}")

        def on_progress(pct):
            print(f"Progress: {pct:.1f}%")

        callbacks = TranscriptionCallbacks(
            new_segment=on_segment,
            progress=on_progress,
        )
        result = model.transcribe("audio.wav", callbacks=callbacks)
    """
    new_segment: Callable[[dict], None] | None = None
    progress: Callable[[float], None] | None = None
    encoder_begin: Callable[[], bool] | None = None
    abort: Callable[[], bool] | None = None
    logits_filter: Callable[[mx.array, list[int]], mx.array] | None = None

    def should_abort(self) -> bool:
        """Check if transcription should be aborted."""
        if self.abort is not None:
            return self.abort()
        return False

    def on_encoder_begin(self) -> bool:
        """
        Called when encoder begins. Returns True to continue, False to abort.
        """
        if self.encoder_begin is not None:
            return self.encoder_begin()
        return True

    def on_progress(self, progress_pct: float) -> None:
        """Report progress percentage (0-100)."""
        if self.progress is not None:
            self.progress(progress_pct)

    def on_new_segment(self, segment: dict) -> None:
        """Called when a new segment is complete."""
        if self.new_segment is not None:
            self.new_segment(segment)

    def filter_logits(self, logits: mx.array, tokens: list[int]) -> mx.array:
        """Apply custom logits filtering before sampling."""
        if self.logits_filter is not None:
            return self.logits_filter(logits, tokens)
        return logits


class WhisperMLX(nn.Module):
    """
    Custom MLX Whisper implementation with:

    1. **Variable-length encoding**: Process actual audio length, not padded 30s
    2. **Dynamic timestamp precision**: Fix decoder hallucinations for short audio
    3. **Preallocated KV-cache**: Zero allocation during decoding (optional)
    4. **Fused attention**: Use MLX's optimized attention (optional)

    Usage:
        # Standard usage (like mlx-whisper)
        model = WhisperMLX.from_pretrained("large-v3")
        result = model.transcribe("audio.wav")

        # Variable-length for speedup
        result = model.transcribe("audio.wav", variable_length=True)
    """

    def __init__(
        self,
        config: WhisperConfig,
        dtype: mx.Dtype = mx.float16,
        use_fused: bool = True,
        preallocate_kv: bool = True,  # OPT-NEW-5: 1.2-1.3x decode speedup
        quantize_kv: bool = True,  # OPT-2.3: INT8 KV cache quantization (enabled by default)
        pad_vocab: bool = True,  # OPT-VOCAB: Vocab padding for GPU efficiency
    ):
        """
        Args:
            config: Model configuration
            dtype: Data type for computation
            use_fused: Use fused attention operations
            preallocate_kv: Use preallocated KV-cache (faster but more memory)
            quantize_kv: Use INT8 quantization for cross-attention KV cache (OPT-2.3).
                         Enabled by default. Provides 50% memory reduction with
                         0% WER impact (verified on 120-file audit with 100% exact match).
                         Requires preallocate_kv=True.
            pad_vocab: Pad vocab dimension to 51872 for GPU efficiency (OPT-VOCAB).
                       Enabled by default. Provides 5-10% speedup on M3/M4 chips
                       where tensor core alignment matters.
        """
        super().__init__()

        self.config = config
        self.dtype = dtype
        self.use_fused = use_fused
        self.preallocate_kv = preallocate_kv
        self.quantize_kv = quantize_kv
        self.pad_vocab = pad_vocab

        # Build encoder
        self.encoder = AudioEncoder(
            n_mels=config.n_mels,
            n_ctx=config.n_audio_ctx,
            n_state=config.n_audio_state,
            n_head=config.n_audio_head,
            n_layer=config.n_audio_layer,
            dtype=dtype,
            use_fused=use_fused,
        )

        # Build decoder
        self.decoder = TextDecoder(
            n_vocab=config.n_vocab,
            n_ctx=config.n_text_ctx,
            n_state=config.n_text_state,
            n_head=config.n_text_head,
            n_layer=config.n_text_layer,
            dtype=dtype,
            use_fused=use_fused,
            pad_vocab=pad_vocab,  # OPT-VOCAB: Pass through vocab padding option
        )

        # KV-cache manager
        head_dim = config.n_text_state // config.n_text_head
        self._kv_cache = KVCacheManager(
            n_layers=config.n_text_layer,
            max_seq_len=config.n_text_ctx,
            n_heads=config.n_text_head,
            head_dim=head_dim,
            dtype=dtype,
            preallocate=preallocate_kv,
            quantize=quantize_kv,  # OPT-2.3: INT8 KV cache quantization
        )

        # Alignment heads (for word-level timestamps)
        # Use last half of decoder layers by default
        import numpy as np
        all_heads = np.zeros((config.n_text_layer, config.n_text_head), dtype=bool)
        all_heads[config.n_text_layer // 2:] = True
        self.alignment_heads = mx.array(np.asarray(all_heads.nonzero()).T)

        # Draft model for speculative decoding (loaded on demand)
        self._draft_model: WhisperMLX | None = None
        self._speculative_decoder: SpeculativeDecoder | None = None

        # Encoder cache for repeated queries (OPT-W4)
        self._encoder_cache: EncoderCache | None = None

        # Medusa multi-token prediction (Phase 2.4)
        self._medusa_module: MedusaModule | None = None
        self._medusa_verifier: MedusaTreeVerifier | None = None

        # Encoder VAD head (Phase 3 Optimization)
        # Enables skipping decoder for silent encoder positions
        self._encoder_vad_head: EncoderVADHead | None = None
        # Default threshold 0.15: optimal for LibriSpeech (91% F1 vs Silero)
        # Model outputs lower probabilities than Silero, so 0.1-0.2 works better than 0.5
        self._encoder_vad_threshold: float = 0.15

    def _warmup(self):
        """
        OPT-NEW-31: Run warmup inference to trigger JIT compilation.

        First inference is always slower due to compilation overhead.
        This method runs a dummy forward pass to pre-compile operations,
        making subsequent real inferences faster.
        """
        # Create minimal dummy inputs
        # Encoder expects (batch, n_frames, n_mels) or (n_frames, n_mels)
        # 3000 frames = 30s max audio (standard Whisper input length)
        # Using smaller size (750 frames = ~7.5s) for faster warmup
        dummy_mel = mx.zeros((1, 750, self.config.n_mels), dtype=self.dtype)
        dummy_tokens = mx.array([[50258]], dtype=mx.int32)  # SOT token

        # Run encoder warmup (use variable_length=True for shorter input)
        encoder_output = self.encoder(dummy_mel, variable_length=True)
        mx.eval(encoder_output)

        # Run decoder warmup (single step to compile all ops)
        decoder_output = self.decoder(dummy_tokens, encoder_output)
        mx.eval(decoder_output)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        cache_dir: str | None = None,
        dtype: mx.Dtype = mx.float16,
        use_fused: bool = True,
        preallocate_kv: bool = True,  # OPT-NEW-5: Enable by default
        quantize_kv: bool = True,  # OPT-2.3: INT8 KV cache (enabled by default, 100% exact match verified)
        warmup: bool = True,  # OPT-NEW-31: faster first inference
        quantize: int | None = None,  # OPT-QUANT: bits for quantization (8=INT8)
    ) -> WhisperMLX:
        """
        Load pretrained model from HuggingFace hub.

        Args:
            model_name: Model name (e.g., "large-v3", "whisper-large-v3-mlx")
            cache_dir: Optional cache directory
            dtype: Data type for computation
            use_fused: Use fused attention
            preallocate_kv: Use preallocated KV-cache
            quantize_kv: Use INT8 quantization for cross-attention KV cache (OPT-2.3).
                         Enabled by default. Provides 50% memory reduction with
                         0% WER impact (verified on 120-file audit with 100% exact match).
                         Requires preallocate_kv=True.
            warmup: Run warmup inference to trigger JIT compilation (OPT-NEW-31)
            quantize: Optional quantization bits (8 for INT8, 4 for INT4).
                      INT8 provides ~1.5-2x memory bandwidth reduction with
                      near-lossless accuracy. Set to None to disable.

        Returns:
            WhisperMLX model instance
        """
        # Get config
        # Parse model name to get config key
        name = model_name.split("/")[-1].replace("whisper-", "").replace("-mlx", "")
        config = get_config(name)

        # Download model
        model_path = download_model(model_name, cache_dir)

        # Create model
        model = cls(
            config,
            dtype=dtype,
            use_fused=use_fused,
            preallocate_kv=preallocate_kv,
            quantize_kv=quantize_kv,
        )

        # Load weights - try different formats
        weights_path = model_path / "weights.npz"  # mlx-whisper format
        if not weights_path.exists():
            weights_path = model_path / "weights.safetensors"
        if not weights_path.exists():
            weights_path = model_path / "model.safetensors"

        if weights_path.exists():
            model.load_weights(str(weights_path))
        else:
            raise FileNotFoundError(f"No weights found in {model_path}")

        # OPT-QUANT: Apply weight quantization if requested
        # Do this after loading weights but before warmup
        if quantize is not None:
            model.quantize_model(bits=quantize)

        # OPT-NEW-31: Model warmup for JIT compilation
        # First inference slower; warmup makes subsequent calls faster
        if warmup:
            model._warmup()

        return model

    def load_weights(self, weights_path: str):
        """
        Load weights from npz or safetensors file.

        Args:
            weights_path: Path to .npz or .safetensors file
        """
        from mlx.utils import tree_unflatten

        # Load weights (mx.load handles both npz and safetensors)
        weights = mx.load(weights_path)

        # Handle alignment_heads separately if present
        if "alignment_heads" in weights:
            self.alignment_heads = weights.pop("alignment_heads")

        # OPT-VOCAB: Pad token embedding weights if vocab padding is enabled
        # Pretrained weights have shape (n_vocab, n_state) where n_vocab is 51864-51866
        # Our padded embedding expects (51872, n_state)
        emb_key = "decoder.token_embedding.weight"
        if self.pad_vocab and emb_key in weights:
            loaded_emb = weights[emb_key]
            loaded_vocab_size = loaded_emb.shape[0]
            padded_vocab_size = self.decoder._padded_vocab

            if loaded_vocab_size < padded_vocab_size:
                # Pad with zeros for unused vocab positions
                pad_size = padded_vocab_size - loaded_vocab_size
                pad_zeros = mx.zeros((pad_size, loaded_emb.shape[1]), dtype=loaded_emb.dtype)
                weights[emb_key] = mx.concatenate([loaded_emb, pad_zeros], axis=0)

        # Convert flat dict to nested structure
        # e.g., {"encoder.conv1.weight": ...} -> {"encoder": {"conv1": {"weight": ...}}}
        weights_list = [(k, v.astype(self.dtype)) for k, v in weights.items()]
        nested_weights = tree_unflatten(weights_list)

        # Update model parameters
        self.update(nested_weights)
        mx.eval(self.parameters())

    def quantize_model(
        self,
        group_size: int = 64,
        bits: int = 8,
    ) -> WhisperMLX:
        """
        Quantize model weights for reduced memory bandwidth.

        INT8 quantization (bits=8) provides ~1.5-2x memory bandwidth reduction
        with near-lossless accuracy. The computation remains in the original
        dtype (typically FP16), only the weight storage is quantized.

        This optimization is particularly effective for memory-bound operations
        like large matrix multiplications in the encoder and decoder.

        Note: Embedding layers are excluded from quantization because:
        1. They're lookup tables, not matrix multiplications
        2. Quantizing embeddings can cause significant accuracy loss
        3. The memory savings are minimal compared to Linear layers

        Args:
            group_size: Quantization group size (default 64, must divide weight dims)
            bits: Bits per weight (default 8 for INT8, can use 4 for more compression)

        Returns:
            Self (model is modified in-place)

        Example:
            >>> model = WhisperMLX.from_pretrained("large-v3")
            >>> model.quantize_model(bits=8)  # INT8 quantization
            >>> # Model now uses quantized weights
        """
        from mlx.nn import quantize

        def _quantize_linear_only(path: str, module: nn.Module) -> bool:
            """Only quantize Linear layers, exclude Embedding."""
            return isinstance(module, nn.Linear)

        # Quantize Linear layers only (exclude Embedding)
        quantize(self, group_size=group_size, bits=bits, class_predicate=_quantize_linear_only)

        # Evaluate to materialize quantized weights
        mx.eval(self.parameters())

        return self

    def embed_audio(
        self,
        mel: mx.array,
        variable_length: bool = False,
    ) -> mx.array:
        """
        Encode audio mel spectrogram.

        Args:
            mel: Mel spectrogram, shape (n_mels, n_frames) or (batch, n_mels, n_frames)
            variable_length: Use variable-length encoding (for speedup)

        Returns:
            Encoded audio features, shape (batch, seq_len, n_state)
        """
        return self.encoder(mel, variable_length=variable_length)

    def logits(
        self,
        tokens: mx.array,
        audio_features: mx.array,
        kv_cache: list | None = None,
    ) -> mx.array:
        """
        Get token logits.

        Args:
            tokens: Token IDs, shape (batch, seq_len)
            audio_features: Encoded audio, shape (batch, encoder_len, n_state)
            kv_cache: Optional KV cache

        Returns:
            Logits, shape (batch, seq_len, n_vocab)
        """
        return self.decoder(tokens, audio_features, kv_cache=kv_cache)[0]

    def __call__(
        self,
        mel: mx.array,
        tokens: mx.array,
    ) -> mx.array:
        """
        Forward pass (like mlx-whisper).

        Args:
            mel: Mel spectrogram
            tokens: Token IDs

        Returns:
            Logits
        """
        return self.decoder(tokens, self.encoder(mel))[0]

    def transcribe(
        self,
        audio: str | numpy.ndarray | mx.array,
        *,
        language: str | None = None,
        task: str = "transcribe",
        variable_length: bool = False,
        temperature: float | tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        max_initial_timestamp: float = 1.0,
        word_timestamps: bool = False,
        verbose: bool = False,
        # VAD preprocessing (P0.1 - ALWAYS ON)
        vad_aggressiveness: int = 2,
        # Accuracy thresholds (matching OpenAI Whisper defaults)
        compression_ratio_threshold: float | None = 2.4,
        logprob_threshold: float | None = -1.0,
        no_speech_threshold: float | None = 0.6,
        # Repetition penalty (F5: helps prevent hallucination loops)
        repetition_penalty: float | None = None,
        # F10 optimization: skip logprobs for greedy (can provide ~2x decode speedup)
        skip_logprobs: bool = False,
        # J9: Initial prompt for terminology injection
        initial_prompt: str | None = None,
        # GAP 44: Callback system (Phase 6 stretch goal)
        callbacks: TranscriptionCallbacks | None = None,
        # Phase 9: Audio cleaning (denoising/dereverberation)
        audio_cleaning: bool = False,
    ) -> dict:
        """
        Transcribe audio file or array.

        VAD (Voice Activity Detection) is ALWAYS enabled. Silero VAD is used to
        filter out silence from audio before transcription. This provides 2-4x
        speedup on audio with silence while maintaining 100% accuracy.

        Args:
            audio: Audio file path or waveform array
            language: Language code (auto-detect if None)
            task: "transcribe" or "translate"
            variable_length: DEPRECATED - produces hallucinations, do not use.
                             Use standard mode (variable_length=False) instead.
            temperature: Sampling temperature. Can be a tuple of temperatures
                         for fallback on quality failures (default: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
            max_initial_timestamp: Maximum timestamp at start
            word_timestamps: NOT IMPLEMENTED - SDPA doesn't return attention
                             weights needed for alignment.
            verbose: Print progress
            vad_aggressiveness: VAD aggressiveness level (0-3, default=2).
                                0=most conservative (keeps more audio),
                                3=most aggressive (filters more). VAD is always ON.
            compression_ratio_threshold: If compression ratio > this, treat as failed
                                         and retry with higher temperature. Set None to disable.
            logprob_threshold: If avg log probability < this, treat as failed
                               and retry with higher temperature. Set None to disable.
            no_speech_threshold: If no_speech_prob > this AND logprob < logprob_threshold,
                                 treat segment as silence. Set None to disable.
            repetition_penalty: Penalty for repeated tokens (default: None = disabled).
                               Values > 1.0 discourage repetition (typical: 1.1-1.5).
                               Helps prevent hallucination loops on certain audio.
            skip_logprobs: Skip computing log probabilities during decoding (default: False).
                          When True, provides ~2x decode speedup but disables quality metrics
                          (avg_logprob will be NaN). Temperature fallback won't work.
                          Recommended for low-latency applications where quality checks
                          are not needed.
            initial_prompt: Optional text to provide as prompt context (J9 terminology injection).
                           Useful for domain-specific vocabulary or formatting guidance.
                           Example: "Technical terms: Kubernetes, microservices, API endpoints"
                           The prompt is tokenized and prepended to guide transcription.
            callbacks: Optional TranscriptionCallbacks for progress reporting and control.
                      Provides whisper.cpp-style callbacks:
                      - new_segment: Called when segment completes
                      - progress: Called with progress percentage (0-100)
                      - encoder_begin: Called before encoding (return False to abort)
                      - abort: Called periodically (return True to abort)
                      - logits_filter: Called to filter logits before sampling
            audio_cleaning: Apply adaptive audio cleaning before transcription (default: False).
                           When enabled, analyzes audio condition (SNR, reverb) and applies
                           denoising/dereverberation if needed. Skips enhancement if audio
                           is already clean (overhead <5ms). Can improve WER on noisy audio.

        Returns:
            Dictionary with "text", "segments", "language", quality metrics, and
            VAD metadata ("vad_speech_ratio", "vad_segments").

        Warning:
            word_timestamps=True is NOT IMPLEMENTED - will be silently ignored.
            SDPA-based attention doesn't return cross_qk for DTW alignment.

            variable_length=True mode is BROKEN and produces hallucinations.
            The decoder timestamp logic assumes 30s padded audio (1500 positions).
            With shorter sequences, the decoder emits garbage tokens.
            Use standard mode (variable_length=False) for 100% accuracy.
        """
        import warnings

        import numpy as np

        # Deprecation warning for variable_length mode
        if variable_length:
            warnings.warn(
                "variable_length=True is DEPRECATED and produces hallucinations. "
                "Decoder timestamps are hardcoded for 30s audio. "
                "Use variable_length=False for 100% accurate transcription. "
                "See WHISPERMLX_QUALITY_AUDIT_2025-12-17.md",
                DeprecationWarning,
                stacklevel=2,
            )

        # Warning for unimplemented word_timestamps
        if word_timestamps:
            warnings.warn(
                "word_timestamps=True is NOT IMPLEMENTED in WhisperMLX. "
                "SDPA-based attention doesn't return cross_qk for DTW alignment. "
                "Use mlx-whisper directly if word timestamps are required.",
                UserWarning,
                stacklevel=2,
            )

        # Load audio if path
        if isinstance(audio, str):
            audio = load_audio(audio, sample_rate=SAMPLE_RATE)

        # Convert to numpy if needed
        if isinstance(audio, mx.array):
            audio = np.array(audio)

        # Store original audio duration for metadata
        original_duration = len(audio) / SAMPLE_RATE

        # ===== AUDIO CLEANING (Phase 9 - Optional) =====
        # Adaptive denoising/dereverberation based on audio condition.
        # Skips clean audio with minimal overhead (<5ms).
        cleaning_result = None
        if audio_cleaning:
            router = _get_audio_router()
            if router is not None:
                cleaning_result = router.route(audio)
                audio = cleaning_result.audio

                if verbose:
                    print(
                        f"Audio cleaning: SNR={cleaning_result.condition.snr_db:.1f}dB, "
                        f"T60={cleaning_result.condition.reverb_t60:.2f}s, "
                        f"enhanced={cleaning_result.was_enhanced} ({cleaning_result.enhancement_type})",
                    )
            elif verbose:
                print("Audio cleaning: not available (tools.audio_cleaning not installed)")
        # ===== END AUDIO CLEANING =====

        # ===== VAD PREPROCESSING (P0.1 - ALWAYS ON) =====
        # Silero VAD filters silence from audio before transcription.
        # This provides 2-4x speedup on audio with silence while maintaining 100% accuracy.
        vad_result: VADResult | None = None
        try:
            speech_audio, vad_result = preprocess_audio_with_vad(
                audio,
                aggressiveness=vad_aggressiveness,
                padding_ms=50,  # Small padding to avoid cutting speech
            )

            if verbose:
                print(f"VAD: {vad_result.speech_ratio:.1%} speech, "
                      f"{len(vad_result.segments)} segments, "
                      f"{vad_result.speech_duration:.1f}s / {vad_result.total_duration:.1f}s")

            # If audio is mostly silent (VAD finds < 5% speech), return early
            if vad_result.is_mostly_silent:
                if verbose:
                    print("VAD: Audio is mostly silent, skipping transcription")
                # GAP 44: Call progress callback even for silent audio
                if callbacks is not None:
                    callbacks.on_progress(100.0)
                return {
                    "text": "",
                    "segments": [],
                    "language": language or "en",
                    "avg_logprob": float("nan"),
                    "no_speech_prob": 1.0,
                    "compression_ratio": 0.0,
                    "temperature": 0.0,
                    "is_silent": True,
                    "vad_speech_ratio": vad_result.speech_ratio,
                    "vad_segments": len(vad_result.segments),
                    "audio_cleaning_applied": cleaning_result.was_enhanced if cleaning_result else None,
                    "audio_cleaning_type": cleaning_result.enhancement_type if cleaning_result else None,
                    "audio_cleaning_snr_db": cleaning_result.condition.snr_db if cleaning_result else None,
                }

            # Use speech-only audio for transcription (faster)
            if len(speech_audio) > 0:
                audio = speech_audio

        except ImportError as e:
            # Silero VAD not available - fall back to no preprocessing
            if verbose:
                print(f"VAD: Silero VAD not available ({e}), using full audio")

        # ===== END VAD PREPROCESSING =====

        # Check for silent audio BEFORE running model (prevents hallucinations)
        # This is a fallback if VAD failed or audio is still too quiet
        from .audio import get_audio_rms_db, is_silent_audio

        if is_silent_audio(audio):
            if verbose:
                rms_db = get_audio_rms_db(audio)
                print(f"Silent audio detected (RMS: {rms_db:.1f} dB < -40 dB)")
            return {
                "text": "",
                "segments": [],
                "language": language or "en",
                "avg_logprob": float("nan"),
                "no_speech_prob": 1.0,
                "compression_ratio": 0.0,
                "temperature": 0.0,
                "is_silent": True,
                "vad_speech_ratio": vad_result.speech_ratio if vad_result else None,
                "vad_segments": len(vad_result.segments) if vad_result else None,
                "audio_cleaning_applied": cleaning_result.was_enhanced if cleaning_result else None,
                "audio_cleaning_type": cleaning_result.enhancement_type if cleaning_result else None,
                "audio_cleaning_snr_db": cleaning_result.condition.snr_db if cleaning_result else None,
            }

        # Compute mel spectrogram with duration awareness
        actual_duration = None  # Track for variable-length mode
        if variable_length:
            mel, encoder_positions, actual_duration = compute_mel_for_duration(
                audio, n_mels=self.config.n_mels, pad_to_30s=False,
            )
            # Set decoder timestamp precision for actual duration
            self.decoder.set_precision(actual_duration, encoder_positions)
            if verbose:
                print(f"Variable-length mode: {actual_duration:.2f}s audio, "
                      f"{encoder_positions} encoder positions, "
                      f"precision={self.decoder.precision:.4f}s")
        else:
            # Standard mode: pad to 30s
            mel = log_mel_spectrogram(audio, n_mels=self.config.n_mels)
            # Pad/trim to standard length (mel shape is (n_frames, n_mels))
            target_len = self.config.n_audio_ctx * 2  # mel frames = 3000
            if mel.shape[0] < target_len:
                mel = mx.pad(mel, [(0, target_len - mel.shape[0]), (0, 0)])
            elif mel.shape[0] > target_len:
                mel = mel[:target_len, :]
            self.decoder.reset_precision()

        # Add batch dimension
        mel = mel[None]

        # Encode audio (with caching if enabled)
        encoder_positions_for_cache = (
            encoder_positions if variable_length
            else self.config.n_audio_ctx
        )

        cache_hit = False
        if self._encoder_cache is not None:
            # Check cache first
            cached = self._encoder_cache.get(mel, variable_length)
            if cached is not None:
                audio_features = cached.audio_features
                cache_hit = True
                if verbose:
                    print(f"Encoder cache HIT (access #{cached.access_count})")

        if not cache_hit:
            # GAP 44: encoder_begin callback - check if should continue
            if callbacks is not None and not callbacks.on_encoder_begin():
                return {
                    "text": "",
                    "segments": [],
                    "language": language or "en",
                    "aborted": True,
                }

            # Cache miss - encode audio
            audio_features = self.embed_audio(mel, variable_length=variable_length)

            # Store in cache if enabled
            if self._encoder_cache is not None:
                self._encoder_cache.put(
                    mel=mel,
                    audio_features=audio_features,
                    variable_length=variable_length,
                    encoder_positions=encoder_positions_for_cache,
                    audio_duration=actual_duration,
                )
                if verbose:
                    n = len(self._encoder_cache)
                    print(f"Encoder cache MISS - stored (entries: {n})")

        # GAP 44: Progress callback after encoding (encoding ~20% of work)
        if callbacks is not None:
            callbacks.on_progress(20.0)

        # Encoder VAD-based silence detection (Phase 3 Optimization)
        # If encoder VAD head is loaded, check speech ratio to skip decoding on silence
        if self._encoder_vad_head is not None:
            speech_mask, speech_ratio = self.get_encoder_vad_mask(audio_features)
            if verbose:
                print(f"Encoder VAD: speech ratio = {speech_ratio:.1%}")

            # If very little speech detected, skip expensive decoding
            # Use 5% threshold - if less than 5% is speech, treat as silence
            if speech_ratio < 0.05:
                if verbose:
                    print(f"Encoder VAD: skipping decoder (speech ratio {speech_ratio:.1%} < 5%)")
                # GAP 44: Call progress callback even for silent audio
                if callbacks is not None:
                    callbacks.on_progress(100.0)
                return {
                    "text": "",
                    "segments": [],
                    "language": language or "en",
                    "avg_logprob": float("nan"),
                    "no_speech_prob": 1.0,
                    "compression_ratio": 0.0,
                    "temperature": 0.0,
                    "is_silent": True,
                    "vad_speech_ratio": speech_ratio,
                }

        # Get tokenizer
        from .tokenizer import get_whisper_tokenizer
        is_multilingual = self.config.n_vocab >= 51865
        num_langs = self.config.n_vocab - 51765 - int(is_multilingual)
        tokenizer = get_whisper_tokenizer(
            multilingual=is_multilingual,
            num_languages=num_langs,
            language=language,
            task=task,
        )

        # Detect language if not specified
        if language is None:
            language = self._detect_language(audio_features, tokenizer)
            if verbose:
                print(f"Detected language: {language}")

        # J9: Tokenize initial_prompt for terminology injection
        prompt_tokens = None
        if initial_prompt:
            prompt_tokens = tokenizer.encode(initial_prompt)
            if verbose:
                print(f"Initial prompt: {len(prompt_tokens)} tokens")

        # Import compression_ratio for quality checking
        from .decoding import compression_ratio

        # Normalize temperature to tuple for fallback support
        if isinstance(temperature, (int, float)):
            temperatures = (float(temperature),)
        else:
            temperatures = tuple(temperature)

        # Temperature fallback loop
        decode_result = None
        final_temperature = temperatures[0]

        for temp_idx, temp in enumerate(temperatures):
            # GAP 44: Abort callback - check if should stop
            if callbacks is not None and callbacks.should_abort():
                return {
                    "text": "",
                    "segments": [],
                    "language": language,
                    "aborted": True,
                }

            # Decode with current temperature
            tokens, segments, avg_logprob, no_speech_prob = self._decode_with_metrics(
                audio_features,
                tokenizer,
                temperature=temp,
                max_initial_timestamp=max_initial_timestamp,
                audio_duration=actual_duration,
                no_speech_threshold=no_speech_threshold,
                prompt_tokens=prompt_tokens,
                repetition_penalty=repetition_penalty,
                skip_logprobs=skip_logprobs,
            )

            # GAP 44: Progress callback after decode (decode is ~80% of work)
            # Progress from 20% to 100% across temperature attempts
            if callbacks is not None:
                progress = 20.0 + (80.0 * (temp_idx + 1) / len(temperatures))
                callbacks.on_progress(progress)

            # Decode text
            text = tokenizer.decode(tokens)

            # Calculate compression ratio
            comp_ratio = compression_ratio(text) if text else 0.0

            # Check quality thresholds
            needs_fallback = False

            # Check compression ratio (detects repetition loops)
            if compression_ratio_threshold is not None and comp_ratio > compression_ratio_threshold:
                needs_fallback = True
                if verbose:
                    print(f"Temperature {temp}: compression ratio {comp_ratio:.2f} > {compression_ratio_threshold} (retry)")

            # Check logprob threshold (detects low confidence)
            if logprob_threshold is not None and avg_logprob < logprob_threshold:
                # Only fail on logprob if we also have high no_speech
                # This avoids false positives on unusual but correct text
                if no_speech_threshold is not None and no_speech_prob > no_speech_threshold:
                    # This is likely silence, not failed transcription
                    text = ""
                    tokens = []
                    segments = []
                    if verbose:
                        print(f"Temperature {temp}: detected silence (no_speech={no_speech_prob:.2f})")
                    break
                needs_fallback = True
                if verbose:
                    print(f"Temperature {temp}: avg_logprob {avg_logprob:.2f} < {logprob_threshold} (retry)")

            # Store result
            decode_result = {
                "text": text,
                "tokens": tokens,
                "segments": segments,
                "avg_logprob": avg_logprob,
                "no_speech_prob": no_speech_prob,
                "compression_ratio": comp_ratio,
                "temperature": temp,
            }
            final_temperature = temp

            # If quality is acceptable or this is the last temperature, stop
            if not needs_fallback or temp_idx == len(temperatures) - 1:
                if verbose and needs_fallback:
                    print(f"Temperature {temp}: accepting despite quality issues (last fallback)")
                break

        # GAP 44: new_segment callback for each completed segment
        if callbacks is not None and decode_result:
            for segment in decode_result["segments"]:
                callbacks.on_new_segment(segment)
            # Final progress callback
            callbacks.on_progress(100.0)

        # Return result with quality metrics and VAD metadata
        return {
            "text": decode_result["text"] if decode_result else "",
            "segments": decode_result["segments"] if decode_result else [],
            "language": language,
            # Quality metrics
            "avg_logprob": decode_result["avg_logprob"] if decode_result else float("nan"),
            "no_speech_prob": decode_result["no_speech_prob"] if decode_result else float("nan"),
            "compression_ratio": decode_result["compression_ratio"] if decode_result else float("nan"),
            "temperature": final_temperature,
            # VAD metadata (P0.1)
            "vad_speech_ratio": vad_result.speech_ratio if vad_result else None,
            "vad_segments": len(vad_result.segments) if vad_result else None,
            "vad_original_duration": original_duration,
            # Audio cleaning metadata (Phase 9)
            "audio_cleaning_applied": cleaning_result.was_enhanced if cleaning_result else None,
            "audio_cleaning_type": cleaning_result.enhancement_type if cleaning_result else None,
            "audio_cleaning_snr_db": cleaning_result.condition.snr_db if cleaning_result else None,
        }

    def transcribe_batch(
        self,
        audio_list: list[str | numpy.ndarray | mx.array],
        *,
        language: str | None = None,
        task: str = "transcribe",
        temperature: float = 0.0,
        max_initial_timestamp: float = 1.0,
        sort_by_length: bool = True,
        dynamic_padding: bool = True,
        verbose: bool = False,
    ) -> list[dict]:
        """
        Transcribe multiple audio files in parallel (C2 batched decoder optimization).

        This method processes multiple audio inputs simultaneously, improving
        throughput by 1.5-2x compared to sequential transcription.

        Key features:
        - Batch encoder: All audio mel spectrograms encoded in one GPU call
        - Batch decoder: All sequences decoded in parallel with shared compute
        - Variable-length stopping: Each sequence finishes independently
        - C4 optimization: Sort by length to reduce padding waste

        Limitations (compared to single transcribe()):
        - No VAD preprocessing (each audio processed as-is)
        - No temperature fallback (uses single temperature)
        - No compression/logprob quality checks
        - All audio assumed to be <= 30s (for longer, use transcribe_long)
        - All audio in batch must use same language

        Use cases:
        - Server batch processing (multiple concurrent requests)
        - Bulk transcription of short audio clips
        - Real-time batch inference pipelines

        Args:
            audio_list: List of audio file paths or waveform arrays
            language: Language code (auto-detect if None, uses first audio)
            task: "transcribe" or "translate"
            temperature: Sampling temperature (0 for greedy)
            max_initial_timestamp: Maximum timestamp at start
            sort_by_length: If True, sort audio by duration before batching
                           to reduce decoder waste (C4 optimization, default True).
                           Shorter sequences finish decoding first, reducing idle time.
            dynamic_padding: Ignored (Whisper requires 30s/3000 mel frames).
                            Kept for API compatibility.
            verbose: Print progress

        Returns:
            List of dictionaries with "text", "segments", "language"
        """
        import numpy as np

        batch_size = len(audio_list)
        if batch_size == 0:
            return []

        if verbose:
            print(f"Batch transcription: {batch_size} audio files")

        # Load and convert all audio to numpy
        audio_arrays = []
        audio_lengths = []  # Track lengths for sorting (C4)
        for _i, audio in enumerate(audio_list):
            if isinstance(audio, str):
                arr = load_audio(audio, sample_rate=SAMPLE_RATE)
            elif isinstance(audio, mx.array):
                arr = np.array(audio)
            else:
                arr = audio
            audio_arrays.append(arr)
            audio_lengths.append(len(arr))

        # C4: Sort by length to reduce padding waste
        if sort_by_length and batch_size > 1:
            # Create sorted indices (shortest to longest)
            sorted_indices = sorted(range(batch_size), key=lambda i: audio_lengths[i])
            # Reorder audio arrays
            audio_arrays = [audio_arrays[i] for i in sorted_indices]
            audio_lengths = [audio_lengths[i] for i in sorted_indices]
            # Create reverse mapping for restoring original order
            reverse_indices = [0] * batch_size
            for new_idx, orig_idx in enumerate(sorted_indices):
                reverse_indices[orig_idx] = new_idx
            if verbose:
                min_len = audio_lengths[0] / SAMPLE_RATE
                max_len = audio_lengths[-1] / SAMPLE_RATE
                print(f"C4: Sorted by length ({min_len:.1f}s - {max_len:.1f}s)")
        else:
            sorted_indices = list(range(batch_size))
            reverse_indices = list(range(batch_size))

        # Compute mel spectrograms
        mels = []
        mel_lengths = []  # Track mel frame lengths for dynamic padding
        max_target_len = self.config.n_audio_ctx * 2  # 3000 mel frames (30s)

        for arr in audio_arrays:
            mel = log_mel_spectrogram(arr, n_mels=self.config.n_mels)
            mel_lengths.append(mel.shape[0])
            mels.append(mel)

        # Whisper encoder REQUIRES 3000 mel frames (1500 encoder positions)
        # Dynamic padding is NOT compatible - decoder timestamps assume 30s audio
        # C4 benefit: sorted audio allows decoder to finish similar-length sequences together
        target_len = max_target_len

        if verbose and dynamic_padding:
            print("Note: Whisper requires 30s padding for encoder; dynamic padding disabled")

        # Pad/truncate mels to target length
        for i, mel in enumerate(mels):
            if mel.shape[0] < target_len:
                mels[i] = mx.pad(mel, [(0, target_len - mel.shape[0]), (0, 0)])
            elif mel.shape[0] > target_len:
                mels[i] = mel[:target_len, :]

        # Stack mels into batch
        mel_batch = mx.stack(mels)  # (batch_size, 3000, n_mels)

        if verbose:
            print(f"Mel batch shape: {mel_batch.shape}")

        # Get tokenizer
        from .tokenizer import get_whisper_tokenizer
        is_multilingual = self.config.n_vocab >= 51865
        num_langs = self.config.n_vocab - 51765 - int(is_multilingual)

        # Create tokenizer (will update with language after detection if needed)
        tokenizer = get_whisper_tokenizer(
            multilingual=is_multilingual,
            num_languages=num_langs,
            language=language,
            task=task,
        )

        # Auto-detect language from first audio if not specified
        if language is None:
            first_features = self.embed_audio(mel_batch[0:1])
            language = self._detect_language(first_features, tokenizer)
            if verbose:
                print(f"Detected language: {language}")
            # Re-create tokenizer with detected language
            tokenizer = get_whisper_tokenizer(
                multilingual=is_multilingual,
                num_languages=num_langs,
                language=language,
                task=task,
            )

        # Batch encode all audio
        if verbose:
            print("Batch encoding...")
        audio_features_batch = self.embed_audio(mel_batch)  # (batch_size, 1500, n_state)
        mx.eval(audio_features_batch)

        if verbose:
            print(f"Audio features shape: {audio_features_batch.shape}")
            print("Batch decoding...")

        # Batch decode
        decode_results = self._decode_batch(
            audio_features_batch,
            tokenizer,
            temperature=temperature,
            max_initial_timestamp=max_initial_timestamp,
        )

        # Format results (in sorted order)
        sorted_results = []
        for tokens, segments in decode_results:
            text = tokenizer.decode(tokens).strip()
            sorted_results.append({
                "text": text,
                "segments": segments,
                "language": language,
            })

        # C4: Restore original order if we sorted
        if sort_by_length and batch_size > 1:
            # Create results in original order using reverse mapping
            results = [None] * batch_size
            for orig_idx in range(batch_size):
                sorted_idx = reverse_indices[orig_idx]
                results[orig_idx] = sorted_results[sorted_idx]
        else:
            results = sorted_results

        if verbose:
            print(f"Batch transcription complete: {len(results)} results")

        return results

    def _detect_language(
        self,
        audio_features: mx.array,
        tokenizer,
        return_probs: bool = False,
        confidence_threshold: float = 0.0,
    ) -> str:
        """Detect language from audio features.

        Args:
            audio_features: Encoder output
            tokenizer: Whisper tokenizer
            return_probs: If True, returns dict with probs instead of just string
            confidence_threshold: Minimum confidence to return a language (0=always return)

        Returns:
            If return_probs=False: language code string (e.g., "en") or "unknown" if below threshold
            If return_probs=True: dict with "language", "confidence", "all_probs" keys
        """
        # Simple language detection using first token prediction
        sot = mx.array([[tokenizer.sot]])
        logits = self.logits(sot, audio_features)

        # Get language token probabilities
        lang_tokens = list(tokenizer.all_language_tokens)
        lang_probs = mx.softmax(logits[0, 0, lang_tokens], axis=-1)
        mx.eval(lang_probs)

        # Convert to numpy for easier processing
        probs_np = lang_probs.tolist()

        # Build language -> probability mapping using tokenizer's language lookup
        # This avoids fragile string parsing
        lang_probs_dict = {}
        for idx, token_id in enumerate(lang_tokens):
            # Use tokenizer's language token lookup if available
            if hasattr(tokenizer, 'to_language_token') and hasattr(tokenizer, 'all_language_codes'):
                # Find language code for this token
                for lang_code in getattr(tokenizer, 'all_language_codes', []):
                    if tokenizer.to_language_token(lang_code) == token_id:
                        lang_probs_dict[lang_code] = probs_np[idx]
                        break
                else:
                    # Fallback: decode and strip special chars
                    decoded = tokenizer.decode([token_id])
                    lang_code = decoded.replace("<|", "").replace("|>", "").strip()
                    if lang_code:
                        lang_probs_dict[lang_code] = probs_np[idx]
            else:
                # Fallback: decode and strip special chars
                decoded = tokenizer.decode([token_id])
                lang_code = decoded.replace("<|", "").replace("|>", "").strip()
                if lang_code:
                    lang_probs_dict[lang_code] = probs_np[idx]

        # Get best language
        best_idx = int(mx.argmax(lang_probs))
        best_prob = float(probs_np[best_idx])
        lang_tokens[best_idx]

        # Get language code from dict keys by finding the one with highest prob
        best_lang = max(lang_probs_dict.keys(), key=lambda k: lang_probs_dict[k]) if lang_probs_dict else "unknown"

        if return_probs:
            # Return top-k languages sorted by probability
            sorted_langs = sorted(lang_probs_dict.items(), key=lambda x: x[1], reverse=True)
            return {
                "language": best_lang if best_prob >= confidence_threshold else "unknown",
                "confidence": best_prob,
                "all_probs": dict(sorted_langs[:10]),  # Top 10
            }

        # Check confidence threshold
        if confidence_threshold > 0 and best_prob < confidence_threshold:
            return "unknown"

        return best_lang

    def _decode_step(
        self,
        tokens: mx.array,
        audio_features: mx.array,
        kv_cache: list | None,
        logit_filters: list,
        all_tokens: mx.array,
        token_pos: int,
        temperature: float,
    ) -> tuple[mx.array, mx.array, list | None, mx.array]:
        """
        Single decode step helper (OPT-NEW-19).

        Returns:
            Tuple of (logits, next_token, kv_cache, filtered_logits)
        """
        # Get logits from decoder
        logits, kv_cache, _, _ = self.decoder(tokens, audio_features, kv_cache=kv_cache)
        logits = logits[:, -1]

        # OPT-NEW-21: Skip redundant float32 conversion if already float32
        if logits.dtype != mx.float32:
            logits = logits.astype(mx.float32)

        # Apply logit filters using current tokens slice
        current_tokens = all_tokens[:token_pos]
        filtered_logits = apply_filters(logits, current_tokens[None], logit_filters)

        # Sample using compiled functions (OPT-NEW-1)
        if temperature == 0:
            next_token = greedy_decode(filtered_logits)
        else:
            next_token = sample_with_temperature(filtered_logits, temperature)

        return logits, next_token, kv_cache, filtered_logits

    def _decode_with_metrics(
        self,
        audio_features: mx.array,
        tokenizer,
        temperature: float = 0.0,
        max_initial_timestamp: float = 1.0,
        audio_duration: float | None = None,
        no_speech_threshold: float = 0.6,
        prompt_tokens: list[int] | None = None,
        repetition_penalty: float | None = None,
        skip_logprobs: bool = False,
    ) -> tuple[list[int], list[dict], float, float]:
        """
        Decode tokens from audio features with proper logit filters.
        Returns quality metrics for temperature fallback.

        Uses mlx-whisper-style decode loop structure (OPT-NEW-19) for better
        async eval batching and cleaner structure.

        OPT-NEW-32: Supports prompt_tokens for condition_on_previous_text.
        When transcribing long audio, tokens from the previous window can be
        passed as prompt context to improve continuity and accuracy.

        Args:
            audio_features: Encoded audio from encoder
            tokenizer: Whisper tokenizer
            temperature: Sampling temperature (0 for greedy)
            max_initial_timestamp: Maximum timestamp at start
            audio_duration: Actual audio duration in seconds (for variable-length mode)
            no_speech_threshold: Probability threshold for no-speech early exit
            prompt_tokens: Optional list of tokens from previous window (OPT-NEW-32).
                          Prepended with sot_prev token to provide context.
            skip_logprobs: Skip computing per-token log probabilities (F10 optimization).
                          Provides ~2x decode speedup. avg_logprob will be NaN when True.

        Returns:
            Tuple of (token_list, segments_list, avg_logprob, no_speech_prob)
        """
        # Build initial tokens (SOT sequence)
        # OPT-NEW-32: If prompt_tokens provided, prepend with sot_prev token
        # Structure: [sot_prev, prompt_tokens..., sot, lang, task, ...]
        initial_tokens = []
        if prompt_tokens:
            # Truncate prompt to fit in context (max half of n_text_ctx)
            max_prompt_len = self.config.n_text_ctx // 2 - 1
            truncated_prompt = prompt_tokens[-max_prompt_len:]
            initial_tokens = [tokenizer.sot_prev] + truncated_prompt
        initial_tokens = initial_tokens + list(tokenizer.sot_sequence)
        tokens = mx.array([initial_tokens])
        sample_begin = len(initial_tokens)

        # Get timestamp precision
        precision = getattr(self.decoder, 'precision', 0.02)

        # Build logit filters
        options = DecodingOptions(
            temperature=temperature,
            max_initial_timestamp=max_initial_timestamp,
            suppress_blank=True,
            suppress_tokens="-1",
            without_timestamps=False,
            repetition_penalty=repetition_penalty,
        )
        logit_filters = build_logit_filters(
            tokenizer,
            options,
            sample_begin=sample_begin,
            n_vocab=self.config.n_vocab,
            precision=precision,
            audio_duration=audio_duration,
        )

        # KV cache
        kv_cache = None

        # OPT-NEW-18: Preallocate tokens array to avoid Python list append overhead
        # Using MLX array for efficient GPU operations
        max_tokens = self.config.n_text_ctx // 2
        all_tokens_arr = mx.zeros((max_tokens + len(initial_tokens),), dtype=mx.int32)
        initial_arr = mx.array(initial_tokens, dtype=mx.int32)
        all_tokens_arr = all_tokens_arr.at[:len(initial_tokens)].add(initial_arr)
        token_pos = len(initial_tokens)  # Current position in preallocated array

        # Track logprobs for quality metrics
        sum_logprob = 0.0
        n_logprobs = 0
        no_speech_prob = 0.0  # Track for quality metrics

        # First decode step - process full initial tokens (OPT-NEW-19)
        logits, next_token, kv_cache, filtered_logits = self._decode_step(
            tokens, audio_features, kv_cache, logit_filters,
            all_tokens_arr, token_pos, temperature,
        )

        # OPT-NEW-19: Batch async_eval on multiple computed values
        mx.async_eval(logits, next_token, kv_cache)

        # Compute no_speech_prob for quality metrics (always, not just for early exit)
        if tokenizer.no_speech is not None:
            probs_at_sot = mx.softmax(logits, axis=-1)
            no_speech_prob = float(probs_at_sot[0, tokenizer.no_speech])

        # F10: Track logprobs only if needed (skip for ~2x speedup)
        if not skip_logprobs:
            logprobs = compute_logprobs(filtered_logits)
            token_logprob = float(logprobs[0, int(next_token[0])])
            sum_logprob += token_logprob
            n_logprobs += 1

        next_token_int = int(next_token[0])

        # Check for early EOT
        if next_token_int == tokenizer.eot:
            all_tokens_list = all_tokens_arr[:token_pos].tolist()
            output_tokens = all_tokens_list[sample_begin:]
            segments = self._parse_segments(output_tokens, tokenizer, precision)
            avg_logprob = float("nan") if skip_logprobs else sum_logprob / max(n_logprobs, 1)
            return output_tokens, segments, avg_logprob, no_speech_prob

        # Update preallocated array
        next_arr = mx.array([next_token_int], dtype=mx.int32)[0]
        all_tokens_arr = all_tokens_arr.at[token_pos].add(next_arr)
        token_pos += 1
        tokens = mx.array([[next_token_int]])

        # Main decode loop - process single token at a time (OPT-NEW-19)
        for _i in range(1, max_tokens):
            # Use _decode_step helper for cleaner structure
            logits, next_token, kv_cache, filtered_logits = self._decode_step(
                tokens, audio_features, kv_cache, logit_filters,
                all_tokens_arr, token_pos, temperature,
            )

            # OPT-NEW-19: Batch async_eval on multiple computed values
            # This allows GPU to continue computing while CPU processes the result
            mx.async_eval(logits, next_token, kv_cache)

            # F10: Track logprobs only if needed (skip for ~2x speedup)
            if not skip_logprobs:
                logprobs = compute_logprobs(filtered_logits)
                token_logprob = float(logprobs[0, int(next_token[0])])
                sum_logprob += token_logprob
                n_logprobs += 1

            next_token_int = int(next_token[0])

            # Check for end
            if next_token_int == tokenizer.eot:
                break

            # OPT-NEW-18: Update preallocated array instead of list append
            next_arr = mx.array([next_token_int], dtype=mx.int32)[0]
            all_tokens_arr = all_tokens_arr.at[token_pos].add(next_arr)
            token_pos += 1
            tokens = mx.array([[next_token_int]])

        # Extract output tokens (after SOT sequence)
        # OPT-NEW-18: Convert preallocated array slice back to list
        all_tokens_list = all_tokens_arr[:token_pos].tolist()
        output_tokens = all_tokens_list[sample_begin:]

        # Parse segments from timestamp tokens
        segments = self._parse_segments(output_tokens, tokenizer, precision)

        # Calculate average logprob for quality metrics (F10: NaN when skipped)
        if skip_logprobs:
            avg_logprob = float("nan")
        else:
            avg_logprob = sum_logprob / max(n_logprobs, 1)

        return output_tokens, segments, avg_logprob, no_speech_prob

    def _decode(
        self,
        audio_features: mx.array,
        tokenizer,
        temperature: float = 0.0,
        max_initial_timestamp: float = 1.0,
        audio_duration: float | None = None,
        prompt_tokens: list[int] | None = None,
    ) -> tuple[list[int], list[dict]]:
        """
        Simple decode wrapper that returns only tokens and segments.

        Used by transcribe_long for chunk-by-chunk decoding without needing
        the quality metrics (avg_logprob, no_speech_prob) returned by
        _decode_with_metrics.

        F10 Optimization: Always uses skip_logprobs=True since logprobs are
        not returned, providing ~2x decode speedup for long audio.

        Args:
            audio_features: Encoded audio from encoder
            tokenizer: Whisper tokenizer
            temperature: Sampling temperature
            max_initial_timestamp: Maximum timestamp at start
            audio_duration: Actual audio duration for variable-length mode
            prompt_tokens: Optional prompt tokens for context continuity

        Returns:
            Tuple of (token_list, segments_list)
        """
        # F10: Always skip logprobs since we discard them (~2x decode speedup)
        tokens, segments, _, _ = self._decode_with_metrics(
            audio_features=audio_features,
            tokenizer=tokenizer,
            temperature=temperature,
            max_initial_timestamp=max_initial_timestamp,
            audio_duration=audio_duration,
            prompt_tokens=prompt_tokens,
            skip_logprobs=True,
        )
        return tokens, segments

    def _decode_batch(
        self,
        audio_features_batch: mx.array,
        tokenizer,
        temperature: float = 0.0,
        max_initial_timestamp: float = 1.0,
    ) -> list[tuple[list[int], list[dict]]]:
        """
        Decode multiple audio sequences in parallel (C2 optimization).

        Processes multiple audio inputs through the decoder simultaneously,
        improving throughput by 1.5-2x for batch transcription workloads.

        Key differences from single-sequence _decode_with_metrics:
        - KV cache grows with batch dimension
        - Logit filters applied per sequence in batch
        - Variable-length stopping: sequences finish independently

        Args:
            audio_features_batch: Batched encoded audio (batch_size, 1500, n_state)
            tokenizer: Whisper tokenizer
            temperature: Sampling temperature (0 for greedy)
            max_initial_timestamp: Maximum timestamp at start

        Returns:
            List of (token_list, segments_list) tuples, one per sequence in batch
        """
        batch_size = audio_features_batch.shape[0]

        # Build initial tokens (SOT sequence) - same for all sequences
        initial_tokens = list(tokenizer.sot_sequence)
        sample_begin = len(initial_tokens)

        # Batch initial tokens
        tokens = mx.array([initial_tokens] * batch_size)

        # Get timestamp precision
        precision = getattr(self.decoder, 'precision', 0.02)

        # Build logit filters (shared config, but applied per-sequence)
        options = DecodingOptions(
            temperature=temperature,
            max_initial_timestamp=max_initial_timestamp,
            suppress_blank=True,
            suppress_tokens="-1",
            without_timestamps=False,
        )
        logit_filters = build_logit_filters(
            tokenizer,
            options,
            sample_begin=sample_begin,
            n_vocab=self.config.n_vocab,
            precision=precision,
        )

        # KV cache (dynamic, grows with batch dimension automatically)
        kv_cache = None

        # Track tokens per sequence
        max_tokens = self.config.n_text_ctx // 2
        all_tokens = [list(initial_tokens) for _ in range(batch_size)]
        done = [False] * batch_size
        eot_token = tokenizer.eot

        # First decode step - process full initial tokens
        logits, kv_cache, _, _ = self.decoder(tokens, audio_features_batch, kv_cache=kv_cache)
        logits = logits[:, -1]  # (batch_size, vocab)

        if logits.dtype != mx.float32:
            logits = logits.astype(mx.float32)

        # Build token context for filters (batch of token sequences)
        current_tokens_batch = mx.array(all_tokens)

        # Apply logit filters
        filtered_logits = apply_filters(logits, current_tokens_batch, logit_filters)

        # Sample next tokens
        if temperature == 0:
            next_tokens = mx.argmax(filtered_logits, axis=-1)
        else:
            probs = mx.softmax(filtered_logits / temperature, axis=-1)
            next_tokens = mx.random.categorical(probs)

        mx.async_eval(logits, next_tokens, kv_cache)

        # Update tokens per sequence
        next_tokens_list = next_tokens.tolist()
        for b in range(batch_size):
            token = next_tokens_list[b]
            if token == eot_token:
                done[b] = True
            else:
                all_tokens[b].append(token)

        # Check if all done
        if all(done):
            return self._finalize_batch_decode(all_tokens, sample_begin, tokenizer, precision)

        # Main decode loop
        for _i in range(1, max_tokens):
            # For sequences that are done, repeat EOT to maintain batch
            batch_next = []
            for b in range(batch_size):
                if done[b]:
                    batch_next.append(eot_token)
                else:
                    batch_next.append(all_tokens[b][-1])
            tokens = mx.array(batch_next)[:, None]  # (batch_size, 1)

            # Decode step
            logits, kv_cache, _, _ = self.decoder(tokens, audio_features_batch, kv_cache=kv_cache)
            logits = logits[:, -1]

            if logits.dtype != mx.float32:
                logits = logits.astype(mx.float32)

            # Build token context (with done sequences having all their tokens)
            max_len = max(len(t) for t in all_tokens)
            padded_tokens = []
            for b in range(batch_size):
                seq = all_tokens[b]
                # Pad with EOT if needed
                padded = seq + [eot_token] * (max_len - len(seq))
                padded_tokens.append(padded)
            current_tokens_batch = mx.array(padded_tokens)

            # Apply logit filters
            filtered_logits = apply_filters(logits, current_tokens_batch, logit_filters)

            # Sample
            if temperature == 0:
                next_tokens = mx.argmax(filtered_logits, axis=-1)
            else:
                probs = mx.softmax(filtered_logits / temperature, axis=-1)
                next_tokens = mx.random.categorical(probs)

            mx.async_eval(logits, next_tokens, kv_cache)

            # Update tokens
            next_tokens_list = next_tokens.tolist()
            any_active = False
            for b in range(batch_size):
                if not done[b]:
                    token = next_tokens_list[b]
                    if token == eot_token:
                        done[b] = True
                    else:
                        all_tokens[b].append(token)
                        any_active = True

            if not any_active:
                break

        return self._finalize_batch_decode(all_tokens, sample_begin, tokenizer, precision)

    def _finalize_batch_decode(
        self,
        all_tokens: list[list[int]],
        sample_begin: int,
        tokenizer,
        precision: float,
    ) -> list[tuple[list[int], list[dict]]]:
        """Finalize batch decode by parsing segments for each sequence."""
        results = []
        for tokens in all_tokens:
            output_tokens = tokens[sample_begin:]
            segments = self._parse_segments(output_tokens, tokenizer, precision)
            results.append((output_tokens, segments))
        return results

    def _parse_segments(
        self,
        tokens: list[int],
        tokenizer,
        precision: float,
    ) -> list[dict]:
        """Parse timestamp tokens into segments."""
        timestamp_begin = tokenizer.timestamp_begin
        segments = []
        current_tokens = []
        start_time = None
        last_time = 0.0

        for token in tokens:
            if token >= timestamp_begin:
                # Timestamp token
                time = (token - timestamp_begin) * precision
                if start_time is None:
                    start_time = time
                else:
                    # End of segment
                    if current_tokens:
                        text = tokenizer.decode(current_tokens).strip()
                        if text:
                            segments.append({
                                "start": start_time,
                                "end": time,
                                "text": text,
                            })
                    current_tokens = []
                    start_time = time
                last_time = time
            else:
                # Text token
                current_tokens.append(token)

        # Handle trailing tokens
        if current_tokens and start_time is not None:
            text = tokenizer.decode(current_tokens).strip()
            if text:
                end_time = (
                    self.decoder.audio_duration
                    if hasattr(self.decoder, 'audio_duration') else last_time
                )
                segments.append({
                    "start": start_time,
                    "end": end_time or last_time,
                    "text": text,
                })

        # If no segments but we have tokens, create one segment
        if not segments and tokens:
            text_tokens = [t for t in tokens if t < timestamp_begin]
            if text_tokens:
                text = tokenizer.decode(text_tokens).strip()
                segments.append({
                    "start": 0.0,
                    "end": (
                        self.decoder.audio_duration
                        if hasattr(self.decoder, 'audio_duration') else 30.0
                    ),
                    "text": text,
                })

        return segments

    def transcribe_long(
        self,
        audio: str | numpy.ndarray | mx.array,
        *,
        language: str | None = None,
        task: str = "transcribe",
        batch_size: int = 4,
        chunk_length: float = 30.0,
        overlap: float = 1.0,
        variable_length_last_chunk: bool = True,
        temperature: float = 0.0,
        max_initial_timestamp: float = 1.0,
        condition_on_previous_text: bool = True,
        async_prefetch: bool = True,
        verbose: bool = False,
        # GAP 44: Callback system (Phase 6 stretch goal)
        callbacks: TranscriptionCallbacks | None = None,
    ) -> dict:
        """
        Transcribe long audio with batch parallel encoding (OPT-W6).

        This method significantly speeds up transcription of audio >30s by:
        1. Splitting audio into overlapping chunks
        2. Encoding all chunks in parallel (batched GPU operation)
        3. Decoding each chunk sequentially
        4. Merging results with proper timestamp offsets

        OPT-NEW-32: Supports condition_on_previous_text for context continuity.
        When enabled (default), tokens from previous windows are passed as prompt
        to subsequent decode calls, improving accuracy for continuous speech.

        OPT-PREFETCH: Supports async_prefetch for pipelined mel preparation.
        When enabled (default), mel spectrograms are prepared in background threads
        while the GPU is decoding, overlapping CPU work with GPU work for 10-15%
        speedup on long audio.

        For short audio (<30s), use transcribe() instead.

        Args:
            audio: Audio file path or waveform array
            language: Language code (auto-detect if None)
            task: "transcribe" or "translate"
            batch_size: Number of chunks to encode in parallel (default: 4)
            chunk_length: Length of each chunk in seconds (default: 30.0)
            overlap: Overlap between chunks in seconds (default: 1.0)
            variable_length_last_chunk: Use variable-length mode for last partial chunk
            temperature: Sampling temperature (0 for greedy)
            max_initial_timestamp: Maximum timestamp at start
            condition_on_previous_text: Pass tokens from previous window as prompt
                for better context continuity (default: True). OPT-NEW-32.
            async_prefetch: Prepare mel spectrograms in background threads while
                decoding (default: True). OPT-PREFETCH. Provides 10-15% speedup.
            verbose: Print progress
            callbacks: Optional TranscriptionCallbacks for progress reporting and control.
                      See TranscriptionCallbacks for details.

        Returns:
            Dictionary with "text", "segments", "language"
        """
        import time

        import numpy as np

        # Load audio if path
        if isinstance(audio, str):
            audio = load_audio(audio, sample_rate=SAMPLE_RATE)

        # Convert to numpy if needed
        if isinstance(audio, mx.array):
            audio = np.array(audio)

        # Calculate duration
        total_duration = len(audio) / SAMPLE_RATE

        # If short audio, use regular transcribe
        if total_duration <= chunk_length:
            if verbose:
                print(f"Audio {total_duration:.2f}s <= {chunk_length}s, standard mode")
            return self.transcribe(
                audio,
                language=language,
                task=task,
                temperature=temperature,
                max_initial_timestamp=max_initial_timestamp,
                verbose=verbose,
                callbacks=callbacks,
            )

        if verbose:
            print(
                f"Processing {total_duration:.2f}s audio in chunks of "
                f"{chunk_length}s with {overlap}s overlap",
            )

        # Split audio into chunks
        chunk_samples = int(chunk_length * SAMPLE_RATE)
        overlap_samples = int(overlap * SAMPLE_RATE)
        stride_samples = chunk_samples - overlap_samples

        chunks = []
        chunk_offsets = []  # Time offset for each chunk
        chunk_durations = []  # Actual duration of each chunk (for variable-length)
        pos = 0
        while pos < len(audio):
            end = min(pos + chunk_samples, len(audio))
            chunk = audio[pos:end]
            actual_chunk_duration = len(chunk) / SAMPLE_RATE

            # Track if this is a partial chunk
            is_partial = len(chunk) < chunk_samples

            # Pad partial chunks unless we're using variable-length mode for them
            if is_partial and not variable_length_last_chunk:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
                actual_chunk_duration = chunk_length

            chunks.append(chunk)
            chunk_offsets.append(pos / SAMPLE_RATE)
            chunk_durations.append(actual_chunk_duration)
            pos += stride_samples

        n_chunks = len(chunks)
        if verbose:
            print(f"Split into {n_chunks} chunks")
            if variable_length_last_chunk and chunk_durations[-1] < chunk_length:
                print(f"Last chunk: variable-length mode ({chunk_durations[-1]:.2f}s)")

        # Compute mel spectrograms for all chunks
        # OPT-PREFETCH: Use async mel preparer when enabled
        mels = []
        mel_infos = []  # Track (is_variable_length, encoder_positions, actual_duration)

        if async_prefetch and n_chunks > 1:
            # Async prefetch mode: submit all chunks to background threads
            if verbose:
                t_mel_start = time.perf_counter()

            # Determine which chunks use variable-length mode
            variable_length_indices = set()
            for i, chunk in enumerate(chunks):
                is_last = (i == n_chunks - 1)
                is_partial = (len(chunk) < chunk_samples)
                if variable_length_last_chunk and is_last and is_partial:
                    variable_length_indices.add(i)

            # Use AsyncMelPreparer to prepare mels in background
            with AsyncMelPreparer(
                n_mels=self.config.n_mels,
                n_audio_ctx=self.config.n_audio_ctx,
                max_workers=min(4, n_chunks),  # Limit workers to avoid overhead
            ) as preparer:
                # Submit all chunks for background processing
                futures = preparer.submit_batch(
                    chunks, chunk_samples, variable_length_indices,
                )

                # Collect results (blocks if not ready)
                for future in futures:
                    mel, mel_info = future.result()
                    mels.append(mel)
                    mel_infos.append(mel_info)

            if verbose:
                t_mel_elapsed = time.perf_counter() - t_mel_start
                print(f"Async mel prep: {t_mel_elapsed*1000:.1f}ms for {n_chunks} chunks")
        else:
            # Synchronous mode: compute mels sequentially
            for i, (chunk, _chunk_dur) in enumerate(zip(chunks, chunk_durations, strict=False)):
                is_last = (i == n_chunks - 1)
                is_partial = (len(chunk) < chunk_samples)
                use_variable = variable_length_last_chunk and is_last and is_partial

                if use_variable:
                    # Variable-length mode for partial last chunk
                    mel, encoder_positions, actual_duration = compute_mel_for_duration(
                        chunk, n_mels=self.config.n_mels, pad_to_30s=False,
                    )
                    mel_infos.append((True, encoder_positions, actual_duration))
                else:
                    # Standard 30s padded mode
                    mel = log_mel_spectrogram(chunk, n_mels=self.config.n_mels)
                    # Pad/trim to standard length (3000 frames)
                    target_len = self.config.n_audio_ctx * 2
                    if mel.shape[0] < target_len:
                        mel = mx.pad(mel, [(0, target_len - mel.shape[0]), (0, 0)])
                    elif mel.shape[0] > target_len:
                        mel = mel[:target_len, :]
                    mel_infos.append((False, self.config.n_audio_ctx, chunk_length))

                mels.append(mel)

        # Get tokenizer
        from .tokenizer import get_whisper_tokenizer
        is_multilingual = self.config.n_vocab >= 51865
        num_langs = self.config.n_vocab - 51765 - int(is_multilingual)
        tokenizer = get_whisper_tokenizer(
            multilingual=is_multilingual,
            num_languages=num_langs,
            language=language,
            task=task,
        )

        # Detect language from first chunk if not specified
        if language is None:
            first_mel = mels[0][None]  # Add batch dim
            first_features = self.embed_audio(first_mel)
            language = self._detect_language(first_features, tokenizer)
            if verbose:
                print(f"Detected language: {language}")

        # GAP 44: encoder_begin callback before batch encoding
        if callbacks is not None and not callbacks.on_encoder_begin():
            return {
                "text": "",
                "segments": [],
                "language": language,
                "aborted": True,
            }

        # Batch encode all chunks in parallel
        # Note: Variable-length chunks must be encoded separately
        if verbose:
            t0 = time.perf_counter()

        all_features = []

        # Find indices of standard and variable-length chunks
        standard_indices = [
            i for i, (is_var, _, _) in enumerate(mel_infos) if not is_var
        ]
        variable_indices = [
            i for i, (is_var, _, _) in enumerate(mel_infos) if is_var
        ]

        # Pre-allocate list for proper ordering
        features_by_idx = [None] * n_chunks

        # Batch encode standard chunks
        if standard_indices:
            for batch_start in range(0, len(standard_indices), batch_size):
                batch_end = min(batch_start + batch_size, len(standard_indices))
                batch_idx = standard_indices[batch_start:batch_end]
                batch_mels = mx.stack([mels[i] for i in batch_idx])

                # Encode batch in parallel
                batch_features = self.embed_audio(batch_mels, variable_length=False)
                mx.eval(batch_features)

                # Store features at correct indices
                for j, idx in enumerate(batch_idx):
                    features_by_idx[idx] = batch_features[j:j+1]

        # Encode variable-length chunks separately
        for idx in variable_indices:
            mel = mels[idx]
            features = self.embed_audio(mel[None], variable_length=True)
            mx.eval(features)
            features_by_idx[idx] = features

        all_features = features_by_idx

        if verbose:
            encode_time = time.perf_counter() - t0
            ms = encode_time / n_chunks * 1000
            print(f"Encoded {n_chunks} chunks in {encode_time:.2f}s ({ms:.1f}ms/ch)")
            if variable_indices:
                ns, nv = len(standard_indices), len(variable_indices)
                print(f"  ({ns} standard + {nv} variable-length)")

        # GAP 44: Progress callback after encoding (encoding ~30% of work for long audio)
        if callbacks is not None:
            callbacks.on_progress(30.0)

        # Decode each chunk sequentially
        if verbose:
            t0 = time.perf_counter()

        all_segments = []
        all_text = []
        all_tokens = []  # OPT-NEW-32: Track all tokens for condition_on_previous_text
        prompt_reset_since = 0  # OPT-NEW-32: Track prompt reset position

        chunk_data = zip(all_features, chunk_offsets, mel_infos, strict=False)
        for i, (features, offset, mel_info) in enumerate(chunk_data):
            # GAP 44: Abort callback - check if should stop
            if callbacks is not None and callbacks.should_abort():
                return {
                    "text": " ".join(all_text) if all_text else "",
                    "segments": all_segments,
                    "language": language,
                    "aborted": True,
                    "chunks_processed": i,
                }

            is_variable, encoder_positions, chunk_duration = mel_info

            # Set decoder precision based on chunk type
            if is_variable:
                self.decoder.set_precision(chunk_duration, encoder_positions)
            else:
                self.decoder.reset_precision()

            # Encoder VAD-based chunk skipping (Phase 3 Optimization)
            # Skip decoding for chunks that are mostly silent
            if self._encoder_vad_head is not None:
                speech_mask, speech_ratio = self.get_encoder_vad_mask(features)
                if speech_ratio < 0.05:
                    # Silent chunk - skip decoding
                    if verbose:
                        print(f"Chunk {i+1}/{n_chunks}: SKIPPED (VAD speech_ratio={speech_ratio:.1%})")
                    continue  # Skip to next chunk

            # OPT-NEW-32: Prepare prompt tokens from previous windows
            prompt_tokens = None
            if condition_on_previous_text and i > 0:
                prompt_tokens = all_tokens[prompt_reset_since:]

            # Decode this chunk
            tokens, segments = self._decode(
                features,
                tokenizer,
                temperature=temperature,
                max_initial_timestamp=max_initial_timestamp,
                audio_duration=chunk_duration if is_variable else None,
                prompt_tokens=prompt_tokens,
            )

            # OPT-NEW-32: Accumulate tokens for next window's prompt
            all_tokens.extend(tokens)

            # OPT-NEW-32: Reset prompt if high temperature (mlx-whisper pattern)
            # High temp = low confidence, don't propagate bad context
            if not condition_on_previous_text or temperature > 0.5:
                prompt_reset_since = len(all_tokens)

            # Adjust timestamps by chunk offset
            for seg in segments:
                seg["start"] += offset
                seg["end"] += offset
                # Clip to actual audio duration
                seg["end"] = min(seg["end"], total_duration)

            all_segments.extend(segments)

            # GAP 44: new_segment callback for each segment in this chunk
            if callbacks is not None:
                for seg in segments:
                    callbacks.on_new_segment(seg)

            # GAP 44: Progress callback after chunk decode
            # Progress from 30% to 100% across chunks (decode is ~70% of work)
            if callbacks is not None:
                progress = 30.0 + (70.0 * (i + 1) / n_chunks)
                callbacks.on_progress(progress)

            # Decode text for this chunk
            chunk_text = tokenizer.decode(tokens)
            all_text.append(chunk_text)

            if verbose:
                mode = "var" if is_variable else "std"
                if condition_on_previous_text:
                    n_prompt = len(prompt_tokens) if prompt_tokens else 0
                    prompt_info = f" (prompt: {n_prompt} tokens)"
                else:
                    prompt_info = ""
                end_t = offset + chunk_duration
                msg = f"Chunk {i+1}/{n_chunks} [{mode}]: {offset:.1f}s-{end_t:.1f}s"
                print(f"{msg}{prompt_info}")

        if verbose:
            decode_time = time.perf_counter() - t0
            ms = decode_time / n_chunks * 1000
            print(f"Decoded {n_chunks} chunks in {decode_time:.2f}s ({ms:.1f}ms/ch)")

        # Merge overlapping segments
        merged_segments = self._merge_overlapping_segments(all_segments, overlap)

        # Merge all text
        full_text = " ".join(all_text)

        return {
            "text": full_text,
            "segments": merged_segments,
            "language": language,
        }

    def _merge_overlapping_segments(
        self,
        segments: list[dict],
        overlap: float,
    ) -> list[dict]:
        """
        Merge overlapping segments from adjacent chunks.

        When chunks overlap, we may get duplicate transcriptions.
        This function deduplicates by preferring segments from the
        chunk that "owns" that time region.

        Args:
            segments: List of segments with start/end times
            overlap: Overlap duration in seconds

        Returns:
            Merged segments without duplicates
        """
        if not segments:
            return []

        # Sort by start time
        segments = sorted(segments, key=lambda s: s["start"])

        merged = []
        for seg in segments:
            if not merged:
                merged.append(seg)
                continue

            prev = merged[-1]

            # Check for significant overlap
            overlap_start = max(prev["start"], seg["start"])
            overlap_end = min(prev["end"], seg["end"])
            overlap_duration = overlap_end - overlap_start

            if overlap_duration > overlap * 0.5:
                # Significant overlap - merge or pick one
                # Prefer the segment with more text (usually more accurate)
                if len(seg["text"]) > len(prev["text"]):
                    merged[-1] = seg
            else:
                merged.append(seg)

        return merged

    def load_draft_model(
        self,
        draft_model_name: str = "mlx-community/distil-whisper-large-v3",
        cache_dir: str | None = None,
        draft_tokens: int = 5,
        quantize_draft: int | None = 8,
    ):
        """
        Load draft model for speculative decoding (OPT-W5).

        The draft model (distil-whisper) has 2 decoder layers vs 32 in large-v3.
        It generates candidate tokens quickly for the main model to verify.

        Args:
            draft_model_name: HuggingFace model name for draft model
            cache_dir: Optional cache directory
            draft_tokens: Number of tokens to draft per iteration (default: 5)
            quantize_draft: Quantization bits for draft model weights (default: 8).
                            INT8 provides ~30% latency reduction for draft generation.
                            Since main model verifies outputs, accuracy degradation
                            is acceptable - wrong drafts are simply rejected.
                            Set to None to disable quantization.
        """
        # Load draft model with optimizations (CTC Plan Phase 0):
        # - INT8 weight quantization: ~30% draft latency reduction
        # - INT8 KV cache: 50% KV memory reduction
        # - JIT warmup: faster first inference
        # - preallocate_kv=True: required for quantize_kv to work
        self._draft_model = WhisperMLX.from_pretrained(
            draft_model_name,
            cache_dir=cache_dir,
            dtype=self.dtype,
            use_fused=self.use_fused,
            preallocate_kv=True,  # Enable for KV quantization
            quantize_kv=True,  # INT8 KV cache (lossless)
            warmup=True,  # JIT compilation warmup
            quantize=quantize_draft,  # INT8 weight quantization
        )

        # Create speculative decoder
        self._speculative_decoder = SpeculativeDecoder(
            main_model=self,
            draft_model=self._draft_model,
            draft_tokens=draft_tokens,
        )

    def transcribe_speculative(
        self,
        audio: str | numpy.ndarray | mx.array,
        *,
        language: str | None = None,
        task: str = "transcribe",
        variable_length: bool = False,
        temperature: float = 0.0,
        max_initial_timestamp: float = 1.0,
        verbose: bool = False,
    ) -> dict:
        """
        Transcribe audio using speculative decoding (OPT-W5).

        This method uses a draft model (distil-whisper) to generate candidate
        tokens, which are then verified by the main model. This can provide
        1.5-2x speedup for autoregressive decoding.

        NOTE: You must call load_draft_model() first.

        Args:
            audio: Audio file path or waveform array
            language: Language code (auto-detect if None)
            task: "transcribe" or "translate"
            variable_length: DEPRECATED - produces hallucinations, do not use.
            temperature: Sampling temperature (0 for greedy)
            max_initial_timestamp: Maximum timestamp at start
            verbose: Print progress

        Returns:
            Dictionary with "text", "segments", "language", "acceptance_rate"

        Warning:
            variable_length=True mode is BROKEN and produces hallucinations.
            Use standard mode (variable_length=False) for 100% accuracy.
        """
        import time
        import warnings

        import numpy as np

        # Deprecation warning for variable_length mode
        if variable_length:
            warnings.warn(
                "variable_length=True is DEPRECATED and produces hallucinations. "
                "Use variable_length=False (default) for accurate transcription.",
                DeprecationWarning,
                stacklevel=2,
            )

        if self._speculative_decoder is None:
            raise RuntimeError(
                "Draft model not loaded. Call load_draft_model() first.",
            )

        # Load audio if path
        if isinstance(audio, str):
            audio = load_audio(audio, sample_rate=SAMPLE_RATE)

        # Convert to numpy if needed
        if isinstance(audio, mx.array):
            audio = np.array(audio)

        # ===== VAD PREPROCESSING (P0.1 - ALWAYS ON for consistency) =====
        # Apply same VAD preprocessing as transcribe() for consistent outputs
        vad_result: VADResult | None = None
        try:
            speech_audio, vad_result = preprocess_audio_with_vad(
                audio,
                aggressiveness=2,  # Default aggressiveness
                padding_ms=50,
            )

            if verbose:
                print(f"VAD: {vad_result.speech_ratio:.1%} speech, "
                      f"{len(vad_result.segments)} segments")

            # If audio is mostly silent, return early
            if vad_result.is_mostly_silent:
                if verbose:
                    print("VAD: Audio is mostly silent, skipping transcription")
                return {
                    "text": "",
                    "segments": [],
                    "language": language or "en",
                    "acceptance_rate": 0.0,
                    "tokens_per_iteration": 0.0,
                    "is_silent": True,
                    "vad_speech_ratio": vad_result.speech_ratio,
                }

            # Use speech-only audio
            if len(speech_audio) > 0:
                audio = speech_audio

        except ImportError as e:
            if verbose:
                print(f"VAD: Silero VAD not available ({e}), using full audio")

        # ===== END VAD PREPROCESSING =====

        # Compute mel spectrogram with duration awareness
        actual_duration = None
        if variable_length:
            mel, encoder_positions, actual_duration = compute_mel_for_duration(
                audio, n_mels=self.config.n_mels, pad_to_30s=False,
            )
            self.decoder.set_precision(actual_duration, encoder_positions)
            # Also set for draft model
            self._draft_model.decoder.set_precision(actual_duration, encoder_positions)
            if verbose:
                print(f"Variable-length mode: {actual_duration:.2f}s audio, "
                      f"{encoder_positions} encoder positions")
        else:
            mel = log_mel_spectrogram(audio, n_mels=self.config.n_mels)
            target_len = self.config.n_audio_ctx * 2
            if mel.shape[0] < target_len:
                mel = mx.pad(mel, [(0, target_len - mel.shape[0]), (0, 0)])
            elif mel.shape[0] > target_len:
                mel = mel[:target_len, :]
            self.decoder.reset_precision()
            self._draft_model.decoder.reset_precision()

        mel = mel[None]

        # Encode audio with both models (with caching if enabled)
        if verbose:
            t0 = time.perf_counter()

        encoder_positions_for_cache = (
            encoder_positions if variable_length
            else self.config.n_audio_ctx
        )

        cache_hit = False
        if self._encoder_cache is not None:
            # Check cache for main model
            cached = self._encoder_cache.get(mel, variable_length)
            if cached is not None:
                audio_features = cached.audio_features
                cache_hit = True
                if verbose:
                    print("Main encoder cache HIT")

        if not cache_hit:
            audio_features = self.embed_audio(mel, variable_length=variable_length)
            if self._encoder_cache is not None:
                self._encoder_cache.put(
                    mel=mel,
                    audio_features=audio_features,
                    variable_length=variable_length,
                    encoder_positions=encoder_positions_for_cache,
                    audio_duration=actual_duration,
                )

        # OPTIMIZATION: Share encoder output with draft model
        # distil-whisper-large-v3 has IDENTICAL encoder architecture:
        # - Same 32 encoder layers, 1280 dim, 20 heads, 128 mels
        # - Only the decoder differs (2 layers vs 32)
        # Reusing encoder output eliminates ~90% of draft model overhead
        draft_audio_features = audio_features
        mx.eval(audio_features)

        if verbose:
            encode_time = time.perf_counter() - t0
            cache_str = " (main cached)" if cache_hit else ""
            print(f"Encoded audio in {encode_time*1000:.1f}ms{cache_str}")

        # Get tokenizer
        from .tokenizer import get_whisper_tokenizer
        is_multilingual = self.config.n_vocab >= 51865
        num_langs = self.config.n_vocab - 51765 - int(is_multilingual)
        tokenizer = get_whisper_tokenizer(
            multilingual=is_multilingual,
            num_languages=num_langs,
            language=language,
            task=task,
        )

        # Detect language if not specified
        if language is None:
            language = self._detect_language(audio_features, tokenizer)
            if verbose:
                print(f"Detected language: {language}")

        # Prepare decoding options
        options = DecodingOptions(
            temperature=temperature,
            max_initial_timestamp=max_initial_timestamp,
            suppress_blank=True,
            suppress_tokens="-1",
            without_timestamps=False,
        )

        # Get sample begin position
        sample_begin = len(tokenizer.sot_sequence)

        # Get timestamp precision
        precision = getattr(self.decoder, 'precision', 0.02)

        # Decode with speculative decoding
        if verbose:
            t0 = time.perf_counter()

        tokens, segments = self._speculative_decoder.decode(
            audio_features=audio_features,
            tokenizer=tokenizer,
            options=options,
            sample_begin=sample_begin,
            n_vocab=self.config.n_vocab,
            precision=precision,
            audio_duration=actual_duration,
            draft_audio_features=draft_audio_features,
        )

        if verbose:
            decode_time = time.perf_counter() - t0
            print(f"Decoded in {decode_time*1000:.1f}ms")
            print(f"Acceptance rate: {self._speculative_decoder.acceptance_rate:.1%}")
            tpi = self._speculative_decoder.tokens_per_iteration
            print(f"Tokens per iteration: {tpi:.2f}")

        # Decode text
        text = tokenizer.decode(tokens)

        return {
            "text": text,
            "segments": segments,
            "language": language,
            "acceptance_rate": self._speculative_decoder.acceptance_rate,
            "tokens_per_iteration": self._speculative_decoder.tokens_per_iteration,
        }

    # =========================================================================
    # Beam Search Decoding (P4)
    # =========================================================================

    def transcribe_beam(
        self,
        audio: str | numpy.ndarray | mx.array,
        *,
        language: str | None = None,
        task: str = "transcribe",
        beam_size: int = 5,
        length_penalty: float = 1.0,
        max_initial_timestamp: float = 1.0,
        verbose: bool = False,
        # VAD preprocessing (P0.1 - ALWAYS ON)
        vad_aggressiveness: int = 2,
    ) -> dict:
        """
        Transcribe audio using beam search decoding.

        Beam search maintains multiple hypotheses during decoding, selecting
        the best one based on log probability. This provides better quality
        than greedy decoding at the cost of beam_size * compute.

        Typical improvement: ~0.5-1% WER reduction with beam_size=5.

        Args:
            audio: Audio file path or waveform array
            language: Language code (auto-detect if None)
            task: "transcribe" or "translate"
            beam_size: Number of beams to maintain (default: 5)
            length_penalty: Length penalty for beam scoring (default: 1.0)
                           > 1.0 encourages longer sequences
                           < 1.0 encourages shorter sequences
            max_initial_timestamp: Maximum timestamp at start
            verbose: Print progress
            vad_aggressiveness: VAD aggressiveness level (0-3, default=2)

        Returns:
            Dictionary with "text", "segments", "language", quality metrics,
            and beam search metadata ("beam_size", "normalized_score").
        """
        import time

        import numpy as np

        from .beam_search import BeamSearchDecoder

        # Load audio if path
        if isinstance(audio, str):
            audio = load_audio(audio, sample_rate=SAMPLE_RATE)

        # Convert to numpy if needed
        if isinstance(audio, mx.array):
            audio = np.array(audio)

        # Store original audio duration for metadata
        original_duration = len(audio) / SAMPLE_RATE

        # ===== VAD PREPROCESSING (P0.1 - ALWAYS ON) =====
        vad_result: VADResult | None = None
        try:
            speech_audio, vad_result = preprocess_audio_with_vad(
                audio,
                aggressiveness=vad_aggressiveness,
                padding_ms=50,
            )

            if verbose:
                print(f"VAD: {vad_result.speech_ratio:.1%} speech, "
                      f"{len(vad_result.segments)} segments, "
                      f"{vad_result.speech_duration:.1f}s / {vad_result.total_duration:.1f}s")

            # If audio is mostly silent, return early
            if vad_result.is_mostly_silent:
                if verbose:
                    print("VAD: Audio is mostly silent, skipping transcription")
                return {
                    "text": "",
                    "segments": [],
                    "language": language or "en",
                    "avg_logprob": float("nan"),
                    "no_speech_prob": 1.0,
                    "beam_size": beam_size,
                    "normalized_score": 0.0,
                    "is_silent": True,
                    "vad_speech_ratio": vad_result.speech_ratio,
                }

            if len(speech_audio) > 0:
                audio = speech_audio

        except ImportError as e:
            if verbose:
                print(f"VAD: Silero VAD not available ({e}), using full audio")

        # Check for silent audio
        from .audio import is_silent_audio

        if is_silent_audio(audio):
            return {
                "text": "",
                "segments": [],
                "language": language or "en",
                "avg_logprob": float("nan"),
                "no_speech_prob": 1.0,
                "beam_size": beam_size,
                "normalized_score": 0.0,
                "is_silent": True,
                "vad_speech_ratio": vad_result.speech_ratio if vad_result else None,
            }

        # Compute mel spectrogram (standard 30s mode for beam search)
        mel = log_mel_spectrogram(audio, n_mels=self.config.n_mels)
        target_len = self.config.n_audio_ctx * 2
        if mel.shape[0] < target_len:
            mel = mx.pad(mel, [(0, target_len - mel.shape[0]), (0, 0)])
        elif mel.shape[0] > target_len:
            mel = mel[:target_len, :]
        self.decoder.reset_precision()
        mel = mel[None]

        # Encode audio
        t0 = time.perf_counter()
        audio_features = self.embed_audio(mel, variable_length=False)

        if verbose:
            encode_time = time.perf_counter() - t0
            print(f"Encoded in {encode_time*1000:.1f}ms")

        # Get tokenizer
        from .tokenizer import get_whisper_tokenizer
        is_multilingual = self.config.n_vocab >= 51865
        num_langs = self.config.n_vocab - 51765 - int(is_multilingual)
        tokenizer = get_whisper_tokenizer(
            multilingual=is_multilingual,
            num_languages=num_langs,
            language=language,
            task=task,
        )

        # Detect language if not specified
        if language is None:
            language = self._detect_language(audio_features, tokenizer)
            if verbose:
                print(f"Detected language: {language}")

        # Create beam search decoder
        beam_decoder = BeamSearchDecoder(
            model=self,
            beam_size=beam_size,
            length_penalty=length_penalty,
        )

        # Build decoding options
        options = DecodingOptions(
            temperature=0.0,  # Beam search uses greedy selection within beams
            max_initial_timestamp=max_initial_timestamp,
            suppress_blank=True,
            suppress_tokens="-1",
            without_timestamps=False,
        )

        sample_begin = len(list(tokenizer.sot_sequence))

        # Decode with beam search
        t0 = time.perf_counter()
        result = beam_decoder.decode(
            audio_features=audio_features,
            tokenizer=tokenizer,
            options=options,
            sample_begin=sample_begin,
            n_vocab=self.config.n_vocab,
            precision=getattr(self.decoder, 'precision', 0.02),
        )

        if verbose:
            decode_time = time.perf_counter() - t0
            print(f"Decoded in {decode_time*1000:.1f}ms with beam_size={beam_size}")
            print(f"Final beams explored: {beam_decoder.final_beam_count}")
            print(f"Decode steps: {beam_decoder.decode_steps}")

        # Parse segments from timestamp tokens
        from .beam_search import _parse_segments
        segments = _parse_segments(
            result.tokens,
            tokenizer,
            getattr(self.decoder, 'precision', 0.02),
        )

        return {
            "text": result.text,
            "segments": segments,
            "language": language,
            # Quality metrics
            "avg_logprob": result.avg_logprob,
            "no_speech_prob": result.no_speech_prob,
            # Beam search metadata
            "beam_size": beam_size,
            "normalized_score": result.normalized_score,
            "log_prob": result.log_prob,
            # VAD metadata
            "vad_speech_ratio": vad_result.speech_ratio if vad_result else None,
            "vad_original_duration": original_duration,
        }

    # =========================================================================
    # Medusa Multi-Token Prediction (Phase 2.4)
    # =========================================================================

    def load_medusa_heads(
        self,
        weights_path: str,
        n_heads: int = 5,
        use_block: bool = False,
        use_aiola: bool = False,
        tree_structure: list[int] | None = None,
        top_k: int = 5,
    ):
        """
        Load Medusa heads for multi-token prediction.

        Medusa adds extra prediction heads that speculate multiple future tokens
        in parallel. This enables 1.5-2.5x speedup for autoregressive decoding.

        Args:
            weights_path: Path to Medusa weights file (.npz format)
            n_heads: Number of Medusa heads (must match saved weights)
            use_block: Use Medusa-Block variant (must match saved weights)
            use_aiola: Use aiola-compatible architecture (residual block + shared proj)
                      Set this to True for aiola/whisper-medusa-v1 weights
            tree_structure: Candidates per tree level (default: [5, 4, 3, 2, 1])
            top_k: Number of top-k candidates per head

        Example:
            # For self-trained weights:
            model = WhisperMLX.from_pretrained("large-v3")
            model.load_medusa_heads("medusa_large_v3.npz")

            # For aiola/whisper-medusa-v1 weights:
            model = WhisperMLX.from_pretrained("large-v2")  # Must use v2!
            model.load_medusa_heads("medusa_aiola_v1.npz", n_heads=10, use_aiola=True)

            # Medusa decoding
            result = model.transcribe_medusa("audio.wav")
            print(f"Speedup: {result['tokens_per_step']:.2f}x")
        """
        from .medusa_training import load_medusa_weights

        # For aiola variant, create wrapper for decoder's vocab projection
        proj_out = None
        if use_aiola:
            # Create a callable wrapper for token_embedding.as_linear()
            # This allows Medusa heads to share the decoder's vocabulary projection
            class ProjOutWrapper(nn.Module):
                def __init__(self, token_embedding):
                    super().__init__()
                    self._token_embedding = token_embedding

                def __call__(self, x):
                    return self._token_embedding.as_linear(x)

            proj_out = ProjOutWrapper(self.decoder.token_embedding)

        # Create Medusa module
        self._medusa_module = create_medusa_module(
            self.config,
            n_heads=n_heads,
            use_block=use_block,
            use_aiola=use_aiola,
            proj_out=proj_out,
            dtype=self.dtype,
        )

        # Load weights
        load_medusa_weights(self._medusa_module, weights_path)
        mx.eval(self._medusa_module.parameters())

        # Create tree verifier
        self._medusa_verifier = MedusaTreeVerifier(
            decoder=self.decoder,
            medusa_module=self._medusa_module,
            tree_structure=tree_structure or [5, 4, 3, 2, 1],
            top_k=top_k,
        )

    def unload_medusa_heads(self):
        """Unload Medusa heads to free memory."""
        self._medusa_module = None
        self._medusa_verifier = None

    @property
    def medusa_loaded(self) -> bool:
        """Whether Medusa heads are loaded."""
        return self._medusa_module is not None

    def transcribe_medusa(
        self,
        audio: str | numpy.ndarray | mx.array,
        *,
        language: str | None = None,
        task: str = "transcribe",
        temperature: float = 0.0,
        max_initial_timestamp: float = 1.0,
        verbose: bool = False,
    ) -> dict:
        """
        Transcribe audio using Medusa multi-token prediction.

        Medusa generates multiple candidate tokens per step and verifies them
        in parallel using tree attention. This achieves 1.5-2.5x speedup
        while maintaining lossless quality.

        NOTE: You must call load_medusa_heads() first with trained weights.

        Args:
            audio: Audio file path or waveform array
            language: Language code (auto-detect if None)
            task: "transcribe" or "translate"
            temperature: Sampling temperature (0 for greedy)
            max_initial_timestamp: Maximum timestamp at start
            verbose: Print progress

        Returns:
            Dictionary with:
            - "text": Transcribed text
            - "segments": List of segments with timestamps
            - "language": Detected or specified language
            - "acceptance_rate": Fraction of proposed tokens accepted
            - "tokens_per_step": Average tokens accepted per decode step

        Example:
            model = WhisperMLX.from_pretrained("large-v3")
            model.load_medusa_heads("medusa_weights.npz")

            result = model.transcribe_medusa("audio.wav")
            print(result["text"])
            print(f"Acceptance rate: {result['acceptance_rate']:.1%}")
        """
        import time

        import numpy as np

        if self._medusa_verifier is None:
            raise RuntimeError(
                "Medusa heads not loaded. Call load_medusa_heads() first.",
            )

        # Load audio if path
        if isinstance(audio, str):
            audio = load_audio(audio, sample_rate=SAMPLE_RATE)

        # Convert to numpy if needed
        if isinstance(audio, mx.array):
            audio = np.array(audio)

        # Check for silent audio
        from .audio import get_audio_rms_db, is_silent_audio

        if is_silent_audio(audio):
            if verbose:
                rms_db = get_audio_rms_db(audio)
                print(f"Silent audio detected (RMS: {rms_db:.1f} dB < -40 dB)")
            return {
                "text": "",
                "segments": [],
                "language": language or "en",
                "acceptance_rate": 0.0,
                "tokens_per_step": 0.0,
                "is_silent": True,
            }

        # Compute mel spectrogram (standard 30s mode)
        mel = log_mel_spectrogram(audio, n_mels=self.config.n_mels)
        target_len = self.config.n_audio_ctx * 2
        if mel.shape[0] < target_len:
            mel = mx.pad(mel, [(0, target_len - mel.shape[0]), (0, 0)])
        elif mel.shape[0] > target_len:
            mel = mel[:target_len, :]
        self.decoder.reset_precision()

        mel = mel[None]

        # Encode audio
        if verbose:
            t0 = time.perf_counter()

        audio_features = self.embed_audio(mel, variable_length=False)
        mx.eval(audio_features)

        if verbose:
            encode_time = time.perf_counter() - t0
            print(f"Encoded audio in {encode_time*1000:.1f}ms")

        # Get tokenizer
        from .tokenizer import get_whisper_tokenizer
        is_multilingual = self.config.n_vocab >= 51865
        num_langs = self.config.n_vocab - 51765 - int(is_multilingual)
        tokenizer = get_whisper_tokenizer(
            multilingual=is_multilingual,
            num_languages=num_langs,
            language=language,
            task=task,
        )

        # Detect language if not specified
        if language is None:
            language = self._detect_language(audio_features, tokenizer)
            if verbose:
                print(f"Detected language: {language}")

        # Reset verifier statistics
        self._medusa_verifier.reset_stats()

        # Decode using Medusa
        if verbose:
            t0 = time.perf_counter()

        tokens, segments = self._decode_medusa(
            audio_features,
            tokenizer,
            temperature=temperature,
            max_initial_timestamp=max_initial_timestamp,
        )

        if verbose:
            decode_time = time.perf_counter() - t0
            print(f"Decoded in {decode_time*1000:.1f}ms")
            print(f"Acceptance rate: {self._medusa_verifier.acceptance_rate:.1%}")
            tpi = self._medusa_verifier.tokens_per_iteration
            print(f"Tokens per step: {tpi:.2f}")

        # Decode text
        text = tokenizer.decode(tokens)

        return {
            "text": text,
            "segments": segments,
            "language": language,
            "acceptance_rate": self._medusa_verifier.acceptance_rate,
            "tokens_per_step": self._medusa_verifier.tokens_per_iteration,
        }

    def _decode_medusa(
        self,
        audio_features: mx.array,
        tokenizer,
        temperature: float = 0.0,
        max_initial_timestamp: float = 1.0,
    ) -> tuple[list[int], list[dict]]:
        """
        Decode tokens using Medusa multi-token prediction.

        The algorithm:
        1. Generate initial token with main decoder
        2. Get Medusa head predictions for multiple future tokens
        3. Build tree of candidate sequences
        4. Verify all candidates in parallel with tree attention
        5. Accept longest valid prefix
        6. Repeat until EOT

        Args:
            audio_features: Encoded audio from encoder
            tokenizer: Whisper tokenizer
            temperature: Sampling temperature (0 for greedy)
            max_initial_timestamp: Maximum timestamp at start

        Returns:
            Tuple of (token_list, segments_list)
        """
        # Build initial tokens (SOT sequence)
        initial_tokens = list(tokenizer.sot_sequence)
        tokens = mx.array([initial_tokens])
        sample_begin = len(initial_tokens)

        # Get timestamp precision
        precision = getattr(self.decoder, 'precision', 0.02)

        # Build logit filters
        options = DecodingOptions(
            temperature=temperature,
            max_initial_timestamp=max_initial_timestamp,
            suppress_blank=True,
            suppress_tokens="-1",
            without_timestamps=False,
        )
        logit_filters = build_logit_filters(
            tokenizer,
            options,
            sample_begin=sample_begin,
            n_vocab=self.config.n_vocab,
            precision=precision,
        )

        # Process initial tokens through decoder to initialize KV cache
        logits, kv_cache, _, hidden_states = self.decoder(
            tokens, audio_features, kv_cache=None, return_hidden=True,
        )
        mx.eval(logits, kv_cache, hidden_states)

        # Apply logit filters and get first token
        logits = logits[:, -1].astype(mx.float32)
        filtered_logits = apply_filters(logits, tokens, logit_filters)

        if temperature == 0:
            next_token = greedy_decode(filtered_logits)
        else:
            next_token = sample_with_temperature(filtered_logits, temperature)

        next_token_int = int(next_token[0])

        # Collect all tokens
        all_tokens = list(initial_tokens)
        all_tokens.append(next_token_int)

        # Check for immediate EOT
        if next_token_int == tokenizer.eot:
            output_tokens = all_tokens[sample_begin:]
            segments = self._parse_segments(output_tokens, tokenizer, precision)
            return output_tokens, segments

        # Update KV cache with first token
        tokens = mx.array([[next_token_int]])
        _, kv_cache, _, _ = self.decoder(tokens, audio_features, kv_cache=kv_cache)
        mx.eval(kv_cache)

        # Medusa decode loop
        max_tokens = self.config.n_text_ctx // 2
        n_ctx = self.config.n_text_ctx
        for _ in range(max_tokens):
            # Check context length limit before decoding
            # Positional embeddings only exist for positions 0 to n_ctx-1
            prefix_len = kv_cache[0][0][0].shape[1] if kv_cache else 0
            if prefix_len >= n_ctx - 1:
                # Context is full, cannot add more tokens
                # Return what we have so far
                break

            # Get decoder output with hidden states for Medusa
            logits, _, _, hidden_states = self.decoder(
                tokens, audio_features, kv_cache=kv_cache, return_hidden=True,
            )
            mx.eval(logits, hidden_states)

            # Generate tree candidates
            tree_tokens, tree_mask, tree_paths, node_depths = self._medusa_verifier.generate_tree_candidates(
                hidden_states, logits,
            )
            mx.eval(tree_tokens, tree_mask)

            # Verify candidates in parallel
            accepted_tokens, n_accepted, kv_cache = self._medusa_verifier.verify_tree_parallel(
                tree_tokens, tree_mask, tree_paths, audio_features, kv_cache, node_depths,
            )
            mx.eval(accepted_tokens, kv_cache)

            # Add accepted tokens
            for i in range(n_accepted):
                token_int = int(accepted_tokens[0, i])

                # Check for EOT
                if token_int == tokenizer.eot:
                    output_tokens = all_tokens[sample_begin:]
                    segments = self._parse_segments(output_tokens, tokenizer, precision)
                    return output_tokens, segments

                all_tokens.append(token_int)

            # Prepare for next iteration
            # Use last accepted token for next step
            tokens = accepted_tokens[:, -1:]

        # Max tokens reached
        output_tokens = all_tokens[sample_begin:]
        segments = self._parse_segments(output_tokens, tokenizer, precision)
        return output_tokens, segments

    @property
    def is_multilingual(self) -> bool:
        """Whether model supports multiple languages."""
        return self.config.n_vocab >= 51865

    @property
    def num_languages(self) -> int:
        """Number of supported languages."""
        return self.config.n_vocab - 51765 - int(self.is_multilingual)

    # =========================================================================
    # Encoder Cache Methods (OPT-W4)
    # =========================================================================

    def enable_encoder_cache(
        self,
        max_entries: int = 16,
        max_memory_mb: float | None = None,
    ):
        """
        Enable encoder output caching for repeated queries (OPT-W4).

        This provides ~2x speedup when the same audio is processed multiple times,
        such as:
        - Language detection + transcription
        - Multiple transcription attempts with different parameters
        - A/B testing decoding strategies

        Memory usage (per entry, large-v3):
        - Standard mode (30s): ~3.8 MB
        - Variable-length (varies): ~0.1-3.8 MB

        Args:
            max_entries: Maximum cached entries (LRU eviction). Default 16.
            max_memory_mb: Optional memory limit in MB.

        Example:
            model = WhisperMLX.from_pretrained("large-v3")
            model.enable_encoder_cache(max_entries=16)

            # First call encodes audio
            result1 = model.transcribe("audio.wav", language="en")

            # Second call uses cached encoder output (2x faster)
            result2 = model.transcribe("audio.wav", language="ja")
        """
        self._encoder_cache = EncoderCache(
            max_entries=max_entries,
            max_memory_mb=max_memory_mb,
        )

    def disable_encoder_cache(self):
        """Disable encoder caching and free cached memory."""
        if self._encoder_cache is not None:
            self._encoder_cache.clear()
            self._encoder_cache = None

    def clear_encoder_cache(self):
        """Clear all cached encoder outputs without disabling caching."""
        if self._encoder_cache is not None:
            self._encoder_cache.clear()

    def get_encoder_cache_stats(self) -> dict | None:
        """
        Get encoder cache statistics.

        Returns:
            Dictionary with cache stats, or None if cache not enabled.
            Keys: entries, max_entries, hits, misses, hit_rate, evictions, memory_mb
        """
        if self._encoder_cache is not None:
            return self._encoder_cache.stats
        return None

    @property
    def encoder_cache_enabled(self) -> bool:
        """Whether encoder caching is enabled."""
        return self._encoder_cache is not None

    # =========================================================================
    # Encoder VAD Head Methods (Phase 3 Optimization)
    # =========================================================================

    def load_encoder_vad_head(
        self,
        weights_path: str,
        hidden_dim: int = 256,
        threshold: float = 0.15,
    ):
        """
        Load encoder VAD head for skipping decoder on silent frames.

        The encoder VAD head predicts speech probability for each encoder
        position. During transcription, positions below the threshold are
        treated as silence and the decoder is skipped for those frames.

        Expected speedup: ~1.2x overall by skipping ~20% silent frames.

        Args:
            weights_path: Path to trained VAD head weights (.npz or .safetensors)
            hidden_dim: Hidden dimension of the VAD head (must match saved weights)
            threshold: Speech probability threshold (0.15 recommended).
                      Lower values include more frames (more conservative).
                      Higher values skip more frames (more aggressive).
                      Note: Model outputs lower probabilities than Silero VAD,
                      so 0.1-0.2 works better than 0.5.

        Example:
            model = WhisperMLX.from_pretrained("large-v3")
            model.load_encoder_vad_head("encoder_vad_best.npz")
            result = model.transcribe("audio.wav")  # Automatically uses VAD
        """
        self._encoder_vad_head = load_encoder_vad_head(
            weights_path,
            n_state=self.config.n_audio_state,
            hidden_dim=hidden_dim,
            dtype=self.dtype,
        )
        self._encoder_vad_threshold = threshold
        mx.eval(self._encoder_vad_head.parameters())

    def unload_encoder_vad_head(self):
        """Unload encoder VAD head to free memory."""
        self._encoder_vad_head = None

    def set_encoder_vad_threshold(self, threshold: float):
        """
        Set the speech probability threshold for encoder VAD.

        Args:
            threshold: Probability threshold (0.0-1.0).
                      Lower values include more frames (conservative).
                      Higher values skip more frames (aggressive).
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be in [0, 1], got {threshold}")
        self._encoder_vad_threshold = threshold

    @property
    def encoder_vad_enabled(self) -> bool:
        """Whether encoder VAD head is loaded."""
        return self._encoder_vad_head is not None

    def get_encoder_vad_mask(
        self,
        audio_features: mx.array,
        threshold: float | None = None,
    ) -> tuple[mx.array, float]:
        """
        Get VAD mask for encoder features.

        Args:
            audio_features: Encoder output (batch, seq_len, n_state)
            threshold: Override threshold (uses default if None)

        Returns:
            Tuple of (speech_mask, speech_ratio):
            - speech_mask: Boolean mask (batch, seq_len), True = speech
            - speech_ratio: Fraction of frames detected as speech
        """
        if self._encoder_vad_head is None:
            raise RuntimeError("Encoder VAD head not loaded. Call load_encoder_vad_head() first.")

        thresh = threshold if threshold is not None else self._encoder_vad_threshold
        speech_mask = self._encoder_vad_head.get_speech_mask(audio_features, threshold=thresh)
        mx.eval(speech_mask)

        speech_ratio = float(mx.mean(speech_mask.astype(mx.float32)))
        return speech_mask, speech_ratio

    # =========================================================================
    # Optimization Presets (OPT-2.5)
    # =========================================================================

    @classmethod
    def from_preset(
        cls,
        preset: OptimizationPreset,
        cache_dir: str | None = None,
        warmup: bool = True,
    ) -> WhisperMLX:
        """
        Create a WhisperMLX model configured for a specific optimization preset.

        This is the recommended way to create a model with quality/speed tradeoffs.
        The preset determines:
        - Which model variant to use (large-v3, turbo, distil)
        - Weight quantization (FP16, INT8, INT4)
        - KV cache quantization

        Args:
            preset: One of OptimizationPreset (MAX_QUALITY, BALANCED, FAST, ULTRA_FAST)
            cache_dir: Optional cache directory for model download
            warmup: Run warmup inference (default True)

        Returns:
            WhisperMLX model configured for the preset

        Example:
            from tools.whisper_mlx import WhisperMLX, OptimizationPreset

            # For balanced speed/quality (recommended for most uses)
            model = WhisperMLX.from_preset(OptimizationPreset.BALANCED)
            result = model.transcribe("audio.wav")

            # For maximum speed (real-time applications)
            model = WhisperMLX.from_preset(OptimizationPreset.ULTRA_FAST)
        """
        from .presets import TranscriptionConfig

        config = TranscriptionConfig.from_preset(preset)

        return cls.from_pretrained(
            model_name=config.get_model_name(),
            cache_dir=cache_dir,
            quantize_kv=config.quantize_kv,
            quantize=config.weight_bits,
            warmup=warmup,
        )

    def transcribe_with_config(
        self,
        audio: str | numpy.ndarray | mx.array,
        config: TranscriptionConfig,
    ) -> dict:
        """
        Transcribe audio using a TranscriptionConfig.

        This method applies all the optimization dials from the config
        to the transcription. Use this when you want fine-grained control
        over the quality/speed tradeoffs.

        Note: The model variant and weight quantization from config are NOT
        applied here (they affect model loading). Use from_preset() to
        create a model with the right variant, then use this method for
        transcription settings.

        Args:
            audio: Audio file path or waveform array
            config: TranscriptionConfig with dial settings

        Returns:
            Dictionary with "text", "segments", "language", and quality metrics

        Example:
            from tools.whisper_mlx import WhisperMLX, TranscriptionConfig, OptimizationPreset

            model = WhisperMLX.from_preset(OptimizationPreset.BALANCED)
            config = TranscriptionConfig.from_preset(OptimizationPreset.BALANCED)
            config.language = "en"  # Override to skip language detection

            result = model.transcribe_with_config("audio.wav", config)
        """
        return self.transcribe(audio, **config.get_transcribe_kwargs())

    def transcribe_long_with_config(
        self,
        audio: str | numpy.ndarray | mx.array,
        config: TranscriptionConfig,
        batch_size: int = 4,
        chunk_length: float = 30.0,
        overlap: float = 1.0,
    ) -> dict:
        """
        Transcribe long audio using a TranscriptionConfig.

        Like transcribe_with_config but for audio longer than 30 seconds.

        Args:
            audio: Audio file path or waveform array
            config: TranscriptionConfig with dial settings
            batch_size: Number of chunks to encode in parallel
            chunk_length: Length of each chunk in seconds
            overlap: Overlap between chunks in seconds

        Returns:
            Dictionary with "text", "segments", "language"

        Example:
            model = WhisperMLX.from_preset(OptimizationPreset.FAST)
            config = TranscriptionConfig.from_preset(OptimizationPreset.FAST)

            # Transcribe a 10-minute audio file
            result = model.transcribe_long_with_config("long_audio.wav", config)
        """
        kwargs = config.get_transcribe_long_kwargs()
        return self.transcribe_long(
            audio,
            batch_size=batch_size,
            chunk_length=chunk_length,
            overlap=overlap,
            **kwargs,
        )
