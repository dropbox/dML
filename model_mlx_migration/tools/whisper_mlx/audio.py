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
Audio processing for WhisperMLX.

Handles mel spectrogram computation with support for:
- Variable-length audio (no padding to 30s)
- Efficient numpy/MLX operations
- Optional Metal acceleration (future)
"""

import concurrent.futures
from functools import lru_cache
from typing import Union

import numpy as np

try:
    import mlx.core as mx
except ImportError:
    mx = None


# Audio constants
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30  # Standard Whisper chunk in seconds
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples for 30s

# Flag to track if av logging has been configured
_av_logging_configured = False


def _configure_av_logging():
    """
    Configure PyAV logging to suppress noisy codec warnings.

    PyAV/ffmpeg emits many warnings for common audio formats (opus timestamp
    warnings, mp3 duration estimates, ogg comment headers) that are not
    actionable and clutter training logs. This sets the logging level to
    ERROR to only show actual errors.
    """
    global _av_logging_configured
    if _av_logging_configured:
        return

    try:
        import av
        # Suppress INFO and WARNING level messages from libav
        # Only show ERROR and FATAL messages
        av.logging.set_level(av.logging.ERROR)
        _av_logging_configured = True
    except (ImportError, AttributeError):
        # PyAV not available or doesn't support logging configuration
        pass


def load_audio(
    file_path: str,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Load audio file and convert to mono float32 at target sample rate.

    Supports:
    - Regular file paths
    - tar:// protocol for TAR-archived audio: tar:///path/to/archive.tar#inner/path.flac

    Backend priority (fastest to slowest):
    1. Native C++ loader (OPT-W3-C): 6-7x faster than ffmpeg subprocess
    2. PyAV (OPT-W3-B): 10-50x faster for short audio via library calls
    3. ffmpeg subprocess: Reference implementation, always available

    All backends use libav resampling for identical output (within 1 LSB).

    Args:
        file_path: Path to audio file (or tar:// URL for TAR archives)
        sample_rate: Target sample rate (default: 16000)

    Returns:
        Audio waveform as float32 numpy array, shape (n_samples,)
    """
    global _use_native, _use_pyav

    # Handle tar:// protocol for TAR-archived audio
    if file_path.startswith("tar://"):
        return _load_audio_from_tar(file_path, sample_rate)

    # Try native C++ loader first (fastest)
    use_native = _use_native
    if use_native is None:
        use_native = _is_native_available()

    if use_native:
        try:
            return _load_audio_native_cpp(file_path, sample_rate)
        except Exception as e:
            import warnings
            warnings.warn(f"Native C++ loader failed, trying next backend: {e}", stacklevel=2)
            # Fall through to PyAV/ffmpeg

    # Try PyAV next
    use_pyav = _use_pyav
    if use_pyav is None:
        use_pyav = _is_pyav_available()

    if use_pyav:
        try:
            return _load_audio_pyav(file_path, sample_rate)
        except Exception as e:
            import warnings
            warnings.warn(f"PyAV failed, falling back to ffmpeg: {e}", stacklevel=2)
            return _load_audio_ffmpeg(file_path, sample_rate)
    else:
        return _load_audio_ffmpeg(file_path, sample_rate)


def get_audio_duration(file_path: str) -> float:
    """
    Get audio file duration in seconds without loading the entire file.

    Uses soundfile.info() for efficient metadata-only access.
    Falls back to ffprobe for unsupported formats.

    Args:
        file_path: Path to audio file

    Returns:
        Duration in seconds, or 0.0 if unable to determine
    """
    try:
        import soundfile as sf
        info = sf.info(file_path)
        return info.duration
    except Exception:
        # Fall back to ffprobe for formats soundfile doesn't support
        try:
            import subprocess
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    file_path,
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
        except Exception:
            pass
        return 0.0


def _load_audio_native(
    file_path: str,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Load audio using soundfile + scipy resampling.

    OPT-W3: Much faster than ffmpeg subprocess for WAV/FLAC files:
    - 16kHz WAV: 24-236x faster (no subprocess overhead)
    - 44.1kHz WAV: 3-45x faster (efficient resampling)
    """
    import soundfile as sf
    from scipy import signal

    # Load audio with soundfile
    audio, file_sr = sf.read(file_path, dtype='float32')

    # Convert stereo to mono if needed
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if file_sr != sample_rate:
        num_samples = int(len(audio) * sample_rate / file_sr)
        audio = signal.resample(audio, num_samples).astype(np.float32)

    return audio


def _load_audio_ffmpeg(
    file_path: str,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Load audio using ffmpeg subprocess.

    CRITICAL: Uses EXACTLY the same ffmpeg command as mlx-whisper for bit-exact output.
    Previous version used different flags which caused sample count differences.

    The key differences from the old version:
    - Uses "-threads 0" for deterministic multi-threading
    - Uses s16le format (16-bit signed) instead of f32le
    - Divides by 32768.0 for float conversion (same as mlx-whisper)
    """
    import subprocess

    # EXACT same command as mlx-whisper - do not modify without validation
    cmd = [
        "ffmpeg", "-nostdin", "-i", file_path,
        "-threads", "0",
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, check=True)
        # CRITICAL: Same conversion as mlx-whisper
        return np.frombuffer(result.stdout, np.int16).flatten().astype(np.float32) / 32768.0
    except FileNotFoundError:
        raise ImportError("ffmpeg is required: brew install ffmpeg") from None
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg error: {e.stderr.decode()}") from e


# Cache for TAR file handles to avoid reopening for each sample
_tar_cache: dict = {}
_TAR_CACHE_MAX_SIZE = 10


def _load_audio_from_tar(
    tar_url: str,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Load audio from a TAR archive using tar:// protocol.

    Format: tar:///path/to/archive.tar#inner/path/to/file.flac

    Uses cached TAR handles for efficiency when loading many files from same archive.

    Args:
        tar_url: TAR URL in format tar://archive_path#inner_path
        sample_rate: Target sample rate (default: 16000)

    Returns:
        Audio waveform as float32 numpy array
    """
    import io
    import tarfile

    # Parse tar:// URL
    # Format: tar:///path/to/archive.tar#inner/path/file.flac
    if not tar_url.startswith("tar://"):
        raise ValueError(f"Invalid tar URL: {tar_url}")

    url_path = tar_url[6:]  # Remove "tar://"
    if "#" not in url_path:
        raise ValueError(f"Invalid tar URL (missing #): {tar_url}")

    tar_path, inner_path = url_path.split("#", 1)

    # Get or create cached TAR handle
    global _tar_cache
    if tar_path not in _tar_cache:
        # Evict oldest entry if cache is full
        if len(_tar_cache) >= _TAR_CACHE_MAX_SIZE:
            oldest_key = next(iter(_tar_cache))
            _tar_cache[oldest_key].close()
            del _tar_cache[oldest_key]
        _tar_cache[tar_path] = tarfile.open(tar_path, "r")

    tar = _tar_cache[tar_path]

    # Extract audio file to memory
    try:
        member = tar.getmember(inner_path)
        f = tar.extractfile(member)
        if f is None:
            raise RuntimeError(f"Could not extract {inner_path} from {tar_path}")
        audio_bytes = f.read()
    except KeyError:
        raise RuntimeError(f"File not found in TAR: {inner_path}") from None

    # Load audio from bytes using soundfile (works with FLAC, WAV, etc.)
    import soundfile as sf
    from scipy import signal

    audio, file_sr = sf.read(io.BytesIO(audio_bytes), dtype='float32')

    # Convert stereo to mono if needed
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if file_sr != sample_rate:
        num_samples = int(len(audio) * sample_rate / file_sr)
        audio = signal.resample(audio, num_samples).astype(np.float32)

    return audio


def _load_audio_pyav(
    file_path: str,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Load audio using PyAV (ffmpeg as library, no subprocess).

    OPT-W3-B: Eliminates ~45ms subprocess spawn overhead.
    Uses exact same libav resampling as ffmpeg CLI for bit-exact output.

    Expected speedup: 10-50x for short audio (2-5s)

    Args:
        file_path: Path to audio file
        sample_rate: Target sample rate (default: 16000)

    Returns:
        Audio waveform as float32 numpy array, shape (n_samples,)
    """
    # Configure av logging to suppress noisy codec warnings
    _configure_av_logging()

    import av

    container = av.open(file_path)

    # Set up resampler to match ffmpeg CLI behavior exactly:
    # -f s16le -ac 1 -acodec pcm_s16le -ar 16000
    resampler = av.AudioResampler(
        format='s16',       # pcm_s16le (16-bit signed)
        layout='mono',      # -ac 1
        rate=sample_rate,   # -ar 16000
    )

    frames = []
    for frame in container.decode(audio=0):
        # Resample frame to target format
        resampled_frames = resampler.resample(frame)
        for resampled in resampled_frames:
            # Convert to numpy array (int16)
            arr = resampled.to_ndarray()
            # Flatten in case of multi-channel (shouldn't happen with mono)
            frames.append(arr.flatten())

    # Flush the resampler to get any remaining buffered samples
    # This is critical for bit-exact match with ffmpeg CLI
    flushed_frames = resampler.resample(None)
    if flushed_frames:
        for resampled in flushed_frames:
            arr = resampled.to_ndarray()
            frames.append(arr.flatten())

    container.close()

    if not frames:
        return np.array([], dtype=np.float32)

    # Concatenate all frames
    audio = np.concatenate(frames)

    # CRITICAL: Same conversion as mlx-whisper (divide by 32768.0)
    return audio.astype(np.float32) / 32768.0


# Global flags to control audio loading backend
_use_native = None  # None = auto-detect, True = force native, False = skip native
_use_pyav = None  # None = auto-detect, True = force PyAV, False = force ffmpeg


def set_audio_backend(backend: str) -> None:
    """
    Set the audio loading backend.

    Args:
        backend: One of 'auto', 'native', 'pyav', 'ffmpeg'
            - 'auto': Use native if available, then PyAV, then ffmpeg
            - 'native': Force native C++ loader (raises ImportError if not built)
            - 'pyav': Force PyAV (raises ImportError if not available)
            - 'ffmpeg': Force ffmpeg subprocess
    """
    global _use_native, _use_pyav
    if backend == 'auto':
        _use_native = None
        _use_pyav = None
    elif backend == 'native':
        # Verify native loader is available
        if not _is_native_available():
            raise ImportError(
                "Native C++ audio loader not available. Build instructions:\n"
                "  cd tools/mlx_audio\n"
                "  mkdir build && cd build\n"
                "  cmake .. -DCMAKE_BUILD_TYPE=Release\n"
                "  make -j\n"
                "  cp mlx_audio_native*.so ../whisper_audio_native/",
            )
        _use_native = True
        _use_pyav = False  # Skip PyAV when native is forced
    elif backend == 'pyav':
        # Verify PyAV is available
        try:
            import av  # noqa: F401
            _use_native = False
            _use_pyav = True
        except ImportError:
            raise ImportError("PyAV not available: pip install av") from None
    elif backend == 'ffmpeg':
        _use_native = False
        _use_pyav = False
    else:
        raise ValueError(
            f"Unknown backend: {backend}. Use 'auto', 'native', 'pyav', or 'ffmpeg'",
        )


def get_audio_backend() -> str:
    """Get the current audio loading backend."""
    if _use_native is None and _use_pyav is None:
        return 'auto'
    if _use_native:
        return 'native'
    if _use_pyav:
        return 'pyav'
    return 'ffmpeg'


def _is_pyav_available() -> bool:
    """Check if PyAV is available."""
    try:
        import av  # noqa: F401
        return True
    except ImportError:
        return False


def _is_native_available() -> bool:
    """Check if native C++ audio loader is available."""
    try:
        import os
        import sys

        # Add mlx_audio parent directory to sys.path so whisper_audio_native can be imported
        mlx_audio_parent = os.path.join(os.path.dirname(__file__), '..', 'mlx_audio')
        mlx_audio_parent = os.path.abspath(mlx_audio_parent)
        if mlx_audio_parent not in sys.path:
            sys.path.insert(0, mlx_audio_parent)

        import whisper_audio_native  # noqa: F401
        return True
    except ImportError:
        return False


def _load_audio_native_cpp(
    file_path: str,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Load audio using native C++ loader (libav-based).

    OPT-W3-C: Eliminates subprocess AND Python FFI overhead.
    Direct C++ implementation using libav libraries.

    Expected speedup: 6-7x vs ffmpeg subprocess

    Args:
        file_path: Path to audio file
        sample_rate: Target sample rate (default: 16000)

    Returns:
        Audio waveform as float32 numpy array, shape (n_samples,)
    """
    import os
    import sys

    # Add mlx_audio parent directory to sys.path so whisper_audio_native can be imported
    mlx_audio_parent = os.path.join(os.path.dirname(__file__), '..', 'mlx_audio')
    mlx_audio_parent = os.path.abspath(mlx_audio_parent)
    if mlx_audio_parent not in sys.path:
        sys.path.insert(0, mlx_audio_parent)

    import whisper_audio_native
    return whisper_audio_native.load_audio(file_path, sample_rate=sample_rate)


def pad_or_trim(
    audio: Union[np.ndarray, "mx.array"],
    length: int = N_SAMPLES,
    axis: int = -1,
) -> Union[np.ndarray, "mx.array"]:
    """
    Pad or trim audio to exact length.

    For standard Whisper, this pads/trims to 30s (480000 samples).
    For dynamic chunking, pass the actual desired length.

    Args:
        audio: Audio waveform
        length: Target length in samples
        axis: Axis along which to pad/trim

    Returns:
        Audio padded or trimmed to exact length
    """
    if isinstance(audio, np.ndarray):
        if audio.shape[axis] > length:
            # Trim
            slices = [slice(None)] * audio.ndim
            slices[axis] = slice(0, length)
            return audio[tuple(slices)]
        if audio.shape[axis] < length:
            # Pad with zeros
            pad_width = [(0, 0)] * audio.ndim
            pad_width[axis] = (0, length - audio.shape[axis])
            return np.pad(audio, pad_width, mode="constant")
        return audio
    # MLX array
    if audio.shape[axis] > length:
        slices = [slice(None)] * audio.ndim
        slices[axis] = slice(0, length)
        return audio[tuple(slices)]
    if audio.shape[axis] < length:
        pad_width = [(0, 0)] * audio.ndim
        pad_width[axis] = (0, length - audio.shape[axis])
        return mx.pad(audio, pad_width)
    return audio


@lru_cache(maxsize=4)
def get_hanning_window(size: int) -> np.ndarray:
    """
    Get cached hanning window.

    OPT-NEW-26: Avoid repeated computation by caching window.
    Whisper uses n_fft=400, so this caches one or two sizes typically.

    Args:
        size: Window size (n_fft)

    Returns:
        Hanning window, shape (size,)
    """
    return np.hanning(size + 1)[:-1].astype(np.float32)


@lru_cache(maxsize=4)
def get_mel_filters(
    sample_rate: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    n_mels: int = 128,
) -> np.ndarray:
    """
    Get mel filterbank matrix (cached).

    OPT-NEW-25: Avoid repeated computation by caching filterbank.
    Uses the same mel scale as OpenAI's Whisper.

    Args:
        sample_rate: Audio sample rate
        n_fft: FFT window size
        n_mels: Number of mel bands (80 for v1/v2, 128 for v3)

    Returns:
        Mel filterbank matrix, shape (n_mels, n_fft // 2 + 1)
    """
    # Frequency bins
    n_freqs = n_fft // 2 + 1
    freqs = np.linspace(0, sample_rate / 2, n_freqs)

    # Mel scale conversion
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    min_mel = hz_to_mel(0)
    max_mel = hz_to_mel(sample_rate / 2)

    # Mel band edges
    mel_points = np.linspace(min_mel, max_mel, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    # Create filterbank
    filters = np.zeros((n_mels, n_freqs))
    for i in range(n_mels):
        left = hz_points[i]
        center = hz_points[i + 1]
        right = hz_points[i + 2]

        # Rising slope
        rising = (freqs - left) / (center - left)
        # Falling slope
        falling = (right - freqs) / (right - center)

        filters[i] = np.maximum(0, np.minimum(rising, falling))

    # Normalize (same as Whisper)
    enorm = 2.0 / (hz_points[2:n_mels+2] - hz_points[:n_mels])
    filters *= enorm[:, np.newaxis]

    return filters.astype(np.float32)


def log_mel_spectrogram(
    audio: Union[np.ndarray, "mx.array"],
    n_mels: int = 128,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    padding: int | None = None,
    mel_filters: np.ndarray | None = None,
) -> "mx.array":
    """
    Compute log mel spectrogram from audio waveform.

    Uses mlx-whisper's implementation for numerical equivalence.

    Args:
        audio: Audio waveform, shape (n_samples,)
        n_mels: Number of mel bands (80 for v1/v2, 128 for v3)
        n_fft: FFT window size (default: 400)
        hop_length: Hop length between frames (default: 160)
        padding: Optional padding to apply
        mel_filters: Precomputed mel filterbank (optional, ignored - uses mlx-whisper's)

    Returns:
        Log mel spectrogram, shape (n_frames, n_mels) to match mlx-whisper convention
    """
    # Use mlx-whisper's implementation for numerical equivalence
    try:
        from mlx_whisper.audio import log_mel_spectrogram as mlx_log_mel_spectrogram
        return mlx_log_mel_spectrogram(audio, n_mels=n_mels, padding=padding or 0)
    except ImportError:
        # Fallback to custom implementation if mlx-whisper not available
        return _log_mel_spectrogram_fallback(audio, n_mels, n_fft, hop_length, padding, mel_filters)


def _log_mel_spectrogram_fallback(
    audio: Union[np.ndarray, "mx.array"],
    n_mels: int = 128,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    padding: int | None = None,
    mel_filters: np.ndarray | None = None,
    use_mlx_fft: bool = True,
) -> "mx.array":
    """
    Fallback mel spectrogram implementation when mlx-whisper is not available.

    OPT-NEW-6: Uses MLX FFT for GPU-accelerated computation when available.

    Args:
        audio: Audio waveform
        n_mels: Number of mel bands
        n_fft: FFT window size
        hop_length: Hop length between frames
        padding: Optional padding
        mel_filters: Precomputed mel filterbank
        use_mlx_fft: If True, use MLX FFT for GPU acceleration (default: True)

    Returns:
        Log mel spectrogram, shape (n_frames, n_mels)
    """
    if mx is None:
        raise ImportError("mlx is required: pip install mlx")

    # Convert to numpy for framing
    if isinstance(audio, mx.array):
        audio = np.array(audio)

    # Apply padding
    if padding is None:
        padding = n_fft // 2
    if padding > 0:
        audio = np.pad(audio, (padding, padding), mode="reflect")

    # Get mel filters (OPT-NEW-25: cached)
    if mel_filters is None:
        mel_filters = get_mel_filters(SAMPLE_RATE, n_fft, n_mels)

    # Window function (OPT-NEW-26: cached)
    window = get_hanning_window(n_fft)

    # Frame the audio
    n_frames = 1 + (len(audio) - n_fft) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        audio,
        shape=(n_frames, n_fft),
        strides=(audio.strides[0] * hop_length, audio.strides[0]),
    )

    # Apply window
    windowed = frames * window

    # OPT-NEW-6: Use MLX FFT for GPU-accelerated computation
    if use_mlx_fft:
        # Convert to MLX array for GPU-accelerated FFT
        windowed_mx = mx.array(windowed)
        stft = mx.fft.rfft(windowed_mx, n=n_fft)
        # Compute magnitude (discard last frequency bin like mlx-whisper)
        magnitudes = mx.abs(stft[:, :-1]) ** 2

        # Apply mel filterbank using MLX
        mel_filters_mx = mx.array(mel_filters[:, :-1].T)
        mel_spec = magnitudes @ mel_filters_mx

        # Log scale with clipping (all MLX operations)
        log_spec = mx.log10(mx.clip(mel_spec, a_min=1e-10, a_max=None))
        max_val = mx.max(log_spec)
        log_spec = mx.maximum(log_spec, max_val - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        return log_spec.astype(mx.float32)
    # Fallback to numpy FFT (CPU)
    stft = np.fft.rfft(windowed, n=n_fft)

    # Compute magnitude (discard last frequency bin like mlx-whisper)
    magnitudes = np.abs(stft[:, :-1]) ** 2

    # Apply mel filterbank
    mel_spec = magnitudes @ mel_filters[:, :-1].T

    # Log scale with clipping
    log_spec = np.log10(np.clip(mel_spec, a_min=1e-10, a_max=None))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    return mx.array(log_spec.astype(np.float32))


# =============================================================================
# Shared FFT for VAD-Whisper Fusion (OPT-SHARED-FFT)
# =============================================================================


def compute_stft_and_mel(
    audio: Union[np.ndarray, "mx.array"],
    n_mels: int = 128,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    padding: int | None = None,
    return_stft: bool = True,
) -> Union["mx.array", tuple["mx.array", "mx.array"]]:
    """
    Compute mel spectrogram and optionally STFT magnitude from single FFT.

    OPT-SHARED-FFT: When running VAD alongside Whisper, computing FFT once
    and deriving both mel spectrogram (for Whisper) and STFT magnitude (for VAD)
    saves 10-15% of audio processing time.

    Args:
        audio: Audio waveform, shape (n_samples,)
        n_mels: Number of mel bands (80 for v1/v2, 128 for v3)
        n_fft: FFT window size (default: 400)
        hop_length: Hop length between frames (default: 160)
        padding: Optional padding to apply
        return_stft: If True, also return STFT magnitude for VAD

    Returns:
        If return_stft=False: mel spectrogram, shape (n_frames, n_mels)
        If return_stft=True: (mel_spectrogram, stft_magnitude)
            - mel_spectrogram: shape (n_frames, n_mels)
            - stft_magnitude: shape (n_frames, n_fft//2) - squared magnitude
    """
    if mx is None:
        raise ImportError("mlx is required: pip install mlx")

    # Convert to numpy for framing
    if isinstance(audio, mx.array):
        audio = np.array(audio)

    # Apply padding
    if padding is None:
        padding = n_fft // 2
    if padding > 0:
        audio = np.pad(audio, (padding, padding), mode="reflect")

    # Get mel filters (OPT-NEW-25: cached)
    mel_filters = get_mel_filters(SAMPLE_RATE, n_fft, n_mels)

    # Window function (OPT-NEW-26: cached)
    window = get_hanning_window(n_fft)

    # Frame the audio
    n_frames = 1 + (len(audio) - n_fft) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        audio,
        shape=(n_frames, n_fft),
        strides=(audio.strides[0] * hop_length, audio.strides[0]),
    )

    # Apply window
    windowed = frames * window

    # Convert to MLX array for GPU-accelerated FFT
    windowed_mx = mx.array(windowed)
    stft = mx.fft.rfft(windowed_mx, n=n_fft)

    # Compute magnitude (discard last frequency bin like mlx-whisper)
    magnitudes = mx.abs(stft[:, :-1]) ** 2

    # Apply mel filterbank using MLX
    mel_filters_mx = mx.array(mel_filters[:, :-1].T)
    mel_spec = magnitudes @ mel_filters_mx

    # Log scale with clipping (all MLX operations)
    log_spec = mx.log10(mx.clip(mel_spec, a_min=1e-10, a_max=None))
    max_val = mx.max(log_spec)
    log_spec = mx.maximum(log_spec, max_val - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    mel = log_spec.astype(mx.float32)

    if return_stft:
        # Return STFT magnitude for VAD use
        # Shape: (n_frames, n_fft//2) = (n_frames, 200) for Whisper's n_fft=400
        return mel, magnitudes.astype(mx.float32)
    return mel


def compute_mel_for_duration(
    audio: np.ndarray,
    duration_seconds: float | None = None,
    n_mels: int = 128,
    pad_to_30s: bool = False,
) -> tuple["mx.array", int, float]:
    """
    Compute mel spectrogram with duration awareness for dynamic chunking.

    This is the key function for variable-length audio processing:
    - Computes mel spectrogram for actual audio length (or duration_seconds)
    - Returns the actual number of encoder positions
    - Returns the actual audio duration for timestamp calibration

    Args:
        audio: Audio waveform, shape (n_samples,)
        duration_seconds: Optional target duration (trims if longer)
        n_mels: Number of mel bands
        pad_to_30s: If True, pad to 30s like standard Whisper

    Returns:
        Tuple of (mel_spectrogram, n_encoder_positions, actual_duration)
        - mel_spectrogram: shape (n_frames, n_mels) to match mlx-whisper convention
        - n_encoder_positions: encoder output sequence length
        - actual_duration: audio duration in seconds
    """
    # Compute actual duration
    actual_duration = len(audio) / SAMPLE_RATE

    # Optionally trim to target duration
    if duration_seconds is not None and actual_duration > duration_seconds:
        n_samples = int(duration_seconds * SAMPLE_RATE)
        audio = audio[:n_samples]
        actual_duration = duration_seconds

    # Optionally pad to 30s
    if pad_to_30s:
        audio = pad_or_trim(audio, N_SAMPLES)
        actual_duration = CHUNK_LENGTH

    # Compute mel spectrogram
    mel = log_mel_spectrogram(audio, n_mels=n_mels)

    # Encoder output length: conv2 has stride=2, so output is half of mel frames
    # mel shape is (n_frames, n_mels), so n_frames is shape[0]
    n_frames = mel.shape[0]
    n_encoder_positions = (n_frames + 1) // 2  # After conv2 with stride 2

    return mel, n_encoder_positions, actual_duration


def is_silent_audio(
    audio: np.ndarray,
    threshold_db: float = -40.0,
) -> bool:
    """
    Detect if audio is silent using RMS energy.

    This prevents Whisper from hallucinating on silent audio.
    Whisper's no_speech token is unreliable for pure silence.

    Args:
        audio: Audio waveform as float32 numpy array
        threshold_db: RMS threshold in dB (default: -40 dB, very quiet)

    Returns:
        True if audio is below silence threshold
    """
    if len(audio) == 0:
        return True

    # Compute RMS energy
    rms = np.sqrt(np.mean(audio ** 2))

    # Convert to dB (with floor to avoid log(0))
    rms_db = 20 * np.log10(max(rms, 1e-10))

    return rms_db < threshold_db


def get_audio_rms_db(audio: np.ndarray) -> float:
    """
    Get audio RMS level in dB.

    Args:
        audio: Audio waveform as float32 numpy array

    Returns:
        RMS level in dB
    """
    if len(audio) == 0:
        return -100.0

    rms = np.sqrt(np.mean(audio ** 2))
    return 20 * np.log10(max(rms, 1e-10))


# =============================================================================
# Async Mel Prefetch (OPT-PREFETCH)
# =============================================================================


class AsyncMelPreparer:
    """
    Asynchronous mel spectrogram preparation for pipelined chunk processing.

    OPT-PREFETCH: Overlaps I/O and mel computation with GPU decode work.
    When processing long audio in chunks, this class prepares the next chunk's
    mel spectrogram in a background thread while the GPU is busy decoding
    the current chunk.

    Expected improvement: 10-15% overall speedup for transcribe_long.

    Usage:
        preparer = AsyncMelPreparer(n_mels=128)

        # Submit chunks for background preparation
        future1 = preparer.submit(chunk1, is_variable_length=False)
        future2 = preparer.submit(chunk2, is_variable_length=False)

        # Get results (blocks if not ready)
        mel1, info1 = future1.result()
        mel2, info2 = future2.result()

        preparer.shutdown()

    The prepare_mel() method handles both standard (30s padded) and
    variable-length modes, returning the mel spectrogram plus metadata.
    """

    def __init__(
        self,
        n_mels: int = 128,
        n_audio_ctx: int = 1500,
        max_workers: int = 2,
    ):
        """
        Initialize async mel preparer.

        Args:
            n_mels: Number of mel bands (128 for large-v3)
            n_audio_ctx: Encoder context length (1500 for Whisper)
            max_workers: Number of background threads (default 2)
        """
        from concurrent.futures import ThreadPoolExecutor

        self.n_mels = n_mels
        self.n_audio_ctx = n_audio_ctx
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._active_futures = []

    def prepare_mel(
        self,
        audio_chunk: np.ndarray,
        chunk_samples: int,
        is_variable_length: bool = False,
    ) -> tuple:
        """
        Prepare mel spectrogram for a chunk.

        This is the core function that runs in background threads.

        Args:
            audio_chunk: Audio waveform for this chunk
            chunk_samples: Expected chunk length in samples (for padding)
            is_variable_length: Use variable-length mode (no padding)

        Returns:
            Tuple of (mel_array, mel_info) where mel_info is:
            (is_variable, encoder_positions, actual_duration)
        """
        if mx is None:
            raise ImportError("mlx is required: pip install mlx")

        actual_duration = len(audio_chunk) / SAMPLE_RATE
        chunk_length = chunk_samples / SAMPLE_RATE

        if is_variable_length:
            # Variable-length mode for partial last chunk
            mel, encoder_positions, actual_duration = compute_mel_for_duration(
                audio_chunk, n_mels=self.n_mels, pad_to_30s=False,
            )
            mel_info = (True, encoder_positions, actual_duration)
        else:
            # Standard 30s padded mode
            mel = log_mel_spectrogram(audio_chunk, n_mels=self.n_mels)
            # Pad/trim to standard length (3000 frames)
            target_len = self.n_audio_ctx * 2
            if mel.shape[0] < target_len:
                mel = mx.pad(mel, [(0, target_len - mel.shape[0]), (0, 0)])
            elif mel.shape[0] > target_len:
                mel = mel[:target_len, :]
            mel_info = (False, self.n_audio_ctx, chunk_length)

        return mel, mel_info

    def submit(
        self,
        audio_chunk: np.ndarray,
        chunk_samples: int,
        is_variable_length: bool = False,
    ) -> "concurrent.futures.Future":
        """
        Submit a chunk for background mel preparation.

        Args:
            audio_chunk: Audio waveform for this chunk
            chunk_samples: Expected chunk length in samples
            is_variable_length: Use variable-length mode

        Returns:
            Future that will contain (mel, mel_info) when ready
        """

        future = self._executor.submit(
            self.prepare_mel,
            audio_chunk,
            chunk_samples,
            is_variable_length,
        )
        self._active_futures.append(future)
        return future

    def submit_batch(
        self,
        chunks: list,
        chunk_samples: int,
        variable_length_indices: set = None,
    ) -> list:
        """
        Submit multiple chunks for background preparation.

        Args:
            chunks: List of audio chunks
            chunk_samples: Expected chunk length in samples
            variable_length_indices: Set of chunk indices to use variable-length

        Returns:
            List of futures in same order as chunks
        """
        if variable_length_indices is None:
            variable_length_indices = set()

        futures = []
        for i, chunk in enumerate(chunks):
            is_variable = i in variable_length_indices
            future = self.submit(chunk, chunk_samples, is_variable)
            futures.append(future)

        return futures

    def get_ready_count(self) -> int:
        """Get count of futures that are ready (non-blocking check)."""
        return sum(1 for f in self._active_futures if f.done())

    def wait_all(self, timeout: float = None) -> list:
        """
        Wait for all submitted futures to complete.

        Args:
            timeout: Maximum seconds to wait (None = wait forever)

        Returns:
            List of (mel, mel_info) results in submission order
        """
        from concurrent.futures import wait

        if self._active_futures:
            wait(self._active_futures, timeout=timeout)

        results = [f.result() for f in self._active_futures]
        self._active_futures = []
        return results

    def shutdown(self, wait: bool = True):
        """
        Shutdown the executor.

        Args:
            wait: If True, wait for pending tasks to complete
        """
        self._executor.shutdown(wait=wait)
        self._active_futures = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown(wait=True)
        return False
