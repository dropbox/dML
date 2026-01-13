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
whisper_audio_native - Native C++ audio loader for WhisperMLX.

This module provides high-performance audio loading using libav (FFmpeg libraries).
Output is identical to the ffmpeg subprocess command used by mlx-whisper.

Usage:
    import whisper_audio_native

    # Simple loading
    audio = whisper_audio_native.load_audio("speech.wav")  # Returns numpy float32

    # With custom sample rate
    audio = whisper_audio_native.load_audio("speech.mp3", sample_rate=22050)

    # Get duration without decoding
    duration = whisper_audio_native.get_duration("speech.wav")  # Returns seconds

    # Advanced: reusable loader
    loader = whisper_audio_native.AudioLoader(sample_rate=16000)
    audio1 = loader.load("file1.wav")
    audio2 = loader.load("file2.mp3")

Performance:
    - 45ms ffmpeg subprocess -> <1ms native (50x faster for short audio)
    - Zero-copy numpy array output
    - Thread-safe loader instances

Output Guarantee:
    The output is identical to:
    ffmpeg -nostdin -threads 0 -i <file> -f s16le -ac 1 -acodec pcm_s16le -ar 16000 -
"""

__version__ = "0.1.0"

try:
    from .mlx_audio_native import (
        AudioLoader,
        get_duration,
        get_sample_count,
        load_audio,
    )
except ImportError as e:
    # Provide helpful error message if native module not built
    _import_error = str(e)
    if "mlx_audio_native" in _import_error:
        raise ImportError(
            "whisper_audio_native module not built. Build instructions:\n"
            "  cd tools/mlx_audio\n"
            "  mkdir build && cd build\n"
            "  cmake .. -DCMAKE_BUILD_TYPE=Release\n"
            "  make -j\n"
            "  cp mlx_audio_native*.so ../whisper_audio_native/\n",
        ) from e
    raise

__all__ = [
    "load_audio",
    "get_duration",
    "get_sample_count",
    "AudioLoader",
]
