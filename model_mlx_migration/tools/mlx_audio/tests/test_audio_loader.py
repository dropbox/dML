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

"""Tests for the native C++ audio loader (mlx_audio)."""

import glob
import subprocess
import sys
import time

import numpy as np
import pytest

# Add the mlx_audio directory to path so whisper_audio_native package can be imported
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

try:
    import whisper_audio_native as mlx_audio
except ImportError:
    pytest.skip(
        "whisper_audio_native module not built. Run cmake and make.",
        allow_module_level=True,
    )


def load_with_ffmpeg(path: str, sample_rate: int = 16000) -> np.ndarray:
    """Load audio using ffmpeg subprocess (reference implementation)."""
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads",
        "0",
        "-i",
        path,
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()}")

    s16_data = np.frombuffer(result.stdout, dtype=np.int16)
    return s16_data.astype(np.float32) / 32768.0


@pytest.fixture
def test_audio_files():
    """Get list of test audio files."""
    base_path = __file__.rsplit("/", 4)[0]
    files = glob.glob(f"{base_path}/tests/prosody/contour_vs_v5_output/*.wav")
    if not files:
        pytest.skip("No test audio files found")
    return files[:5]


class TestAudioLoader:
    """Tests for mlx_audio load_audio function."""

    def test_load_audio_basic(self, test_audio_files):
        """Test basic audio loading."""
        audio = mlx_audio.load_audio(test_audio_files[0])

        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert len(audio) > 0
        assert audio.min() >= -1.0
        assert audio.max() <= 1.0

    def test_load_audio_custom_sample_rate(self, test_audio_files):
        """Test loading with custom sample rate."""
        audio_16k = mlx_audio.load_audio(test_audio_files[0], sample_rate=16000)
        audio_8k = mlx_audio.load_audio(test_audio_files[0], sample_rate=8000)

        # 8kHz should have half the samples
        ratio = len(audio_16k) / len(audio_8k)
        assert 1.9 < ratio < 2.1

    def test_output_matches_ffmpeg(self, test_audio_files):
        """Test that output matches ffmpeg subprocess within tolerance."""
        for path in test_audio_files:
            native = mlx_audio.load_audio(path, 16000)
            ffmpeg = load_with_ffmpeg(path, 16000)

            # Lengths must match exactly
            assert len(native) == len(ffmpeg), (
                f"Length mismatch for {path}: {len(native)} vs {len(ffmpeg)}"
            )

            # Values must match within 1 LSB (1/32768 ≈ 3e-5)
            # This is acceptable resampler rounding difference
            max_diff = np.max(np.abs(native - ffmpeg))
            assert max_diff <= 3.1e-5, (
                f"Max diff {max_diff} exceeds 1 LSB for {path}"
            )

    def test_get_duration(self, test_audio_files):
        """Test get_duration function."""
        for path in test_audio_files:
            duration = mlx_audio.get_duration(path)

            # Duration should be positive and reasonable
            assert duration > 0
            assert duration < 3600  # Less than 1 hour

            # Cross-check with loaded sample count
            audio = mlx_audio.load_audio(path, 16000)
            expected_duration = len(audio) / 16000

            # Should match within 1%
            assert abs(duration - expected_duration) / expected_duration < 0.01

    def test_get_sample_count(self, test_audio_files):
        """Test get_sample_count function."""
        for path in test_audio_files:
            estimated = mlx_audio.get_sample_count(path, 16000)
            actual = len(mlx_audio.load_audio(path, 16000))

            # Estimate should be within 1% of actual
            assert abs(estimated - actual) / actual < 0.01

    def test_invalid_file(self):
        """Test that invalid file raises error."""
        with pytest.raises(RuntimeError):
            mlx_audio.load_audio("/nonexistent/file.wav")

    def test_audio_loader_class(self, test_audio_files):
        """Test AudioLoader class interface."""
        loader = mlx_audio.AudioLoader()

        assert loader.sample_rate == 16000
        assert loader.ok

        audio = loader.load(test_audio_files[0])
        assert isinstance(audio, np.ndarray)
        assert loader.ok


class TestPerformance:
    """Performance benchmarks for the native audio loader."""

    def test_speedup_vs_ffmpeg(self, test_audio_files):
        """Verify native loader is faster than ffmpeg subprocess."""
        path = test_audio_files[0]

        # Warm up
        mlx_audio.load_audio(path, 16000)
        load_with_ffmpeg(path, 16000)

        # Benchmark native (10 runs)
        start = time.time()
        for _ in range(10):
            mlx_audio.load_audio(path, 16000)
        native_time = (time.time() - start) / 10

        # Benchmark ffmpeg (10 runs)
        start = time.time()
        for _ in range(10):
            load_with_ffmpeg(path, 16000)
        ffmpeg_time = (time.time() - start) / 10

        speedup = ffmpeg_time / native_time
        print(f"\nNative: {native_time*1000:.2f}ms")
        print(f"FFmpeg: {ffmpeg_time*1000:.2f}ms")
        print(f"Speedup: {speedup:.1f}x")

        # Should be at least 2x faster
        assert speedup > 2.0, f"Speedup {speedup:.1f}x is below minimum 2x"

    def test_repeated_loading_performance(self, test_audio_files):
        """Test that repeated loading doesn't degrade performance."""
        path = test_audio_files[0]

        # Warm up
        mlx_audio.load_audio(path, 16000)

        # First batch
        start = time.time()
        for _ in range(20):
            mlx_audio.load_audio(path, 16000)
        first_batch = time.time() - start

        # Second batch
        start = time.time()
        for _ in range(20):
            mlx_audio.load_audio(path, 16000)
        second_batch = time.time() - start

        # Second batch should not be significantly slower
        assert second_batch < first_batch * 1.5
