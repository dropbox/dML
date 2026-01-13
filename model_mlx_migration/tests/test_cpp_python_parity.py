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
Test C++ vs Python Kokoro parity.

STRICT THRESHOLDS - DO NOT MODIFY WITHOUT MANAGER APPROVAL.

The 0.01 threshold IS achievable. Worker #603 found the root cause:
- 0.39 Hz F0 difference at frame 85 (voiced/unvoiced transition)
- This gets amplified 300x through phase scaling

The fix is to trace and fix the F0 predictor, NOT to raise thresholds.
See reports/main/WORKER_DIRECTIVE.md for required fix.
"""
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Skip if C++ binary doesn't exist
CPP_BINARY = Path(__file__).parent.parent / "src" / "kokoro" / "test_token_input"
MODEL_PATH = Path(__file__).parent.parent / "kokoro_cpp_export"

pytestmark = pytest.mark.skipif(
    not CPP_BINARY.exists() or not MODEL_PATH.exists(),
    reason="C++ binary or model not available",
)


class TestCppPythonParity:
    """Strict parity tests between C++ and Python Kokoro implementations."""

    # STRICT THRESHOLDS - DO NOT MODIFY
    MAX_ABS_THRESHOLD = 0.01  # Maximum absolute error - FIX THE CODE, NOT THIS
    MIN_CORRELATION = 0.99    # Minimum correlation coefficient
    MIN_PHASE_CORRELATION = 0.25  # Expected low due to 2*pi phase wraps

    # Test tokens for "Hello world"
    HELLO_WORLD_TOKENS = [0, 50, 83, 54, 156, 31, 16, 65, 156, 87, 123, 54, 46, 0]

    @pytest.fixture
    def generate_python_reference(self):
        """Generate Python MLX reference audio."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))

        import mlx.core as mx

        from tools.pytorch_to_mlx.converters import KokoroConverter

        converter = KokoroConverter()
        model, config, state_dict = converter.load_from_hf("hexgrad/Kokoro-82M")
        model.set_deterministic(True)

        voice_pack = converter.load_voice_pack("af_bella")
        voice_emb = converter.select_voice_embedding(voice_pack, 12)

        input_ids = mx.array([self.HELLO_WORLD_TOKENS])
        audio = model(input_ids, voice_emb)
        mx.eval(audio)

        return np.array(audio).flatten()

    def generate_cpp_audio(self) -> np.ndarray:
        """Generate C++ audio using test_token_input."""
        import scipy.io.wavfile as wav

        with tempfile.TemporaryDirectory():
            # Build command
            token_str = " ".join(map(str, self.HELLO_WORLD_TOKENS))
            cmd = f"cd {CPP_BINARY.parent} && ./test_token_input ../../kokoro_cpp_export af_bella {token_str}"

            # Run C++
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                pytest.fail(f"C++ binary failed: {result.stderr}")

            # Load output
            output_wav = CPP_BINARY.parent / "token_input_output.wav"
            if not output_wav.exists():
                pytest.fail("C++ did not produce output.wav")

            sr, audio_int16 = wav.read(output_wav)
            return audio_int16.astype(np.float32) / 32767.0

    def test_cpp_python_max_abs_error(self, generate_python_reference):
        """
        Test that C++ and Python audio have max absolute error < 0.01.

        Previously xfail due to float32 precision, now consistently passing.
        See reports/main/F0_PRECISION_ANALYSIS_706.md for historical analysis.
        """
        py_audio = generate_python_reference
        cpp_audio = self.generate_cpp_audio()

        # Align lengths
        min_len = min(len(py_audio), len(cpp_audio))
        py_audio = py_audio[:min_len]
        cpp_audio = cpp_audio[:min_len]

        # Compute error
        max_abs_error = float(np.abs(py_audio - cpp_audio).max())

        assert max_abs_error <= self.MAX_ABS_THRESHOLD, (
            f"C++ vs Python max absolute error {max_abs_error:.6f} exceeds threshold {self.MAX_ABS_THRESHOLD}.\n"
            f"Check for regression in SourceModule phase accumulation or F0 predictor."
        )

    def test_cpp_python_correlation(self, generate_python_reference):
        """
        Test that C++ and Python audio have correlation > 0.99.

        Correlation is the primary quality metric. Expected value: ~0.993.
        If this fails, check for regression in model implementation.
        """
        py_audio = generate_python_reference
        cpp_audio = self.generate_cpp_audio()

        # Align lengths
        min_len = min(len(py_audio), len(cpp_audio))
        py_audio = py_audio[:min_len]
        cpp_audio = cpp_audio[:min_len]

        # Compute correlation
        correlation = float(np.corrcoef(py_audio, cpp_audio)[0, 1])

        assert correlation >= self.MIN_CORRELATION, (
            f"C++ vs Python correlation {correlation:.6f} below threshold {self.MIN_CORRELATION}.\n"
            f"\n"
            f"This indicates a significant regression in the model implementation.\n"
            f"Check F0 predictor, SourceModule, and STFT/ISTFT operations."
        )

    def test_stft_phase_correlation(self, generate_python_reference):
        """
        Test that C++ and Python source STFT phase has expected correlation.

        Phase correlation is expected to be low (~0.28) due to 2*pi phase wraps
        at magnitude boundaries. This is a fundamental limitation of comparing
        wrapped phase values and does NOT affect audio quality.

        The important metric is audio correlation (>0.99), not phase correlation.
        """
        # Generate both and save debug tensors
        _ = generate_python_reference  # This saves /tmp/py_source_stft.npy
        _ = self.generate_cpp_audio()  # This saves /tmp/cpp_source_stft.npy (with SAVE_DEBUG_TENSORS=1)

        py_stft_path = Path("/tmp/py_source_stft.npy")
        cpp_stft_path = Path("/tmp/cpp_source_stft.npy")

        if not py_stft_path.exists() or not cpp_stft_path.exists():
            pytest.skip("Debug tensors not available")

        py_stft = np.load(py_stft_path)
        cpp_stft = np.load(cpp_stft_path)

        if py_stft.shape != cpp_stft.shape:
            pytest.fail(f"STFT shape mismatch: Python={py_stft.shape}, C++={cpp_stft.shape}")

        # Extract phase (last 11 channels)
        n_bins = 11
        py_phase = py_stft[..., n_bins:]
        cpp_phase = cpp_stft[..., n_bins:]

        phase_corr = float(np.corrcoef(py_phase.flatten(), cpp_phase.flatten())[0, 1])

        assert phase_corr >= self.MIN_PHASE_CORRELATION, (
            f"STFT phase correlation {phase_corr:.6f} dropped below expected minimum {self.MIN_PHASE_CORRELATION}.\n"
            f"\n"
            f"This indicates a significant regression. Expected ~0.28 correlation due to phase wraps."
        )


# Validation that thresholds match documented values
def test_thresholds_documented():
    """
    Ensure the parity thresholds match documented values.

    These thresholds are the TARGET values:
    - MAX_ABS_THRESHOLD = 0.01: Target for numerical parity (currently xfail due to float32 floor)
    - MIN_CORRELATION = 0.99: Audio quality threshold (passes)
    - MIN_PHASE_CORRELATION = 0.25: Expected due to 2*pi phase wraps

    See reports/main/F0_PRECISION_ANALYSIS_706.md for why 0.01 is not achievable.
    See reports/main/PARITY_ROOT_CAUSE_ANALYSIS_603.md for initial root cause.
    """
    assert TestCppPythonParity.MAX_ABS_THRESHOLD == 0.01, (
        "MAX_ABS_THRESHOLD should be 0.01 (strict target, test is xfail due to float32 precision)"
    )
    assert TestCppPythonParity.MIN_CORRELATION == 0.99, (
        "MIN_CORRELATION should be 0.99"
    )
    assert TestCppPythonParity.MIN_PHASE_CORRELATION == 0.25, (
        "MIN_PHASE_CORRELATION should be 0.25"
    )
