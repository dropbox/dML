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
Tests for Silero VAD MLX implementation.

Validates:
1. Model creation and initialization
2. Weight conversion from PyTorch
3. Numerical equivalence with PyTorch (tolerance ~0.02)
4. Streaming mode with stateful inference
5. detect() convenience method
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Module-level random generator for reproducibility
_rng = np.random.default_rng(42)

# Add converters/models to path
sys.path.insert(0, str(Path(__file__).parent.parent / "tools/pytorch_to_mlx/converters/models"))


@pytest.fixture(scope="module")
def torch_model():
    """Load PyTorch Silero VAD model from JIT file."""
    import torch

    jit_path = Path.home() / ".cache/torch/hub/snakers4_silero-vad_master/src/silero_vad/data/silero_vad.jit"

    if not jit_path.exists():
        pytest.skip(f"Silero VAD JIT model not found at {jit_path}")

    model = torch.jit.load(str(jit_path))
    model.eval()
    return model


@pytest.fixture(scope="module")
def mlx_model(torch_model):
    """Convert PyTorch model to MLX."""
    from silero_vad_mlx import SileroVAD

    return SileroVAD.from_torch(torch_model, sample_rate=16000)


class TestSileroVADMLXCreation:
    """Tests for model creation and initialization."""

    def test_create_model_16k(self):
        """Test creating 16kHz model."""
        from silero_vad_mlx import SileroVAD

        model = SileroVAD(sample_rate=16000)
        assert model.sample_rate == 16000
        assert model.context_size == 64
        assert len(model.encoder) == 4

    def test_create_model_8k(self):
        """Test creating 8kHz model."""
        from silero_vad_mlx import SileroVAD

        model = SileroVAD(sample_rate=8000)
        assert model.sample_rate == 8000
        assert model.context_size == 32
        assert len(model.encoder) == 4


class TestSileroVADMLXConversion:
    """Tests for PyTorch to MLX weight conversion."""

    def test_conversion_produces_model(self, mlx_model):
        """Test that conversion produces a valid model."""

        assert mlx_model is not None
        assert mlx_model.sample_rate == 16000
        # Check STFT buffer was converted
        assert mlx_model.stft.forward_basis_buffer.shape == (258, 1, 256)

    def test_encoder_weights_converted(self, mlx_model):
        """Test encoder weights are converted correctly."""

        # Check each encoder block
        expected_channels = [(129, 128), (128, 64), (64, 64), (64, 128)]
        for i, block in enumerate(mlx_model.encoder):
            in_ch, out_ch = expected_channels[i]
            # MLX conv weight shape: [out, kernel, in]
            weight_shape = block.conv.weight.shape
            assert weight_shape[0] == out_ch, f"Block {i} out_channels mismatch"
            assert weight_shape[2] == in_ch, f"Block {i} in_channels mismatch"

    def test_lstm_weights_converted(self, mlx_model):
        """Test LSTM weights are converted correctly."""
        lstm = mlx_model.decoder.lstm
        # LSTM weights: 4*hidden_size x input_size
        assert lstm.weight_ih.shape == (512, 128)  # 4*128 x 128
        assert lstm.weight_hh.shape == (512, 128)  # 4*128 x 128
        assert lstm.bias_ih.shape == (512,)
        assert lstm.bias_hh.shape == (512,)


class TestSileroVADMLXNumericalEquivalence:
    """Tests for numerical equivalence with PyTorch."""

    def test_single_chunk_equivalence(self, torch_model, mlx_model):
        """Test single chunk produces similar results to PyTorch."""
        import mlx.core as mx
        import torch

        audio = _rng.standard_normal(512).astype(np.float32) * 0.1

        # MLX
        mlx_audio = mx.array(audio[None, :])
        mlx_model.reset_state()
        mlx_prob, _ = mlx_model(mlx_audio)
        mx.eval(mlx_prob)
        mlx_val = float(mlx_prob[0, 0])

        # PyTorch
        torch_audio = torch.tensor(audio)
        pt_prob = torch_model(torch_audio, 16000).item()

        diff = abs(mlx_val - pt_prob)
        assert diff < 0.02, f"Single chunk difference {diff} exceeds tolerance"

    def test_multiple_chunks_equivalence(self, torch_model, mlx_model):
        """Test multiple chunks maintain reasonable accuracy."""
        import mlx.core as mx
        import torch

        rng = np.random.default_rng(123)
        max_diff = 0.0
        avg_diff = 0.0
        n_chunks = 10

        for _i in range(n_chunks):
            audio = rng.standard_normal(512).astype(np.float32) * 0.1

            # MLX
            mlx_audio = mx.array(audio[None, :])
            mlx_model.reset_state()
            mlx_prob, _ = mlx_model(mlx_audio)
            mx.eval(mlx_prob)
            mlx_val = float(mlx_prob[0, 0])

            # PyTorch
            torch_audio = torch.tensor(audio)
            pt_prob = torch_model(torch_audio, 16000).item()

            diff = abs(mlx_val - pt_prob)
            max_diff = max(max_diff, diff)
            avg_diff += diff

        avg_diff /= n_chunks
        assert max_diff < 0.05, f"Max difference {max_diff} exceeds tolerance"
        assert avg_diff < 0.02, f"Average difference {avg_diff} exceeds tolerance"


class TestSileroVADMLXStreaming:
    """Tests for streaming mode with stateful inference."""

    def test_state_reset(self, mlx_model):
        """Test state reset clears LSTM state and context."""
        import mlx.core as mx

        # Run a chunk to create state
        audio = mx.array(_rng.standard_normal((1, 512)).astype(np.float32))
        mlx_model(audio)

        assert mlx_model._state is not None
        assert mlx_model._context is not None

        # Reset
        mlx_model.reset_state()
        assert mlx_model._state is None
        assert mlx_model._context is None

    def test_stateful_inference_produces_output(self, mlx_model):
        """Test stateful inference produces valid output."""
        import mlx.core as mx

        mlx_model.reset_state()

        # Process multiple chunks
        probs = []
        for _ in range(5):
            audio = mx.array(_rng.standard_normal((1, 512)).astype(np.float32) * 0.1)
            prob, _ = mlx_model(audio)
            mx.eval(prob)
            probs.append(float(prob[0, 0]))

        # All probabilities should be between 0 and 1
        assert all(0.0 <= p <= 1.0 for p in probs), f"Invalid probabilities: {probs}"


class TestSileroVADMLXDetect:
    """Tests for detect() convenience method."""

    def test_detect_method(self, mlx_model):
        """Test detect() method with longer audio."""
        import mlx.core as mx

        # Create 2 seconds of audio (32000 samples at 16kHz)
        audio = mx.array(_rng.standard_normal(32000).astype(np.float32) * 0.1)

        result = mlx_model.detect(audio, threshold=0.5)
        mx.eval(result)

        # Should produce ~62 chunks (32000/512)
        assert result.ndim == 2
        assert result.shape[0] == 1  # batch size
        assert result.shape[1] >= 60  # approx number of chunks

    def test_detect_with_batch(self, mlx_model):
        """Test detect() with batched audio."""
        import mlx.core as mx

        # Batch of 2 audio samples
        audio = mx.array(_rng.standard_normal((2, 16000)).astype(np.float32) * 0.1)

        result = mlx_model.detect(audio, threshold=0.5)
        mx.eval(result)

        assert result.shape[0] == 2  # batch size


class TestSileroVADMLXComponents:
    """Tests for individual components."""

    def test_stft_produces_correct_shape(self, mlx_model):
        """Test STFT produces correct spectrogram shape."""
        import mlx.core as mx

        # 512 samples + context
        audio = mx.array(_rng.standard_normal((1, 576)).astype(np.float32))
        spec = mlx_model.stft(audio)
        mx.eval(spec)

        # Should be [batch, n_fft//2+1, frames]
        assert spec.shape[0] == 1
        assert spec.shape[1] == 129  # 256//2 + 1

    def test_lstm_cell_produces_state(self):
        """Test LSTM cell produces correct state shape."""
        import mlx.core as mx
        from silero_vad_mlx import LSTMCell

        lstm = LSTMCell(input_size=128, hidden_size=128)
        # Initialize weights
        lstm.weight_ih = mx.random.normal((512, 128))
        lstm.weight_hh = mx.random.normal((512, 128))
        lstm.bias_ih = mx.zeros((512,))
        lstm.bias_hh = mx.zeros((512,))

        x = mx.random.normal((1, 128))
        h, (h_new, c_new) = lstm(x)
        mx.eval(h, h_new, c_new)

        assert h.shape == (1, 128)
        assert h_new.shape == (1, 128)
        assert c_new.shape == (1, 128)
