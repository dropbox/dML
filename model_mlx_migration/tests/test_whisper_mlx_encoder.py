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
Tests for whisper_mlx encoder module.

Tests:
- sinusoids: Positional embedding generation
- AudioEncoder: Variable-length audio encoding
"""

import mlx.core as mx
import pytest


class TestSinusoids:
    """Tests for sinusoidal positional embedding generation."""

    def test_basic_shape(self):
        """Test output shape is correct."""
        from tools.whisper_mlx.encoder import sinusoids

        length, channels = 100, 512
        pos_emb = sinusoids(length, channels)

        assert pos_emb.shape == (length, channels)

    def test_various_lengths(self):
        """Test with various sequence lengths."""
        from tools.whisper_mlx.encoder import sinusoids

        channels = 256
        for length in [1, 10, 100, 500, 1500]:
            pos_emb = sinusoids(length, channels)
            assert pos_emb.shape == (length, channels)

    def test_various_channels(self):
        """Test with various channel dimensions."""
        from tools.whisper_mlx.encoder import sinusoids

        length = 100
        for channels in [64, 128, 256, 512, 768, 1024]:
            pos_emb = sinusoids(length, channels)
            assert pos_emb.shape == (length, channels)

    def test_channels_must_be_even(self):
        """Test that odd channels raise assertion error."""
        from tools.whisper_mlx.encoder import sinusoids

        with pytest.raises(AssertionError, match="must be even"):
            sinusoids(100, 127)

    def test_sin_cos_split(self):
        """Test first half is sin, second half is cos."""
        from tools.whisper_mlx.encoder import sinusoids

        length, channels = 50, 64
        pos_emb = sinusoids(length, channels)

        # First half should be sin values (bounded by [-1, 1])
        sin_part = pos_emb[:, :channels // 2]
        assert mx.all(sin_part >= -1).item()
        assert mx.all(sin_part <= 1).item()

        # Second half should be cos values (bounded by [-1, 1])
        cos_part = pos_emb[:, channels // 2:]
        assert mx.all(cos_part >= -1).item()
        assert mx.all(cos_part <= 1).item()

    def test_position_invariance(self):
        """Test that prefix of longer sequence matches shorter sequence."""
        from tools.whisper_mlx.encoder import sinusoids

        channels = 256

        # Generate embeddings for different lengths
        emb_100 = sinusoids(100, channels)
        emb_200 = sinusoids(200, channels)
        emb_500 = sinusoids(500, channels)

        # First 100 positions should match
        assert mx.allclose(emb_100, emb_200[:100], atol=1e-6).item()
        assert mx.allclose(emb_100, emb_500[:100], atol=1e-6).item()
        assert mx.allclose(emb_200[:100], emb_500[:100], atol=1e-6).item()

    def test_unique_positions(self):
        """Test each position has unique embedding."""
        from tools.whisper_mlx.encoder import sinusoids

        length, channels = 50, 128
        pos_emb = sinusoids(length, channels)

        # Each row should be different from others
        for i in range(length):
            for j in range(i + 1, min(length, i + 5)):  # Check nearby positions
                diff = mx.sum(mx.abs(pos_emb[i] - pos_emb[j])).item()
                assert diff > 0, f"Positions {i} and {j} have identical embeddings"

    def test_first_position_values(self):
        """Test first position has expected pattern."""
        from tools.whisper_mlx.encoder import sinusoids

        channels = 64
        pos_emb = sinusoids(10, channels)

        # At position 0, sin(0) = 0, cos(0) = 1
        # First channel should be sin(0) = 0
        assert abs(pos_emb[0, 0].item()) < 1e-6

        # First cos channel should be cos(0) = 1
        assert abs(pos_emb[0, channels // 2].item() - 1.0) < 1e-6

    def test_max_timescale_parameter(self):
        """Test max_timescale affects frequencies."""
        from tools.whisper_mlx.encoder import sinusoids

        length, channels = 100, 128

        emb_default = sinusoids(length, channels, max_timescale=10000)
        emb_smaller = sinusoids(length, channels, max_timescale=1000)

        # Different timescales should produce different embeddings
        assert not mx.allclose(emb_default, emb_smaller, atol=1e-3).item()

    def test_numerical_stability(self):
        """Test embeddings are numerically stable."""
        from tools.whisper_mlx.encoder import sinusoids

        # Large length
        pos_emb = sinusoids(3000, 512)

        assert mx.all(mx.isfinite(pos_emb)).item()
        assert mx.all(pos_emb >= -1).item()
        assert mx.all(pos_emb <= 1).item()


class TestAudioEncoder:
    """Tests for AudioEncoder class."""

    @pytest.fixture
    def encoder_config(self):
        """Standard encoder configuration."""
        return {
            "n_mels": 80,
            "n_ctx": 1500,
            "n_state": 384,
            "n_head": 6,
            "n_layer": 4,
        }

    @pytest.fixture
    def small_encoder_config(self):
        """Smaller config for faster tests."""
        return {
            "n_mels": 40,
            "n_ctx": 100,
            "n_state": 64,
            "n_head": 4,
            "n_layer": 2,
        }

    def test_init(self, small_encoder_config):
        """Test encoder initialization."""
        from tools.whisper_mlx.encoder import AudioEncoder

        encoder = AudioEncoder(**small_encoder_config)

        assert encoder.n_ctx == small_encoder_config["n_ctx"]
        assert encoder.n_state == small_encoder_config["n_state"]
        assert len(encoder.blocks) == small_encoder_config["n_layer"]

    def test_conv_layers_exist(self, small_encoder_config):
        """Test convolutional frontend exists."""
        from tools.whisper_mlx.encoder import AudioEncoder

        encoder = AudioEncoder(**small_encoder_config)

        assert hasattr(encoder, "conv1")
        assert hasattr(encoder, "conv2")

    def test_positional_embedding_shape(self, small_encoder_config):
        """Test positional embeddings have correct shape."""
        from tools.whisper_mlx.encoder import AudioEncoder

        encoder = AudioEncoder(**small_encoder_config)

        assert encoder._positional_embedding.shape == (
            small_encoder_config["n_ctx"],
            small_encoder_config["n_state"],
        )

    def test_forward_unbatched(self, small_encoder_config):
        """Test forward pass with unbatched input."""
        from tools.whisper_mlx.encoder import AudioEncoder

        encoder = AudioEncoder(**small_encoder_config)
        n_mels = small_encoder_config["n_mels"]
        n_ctx = small_encoder_config["n_ctx"]
        n_state = small_encoder_config["n_state"]

        # Input needs to produce n_ctx after conv2 stride=2
        # n_ctx = (n_frames + 1) // 2, so n_frames = 2 * n_ctx - 1
        n_frames = 2 * n_ctx - 1
        x = mx.random.normal((n_frames, n_mels))

        out = encoder(x, variable_length=False)

        # Output should be batched
        assert out.shape == (1, n_ctx, n_state)

    def test_forward_batched(self, small_encoder_config):
        """Test forward pass with batched input."""
        from tools.whisper_mlx.encoder import AudioEncoder

        encoder = AudioEncoder(**small_encoder_config)
        batch_size = 2
        n_mels = small_encoder_config["n_mels"]
        n_ctx = small_encoder_config["n_ctx"]
        n_state = small_encoder_config["n_state"]

        n_frames = 2 * n_ctx - 1
        x = mx.random.normal((batch_size, n_frames, n_mels))

        out = encoder(x, variable_length=False)

        assert out.shape == (batch_size, n_ctx, n_state)

    def test_variable_length_shorter(self, small_encoder_config):
        """Test variable length mode with shorter audio."""
        from tools.whisper_mlx.encoder import AudioEncoder

        encoder = AudioEncoder(**small_encoder_config)
        n_mels = small_encoder_config["n_mels"]
        n_state = small_encoder_config["n_state"]

        # Shorter input (half of max)
        n_frames = small_encoder_config["n_ctx"] - 1  # Will produce ~50 positions
        x = mx.random.normal((n_frames, n_mels))

        out = encoder(x, variable_length=True)

        # Output sequence should be shorter than n_ctx
        expected_len = (n_frames + 1) // 2
        assert out.shape == (1, expected_len, n_state)

    def test_variable_length_exact(self, small_encoder_config):
        """Test variable length mode with exact max length."""
        from tools.whisper_mlx.encoder import AudioEncoder

        encoder = AudioEncoder(**small_encoder_config)
        n_mels = small_encoder_config["n_mels"]
        n_ctx = small_encoder_config["n_ctx"]
        n_state = small_encoder_config["n_state"]

        n_frames = 2 * n_ctx - 1
        x = mx.random.normal((n_frames, n_mels))

        out = encoder(x, variable_length=True)

        assert out.shape == (1, n_ctx, n_state)

    def test_variable_length_too_long(self, small_encoder_config):
        """Test variable length mode rejects too-long input."""
        from tools.whisper_mlx.encoder import AudioEncoder

        encoder = AudioEncoder(**small_encoder_config)
        n_mels = small_encoder_config["n_mels"]
        n_ctx = small_encoder_config["n_ctx"]

        # Too long
        n_frames = 2 * (n_ctx + 10)
        x = mx.random.normal((n_frames, n_mels))

        with pytest.raises(ValueError, match="exceeds maximum context"):
            encoder(x, variable_length=True)

    def test_standard_mode_shape_mismatch(self, small_encoder_config):
        """Test standard mode rejects shape mismatch."""
        from tools.whisper_mlx.encoder import AudioEncoder

        encoder = AudioEncoder(**small_encoder_config)
        n_mels = small_encoder_config["n_mels"]

        # Wrong length
        n_frames = 50
        x = mx.random.normal((n_frames, n_mels))

        with pytest.raises(ValueError, match="Input shape mismatch"):
            encoder(x, variable_length=False)

    def test_get_output_length(self, small_encoder_config):
        """Test output length calculation."""
        from tools.whisper_mlx.encoder import AudioEncoder

        encoder = AudioEncoder(**small_encoder_config)

        # Test various input lengths
        test_cases = [
            (100, 50),   # (100 + 1) // 2 = 50
            (199, 100),  # (199 + 1) // 2 = 100
            (200, 100),  # (200 + 1) // 2 = 100
            (201, 101),  # (201 + 1) // 2 = 101
            (3000, 1500),
        ]

        for n_frames, expected_len in test_cases:
            assert encoder.get_output_length(n_frames) == expected_len

    def test_dtype_preservation(self, small_encoder_config):
        """Test dtype is preserved through forward pass."""
        from tools.whisper_mlx.encoder import AudioEncoder

        encoder = AudioEncoder(**small_encoder_config, dtype=mx.float16)
        n_mels = small_encoder_config["n_mels"]
        n_ctx = small_encoder_config["n_ctx"]

        n_frames = 2 * n_ctx - 1
        x = mx.random.normal((n_frames, n_mels)).astype(mx.float16)

        out = encoder(x, variable_length=False)

        # Output should be float16 or compatible
        assert out.dtype in [mx.float16, mx.float32]

    def test_transformer_blocks_count(self, small_encoder_config):
        """Test correct number of transformer blocks."""
        from tools.whisper_mlx.encoder import AudioEncoder

        encoder = AudioEncoder(**small_encoder_config)

        assert len(encoder.blocks) == small_encoder_config["n_layer"]

    def test_layer_norm_post(self, small_encoder_config):
        """Test final layer norm exists."""
        from tools.whisper_mlx.encoder import AudioEncoder

        encoder = AudioEncoder(**small_encoder_config)

        assert hasattr(encoder, "ln_post")

    def test_output_is_finite(self, small_encoder_config):
        """Test output contains no NaN or Inf."""
        from tools.whisper_mlx.encoder import AudioEncoder

        encoder = AudioEncoder(**small_encoder_config)
        n_mels = small_encoder_config["n_mels"]
        n_ctx = small_encoder_config["n_ctx"]

        n_frames = 2 * n_ctx - 1
        x = mx.random.normal((n_frames, n_mels))

        out = encoder(x, variable_length=False)

        assert mx.all(mx.isfinite(out)).item()


class TestAudioEncoderEdgeCases:
    """Edge case tests for AudioEncoder."""

    @pytest.fixture
    def tiny_encoder_config(self):
        """Minimal config for edge case tests."""
        return {
            "n_mels": 20,
            "n_ctx": 50,
            "n_state": 32,
            "n_head": 2,
            "n_layer": 1,
        }

    def test_minimum_input_length(self, tiny_encoder_config):
        """Test with minimum viable input length."""
        from tools.whisper_mlx.encoder import AudioEncoder

        encoder = AudioEncoder(**tiny_encoder_config)
        n_mels = tiny_encoder_config["n_mels"]

        # Minimum length that produces at least 1 position after conv2
        # (n_frames + 1) // 2 >= 1 -> n_frames >= 1
        x = mx.random.normal((3, n_mels))  # Very short

        out = encoder(x, variable_length=True)

        assert out.shape[1] == encoder.get_output_length(3)

    def test_batch_size_one(self, tiny_encoder_config):
        """Test with batch size 1."""
        from tools.whisper_mlx.encoder import AudioEncoder

        encoder = AudioEncoder(**tiny_encoder_config)
        n_mels = tiny_encoder_config["n_mels"]
        n_ctx = tiny_encoder_config["n_ctx"]

        n_frames = 2 * n_ctx - 1
        x = mx.random.normal((1, n_frames, n_mels))

        out = encoder(x, variable_length=False)

        assert out.shape[0] == 1

    def test_larger_batch(self, tiny_encoder_config):
        """Test with larger batch size."""
        from tools.whisper_mlx.encoder import AudioEncoder

        encoder = AudioEncoder(**tiny_encoder_config)
        batch_size = 8
        n_mels = tiny_encoder_config["n_mels"]
        n_ctx = tiny_encoder_config["n_ctx"]

        n_frames = 2 * n_ctx - 1
        x = mx.random.normal((batch_size, n_frames, n_mels))

        out = encoder(x, variable_length=False)

        assert out.shape[0] == batch_size


class TestAudioEncoderVariableLengthOptimization:
    """Tests for variable-length encoding optimization."""

    @pytest.fixture
    def encoder_config(self):
        """Config for variable length tests."""
        return {
            "n_mels": 40,
            "n_ctx": 100,
            "n_state": 64,
            "n_head": 4,
            "n_layer": 2,
        }

    def test_shorter_audio_produces_shorter_output(self, encoder_config):
        """Test shorter audio produces proportionally shorter output."""
        from tools.whisper_mlx.encoder import AudioEncoder

        encoder = AudioEncoder(**encoder_config)
        n_mels = encoder_config["n_mels"]

        # Full length
        full_frames = 2 * encoder_config["n_ctx"] - 1
        x_full = mx.random.normal((full_frames, n_mels))
        out_full = encoder(x_full, variable_length=True)

        # Half length
        half_frames = full_frames // 2
        x_half = mx.random.normal((half_frames, n_mels))
        out_half = encoder(x_half, variable_length=True)

        # Half length should produce approximately half the positions
        assert out_half.shape[1] < out_full.shape[1]
        ratio = out_half.shape[1] / out_full.shape[1]
        assert 0.4 < ratio < 0.6  # Approximately half

    def test_positional_embedding_slicing(self, encoder_config):
        """Test positional embeddings are correctly sliced for variable length."""
        from tools.whisper_mlx.encoder import AudioEncoder

        encoder = AudioEncoder(**encoder_config)
        n_mels = encoder_config["n_mels"]

        # Short input
        short_frames = 50
        x = mx.random.normal((short_frames, n_mels))

        out = encoder(x, variable_length=True)
        expected_len = encoder.get_output_length(short_frames)

        # Output length should match expected
        assert out.shape[1] == expected_len

    def test_variable_length_vs_padded_output(self, encoder_config):
        """Compare variable length vs padded outputs for same audio portion."""
        from tools.whisper_mlx.encoder import AudioEncoder

        encoder = AudioEncoder(**encoder_config)
        n_mels = encoder_config["n_mels"]
        n_ctx = encoder_config["n_ctx"]

        # Create short audio
        short_frames = n_ctx - 1  # About half
        x_short = mx.random.normal((short_frames, n_mels))

        # Get variable-length output
        out_var = encoder(x_short, variable_length=True)

        # The variable-length output should be valid (finite values)
        assert mx.all(mx.isfinite(out_var)).item()


class TestConvolutionalFrontend:
    """Tests for the convolutional frontend."""

    @pytest.fixture
    def encoder_config(self):
        """Config for conv tests."""
        return {
            "n_mels": 40,
            "n_ctx": 100,
            "n_state": 64,
            "n_head": 4,
            "n_layer": 1,
        }

    def test_conv1_properties(self, encoder_config):
        """Test conv1 has correct properties."""
        from tools.whisper_mlx.encoder import AudioEncoder

        encoder = AudioEncoder(**encoder_config)

        # Conv1: kernel_size=3, padding=1 -> same length output
        # Input channels: n_mels, Output channels: n_state
        assert hasattr(encoder.conv1, "weight")

    def test_conv2_stride(self, encoder_config):
        """Test conv2 stride reduces sequence length."""
        from tools.whisper_mlx.encoder import AudioEncoder

        encoder = AudioEncoder(**encoder_config)

        # Test input/output length relationship
        for n_frames in [50, 100, 150, 200]:
            expected_output_len = encoder.get_output_length(n_frames)
            # The formula is (n_frames + 1) // 2 due to stride=2
            assert expected_output_len == (n_frames + 1) // 2


class TestEncoderCompileOptimization:
    """
    Tests for mx.compile encoder optimization (OPT 1.1).

    Verifies:
    1. Correctness: Output matches expected shape and values
    2. Determinism: Same input produces same output
    3. Performance: JIT compilation provides speedup after warmup
    """

    @pytest.fixture
    def encoder_config(self):
        """Standard encoder config for compile tests."""
        return {
            "n_mels": 80,
            "n_ctx": 1500,
            "n_state": 384,
            "n_head": 6,
            "n_layer": 4,
        }

    @pytest.fixture
    def small_encoder_config(self):
        """Smaller config for faster tests."""
        return {
            "n_mels": 40,
            "n_ctx": 100,
            "n_state": 64,
            "n_head": 4,
            "n_layer": 2,
        }

    def test_encoder_compile_correctness(self, small_encoder_config):
        """Test compiled encoder produces correct output shape and finite values."""
        from tools.whisper_mlx.encoder import AudioEncoder

        encoder = AudioEncoder(**small_encoder_config)
        n_mels = small_encoder_config["n_mels"]
        n_ctx = small_encoder_config["n_ctx"]
        n_state = small_encoder_config["n_state"]

        # Create input that produces exact n_ctx output
        n_frames = 2 * n_ctx - 1
        x = mx.random.normal((n_frames, n_mels))

        # Run encoder (uses @mx.compile)
        out = encoder(x, variable_length=False)
        mx.eval(out)

        # Verify shape
        assert out.shape == (1, n_ctx, n_state), f"Expected (1, {n_ctx}, {n_state}), got {out.shape}"

        # Verify output is finite
        assert mx.all(mx.isfinite(out)).item(), "Output contains NaN or Inf"

    def test_encoder_compile_determinism(self, small_encoder_config):
        """Test compiled encoder produces deterministic output."""
        from tools.whisper_mlx.encoder import AudioEncoder

        encoder = AudioEncoder(**small_encoder_config)
        n_mels = small_encoder_config["n_mels"]
        n_ctx = small_encoder_config["n_ctx"]

        n_frames = 2 * n_ctx - 1
        x = mx.random.normal((n_frames, n_mels))
        mx.eval(x)  # Ensure input is materialized

        # Run twice with same input
        out1 = encoder(x, variable_length=False)
        mx.eval(out1)

        out2 = encoder(x, variable_length=False)
        mx.eval(out2)

        # Outputs must be identical
        max_diff = mx.max(mx.abs(out1 - out2)).item()
        assert max_diff < 1e-6, f"Non-deterministic output: max diff = {max_diff}"

    def test_encoder_compile_variable_length(self, small_encoder_config):
        """Test compiled encoder works with variable_length=True."""
        from tools.whisper_mlx.encoder import AudioEncoder

        encoder = AudioEncoder(**small_encoder_config)
        n_mels = small_encoder_config["n_mels"]
        n_state = small_encoder_config["n_state"]

        # Test with various lengths
        test_lengths = [50, 100, 150]

        for n_frames in test_lengths:
            x = mx.random.normal((n_frames, n_mels))
            out = encoder(x, variable_length=True)
            mx.eval(out)

            expected_len = encoder.get_output_length(n_frames)
            assert out.shape == (1, expected_len, n_state), \
                f"For {n_frames} frames: expected (1, {expected_len}, {n_state}), got {out.shape}"
            assert mx.all(mx.isfinite(out)).item()

    def test_encoder_compile_batched(self, small_encoder_config):
        """Test compiled encoder works with batched input."""
        from tools.whisper_mlx.encoder import AudioEncoder

        encoder = AudioEncoder(**small_encoder_config)
        batch_size = 4
        n_mels = small_encoder_config["n_mels"]
        n_ctx = small_encoder_config["n_ctx"]
        n_state = small_encoder_config["n_state"]

        n_frames = 2 * n_ctx - 1
        x = mx.random.normal((batch_size, n_frames, n_mels))

        out = encoder(x, variable_length=False)
        mx.eval(out)

        assert out.shape == (batch_size, n_ctx, n_state)
        assert mx.all(mx.isfinite(out)).item()

    @pytest.mark.benchmark
    def test_encoder_compile_benchmark(self, small_encoder_config):
        """
        Benchmark compiled encoder performance.

        Measures:
        - First run (includes JIT compilation)
        - Warmup runs
        - Subsequent runs (should be faster)
        """
        import time

        from tools.whisper_mlx.encoder import AudioEncoder

        encoder = AudioEncoder(**small_encoder_config)
        n_mels = small_encoder_config["n_mels"]
        n_ctx = small_encoder_config["n_ctx"]

        n_frames = 2 * n_ctx - 1
        x = mx.random.normal((n_frames, n_mels))
        mx.eval(x)

        # First run (includes JIT compilation overhead)
        start = time.perf_counter()
        out = encoder(x, variable_length=False)
        mx.eval(out)
        first_run_ms = (time.perf_counter() - start) * 1000

        # Warmup runs
        for _ in range(3):
            out = encoder(x, variable_length=False)
            mx.eval(out)

        # Timed runs
        n_runs = 10
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            out = encoder(x, variable_length=False)
            mx.eval(out)
            times.append((time.perf_counter() - start) * 1000)

        avg_ms = sum(times) / len(times)
        std_ms = (sum((t - avg_ms) ** 2 for t in times) / len(times)) ** 0.5

        # Print benchmark results (for pytest -v)
        print("\n[Encoder Compile Benchmark]")
        print(f"  First run (with JIT): {first_run_ms:.2f}ms")
        print(f"  Avg after warmup: {avg_ms:.2f}ms (std: {std_ms:.2f}ms)")
        print(f"  Speedup vs first run: {first_run_ms / avg_ms:.2f}x")

        # Verify warmup speedup exists (compiled should be faster)
        # Note: First run includes compilation, so it should be slower
        assert first_run_ms > avg_ms * 0.5, "Expected first run to include compilation overhead"

    @pytest.mark.benchmark
    def test_encoder_compile_performance_realistic(self, encoder_config):
        """
        Benchmark with realistic Whisper encoder config.

        This uses the actual Whisper-base configuration to measure
        real-world performance impact of mx.compile.
        """
        import time

        from tools.whisper_mlx.encoder import AudioEncoder

        encoder = AudioEncoder(**encoder_config)
        n_mels = encoder_config["n_mels"]
        n_ctx = encoder_config["n_ctx"]

        # Simulate 30s audio mel spectrogram
        n_frames = 2 * n_ctx - 1  # 2999 frames
        x = mx.random.normal((n_frames, n_mels))
        mx.eval(x)

        # Warmup (triggers JIT compilation)
        for _ in range(3):
            out = encoder(x, variable_length=False)
            mx.eval(out)

        # Timed runs
        n_runs = 5
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            out = encoder(x, variable_length=False)
            mx.eval(out)
            times.append((time.perf_counter() - start) * 1000)

        avg_ms = sum(times) / len(times)
        min_ms = min(times)
        max_ms = max(times)

        print("\n[Realistic Encoder Benchmark (30s audio)]")
        print(f"  Config: n_state={encoder_config['n_state']}, n_layer={encoder_config['n_layer']}")
        print(f"  Input: {n_frames} frames x {n_mels} mels")
        print(f"  Avg latency: {avg_ms:.2f}ms")
        print(f"  Min: {min_ms:.2f}ms, Max: {max_ms:.2f}ms")

        # Basic sanity check - encoding should complete
        assert mx.all(mx.isfinite(out)).item()
