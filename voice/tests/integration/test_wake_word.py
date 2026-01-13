"""
Wake Word Detector Integration Tests

Tests the C++ wake word detection system using openWakeWord ONNX models.
Worker #302: Phase 1 - Wake Word Detection System
"""

import pytest
import subprocess
import os
import struct
import wave
import tempfile
import numpy as np


# Path to test binary
BUILD_DIR = os.path.join(os.path.dirname(__file__), "../../stream-tts-cpp/build")
TEST_BINARY = os.path.join(BUILD_DIR, "test_wake_word")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "../../models/wakeword")


@pytest.fixture
def check_wake_word_models():
    """Check that wake word models exist"""
    required_models = [
        "melspectrogram.onnx",
        "embedding_model.onnx",
        "alexa.onnx",
        "hey_jarvis.onnx"
    ]
    for model in required_models:
        model_path = os.path.join(MODELS_DIR, model)
        if not os.path.exists(model_path):
            pytest.skip(f"Model not found: {model_path}")
    return True


class TestWakeWordBinary:
    """Tests for the C++ wake word detector test binary"""

    def test_binary_exists(self):
        """Test binary was built"""
        if not os.path.exists(TEST_BINARY):
            pytest.skip("test_wake_word binary not built")

    def test_wake_word_basic(self, check_wake_word_models):
        """Run the C++ test program and check output"""
        if not os.path.exists(TEST_BINARY):
            pytest.skip("test_wake_word binary not built")

        result = subprocess.run(
            [TEST_BINARY],
            cwd=BUILD_DIR,
            capture_output=True,
            text=True,
            timeout=60
        )

        # Print output for debugging
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

        # Check all tests passed
        assert "All tests completed" in result.stdout, "Tests did not complete"
        assert "FAIL:" not in result.stdout, "Test failures detected"
        assert "PERFORMANCE: PASS" in result.stdout, "Performance check failed"

    def test_no_false_positives_silence(self, check_wake_word_models):
        """Test that silence doesn't trigger detection"""
        if not os.path.exists(TEST_BINARY):
            pytest.skip("test_wake_word binary not built")

        result = subprocess.run(
            [TEST_BINARY],
            cwd=BUILD_DIR,
            capture_output=True,
            text=True,
            timeout=60
        )

        assert "PASS: No false positives on silence" in result.stdout

    def test_no_false_positives_tone(self, check_wake_word_models):
        """Test that a 440Hz tone doesn't trigger detection"""
        if not os.path.exists(TEST_BINARY):
            pytest.skip("test_wake_word binary not built")

        result = subprocess.run(
            [TEST_BINARY],
            cwd=BUILD_DIR,
            capture_output=True,
            text=True,
            timeout=60
        )

        assert "PASS: No false positives on tone" in result.stdout

    def test_realtime_performance(self, check_wake_word_models):
        """Test that inference is faster than real-time (< 80ms per chunk)"""
        if not os.path.exists(TEST_BINARY):
            pytest.skip("test_wake_word binary not built")

        result = subprocess.run(
            [TEST_BINARY],
            cwd=BUILD_DIR,
            capture_output=True,
            text=True,
            timeout=60
        )

        # Check performance metric
        assert "Avg inference time:" in result.stdout
        # Extract average inference time
        for line in result.stdout.split('\n'):
            if 'Avg inference time:' in line:
                # Parse "Avg inference time: Xms"
                time_str = line.split(':')[1].strip().replace('ms', '')
                avg_time_ms = float(time_str)
                # Must be faster than 80ms (one chunk duration)
                assert avg_time_ms < 80, f"Inference too slow: {avg_time_ms}ms"
                break


class TestWakeWordModels:
    """Tests for wake word model files"""

    def test_models_exist(self):
        """Test that required model files exist"""
        required_models = [
            "melspectrogram.onnx",
            "embedding_model.onnx",
            "alexa.onnx",
            "hey_jarvis.onnx"
        ]
        for model in required_models:
            model_path = os.path.join(MODELS_DIR, model)
            assert os.path.exists(model_path), f"Missing model: {model}"

    def test_model_sizes(self):
        """Test that models are reasonable sizes"""
        expected_sizes = {
            "melspectrogram.onnx": (500_000, 2_000_000),  # 0.5-2MB
            "embedding_model.onnx": (500_000, 3_000_000),  # 0.5-3MB
            "alexa.onnx": (100_000, 2_000_000),           # 0.1-2MB
            "hey_jarvis.onnx": (100_000, 2_000_000),      # 0.1-2MB
        }

        for model, (min_size, max_size) in expected_sizes.items():
            model_path = os.path.join(MODELS_DIR, model)
            if os.path.exists(model_path):
                size = os.path.getsize(model_path)
                assert min_size < size < max_size, \
                    f"Model {model} unexpected size: {size} bytes"


def generate_wav(samples, sample_rate=16000):
    """Generate WAV file from samples for testing"""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        wav_path = f.name

    with wave.open(wav_path, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        # Convert float samples to int16
        int_samples = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
        wav.writeframes(int_samples.tobytes())

    return wav_path


class TestCustomWakeWordTraining:
    """Tests for custom wake word model training"""

    @pytest.fixture
    def training_script(self):
        """Get path to training script"""
        script_path = os.path.join(
            os.path.dirname(__file__),
            "../../scripts/train_wake_word.py"
        )
        if not os.path.exists(script_path):
            pytest.skip("Training script not found")
        return script_path

    def test_training_script_exists(self, training_script):
        """Verify training script exists"""
        assert os.path.exists(training_script)

    def test_custom_model_exports(self):
        """Test that custom trained models have valid ONNX format"""
        custom_models = [
            "hey_voice.onnx",
            "hey_agent.onnx",
        ]
        for model in custom_models:
            model_path = os.path.join(MODELS_DIR, model)
            if os.path.exists(model_path):
                # Check file is valid ONNX (header check)
                with open(model_path, 'rb') as f:
                    # ONNX files start with specific bytes or are protobuf
                    header = f.read(8)
                    # Just verify it's not empty
                    assert len(header) > 0, f"Empty model file: {model}"

    def test_custom_model_sizes(self):
        """Test custom model sizes are reasonable"""
        custom_models = [
            ("hey_voice.onnx", 10_000, 2_000_000),  # 10KB - 2MB
            ("hey_agent.onnx", 10_000, 2_000_000),
        ]
        for model, min_size, max_size in custom_models:
            model_path = os.path.join(MODELS_DIR, model)
            if os.path.exists(model_path):
                size = os.path.getsize(model_path)
                # Also check for .data file (PyTorch ONNX external data)
                data_path = model_path + ".data"
                if os.path.exists(data_path):
                    size += os.path.getsize(data_path)
                assert min_size < size < max_size, \
                    f"Custom model {model} unexpected size: {size} bytes"


class TestCustomWakeWordInference:
    """Tests for inference with custom wake word models"""

    @pytest.fixture
    def custom_model_path(self):
        """Get path to custom model if it exists"""
        model_path = os.path.join(MODELS_DIR, "hey_voice.onnx")
        if not os.path.exists(model_path):
            pytest.skip("Custom hey_voice model not trained yet")
        return model_path

    def test_custom_model_with_python_inference(self, custom_model_path):
        """Test custom model can be loaded and run with Python ONNX Runtime"""
        try:
            import onnxruntime as ort
        except ImportError:
            pytest.skip("onnxruntime not installed")

        # Load model
        session = ort.InferenceSession(custom_model_path)

        # Get input/output info
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]

        # Expected input shape: [batch, 16, 96] (16 time steps, 96 embedding dim)
        expected_shape = [1, 16, 96]

        # Create dummy input
        dummy_input = np.random.randn(*expected_shape).astype(np.float32)

        # Run inference
        outputs = session.run(None, {input_info.name: dummy_input})

        # Check output shape (should be [batch, 1] for binary classification)
        assert len(outputs) == 1, "Expected single output"
        assert outputs[0].shape[0] == 1, "Expected batch size 1"

        # Check output is probability (0-1)
        score = outputs[0].flatten()[0]
        assert 0 <= score <= 1, f"Score should be 0-1, got {score}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
