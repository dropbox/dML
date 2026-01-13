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

"""Tests for confidence calibration module."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tools.whisper_mlx.confidence_calibration import (
    CalibrationConfig,
    CalibrationMethod,
    CalibrationParams,
    ConfidenceCalibrator,
    IsotonicCalibration,
    PlattScaling,
    TemperatureScaling,
    calibrate_for_streaming,
    compute_calibration_metrics,
)

# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def binary_classification_data():
    """Generate synthetic binary classification data."""
    rng = np.random.default_rng(42)
    n_samples = 1000

    # Generate logits
    logits = rng.standard_normal(n_samples)

    # Generate labels (correlated with logits but with noise)
    probs = 1.0 / (1.0 + np.exp(-logits))
    labels = (rng.random(n_samples) < probs).astype(int)

    return logits, labels


@pytest.fixture
def multiclass_data():
    """Generate synthetic multiclass classification data."""
    rng = np.random.default_rng(42)
    n_samples = 1000
    n_classes = 8

    # Generate logits
    logits = rng.standard_normal((n_samples, n_classes))

    # Generate labels
    labels = np.argmax(logits + rng.standard_normal((n_samples, n_classes)) * 0.5, axis=1)

    return logits, labels


@pytest.fixture
def overconfident_model_data():
    """Generate data from an overconfident model (typical NN behavior)."""
    rng = np.random.default_rng(42)
    n_samples = 500

    # Overconfident: high confidence even when wrong
    logits = rng.standard_normal((n_samples, 4)) * 2  # Large magnitudes

    # True labels have some randomness
    true_probs = np.array([0.4, 0.3, 0.2, 0.1])
    labels = rng.choice(4, n_samples, p=true_probs)

    return logits, labels


# =============================================================================
# Temperature Scaling Tests
# =============================================================================


class TestTemperatureScaling:
    """Tests for TemperatureScaling class."""

    def test_init(self):
        """Test initialization."""
        ts = TemperatureScaling(initial_temperature=1.5)
        assert ts.temperature == 1.5
        assert not ts._fitted

    def test_fit_binary(self, binary_classification_data):
        """Test fitting on binary classification data."""
        logits, labels = binary_classification_data
        logits_2d = logits.reshape(-1, 1)
        labels = labels.astype(int)

        ts = TemperatureScaling(initial_temperature=1.0)
        temp = ts.fit(logits_2d, labels)

        assert ts._fitted
        assert 0.1 <= temp <= 10.0  # Should be within bounds
        assert temp == ts.temperature

    def test_fit_multiclass(self, multiclass_data):
        """Test fitting on multiclass data."""
        logits, labels = multiclass_data

        ts = TemperatureScaling(initial_temperature=1.0)
        temp = ts.fit(logits, labels)

        assert ts._fitted
        assert 0.1 <= temp <= 10.0

    def test_calibrate_scalar(self):
        """Test calibrating a single confidence value."""
        ts = TemperatureScaling(initial_temperature=1.5)
        ts._fitted = True

        calibrated = ts.calibrate(0.9)
        assert 0 <= calibrated <= 1

        # Higher temperature should reduce confidence
        ts2 = TemperatureScaling(initial_temperature=3.0)
        ts2._fitted = True
        calibrated2 = ts2.calibrate(0.9)

        assert calibrated2 < calibrated  # Higher temp = lower confidence

    def test_calibrate_array(self):
        """Test calibrating an array of confidences."""
        ts = TemperatureScaling(initial_temperature=1.5)
        ts._fitted = True

        confidences = np.array([0.5, 0.7, 0.9, 0.99])
        calibrated = ts.calibrate(confidences)

        assert calibrated.shape == confidences.shape
        assert np.all((calibrated >= 0) & (calibrated <= 1))

    def test_calibrate_logits(self, multiclass_data):
        """Test calibrating logits to probabilities."""
        logits, _ = multiclass_data

        ts = TemperatureScaling(initial_temperature=2.0)
        probs = ts.calibrate_logits(logits[:10])

        assert probs.shape == logits[:10].shape
        assert np.allclose(np.sum(probs, axis=1), 1.0)  # Sum to 1


# =============================================================================
# Platt Scaling Tests
# =============================================================================


class TestPlattScaling:
    """Tests for PlattScaling class."""

    def test_init(self):
        """Test initialization."""
        ps = PlattScaling()
        assert ps.A == 1.0
        assert ps.B == 0.0
        assert not ps._fitted

    def test_fit(self, binary_classification_data):
        """Test fitting Platt scaling."""
        logits, labels = binary_classification_data

        ps = PlattScaling()
        A, B = ps.fit(logits, labels)

        assert ps._fitted
        assert ps.A == A
        assert ps.B == B

    def test_fit_multiclass(self, multiclass_data):
        """Test fitting on multiclass data (uses max logit)."""
        logits, labels = multiclass_data

        ps = PlattScaling()
        A, B = ps.fit(logits, labels)

        assert ps._fitted

    def test_calibrate(self):
        """Test calibration."""
        ps = PlattScaling()
        ps.A = 0.8
        ps.B = -0.1
        ps._fitted = True

        calibrated = ps.calibrate(0.9)
        assert 0 <= calibrated <= 1

    def test_calibrate_array(self):
        """Test calibrating array."""
        ps = PlattScaling()
        ps.A = 1.0
        ps.B = 0.0
        ps._fitted = True

        confidences = np.array([0.3, 0.5, 0.7, 0.9])
        calibrated = ps.calibrate(confidences)

        assert calibrated.shape == confidences.shape
        assert np.all((calibrated >= 0) & (calibrated <= 1))


# =============================================================================
# Isotonic Calibration Tests
# =============================================================================


class TestIsotonicCalibration:
    """Tests for IsotonicCalibration class."""

    def test_init(self):
        """Test initialization."""
        ic = IsotonicCalibration()
        assert ic.bins is None
        assert ic.values is None
        assert not ic._fitted

    def test_fit(self, binary_classification_data):
        """Test fitting isotonic calibration."""
        logits, labels = binary_classification_data
        confidences = 1.0 / (1.0 + np.exp(-logits))

        ic = IsotonicCalibration()
        ic.fit(confidences, labels, n_bins=20)

        assert ic._fitted
        assert ic.bins is not None
        assert ic.values is not None
        assert len(ic.bins) == 20
        assert len(ic.values) == 20

    def test_monotonicity(self, binary_classification_data):
        """Test that isotonic values are monotonically increasing."""
        logits, labels = binary_classification_data
        confidences = 1.0 / (1.0 + np.exp(-logits))

        ic = IsotonicCalibration()
        ic.fit(confidences, labels)

        # Values should be monotonically increasing
        assert np.all(np.diff(ic.values) >= 0)

    def test_calibrate(self, binary_classification_data):
        """Test calibration."""
        logits, labels = binary_classification_data
        confidences = 1.0 / (1.0 + np.exp(-logits))

        ic = IsotonicCalibration()
        ic.fit(confidences, labels)

        calibrated = ic.calibrate(0.7)
        assert 0 <= calibrated <= 1

    def test_calibrate_unfitted(self):
        """Test that unfitted calibrator returns original confidence."""
        ic = IsotonicCalibration()
        result = ic.calibrate(0.8)
        assert result == 0.8


# =============================================================================
# CalibrationParams Tests
# =============================================================================


class TestCalibrationParams:
    """Tests for CalibrationParams dataclass."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        params = CalibrationParams(
            method=CalibrationMethod.TEMPERATURE,
            temperature=1.5,
        )
        d = params.to_dict()

        assert d["method"] == "temperature"
        assert d["temperature"] == 1.5

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        d = {
            "method": "platt",
            "platt_a": 0.9,
            "platt_b": -0.2,
        }
        params = CalibrationParams.from_dict(d)

        assert params.method == CalibrationMethod.PLATT
        assert params.platt_a == 0.9
        assert params.platt_b == -0.2

    def test_round_trip(self):
        """Test serialization round trip."""
        params = CalibrationParams(
            method=CalibrationMethod.ISOTONIC,
            isotonic_bins=[0.1, 0.3, 0.5, 0.7, 0.9],
            isotonic_values=[0.05, 0.25, 0.5, 0.75, 0.95],
        )
        d = params.to_dict()
        restored = CalibrationParams.from_dict(d)

        assert restored.method == params.method
        assert restored.isotonic_bins == params.isotonic_bins
        assert restored.isotonic_values == params.isotonic_values


# =============================================================================
# ConfidenceCalibrator Tests
# =============================================================================


class TestConfidenceCalibrator:
    """Tests for ConfidenceCalibrator class."""

    def test_init_default(self):
        """Test default initialization."""
        calibrator = ConfidenceCalibrator()
        assert calibrator.config.method == CalibrationMethod.TEMPERATURE
        assert calibrator.config.n_bins == 15

    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = CalibrationConfig(
            method=CalibrationMethod.PLATT,
            n_bins=20,
        )
        calibrator = ConfidenceCalibrator(config)
        assert calibrator.config.method == CalibrationMethod.PLATT
        assert calibrator.config.n_bins == 20

    def test_fit_temperature(self, multiclass_data):
        """Test fitting with temperature scaling."""
        logits, labels = multiclass_data

        config = CalibrationConfig(method=CalibrationMethod.TEMPERATURE)
        calibrator = ConfidenceCalibrator(config)

        params = calibrator.fit(logits, labels, output_type="text")

        assert params.method == CalibrationMethod.TEMPERATURE
        assert params.temperature > 0

    def test_fit_platt(self, multiclass_data):
        """Test fitting with Platt scaling."""
        logits, labels = multiclass_data

        config = CalibrationConfig(method=CalibrationMethod.PLATT)
        calibrator = ConfidenceCalibrator(config)

        params = calibrator.fit(logits, labels, output_type="emotion")

        assert params.method == CalibrationMethod.PLATT

    def test_fit_isotonic(self, multiclass_data):
        """Test fitting with isotonic regression."""
        logits, labels = multiclass_data

        config = CalibrationConfig(method=CalibrationMethod.ISOTONIC)
        calibrator = ConfidenceCalibrator(config)

        params = calibrator.fit(logits, labels, output_type="phoneme")

        assert params.method == CalibrationMethod.ISOTONIC

    def test_calibrate_per_output_type(self, multiclass_data):
        """Test calibration for different output types."""
        logits, labels = multiclass_data

        calibrator = ConfidenceCalibrator()

        # Fit for different output types
        calibrator.fit(logits, labels, output_type="text")
        calibrator.fit(logits, labels, output_type="emotion")

        # Calibrate
        text_cal = calibrator.calibrate(0.9, output_type="text")
        emotion_cal = calibrator.calibrate(0.9, output_type="emotion")

        assert 0 <= text_cal <= 1
        assert 0 <= emotion_cal <= 1

    def test_calibrate_unfitted_output(self):
        """Test that unfitted output type returns original confidence."""
        calibrator = ConfidenceCalibrator()

        # Don't fit anything
        result = calibrator.calibrate(0.85, output_type="unknown")
        assert result == 0.85

    def test_expected_calibration_error_perfect(self):
        """Test ECE for perfectly calibrated predictions."""
        calibrator = ConfidenceCalibrator()

        # Create perfectly calibrated data
        rng = np.random.default_rng(42)
        n_samples = 1000

        # Confidence equals accuracy
        confidences = rng.uniform(0.1, 0.9, n_samples)
        labels = (rng.random(n_samples) < confidences).astype(int)

        ece = calibrator.expected_calibration_error(confidences, labels)

        # ECE should be low (but not exactly 0 due to sampling)
        assert ece < 0.1

    def test_expected_calibration_error_overconfident(self, overconfident_model_data):
        """Test ECE for overconfident model."""
        logits, labels = overconfident_model_data

        calibrator = ConfidenceCalibrator()

        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        ece = calibrator.expected_calibration_error(probs, labels)

        # Overconfident model should have higher ECE
        assert ece > 0.1

    def test_maximum_calibration_error(self, multiclass_data):
        """Test MCE computation."""
        logits, labels = multiclass_data

        calibrator = ConfidenceCalibrator()
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

        mce = calibrator.maximum_calibration_error(probs, labels)

        assert 0 <= mce <= 1

    def test_reliability_diagram_data(self, multiclass_data):
        """Test reliability diagram data generation."""
        logits, labels = multiclass_data

        calibrator = ConfidenceCalibrator()
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

        data = calibrator.reliability_diagram_data(probs, labels, n_bins=10)

        assert "bin_centers" in data
        assert "accuracies" in data
        assert "confidences" in data
        assert "counts" in data
        assert len(data["bin_centers"]) == 10

    def test_get_params(self, multiclass_data):
        """Test getting calibration parameters."""
        logits, labels = multiclass_data

        calibrator = ConfidenceCalibrator()
        calibrator.fit(logits, labels, output_type="text")

        params = calibrator.get_params("text")
        assert params is not None
        assert params.method == CalibrationMethod.TEMPERATURE

        params_none = calibrator.get_params("nonexistent")
        assert params_none is None

    def test_save_and_load(self, multiclass_data):
        """Test saving and loading calibration."""
        logits, labels = multiclass_data

        calibrator = ConfidenceCalibrator()
        calibrator.fit(logits, labels, output_type="text")
        calibrator.fit(logits, labels, output_type="emotion")

        # Save
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            save_path = Path(f.name)

        calibrator.save(save_path)

        # Load into new calibrator
        loaded = ConfidenceCalibrator()
        loaded.load(save_path)

        # Verify same calibration
        orig_cal = calibrator.calibrate(0.9, "text")
        loaded_cal = loaded.calibrate(0.9, "text")

        assert abs(orig_cal - loaded_cal) < 1e-6

        # Clean up
        save_path.unlink()

    def test_calibration_modifies_confidence(self, multiclass_data):
        """Test that calibration actually modifies confidence scores."""
        logits, labels = multiclass_data

        # Compute original probs
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        max_probs = np.max(probs, axis=-1)

        calibrator = ConfidenceCalibrator()

        # Fit calibration
        calibrator.fit(logits, labels, output_type="text")

        # Apply calibration
        calibrated_probs = np.array([
            calibrator.calibrate(p, "text") for p in max_probs
        ])

        # Calibration should modify values (not be identity)
        # For typical overconfident networks, temp > 1, so confidence decreases
        # But we just check that SOMETHING changed
        differences = np.abs(calibrated_probs - max_probs)

        # At least some confidences should be modified
        assert np.mean(differences) > 0.001, "Calibration had no effect"

        # All outputs should still be valid probabilities
        assert np.all((calibrated_probs >= 0) & (calibrated_probs <= 1))


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_compute_calibration_metrics(self, multiclass_data):
        """Test compute_calibration_metrics function."""
        logits, labels = multiclass_data
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

        metrics = compute_calibration_metrics(probs, labels)

        assert "ece" in metrics
        assert "mce" in metrics
        assert "accuracy" in metrics
        assert 0 <= metrics["ece"] <= 1
        assert 0 <= metrics["mce"] <= 1
        assert 0 <= metrics["accuracy"] <= 1

    def test_compute_calibration_metrics_binary(self, binary_classification_data):
        """Test compute_calibration_metrics for binary case."""
        logits, labels = binary_classification_data
        probs = 1.0 / (1.0 + np.exp(-logits))

        metrics = compute_calibration_metrics(probs, labels)

        assert 0 <= metrics["accuracy"] <= 1

    def test_calibrate_for_streaming_no_calibrator(self):
        """Test streaming calibration without calibrator."""
        result = calibrate_for_streaming(0.9, calibrator=None, conservative=False)
        assert result == 0.9

    def test_calibrate_for_streaming_conservative(self):
        """Test conservative streaming calibration."""
        result = calibrate_for_streaming(0.9, calibrator=None, conservative=True)
        assert result == 0.8  # Shifted down by 0.1

    def test_calibrate_for_streaming_with_calibrator(self, multiclass_data):
        """Test streaming calibration with calibrator."""
        logits, labels = multiclass_data

        calibrator = ConfidenceCalibrator()
        calibrator.fit(logits, labels, output_type="text")

        result = calibrate_for_streaming(
            0.9,
            calibrator=calibrator,
            output_type="text",
            conservative=False,
        )

        assert 0 <= result <= 1

    def test_calibrate_for_streaming_conservative_clamp(self):
        """Test that conservative calibration clamps to 0."""
        result = calibrate_for_streaming(0.05, calibrator=None, conservative=True)
        assert result == 0.0  # Should be clamped to 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestCalibrationIntegration:
    """Integration tests for calibration system."""

    def test_all_methods_work(self, multiclass_data):
        """Test that all calibration methods can be used."""
        logits, labels = multiclass_data

        for method in CalibrationMethod:
            config = CalibrationConfig(method=method)
            calibrator = ConfidenceCalibrator(config)

            params = calibrator.fit(logits, labels, output_type="test")
            assert params is not None, f"Method {method} returned no params"
            result = calibrator.calibrate(0.8, output_type="test")

            assert 0 <= result <= 1, f"Method {method} failed"

    def test_multiple_output_types(self, multiclass_data):
        """Test calibrating multiple output types independently."""
        logits, labels = multiclass_data

        calibrator = ConfidenceCalibrator()

        # Different configs per output type would require multiple calibrators
        # but we can at least test fitting multiple types
        calibrator.fit(logits, labels, output_type="text")
        calibrator.fit(logits[:, :4], labels % 4, output_type="emotion")

        text_cal = calibrator.calibrate(0.9, "text")
        emotion_cal = calibrator.calibrate(0.9, "emotion")

        # Both should produce valid confidences
        assert 0 <= text_cal <= 1
        assert 0 <= emotion_cal <= 1

    def test_json_serialization(self, multiclass_data):
        """Test that all calibration methods can be JSON serialized."""
        logits, labels = multiclass_data

        for method in [CalibrationMethod.TEMPERATURE, CalibrationMethod.PLATT]:
            config = CalibrationConfig(method=method)
            calibrator = ConfidenceCalibrator(config)
            calibrator.fit(logits, labels, output_type="test")

            params = calibrator.get_params("test")
            json_str = json.dumps(params.to_dict())
            restored = CalibrationParams.from_dict(json.loads(json_str))

            assert restored.method == method


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
