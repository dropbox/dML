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
Confidence Calibration for Rich Audio Understanding.

Calibrates model confidence scores so they accurately reflect prediction accuracy.
A well-calibrated model with 80% confidence should be correct 80% of the time.

Key components:
- TemperatureScaling: Simple post-hoc calibration using a single temperature
- PlattScaling: Logistic regression calibration with scale and shift
- ConfidenceCalibrator: Unified interface for all calibration methods

Usage:
    from tools.whisper_mlx.confidence_calibration import ConfidenceCalibrator

    # Create calibrator
    calibrator = ConfidenceCalibrator(method="temperature")

    # Fit on validation data
    calibrator.fit(logits, labels)

    # Apply calibration
    calibrated_conf = calibrator.calibrate(raw_confidence)

    # Compute reliability metrics
    ece = calibrator.expected_calibration_error(probs, labels)

Reference:
- Guo et al. "On Calibration of Modern Neural Networks" (ICML 2017)
- Platt "Probabilistic Outputs for SVMs" (1999)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tools.whisper_mlx.rich_ctc_head import RichToken

import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as nn

    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None
    nn = None


class CalibrationMethod(Enum):
    """Available calibration methods."""

    TEMPERATURE = "temperature"  # Single temperature parameter
    PLATT = "platt"  # Logistic regression (scale + shift)
    ISOTONIC = "isotonic"  # Non-parametric isotonic regression
    NONE = "none"  # No calibration (pass-through)


@dataclass
class CalibrationConfig:
    """Configuration for confidence calibration."""

    method: CalibrationMethod = CalibrationMethod.TEMPERATURE
    n_bins: int = 15  # Number of bins for ECE computation

    # Temperature scaling
    initial_temperature: float = 1.5

    # Platt scaling
    platt_lr: float = 0.01
    platt_iterations: int = 100

    # Isotonic regression
    isotonic_increasing: bool = True

    # Per-output calibration
    calibrate_text: bool = True
    calibrate_emotion: bool = True
    calibrate_phoneme: bool = True
    calibrate_para: bool = True


@dataclass
class CalibrationParams:
    """Learned calibration parameters."""

    method: CalibrationMethod
    temperature: float = 1.0  # For temperature scaling
    platt_a: float = 1.0  # For Platt scaling: 1 / (1 + exp(a * conf + b))
    platt_b: float = 0.0
    isotonic_bins: list[float] | None = None  # For isotonic regression
    isotonic_values: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "method": self.method.value,
            "temperature": self.temperature,
            "platt_a": self.platt_a,
            "platt_b": self.platt_b,
            "isotonic_bins": self.isotonic_bins,
            "isotonic_values": self.isotonic_values,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationParams:
        """Create from dictionary."""
        return cls(
            method=CalibrationMethod(data["method"]),
            temperature=data.get("temperature", 1.0),
            platt_a=data.get("platt_a", 1.0),
            platt_b=data.get("platt_b", 0.0),
            isotonic_bins=data.get("isotonic_bins"),
            isotonic_values=data.get("isotonic_values"),
        )


class TemperatureScaling:
    """
    Temperature scaling calibration.

    Learns a single temperature T to scale logits: softmax(logits / T).
    Simple but effective for most neural networks.

    Reference: Guo et al. "On Calibration of Modern Neural Networks" (ICML 2017)
    """

    def __init__(self, initial_temperature: float = 1.5):
        """Initialize temperature scaling."""
        self.temperature = initial_temperature
        self._fitted = False

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        lr: float = 0.01,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> float:
        """
        Fit temperature using gradient descent on NLL loss.

        Args:
            logits: Model logits, shape (N, C) or (N,)
            labels: Ground truth labels, shape (N,)
            lr: Learning rate
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance

        Returns:
            Optimal temperature
        """
        # Handle binary case: convert 1D or (N, 1) logits to 2-class format
        if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1):
            # Binary: logit represents class 1, class 0 is -logit
            logits_flat = logits.flatten()
            logits = np.stack([-logits_flat, logits_flat], axis=-1)

        N = logits.shape[0]
        logits.shape[1]
        T = self.temperature

        for _iteration in range(max_iterations):
            # Compute scaled softmax
            scaled_logits = logits / T
            max_logits = np.max(scaled_logits, axis=-1, keepdims=True)
            exp_logits = np.exp(scaled_logits - max_logits)
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

            # NLL loss
            labels_int = labels.astype(int)
            label_probs = probs[np.arange(N), labels_int]
            -np.mean(np.log(np.clip(label_probs, 1e-10, 1.0)))

            # Gradient of NLL w.r.t. temperature
            # d(NLL)/dT = 1/T^2 * sum_i(logits_i * (p_i - y_i))
            one_hot = np.zeros_like(probs)
            one_hot[np.arange(N), labels_int] = 1.0
            grad = np.mean(
                np.sum(logits * (probs - one_hot), axis=-1),
            ) / (T * T)

            # Update temperature
            T_new = T - lr * grad
            T_new = max(0.1, min(10.0, T_new))  # Clamp

            if abs(T_new - T) < tolerance:
                break

            T = T_new

        self.temperature = T
        self._fitted = True
        return T

    def calibrate(self, confidence: float | np.ndarray) -> float | np.ndarray:
        """
        Apply temperature scaling to confidence scores.

        For a confidence score p, returns calibrated confidence.

        Args:
            confidence: Raw confidence score(s) in [0, 1]

        Returns:
            Calibrated confidence score(s)
        """
        # Convert confidence back to logit, scale, convert back
        # For binary: logit = log(p / (1-p))
        # For softmax: we approximate by scaling the "effective logit"
        conf = np.asarray(confidence)
        conf = np.clip(conf, 1e-10, 1.0 - 1e-10)

        # Effective logit transformation
        logit = np.log(conf / (1.0 - conf))
        scaled_logit = logit / self.temperature
        calibrated = 1.0 / (1.0 + np.exp(-scaled_logit))

        return float(calibrated) if np.isscalar(confidence) else calibrated

    def calibrate_logits(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to logits and return probabilities."""
        scaled = logits / self.temperature
        max_logits = np.max(scaled, axis=-1, keepdims=True)
        exp_logits = np.exp(scaled - max_logits)
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


class PlattScaling:
    """
    Platt scaling calibration.

    Learns parameters A and B for sigmoid: 1 / (1 + exp(A * logit + B)).
    More flexible than temperature scaling.

    Reference: Platt "Probabilistic Outputs for SVMs" (1999)
    """

    def __init__(self):
        """Initialize Platt scaling."""
        self.A = 1.0
        self.B = 0.0
        self._fitted = False

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        lr: float = 0.01,
        max_iterations: int = 100,
    ) -> tuple[float, float]:
        """
        Fit Platt scaling parameters using gradient descent.

        Args:
            logits: Model logits or confidences, shape (N,) or (N, C)
            labels: Ground truth labels, shape (N,)
            lr: Learning rate
            max_iterations: Maximum iterations

        Returns:
            Tuple of (A, B) parameters
        """
        # If multi-class, use max logit
        if logits.ndim > 1:
            max_logits = np.max(logits, axis=-1)
            pred_labels = np.argmax(logits, axis=-1)
            correct = (pred_labels == labels).astype(float)
            logits_1d = max_logits
            labels_1d = correct
        else:
            logits_1d = logits
            labels_1d = labels.astype(float)

        len(logits_1d)
        A, B = self.A, self.B

        for _ in range(max_iterations):
            # Sigmoid: p = 1 / (1 + exp(A * x + B))
            z = A * logits_1d + B
            p = 1.0 / (1.0 + np.exp(np.clip(-z, -500, 500)))

            # Binary cross-entropy loss gradient
            grad_A = np.mean((p - labels_1d) * logits_1d)
            grad_B = np.mean(p - labels_1d)

            A = A - lr * grad_A
            B = B - lr * grad_B

        self.A = A
        self.B = B
        self._fitted = True
        return A, B

    def calibrate(self, confidence: float | np.ndarray) -> float | np.ndarray:
        """Apply Platt scaling to confidence scores."""
        conf = np.asarray(confidence)
        conf = np.clip(conf, 1e-10, 1.0 - 1e-10)

        # Convert confidence to logit
        logit = np.log(conf / (1.0 - conf))

        # Apply Platt scaling
        z = self.A * logit + self.B
        calibrated = 1.0 / (1.0 + np.exp(np.clip(-z, -500, 500)))

        return float(calibrated) if np.isscalar(confidence) else calibrated


class IsotonicCalibration:
    """
    Isotonic regression calibration.

    Non-parametric method that learns a monotonic mapping from
    raw confidence to calibrated confidence.
    """

    def __init__(self):
        """Initialize isotonic calibration."""
        self.bins: np.ndarray | None = None
        self.values: np.ndarray | None = None
        self._fitted = False

    def fit(
        self,
        confidences: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 100,
    ) -> None:
        """
        Fit isotonic regression.

        Args:
            confidences: Raw confidence scores, shape (N,)
            labels: Ground truth labels (binary), shape (N,)
            n_bins: Number of bins for binning
        """
        # Sort by confidence
        sorted_idx = np.argsort(confidences)
        sorted_conf = confidences[sorted_idx]
        sorted_labels = labels[sorted_idx].astype(float)

        # Pool adjacent violators algorithm (simplified)
        # Create bins
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_values = np.zeros(n_bins)

        for i in range(n_bins):
            mask = (sorted_conf >= bin_edges[i]) & (sorted_conf < bin_edges[i + 1])
            if np.sum(mask) > 0:
                bin_values[i] = np.mean(sorted_labels[mask])
            else:
                bin_values[i] = bin_centers[i]

        # Enforce monotonicity with pool-adjacent-violators
        bin_values = self._pool_adjacent_violators(bin_values)

        self.bins = bin_centers
        self.values = bin_values
        self._fitted = True

    def _pool_adjacent_violators(self, values: np.ndarray) -> np.ndarray:
        """Pool adjacent violators to ensure monotonicity."""
        result = values.copy()
        n = len(result)

        # Forward pass
        for i in range(1, n):
            if result[i] < result[i - 1]:
                result[i] = result[i - 1]

        return result

    def calibrate(self, confidence: float | np.ndarray) -> float | np.ndarray:
        """Apply isotonic calibration via interpolation."""
        if not self._fitted or self.bins is None or self.values is None:
            return confidence

        conf = np.asarray(confidence)
        calibrated = np.interp(conf, self.bins, self.values)

        return float(calibrated) if np.isscalar(confidence) else calibrated


class ConfidenceCalibrator:
    """
    Unified confidence calibration interface.

    Supports multiple calibration methods and per-output calibration.

    Example:
        calibrator = ConfidenceCalibrator(method="temperature")

        # Fit on validation data
        calibrator.fit(logits, labels, output_type="text")

        # Apply calibration
        calibrated = calibrator.calibrate(0.95, output_type="text")

        # Compute metrics
        ece = calibrator.expected_calibration_error(probs, labels)
    """

    def __init__(
        self,
        config: CalibrationConfig | None = None,
    ):
        """Initialize calibrator."""
        self.config = config or CalibrationConfig()

        # Per-output calibrators
        self._calibrators: dict[str, Any] = {}
        self._params: dict[str, CalibrationParams] = {}

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        output_type: str = "text",
        **kwargs,
    ) -> CalibrationParams:
        """
        Fit calibration for a specific output type.

        Args:
            logits: Model logits, shape (N, C) or (N,)
            labels: Ground truth labels, shape (N,)
            output_type: Type of output ("text", "emotion", "phoneme", "para")
            **kwargs: Additional arguments for calibration method

        Returns:
            Fitted calibration parameters
        """
        method = self.config.method

        if method == CalibrationMethod.TEMPERATURE:
            calibrator = TemperatureScaling(self.config.initial_temperature)
            temp = calibrator.fit(
                logits, labels,
                lr=kwargs.get("lr", 0.01),
                max_iterations=kwargs.get("max_iterations", 100),
            )
            params = CalibrationParams(
                method=method,
                temperature=temp,
            )

        elif method == CalibrationMethod.PLATT:
            calibrator = PlattScaling()
            A, B = calibrator.fit(
                logits, labels,
                lr=self.config.platt_lr,
                max_iterations=self.config.platt_iterations,
            )
            params = CalibrationParams(
                method=method,
                platt_a=A,
                platt_b=B,
            )

        elif method == CalibrationMethod.ISOTONIC:
            # For isotonic, need confidence scores not logits
            if logits.ndim > 1:
                max_logits = np.max(logits, axis=-1)
                exp_logits = np.exp(max_logits - np.max(max_logits))
                confidences = exp_logits / np.sum(np.exp(logits - np.max(logits, axis=-1, keepdims=True)), axis=-1)
                pred_labels = np.argmax(logits, axis=-1)
                correct = (pred_labels == labels).astype(float)
            else:
                confidences = 1.0 / (1.0 + np.exp(-logits))
                correct = labels.astype(float)

            calibrator = IsotonicCalibration()
            calibrator.fit(confidences, correct, n_bins=self.config.n_bins)
            params = CalibrationParams(
                method=method,
                isotonic_bins=calibrator.bins.tolist() if calibrator.bins is not None else None,
                isotonic_values=calibrator.values.tolist() if calibrator.values is not None else None,
            )

        else:
            calibrator = None
            params = CalibrationParams(method=CalibrationMethod.NONE)

        self._calibrators[output_type] = calibrator
        self._params[output_type] = params
        return params

    def calibrate(
        self,
        confidence: float | np.ndarray,
        output_type: str = "text",
    ) -> float | np.ndarray:
        """
        Apply calibration to confidence scores.

        Args:
            confidence: Raw confidence score(s)
            output_type: Type of output

        Returns:
            Calibrated confidence score(s)
        """
        if output_type not in self._calibrators or self._calibrators[output_type] is None:
            return confidence

        return self._calibrators[output_type].calibrate(confidence)

    def calibrate_rich_token(
        self,
        token: RichToken,
    ) -> RichToken:
        """
        Apply calibration to all confidence fields in a RichToken.

        Args:
            token: RichToken with raw confidence scores

        Returns:
            RichToken with calibrated confidence scores
        """
        from dataclasses import replace

        # Create updated confidence values
        new_confidence = self.calibrate(token.confidence, "text")
        new_emotion_confidence = self.calibrate(token.emotion_confidence, "emotion")

        # Calibrate phoneme confidence
        if token.phoneme_confidence:
            new_phoneme_conf = [
                float(self.calibrate(c, "phoneme"))
                for c in token.phoneme_confidence
            ]
        else:
            new_phoneme_conf = token.phoneme_confidence

        # Calibrate para confidence
        new_para_conf = None
        if token.para_confidence is not None:
            new_para_conf = float(self.calibrate(token.para_confidence, "para"))

        return replace(
            token,
            confidence=float(new_confidence),
            emotion_confidence=float(new_emotion_confidence),
            phoneme_confidence=new_phoneme_conf,
            para_confidence=new_para_conf,
        )

    def expected_calibration_error(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        n_bins: int | None = None,
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).

        ECE measures how well confidence matches accuracy across confidence bins.
        Lower is better (0 = perfectly calibrated).

        Args:
            probs: Predicted probabilities/confidences, shape (N,) or (N, C)
            labels: Ground truth labels, shape (N,)
            n_bins: Number of bins (default: config.n_bins)

        Returns:
            ECE value in [0, 1]
        """
        n_bins = n_bins or self.config.n_bins

        if probs.ndim > 1:
            confidences = np.max(probs, axis=-1)
            predictions = np.argmax(probs, axis=-1)
            accuracies = (predictions == labels).astype(float)
        else:
            confidences = probs
            accuracies = labels.astype(float)

        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        total_samples = len(confidences)

        for i in range(n_bins):
            mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
            bin_size = np.sum(mask)

            if bin_size > 0:
                bin_accuracy = np.mean(accuracies[mask])
                bin_confidence = np.mean(confidences[mask])
                ece += (bin_size / total_samples) * abs(bin_accuracy - bin_confidence)

        return float(ece)

    def maximum_calibration_error(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        n_bins: int | None = None,
    ) -> float:
        """
        Compute Maximum Calibration Error (MCE).

        MCE is the maximum gap between confidence and accuracy across bins.

        Args:
            probs: Predicted probabilities/confidences
            labels: Ground truth labels
            n_bins: Number of bins

        Returns:
            MCE value in [0, 1]
        """
        n_bins = n_bins or self.config.n_bins

        if probs.ndim > 1:
            confidences = np.max(probs, axis=-1)
            predictions = np.argmax(probs, axis=-1)
            accuracies = (predictions == labels).astype(float)
        else:
            confidences = probs
            accuracies = labels.astype(float)

        bin_edges = np.linspace(0, 1, n_bins + 1)
        mce = 0.0

        for i in range(n_bins):
            mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
            if np.sum(mask) > 0:
                bin_accuracy = np.mean(accuracies[mask])
                bin_confidence = np.mean(confidences[mask])
                mce = max(mce, abs(bin_accuracy - bin_confidence))

        return float(mce)

    def reliability_diagram_data(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        n_bins: int | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Get data for plotting reliability diagram.

        Returns:
            Dictionary with 'bin_centers', 'accuracies', 'confidences', 'counts'
        """
        n_bins = n_bins or self.config.n_bins

        if probs.ndim > 1:
            confidences = np.max(probs, axis=-1)
            predictions = np.argmax(probs, axis=-1)
            correct = (predictions == labels).astype(float)
        else:
            confidences = probs
            correct = labels.astype(float)

        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_accuracies = np.zeros(n_bins)
        bin_confidences = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)

        for i in range(n_bins):
            mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
            bin_counts[i] = np.sum(mask)
            if bin_counts[i] > 0:
                bin_accuracies[i] = np.mean(correct[mask])
                bin_confidences[i] = np.mean(confidences[mask])

        return {
            "bin_centers": bin_centers,
            "accuracies": bin_accuracies,
            "confidences": bin_confidences,
            "counts": bin_counts,
        }

    def get_params(self, output_type: str = "text") -> CalibrationParams | None:
        """Get calibration parameters for an output type."""
        return self._params.get(output_type)

    def save(self, path: str | Path) -> None:
        """Save calibration parameters to file."""
        path = Path(path)
        data = {
            "config": {
                "method": self.config.method.value,
                "n_bins": self.config.n_bins,
                "initial_temperature": self.config.initial_temperature,
            },
            "params": {
                output_type: params.to_dict()
                for output_type, params in self._params.items()
            },
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str | Path) -> None:
        """Load calibration parameters from file."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        # Restore config
        self.config.method = CalibrationMethod(data["config"]["method"])
        self.config.n_bins = data["config"]["n_bins"]
        self.config.initial_temperature = data["config"].get("initial_temperature", 1.5)

        # Restore params and calibrators
        self._params = {}
        self._calibrators = {}

        for output_type, param_dict in data.get("params", {}).items():
            params = CalibrationParams.from_dict(param_dict)
            self._params[output_type] = params

            # Recreate calibrator from params
            if params.method == CalibrationMethod.TEMPERATURE:
                cal = TemperatureScaling(params.temperature)
                cal._fitted = True
                self._calibrators[output_type] = cal

            elif params.method == CalibrationMethod.PLATT:
                cal = PlattScaling()
                cal.A = params.platt_a
                cal.B = params.platt_b
                cal._fitted = True
                self._calibrators[output_type] = cal

            elif params.method == CalibrationMethod.ISOTONIC:
                cal = IsotonicCalibration()
                if params.isotonic_bins and params.isotonic_values:
                    cal.bins = np.array(params.isotonic_bins)
                    cal.values = np.array(params.isotonic_values)
                    cal._fitted = True
                self._calibrators[output_type] = cal


# =============================================================================
# Utility Functions
# =============================================================================


def compute_calibration_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> dict[str, float]:
    """
    Compute all calibration metrics.

    Args:
        probs: Predicted probabilities
        labels: Ground truth labels
        n_bins: Number of bins

    Returns:
        Dictionary with ECE, MCE, and accuracy
    """
    calibrator = ConfidenceCalibrator()
    calibrator.config.n_bins = n_bins

    if probs.ndim > 1:
        predictions = np.argmax(probs, axis=-1)
        accuracy = np.mean(predictions == labels)
    else:
        accuracy = np.mean((probs > 0.5) == labels)

    return {
        "ece": calibrator.expected_calibration_error(probs, labels, n_bins),
        "mce": calibrator.maximum_calibration_error(probs, labels, n_bins),
        "accuracy": float(accuracy),
    }


def calibrate_for_streaming(
    confidence: float,
    calibrator: ConfidenceCalibrator | None = None,
    output_type: str = "text",
    conservative: bool = True,
) -> float:
    """
    Apply calibration for streaming decisions.

    For streaming ASR, we want conservative calibration to avoid
    premature commits. This function optionally applies a conservative
    shift to calibrated confidences.

    Args:
        confidence: Raw confidence score
        calibrator: Confidence calibrator (optional)
        output_type: Type of output
        conservative: If True, shift calibrated confidence down by 0.1

    Returns:
        Calibrated confidence, optionally conservative
    """
    if calibrator is not None:
        calibrated = calibrator.calibrate(confidence, output_type)
    else:
        calibrated = confidence

    if conservative:
        # Conservative shift for streaming (avoid premature commits)
        calibrated = max(0.0, float(calibrated) - 0.1)

    return float(calibrated)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums and configs
    "CalibrationMethod",
    "CalibrationConfig",
    "CalibrationParams",
    # Calibration methods
    "TemperatureScaling",
    "PlattScaling",
    "IsotonicCalibration",
    # Main interface
    "ConfidenceCalibrator",
    # Utilities
    "compute_calibration_metrics",
    "calibrate_for_streaming",
]
