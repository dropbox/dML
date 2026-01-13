#!/usr/bin/env python3
"""PyTorch ecosystem version validation tests.

These tests ensure that PyTorch, torchvision, torchaudio, and libtorch
are all compatible versions to prevent segfaults and ABI issues.

Worker #442: Created after discovering PyTorch 2.9.1 upgrade broke imports
due to torchvision/torchaudio version mismatch.
"""

import subprocess
import sys
import pytest


# Expected versions - update these when upgrading PyTorch
PYTORCH_MIN_VERSION = "2.9.0"
PYTORCH_MAJOR_MINOR = "2.9"


class TestPyTorchVersions:
    """Test PyTorch ecosystem version compatibility."""

    def test_pytorch_version_minimum(self):
        """PyTorch must be at least 2.9.0 for native MPS torch.angle support."""
        import torch

        version = torch.__version__
        major_minor = ".".join(version.split(".")[:2])

        assert version >= PYTORCH_MIN_VERSION, (
            f"PyTorch {version} is too old. "
            f"Need >= {PYTORCH_MIN_VERSION} for native MPS torch.angle() support. "
            f"Run: pip install --upgrade torch"
        )

    def test_torchvision_version_match(self):
        """torchvision must match PyTorch major.minor version."""
        import torch
        try:
            import torchvision
        except ImportError:
            pytest.skip("torchvision not installed (not required for TTS)")

        torch_version = ".".join(torch.__version__.split(".")[:2])

        # torchvision version mapping (from PyTorch release notes)
        # PyTorch 2.9 -> torchvision 0.24.x
        # PyTorch 2.6 -> torchvision 0.21.x
        expected_vision_prefix = {
            "2.9": "0.24",
            "2.8": "0.23",
            "2.7": "0.22",
            "2.6": "0.21",
        }.get(torch_version, None)

        if expected_vision_prefix:
            vision_prefix = ".".join(torchvision.__version__.split(".")[:2])
            assert vision_prefix == expected_vision_prefix, (
                f"torchvision {torchvision.__version__} incompatible with PyTorch {torch.__version__}. "
                f"Expected torchvision {expected_vision_prefix}.x. "
                f"Run: pip install --upgrade torchvision"
            )

    def test_torchaudio_version_match(self):
        """torchaudio must match PyTorch major.minor version."""
        import torch
        import torchaudio

        torch_version = ".".join(torch.__version__.split(".")[:2])
        audio_version = ".".join(torchaudio.__version__.split(".")[:2])

        assert torch_version == audio_version, (
            f"torchaudio {torchaudio.__version__} incompatible with PyTorch {torch.__version__}. "
            f"Expected torchaudio {torch_version}.x. "
            f"Run: pip install --upgrade torchaudio"
        )

    def test_mps_torch_angle_native(self):
        """torch.angle() must work natively on MPS (no fallback needed)."""
        import torch

        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available")

        # This operation previously required PYTORCH_ENABLE_MPS_FALLBACK=1
        # PyTorch 2.9.1+ supports it natively
        c = torch.randn(5, dtype=torch.cfloat, device='mps')
        a = torch.angle(c)

        assert a.device.type == 'mps', "torch.angle result should be on MPS"
        assert a.shape == (5,), f"Expected shape (5,), got {a.shape}"

    def test_cpp_binary_synthesizes_audio(self):
        """C++ TTS binary must synthesize audio without Python.

        This verifies:
        1. libtorch loads TorchScript model correctly
        2. No ABI mismatches cause crashes
        3. Audio output is generated
        """
        import os
        import tempfile

        binary = os.path.join(
            os.path.dirname(__file__),
            "../../stream-tts-cpp/build/stream-tts-cpp"
        )
        binary = os.path.normpath(binary)

        if not os.path.exists(binary):
            pytest.skip("stream-tts-cpp binary not built")

        # Generate audio to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = f.name

        try:
            result = subprocess.run(
                [binary, "--speak", "Test.", "--lang", "en", "--save-audio", output_path],
                capture_output=True,
                text=True,
                timeout=60
            )

            # Must succeed
            assert result.returncode == 0, f"TTS failed: {result.stderr}"

            # Must produce audio file
            assert os.path.exists(output_path), "No audio file generated"
            file_size = os.path.getsize(output_path)
            assert file_size > 1000, f"Audio file too small: {file_size} bytes"

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_libtorch_version_match(self):
        """libtorch version should match Python PyTorch for model compatibility."""
        import os
        import torch

        libtorch_version_file = os.path.join(
            os.path.dirname(__file__),
            "../../stream-tts-cpp/external/libtorch-mps/build-version"
        )
        libtorch_version_file = os.path.normpath(libtorch_version_file)

        if not os.path.exists(libtorch_version_file):
            pytest.skip("libtorch not installed")

        with open(libtorch_version_file) as f:
            libtorch_version = f.read().strip()

        torch_version = torch.__version__.split("+")[0]  # Remove +cu118 etc

        # Warn if versions don't match (not necessarily fatal)
        if libtorch_version != torch_version:
            pytest.xfail(
                f"libtorch {libtorch_version} != PyTorch {torch_version}. "
                f"This may cause issues. Run: ./stream-tts-cpp/scripts/setup_libtorch.sh"
            )


class TestModelExport:
    """Test that model export produces valid files."""

    def test_kokoro_mps_model_exists(self):
        """kokoro_mps.pt must exist and be reasonable size."""
        import os

        model_path = os.path.join(
            os.path.dirname(__file__),
            "../../models/kokoro/kokoro_mps.pt"
        )
        model_path = os.path.normpath(model_path)

        assert os.path.exists(model_path), (
            f"MPS model not found: {model_path}. "
            f"Run: python scripts/export_kokoro_torchscript.py --device mps"
        )

        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        assert 300 < size_mb < 400, (
            f"Model size {size_mb:.1f}MB is unexpected (expected ~328MB). "
            f"Model may be corrupted."
        )

    def test_kokoro_mps_model_loads(self):
        """kokoro_mps.pt must load successfully in libtorch."""
        import os
        import torch

        model_path = os.path.join(
            os.path.dirname(__file__),
            "../../models/kokoro/kokoro_mps.pt"
        )
        model_path = os.path.normpath(model_path)

        if not os.path.exists(model_path):
            pytest.skip("MPS model not found")

        # Load the traced model
        try:
            model = torch.jit.load(model_path)
        except Exception as e:
            pytest.fail(f"Failed to load model: {e}")

        # Verify it's a valid traced module
        assert isinstance(model, torch.jit.ScriptModule), "Not a valid TorchScript module"
