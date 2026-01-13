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
Tests for CosyVoice2 Converter

Phase 5 tests - Flow-matching TTS model conversion.
"""

import tempfile
from pathlib import Path


class TestCosyVoice2Converter:
    """Test CosyVoice2Converter class."""

    def test_init(self):
        """Test converter initialization."""
        from tools.pytorch_to_mlx.converters import CosyVoice2Converter

        converter = CosyVoice2Converter()
        assert converter is not None

    def test_list_supported_models(self):
        """Test listing supported models."""
        from tools.pytorch_to_mlx.converters import CosyVoice2Converter

        models = CosyVoice2Converter.list_supported_models()
        assert len(models) >= 2
        assert "FunAudioLLM/CosyVoice2-0.5B" in models
        assert "FunAudioLLM/CosyVoice-300M" in models

    def test_model_files_constant(self):
        """Test MODEL_FILES constant."""
        from tools.pytorch_to_mlx.converters import CosyVoice2Converter

        assert "llm" in CosyVoice2Converter.MODEL_FILES
        assert "flow" in CosyVoice2Converter.MODEL_FILES
        assert "vocoder" in CosyVoice2Converter.MODEL_FILES

        assert CosyVoice2Converter.MODEL_FILES["llm"] == "llm.pt"
        assert CosyVoice2Converter.MODEL_FILES["flow"] == "flow.pt"
        assert CosyVoice2Converter.MODEL_FILES["vocoder"] == "hift.pt"

    def test_locate_model_files_empty_dir(self):
        """Test locating files in empty directory."""
        from tools.pytorch_to_mlx.converters import CosyVoice2Converter

        converter = CosyVoice2Converter()

        with tempfile.TemporaryDirectory() as tmpdir:
            files = converter.locate_model_files(Path(tmpdir))
            assert files == {}

    def test_convert_nonexistent_path(self):
        """Test convert with non-existent path."""
        from tools.pytorch_to_mlx.converters import CosyVoice2Converter

        converter = CosyVoice2Converter()

        with tempfile.TemporaryDirectory() as tmpdir:
            result = converter.convert(
                "/nonexistent/path/to/model",
                tmpdir,
            )
            assert not result.success
            assert "not found" in result.error.lower()

    def test_validate_not_implemented(self):
        """Test that validate returns not implemented status."""
        from tools.pytorch_to_mlx.converters import CosyVoice2Converter

        converter = CosyVoice2Converter()

        with tempfile.TemporaryDirectory() as tmpdir:
            result = converter.validate(tmpdir)
            assert not result.passed
            assert "not yet implemented" in result.error.lower()

    def test_benchmark_not_implemented(self):
        """Test that benchmark returns placeholder values."""
        from tools.pytorch_to_mlx.converters import CosyVoice2Converter

        converter = CosyVoice2Converter()

        with tempfile.TemporaryDirectory() as tmpdir:
            result = converter.benchmark(tmpdir)
            assert result.mlx_audio_per_second == 0.0
            assert result.speedup == 0.0


class TestCLICommands:
    """Test CLI commands for CosyVoice2."""

    def test_cosyvoice2_help_command(self):
        """Test cosyvoice2 help command."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "tools.pytorch_to_mlx", "cosyvoice2", "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "cosyvoice2" in result.stdout.lower() or "tts" in result.stdout.lower()

    def test_cosyvoice2_list_command(self):
        """Test cosyvoice2 list command."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "tools.pytorch_to_mlx", "cosyvoice2", "list"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "FunAudioLLM/CosyVoice2-0.5B" in result.stdout
