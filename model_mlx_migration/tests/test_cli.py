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

"""Tests for the PyTorch to MLX CLI module."""

import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add tools to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from pytorch_to_mlx.cli import (
    cmd_analyze,
    cmd_cosyvoice2_list,
    cmd_kokoro_list,
    cmd_llama_list,
    cmd_nllb_list,
    cmd_whisper_list,
    main,
    run_benchmark,
    validate_conversion,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_args():
    """Create a mock args namespace."""
    return argparse.Namespace()


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test outputs."""
    return tmp_path


# =============================================================================
# Test main() Entry Point
# =============================================================================


class TestMain:
    """Test main() entry point and command dispatch."""

    def test_no_command_shows_help(self):
        """Test that running with no command shows help."""
        with patch.object(sys, "argv", ["pytorch_to_mlx"]):
            result = main()
            assert result == 1

    def test_invalid_command_shows_help(self):
        """Test that invalid command returns error."""
        with patch.object(sys, "argv", ["pytorch_to_mlx", "invalid_command"]):
            # Invalid command returns 1
            # argparse will raise SystemExit on unrecognized command
            with pytest.raises(SystemExit):
                main()

    def test_analyze_command_parsed(self):
        """Test analyze command argument parsing."""
        with patch.object(
            sys, "argv", ["pytorch_to_mlx", "analyze", "--input", "model.pt"],
        ):
            with patch("pytorch_to_mlx.cli.cmd_analyze") as mock_cmd:
                mock_cmd.return_value = 0
                result = main()
                assert result == 0
                mock_cmd.assert_called_once()

    def test_convert_command_parsed(self):
        """Test convert command argument parsing."""
        with patch.object(
            sys,
            "argv",
            ["pytorch_to_mlx", "convert", "--input", "model.pt", "--output", "out/"],
        ):
            with patch("pytorch_to_mlx.cli.cmd_convert") as mock_cmd:
                mock_cmd.return_value = 0
                result = main()
                assert result == 0
                mock_cmd.assert_called_once()

    def test_validate_command_parsed(self):
        """Test validate command argument parsing."""
        with patch.object(
            sys,
            "argv",
            [
                "pytorch_to_mlx",
                "validate",
                "--pytorch",
                "model.pt",
                "--mlx",
                "weights.safetensors",
            ],
        ):
            with patch("pytorch_to_mlx.cli.cmd_validate") as mock_cmd:
                mock_cmd.return_value = 0
                result = main()
                assert result == 0
                mock_cmd.assert_called_once()

    def test_benchmark_command_parsed(self):
        """Test benchmark command argument parsing."""
        with patch.object(
            sys, "argv", ["pytorch_to_mlx", "benchmark", "--mlx", "model_mlx/"],
        ):
            with patch("pytorch_to_mlx.cli.cmd_benchmark") as mock_cmd:
                mock_cmd.return_value = 0
                result = main()
                assert result == 0
                mock_cmd.assert_called_once()


class TestLlamaSubcommand:
    """Test llama subcommand group."""

    def test_llama_no_subcommand_shows_help(self):
        """Test llama with no subcommand shows help."""
        with patch.object(sys, "argv", ["pytorch_to_mlx", "llama"]):
            result = main()
            assert result == 1

    def test_llama_convert_parsed(self):
        """Test llama convert command parsing."""
        with patch.object(
            sys,
            "argv",
            [
                "pytorch_to_mlx",
                "llama",
                "convert",
                "--hf-path",
                "meta-llama/test",
                "--output",
                "out/",
            ],
        ):
            with patch("pytorch_to_mlx.cli.cmd_llama_convert") as mock_cmd:
                mock_cmd.return_value = 0
                result = main()
                assert result == 0
                mock_cmd.assert_called_once()

    def test_llama_validate_parsed(self):
        """Test llama validate command parsing."""
        with patch.object(
            sys,
            "argv",
            ["pytorch_to_mlx", "llama", "validate", "--mlx-path", "model_mlx/"],
        ):
            with patch("pytorch_to_mlx.cli.cmd_llama_validate") as mock_cmd:
                mock_cmd.return_value = 0
                result = main()
                assert result == 0
                mock_cmd.assert_called_once()

    def test_llama_benchmark_parsed(self):
        """Test llama benchmark command parsing."""
        with patch.object(
            sys,
            "argv",
            ["pytorch_to_mlx", "llama", "benchmark", "--mlx-path", "model_mlx/"],
        ):
            with patch("pytorch_to_mlx.cli.cmd_llama_benchmark") as mock_cmd:
                mock_cmd.return_value = 0
                result = main()
                assert result == 0
                mock_cmd.assert_called_once()

    def test_llama_list_parsed(self):
        """Test llama list command parsing."""
        with patch.object(sys, "argv", ["pytorch_to_mlx", "llama", "list"]):
            with patch("pytorch_to_mlx.cli.cmd_llama_list") as mock_cmd:
                mock_cmd.return_value = 0
                result = main()
                assert result == 0
                mock_cmd.assert_called_once()


class TestNllbSubcommand:
    """Test nllb subcommand group."""

    def test_nllb_no_subcommand_shows_help(self):
        """Test nllb with no subcommand shows help."""
        with patch.object(sys, "argv", ["pytorch_to_mlx", "nllb"]):
            result = main()
            assert result == 1

    def test_nllb_convert_parsed(self):
        """Test nllb convert command parsing."""
        with patch.object(
            sys,
            "argv",
            [
                "pytorch_to_mlx",
                "nllb",
                "convert",
                "--hf-path",
                "facebook/nllb-200-distilled-600M",
                "--output",
                "out/",
            ],
        ):
            with patch("pytorch_to_mlx.cli.cmd_nllb_convert") as mock_cmd:
                mock_cmd.return_value = 0
                result = main()
                assert result == 0
                mock_cmd.assert_called_once()

    def test_nllb_validate_parsed(self):
        """Test nllb validate command parsing."""
        with patch.object(
            sys,
            "argv",
            ["pytorch_to_mlx", "nllb", "validate", "--mlx-path", "model_mlx/"],
        ):
            with patch("pytorch_to_mlx.cli.cmd_nllb_validate") as mock_cmd:
                mock_cmd.return_value = 0
                result = main()
                assert result == 0
                mock_cmd.assert_called_once()

    def test_nllb_benchmark_parsed(self):
        """Test nllb benchmark command parsing."""
        with patch.object(
            sys,
            "argv",
            ["pytorch_to_mlx", "nllb", "benchmark", "--mlx-path", "model_mlx/"],
        ):
            with patch("pytorch_to_mlx.cli.cmd_nllb_benchmark") as mock_cmd:
                mock_cmd.return_value = 0
                result = main()
                assert result == 0
                mock_cmd.assert_called_once()

    def test_nllb_translate_parsed(self):
        """Test nllb translate command parsing."""
        with patch.object(
            sys,
            "argv",
            [
                "pytorch_to_mlx",
                "nllb",
                "translate",
                "--mlx-path",
                "model_mlx/",
                "--text",
                "Hello world",
            ],
        ):
            with patch("pytorch_to_mlx.cli.cmd_nllb_translate") as mock_cmd:
                mock_cmd.return_value = 0
                result = main()
                assert result == 0
                mock_cmd.assert_called_once()

    def test_nllb_list_parsed(self):
        """Test nllb list command parsing."""
        with patch.object(sys, "argv", ["pytorch_to_mlx", "nllb", "list"]):
            with patch("pytorch_to_mlx.cli.cmd_nllb_list") as mock_cmd:
                mock_cmd.return_value = 0
                result = main()
                assert result == 0
                mock_cmd.assert_called_once()


class TestKokoroSubcommand:
    """Test kokoro subcommand group."""

    def test_kokoro_no_subcommand_shows_help(self):
        """Test kokoro with no subcommand shows help."""
        with patch.object(sys, "argv", ["pytorch_to_mlx", "kokoro"]):
            result = main()
            assert result == 1

    def test_kokoro_convert_parsed(self):
        """Test kokoro convert command parsing."""
        with patch.object(
            sys,
            "argv",
            ["pytorch_to_mlx", "kokoro", "convert", "--output", "out/"],
        ):
            with patch("pytorch_to_mlx.cli.cmd_kokoro_convert") as mock_cmd:
                mock_cmd.return_value = 0
                result = main()
                assert result == 0
                mock_cmd.assert_called_once()

    def test_kokoro_validate_parsed(self):
        """Test kokoro validate command parsing."""
        with patch.object(sys, "argv", ["pytorch_to_mlx", "kokoro", "validate"]):
            with patch("pytorch_to_mlx.cli.cmd_kokoro_validate") as mock_cmd:
                mock_cmd.return_value = 0
                result = main()
                assert result == 0
                mock_cmd.assert_called_once()

    def test_kokoro_list_parsed(self):
        """Test kokoro list command parsing."""
        with patch.object(sys, "argv", ["pytorch_to_mlx", "kokoro", "list"]):
            with patch("pytorch_to_mlx.cli.cmd_kokoro_list") as mock_cmd:
                mock_cmd.return_value = 0
                result = main()
                assert result == 0
                mock_cmd.assert_called_once()

    def test_kokoro_synthesize_parsed(self):
        """Test kokoro synthesize command parsing."""
        with patch.object(
            sys,
            "argv",
            ["pytorch_to_mlx", "kokoro", "synthesize", "--text", "Hello world"],
        ):
            with patch("pytorch_to_mlx.cli.cmd_kokoro_synthesize") as mock_cmd:
                mock_cmd.return_value = 0
                result = main()
                assert result == 0
                mock_cmd.assert_called_once()


class TestCosyvoice2Subcommand:
    """Test cosyvoice2 subcommand group."""

    def test_cosyvoice2_no_subcommand_shows_help(self):
        """Test cosyvoice2 with no subcommand shows help."""
        with patch.object(sys, "argv", ["pytorch_to_mlx", "cosyvoice2"]):
            result = main()
            assert result == 1

    def test_cosyvoice2_convert_parsed(self):
        """Test cosyvoice2 convert command parsing."""
        with patch.object(
            sys,
            "argv",
            [
                "pytorch_to_mlx",
                "cosyvoice2",
                "convert",
                "--model-path",
                "model/",
                "--output",
                "out/",
            ],
        ):
            with patch("pytorch_to_mlx.cli.cmd_cosyvoice2_convert") as mock_cmd:
                mock_cmd.return_value = 0
                result = main()
                assert result == 0
                mock_cmd.assert_called_once()

    def test_cosyvoice2_inspect_parsed(self):
        """Test cosyvoice2 inspect command parsing."""
        with patch.object(
            sys,
            "argv",
            ["pytorch_to_mlx", "cosyvoice2", "inspect", "--model-path", "model/"],
        ):
            with patch("pytorch_to_mlx.cli.cmd_cosyvoice2_inspect") as mock_cmd:
                mock_cmd.return_value = 0
                result = main()
                assert result == 0
                mock_cmd.assert_called_once()

    def test_cosyvoice2_validate_parsed(self):
        """Test cosyvoice2 validate command parsing."""
        with patch.object(
            sys,
            "argv",
            ["pytorch_to_mlx", "cosyvoice2", "validate", "--mlx-path", "model_mlx/"],
        ):
            with patch("pytorch_to_mlx.cli.cmd_cosyvoice2_validate") as mock_cmd:
                mock_cmd.return_value = 0
                result = main()
                assert result == 0
                mock_cmd.assert_called_once()

    def test_cosyvoice2_synthesize_parsed(self):
        """Test cosyvoice2 synthesize command parsing."""
        with patch.object(
            sys,
            "argv",
            ["pytorch_to_mlx", "cosyvoice2", "synthesize", "--text", "Hello world"],
        ):
            with patch("pytorch_to_mlx.cli.cmd_cosyvoice2_synthesize") as mock_cmd:
                mock_cmd.return_value = 0
                result = main()
                assert result == 0
                mock_cmd.assert_called_once()

    def test_cosyvoice2_list_parsed(self):
        """Test cosyvoice2 list command parsing."""
        with patch.object(sys, "argv", ["pytorch_to_mlx", "cosyvoice2", "list"]):
            with patch("pytorch_to_mlx.cli.cmd_cosyvoice2_list") as mock_cmd:
                mock_cmd.return_value = 0
                result = main()
                assert result == 0
                mock_cmd.assert_called_once()


class TestWhisperSubcommand:
    """Test whisper subcommand group."""

    def test_whisper_no_subcommand_shows_help(self):
        """Test whisper with no subcommand shows help."""
        with patch.object(sys, "argv", ["pytorch_to_mlx", "whisper"]):
            result = main()
            assert result == 1

    def test_whisper_transcribe_parsed(self):
        """Test whisper transcribe command parsing."""
        with patch.object(
            sys,
            "argv",
            ["pytorch_to_mlx", "whisper", "transcribe", "--audio", "audio.wav"],
        ):
            with patch("pytorch_to_mlx.cli.cmd_whisper_transcribe") as mock_cmd:
                mock_cmd.return_value = 0
                result = main()
                assert result == 0
                mock_cmd.assert_called_once()

    def test_whisper_benchmark_parsed(self):
        """Test whisper benchmark command parsing."""
        with patch.object(
            sys,
            "argv",
            ["pytorch_to_mlx", "whisper", "benchmark", "--audio", "audio.wav"],
        ):
            with patch("pytorch_to_mlx.cli.cmd_whisper_benchmark") as mock_cmd:
                mock_cmd.return_value = 0
                result = main()
                assert result == 0
                mock_cmd.assert_called_once()

    def test_whisper_list_parsed(self):
        """Test whisper list command parsing."""
        with patch.object(sys, "argv", ["pytorch_to_mlx", "whisper", "list"]):
            with patch("pytorch_to_mlx.cli.cmd_whisper_list") as mock_cmd:
                mock_cmd.return_value = 0
                result = main()
                assert result == 0
                mock_cmd.assert_called_once()


class TestWakewordSubcommand:
    """Test wakeword subcommand group."""

    def test_wakeword_no_subcommand_shows_help(self):
        """Test wakeword with no subcommand shows help."""
        with patch.object(sys, "argv", ["pytorch_to_mlx", "wakeword"]):
            result = main()
            assert result == 1

    def test_wakeword_status_parsed(self):
        """Test wakeword status command parsing."""
        with patch.object(sys, "argv", ["pytorch_to_mlx", "wakeword", "status"]):
            with patch("pytorch_to_mlx.cli.cmd_wakeword_status") as mock_cmd:
                mock_cmd.return_value = 0
                result = main()
                assert result == 0
                mock_cmd.assert_called_once()

    def test_wakeword_analyze_parsed(self):
        """Test wakeword analyze command parsing."""
        with patch.object(sys, "argv", ["pytorch_to_mlx", "wakeword", "analyze"]):
            with patch("pytorch_to_mlx.cli.cmd_wakeword_analyze") as mock_cmd:
                mock_cmd.return_value = 0
                result = main()
                assert result == 0
                mock_cmd.assert_called_once()

    def test_wakeword_detect_parsed(self):
        """Test wakeword detect command parsing."""
        with patch.object(
            sys,
            "argv",
            ["pytorch_to_mlx", "wakeword", "detect", "--audio", "audio.wav"],
        ):
            with patch("pytorch_to_mlx.cli.cmd_wakeword_detect") as mock_cmd:
                mock_cmd.return_value = 0
                result = main()
                assert result == 0
                mock_cmd.assert_called_once()

    def test_wakeword_benchmark_parsed(self):
        """Test wakeword benchmark command parsing."""
        with patch.object(sys, "argv", ["pytorch_to_mlx", "wakeword", "benchmark"]):
            with patch("pytorch_to_mlx.cli.cmd_wakeword_benchmark") as mock_cmd:
                mock_cmd.return_value = 0
                result = main()
                assert result == 0
                mock_cmd.assert_called_once()

    def test_wakeword_convert_parsed(self):
        """Test wakeword convert command parsing."""
        with patch.object(sys, "argv", ["pytorch_to_mlx", "wakeword", "convert"]):
            with patch("pytorch_to_mlx.cli.cmd_wakeword_convert") as mock_cmd:
                mock_cmd.return_value = 0
                result = main()
                assert result == 0
                mock_cmd.assert_called_once()


# =============================================================================
# Test Command Argument Parsing Details
# =============================================================================


class TestConvertCommandArgs:
    """Test convert command argument parsing in detail."""

    def test_convert_dtype_choices(self):
        """Test convert accepts valid dtype choices."""
        with patch.object(
            sys,
            "argv",
            [
                "pytorch_to_mlx",
                "convert",
                "--input",
                "model.pt",
                "--output",
                "out/",
                "--dtype",
                "float16",
            ],
        ):
            with patch("pytorch_to_mlx.cli.cmd_convert") as mock_cmd:
                mock_cmd.return_value = 0
                main()
                args = mock_cmd.call_args[0][0]
                assert args.dtype == "float16"

    def test_convert_validate_flag(self):
        """Test convert --validate flag."""
        with patch.object(
            sys,
            "argv",
            [
                "pytorch_to_mlx",
                "convert",
                "--input",
                "model.pt",
                "--output",
                "out/",
                "--validate",
            ],
        ):
            with patch("pytorch_to_mlx.cli.cmd_convert") as mock_cmd:
                mock_cmd.return_value = 0
                main()
                args = mock_cmd.call_args[0][0]
                assert args.validate is True

    def test_convert_benchmark_flag(self):
        """Test convert --benchmark flag."""
        with patch.object(
            sys,
            "argv",
            [
                "pytorch_to_mlx",
                "convert",
                "--input",
                "model.pt",
                "--output",
                "out/",
                "--benchmark",
            ],
        ):
            with patch("pytorch_to_mlx.cli.cmd_convert") as mock_cmd:
                mock_cmd.return_value = 0
                main()
                args = mock_cmd.call_args[0][0]
                assert args.benchmark is True

    def test_convert_tolerance_arg(self):
        """Test convert --tolerance argument."""
        with patch.object(
            sys,
            "argv",
            [
                "pytorch_to_mlx",
                "convert",
                "--input",
                "model.pt",
                "--output",
                "out/",
                "--tolerance",
                "1e-3",
            ],
        ):
            with patch("pytorch_to_mlx.cli.cmd_convert") as mock_cmd:
                mock_cmd.return_value = 0
                main()
                args = mock_cmd.call_args[0][0]
                assert args.tolerance == 1e-3

    def test_convert_verbose_flag(self):
        """Test convert -v verbose flag."""
        with patch.object(
            sys,
            "argv",
            [
                "pytorch_to_mlx",
                "convert",
                "--input",
                "model.pt",
                "--output",
                "out/",
                "-v",
            ],
        ):
            with patch("pytorch_to_mlx.cli.cmd_convert") as mock_cmd:
                mock_cmd.return_value = 0
                main()
                args = mock_cmd.call_args[0][0]
                assert args.verbose is True


class TestLlamaConvertCommandArgs:
    """Test llama convert command argument parsing in detail."""

    def test_llama_convert_quantize_flag(self):
        """Test llama convert --quantize flag."""
        with patch.object(
            sys,
            "argv",
            [
                "pytorch_to_mlx",
                "llama",
                "convert",
                "--hf-path",
                "meta-llama/test",
                "--output",
                "out/",
                "--quantize",
            ],
        ):
            with patch("pytorch_to_mlx.cli.cmd_llama_convert") as mock_cmd:
                mock_cmd.return_value = 0
                main()
                args = mock_cmd.call_args[0][0]
                assert args.quantize is True

    def test_llama_convert_q_bits_arg(self):
        """Test llama convert --q-bits argument."""
        with patch.object(
            sys,
            "argv",
            [
                "pytorch_to_mlx",
                "llama",
                "convert",
                "--hf-path",
                "meta-llama/test",
                "--output",
                "out/",
                "--q-bits",
                "8",
            ],
        ):
            with patch("pytorch_to_mlx.cli.cmd_llama_convert") as mock_cmd:
                mock_cmd.return_value = 0
                main()
                args = mock_cmd.call_args[0][0]
                assert args.q_bits == 8

    def test_llama_convert_dtype_choices(self):
        """Test llama convert --dtype accepts valid choices."""
        with patch.object(
            sys,
            "argv",
            [
                "pytorch_to_mlx",
                "llama",
                "convert",
                "--hf-path",
                "meta-llama/test",
                "--output",
                "out/",
                "--dtype",
                "bfloat16",
            ],
        ):
            with patch("pytorch_to_mlx.cli.cmd_llama_convert") as mock_cmd:
                mock_cmd.return_value = 0
                main()
                args = mock_cmd.call_args[0][0]
                assert args.dtype == "bfloat16"


class TestWhisperTranscribeCommandArgs:
    """Test whisper transcribe command argument parsing in detail."""

    def test_whisper_transcribe_format_choices(self):
        """Test whisper transcribe --format accepts valid choices."""
        for fmt in ["text", "json", "srt", "vtt"]:
            with patch.object(
                sys,
                "argv",
                [
                    "pytorch_to_mlx",
                    "whisper",
                    "transcribe",
                    "--audio",
                    "audio.wav",
                    "--format",
                    fmt,
                ],
            ):
                with patch("pytorch_to_mlx.cli.cmd_whisper_transcribe") as mock_cmd:
                    mock_cmd.return_value = 0
                    main()
                    args = mock_cmd.call_args[0][0]
                    assert args.format == fmt

    def test_whisper_transcribe_word_timestamps_flag(self):
        """Test whisper transcribe --word-timestamps flag."""
        with patch.object(
            sys,
            "argv",
            [
                "pytorch_to_mlx",
                "whisper",
                "transcribe",
                "--audio",
                "audio.wav",
                "--word-timestamps",
            ],
        ):
            with patch("pytorch_to_mlx.cli.cmd_whisper_transcribe") as mock_cmd:
                mock_cmd.return_value = 0
                main()
                args = mock_cmd.call_args[0][0]
                assert args.word_timestamps is True

    def test_whisper_transcribe_temperature_arg(self):
        """Test whisper transcribe --temperature argument."""
        with patch.object(
            sys,
            "argv",
            [
                "pytorch_to_mlx",
                "whisper",
                "transcribe",
                "--audio",
                "audio.wav",
                "--temperature",
                "0.5",
            ],
        ):
            with patch("pytorch_to_mlx.cli.cmd_whisper_transcribe") as mock_cmd:
                mock_cmd.return_value = 0
                main()
                args = mock_cmd.call_args[0][0]
                assert args.temperature == 0.5


class TestCosyvoice2SynthesizeCommandArgs:
    """Test cosyvoice2 synthesize command argument parsing in detail."""

    def test_cosyvoice2_synthesize_max_tokens(self):
        """Test cosyvoice2 synthesize --max-tokens argument."""
        with patch.object(
            sys,
            "argv",
            [
                "pytorch_to_mlx",
                "cosyvoice2",
                "synthesize",
                "--text",
                "Hello",
                "--max-tokens",
                "500",
            ],
        ):
            with patch("pytorch_to_mlx.cli.cmd_cosyvoice2_synthesize") as mock_cmd:
                mock_cmd.return_value = 0
                main()
                args = mock_cmd.call_args[0][0]
                assert args.max_tokens == 500

    def test_cosyvoice2_synthesize_temperature(self):
        """Test cosyvoice2 synthesize --temperature argument."""
        with patch.object(
            sys,
            "argv",
            [
                "pytorch_to_mlx",
                "cosyvoice2",
                "synthesize",
                "--text",
                "Hello",
                "--temperature",
                "0.8",
            ],
        ):
            with patch("pytorch_to_mlx.cli.cmd_cosyvoice2_synthesize") as mock_cmd:
                mock_cmd.return_value = 0
                main()
                args = mock_cmd.call_args[0][0]
                assert args.temperature == 0.8

    def test_cosyvoice2_synthesize_speaker_seed(self):
        """Test cosyvoice2 synthesize --speaker-seed argument."""
        with patch.object(
            sys,
            "argv",
            [
                "pytorch_to_mlx",
                "cosyvoice2",
                "synthesize",
                "--text",
                "Hello",
                "--speaker-seed",
                "123",
            ],
        ):
            with patch("pytorch_to_mlx.cli.cmd_cosyvoice2_synthesize") as mock_cmd:
                mock_cmd.return_value = 0
                main()
                args = mock_cmd.call_args[0][0]
                assert args.speaker_seed == 123


# =============================================================================
# Test Individual Command Handlers
# =============================================================================


class TestCmdAnalyze:
    """Test cmd_analyze command handler."""

    def test_analyze_nonexistent_model(self, mock_args):
        """Test analyze with nonexistent model returns error."""
        mock_args.input = "/nonexistent/model.pt"
        mock_args.output = None

        result = cmd_analyze(mock_args)
        assert result == 1  # Error loading model


class TestCmdLlamaList:
    """Test cmd_llama_list command handler."""

    def test_llama_list_output(self, mock_args, capsys):
        """Test llama list command outputs models."""
        # Patch at the source module where the import actually happens
        with patch(
            "pytorch_to_mlx.converters.llama_converter.LLaMAConverter",
        ) as mock_cls:
            mock_cls.list_supported_models.return_value = [
                "meta-llama/Llama-3.2-1B",
                "meta-llama/Llama-3.2-3B",
            ]

            result = cmd_llama_list(mock_args)
            assert result == 0

            captured = capsys.readouterr()
            assert "LLaMA" in captured.out or "Supported" in captured.out


class TestCmdNllbList:
    """Test cmd_nllb_list command handler."""

    def test_nllb_list_output(self, mock_args, capsys):
        """Test nllb list command outputs models."""
        with patch(
            "pytorch_to_mlx.converters.nllb_converter.NLLBConverter",
        ) as mock_cls:
            mock_cls.list_supported_models.return_value = [
                "facebook/nllb-200-distilled-600M",
                "facebook/nllb-200-1.3B",
            ]

            result = cmd_nllb_list(mock_args)
            assert result == 0

            captured = capsys.readouterr()
            assert "NLLB" in captured.out or "Supported" in captured.out


class TestCmdKokoroList:
    """Test cmd_kokoro_list command handler."""

    def test_kokoro_list_output(self, mock_args, capsys):
        """Test kokoro list command outputs models."""
        with patch(
            "pytorch_to_mlx.converters.kokoro_converter.KokoroConverter",
        ) as mock_cls:
            mock_cls.list_supported_models.return_value = ["hexgrad/Kokoro-82M"]

            result = cmd_kokoro_list(mock_args)
            assert result == 0

            captured = capsys.readouterr()
            assert "Kokoro" in captured.out or "Supported" in captured.out


class TestCmdCosyvoice2List:
    """Test cmd_cosyvoice2_list command handler."""

    def test_cosyvoice2_list_output(self, mock_args, capsys):
        """Test cosyvoice2 list command outputs models."""
        with patch(
            "pytorch_to_mlx.converters.cosyvoice2_converter.CosyVoice2Converter",
        ) as mock_cls:
            mock_cls.list_supported_models.return_value = ["CosyVoice2-0.5B"]

            result = cmd_cosyvoice2_list(mock_args)
            assert result == 0

            captured = capsys.readouterr()
            assert "CosyVoice2" in captured.out or "Supported" in captured.out


class TestCmdWhisperList:
    """Test cmd_whisper_list command handler."""

    def test_whisper_list_output(self, mock_args, capsys):
        """Test whisper list command outputs models."""
        with patch(
            "pytorch_to_mlx.converters.whisper_converter.WhisperConverter",
        ) as mock_cls:
            mock_cls.list_models.return_value = [
                "mlx-community/whisper-large-v3-turbo",
                "mlx-community/whisper-large-v3",
            ]
            mock_cls.get_model_info.return_value = {
                "size": "1.55GB",
                "multilingual": True,
            }

            result = cmd_whisper_list(mock_args)
            assert result == 0

            captured = capsys.readouterr()
            assert "whisper" in captured.out.lower() or "Available" in captured.out


# =============================================================================
# Test Helper Functions
# =============================================================================


class TestValidateConversion:
    """Test validate_conversion helper function."""

    def test_validate_no_weights_file(self, temp_dir):
        """Test validation fails when no weights file exists."""
        result = validate_conversion("model.pt", temp_dir, 1e-5)
        assert result is False

    def test_validate_with_safetensors(self, temp_dir):
        """Test validation finds safetensors file."""
        # Create dummy weights file
        (temp_dir / "weights.safetensors").write_text("")

        mock_report = {"verified": True}

        # Patch at the source module level
        with patch(
            "pytorch_to_mlx.generator.weight_converter.WeightConverter",
        ) as mock_cls:
            mock_converter = MagicMock()
            mock_converter.verify_conversion.return_value = mock_report
            mock_cls.return_value = mock_converter

            result = validate_conversion("model.pt", temp_dir, 1e-5)
            assert result is True


class TestRunBenchmark:
    """Test run_benchmark helper function."""

    def test_run_benchmark_output(self, temp_dir, capsys):
        """Test run_benchmark prints message."""
        run_benchmark("model.pt", temp_dir)
        captured = capsys.readouterr()
        assert "Benchmark" in captured.out


# =============================================================================
# Test Default Argument Values
# =============================================================================


class TestDefaultArgValues:
    """Test default argument values are correctly set."""

    def test_convert_default_dtype(self):
        """Test convert command default dtype is float32."""
        with patch.object(
            sys,
            "argv",
            [
                "pytorch_to_mlx",
                "convert",
                "--input",
                "model.pt",
                "--output",
                "out/",
            ],
        ):
            with patch("pytorch_to_mlx.cli.cmd_convert") as mock_cmd:
                mock_cmd.return_value = 0
                main()
                args = mock_cmd.call_args[0][0]
                assert args.dtype == "float32"

    def test_convert_default_tolerance(self):
        """Test convert command default tolerance is 1e-5."""
        with patch.object(
            sys,
            "argv",
            [
                "pytorch_to_mlx",
                "convert",
                "--input",
                "model.pt",
                "--output",
                "out/",
            ],
        ):
            with patch("pytorch_to_mlx.cli.cmd_convert") as mock_cmd:
                mock_cmd.return_value = 0
                main()
                args = mock_cmd.call_args[0][0]
                assert args.tolerance == 1e-5

    def test_llama_convert_default_q_bits(self):
        """Test llama convert default q_bits is 4."""
        with patch.object(
            sys,
            "argv",
            [
                "pytorch_to_mlx",
                "llama",
                "convert",
                "--hf-path",
                "test",
                "--output",
                "out/",
            ],
        ):
            with patch("pytorch_to_mlx.cli.cmd_llama_convert") as mock_cmd:
                mock_cmd.return_value = 0
                main()
                args = mock_cmd.call_args[0][0]
                assert args.q_bits == 4

    def test_whisper_transcribe_default_temperature(self):
        """Test whisper transcribe default temperature is 0.0."""
        with patch.object(
            sys,
            "argv",
            [
                "pytorch_to_mlx",
                "whisper",
                "transcribe",
                "--audio",
                "test.wav",
            ],
        ):
            with patch("pytorch_to_mlx.cli.cmd_whisper_transcribe") as mock_cmd:
                mock_cmd.return_value = 0
                main()
                args = mock_cmd.call_args[0][0]
                assert args.temperature == 0.0

    def test_cosyvoice2_synthesize_default_speaker_seed(self):
        """Test cosyvoice2 synthesize default speaker_seed is 42."""
        with patch.object(
            sys,
            "argv",
            [
                "pytorch_to_mlx",
                "cosyvoice2",
                "synthesize",
                "--text",
                "Hello",
            ],
        ):
            with patch("pytorch_to_mlx.cli.cmd_cosyvoice2_synthesize") as mock_cmd:
                mock_cmd.return_value = 0
                main()
                args = mock_cmd.call_args[0][0]
                assert args.speaker_seed == 42


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
