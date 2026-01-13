"""
Dialect Instruction Consistency Tests

Ensures all CosyVoice2 dialect voices use consistent simple instruction format.
This prevents regression where one dialect uses a complex instruction while others
use simple format.

The expected format for all dialects is: "用X话说" or "用X语说"
Examples:
- sichuan: "用四川话说"
- cantonese: "用粤语说"
- shanghainese: "用上海话说"

Complex instructions like "用四川话说，像一个四川婆婆在讲故事" are NOT allowed
as they produce inconsistent quality compared to simple instructions.
"""

import os
import re
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
STREAM_TTS_CPP = PROJECT_ROOT / "stream-tts-cpp"
TTS_BINARY = STREAM_TTS_CPP / "build" / "stream-tts-cpp"

# All CosyVoice2 dialect voices that should use simple instructions
DIALECT_VOICES = ["sichuan", "cantonese", "shanghainese"]

# Pattern for valid simple dialect instruction: "用X话说" or "用X语说"
# Must NOT contain comma or additional phrases
SIMPLE_INSTRUCTION_PATTERN = re.compile(r"^用.{1,4}[话语]说$")


def get_tts_env():
    """Get environment variables for TTS subprocess."""
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["OMP_NUM_THREADS"] = "1"
    return env


@pytest.mark.unit
@pytest.mark.skipif(not TTS_BINARY.exists(), reason="Binary not built")
class TestDialectInstructionConsistency:
    """Tests to enforce consistent dialect instruction format."""

    @pytest.mark.parametrize("voice", DIALECT_VOICES)
    def test_dialect_uses_simple_instruction(self, voice):
        """Verify each dialect voice uses simple instruction format.

        This test prevents regression where a dialect voice is configured
        with a complex instruction (e.g., "像一个四川婆婆在讲故事") that
        produces inconsistent quality compared to other dialects.

        All dialect voices should use the format: "用X话说" or "用X语说"
        """
        # Run the binary and capture the instruction it uses
        result = subprocess.run(
            [str(TTS_BINARY), "--voice-name", voice, "--speak", "测试", "--lang", "zh"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(STREAM_TTS_CPP),
            env=get_tts_env(),
        )

        # Extract instruction from log output
        # Format: VoiceEngine initialized (backend=cosyvoice2, voice=X, instruction='Y')
        output = result.stderr + result.stdout

        instruction_match = re.search(r"instruction='([^']+)'", output)
        assert instruction_match, f"Could not find instruction in output for voice '{voice}'"

        instruction = instruction_match.group(1)

        # Verify instruction is simple format (no comma, no additional phrases)
        assert "," not in instruction, (
            f"Voice '{voice}' uses complex instruction with comma: '{instruction}'. "
            f"All dialect voices should use simple format like '用X话说'"
        )

        assert SIMPLE_INSTRUCTION_PATTERN.match(instruction), (
            f"Voice '{voice}' instruction '{instruction}' doesn't match simple format. "
            f"Expected pattern: '用X话说' or '用X语说' (e.g., '用四川话说', '用粤语说')"
        )

    def test_all_dialects_instruction_length_similar(self):
        """Verify all dialect instructions have similar length (consistency check).

        If one dialect has a much longer instruction than others, it's likely
        using a complex format that should be simplified.
        """
        instructions = {}

        for voice in DIALECT_VOICES:
            result = subprocess.run(
                [str(TTS_BINARY), "--voice-name", voice, "--speak", "测试", "--lang", "zh"],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(STREAM_TTS_CPP),
                env=get_tts_env(),
            )

            output = result.stderr + result.stdout
            match = re.search(r"instruction='([^']+)'", output)
            if match:
                instructions[voice] = match.group(1)

        # All instructions should be within 2x length of shortest
        if instructions:
            lengths = [len(v) for v in instructions.values()]
            min_len = min(lengths)
            max_len = max(lengths)

            assert max_len <= min_len * 2, (
                f"Instruction lengths vary too much: {instructions}. "
                f"This suggests inconsistent instruction format between dialects."
            )
