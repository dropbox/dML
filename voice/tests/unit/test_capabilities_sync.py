"""
Capabilities Registry Sync Tests

Ensures the supported-language grid in README.md matches the C++ capabilities
registry exposed via --capabilities-json.
"""

import json
import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).parent.parent.parent


def readme_language_codes():
    """Extract language codes from README Supported Languages table."""
    readme_path = PROJECT_ROOT / "README.md"
    text = readme_path.read_text().splitlines()

    # Find the supported languages table
    start_idx = None
    for i, line in enumerate(text):
        if line.strip().startswith("## Supported Languages"):
            start_idx = i
            break
    assert start_idx is not None, "README Supported Languages section not found"

    codes = []
    in_table = False
    for line in text[start_idx:]:
        if line.startswith("| Language |"):
            in_table = True
            continue
        if in_table:
            if not line.startswith("|"):
                break
            if line.startswith("|---") or line.startswith("|----------"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 3:
                continue
            code = parts[2]
            if code:
                codes.append(code)
    return codes


@pytest.mark.unit
def test_readme_matches_capabilities(cpp_binary):
    """README codes should match canonical codes from capabilities registry."""
    result = subprocess.run(
        [str(cpp_binary), "--capabilities-json"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, f"Failed to get capabilities: {result.stderr}"
    caps = json.loads(result.stdout)

    registry_codes = [l["code"] for l in caps["languages"]]
    readme_codes = readme_language_codes()

    assert set(readme_codes) == set(registry_codes), \
        f"README codes {readme_codes} != registry codes {registry_codes}"

