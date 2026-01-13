"""
Optional Backend Preflight Tests

Verifies that --check surfaces optional backend availability accurately.
In particular, connectability (not just file existence) is required for
socket-based backends.
"""

import os
import socket
import subprocess
from pathlib import Path
from uuid import uuid4


PROJECT_ROOT = Path(__file__).parent.parent.parent

import pytest


def _env_with_socket(var_name: str, path: str):
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env[var_name] = path
    return env


@pytest.mark.unit
@pytest.mark.requires_binary
@pytest.mark.requires_models
def test_orpheus_socket_preflight_detects_stale_socket(tmp_path, cpp_binary, english_config):
    """
    When ORPHEUS_TTS_SOCKET points at a live Unix socket, --check should not warn.
    When the socket file exists but no server is listening, --check should warn.
    """
    # Use a short path inside the workspace (sandbox-safe) to stay under AF_UNIX sun_path limit.
    sock_dir = PROJECT_ROOT / "test_output"
    sock_dir.mkdir(exist_ok=True)
    sock_path = sock_dir / f"orpheus_test_{os.getpid()}_{uuid4().hex}.sock"
    if sock_path.exists():
        sock_path.unlink()

    env = _env_with_socket("ORPHEUS_TTS_SOCKET", str(sock_path))

    # Attempt to start a minimal Unix socket server. In sandboxed environments,
    # AF_UNIX bind may be disallowed; fall back to validating the negative case.
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(sock_path))
        server.listen(1)

        # Live socket: no warning expected.
        result_live = subprocess.run(
            [str(cpp_binary), "--check", str(english_config)],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(cpp_binary.parent.parent),
            env=env,
        )
        assert result_live.returncode == 0, result_live.stderr
        combined_live = (result_live.stdout or "") + (result_live.stderr or "")
        assert "Optional backend unavailable: Orpheus-TTS socket not reachable" not in combined_live

        # Close server but leave stale socket path in place.
        server.close()

        # Stale socket file: warning expected.
        result_stale = subprocess.run(
            [str(cpp_binary), "--check", str(english_config)],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(cpp_binary.parent.parent),
            env=env,
        )
        assert result_stale.returncode == 0, result_stale.stderr
        combined_stale = (result_stale.stdout or "") + (result_stale.stderr or "")
        assert "Optional backend unavailable: Orpheus-TTS socket not reachable" in combined_stale

    except PermissionError:
        server.close()
        # Negative-case fallback: create a regular file at the path.
        sock_path.write_text("not a socket")
        result_bad = subprocess.run(
            [str(cpp_binary), "--check", str(english_config)],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(cpp_binary.parent.parent),
            env=env,
        )
        assert result_bad.returncode == 0, result_bad.stderr
        combined_bad = (result_bad.stdout or "") + (result_bad.stderr or "")
        assert "Optional backend unavailable: Orpheus-TTS socket not reachable" in combined_bad

    finally:
        if sock_path.exists():
            sock_path.unlink()
