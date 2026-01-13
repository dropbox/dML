#!/usr/bin/env python3
"""
AGX Fix Check Module

NOTE (2025-12-23): PyTorch 2.9.1 has improved MPS threading support, but
multi-threaded MPS workloads can still intermittently SIGSEGV due to an
underlying AGX driver race.

For reliable multi-threaded verification runs in this repo, prefer:
- Crash-check wrapper (recommended): `./scripts/run_test_with_crash_check.sh python3 tests/<test>.py`
- Crash-check wrapper override: `AGX_FIX_DYLIB=./agx_fix/build/libagx_fix_v2_9.dylib ./scripts/run_test_with_crash_check.sh python3 tests/<test>.py`
- Direct injection (recommended version): `DYLD_INSERT_LIBRARIES=./agx_fix/build/libagx_fix_v2_9.dylib MPS_FORCE_GRAPH_PATH=1 python3 tests/<test>.py`
  (Note: heavy workloads can still crash at 3+ concurrent MPS ops; see Semaphore(2) throttling in agx_fix/README.md)

This module provides a small guard for tests that use MPS threading so they can:
- Warn when running without the AGX fix dylib (may crash/flaky)
- Skip gracefully on older PyTorch versions when dylib injection is required

Usage:
    from agx_fix_check import require_agx_fix_for_threading
    require_agx_fix_for_threading()  # Warns on 2.9.1+ if no dylib is loaded

Or run tests through the wrapper:
    ./scripts/run_test_with_crash_check.sh python3 tests/your_test.py
"""

import os
import sys


def get_pytorch_version():
    """Get PyTorch version tuple (major, minor, patch)."""
    try:
        import torch
        version = torch.__version__.split('+')[0].split('a')[0]  # Strip nightly suffix
        parts = version.split('.')
        major = int(parts[0]) if parts else 0
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch = int(parts[2]) if len(parts) > 2 else 0
        return (major, minor, patch)
    except Exception:
        return (0, 0, 0)


def is_pytorch_291_or_newer():
    """Check if PyTorch version is 2.9.1 or newer (has native MPS threading)."""
    version = get_pytorch_version()
    return version >= (2, 9, 1)


def is_agx_fix_loaded() -> bool:
    """Check if libagx_fix.dylib is loaded via DYLD_INSERT_LIBRARIES."""
    dyld_libs = os.environ.get('DYLD_INSERT_LIBRARIES', '')
    return 'libagx_fix' in dyld_libs


def require_agx_fix_for_threading(allow_skip: bool = True) -> None:
    """
    Check MPS threading safety.

    For PyTorch 2.9.1+:
        - Native MPS threading is supported, no dylib needed
        - This function is a no-op

    For older PyTorch:
        - AGX fix dylib is required
        - Will skip or exit if not loaded

    Args:
        allow_skip: If True, skip gracefully when AGX fix not loaded on older PyTorch.
                   If False, exit with error.
    """
    # PyTorch 2.9.1+ improved MPS threading support, but can still crash
    # due to the underlying AGX driver race. Do not hard-require dylib
    # injection here, but warn so direct execution isn't silently flaky.
    if is_pytorch_291_or_newer():
        if is_agx_fix_loaded():
            return
        if os.environ.get("MPS_SUPPRESS_AGX_FIX_WARNING", "0") != "1":
            print("=" * 60)
            print("WARNING: Running multi-threaded MPS test without AGX fix dylib")
            print("=" * 60)
            print()
            print("PyTorch 2.9.1+ is improved, but complex multi-threaded MPS")
            print("workloads can still intermittently crash (AGX driver race).")
            print()
            print("For best stability, run via the crash-check wrapper:")
            print("    ./scripts/run_test_with_crash_check.sh python3 tests/<this_test>.py")
            print()
            print("To force a specific dylib (e.g., v2.9):")
            print("    AGX_FIX_DYLIB=./agx_fix/build/libagx_fix_v2_9.dylib ./scripts/run_test_with_crash_check.sh python3 tests/<this_test>.py")
            print()
            print("Or inject directly:")
            print("    DYLD_INSERT_LIBRARIES=./agx_fix/build/libagx_fix_v2_9.dylib MPS_FORCE_GRAPH_PATH=1 python3 tests/<this_test>.py")
            print("=" * 60)
        return

    # Older PyTorch requires AGX fix
    if is_agx_fix_loaded():
        return  # Good to go

    if allow_skip:
        print("=" * 60)
        print("SKIPPING: Multi-threaded MPS test requires AGX fix")
        print("=" * 60)
        print()
        print("The AGX driver has a race condition that causes crashes")
        print("during multi-threaded MPS use. To run this test safely:")
        print()
        print("Option 1: Use the crash-check wrapper:")
        print("    ./scripts/run_test_with_crash_check.sh python3 tests/<this_test>.py")
        print()
        print("Option 2: Upgrade to PyTorch 2.9.1+ (has native MPS threading)")
        print()
        print("See AGX_RESEARCH_ROADMAP.md for details on the driver bug.")
        print("=" * 60)
        sys.exit(0)  # Skip gracefully, exit code 0
    else:
        print("ERROR: Multi-threaded MPS test requires AGX fix", file=sys.stderr)
        print("Run with: ./scripts/run_test_with_crash_check.sh python3 tests/<test>.py", file=sys.stderr)
        sys.exit(1)


def print_agx_fix_status() -> None:
    """Print AGX fix status for diagnostic purposes."""
    if is_agx_fix_loaded():
        dyld_libs = os.environ.get('DYLD_INSERT_LIBRARIES', '')
        print(f"AGX fix: LOADED ({dyld_libs})")
    else:
        print("AGX fix: NOT LOADED")


if __name__ == "__main__":
    print_agx_fix_status()
    if is_agx_fix_loaded():
        print("AGX fix is loaded. Multi-threaded MPS tests are safe to run.")
    else:
        print("AGX fix NOT loaded. Multi-threaded MPS tests may crash.")
        print("Use: ./scripts/run_test_with_crash_check.sh python3 tests/<test>.py")
