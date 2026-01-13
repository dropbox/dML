#!/usr/bin/env python3
"""
Verify Gap 9 (roadmap) deadlock/lock-wait diagnostics APIs exist and behave sanely.

This test only validates the exported symbols and configuration wiring; it does not
attempt to force a deadlock.
"""

import ctypes
import os
import sys


def load_agx_fix():
    dylib_paths = [
        "/Users/ayates/metal_mps_parallel/agx_fix/build/libagx_fix_v2_9.dylib",
        "./agx_fix/build/libagx_fix_v2_9.dylib",
    ]

    for path in dylib_paths:
        try:
            return ctypes.CDLL(path)
        except OSError:
            continue
    return None


def main():
    agx = load_agx_fix()
    if not agx:
        print("ERROR: Could not load libagx_fix_v2_9.dylib")
        return 1

    # Core status check
    agx.agx_fix_v2_9_is_enabled.restype = ctypes.c_bool
    if not agx.agx_fix_v2_9_is_enabled():
        print("FAIL: agx_fix_v2_9_is_enabled() returned false")
        return 1

    # Gap 9 diagnostics
    agx.agx_fix_v2_9_deadlock_detection_enabled.restype = ctypes.c_bool
    agx.agx_fix_v2_9_get_mutex_long_wait_warnings.restype = ctypes.c_uint64
    agx.agx_fix_v2_9_get_mutex_lock_timeouts.restype = ctypes.c_uint64
    agx.agx_fix_v2_9_get_mutex_max_wait_ms.restype = ctypes.c_uint64

    enabled = agx.agx_fix_v2_9_deadlock_detection_enabled()
    long_waits = agx.agx_fix_v2_9_get_mutex_long_wait_warnings()
    timeouts = agx.agx_fix_v2_9_get_mutex_lock_timeouts()
    max_wait_ms = agx.agx_fix_v2_9_get_mutex_max_wait_ms()

    print(f"agx_fix_v2_9_deadlock_detection_enabled() = {enabled}")
    print(f"agx_fix_v2_9_get_mutex_long_wait_warnings() = {long_waits}")
    print(f"agx_fix_v2_9_get_mutex_lock_timeouts() = {timeouts}")
    print(f"agx_fix_v2_9_get_mutex_max_wait_ms() = {max_wait_ms}")

    expected_enabled = os.getenv("AGX_FIX_DEADLOCK_DETECT") is not None
    if expected_enabled and not enabled:
        print("FAIL: Deadlock detection expected enabled but API reports disabled")
        return 1

    # Sanity: max wait should be >= 0 (uint64); no further constraints.
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

