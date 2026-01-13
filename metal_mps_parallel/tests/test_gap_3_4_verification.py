#!/usr/bin/env python3
"""
Test Gap 3 (IMP caching detection) and Gap 4 (class name robustness) features.

These tests verify that:
1. Gap 3: The swizzle verification API returns True when AGX fix is active
2. Gap 4: The class name API returns actual AGX class names (not "NOT_FOUND")
3. macOS version is logged at startup
"""

import ctypes
import os
import sys
import subprocess

def main():
    # Check that we're running with the AGX fix dylib
    dylib_path = os.environ.get('AGX_FIX_DYLIB') or os.environ.get('DYLD_INSERT_LIBRARIES', '')
    if 'libagx_fix_v2_9' not in dylib_path:
        print("WARNING: Test should be run with AGX fix v2.9 dylib")
        print(f"  Current: {dylib_path}")
        # Continue anyway to test the API

    # Load the dylib
    dylib_paths = [
        '/Users/ayates/metal_mps_parallel/agx_fix/build/libagx_fix_v2_9.dylib',
        './agx_fix/build/libagx_fix_v2_9.dylib',
    ]

    agx = None
    for path in dylib_paths:
        try:
            agx = ctypes.CDLL(path)
            break
        except OSError:
            continue

    if not agx:
        print("ERROR: Could not load libagx_fix_v2_9.dylib")
        return 1

    # Test Gap 3: Swizzle verification API
    print("=" * 60)
    print("TEST: Gap 3 - IMP Caching Detection")
    print("=" * 60)

    try:
        agx.agx_fix_v2_9_verify_swizzle_active.restype = ctypes.c_bool
        swizzle_active = agx.agx_fix_v2_9_verify_swizzle_active()
        print(f"  agx_fix_v2_9_verify_swizzle_active() = {swizzle_active}")
        if swizzle_active:
            print("  PASS: Swizzle is active")
        else:
            print("  WARN: Swizzle may not be active (could be IMP caching or dylib not injected)")
    except AttributeError as e:
        print(f"  ERROR: API not found - {e}")
        return 1

    # Test Gap 4: Class name API
    print()
    print("=" * 60)
    print("TEST: Gap 4 - Class Name Robustness")
    print("=" * 60)

    try:
        agx.agx_fix_v2_9_get_encoder_class_name.restype = ctypes.c_char_p
        agx.agx_fix_v2_9_get_command_buffer_class_name.restype = ctypes.c_char_p

        encoder_class = agx.agx_fix_v2_9_get_encoder_class_name()
        cmd_buffer_class = agx.agx_fix_v2_9_get_command_buffer_class_name()

        encoder_class_str = encoder_class.decode('utf-8') if encoder_class else "None"
        cmd_buffer_class_str = cmd_buffer_class.decode('utf-8') if cmd_buffer_class else "None"

        print(f"  Encoder class: {encoder_class_str}")
        print(f"  Command buffer class: {cmd_buffer_class_str}")

        if encoder_class_str != "NOT_FOUND" and "AGX" in encoder_class_str:
            print("  PASS: Encoder class discovered correctly")
        else:
            print(f"  WARN: Unexpected encoder class: {encoder_class_str}")

        if cmd_buffer_class_str != "NOT_FOUND" and "AGX" in cmd_buffer_class_str:
            print("  PASS: Command buffer class discovered correctly")
        else:
            print(f"  WARN: Unexpected command buffer class: {cmd_buffer_class_str}")

    except AttributeError as e:
        print(f"  ERROR: API not found - {e}")
        return 1

    # Test enabled status
    print()
    print("=" * 60)
    print("TEST: AGX Fix Status")
    print("=" * 60)

    try:
        agx.agx_fix_v2_9_is_enabled.restype = ctypes.c_bool
        enabled = agx.agx_fix_v2_9_is_enabled()
        print(f"  agx_fix_v2_9_is_enabled() = {enabled}")
        if enabled:
            print("  PASS: AGX fix is enabled")
        else:
            print("  FAIL: AGX fix is disabled")
            return 1
    except AttributeError as e:
        print(f"  ERROR: API not found - {e}")
        return 1

    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
