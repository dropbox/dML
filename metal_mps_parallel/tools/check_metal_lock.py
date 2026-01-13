#!/usr/bin/env python3
"""
Metal API Lock Checker for MPS Thread Safety

Scans Objective-C++ files for Metal API calls and verifies they are
within an MPSEncodingLock scope or dispatch_sync_with_rethrow block.

Usage:
    python3 tools/check_metal_lock.py [--verbose]

Exit codes:
    0 - All Metal calls are protected
    1 - Unprotected Metal calls found
    2 - Error (file not found, etc.)
"""

import os
import re
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Set

# Metal APIs that need lock protection
METAL_API_PATTERNS = [
    (r'\[\s*\w+\s+dispatchThread', 'dispatchThread*'),
    (r'\[\s*\w+\s+computeCommandEncoder\s*\]', 'computeCommandEncoder'),
    (r'\[\s*\w+\s+blitCommandEncoder\s*\]', 'blitCommandEncoder'),
    (r'\[\s*\w+\s+newLibraryWith', 'newLibraryWith*'),
    (r'\[\s*\w+\s+newFunctionWithName', 'newFunctionWithName'),
    (r'\[\s*\w+\s+newComputePipelineStateWith', 'newComputePipelineState'),
    (r'\[\s*\w+\s+encodeToCommandBuffer', 'encodeToCommandBuffer'),
    (r'\[\s*\w+\s+encodeSignalEvent', 'encodeSignalEvent'),
    (r'\[\s*\w+\s+encodeWaitForEvent', 'encodeWaitForEvent'),
]

# Patterns that indicate the call is protected
PROTECTION_PATTERNS = [
    r'MPSEncodingLock\s+\w+',           # Direct lock variable
    r'dispatch_sync_with_rethrow\s*\(', # Wrapper function (has lock inside)
]

# Known safe patterns (false positives to ignore)
SAFE_PATTERNS = [
    r'^\s*//',              # C++ comment lines
    r'^\s*\*',              # Block comment continuation
    r'^\s*#',               # Preprocessor
]

# Files to exclude (tests, etc.)
EXCLUDE_PATHS = [
    '/test/',               # Test files
    '/tests/',
]

def is_lock_in_scope(lines: List[str], lock_line: int, call_line: int) -> bool:
    """
    Check if a lock at lock_line is still in scope at call_line.
    Uses brace counting - if we exit a scope between lock and call, lock is not valid.
    Both line numbers are 0-indexed.
    """
    # Count braces from lock_line to call_line
    # If we ever go negative (exited a scope), the lock is out of scope
    brace_count = 0
    for i in range(lock_line, call_line):
        line = lines[i]
        brace_count += line.count('{') - line.count('}')
        # Went negative means we exited the scope where lock was declared
        if brace_count < 0:
            return False
    return True

def is_call_protected(lines: List[str], call_line: int) -> Tuple[bool, str]:
    """
    Check if a Metal API call at call_line is protected by MPSEncodingLock.
    call_line is 1-indexed (from enumerate).
    Returns (is_protected, reason).
    """
    # Convert to 0-indexed for array access
    call_idx = call_line - 1

    # Look back up to 150 lines for protection
    start_idx = max(0, call_idx - 150)

    # Check for MPSEncodingLock declarations
    lock_pattern = re.compile(r'MPSEncodingLock\s+\w+')
    for i in range(call_idx - 1, start_idx - 1, -1):
        if lock_pattern.search(lines[i]):
            # Verify lock is still in scope at call site
            if is_lock_in_scope(lines, i, call_idx):
                return True, f"Protected by MPSEncodingLock (line {i+1})"

    # Check for dispatch_sync_with_rethrow blocks
    dispatch_pattern = re.compile(r'dispatch_sync_with_rethrow\s*\(')
    for i in range(call_idx - 1, start_idx - 1, -1):
        if dispatch_pattern.search(lines[i]):
            # Check if we're inside this dispatch block
            brace_count = 0
            for j in range(i, call_idx + 1):
                brace_count += lines[j].count('{') - lines[j].count('}')
            # If brace_count > 0, we're still inside the block
            if brace_count > 0:
                return True, f"Inside dispatch_sync_with_rethrow (line {i+1})"

    return False, "No protection found"

def check_file(filepath: Path, verbose: bool = False) -> List[str]:
    """Check a single file for unprotected Metal API calls."""
    errors = []

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            lines = content.split('\n')
    except Exception as e:
        return [f"{filepath}: Error reading file: {e}"]

    for line_num, line in enumerate(lines, 1):
        # Skip safe patterns
        if any(re.search(p, line) for p in SAFE_PATTERNS):
            continue

        for pattern, api_name in METAL_API_PATTERNS:
            if re.search(pattern, line):
                is_protected, reason = is_call_protected(lines, line_num)

                if verbose:
                    status = "OK" if is_protected else "UNPROTECTED"
                    print(f"  {filepath}:{line_num}: {api_name} - {status} ({reason})")

                if not is_protected:
                    errors.append(
                        f"{filepath}:{line_num}: Unprotected {api_name}: {line.strip()[:80]}"
                    )

    return errors

def main():
    parser = argparse.ArgumentParser(description='Check Metal API calls for lock protection')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show all checks, not just errors')
    parser.add_argument('--path', default=None, help='Path to check (default: pytorch-mps-fork)')
    args = parser.parse_args()

    # Find repo root
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent

    if args.path:
        search_path = Path(args.path)
    else:
        search_path = repo_root / 'pytorch-mps-fork' / 'aten' / 'src' / 'ATen'

    if not search_path.exists():
        print(f"ERROR: Path not found: {search_path}")
        sys.exit(2)

    print(f"Checking Metal API lock protection in: {search_path}")
    print()

    # Find all .mm files
    mm_files = list(search_path.rglob('*.mm'))

    # Filter to only MPS backend files (exclude native/metal/ which is a separate MPSCNN backend)
    # Also exclude test files
    mps_files = [f for f in mm_files
                 if ('mps' in str(f).lower() or 'MPS' in str(f))
                 and '/native/metal/' not in str(f)
                 and not any(excl in str(f) for excl in EXCLUDE_PATHS)]

    print(f"Found {len(mps_files)} MPS-related .mm files")
    print()

    all_errors = []
    files_with_errors = set()

    for filepath in sorted(mps_files):
        if args.verbose:
            print(f"Checking: {filepath}")

        errors = check_file(filepath, args.verbose)
        if errors:
            all_errors.extend(errors)
            files_with_errors.add(str(filepath))

    print()
    print("=" * 60)
    if all_errors:
        print(f"FAILED: Found {len(all_errors)} unprotected Metal API call(s)")
        print(f"        in {len(files_with_errors)} file(s)")
        print()
        print("Unprotected calls:")
        for error in all_errors:
            print(f"  {error}")
        print()
        print("FIX: Add MPSEncodingLock before each unprotected call:")
        print("  at::mps::MPSEncodingLock encodingLock;")
        sys.exit(1)
    else:
        print("PASSED: All Metal API calls are protected")
        sys.exit(0)

if __name__ == '__main__':
    main()
