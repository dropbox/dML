#!/usr/bin/env python3
"""
Export Misaki JSON lexicons to binary format for fast C++ loading.

Binary format v2 (sorted, zero-copy):
┌─────────────────────────────────────┐
│ Header (32 bytes)                   │
│   - magic: 4 bytes "MLX2"           │
│   - version: 4 bytes (uint32) = 2   │
│   - entry_count: 4 bytes (uint32)   │
│   - string_table_size: 4 bytes      │
│   - reserved: 16 bytes              │
├─────────────────────────────────────┤
│ Index table (entry_count * 12 bytes)│
│   - key_offset: 4 bytes (uint32)    │
│   - key_length: 2 bytes (uint16)    │
│   - value_offset: 4 bytes (uint32)  │
│   - value_length: 2 bytes (uint16)  │
├─────────────────────────────────────┤
│ String table (packed, no null-term) │
│   - all keys followed by values     │
│   - SORTED BY KEY for binary search │
└─────────────────────────────────────┘

Key features:
- Entries sorted by key for O(log n) binary search
- Key/value lengths stored, no null terminators needed
- Zero-copy: mmap file and use string_view directly
- Target: <5ms load time (vs 68ms with hash table)

Usage:
    python export_binary_lexicon.py misaki_export/en/us_golds.json
    python export_binary_lexicon.py --all misaki_export/
"""

import argparse
import json
import struct
import sys
from pathlib import Path


MAGIC = b"MLX2"
VERSION = 2


def flatten_value(value):
    """Flatten a value that may be a string, dict, or list."""
    if isinstance(value, str):
        return value
    elif isinstance(value, dict):
        # Heterophonic words have {DEFAULT: ..., VERB: ..., NOUN: ...}
        # Use DEFAULT if available, otherwise first value
        if "DEFAULT" in value:
            return flatten_value(value["DEFAULT"])
        else:
            return flatten_value(next(iter(value.values())))
    elif isinstance(value, list):
        # Array format ["ipa"] or ["ipa1", "ipa2"]
        return value[0] if value else ""
    else:
        return str(value) if value else ""


def export_lexicon(json_path: Path, output_path: Path = None) -> None:
    """Convert a JSON lexicon to binary format v2 (sorted, zero-copy)."""
    if output_path is None:
        output_path = json_path.with_suffix(".bin")

    print(f"Loading {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle both dict and array-of-arrays formats
    entries = []
    if isinstance(data, dict):
        for k, v in data.items():
            entries.append((k, flatten_value(v)))
    elif isinstance(data, list):
        # Array format like [["key", "value"], ...]
        for item in data:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                entries.append((item[0], flatten_value(item[1])))
    else:
        print(f"  Skipping {json_path}: unsupported format")
        return

    if not entries:
        print(f"  Skipping {json_path}: empty")
        return

    # SORT entries by key for binary search
    entries.sort(key=lambda x: x[0])
    print(f"  {len(entries)} entries (sorted)")

    # Build string table and index with lengths (no null terminators)
    string_table = bytearray()
    index_table = []

    for key, value in entries:
        key_bytes = key.encode("utf-8")
        value_bytes = value.encode("utf-8") if value else b""

        key_offset = len(string_table)
        key_length = len(key_bytes)
        string_table.extend(key_bytes)

        value_offset = len(string_table)
        value_length = len(value_bytes)
        string_table.extend(value_bytes)

        # Check length limits (uint16)
        if key_length > 65535 or value_length > 65535:
            print(f"  Warning: truncating entry '{key[:20]}...' (lengths exceed uint16)")
            continue

        index_table.append((key_offset, key_length, value_offset, value_length))

    # Write binary file v2
    with open(output_path, "wb") as f:
        # Header (32 bytes)
        header = struct.pack(
            "<4sIII16s",
            MAGIC,
            VERSION,
            len(index_table),
            len(string_table),
            b"\x00" * 16  # reserved
        )
        f.write(header)

        # Index table (12 bytes per entry)
        for key_offset, key_length, value_offset, value_length in index_table:
            f.write(struct.pack("<IHIH", key_offset, key_length, value_offset, value_length))

        # String table (packed, no null terminators)
        f.write(bytes(string_table))

    # Report sizes
    json_size = json_path.stat().st_size
    bin_size = output_path.stat().st_size
    ratio = bin_size / json_size * 100
    print(f"  Output: {output_path}")
    print(f"  JSON: {json_size:,} bytes -> Binary: {bin_size:,} bytes ({ratio:.1f}%)")


def export_all(misaki_dir: Path) -> None:
    """Export all JSON lexicons in the misaki_export directory."""
    json_files = [
        # English
        "en/us_golds.json",
        "en/us_silvers.json",
        "en/gb_golds.json",
        "en/gb_silvers.json",
        "en/symbols.json",
        "en/add_symbols.json",
        # Japanese
        "ja/hepburn.json",
        # Chinese
        "zh/pinyin_to_ipa.json",
    ]

    for rel_path in json_files:
        json_path = misaki_dir / rel_path
        if json_path.exists():
            try:
                export_lexicon(json_path)
            except Exception as e:
                print(f"  Error: {e}")
        else:
            print(f"Skipping {rel_path}: not found")


def main():
    parser = argparse.ArgumentParser(description="Export Misaki lexicons to binary format")
    parser.add_argument("path", help="JSON file or misaki_export directory")
    parser.add_argument("--all", action="store_true", help="Export all lexicons in directory")
    parser.add_argument("-o", "--output", help="Output file (for single file export)")
    args = parser.parse_args()

    path = Path(args.path)

    if args.all or path.is_dir():
        export_all(path)
    elif path.is_file():
        output = Path(args.output) if args.output else None
        export_lexicon(path, output)
    else:
        print(f"Error: {path} not found")
        sys.exit(1)


if __name__ == "__main__":
    main()
