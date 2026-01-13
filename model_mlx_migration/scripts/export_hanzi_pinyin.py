#!/usr/bin/env python3
"""
Export hanzi-to-pinyin dictionary from pypinyin for C++ loading.

This creates a dictionary mapping Chinese characters (hanzi) to pinyin with tone numbers.
The pinyin output can then be processed by the existing pinyin_to_ipa pipeline.

Usage:
    .venv/bin/python scripts/export_hanzi_pinyin.py
"""

import json
import struct
from pathlib import Path

# Import pypinyin
from pypinyin import pinyin, Style


def get_pinyin_with_tone(char):
    """Get pinyin with tone number for a single character."""
    try:
        result = pinyin(char, style=Style.TONE3, heteronym=False)
        if result and result[0]:
            return result[0][0]
    except:
        pass
    return None


def export_single_char_dict():
    """Export dictionary for single characters (most common ~20K chars)."""
    entries = {}

    # CJK Unified Ideographs: U+4E00 to U+9FFF (most common ~20K chars)
    print("Processing CJK Unified Ideographs (U+4E00-U+9FFF)...")
    count = 0
    for cp in range(0x4E00, 0x9FFF + 1):
        char = chr(cp)
        py = get_pinyin_with_tone(char)
        if py and py != char:  # Skip if pypinyin returns the char itself
            entries[char] = py
            count += 1
        if count % 5000 == 0 and count > 0:
            print(f"  Processed {count} characters...")

    print(f"  Total from main block: {len(entries)}")

    # CJK Extension A: U+3400 to U+4DBF (less common ~6K chars)
    print("Processing CJK Extension A (U+3400-U+4DBF)...")
    ext_count = 0
    for cp in range(0x3400, 0x4DBF + 1):
        char = chr(cp)
        py = get_pinyin_with_tone(char)
        if py and py != char:
            entries[char] = py
            ext_count += 1

    print(f"  Total from Extension A: {ext_count}")
    print(f"Total single characters: {len(entries)}")
    return entries


def export_to_json(entries, output_path):
    """Export to JSON format."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(entries, f, ensure_ascii=False, indent=None, separators=(',', ':'))
    print(f"Exported {len(entries)} entries to {output_path}")
    print(f"File size: {output_path.stat().st_size:,} bytes")


def export_to_binary_v2(entries, output_path):
    """Export to v2 binary format (sorted, zero-copy compatible)."""
    # Sort entries by key
    sorted_entries = sorted(entries.items(), key=lambda x: x[0])

    # Build string table and index
    string_table = bytearray()
    index_table = []

    for key, value in sorted_entries:
        key_bytes = key.encode('utf-8')
        value_bytes = value.encode('utf-8')

        key_offset = len(string_table)
        key_length = len(key_bytes)
        string_table.extend(key_bytes)

        value_offset = len(string_table)
        value_length = len(value_bytes)
        string_table.extend(value_bytes)

        index_table.append((key_offset, key_length, value_offset, value_length))

    # Write binary file
    with open(output_path, 'wb') as f:
        # Header (32 bytes)
        header = struct.pack(
            '<4sIII16s',
            b'MLX2',
            2,  # version
            len(index_table),
            len(string_table),
            b'\x00' * 16
        )
        f.write(header)

        # Index table (12 bytes per entry)
        for key_offset, key_length, value_offset, value_length in index_table:
            f.write(struct.pack('<IHIH', key_offset, key_length, value_offset, value_length))

        # String table
        f.write(bytes(string_table))

    print(f"Exported {len(index_table)} entries to {output_path}")
    print(f"File size: {output_path.stat().st_size:,} bytes")


def main():
    output_dir = Path('misaki_export/zh')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Exporting Hanzi-to-Pinyin Dictionary ===\n")

    # Export single characters
    entries = export_single_char_dict()

    # Export to JSON
    json_path = output_dir / 'hanzi_to_pinyin.json'
    export_to_json(entries, json_path)

    # Export to v2 binary
    bin_path = output_dir / 'hanzi_to_pinyin.bin'
    export_to_binary_v2(entries, bin_path)

    # Test a few conversions
    print("\n=== Test Conversions ===")
    test_chars = ['你', '好', '中', '国', '日', '本', '世', '界', '语', '言']
    for char in test_chars:
        py = entries.get(char, '?')
        print(f"  '{char}' -> '{py}'")

    print("\nDone!")


if __name__ == '__main__':
    main()
