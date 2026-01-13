# Misaki G2P C++ Implementation

C++ implementation of grapheme-to-phoneme conversion for 11 languages, optimized with mmap binary lexicons.

## Supported Languages

| Language | Code | Method | Binary Lexicons | Writing Systems |
|----------|------|--------|-----------------|-----------------|
| English (US) | `en-us` | Misaki lexicons | `us_golds.bin`, `us_silvers.bin` | Latin |
| English (GB) | `en-gb` | Misaki lexicons | `gb_golds.bin`, `gb_silvers.bin` | Latin |
| Japanese | `ja` | Misaki + MeCab | `hepburn.bin` | Hiragana, Katakana, Kanji |
| Chinese | `zh` | Misaki lexicons | `hanzi_to_pinyin.bin`, `pinyin_to_ipa.bin` | Hanzi, Pinyin |
| Spanish | `es` | espeak-ng | - | Latin |
| French | `fr` | espeak-ng | - | Latin |
| Hindi | `hi` | espeak-ng | - | Devanagari |
| Italian | `it` | espeak-ng | - | Latin |
| Portuguese (BR) | `pt-br` | espeak-ng | - | Latin |
| Korean | `ko` | espeak-ng | - | Hangul |
| Vietnamese | `vi` | espeak-ng | - | Latin with diacritics |

## Performance

| Language | Entries | Load Time | Method |
|----------|---------|-----------|--------|
| English (US) | 362,842 | <1ms | mmap binary |
| English (GB) | 389,466 | <1ms | mmap binary |
| Japanese | 194 + MeCab | <1ms | mmap binary |
| Chinese | 26,703 hanzi + 4,095 pinyin | <1ms | mmap binary |
| espeak-ng languages | - | instant | system library |

## Usage

```cpp
#include "misaki_g2p.h"

// Basic usage
misaki::MisakiG2P g2p;
g2p.initialize("misaki_export", "en-us");
std::string ipa = g2p.phonemize("Hello world");
// Result: "həlˈO wˈɜɹld"

// Japanese (all writing systems)
g2p.initialize("misaki_export", "ja");
g2p.phonemize("こんにちは");  // Hiragana -> "koɴɲiʨiha"
g2p.phonemize("コンピュータ"); // Katakana -> "koɴpʲɨːta"
g2p.phonemize("日本語");       // Kanji -> "ɲihoɴɡo" (via MeCab)

// Chinese (both writing systems)
g2p.initialize("misaki_export", "zh");
g2p.phonemize("ni3hao3");  // Pinyin -> "ni˧˩˧xau̯˧˩˧"
g2p.phonemize("你好");     // Hanzi -> "ni˧˩˧xau̯˧˩˧"
```

## Path Configuration

**IMPORTANT**: The lexicon path must be correct or loading will fail silently and fall back to slow JSON parsing.

### Option 1: Run from project root (recommended)
```bash
./build/test_misaki_g2p
# Uses relative path "misaki_export" from current directory
```

### Option 2: Explicit path argument
```bash
./build/test_misaki_g2p /absolute/path/to/misaki_export
```

### Option 3: Symlink in build directory
```bash
ln -sf /path/to/misaki_export build/misaki_export
cd build && ./test_misaki_g2p
```

## Binary Lexicon Format

Binary lexicons use mmap for zero-copy loading. Format:

```
Header (32 bytes):
  - Magic: "MLX2" (4 bytes)
  - Version: 2 (4 bytes)
  - Entry count (4 bytes)
  - String table size (4 bytes)
  - Reserved (16 bytes)

Index table (12 bytes per entry):
  - Key offset (4 bytes)
  - Key length (2 bytes)
  - Value offset (4 bytes)
  - Value length (2 bytes)

String table:
  - Concatenated key-value strings
```

## Japanese Writing System Support

| Writing | Range | Handling |
|---------|-------|----------|
| Hiragana | U+3040-U+309F | Direct lookup in `hepburn.bin` |
| Katakana | U+30A0-U+30FF | Auto-converted to hiragana, then lookup |
| Kanji | U+4E00-U+9FFF | MeCab extracts reading -> hiragana -> lookup |

**MeCab dependency**: Kanji support requires MeCab to be installed and compiled with `USE_MECAB`:
```bash
brew install mecab mecab-ipadic
cmake -DUSE_MECAB=ON ..
```

## Chinese Writing System Support

| Writing | Handling |
|---------|----------|
| Hanzi (汉字) | `hanzi_to_pinyin.bin` (26,703 chars) -> pinyin -> IPA |
| Pinyin with tones | Direct lookup in `pinyin_to_ipa.bin` (4,095 syllables) |

Tone numbers (1-5) are required for pinyin: `ni3hao3`, `zhong1guo2`

## Script Warnings

The following languages require native script input:

| Language | Required Script | Example |
|----------|-----------------|---------|
| Hindi | Devanagari | नमस्ते (not "namaste") |
| Korean | Hangul | 안녕하세요 (not "annyeonghaseyo") |

Romanized input will trigger a warning and produce incorrect output.

## Generating Binary Lexicons

If binary lexicons don't exist, generate them:

```bash
python scripts/export_misaki_lexicons.py
python scripts/convert_lexicons_to_binary.py
```

## Files

| File | Purpose |
|------|---------|
| `misaki_g2p.h` | Header with MisakiG2P class |
| `misaki_g2p.cpp` | Implementation (~1800 lines) |
| `test_misaki_g2p.cpp` | Test all 11 languages |
| `misaki_export/` | Lexicon directory |

## Troubleshooting

### "MisakiG2P: v2 binary not found, falling back to hash map loading"
Binary lexicons not found. Check path or generate them.

### Test hangs for hours
Path is wrong - falling back to slow JSON parsing. Fix path or use explicit argument.

### Kanji not converting
MeCab not installed or not compiled with `USE_MECAB`.

### Hindi/Korean output is wrong
Using romanized input instead of native script (Devanagari/Hangul).

## Commit History

- `e01f846` - Fixed path issue, added command-line argument support
- Binary mmap lexicons provide 241x speedup over JSON (82ms -> 0.34ms)
