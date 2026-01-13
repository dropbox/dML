#!/usr/bin/env python3
"""
Generate Japanese pitch accent lexicon from pyopenjtalk.

This script creates a C++ header file with:
1. Common Japanese words with their pitch accent patterns
2. Helper functions to apply accent markers to IPA output

Japanese pitch accent:
- Accent position indicates where pitch drops (downstep)
- Position 0 = heiban (flat) - no drop, stays high after first mora
- Position 1 = atamadaka (head-high) - drops after first mora
- Position 2+ = nakadaka/odaka - drops after that mora

Output format: word -> "phonemes|accent_pos|mora_count"
The C++ code will use this to add ↓ markers at correct positions.

Usage:
    python scripts/generate_japanese_lexicon.py > stream-tts-cpp/include/japanese_pitch_lexicon.hpp
"""

import re
import sys

try:
    import pyopenjtalk
except ImportError:
    print("Error: pyopenjtalk not installed. Run: pip install pyopenjtalk", file=sys.stderr)
    sys.exit(1)


def extract_accent(word: str) -> tuple[str, int, int]:
    """Extract phonemes and accent info from a Japanese word.

    Returns: (phonemes, accent_pos, mora_count)
    - phonemes: space-separated phoneme string
    - accent_pos: position of downstep (0 = flat)
    - mora_count: total morae in the word
    """
    try:
        labels = pyopenjtalk.extract_fullcontext(word)
        phonemes = pyopenjtalk.g2p(word, kana=False)

        # Extract accent from F: field (accent_type_mora_count)
        accent_pos = 0
        mora_count = 1

        for label in labels:
            match = re.search(r'/F:(\d+)_(\d+)', label)
            if match:
                accent_pos = int(match.group(1))
                mora_count = int(match.group(2))
                break

        return phonemes, accent_pos, mora_count
    except Exception as e:
        return "", 0, 0


# Common Japanese words for the lexicon
# These are words likely to appear in Claude Code voice output
COMMON_WORDS = [
    # Greetings
    "こんにちは", "こんばんは", "おはよう", "おやすみ",
    "さようなら", "ありがとう", "すみません", "ごめんなさい",
    "はい", "いいえ", "お願いします", "失礼します",

    # Pronouns
    "私", "僕", "俺", "彼", "彼女", "あなた", "誰", "何",
    "これ", "それ", "あれ", "ここ", "そこ", "あそこ",

    # Numbers
    "一", "二", "三", "四", "五", "六", "七", "八", "九", "十",
    "百", "千", "万", "零",
    "一つ", "二つ", "三つ", "四つ", "五つ",

    # Time
    "今", "今日", "明日", "昨日", "今年", "来年", "去年",
    "朝", "昼", "夜", "午前", "午後",
    "時間", "分", "秒", "日", "週", "月", "年",

    # Programming terms (common in coding context)
    "関数", "変数", "配列", "文字列", "数値", "型",
    "クラス", "オブジェクト", "メソッド", "プロパティ",
    "エラー", "バグ", "テスト", "デバッグ", "コード",
    "コンパイル", "実行", "処理", "入力", "出力",
    "ファイル", "フォルダ", "ディレクトリ",
    "データ", "データベース", "サーバー", "クライアント",
    "ネットワーク", "インターネット", "ウェブ",
    "プログラム", "ソフトウェア", "ハードウェア",
    "アルゴリズム", "ロジック", "条件", "ループ",

    # Actions
    "する", "なる", "ある", "いる", "行く", "来る", "見る", "聞く",
    "言う", "書く", "読む", "作る", "使う", "分かる", "知る",
    "思う", "考える", "決める", "始める", "終わる", "続ける",
    "開く", "閉じる", "動く", "止まる", "変わる", "変える",

    # Adjectives
    "良い", "悪い", "大きい", "小さい", "多い", "少ない",
    "新しい", "古い", "高い", "低い", "長い", "短い",
    "早い", "遅い", "速い", "難しい", "簡単", "正しい",
    "同じ", "違う", "必要", "重要", "大切", "大丈夫",

    # Particles and conjunctions (for context)
    "です", "ます", "だから", "しかし", "また", "そして",
    "もし", "ただし", "例えば", "つまり", "それで",

    # Technical status words
    "成功", "失敗", "完了", "開始", "停止", "待機",
    "実行中", "処理中", "読み込み中", "保存中",
    "エラー", "警告", "情報", "確認",

    # Common nouns
    "日本", "日本語", "英語", "言語", "言葉",
    "人", "名前", "場所", "時", "物", "事", "所",
    "問題", "答え", "質問", "説明", "例",
    "結果", "理由", "方法", "目的", "意味",

    # Tokyo and common places
    "東京", "大阪", "京都", "横浜", "名古屋",

    # Coding-specific phrases
    "インポート", "エクスポート", "インストール",
    "アップデート", "ダウンロード", "アップロード",
    "セーブ", "ロード", "リセット", "クリア",
    "コピー", "ペースト", "カット", "アンドゥ",
    "プルリクエスト", "コミット", "プッシュ", "プル",
    "マージ", "ブランチ", "リポジトリ",
]

# Additional common words from frequency lists
FREQUENCY_WORDS = [
    # Top 100 most common Japanese words
    "の", "に", "は", "を", "た", "が", "で", "て", "と", "し",
    "れ", "さ", "ある", "いる", "も", "する", "から", "な", "こと", "として",
    "い", "や", "れる", "など", "なっ", "ない", "この", "ため", "その", "あっ",
    "よう", "また", "もの", "という", "あり", "まで", "られ", "なる", "へ", "か",
    "だ", "これ", "によって", "により", "おり", "より", "による", "ず", "なり", "られる",
    "において", "ば", "なかっ", "なく", "しかし", "について", "せ", "だっ", "その後", "できる",
    "それ", "う", "ので", "なお", "のみ", "でき", "き", "つ", "における", "および",
    "いう", "さらに", "でも", "ら", "たり", "その他", "に関する", "たち", "ます", "ん",
    "なら", "に対して", "特に", "せる", "及び", "これら", "とき", "では", "にて", "ほか",
    "ながら", "うち", "そう", "もっとも", "ところ", "ただし", "にあたって", "として", "場合", "において",
]


def phonemes_to_ipa(phonemes: str) -> str:
    """Convert pyopenjtalk phonemes to Kokoro-compatible IPA.

    pyopenjtalk uses: a, i, u, e, o, N, k, s, t, n, h, m, y, r, w, g, z, d, b, p, ch, sh, j, f, ts, ky, ...
    Kokoro expects: similar IPA with specific mappings
    """
    # Mapping from pyopenjtalk to IPA (matching existing japanese_g2p.hpp style)
    mapping = {
        'N': 'n',      # Syllabic N
        'ch': 'ʨ',     # Chi
        'sh': 'ɕ',     # Shi
        'j': 'ʥ',      # Ji
        'ts': 'ʦ',     # Tsu
        'ky': 'kj',    # Kya, kyu, kyo palatalized
        'gy': 'gj',
        'ny': 'ɲ',
        'hy': 'çj',
        'my': 'mj',
        'ry': 'ɾj',
        'by': 'bj',
        'py': 'pj',
        'dy': 'dj',
        'ty': 'tj',
        'r': 'ɾ',      # Japanese r is a flap
        'u': 'ɯ',      # Japanese u is unrounded
        'f': 'ɸ',      # Japanese f is bilabial fricative
        'hi': 'çi',    # hi has palatal fricative
    }

    result = phonemes
    # Apply mappings (order matters - do longer patterns first)
    for old, new in sorted(mapping.items(), key=lambda x: -len(x[0])):
        result = result.replace(old, new)

    # Remove spaces (join into continuous IPA)
    result = result.replace(' ', '')

    return result


def generate_lexicon():
    """Generate the C++ header file content."""

    all_words = list(set(COMMON_WORDS + FREQUENCY_WORDS))
    all_words.sort()  # Sort for deterministic output

    entries = []

    for word in all_words:
        phonemes, accent_pos, mora_count = extract_accent(word)
        if phonemes and mora_count > 0:
            ipa = phonemes_to_ipa(phonemes)
            # Store as: word -> "ipa|accent_pos|mora_count"
            entries.append((word, ipa, accent_pos, mora_count))

    # Generate header
    print("""#pragma once
// Japanese pitch accent lexicon - AUTO-GENERATED
// Generated by scripts/generate_japanese_lexicon.py using pyopenjtalk
//
// Format: character/word -> "ipa|accent_position|mora_count"
// - accent_position: position where pitch drops (0 = flat/heiban)
// - mora_count: total morae in the word
//
// Japanese pitch accent patterns:
// - Position 0 (heiban): stays high after initial rise, no drop
// - Position 1 (atamadaka): high on first mora, drops after
// - Position 2+ (nakadaka/odaka): rises, drops after that position
//
// Copyright 2025 Andrew Yates. All rights reserved.

#include <string>
#include <unordered_map>

namespace japanese_pitch {

// Pitch accent lexicon: word -> "ipa|accent_pos|mora_count"
inline const std::unordered_map<std::string, std::string> PITCH_LEXICON = {""")

    for word, ipa, accent, moras in entries:
        # Escape any special characters
        ipa_escaped = ipa.replace('\\', '\\\\').replace('"', '\\"')
        print(f'    {{"{word}", "{ipa_escaped}|{accent}|{moras}"}},')

    print("""};

// Parse a lexicon entry to extract IPA, accent position, and mora count
struct PitchEntry {
    std::string ipa;
    int accent_pos;  // 0 = flat, 1+ = downstep after that mora
    int mora_count;
};

inline PitchEntry parse_pitch_entry(const std::string& entry) {
    PitchEntry result;
    result.accent_pos = 0;
    result.mora_count = 1;

    // Format: "ipa|accent_pos|mora_count"
    size_t pos1 = entry.find('|');
    if (pos1 == std::string::npos) {
        result.ipa = entry;
        return result;
    }

    result.ipa = entry.substr(0, pos1);

    size_t pos2 = entry.find('|', pos1 + 1);
    if (pos2 == std::string::npos) {
        return result;
    }

    try {
        result.accent_pos = std::stoi(entry.substr(pos1 + 1, pos2 - pos1 - 1));
        result.mora_count = std::stoi(entry.substr(pos2 + 1));
    } catch (...) {
        // Keep defaults on parse error
    }

    return result;
}

// Add pitch accent marker (↓) to IPA string based on accent position
// The ↓ is placed after the mora where pitch drops
inline std::string add_pitch_marker(const std::string& ipa, int accent_pos, int mora_count) {
    if (accent_pos == 0 || accent_pos > mora_count) {
        // Flat pattern (heiban) - no marker needed
        // Or invalid accent position
        return ipa;
    }

    // Count morae in the IPA string
    // Japanese morae: each vowel (a, i, u, e, o, ɯ) or N counts as one mora
    // Long vowels (oo, ee, etc.) count as two morae
    std::string result;
    int current_mora = 0;
    bool marker_added = false;

    size_t i = 0;
    while (i < ipa.size()) {
        unsigned char c = static_cast<unsigned char>(ipa[i]);

        // Determine character length (UTF-8)
        int len = 1;
        if ((c & 0xF0) == 0xF0) len = 4;
        else if ((c & 0xE0) == 0xE0) len = 3;
        else if ((c & 0xC0) == 0xC0) len = 2;

        if (i + len > ipa.size()) break;

        std::string ch = ipa.substr(i, len);
        result += ch;

        // Check if this is a vowel (mora boundary)
        bool is_vowel = (ch == "a" || ch == "i" || ch == "u" || ch == "e" || ch == "o" ||
                         ch == "ɯ" || ch == "ɪ" || ch == "ʊ" || ch == "ə");

        // Also check for syllabic N (ん)
        bool is_n = (ch == "n" && (i + len >= ipa.size() ||
                    (ipa[i + len] != 'a' && ipa[i + len] != 'i' && ipa[i + len] != 'u' &&
                     ipa[i + len] != 'e' && ipa[i + len] != 'o')));

        if (is_vowel || is_n) {
            current_mora++;

            // Add downstep marker after the accent position
            if (!marker_added && current_mora == accent_pos) {
                result += "↓";  // Unicode U+2193 DOWNWARDS ARROW
                marker_added = true;
            }
        }

        i += len;
    }

    return result;
}

// Look up a word and return IPA with pitch accent marker
// Returns empty string if word not found
inline std::string lookup_with_pitch(const std::string& word) {
    auto it = PITCH_LEXICON.find(word);
    if (it == PITCH_LEXICON.end()) {
        return "";  // Not found
    }

    PitchEntry entry = parse_pitch_entry(it->second);
    return add_pitch_marker(entry.ipa, entry.accent_pos, entry.mora_count);
}

}  // namespace japanese_pitch
""")


if __name__ == "__main__":
    generate_lexicon()
