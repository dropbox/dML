// Copyright 2024-2025 Andrew Yates
//
// Test for Misaki G2P C++ implementation - All 9 Kokoro languages

#include "misaki_g2p.h"
#include <iostream>
#include <cassert>

// Forward declaration of global lexicon path
extern std::string g_lexicon_path;

void test_english() {
    std::cout << "\n=== Testing English (en-us) ===\n";
    misaki::MisakiG2P g2p;

    if (!g2p.initialize(g_lexicon_path, "en-us")) {
        std::cerr << "  FAIL: Could not initialize English G2P\n";
        return;
    }
    std::cout << "  Loaded " << g2p.lexicon_size() << " entries\n";

    // Test words
    std::cout << "  'hello' -> '" << g2p.phonemize("hello") << "'\n";
    std::cout << "  'world' -> '" << g2p.phonemize("world") << "'\n";
    std::cout << "  'Hello world' -> '" << g2p.phonemize("Hello world") << "'\n";

    // Test symbols
    std::cout << "  '%' -> '" << g2p.phonemize("%") << "'\n";
    std::cout << "  '&' -> '" << g2p.phonemize("&") << "'\n";
}

void test_english_gb() {
    std::cout << "\n=== Testing British English (en-gb) ===\n";
    misaki::MisakiG2P g2p;

    if (!g2p.initialize(g_lexicon_path, "en-gb")) {
        std::cerr << "  FAIL: Could not initialize British English G2P\n";
        return;
    }
    std::cout << "  Loaded " << g2p.lexicon_size() << " entries\n";

    std::cout << "  'hello' -> '" << g2p.phonemize("hello") << "'\n";
    std::cout << "  'colour' -> '" << g2p.phonemize("colour") << "'\n";
}

void test_japanese() {
    std::cout << "\n=== Testing Japanese (ja) ===\n";
    misaki::MisakiG2P g2p;

    if (!g2p.initialize(g_lexicon_path, "ja")) {
        std::cerr << "  FAIL: Could not initialize Japanese G2P\n";
        return;
    }
    std::cout << "  Loaded " << g2p.lexicon_size() << " entries\n";

    // Test hiragana
    std::cout << "\n  --- Hiragana input ---\n";
    std::cout << "  'こんにちは' -> '" << g2p.phonemize("こんにちは") << "'\n";
    std::cout << "  'さくら' -> '" << g2p.phonemize("さくら") << "'\n";
    std::cout << "  'きょう' -> '" << g2p.phonemize("きょう") << "'\n";  // Should handle combo

    // Test katakana (should convert to hiragana then IPA)
    std::cout << "\n  --- Katakana input ---\n";
    std::cout << "  'コンピュータ' -> '" << g2p.phonemize("コンピュータ") << "'\n";

    // Test kanji (NEW - requires MeCab)
    std::cout << "\n  --- Kanji input (NEW - requires MeCab) ---\n";
    std::cout << "  '日本語' (nihongo) -> '" << g2p.phonemize("日本語") << "'\n";
    std::cout << "  '東京' (tokyo) -> '" << g2p.phonemize("東京") << "'\n";
    std::cout << "  '今日' (kyou) -> '" << g2p.phonemize("今日") << "'\n";
    std::cout << "  '私' (watashi) -> '" << g2p.phonemize("私") << "'\n";
    std::cout << "  '日本語を話す' (nihongo wo hanasu) -> '" << g2p.phonemize("日本語を話す") << "'\n";

    // Test punctuation
    std::cout << "\n  --- Punctuation ---\n";
    std::cout << "  '。' -> '" << g2p.phonemize("。") << "'\n";
    std::cout << "  '、' -> '" << g2p.phonemize("、") << "'\n";
}

void test_chinese() {
    std::cout << "\n=== Testing Chinese (zh) ===\n";
    misaki::MisakiG2P g2p;

    if (!g2p.initialize(g_lexicon_path, "zh")) {
        std::cerr << "  FAIL: Could not initialize Chinese G2P\n";
        return;
    }
    std::cout << "  Loaded " << g2p.lexicon_size() << " entries\n";

    // Test pinyin with tones (original functionality)
    std::cout << "\n  --- Pinyin input ---\n";
    std::cout << "  'ni3' -> '" << g2p.phonemize("ni3") << "'\n";
    std::cout << "  'hao3' -> '" << g2p.phonemize("hao3") << "'\n";
    std::cout << "  'ni3hao3' -> '" << g2p.phonemize("ni3hao3") << "'\n";
    std::cout << "  'zhong1guo2' -> '" << g2p.phonemize("zhong1guo2") << "'\n";

    // Test hanzi characters (NEW functionality)
    std::cout << "\n  --- Hanzi input (NEW) ---\n";
    std::cout << "  '你好' -> '" << g2p.phonemize("你好") << "'\n";
    std::cout << "  '中国' -> '" << g2p.phonemize("中国") << "'\n";
    std::cout << "  '世界' -> '" << g2p.phonemize("世界") << "'\n";
    std::cout << "  '你好世界' -> '" << g2p.phonemize("你好世界") << "'\n";
    std::cout << "  '我爱中国' -> '" << g2p.phonemize("我爱中国") << "'\n";
}

void test_spanish() {
    std::cout << "\n=== Testing Spanish (es) ===\n";
    misaki::MisakiG2P g2p;

    if (!g2p.initialize(g_lexicon_path, "es")) {
        std::cerr << "  FAIL: Could not initialize Spanish G2P (requires espeak-ng)\n";
        return;
    }

    std::cout << "  'hola' -> '" << g2p.phonemize("hola") << "'\n";
    std::cout << "  'mundo' -> '" << g2p.phonemize("mundo") << "'\n";
    std::cout << "  'Buenos dias' -> '" << g2p.phonemize("Buenos dias") << "'\n";
}

void test_french() {
    std::cout << "\n=== Testing French (fr) ===\n";
    misaki::MisakiG2P g2p;

    if (!g2p.initialize(g_lexicon_path, "fr")) {
        std::cerr << "  FAIL: Could not initialize French G2P (requires espeak-ng)\n";
        return;
    }

    std::cout << "  'bonjour' -> '" << g2p.phonemize("bonjour") << "'\n";
    std::cout << "  'monde' -> '" << g2p.phonemize("monde") << "'\n";
    std::cout << "  'Bonjour monde' -> '" << g2p.phonemize("Bonjour monde") << "'\n";
}

void test_hindi() {
    std::cout << "\n=== Testing Hindi (hi) ===\n";
    std::cout << "  NOTE: Hindi requires Devanagari script, not romanized text\n";
    misaki::MisakiG2P g2p;

    if (!g2p.initialize(g_lexicon_path, "hi")) {
        std::cerr << "  FAIL: Could not initialize Hindi G2P (requires espeak-ng)\n";
        return;
    }

    // Devanagari script (correct)
    std::cout << "  'नमस्ते' (namaste) -> '" << g2p.phonemize("नमस्ते") << "'\n";
    std::cout << "  'धन्यवाद' (dhanyavad) -> '" << g2p.phonemize("धन्यवाद") << "'\n";
    std::cout << "  'भारत' (bharat) -> '" << g2p.phonemize("भारत") << "'\n";
    std::cout << "  'हिंदी' (hindi) -> '" << g2p.phonemize("हिंदी") << "'\n";

    // Show what happens with romanized (incorrect - for documentation)
    std::cout << "  'namaste' (romanized, wrong) -> '" << g2p.phonemize("namaste") << "'\n";
}

void test_italian() {
    std::cout << "\n=== Testing Italian (it) ===\n";
    misaki::MisakiG2P g2p;

    if (!g2p.initialize(g_lexicon_path, "it")) {
        std::cerr << "  FAIL: Could not initialize Italian G2P (requires espeak-ng)\n";
        return;
    }

    std::cout << "  'ciao' -> '" << g2p.phonemize("ciao") << "'\n";
    std::cout << "  'mondo' -> '" << g2p.phonemize("mondo") << "'\n";
    std::cout << "  'Buongiorno' -> '" << g2p.phonemize("Buongiorno") << "'\n";
}

void test_portuguese() {
    std::cout << "\n=== Testing Brazilian Portuguese (pt-br) ===\n";
    misaki::MisakiG2P g2p;

    if (!g2p.initialize(g_lexicon_path, "pt-br")) {
        std::cerr << "  FAIL: Could not initialize Portuguese G2P (requires espeak-ng)\n";
        return;
    }

    std::cout << "  'ola' -> '" << g2p.phonemize("ola") << "'\n";
    std::cout << "  'mundo' -> '" << g2p.phonemize("mundo") << "'\n";
    std::cout << "  'Bom dia' -> '" << g2p.phonemize("Bom dia") << "'\n";
}

void test_korean() {
    std::cout << "\n=== Testing Korean (ko) ===\n";
    std::cout << "  NOTE: Korean requires Hangul script, not romanized text\n";
    misaki::MisakiG2P g2p;

    if (!g2p.initialize(g_lexicon_path, "ko")) {
        std::cerr << "  FAIL: Could not initialize Korean G2P (requires espeak-ng)\n";
        return;
    }

    // Hangul script (correct)
    std::cout << "  '안녕하세요' (hello) -> '" << g2p.phonemize("안녕하세요") << "'\n";
    std::cout << "  '감사합니다' (thank you) -> '" << g2p.phonemize("감사합니다") << "'\n";
    std::cout << "  '한국어' (Korean) -> '" << g2p.phonemize("한국어") << "'\n";
    std::cout << "  '서울' (Seoul) -> '" << g2p.phonemize("서울") << "'\n";

    // Show what happens with romanized (incorrect - for documentation)
    std::cout << "  'annyeonghaseyo' (romanized, wrong) -> '" << g2p.phonemize("annyeonghaseyo") << "'\n";
}

void test_vietnamese() {
    std::cout << "\n=== Testing Vietnamese (vi) ===\n";
    misaki::MisakiG2P g2p;

    if (!g2p.initialize(g_lexicon_path, "vi")) {
        std::cerr << "  FAIL: Could not initialize Vietnamese G2P (requires espeak-ng)\n";
        return;
    }

    // Vietnamese uses Latin script with diacritics
    std::cout << "  'xin chào' (hello) -> '" << g2p.phonemize("xin chào") << "'\n";
    std::cout << "  'cảm ơn' (thank you) -> '" << g2p.phonemize("cảm ơn") << "'\n";
    std::cout << "  'Việt Nam' -> '" << g2p.phonemize("Việt Nam") << "'\n";
    std::cout << "  'Hà Nội' -> '" << g2p.phonemize("Hà Nội") << "'\n";
}

// Global lexicon path (can be overridden by command line)
std::string g_lexicon_path = "misaki_export";

int main(int argc, char* argv[]) {
    // Allow specifying lexicon path as first argument
    if (argc > 1) {
        g_lexicon_path = argv[1];
    }

    std::cout << "=== Misaki G2P Test - All 11 Supported Languages ===\n";
    std::cout << "Using lexicon path: " << g_lexicon_path << "\n";

    // Misaki-based languages (lexicon lookup)
    test_english();
    test_english_gb();
    test_japanese();
    test_chinese();

    // espeak-ng based languages
    test_spanish();
    test_french();
    test_hindi();
    test_italian();
    test_portuguese();
    test_korean();
    test_vietnamese();

    std::cout << "\n=== All language tests completed! ===\n";
    return 0;
}
