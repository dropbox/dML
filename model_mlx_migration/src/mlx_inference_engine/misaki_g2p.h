// Copyright 2024-2025 Andrew Yates
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Misaki G2P - Multilingual phonemizer for Kokoro TTS
//
// This implementation uses exported Misaki lexicons (JSON format) for
// text-to-phoneme conversion. Misaki is Kokoro's official G2P and produces
// phonemes that match what the phoneme head was trained on.
//
// Supported languages (all 9 Kokoro languages):
//   - English (en-us, en-gb): 362K+ word lexicons (Misaki)
//   - Japanese (ja): Hiragana/katakana character-to-IPA (Misaki)
//   - Chinese (zh): Pinyin-to-IPA mapping (Misaki)
//   - Spanish (es): espeak-ng backend
//   - French (fr): espeak-ng backend
//   - Hindi (hi): espeak-ng backend
//   - Italian (it): espeak-ng backend
//   - Brazilian Portuguese (pt-br): espeak-ng backend
//
// Languages with Misaki support use lexicons for accurate phonemization.
// Other languages fall back to espeak-ng which produces compatible IPA.

#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <memory>

namespace misaki {

/**
 * Language/dialect identifiers for all 11 supported languages.
 * Includes the 9 Kokoro languages plus Korean and Vietnamese from Misaki.
 */
enum class Language {
    // Misaki-based (lexicon lookup)
    EN_US,   // American English (a)
    EN_GB,   // British English (b)
    JA,      // Japanese (j)
    ZH,      // Chinese/Mandarin (z)

    // espeak-ng based (requires espeak-ng library)
    ES,      // Spanish (e)
    FR,      // French (f)
    HI,      // Hindi (h) - requires Devanagari script
    IT,      // Italian (i)
    PT_BR,   // Brazilian Portuguese (p)
    KO,      // Korean (k) - requires Hangul script
    VI,      // Vietnamese (v)
};

/**
 * Convert language string to enum.
 */
Language parse_language(const std::string& lang);

/**
 * Convert language enum to string.
 */
std::string language_to_string(Language lang);

/**
 * Misaki G2P configuration.
 */
struct MisakiConfig {
    std::string lexicon_path;      // Path to misaki_export/ directory
    Language language = Language::EN_US;
    bool normalize_text = true;    // Lowercase, basic normalization
    bool use_espeak_fallback = false;  // Optional espeak-ng fallback for OOV words
    // Note: Misaki is ALWAYS priority - espeak only for truly unknown words
};

/**
 * Zero-copy memory-mapped lexicon with binary search.
 *
 * Binary format v2:
 * - Header: magic(4) + version(4) + entry_count(4) + string_table_size(4) + reserved(16)
 * - Index: entry_count * (key_offset(4) + key_len(2) + val_offset(4) + val_len(2))
 * - Strings: packed key/value strings (sorted by key)
 *
 * Entries are sorted by key for O(log n) binary search lookup.
 * File remains mmap'd for the lifetime of this object - no copies made.
 */
class MmapLexicon {
public:
    MmapLexicon();
    ~MmapLexicon();

    // Disable copy
    MmapLexicon(const MmapLexicon&) = delete;
    MmapLexicon& operator=(const MmapLexicon&) = delete;

    // Enable move
    MmapLexicon(MmapLexicon&& other) noexcept;
    MmapLexicon& operator=(MmapLexicon&& other) noexcept;

    /**
     * Load binary lexicon (v2 format) from file.
     * File remains memory-mapped until destruction.
     * @return true on success
     */
    bool load(const std::string& bin_path);

    /**
     * Look up a key using binary search.
     * @return string_view into mapped memory, empty if not found
     */
    std::string_view lookup(std::string_view key) const;

    /**
     * Get number of entries.
     */
    size_t size() const { return entry_count_; }

    /**
     * Check if loaded.
     */
    bool loaded() const { return mapped_ != nullptr; }

private:
    void* mapped_ = nullptr;
    size_t file_size_ = 0;
    uint32_t entry_count_ = 0;
    const uint8_t* index_table_ = nullptr;
    const char* string_table_ = nullptr;
};

/**
 * Misaki G2P - Lexicon-based grapheme-to-phoneme converter.
 *
 * Uses exported Misaki lexicons for accurate phonemization.
 * The lexicons contain word-to-IPA mappings in JSON format:
 *   - golds: High-confidence pronunciations
 *   - silvers: Lower-confidence (derived) pronunciations
 */
class MisakiG2P {
public:
    MisakiG2P();
    ~MisakiG2P();

    // Move semantics
    MisakiG2P(MisakiG2P&&) noexcept;
    MisakiG2P& operator=(MisakiG2P&&) noexcept;

    // Disable copy
    MisakiG2P(const MisakiG2P&) = delete;
    MisakiG2P& operator=(const MisakiG2P&) = delete;

    /**
     * Initialize with configuration.
     * @param config MisakiConfig with lexicon path and language
     * @return true if successful
     */
    bool initialize(const MisakiConfig& config);

    /**
     * Initialize with default path and language.
     * @param lexicon_path Path to misaki_export/ directory
     * @param language Language code ("en-us", "en-gb", "ja", "zh")
     * @return true if successful
     */
    bool initialize(const std::string& lexicon_path,
                   const std::string& language = "en-us");

    /**
     * Check if initialized.
     */
    bool initialized() const { return initialized_; }

    /**
     * Convert text to IPA phonemes.
     * @param text Input text
     * @return IPA phoneme string
     */
    std::string phonemize(const std::string& text);

    /**
     * Convert text to phoneme token IDs.
     * Uses the phoneme vocabulary to map IPA to token IDs.
     * @param text Input text
     * @return Vector of phoneme token IDs
     */
    std::vector<int> phonemize_to_ids(const std::string& text);

    /**
     * Look up a single word in the lexicon.
     * @param word Word to look up (lowercase)
     * @return IPA string, or empty if not found
     */
    std::string lookup_word(const std::string& word) const;

    /**
     * Get number of entries in lexicon.
     */
    size_t lexicon_size() const;

    /**
     * Get current language.
     */
    Language language() const { return config_.language; }

    /**
     * Load phoneme vocabulary for tokenization.
     * @param vocab_path Path to vocab.json
     * @return true if successful
     */
    bool load_vocab(const std::string& vocab_path);

    /**
     * Convert IPA string to token IDs.
     * @param ipa IPA phoneme string
     * @return Vector of token IDs
     */
    std::vector<int> ipa_to_ids(const std::string& ipa) const;

    /**
     * Convert token IDs to IPA string.
     * @param ids Token IDs
     * @return IPA string
     */
    std::string ids_to_ipa(const std::vector<int>& ids) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    MisakiConfig config_;
    bool initialized_ = false;

    // Zero-copy mmap lexicons (v2 format, sorted, binary search)
    // These are preferred over hash maps for fast loading
    MmapLexicon mmap_manual_;
    MmapLexicon mmap_symbols_;
    MmapLexicon mmap_golds_;
    MmapLexicon mmap_silvers_;
    MmapLexicon mmap_hepburn_;
    MmapLexicon mmap_pinyin_;

    // Fallback hash maps for v1 format or JSON-only lexicons
    // Used when mmap lexicons are not available
    std::unordered_map<std::string, std::string> manual_;
    std::unordered_map<std::string, std::string> symbols_;
    std::unordered_map<std::string, std::string> golds_;
    std::unordered_map<std::string, std::string> silvers_;

    // Phoneme vocabulary (phoneme -> id, id -> phoneme)
    std::unordered_map<std::string, int> phoneme_to_id_;
    std::unordered_map<int, std::string> id_to_phoneme_;

    // Japanese-specific data (fallback)
    std::unordered_map<std::string, std::string> ja_words_;
    std::unordered_map<std::string, std::string> hepburn_;

    // MeCab tokenizer for Japanese kanji (opaque pointer)
    void* mecab_tagger_ = nullptr;

    // Chinese-specific data (fallback)
    std::unordered_map<std::string, std::string> pinyin_to_ipa_;

    // Chinese hanzi-to-pinyin (zero-copy mmap)
    MmapLexicon mmap_hanzi_;

    // Internal methods
    bool load_english_lexicons();
    bool load_japanese_lexicons();
    bool load_chinese_lexicons();
    bool load_json_lexicon(const std::string& path,
                          std::unordered_map<std::string, std::string>& lexicon);
    bool load_binary_lexicon(const std::string& path,
                            std::unordered_map<std::string, std::string>& lexicon);
    bool load_lexicon_prefer_binary(const std::string& json_path,
                                    std::unordered_map<std::string, std::string>& lexicon);

    std::string phonemize_english(const std::string& text);
    std::string phonemize_japanese(const std::string& text);
    std::string phonemize_chinese(const std::string& text);
    std::string phonemize_espeak(const std::string& text);

    std::string normalize_text(const std::string& text) const;
    std::vector<std::string> tokenize_words(const std::string& text) const;
    std::string fallback_phonemize(const std::string& word) const;
};

/**
 * Global Misaki G2P instance for convenience.
 * Thread-safe after initialization.
 */
MisakiG2P& get_global_g2p();

/**
 * Initialize global G2P.
 */
bool init_global_g2p(const std::string& lexicon_path,
                    const std::string& language = "en-us");

}  // namespace misaki
