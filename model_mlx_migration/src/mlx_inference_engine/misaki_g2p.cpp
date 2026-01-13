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

#include "misaki_g2p.h"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <iostream>
#include <regex>
#include <mutex>

// Memory-mapped file support for fast binary lexicon loading
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>

// Optional espeak-ng fallback for out-of-vocabulary words
// Misaki is ALWAYS priority - espeak only as last resort
#ifdef USE_ESPEAK_FALLBACK
#include <espeak-ng/speak_lib.h>
#endif

// Optional MeCab for Japanese kanji tokenization
#ifdef USE_MECAB
#include <mecab.h>
#endif

namespace misaki {

// ============================================================================
// MmapLexicon implementation - zero-copy binary search lexicon
// ============================================================================

MmapLexicon::MmapLexicon() = default;

MmapLexicon::~MmapLexicon() {
    if (mapped_) {
        munmap(mapped_, file_size_);
        mapped_ = nullptr;
    }
}

MmapLexicon::MmapLexicon(MmapLexicon&& other) noexcept
    : mapped_(other.mapped_)
    , file_size_(other.file_size_)
    , entry_count_(other.entry_count_)
    , index_table_(other.index_table_)
    , string_table_(other.string_table_) {
    other.mapped_ = nullptr;
    other.file_size_ = 0;
    other.entry_count_ = 0;
    other.index_table_ = nullptr;
    other.string_table_ = nullptr;
}

MmapLexicon& MmapLexicon::operator=(MmapLexicon&& other) noexcept {
    if (this != &other) {
        if (mapped_) {
            munmap(mapped_, file_size_);
        }
        mapped_ = other.mapped_;
        file_size_ = other.file_size_;
        entry_count_ = other.entry_count_;
        index_table_ = other.index_table_;
        string_table_ = other.string_table_;
        other.mapped_ = nullptr;
        other.file_size_ = 0;
        other.entry_count_ = 0;
        other.index_table_ = nullptr;
        other.string_table_ = nullptr;
    }
    return *this;
}

bool MmapLexicon::load(const std::string& bin_path) {
    // Open file
    int fd = open(bin_path.c_str(), O_RDONLY);
    if (fd < 0) {
        return false;
    }

    // Get file size
    struct stat st;
    if (fstat(fd, &st) < 0) {
        close(fd);
        return false;
    }
    file_size_ = st.st_size;

    // Memory map the file
    mapped_ = mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    if (mapped_ == MAP_FAILED) {
        mapped_ = nullptr;
        return false;
    }

    const uint8_t* data = static_cast<const uint8_t*>(mapped_);

    // Parse header (32 bytes)
    if (file_size_ < 32) {
        munmap(mapped_, file_size_);
        mapped_ = nullptr;
        return false;
    }

    // Check magic "MLX2" (v2 format)
    if (memcmp(data, "MLX2", 4) != 0) {
        // Not v2 format, unmap and return false
        munmap(mapped_, file_size_);
        mapped_ = nullptr;
        return false;
    }

    uint32_t version = *reinterpret_cast<const uint32_t*>(data + 4);
    entry_count_ = *reinterpret_cast<const uint32_t*>(data + 8);
    uint32_t string_table_size = *reinterpret_cast<const uint32_t*>(data + 12);

    if (version != 2) {
        munmap(mapped_, file_size_);
        mapped_ = nullptr;
        return false;
    }

    // Index table: 12 bytes per entry (key_off(4) + key_len(2) + val_off(4) + val_len(2))
    size_t index_table_offset = 32;
    size_t string_table_offset = 32 + entry_count_ * 12;

    if (string_table_offset + string_table_size > file_size_) {
        munmap(mapped_, file_size_);
        mapped_ = nullptr;
        return false;
    }

    index_table_ = data + index_table_offset;
    string_table_ = reinterpret_cast<const char*>(data + string_table_offset);

    return true;
}

std::string_view MmapLexicon::lookup(std::string_view key) const {
    if (!mapped_ || entry_count_ == 0) {
        return {};
    }

    // Binary search on sorted entries
    uint32_t lo = 0, hi = entry_count_;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;

        // Read index entry at mid
        const uint8_t* entry_ptr = index_table_ + mid * 12;
        uint32_t key_offset = *reinterpret_cast<const uint32_t*>(entry_ptr);
        uint16_t key_length = *reinterpret_cast<const uint16_t*>(entry_ptr + 4);
        uint32_t val_offset = *reinterpret_cast<const uint32_t*>(entry_ptr + 6);
        uint16_t val_length = *reinterpret_cast<const uint16_t*>(entry_ptr + 10);

        std::string_view mid_key(string_table_ + key_offset, key_length);

        int cmp = key.compare(mid_key);
        if (cmp < 0) {
            hi = mid;
        } else if (cmp > 0) {
            lo = mid + 1;
        } else {
            // Found!
            return std::string_view(string_table_ + val_offset, val_length);
        }
    }

    return {};  // Not found
}

// ============================================================================
// Language helpers
// ============================================================================

Language parse_language(const std::string& lang) {
    std::string lower = lang;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    // English
    if (lower == "en-us" || lower == "en_us" || lower == "en" || lower == "a") {
        return Language::EN_US;
    } else if (lower == "en-gb" || lower == "en_gb" || lower == "b") {
        return Language::EN_GB;
    }
    // Japanese
    else if (lower == "ja" || lower == "jp" || lower == "japanese" || lower == "j") {
        return Language::JA;
    }
    // Chinese
    else if (lower == "zh" || lower == "cn" || lower == "chinese" || lower == "mandarin" || lower == "z") {
        return Language::ZH;
    }
    // Spanish
    else if (lower == "es" || lower == "spanish" || lower == "e") {
        return Language::ES;
    }
    // French
    else if (lower == "fr" || lower == "french" || lower == "f") {
        return Language::FR;
    }
    // Hindi
    else if (lower == "hi" || lower == "hindi" || lower == "h") {
        return Language::HI;
    }
    // Italian
    else if (lower == "it" || lower == "italian" || lower == "i") {
        return Language::IT;
    }
    // Brazilian Portuguese
    else if (lower == "pt-br" || lower == "pt_br" || lower == "pt" || lower == "portuguese" || lower == "p") {
        return Language::PT_BR;
    }
    // Korean
    else if (lower == "ko" || lower == "korean" || lower == "k") {
        return Language::KO;
    }
    // Vietnamese
    else if (lower == "vi" || lower == "vietnamese" || lower == "v") {
        return Language::VI;
    }

    return Language::EN_US;  // Default
}

std::string language_to_string(Language lang) {
    switch (lang) {
        case Language::EN_US: return "en-us";
        case Language::EN_GB: return "en-gb";
        case Language::JA: return "ja";
        case Language::ZH: return "zh";
        case Language::ES: return "es";
        case Language::FR: return "fr";
        case Language::HI: return "hi";
        case Language::IT: return "it";
        case Language::PT_BR: return "pt-br";
        case Language::KO: return "ko";
        case Language::VI: return "vi";
        default: return "en-us";
    }
}

// Get espeak-ng voice name for a language
std::string get_espeak_voice(Language lang) {
    switch (lang) {
        case Language::EN_US: return "en-us";
        case Language::EN_GB: return "en-gb";
        case Language::ES: return "es";
        case Language::FR: return "fr";
        case Language::HI: return "hi";
        case Language::IT: return "it";
        case Language::PT_BR: return "pt-br";
        case Language::KO: return "ko";
        case Language::VI: return "vi";
        default: return "en-us";
    }
}

// Check if language uses espeak-ng (vs Misaki lexicons)
bool is_espeak_language(Language lang) {
    switch (lang) {
        case Language::ES:
        case Language::FR:
        case Language::HI:
        case Language::IT:
        case Language::PT_BR:
        case Language::KO:
        case Language::VI:
            return true;
        default:
            return false;
    }
}

// ============================================================================
// Fast JSON parser for lexicon files - optimized for large files
// Only handles {"key": "value", ...} format with optional nested objects
// ============================================================================

namespace {

// Fast hex digit to int
inline int hex_digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

// Fast inline whitespace skip using pointer arithmetic
inline void skip_ws_fast(const char*& p, const char* end) {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) {
        p++;
    }
}

// Fast JSON string parser using pointers - avoids substr allocations
inline std::string parse_json_string_fast(const char*& p, const char* end) {
    if (p >= end || *p != '"') {
        return "";
    }
    p++;  // Skip opening quote

    std::string result;
    result.reserve(32);  // Pre-allocate for typical word length

    while (p < end && *p != '"') {
        if (*p == '\\' && p + 1 < end) {
            p++;
            switch (*p) {
                case 'n': result += '\n'; break;
                case 't': result += '\t'; break;
                case 'r': result += '\r'; break;
                case '"': result += '"'; break;
                case '\\': result += '\\'; break;
                case 'u': {
                    // Unicode escape \uXXXX
                    if (p + 4 < end) {
                        int cp = 0;
                        for (int i = 1; i <= 4; i++) {
                            int d = hex_digit(p[i]);
                            if (d < 0) { cp = -1; break; }
                            cp = (cp << 4) | d;
                        }
                        if (cp >= 0) {
                            if (cp < 0x80) {
                                result += static_cast<char>(cp);
                            } else if (cp < 0x800) {
                                result += static_cast<char>(0xC0 | (cp >> 6));
                                result += static_cast<char>(0x80 | (cp & 0x3F));
                            } else {
                                result += static_cast<char>(0xE0 | (cp >> 12));
                                result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                                result += static_cast<char>(0x80 | (cp & 0x3F));
                            }
                            p += 4;
                        }
                    }
                    break;
                }
                default: result += *p; break;
            }
        } else {
            result += *p;
        }
        p++;
    }

    if (p < end) {
        p++;  // Skip closing quote
    }
    return result;
}

// Parse simple JSON object {"key": "value", ...} - optimized version
bool parse_json_dict(const std::string& json,
                    std::unordered_map<std::string, std::string>& result) {
    const char* p = json.data();
    const char* end = p + json.size();

    skip_ws_fast(p, end);

    if (p >= end || *p != '{') {
        std::cerr << "JSON parse error: Expected '{' at start\n";
        return false;
    }
    p++;  // Skip '{'

    while (p < end) {
        skip_ws_fast(p, end);

        if (*p == '}') {
            break;
        }

        // Parse key
        std::string key = parse_json_string_fast(p, end);
        if (key.empty()) {
            std::cerr << "JSON parse error: Empty key\n";
            return false;
        }

        skip_ws_fast(p, end);
        if (p >= end || *p != ':') {
            std::cerr << "JSON parse error: Expected ':' after key '" << key << "'\n";
            return false;
        }
        p++;  // Skip ':'

        skip_ws_fast(p, end);

        // Parse value - can be string OR nested object {"DEFAULT": "...", "VERB": "..."}
        if (p < end && *p == '{') {
            // Nested object - extract DEFAULT value
            p++;  // Skip '{'
            std::string default_value;

            while (p < end && *p != '}') {
                skip_ws_fast(p, end);
                if (p >= end || *p == '}') break;

                // Must have a string key (starts with '"')
                if (*p != '"') {
                    // Skip to end of object
                    while (p < end && *p != '}') p++;
                    if (p < end) p++;
                    result.emplace(std::move(key), std::move(default_value));
                    goto next_entry;
                }

                std::string inner_key = parse_json_string_fast(p, end);

                skip_ws_fast(p, end);
                if (p < end && *p == ':') {
                    p++;
                }

                skip_ws_fast(p, end);
                std::string inner_val = parse_json_string_fast(p, end);

                // Use DEFAULT value, or first value if no DEFAULT
                if (inner_key == "DEFAULT" || default_value.empty()) {
                    default_value = std::move(inner_val);
                }

                skip_ws_fast(p, end);
                if (p < end && *p == ',') {
                    p++;
                }
            }

            if (p < end && *p == '}') {
                p++;  // Skip closing '}'
            }

            result.emplace(std::move(key), std::move(default_value));
        } else {
            // Simple string value
            std::string value = parse_json_string_fast(p, end);
            result.emplace(std::move(key), std::move(value));
        }

        next_entry:
        skip_ws_fast(p, end);
        if (p < end && *p == ',') {
            p++;  // Skip ','
        }
    }

    return true;
}

// Parse JSON dict with array values {"key": ["value"]} - takes first element
bool parse_json_dict_array(const std::string& json,
                           std::unordered_map<std::string, std::string>& result) {
    const char* p = json.data();
    const char* end = p + json.size();

    skip_ws_fast(p, end);

    if (p >= end || *p != '{') {
        return false;
    }
    p++;

    while (p < end) {
        skip_ws_fast(p, end);

        if (*p == '}') {
            break;
        }

        std::string key = parse_json_string_fast(p, end);
        if (key.empty()) {
            return false;
        }

        skip_ws_fast(p, end);
        if (p >= end || *p != ':') {
            return false;
        }
        p++;

        skip_ws_fast(p, end);

        // Expect array value: ["string"]
        if (p >= end || *p != '[') {
            // Skip non-array values
            while (p < end && *p != ',' && *p != '}') p++;
            if (p < end && *p == ',') p++;
            continue;
        }
        p++;  // Skip '['

        skip_ws_fast(p, end);

        // Parse first string in array
        std::string value;
        if (p < end && *p == '"') {
            value = parse_json_string_fast(p, end);
        }

        // Skip to end of array
        while (p < end && *p != ']') p++;
        if (p < end) p++;  // Skip ']'

        if (!value.empty()) {
            result.emplace(std::move(key), std::move(value));
        }

        skip_ws_fast(p, end);
        if (p < end && *p == ',') {
            p++;
        }
    }

    return true;
}

// Parse vocab JSON (phoneme -> int) - optimized version
bool parse_vocab_json(const std::string& json,
                     std::unordered_map<std::string, int>& p2i,
                     std::unordered_map<int, std::string>& i2p) {
    const char* p = json.data();
    const char* end = p + json.size();

    skip_ws_fast(p, end);

    if (p >= end || *p != '{') {
        return false;
    }
    p++;

    while (p < end) {
        skip_ws_fast(p, end);

        if (*p == '}') {
            break;
        }

        std::string key = parse_json_string_fast(p, end);
        if (key.empty()) {
            return false;
        }

        skip_ws_fast(p, end);
        if (p >= end || *p != ':') {
            return false;
        }
        p++;

        skip_ws_fast(p, end);

        // Parse integer value directly
        int value = 0;
        bool neg = false;
        if (p < end && *p == '-') {
            neg = true;
            p++;
        }
        while (p < end && *p >= '0' && *p <= '9') {
            value = value * 10 + (*p - '0');
            p++;
        }
        if (neg) value = -value;

        p2i[key] = value;
        i2p[value] = key;

        skip_ws_fast(p, end);
        if (p < end && *p == ',') {
            p++;
        }
    }

    return true;
}

}  // anonymous namespace

// ============================================================================
// MisakiG2P implementation
// ============================================================================

struct MisakiG2P::Impl {
#ifdef USE_ESPEAK_FALLBACK
    bool espeak_initialized = false;
#endif
};

MisakiG2P::MisakiG2P() : impl_(std::make_unique<Impl>()) {}

MisakiG2P::~MisakiG2P() {
#ifdef USE_MECAB
    if (mecab_tagger_) {
        MeCab::Tagger* tagger = static_cast<MeCab::Tagger*>(mecab_tagger_);
        delete tagger;
        mecab_tagger_ = nullptr;
    }
#endif
}

MisakiG2P::MisakiG2P(MisakiG2P&&) noexcept = default;
MisakiG2P& MisakiG2P::operator=(MisakiG2P&&) noexcept = default;

bool MisakiG2P::initialize(const MisakiConfig& config) {
    config_ = config;

    // Load lexicons based on language
    bool success = false;

    if (is_espeak_language(config_.language)) {
        // espeak-ng based languages (ES, FR, HI, IT, PT-BR)
#ifdef USE_ESPEAK_FALLBACK
        // Initialize espeak-ng
        int result = espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, nullptr, 0);
        if (result == EE_INTERNAL_ERROR) {
            std::cerr << "MisakiG2P: Failed to initialize espeak-ng" << std::endl;
            return false;
        }

        // Set voice for this language
        std::string voice = get_espeak_voice(config_.language);
        espeak_SetVoiceByName(voice.c_str());
        impl_->espeak_initialized = true;
        success = true;

        std::cout << "MisakiG2P: Initialized espeak-ng for " << voice << std::endl;
#else
        std::cerr << "MisakiG2P: espeak-ng not compiled in. Rebuild with USE_ESPEAK_FALLBACK=1" << std::endl;
        std::cerr << "  Languages ES, FR, HI, IT, PT-BR require espeak-ng" << std::endl;
        return false;
#endif
    } else {
        // Misaki-based languages (EN, JA, ZH)
        switch (config_.language) {
            case Language::EN_US:
            case Language::EN_GB:
                success = load_english_lexicons();
                break;
            case Language::JA:
                success = load_japanese_lexicons();
                break;
            case Language::ZH:
                success = load_chinese_lexicons();
                break;
            default:
                success = load_english_lexicons();
                break;
        }

        if (!success) {
            std::cerr << "MisakiG2P: Failed to load lexicons for "
                      << language_to_string(config_.language) << std::endl;
            return false;
        }
    }

    initialized_ = true;
    return true;
}

bool MisakiG2P::initialize(const std::string& lexicon_path,
                          const std::string& language) {
    MisakiConfig config;
    config.lexicon_path = lexicon_path;
    config.language = parse_language(language);
    return initialize(config);
}

bool MisakiG2P::load_json_lexicon(const std::string& path,
                                  std::unordered_map<std::string, std::string>& lexicon) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "MisakiG2P: Cannot open " << path << std::endl;
        return false;
    }

    // Pre-allocate for large lexicons
    lexicon.reserve(200000);

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string json = buffer.str();

    std::cout << "MisakiG2P: Parsing " << path << " (" << json.size() / 1024 << " KB)..." << std::flush;

    if (!parse_json_dict(json, lexicon)) {
        std::cout << " FAILED\n";
        std::cerr << "MisakiG2P: Failed to parse " << path << std::endl;
        return false;
    }

    std::cout << " done (" << lexicon.size() << " entries)\n";
    return true;
}

// Binary lexicon format:
// Header (32 bytes): magic(4) + version(4) + entry_count(4) + string_table_size(4) + reserved(16)
// Index table: entry_count * (key_offset(4) + value_offset(4))
// String table: null-terminated strings
bool MisakiG2P::load_binary_lexicon(const std::string& path,
                                    std::unordered_map<std::string, std::string>& lexicon) {
    // Open file
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        return false;  // File doesn't exist, fall back to JSON
    }

    // Get file size
    struct stat st;
    if (fstat(fd, &st) < 0) {
        close(fd);
        return false;
    }
    size_t file_size = st.st_size;

    // Memory map the file
    void* mapped = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);  // Can close fd after mmap

    if (mapped == MAP_FAILED) {
        std::cerr << "MisakiG2P: Failed to mmap " << path << std::endl;
        return false;
    }

    const uint8_t* data = static_cast<const uint8_t*>(mapped);

    // Parse header (32 bytes)
    if (file_size < 32) {
        munmap(mapped, file_size);
        return false;
    }

    // Check magic "MLEX"
    if (memcmp(data, "MLEX", 4) != 0) {
        munmap(mapped, file_size);
        return false;
    }

    uint32_t version = *reinterpret_cast<const uint32_t*>(data + 4);
    uint32_t entry_count = *reinterpret_cast<const uint32_t*>(data + 8);
    uint32_t string_table_size = *reinterpret_cast<const uint32_t*>(data + 12);

    if (version != 1) {
        std::cerr << "MisakiG2P: Unsupported binary lexicon version " << version << std::endl;
        munmap(mapped, file_size);
        return false;
    }

    // Calculate offsets
    size_t index_table_offset = 32;
    size_t string_table_offset = 32 + entry_count * 8;

    if (string_table_offset + string_table_size > file_size) {
        std::cerr << "MisakiG2P: Invalid binary lexicon size" << std::endl;
        munmap(mapped, file_size);
        return false;
    }

    const char* string_table = reinterpret_cast<const char*>(data + string_table_offset);

    // Pre-allocate
    lexicon.reserve(entry_count);

    std::cout << "MisakiG2P: Loading binary " << path << " (" << entry_count << " entries)..." << std::flush;

    // Load entries
    const uint32_t* index = reinterpret_cast<const uint32_t*>(data + index_table_offset);
    for (uint32_t i = 0; i < entry_count; i++) {
        uint32_t key_offset = index[i * 2];
        uint32_t value_offset = index[i * 2 + 1];

        if (key_offset >= string_table_size || value_offset >= string_table_size) {
            continue;  // Skip invalid entries
        }

        const char* key = string_table + key_offset;
        const char* value = string_table + value_offset;

        lexicon[key] = value;
    }

    std::cout << " done\n";

    munmap(mapped, file_size);
    return true;
}

// Try binary first, fall back to JSON
bool MisakiG2P::load_lexicon_prefer_binary(const std::string& json_path,
                                            std::unordered_map<std::string, std::string>& lexicon) {
    // Try binary version first (replace .json with .bin)
    std::string bin_path = json_path;
    size_t pos = bin_path.rfind(".json");
    if (pos != std::string::npos) {
        bin_path = bin_path.substr(0, pos) + ".bin";
        if (load_binary_lexicon(bin_path, lexicon)) {
            return true;
        }
    }

    // Fall back to JSON
    return load_json_lexicon(json_path, lexicon);
}

bool MisakiG2P::load_english_lexicons() {
    std::string prefix = config_.lexicon_path + "/en/";
    std::string dialect = (config_.language == Language::EN_GB) ? "gb" : "us";

    // Try zero-copy mmap lexicons first (v2 format)
    std::string manual_bin = prefix + "add_symbols.bin";
    std::string symbols_bin = prefix + "symbols.bin";
    std::string golds_bin = prefix + dialect + "_golds.bin";
    std::string silvers_bin = prefix + dialect + "_silvers.bin";

    bool use_mmap = false;
    if (mmap_golds_.load(golds_bin)) {
        use_mmap = true;
        std::cout << "MisakiG2P: Loaded mmap " << golds_bin << " (" << mmap_golds_.size() << " entries)" << std::endl;

        mmap_manual_.load(manual_bin);
        mmap_symbols_.load(symbols_bin);
        mmap_silvers_.load(silvers_bin);

        std::cout << "MisakiG2P: Zero-copy mmap: "
                  << mmap_manual_.size() << " manual + "
                  << mmap_symbols_.size() << " symbols + "
                  << mmap_golds_.size() << " gold + "
                  << mmap_silvers_.size() << " silver entries for " << dialect << std::endl;
        return true;
    }

    // Fall back to hash map loading (v1 format or JSON)
    std::cout << "MisakiG2P: v2 binary not found, falling back to hash map loading" << std::endl;

    // Load manual overrides (add_symbols.json) - HIGHEST PRIORITY
    std::string manual_path = prefix + "add_symbols.json";
    if (!load_lexicon_prefer_binary(manual_path, manual_)) {
        // Manual overrides are optional
        std::cerr << "MisakiG2P: Note - add_symbols.json not found" << std::endl;
    }

    // Load symbols (symbols.json) - HIGH PRIORITY
    std::string symbols_path = prefix + "symbols.json";
    if (!load_lexicon_prefer_binary(symbols_path, symbols_)) {
        // Symbols are optional
        std::cerr << "MisakiG2P: Note - symbols.json not found" << std::endl;
    }

    // Load gold lexicon (high confidence) - try binary first
    std::string golds_path = prefix + dialect + "_golds.json";
    if (!load_lexicon_prefer_binary(golds_path, golds_)) {
        return false;
    }

    // Load silver lexicon (derived pronunciations) - try binary first
    std::string silvers_path = prefix + dialect + "_silvers.json";
    if (!load_lexicon_prefer_binary(silvers_path, silvers_)) {
        // Silvers are optional
        std::cerr << "MisakiG2P: Warning - silvers not loaded" << std::endl;
    }

    std::cout << "MisakiG2P: Loaded "
              << manual_.size() << " manual + "
              << symbols_.size() << " symbols + "
              << golds_.size() << " gold + "
              << silvers_.size() << " silver entries for " << dialect << std::endl;

    return true;
}

bool MisakiG2P::load_japanese_lexicons() {
    std::string prefix = config_.lexicon_path + "/ja/";

    // Try zero-copy mmap first (v2 format)
    std::string hepburn_bin = prefix + "hepburn.bin";
    if (mmap_hepburn_.load(hepburn_bin)) {
        std::cout << "MisakiG2P: Loaded mmap " << hepburn_bin << " (" << mmap_hepburn_.size() << " entries)" << std::endl;
    } else {
        // Fall back to hash map loading
        std::cout << "MisakiG2P: v2 binary not found for Japanese, falling back to hash map" << std::endl;

        // Load hepburn character-to-IPA mappings (required) - try v1 binary first
        if (!load_lexicon_prefer_binary(prefix + "hepburn.json", hepburn_)) {
            std::cerr << "MisakiG2P: Failed to load hepburn.json (required for Japanese)" << std::endl;
            return false;
        }
        std::cout << "MisakiG2P: Loaded " << hepburn_.size()
                  << " Japanese hepburn entries" << std::endl;
    }

    // Initialize MeCab for kanji tokenization
#ifdef USE_MECAB
    mecab_tagger_ = mecab_new2("");
    if (mecab_tagger_) {
        std::cout << "MisakiG2P: MeCab initialized for Japanese kanji" << std::endl;
    } else {
        std::cerr << "MisakiG2P: Warning - MeCab initialization failed, kanji won't be converted" << std::endl;
    }
#else
    std::cout << "MisakiG2P: Note - MeCab not compiled in, kanji won't be converted" << std::endl;
#endif

    return true;
}

bool MisakiG2P::load_chinese_lexicons() {
    std::string prefix = config_.lexicon_path + "/zh/";

    // Load hanzi-to-pinyin dictionary (for Chinese character input)
    std::string hanzi_bin = prefix + "hanzi_to_pinyin.bin";
    if (mmap_hanzi_.load(hanzi_bin)) {
        std::cout << "MisakiG2P: Loaded mmap " << hanzi_bin << " (" << mmap_hanzi_.size() << " hanzi entries)" << std::endl;
    } else {
        std::cerr << "MisakiG2P: Note - hanzi_to_pinyin.bin not found, Chinese characters won't be converted" << std::endl;
    }

    // Try zero-copy mmap first for pinyin-to-IPA (v2 format)
    std::string pinyin_bin = prefix + "pinyin_to_ipa.bin";
    if (mmap_pinyin_.load(pinyin_bin)) {
        std::cout << "MisakiG2P: Loaded mmap " << pinyin_bin << " (" << mmap_pinyin_.size() << " pinyin entries)" << std::endl;
        return true;
    }

    // Fall back: Try v1 binary format (flattened array values)
    std::cout << "MisakiG2P: v2 binary not found for Chinese pinyin, trying v1 binary" << std::endl;
    if (load_binary_lexicon(pinyin_bin, pinyin_to_ipa_)) {
        std::cout << "MisakiG2P: Loaded " << pinyin_to_ipa_.size()
                  << " Chinese pinyin entries" << std::endl;
        return true;
    }

    // Fall back to JSON (special format: {"pinyin": ["ipa"]})
    std::ifstream file(prefix + "pinyin_to_ipa.json");
    if (!file.is_open()) {
        std::cerr << "MisakiG2P: Cannot open pinyin_to_ipa.json" << std::endl;
        return false;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string json = buffer.str();

    if (!parse_json_dict_array(json, pinyin_to_ipa_)) {
        std::cerr << "MisakiG2P: Failed to parse pinyin_to_ipa.json" << std::endl;
        return false;
    }

    std::cout << "MisakiG2P: Loaded " << pinyin_to_ipa_.size()
              << " Chinese pinyin entries" << std::endl;

    return true;
}

bool MisakiG2P::load_vocab(const std::string& vocab_path) {
    std::ifstream file(vocab_path);
    if (!file.is_open()) {
        std::cerr << "MisakiG2P: Cannot open vocab " << vocab_path << std::endl;
        return false;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string json = buffer.str();

    if (!parse_vocab_json(json, phoneme_to_id_, id_to_phoneme_)) {
        std::cerr << "MisakiG2P: Failed to parse vocab" << std::endl;
        return false;
    }

    std::cout << "MisakiG2P: Loaded vocab with " << phoneme_to_id_.size()
              << " phonemes" << std::endl;

    return true;
}

std::string MisakiG2P::normalize_text(const std::string& text) const {
    std::string result;
    result.reserve(text.size());

    for (unsigned char c : text) {
        if (std::isupper(c)) {
            result += static_cast<char>(std::tolower(c));
        } else {
            result += static_cast<char>(c);
        }
    }

    return result;
}

std::vector<std::string> MisakiG2P::tokenize_words(const std::string& text) const {
    std::vector<std::string> words;
    std::string current;

    for (char c : text) {
        if (std::isalpha(static_cast<unsigned char>(c)) || c == '\'' || c == '-') {
            current += c;
        } else {
            if (!current.empty()) {
                words.push_back(current);
                current.clear();
            }
            // Keep punctuation and symbols as separate tokens
            if (c == '.' || c == ',' || c == '!' || c == '?' ||
                c == ';' || c == ':' || c == '"' || c == '(' || c == ')' ||
                c == '%' || c == '&' || c == '+' || c == '@' || c == '/' ||
                c == '#' || c == '$' || c == '*') {
                words.push_back(std::string(1, c));
            } else if (c == ' ') {
                words.push_back(" ");
            }
        }
    }

    if (!current.empty()) {
        words.push_back(current);
    }

    return words;
}

std::string MisakiG2P::lookup_word(const std::string& word) const {
    // Try mmap lexicons first (zero-copy, O(log n) binary search)
    if (mmap_golds_.loaded()) {
        // 1. Manual overrides (add_symbols.bin)
        auto result = mmap_manual_.lookup(word);
        if (!result.empty()) {
            return std::string(result);
        }

        // 2. Symbols (symbols.bin)
        result = mmap_symbols_.lookup(word);
        if (!result.empty()) {
            return std::string(result);
        }

        // 3. Gold lexicon (high confidence)
        result = mmap_golds_.lookup(word);
        if (!result.empty()) {
            return std::string(result);
        }

        // 4. Silver lexicon (derived)
        result = mmap_silvers_.lookup(word);
        if (!result.empty()) {
            return std::string(result);
        }

        return "";
    }

    // Fall back to hash maps (for v1 format or JSON-only)
    // Priority order:
    // 1. Manual overrides (add_symbols.json)
    auto it = manual_.find(word);
    if (it != manual_.end()) {
        return it->second;
    }

    // 2. Symbols (symbols.json)
    it = symbols_.find(word);
    if (it != symbols_.end()) {
        return it->second;
    }

    // 3. Gold lexicon (high confidence)
    it = golds_.find(word);
    if (it != golds_.end()) {
        return it->second;
    }

    // 4. Silver lexicon (derived)
    it = silvers_.find(word);
    if (it != silvers_.end()) {
        return it->second;
    }

    return "";
}

std::string MisakiG2P::fallback_phonemize(const std::string& word) const {
    // NO espeak fallback - Misaki lexicons should be comprehensive
    // Unknown words are returned as-is (will be skipped during tokenization)
    // This is intentional: phoneme head was trained on Misaki, not espeak
    return "";
}

std::string MisakiG2P::phonemize_english(const std::string& text) {
    std::string normalized = config_.normalize_text ? normalize_text(text) : text;
    std::vector<std::string> words = tokenize_words(normalized);

    std::string result;
    for (size_t i = 0; i < words.size(); i++) {
        const std::string& word = words[i];

        // Handle spaces
        if (word == " ") {
            if (!result.empty() && result.back() != ' ') {
                result += " ";
            }
            continue;
        }

        // Handle single-char symbols - try to look up in symbols dictionary
        if (word.size() == 1 && !std::isalpha(static_cast<unsigned char>(word[0]))) {
            std::string sym_phoneme = lookup_word(word);
            if (!sym_phoneme.empty()) {
                if (!result.empty() && result.back() != ' ') {
                    result += " ";
                }
                result += sym_phoneme;
            } else {
                result += word;  // Keep as-is if no phoneme found
            }
            continue;
        }

        // Look up word
        std::string phonemes = lookup_word(word);
        if (phonemes.empty() && config_.use_espeak_fallback) {
            phonemes = fallback_phonemize(word);
        }

        if (!phonemes.empty()) {
            if (!result.empty() && result.back() != ' ' &&
                std::isalpha(static_cast<unsigned char>(result.back()))) {
                result += " ";
            }
            result += phonemes;
        }
    }

    return result;
}

// Helper to check if a Unicode codepoint is a CJK character (kanji)
namespace {
bool is_kanji(uint32_t cp) {
    // CJK Unified Ideographs: U+4E00-U+9FFF
    // CJK Extension A: U+3400-U+4DBF
    return (cp >= 0x4E00 && cp <= 0x9FFF) ||
           (cp >= 0x3400 && cp <= 0x4DBF);
}

// Check if string contains any kanji characters
bool contains_kanji(const std::string& text) {
    size_t i = 0;
    while (i < text.size()) {
        unsigned char c = static_cast<unsigned char>(text[i]);

        // Check for UTF-8 3-byte sequence (where CJK lives)
        if ((c & 0xF0) == 0xE0 && i + 2 < text.size()) {
            unsigned char c2 = static_cast<unsigned char>(text[i + 1]);
            unsigned char c3 = static_cast<unsigned char>(text[i + 2]);

            // Decode UTF-8 codepoint
            uint32_t cp = ((c & 0x0F) << 12) | ((c2 & 0x3F) << 6) | (c3 & 0x3F);

            if (is_kanji(cp)) {
                return true;
            }
            i += 3;
        } else {
            // Skip other characters
            size_t len = 1;
            if ((c & 0xE0) == 0xC0) len = 2;
            else if ((c & 0xF0) == 0xE0) len = 3;
            else if ((c & 0xF8) == 0xF0) len = 4;
            i += len;
        }
    }
    return false;
}
}  // anonymous namespace

// Helper to convert katakana to hiragana (both are valid Japanese scripts)
namespace {
std::string katakana_to_hiragana(const std::string& text) {
    std::string result;
    result.reserve(text.size());

    size_t i = 0;
    while (i < text.size()) {
        unsigned char c = static_cast<unsigned char>(text[i]);

        // Check for UTF-8 3-byte sequence (Japanese characters)
        if ((c & 0xF0) == 0xE0 && i + 2 < text.size()) {
            unsigned char c2 = static_cast<unsigned char>(text[i + 1]);
            unsigned char c3 = static_cast<unsigned char>(text[i + 2]);

            // Decode UTF-8 codepoint
            uint32_t cp = ((c & 0x0F) << 12) | ((c2 & 0x3F) << 6) | (c3 & 0x3F);

            // Katakana range: U+30A0 to U+30FF -> Hiragana: U+3040 to U+309F
            if (cp >= 0x30A1 && cp <= 0x30F6) {
                // Convert to hiragana (subtract 0x60)
                cp -= 0x60;
                // Re-encode as UTF-8
                result += static_cast<char>(0xE0 | (cp >> 12));
                result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                result += static_cast<char>(0x80 | (cp & 0x3F));
                i += 3;
                continue;
            }
        }

        // Copy as-is
        size_t len = 1;
        if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;

        for (size_t j = 0; j < len && i + j < text.size(); j++) {
            result += text[i + j];
        }
        i += len;
    }

    return result;
}
}  // anonymous namespace

std::string MisakiG2P::phonemize_japanese(const std::string& text) {
    std::string input_text = text;

    // If text contains kanji, use MeCab to get readings
#ifdef USE_MECAB
    if (mecab_tagger_ && contains_kanji(text)) {
        MeCab::Tagger* tagger = static_cast<MeCab::Tagger*>(mecab_tagger_);

        // MeCab output format: surface\tPOS,POS,POS,POS,conj,conj,base,reading,pronunciation
        // We want the reading (field 7, 0-indexed) which is in katakana
        const char* result = tagger->parse(text.c_str());
        if (result) {
            std::string reading_text;
            reading_text.reserve(text.size() * 2);

            std::istringstream stream(result);
            std::string line;
            while (std::getline(stream, line)) {
                // Skip EOS marker
                if (line == "EOS" || line.empty()) {
                    continue;
                }

                // Parse: surface\tfeatures
                size_t tab_pos = line.find('\t');
                if (tab_pos == std::string::npos) {
                    continue;
                }

                std::string surface = line.substr(0, tab_pos);
                std::string features = line.substr(tab_pos + 1);

                // Split features by comma
                std::vector<std::string> fields;
                std::istringstream feat_stream(features);
                std::string field;
                while (std::getline(feat_stream, field, ',')) {
                    fields.push_back(field);
                }

                // Field 7 is the reading (0-indexed), field 8 is pronunciation
                // Some entries don't have readings (e.g., punctuation)
                std::string reading;
                if (fields.size() > 7 && fields[7] != "*") {
                    reading = fields[7];
                } else if (fields.size() > 8 && fields[8] != "*") {
                    reading = fields[8];
                } else {
                    // No reading available, use surface form
                    reading = surface;
                }

                reading_text += reading;
            }

            if (!reading_text.empty()) {
                input_text = reading_text;
            }
        }
    }
#endif

    // Convert katakana to hiragana for unified lookup
    std::string normalized = katakana_to_hiragana(input_text);

    std::string result;
    result.reserve(normalized.size() * 2);  // IPA often longer than input

    // Helper lambda for hepburn lookup (mmap first, then hash map)
    auto lookup_hepburn = [this](std::string_view key) -> std::string_view {
        if (mmap_hepburn_.loaded()) {
            return mmap_hepburn_.lookup(key);
        }
        auto it = hepburn_.find(std::string(key));
        if (it != hepburn_.end()) {
            return std::string_view(it->second);
        }
        return {};
    };

    size_t i = 0;
    while (i < normalized.size()) {
        unsigned char c = static_cast<unsigned char>(normalized[i]);

        // Determine UTF-8 character length
        size_t char_len = 1;
        if ((c & 0xE0) == 0xC0) char_len = 2;
        else if ((c & 0xF0) == 0xE0) char_len = 3;
        else if ((c & 0xF8) == 0xF0) char_len = 4;

        // Greedy matching: try longest match first (up to 3 characters for Japanese)
        bool found = false;

        // Try 3-character match (e.g., multi-character combinations are rare but possible)
        if (i + char_len * 3 <= normalized.size()) {
            // Get 3 UTF-8 characters
            size_t len1 = char_len;
            size_t pos2 = i + len1;
            if (pos2 < normalized.size()) {
                unsigned char c2 = static_cast<unsigned char>(normalized[pos2]);
                size_t len2 = 1;
                if ((c2 & 0xE0) == 0xC0) len2 = 2;
                else if ((c2 & 0xF0) == 0xE0) len2 = 3;
                else if ((c2 & 0xF8) == 0xF0) len2 = 4;

                size_t pos3 = pos2 + len2;
                if (pos3 < normalized.size()) {
                    unsigned char c3 = static_cast<unsigned char>(normalized[pos3]);
                    size_t len3 = 1;
                    if ((c3 & 0xE0) == 0xC0) len3 = 2;
                    else if ((c3 & 0xF0) == 0xE0) len3 = 3;
                    else if ((c3 & 0xF8) == 0xF0) len3 = 4;

                    if (pos3 + len3 <= normalized.size()) {
                        std::string_view three_chars(&normalized[i], len1 + len2 + len3);
                        auto ipa = lookup_hepburn(three_chars);
                        if (!ipa.empty()) {
                            result += ipa;
                            i += len1 + len2 + len3;
                            found = true;
                        }
                    }
                }
            }
        }

        // Try 2-character match (common for combinations like きゃ, しょ, etc.)
        if (!found && i + char_len < normalized.size()) {
            size_t pos2 = i + char_len;
            unsigned char c2 = static_cast<unsigned char>(normalized[pos2]);
            size_t len2 = 1;
            if ((c2 & 0xE0) == 0xC0) len2 = 2;
            else if ((c2 & 0xF0) == 0xE0) len2 = 3;
            else if ((c2 & 0xF8) == 0xF0) len2 = 4;

            if (pos2 + len2 <= normalized.size()) {
                std::string_view two_chars(&normalized[i], char_len + len2);
                auto ipa = lookup_hepburn(two_chars);
                if (!ipa.empty()) {
                    result += ipa;
                    i += char_len + len2;
                    found = true;
                }
            }
        }

        // Try single character match
        if (!found && i + char_len <= normalized.size()) {
            std::string_view one_char(&normalized[i], char_len);
            auto ipa = lookup_hepburn(one_char);
            if (!ipa.empty()) {
                result += ipa;
                i += char_len;
                found = true;
            }
        }

        // If no match found, pass through (handles ASCII, punctuation, etc.)
        if (!found) {
            result += normalized[i];
            i++;
        }
    }

    return result;
}

std::string MisakiG2P::phonemize_chinese(const std::string& text) {
    // Chinese input can be:
    // 1. Hanzi characters (e.g., "你好") -> converted to pinyin -> then IPA
    // 2. Pinyin with tone numbers (e.g., "ni3hao3") -> directly to IPA
    //
    // Pipeline: hanzi -> pinyin (with tone) -> IPA

    std::string normalized = config_.normalize_text ? normalize_text(text) : text;

    // Helper lambda for pinyin-to-IPA lookup (mmap first, then hash map)
    auto lookup_pinyin_ipa = [this](const std::string& key) -> std::string_view {
        if (mmap_pinyin_.loaded()) {
            return mmap_pinyin_.lookup(key);
        }
        auto it = pinyin_to_ipa_.find(key);
        if (it != pinyin_to_ipa_.end()) {
            return std::string_view(it->second);
        }
        return {};
    };

    // Helper to check if a codepoint is a CJK character
    auto is_cjk = [](uint32_t cp) -> bool {
        // CJK Unified Ideographs: U+4E00-U+9FFF
        // CJK Extension A: U+3400-U+4DBF
        return (cp >= 0x4E00 && cp <= 0x9FFF) ||
               (cp >= 0x3400 && cp <= 0x4DBF);
    };

    // First pass: convert hanzi to pinyin
    std::string pinyin_text;
    pinyin_text.reserve(normalized.size() * 2);

    size_t i = 0;
    while (i < normalized.size()) {
        unsigned char c = static_cast<unsigned char>(normalized[i]);

        // Check for UTF-8 3-byte sequence (where CJK lives)
        if ((c & 0xF0) == 0xE0 && i + 2 < normalized.size()) {
            unsigned char c2 = static_cast<unsigned char>(normalized[i + 1]);
            unsigned char c3 = static_cast<unsigned char>(normalized[i + 2]);

            // Decode UTF-8 codepoint
            uint32_t cp = ((c & 0x0F) << 12) | ((c2 & 0x3F) << 6) | (c3 & 0x3F);

            if (is_cjk(cp) && mmap_hanzi_.loaded()) {
                // Look up hanzi -> pinyin
                std::string_view hanzi(&normalized[i], 3);
                auto pinyin = mmap_hanzi_.lookup(hanzi);
                if (!pinyin.empty()) {
                    pinyin_text += pinyin;
                    i += 3;
                    continue;
                }
            }
            // Pass through non-CJK or unknown CJK
            pinyin_text += normalized.substr(i, 3);
            i += 3;
        } else if ((c & 0xF8) == 0xF0 && i + 3 < normalized.size()) {
            // 4-byte UTF-8 (CJK Extension B, etc.)
            pinyin_text += normalized.substr(i, 4);
            i += 4;
        } else if ((c & 0xE0) == 0xC0 && i + 1 < normalized.size()) {
            // 2-byte UTF-8
            pinyin_text += normalized.substr(i, 2);
            i += 2;
        } else {
            pinyin_text += c;
            i++;
        }
    }

    // Second pass: convert pinyin to IPA
    std::string result;
    result.reserve(pinyin_text.size() * 2);

    i = 0;
    while (i < pinyin_text.size()) {
        // Skip whitespace
        if (std::isspace(static_cast<unsigned char>(pinyin_text[i]))) {
            if (!result.empty() && result.back() != ' ') {
                result += ' ';
            }
            i++;
            continue;
        }

        // Handle punctuation
        if (std::ispunct(static_cast<unsigned char>(pinyin_text[i]))) {
            result += pinyin_text[i];
            i++;
            continue;
        }

        // Try to match pinyin syllable (greedy: longest match first)
        // Pinyin syllables can be 1-6 letters + 1 tone digit
        bool found = false;

        // Try matches from longest to shortest
        for (int len = 7; len >= 2; len--) {
            if (i + static_cast<size_t>(len) > pinyin_text.size()) continue;

            std::string syllable = pinyin_text.substr(i, len);

            // Check if last char is a tone digit (1-5)
            char last = syllable.back();
            if (last < '1' || last > '5') continue;

            auto ipa = lookup_pinyin_ipa(syllable);
            if (!ipa.empty()) {
                result += ipa;
                i += len;
                found = true;
                break;
            }
        }

        // If no match, try without tone (use neutral tone 5)
        if (!found) {
            for (int len = 6; len >= 1; len--) {
                if (i + static_cast<size_t>(len) > pinyin_text.size()) continue;

                std::string syllable = pinyin_text.substr(i, len);

                // Skip if contains digit (already handled above)
                bool has_digit = false;
                for (char c : syllable) {
                    if (std::isdigit(static_cast<unsigned char>(c))) {
                        has_digit = true;
                        break;
                    }
                }
                if (has_digit) continue;

                // Try with neutral tone (5)
                auto ipa = lookup_pinyin_ipa(syllable + "5");
                if (!ipa.empty()) {
                    result += ipa;
                    i += len;
                    found = true;
                    break;
                }
            }
        }

        // Pass through unrecognized characters
        if (!found) {
            result += pinyin_text[i];
            i++;
        }
    }

    return result;
}

// Script detection helpers
namespace {
// Check if text contains Devanagari characters (U+0900-U+097F)
bool has_devanagari(const std::string& text) {
    size_t i = 0;
    while (i < text.size()) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        // UTF-8 3-byte sequence starting with 0xE0
        if ((c & 0xF0) == 0xE0 && i + 2 < text.size()) {
            unsigned char c2 = static_cast<unsigned char>(text[i + 1]);
            unsigned char c3 = static_cast<unsigned char>(text[i + 2]);
            uint32_t cp = ((c & 0x0F) << 12) | ((c2 & 0x3F) << 6) | (c3 & 0x3F);
            // Devanagari range: U+0900 to U+097F
            if (cp >= 0x0900 && cp <= 0x097F) {
                return true;
            }
            i += 3;
        } else if ((c & 0xE0) == 0xC0) {
            i += 2;  // 2-byte UTF-8
        } else if ((c & 0x80) == 0) {
            i += 1;  // ASCII
        } else {
            i += 1;  // Invalid, skip
        }
    }
    return false;
}

// Check if text contains Hangul characters (U+AC00-U+D7A3 syllables, U+1100-U+11FF jamo)
bool has_hangul(const std::string& text) {
    size_t i = 0;
    while (i < text.size()) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        // UTF-8 3-byte sequence
        if ((c & 0xF0) == 0xE0 && i + 2 < text.size()) {
            unsigned char c2 = static_cast<unsigned char>(text[i + 1]);
            unsigned char c3 = static_cast<unsigned char>(text[i + 2]);
            uint32_t cp = ((c & 0x0F) << 12) | ((c2 & 0x3F) << 6) | (c3 & 0x3F);
            // Hangul syllables: U+AC00 to U+D7A3
            // Hangul jamo: U+1100 to U+11FF
            // Hangul compatibility jamo: U+3130 to U+318F
            if ((cp >= 0xAC00 && cp <= 0xD7A3) ||
                (cp >= 0x1100 && cp <= 0x11FF) ||
                (cp >= 0x3130 && cp <= 0x318F)) {
                return true;
            }
            i += 3;
        } else if ((c & 0xE0) == 0xC0) {
            i += 2;
        } else if ((c & 0x80) == 0) {
            i += 1;
        } else {
            i += 1;
        }
    }
    return false;
}

// Check if text is mostly ASCII/Latin (indicates romanized input)
bool is_mostly_latin(const std::string& text) {
    int latin_count = 0;
    int total_alpha = 0;
    for (unsigned char c : text) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
            latin_count++;
            total_alpha++;
        } else if ((c & 0x80) != 0) {
            total_alpha++;  // Non-ASCII character
        }
    }
    return total_alpha > 0 && (latin_count * 100 / total_alpha) > 80;
}
}  // namespace

// espeak-ng phonemization (for ES, FR, HI, IT, PT-BR)
std::string MisakiG2P::phonemize_espeak(const std::string& text) {
#ifdef USE_ESPEAK_FALLBACK
    if (!impl_->espeak_initialized) {
        std::cerr << "MisakiG2P: espeak-ng not initialized" << std::endl;
        return "";
    }

    // Warn if Hindi text is not in Devanagari script
    if (config_.language == Language::HI && !has_devanagari(text) && is_mostly_latin(text)) {
        std::cerr << "MisakiG2P: Warning - Hindi requires Devanagari script (e.g., नमस्ते), "
                  << "not romanized text (e.g., namaste). Output may be incorrect." << std::endl;
    }

    // Warn if Korean text is not in Hangul script
    if (config_.language == Language::KO && !has_hangul(text) && is_mostly_latin(text)) {
        std::cerr << "MisakiG2P: Warning - Korean requires Hangul script (e.g., 안녕하세요), "
                  << "not romanized text (e.g., annyeonghaseyo). Output may be incorrect." << std::endl;
    }

    // Set voice for current language
    std::string voice = get_espeak_voice(config_.language);
    espeak_SetVoiceByName(voice.c_str());

    // Convert to phonemes using espeak-ng
    const void* input = text.c_str();
    const char* phonemes = espeak_TextToPhonemes(&input, espeakCHARS_UTF8, espeakPHONEMES_IPA);

    if (phonemes) {
        return std::string(phonemes);
    }
    return "";
#else
    (void)text;  // Suppress unused warning
    return "";
#endif
}

std::string MisakiG2P::phonemize(const std::string& text) {
    if (!initialized_) {
        std::cerr << "MisakiG2P: Not initialized" << std::endl;
        return "";
    }

    switch (config_.language) {
        // Misaki-based languages
        case Language::EN_US:
        case Language::EN_GB:
            return phonemize_english(text);
        case Language::JA:
            return phonemize_japanese(text);
        case Language::ZH:
            return phonemize_chinese(text);

        // espeak-ng based languages
        case Language::ES:
        case Language::FR:
        case Language::HI:
        case Language::IT:
        case Language::PT_BR:
        case Language::KO:
        case Language::VI:
            return phonemize_espeak(text);

        default:
            return phonemize_english(text);
    }
}

std::vector<int> MisakiG2P::ipa_to_ids(const std::string& ipa) const {
    std::vector<int> ids;
    ids.push_back(0);  // BOS token

    // UTF-8 aware iteration
    size_t i = 0;
    while (i < ipa.size()) {
        // Determine character length
        unsigned char c = static_cast<unsigned char>(ipa[i]);
        size_t len = 1;
        if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;

        if (i + len > ipa.size()) break;

        std::string phoneme = ipa.substr(i, len);
        auto it = phoneme_to_id_.find(phoneme);
        if (it != phoneme_to_id_.end()) {
            ids.push_back(it->second);
        }
        // Skip unknown phonemes (could also map to UNK)

        i += len;
    }

    ids.push_back(0);  // EOS token
    return ids;
}

std::string MisakiG2P::ids_to_ipa(const std::vector<int>& ids) const {
    std::string result;
    for (int id : ids) {
        if (id == 0) continue;  // Skip BOS/EOS

        auto it = id_to_phoneme_.find(id);
        if (it != id_to_phoneme_.end()) {
            result += it->second;
        }
    }
    return result;
}

std::vector<int> MisakiG2P::phonemize_to_ids(const std::string& text) {
    std::string ipa = phonemize(text);
    return ipa_to_ids(ipa);
}

size_t MisakiG2P::lexicon_size() const {
    // Return mmap sizes if using zero-copy lexicons
    if (mmap_golds_.loaded()) {
        return mmap_manual_.size() + mmap_symbols_.size() + mmap_golds_.size() + mmap_silvers_.size() +
               mmap_hepburn_.size() + mmap_pinyin_.size();
    }
    // Otherwise return hash map sizes
    return manual_.size() + symbols_.size() + golds_.size() + silvers_.size() +
           ja_words_.size() + pinyin_to_ipa_.size() + hepburn_.size();
}

// ============================================================================
// Global G2P instance
// ============================================================================

static std::unique_ptr<MisakiG2P> g_misaki_g2p;
static std::once_flag g_misaki_init_flag;

MisakiG2P& get_global_g2p() {
    std::call_once(g_misaki_init_flag, []() {
        g_misaki_g2p = std::make_unique<MisakiG2P>();
    });
    return *g_misaki_g2p;
}

bool init_global_g2p(const std::string& lexicon_path,
                    const std::string& language) {
    return get_global_g2p().initialize(lexicon_path, language);
}

}  // namespace misaki
