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

#include "tokenizer.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace kokoro {

std::string Tokenizer::next_utf8_char(const std::string& str, size_t& pos) {
    if (pos >= str.size()) {
        return "";
    }

    unsigned char c = str[pos];
    size_t char_len = 1;

    // Determine UTF-8 character length from first byte
    if ((c & 0x80) == 0) {
        // ASCII (0xxxxxxx)
        char_len = 1;
    } else if ((c & 0xE0) == 0xC0) {
        // 2-byte (110xxxxx)
        char_len = 2;
    } else if ((c & 0xF0) == 0xE0) {
        // 3-byte (1110xxxx)
        char_len = 3;
    } else if ((c & 0xF8) == 0xF0) {
        // 4-byte (11110xxx)
        char_len = 4;
    }

    // Handle combining characters (like stress marks)
    // These follow the base character and should be included
    std::string result = str.substr(pos, char_len);
    pos += char_len;

    // Check for combining diacritical marks (U+0300 to U+036F)
    // These are 2-byte sequences starting with 0xCC or 0xCD
    while (pos < str.size()) {
        unsigned char next = str[pos];
        if (next == 0xCC || next == 0xCD) {
            // Potential combining character, include it
            size_t combine_len = 2;
            if (pos + combine_len <= str.size()) {
                result += str.substr(pos, combine_len);
                pos += combine_len;
            } else {
                break;
            }
        } else {
            break;
        }
    }

    return result;
}

bool Tokenizer::load_vocab(const std::string& vocab_path) {
    std::ifstream file(vocab_path);
    if (!file.is_open()) {
        return false;
    }

    // Simple JSON parsing for {"key": value, ...} format
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string json = buffer.str();

    vocab_.clear();

    // Find the object content between { and }
    size_t start = json.find('{');
    size_t end = json.rfind('}');
    if (start == std::string::npos || end == std::string::npos || start >= end) {
        return false;
    }

    std::string content = json.substr(start + 1, end - start - 1);

    // Parse key-value pairs
    size_t pos = 0;
    while (pos < content.size()) {
        // Find opening quote for key
        size_t key_start = content.find('"', pos);
        if (key_start == std::string::npos) break;

        // Find closing quote for key (handling escape sequences)
        size_t key_end = key_start + 1;
        while (key_end < content.size()) {
            if (content[key_end] == '"' && content[key_end - 1] != '\\') {
                break;
            }
            key_end++;
        }
        if (key_end >= content.size()) break;

        std::string key = content.substr(key_start + 1, key_end - key_start - 1);

        // Unescape the key (handle \\, \")
        std::string unescaped;
        for (size_t i = 0; i < key.size(); i++) {
            if (key[i] == '\\' && i + 1 < key.size()) {
                unescaped += key[i + 1];
                i++;
            } else {
                unescaped += key[i];
            }
        }
        key = unescaped;

        // Find colon
        size_t colon = content.find(':', key_end);
        if (colon == std::string::npos) break;

        // Find number value
        size_t num_start = colon + 1;
        while (num_start < content.size() && (content[num_start] == ' ' || content[num_start] == '\t' || content[num_start] == '\n')) {
            num_start++;
        }

        size_t num_end = num_start;
        while (num_end < content.size() && (isdigit(content[num_end]) || content[num_end] == '-')) {
            num_end++;
        }

        if (num_start < num_end) {
            int32_t value = std::stoi(content.substr(num_start, num_end - num_start));
            vocab_[key] = value;
        }

        pos = num_end;
    }

    return !vocab_.empty();
}

void Tokenizer::set_vocab(const std::unordered_map<std::string, int32_t>& vocab) {
    vocab_ = vocab;
}

std::vector<int32_t> Tokenizer::tokenize(const std::string& phonemes) const {
    std::vector<int32_t> tokens;
    tokens.push_back(BOS_TOKEN);

    auto raw = tokenize_raw(phonemes);
    tokens.insert(tokens.end(), raw.begin(), raw.end());

    tokens.push_back(EOS_TOKEN);
    return tokens;
}

std::vector<int32_t> Tokenizer::tokenize_raw(const std::string& phonemes) const {
    std::vector<int32_t> tokens;

    size_t pos = 0;
    while (pos < phonemes.size()) {
        std::string ch = next_utf8_char(phonemes, pos);
        if (ch.empty()) break;

        auto it = vocab_.find(ch);
        if (it != vocab_.end()) {
            tokens.push_back(it->second);
        } else {
            // Unknown character - use UNK token
            tokens.push_back(UNK_TOKEN);
        }
    }

    return tokens;
}

bool Tokenizer::has_token(const std::string& token) const {
    return vocab_.find(token) != vocab_.end();
}

}  // namespace kokoro
