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

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace kokoro {

/**
 * Tokenizer for converting IPA phoneme strings to token IDs.
 *
 * The Kokoro model uses a fixed vocabulary of ~114 IPA symbols.
 * This tokenizer handles UTF-8 IPA strings and maps them to integer token IDs.
 */
class Tokenizer {
public:
    // Special token IDs
    static constexpr int32_t BOS_TOKEN = 0;  // Beginning of sequence
    static constexpr int32_t EOS_TOKEN = 0;  // End of sequence (same as BOS)
    static constexpr int32_t PAD_TOKEN = 0;  // Padding token
    static constexpr int32_t UNK_TOKEN = 0;  // Unknown token (same as PAD)
    static constexpr int32_t SPACE_TOKEN = 16;  // Space character

    /**
     * Load vocabulary from JSON file.
     *
     * Expected format: {"phoneme": token_id, ...}
     * Example: {";": 1, ":": 2, " ": 16, "a": 43, ...}
     *
     * @param vocab_path Path to vocab JSON file
     * @return true if loaded successfully
     */
    bool load_vocab(const std::string& vocab_path);

    /**
     * Load vocabulary from C++ map directly.
     * Useful for embedding vocab in compiled binary.
     */
    void set_vocab(const std::unordered_map<std::string, int32_t>& vocab);

    /**
     * Tokenize IPA phoneme string to token IDs.
     *
     * Adds BOS and EOS tokens automatically.
     * Unknown characters map to UNK_TOKEN (0).
     *
     * @param phonemes UTF-8 IPA string (e.g., "həlˈoʊ wˈɜːld")
     * @return Vector of token IDs with BOS/EOS
     */
    std::vector<int32_t> tokenize(const std::string& phonemes) const;

    /**
     * Tokenize without adding BOS/EOS.
     *
     * @param phonemes UTF-8 IPA string
     * @return Vector of token IDs (no BOS/EOS)
     */
    std::vector<int32_t> tokenize_raw(const std::string& phonemes) const;

    /**
     * Check if a character is in the vocabulary.
     */
    bool has_token(const std::string& token) const;

    /**
     * Get vocabulary size.
     */
    size_t vocab_size() const { return vocab_.size(); }

    /**
     * Check if vocab is loaded.
     */
    bool is_loaded() const { return !vocab_.empty(); }

private:
    // Map from UTF-8 character string to token ID
    std::unordered_map<std::string, int32_t> vocab_;

    /**
     * Extract next UTF-8 character from string.
     * Returns the character and advances the position.
     */
    static std::string next_utf8_char(const std::string& str, size_t& pos);
};

}  // namespace kokoro
