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
#include <cstdint>
#include <memory>

namespace kokoro {

/**
 * Grapheme-to-Phoneme converter using Misaki lexicons (primary) with optional
 * espeak-ng fallback for out-of-vocabulary words.
 *
 * Converts English text to IPA phoneme strings that can be tokenized
 * for the Kokoro TTS model.
 *
 * IMPORTANT: Misaki is the correct G2P for Kokoro. The model was trained on
 * Misaki phonemes, not espeak-ng. Using espeak-ng alone produces incorrect
 * phoneme sequences.
 */
class G2P {
public:
    G2P();
    ~G2P();

    // Prevent copying (espeak-ng has global state)
    G2P(const G2P&) = delete;
    G2P& operator=(const G2P&) = delete;

    /**
     * Initialize G2P with Misaki lexicons (primary) and optional espeak-ng fallback.
     *
     * Must be called before any phonemize() calls.
     * Loads lexicons from "misaki_export/" directory (relative to working dir).
     *
     * @param voice Voice/language code (default "en-us" for American English)
     * @param lexicon_path Path to Misaki lexicons (default "misaki_export")
     * @return true if initialized successfully
     */
    bool initialize(const std::string& voice = "en-us",
                   const std::string& lexicon_path = "misaki_export");

    /**
     * Check if G2P is initialized.
     */
    bool is_initialized() const { return initialized_; }

    /**
     * Convert text to IPA phoneme string.
     *
     * @param text Input text (UTF-8)
     * @return IPA phoneme string (UTF-8)
     */
    std::string phonemize(const std::string& text) const;

    /**
     * Switch to a different language.
     *
     * @param voice espeak-ng voice code (e.g., "en-us", "ja", "cmn")
     * @return true if language switch succeeded
     */
    bool set_language(const std::string& voice);

    /**
     * Get current language/voice code.
     */
    std::string current_language() const { return voice_; }

    /**
     * Terminate espeak-ng.
     * Called automatically by destructor.
     */
    void terminate();

private:
    bool initialized_ = false;
    std::string voice_;
    std::string lexicon_path_;

    // Forward declare - actual type is in cpp file
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace kokoro
