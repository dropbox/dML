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

#include "g2p.h"
#include "../mlx_inference_engine/misaki_g2p.h"

#include <cstring>
#include <stdexcept>
#include <iostream>

namespace kokoro {

// Implementation struct holding Misaki G2P
struct G2P::Impl {
    misaki::MisakiG2P misaki;
};

G2P::G2P() : impl_(std::make_unique<Impl>()) {}

G2P::~G2P() {
    terminate();
}

bool G2P::initialize(const std::string& voice, const std::string& lexicon_path) {
    if (initialized_) {
        return true;
    }

    lexicon_path_ = lexicon_path;
    voice_ = voice;

    // Try multiple paths for lexicon directory
    std::vector<std::string> search_paths = {
        lexicon_path,
        "../" + lexicon_path,
        "../../" + lexicon_path,
        "models/" + lexicon_path,
    };

    bool misaki_ok = false;
    for (const auto& path : search_paths) {
        if (impl_->misaki.initialize(path, voice)) {
            lexicon_path_ = path;
            misaki_ok = true;
            break;
        }
    }

    if (!misaki_ok) {
        std::cerr << "kokoro::G2P: WARNING - Could not initialize Misaki G2P. "
                  << "Make sure misaki_export/ directory exists.\n";
        std::cerr << "  Searched paths: ";
        for (const auto& p : search_paths) {
            std::cerr << p << " ";
        }
        std::cerr << "\n";
        // Note: We don't return false here - allow espeak-ng fallback mode
        // But this is NOT recommended for production use with Kokoro models
    }

    // Load vocab for tokenization
    std::string vocab_path = lexicon_path_ + "/vocab.json";
    if (!impl_->misaki.load_vocab(vocab_path)) {
        // Try alternate paths
        for (const auto& path : search_paths) {
            vocab_path = path + "/vocab.json";
            if (impl_->misaki.load_vocab(vocab_path)) {
                break;
            }
        }
    }

    initialized_ = true;
    return true;
}

void G2P::terminate() {
    if (initialized_) {
        // MisakiG2P cleanup is handled by its destructor
        initialized_ = false;
    }
}

bool G2P::set_language(const std::string& voice) {
    if (!initialized_) {
        return false;
    }

    // Already using this voice
    if (voice_ == voice) {
        return true;
    }

    // Reinitialize Misaki with new language
    if (!impl_->misaki.initialize(lexicon_path_, voice)) {
        return false;
    }

    voice_ = voice;
    return true;
}

std::string G2P::phonemize(const std::string& text) const {
    if (!initialized_) {
        throw std::runtime_error("G2P not initialized");
    }

    // Use Misaki G2P (primary)
    return impl_->misaki.phonemize(text);
}

}  // namespace kokoro
