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
#include "tokenizer.h"

#include <iostream>
#include <cstdlib>
#include <fstream>

void test_g2p_to_tokens() {
    std::cout << "Testing G2P -> Tokenizer integration..." << std::endl;

    // Initialize G2P
    kokoro::G2P g2p;
    if (!g2p.initialize("en-us")) {
        std::cerr << "  FAILED: G2P initialization failed" << std::endl;
        std::exit(1);
    }

    // Initialize Tokenizer with subset of Kokoro vocab
    kokoro::Tokenizer tokenizer;
    tokenizer.set_vocab({
        // Punctuation
        {" ", 16},
        // Consonants
        {"b", 44}, {"d", 46}, {"f", 48}, {"h", 50}, {"k", 53}, {"l", 54},
        {"m", 55}, {"n", 56}, {"p", 58}, {"s", 61}, {"t", 62}, {"v", 64},
        {"w", 65}, {"z", 68},
        // Vowels
        {"a", 43}, {"e", 47}, {"i", 51}, {"o", 57}, {"u", 63},
        // IPA vowels
        {"ə", 83}, {"ɪ", 102}, {"æ", 72}, {"ʌ", 138}, {"ʊ", 135},
        {"ɛ", 86}, {"ɔ", 76}, {"ɑ", 69}, {"ɜ", 87},
        {"O", 31},  // GOAT vowel (misaki style)
        // IPA consonants
        {"ð", 81}, {"θ", 119}, {"ʃ", 131}, {"ʒ", 147}, {"ŋ", 112},
        {"ɹ", 123}, {"ɚ", 85}, {"ʤ", 82}, {"ʧ", 133},
        // Diacritics
        {"ˈ", 156}, {"ˌ", 157}, {"ː", 158},
    });

    // Test G2P -> Tokenizer pipeline
    std::string text = "Hello world";
    std::string phonemes = g2p.phonemize(text);
    auto tokens = tokenizer.tokenize(phonemes);

    std::cout << "  Input text: '" << text << "'" << std::endl;
    std::cout << "  Phonemes: '" << phonemes << "'" << std::endl;
    std::cout << "  Tokens (" << tokens.size() << "): [";
    for (size_t i = 0; i < tokens.size(); i++) {
        std::cout << tokens[i];
        if (i < tokens.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Verify basic structure
    if (tokens.size() < 3) {
        std::cerr << "  FAILED: Need at least BOS + something + EOS" << std::endl;
        std::exit(1);
    }
    if (tokens.front() != 0) {
        std::cerr << "  FAILED: First token should be BOS (0)" << std::endl;
        std::exit(1);
    }
    if (tokens.back() != 0) {
        std::cerr << "  FAILED: Last token should be EOS (0)" << std::endl;
        std::exit(1);
    }

    // Verify no unknown tokens (except BOS/EOS which are also 0)
    int unknown_count = 0;
    for (size_t i = 1; i < tokens.size() - 1; i++) {
        if (tokens[i] == 0) {
            unknown_count++;
        }
    }
    std::cout << "  Unknown tokens: " << unknown_count << std::endl;

    std::cout << "  PASSED" << std::endl;
}

void test_multiple_sentences() {
    std::cout << "Testing multiple sentences..." << std::endl;

    kokoro::G2P g2p;
    if (!g2p.initialize("en-us")) {
        std::cerr << "  FAILED: G2P initialization failed" << std::endl;
        std::exit(1);
    }

    kokoro::Tokenizer tokenizer;
    tokenizer.set_vocab({
        {" ", 16}, {".", 4}, {",", 3}, {"!", 5}, {"?", 6},
        {"b", 44}, {"d", 46}, {"f", 48}, {"h", 50}, {"k", 53}, {"l", 54},
        {"m", 55}, {"n", 56}, {"p", 58}, {"s", 61}, {"t", 62}, {"v", 64},
        {"w", 65}, {"z", 68}, {"a", 43}, {"e", 47}, {"i", 51}, {"o", 57},
        {"u", 63}, {"ə", 83}, {"ɪ", 102}, {"æ", 72}, {"ʌ", 138}, {"ʊ", 135},
        {"ɛ", 86}, {"ɔ", 76}, {"ɑ", 69}, {"ɜ", 87}, {"O", 31},
        {"ð", 81}, {"θ", 119}, {"ʃ", 131}, {"ʒ", 147}, {"ŋ", 112},
        {"ɹ", 123}, {"ɚ", 85}, {"ʤ", 82}, {"ʧ", 133},
        {"ˈ", 156}, {"ˌ", 157}, {"ː", 158},
    });

    std::vector<std::string> test_texts = {
        "Hello.",
        "How are you?",
        "The weather is nice today!",
        "One, two, three.",
    };

    for (const auto& text : test_texts) {
        std::string phonemes = g2p.phonemize(text);
        auto tokens = tokenizer.tokenize(phonemes);

        std::cout << "  '" << text << "'" << std::endl;
        std::cout << "    -> '" << phonemes << "'" << std::endl;
        std::cout << "    -> " << tokens.size() << " tokens" << std::endl;

        if (tokens.size() < 3) {
            std::cerr << "  FAILED: Need at least BOS + something + EOS for: " << text << std::endl;
            std::exit(1);
        }
        if (tokens.front() != 0) {
            std::cerr << "  FAILED: First token should be BOS (0) for: " << text << std::endl;
            std::exit(1);
        }
        if (tokens.back() != 0) {
            std::cerr << "  FAILED: Last token should be EOS (0) for: " << text << std::endl;
            std::exit(1);
        }
    }

    std::cout << "  PASSED" << std::endl;
}

int main() {
    std::cout << "=== Kokoro Integration Tests ===" << std::endl;

    test_g2p_to_tokens();
    test_multiple_sentences();

    std::cout << "\n=== All tests passed! ===" << std::endl;
    return 0;
}
