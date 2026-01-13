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

#include <iostream>
#include <cstdlib>

void test_initialization() {
    std::cout << "Testing G2P initialization..." << std::endl;

    kokoro::G2P g2p;
    if (g2p.is_initialized()) {
        std::cerr << "  FAILED: G2P should not be initialized initially" << std::endl;
        std::exit(1);
    }

    bool ok = g2p.initialize("en-us");
    if (!ok) {
        std::cerr << "  FAILED: G2P initialization failed" << std::endl;
        std::exit(1);
    }
    if (!g2p.is_initialized()) {
        std::cerr << "  FAILED: G2P should be initialized after init()" << std::endl;
        std::exit(1);
    }

    std::cout << "  PASSED" << std::endl;
}

void test_phonemize() {
    std::cout << "Testing phonemization..." << std::endl;

    kokoro::G2P g2p;
    if (!g2p.initialize("en-us")) {
        std::cerr << "  FAILED: G2P initialization failed" << std::endl;
        std::exit(1);
    }

    // Test simple word
    std::string phonemes = g2p.phonemize("Hello");
    std::cout << "  'Hello' -> '" << phonemes << "'" << std::endl;
    if (phonemes.empty()) {
        std::cerr << "  FAILED: phonemes should not be empty" << std::endl;
        std::exit(1);
    }

    // Test phrase
    phonemes = g2p.phonemize("Hello world");
    std::cout << "  'Hello world' -> '" << phonemes << "'" << std::endl;
    if (phonemes.empty()) {
        std::cerr << "  FAILED: phonemes should not be empty" << std::endl;
        std::exit(1);
    }

    // Test longer sentence
    phonemes = g2p.phonemize("The quick brown fox jumps over the lazy dog.");
    std::cout << "  'The quick...' -> '" << phonemes << "'" << std::endl;
    if (phonemes.empty()) {
        std::cerr << "  FAILED: phonemes should not be empty" << std::endl;
        std::exit(1);
    }

    std::cout << "  PASSED" << std::endl;
}

void test_different_inputs() {
    std::cout << "Testing different inputs produce different outputs..." << std::endl;

    kokoro::G2P g2p;
    if (!g2p.initialize("en-us")) {
        std::cerr << "  FAILED: G2P initialization failed" << std::endl;
        std::exit(1);
    }

    std::string phonemes1 = g2p.phonemize("cat");
    std::string phonemes2 = g2p.phonemize("dog");
    std::string phonemes3 = g2p.phonemize("bird");

    std::cout << "  'cat' -> '" << phonemes1 << "'" << std::endl;
    std::cout << "  'dog' -> '" << phonemes2 << "'" << std::endl;
    std::cout << "  'bird' -> '" << phonemes3 << "'" << std::endl;

    if (phonemes1 == phonemes2) {
        std::cerr << "  FAILED: cat and dog should produce different phonemes" << std::endl;
        std::exit(1);
    }
    if (phonemes2 == phonemes3) {
        std::cerr << "  FAILED: dog and bird should produce different phonemes" << std::endl;
        std::exit(1);
    }
    if (phonemes1 == phonemes3) {
        std::cerr << "  FAILED: cat and bird should produce different phonemes" << std::endl;
        std::exit(1);
    }

    std::cout << "  PASSED" << std::endl;
}

int main() {
    std::cout << "=== Kokoro G2P Tests ===" << std::endl;

    test_initialization();
    test_phonemize();
    test_different_inputs();

    std::cout << "\n=== All tests passed! ===" << std::endl;
    return 0;
}
