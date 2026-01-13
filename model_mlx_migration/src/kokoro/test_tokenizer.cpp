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

#include <iostream>
#include <cassert>
#include <fstream>

void test_vocab_loading() {
    std::cout << "Testing vocab loading..." << std::endl;

    kokoro::Tokenizer tokenizer;

    // Create a simple test vocab file
    std::ofstream vocab_file("/tmp/test_vocab.json");
    vocab_file << R"({
        " ": 16,
        "h": 50,
        "ə": 83,
        "l": 54,
        "ˈ": 156,
        "O": 31,
        "w": 65,
        "ɜ": 87,
        "ɹ": 123,
        "d": 46
    })";
    vocab_file.close();

    assert(tokenizer.load_vocab("/tmp/test_vocab.json"));
    assert(tokenizer.vocab_size() == 10);
    assert(tokenizer.has_token(" "));
    assert(tokenizer.has_token("h"));
    assert(tokenizer.has_token("ə"));
    assert(!tokenizer.has_token("x"));

    std::cout << "  vocab_size = " << tokenizer.vocab_size() << std::endl;
    std::cout << "  PASSED" << std::endl;
}

void test_tokenize_raw() {
    std::cout << "Testing raw tokenization..." << std::endl;

    kokoro::Tokenizer tokenizer;
    tokenizer.set_vocab({
        {" ", 16},
        {"h", 50},
        {"ə", 83},
        {"l", 54},
        {"ˈ", 156},
        {"O", 31},
        {"w", 65},
        {"ɜ", 87},
        {"ɹ", 123},
        {"d", 46},
    });

    // Test: "həlˈO wˈɜɹld" (from misaki for "Hello world")
    std::string phonemes = "həlˈO wˈɜɹld";
    auto tokens = tokenizer.tokenize_raw(phonemes);

    std::cout << "  Input: " << phonemes << std::endl;
    std::cout << "  Tokens: [";
    for (size_t i = 0; i < tokens.size(); i++) {
        std::cout << tokens[i];
        if (i < tokens.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Expected: h=50, ə=83, l=54, ˈ=156, O=31, space=16, w=65, ˈ=156, ɜ=87, ɹ=123, l=54, d=46
    std::vector<int32_t> expected = {50, 83, 54, 156, 31, 16, 65, 156, 87, 123, 54, 46};
    assert(tokens.size() == expected.size());
    for (size_t i = 0; i < expected.size(); i++) {
        if (tokens[i] != expected[i]) {
            std::cerr << "  Mismatch at " << i << ": got " << tokens[i] << ", expected " << expected[i] << std::endl;
        }
        assert(tokens[i] == expected[i]);
    }

    std::cout << "  PASSED" << std::endl;
}

void test_tokenize_with_bos_eos() {
    std::cout << "Testing tokenization with BOS/EOS..." << std::endl;

    kokoro::Tokenizer tokenizer;
    tokenizer.set_vocab({
        {"h", 50},
        {"ə", 83},
        {"l", 54},
        {"ˈ", 156},
        {"O", 31},
    });

    std::string phonemes = "həlˈO";
    auto tokens = tokenizer.tokenize(phonemes);

    std::cout << "  Input: " << phonemes << std::endl;
    std::cout << "  Tokens: [";
    for (size_t i = 0; i < tokens.size(); i++) {
        std::cout << tokens[i];
        if (i < tokens.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Expected: BOS=0, h=50, ə=83, l=54, ˈ=156, O=31, EOS=0
    std::vector<int32_t> expected = {0, 50, 83, 54, 156, 31, 0};
    assert(tokens.size() == expected.size());
    for (size_t i = 0; i < expected.size(); i++) {
        assert(tokens[i] == expected[i]);
    }

    std::cout << "  PASSED" << std::endl;
}

void test_unknown_tokens() {
    std::cout << "Testing unknown tokens..." << std::endl;

    kokoro::Tokenizer tokenizer;
    tokenizer.set_vocab({
        {"a", 43},
        {"b", 44},
    });

    std::string input = "abc";  // 'c' is unknown
    auto tokens = tokenizer.tokenize_raw(input);

    std::cout << "  Input: " << input << std::endl;
    std::cout << "  Tokens: [";
    for (size_t i = 0; i < tokens.size(); i++) {
        std::cout << tokens[i];
        if (i < tokens.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Expected: a=43, b=44, UNK=0
    std::vector<int32_t> expected = {43, 44, 0};
    assert(tokens.size() == expected.size());
    for (size_t i = 0; i < expected.size(); i++) {
        assert(tokens[i] == expected[i]);
    }

    std::cout << "  PASSED" << std::endl;
}

int main() {
    std::cout << "=== Kokoro Tokenizer Tests ===" << std::endl;

    test_vocab_loading();
    test_tokenize_raw();
    test_tokenize_with_bos_eos();
    test_unknown_tokens();

    std::cout << "\n=== All tests passed! ===" << std::endl;
    return 0;
}
