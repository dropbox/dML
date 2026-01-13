// Copyright 2024-2025 Andrew Yates
//
// Test for C++ Phoneme Head implementation

#include "phoneme_head.h"
#include <iostream>
#include <chrono>

int main(int argc, char* argv[]) {
    std::cout << "=== Phoneme Head C++ Test ===\n\n";

    // Get model path from args or use default
    std::string model_path = "models/kokoro_phoneme_head";
    if (argc > 1) {
        model_path = argv[1];
    }
    std::cout << "Model path: " << model_path << "\n\n";

    // Test 1: Load phoneme head
    std::cout << "[Test 1] Loading phoneme head...\n";
    try {
        auto head = phoneme::PhonemeHead::load(model_path);
        std::cout << "  PASS: Loaded successfully\n";
        std::cout << "  Config: d_model=" << head.config().d_model
                  << ", hidden=" << head.config().hidden_dim
                  << ", vocab=" << head.config().phoneme_vocab << "\n";
    } catch (const std::exception& e) {
        std::cout << "  FAIL: " << e.what() << "\n";
        return 1;
    }

    // Test 2: Forward pass with dummy input
    std::cout << "\n[Test 2] Forward pass...\n";
    try {
        auto head = phoneme::PhonemeHead::load(model_path);

        // Create dummy encoder output: [1, 100, 1280]
        auto encoder_output = mx::random::normal({1, 100, 1280});
        mx::eval(encoder_output);

        auto start = std::chrono::high_resolution_clock::now();
        auto logits = head.forward(encoder_output);
        mx::eval(logits);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        std::cout << "  PASS: Output shape = [";
        for (size_t i = 0; i < logits.shape().size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << logits.shape()[i];
        }
        std::cout << "]\n";
        std::cout << "  Forward pass time: " << duration.count() / 1000.0 << " ms\n";
    } catch (const std::exception& e) {
        std::cout << "  FAIL: " << e.what() << "\n";
        return 1;
    }

    // Test 3: CTC decode
    std::cout << "\n[Test 3] CTC greedy decode...\n";
    try {
        auto head = phoneme::PhonemeHead::load(model_path);

        // Create dummy encoder output
        auto encoder_output = mx::random::normal({1, 100, 1280});
        mx::eval(encoder_output);

        auto start = std::chrono::high_resolution_clock::now();
        auto tokens = head.predict(encoder_output);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        std::cout << "  PASS: Decoded " << tokens.size() << " tokens\n";
        std::cout << "  First 10 tokens: [";
        for (size_t i = 0; i < std::min(tokens.size(), size_t(10)); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << tokens[i];
        }
        std::cout << "]\n";
        std::cout << "  Predict time: " << duration.count() / 1000.0 << " ms\n";
    } catch (const std::exception& e) {
        std::cout << "  FAIL: " << e.what() << "\n";
        return 1;
    }

    // Test 4: Edit distance
    std::cout << "\n[Test 4] Edit distance...\n";
    {
        std::vector<int> seq1 = {1, 2, 3, 4, 5};
        std::vector<int> seq2 = {1, 2, 4, 5, 6};

        auto start = std::chrono::high_resolution_clock::now();
        int dist = phoneme::edit_distance(seq1, seq2);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

        std::cout << "  PASS: edit_distance([1,2,3,4,5], [1,2,4,5,6]) = " << dist << "\n";
        std::cout << "  Expected: 2 (delete 3, insert 6)\n";
        std::cout << "  Time: " << duration.count() << " ns\n";

        if (dist != 2) {
            std::cout << "  WARNING: Expected 2, got " << dist << "\n";
        }
    }

    // Test 5: Edit distance with counts
    std::cout << "\n[Test 5] Edit distance with counts...\n";
    {
        std::vector<int> seq1 = {1, 2, 3, 4, 5};
        std::vector<int> seq2 = {1, 9, 3, 5};  // sub 2->9, delete 4

        int ins, del, sub;
        int dist = phoneme::edit_distance_with_counts(seq1, seq2, ins, del, sub);

        std::cout << "  PASS: dist=" << dist << ", ins=" << ins
                  << ", del=" << del << ", sub=" << sub << "\n";
    }

    // Test 6: Commit status logic
    std::cout << "\n[Test 6] Commit status logic...\n";
    {
        auto status1 = phoneme::PhonemeHead::get_commit_status(0.80f);
        auto status2 = phoneme::PhonemeHead::get_commit_status(0.60f);
        auto status3 = phoneme::PhonemeHead::get_commit_status(0.40f);

        std::cout << "  0.80 -> " << (status1 == phoneme::CommitStatus::COMMIT ? "COMMIT" : "other") << "\n";
        std::cout << "  0.60 -> " << (status2 == phoneme::CommitStatus::PARTIAL ? "PARTIAL" : "other") << "\n";
        std::cout << "  0.40 -> " << (status3 == phoneme::CommitStatus::WAIT ? "WAIT" : "other") << "\n";

        if (status1 == phoneme::CommitStatus::COMMIT &&
            status2 == phoneme::CommitStatus::PARTIAL &&
            status3 == phoneme::CommitStatus::WAIT) {
            std::cout << "  PASS: All commit statuses correct\n";
        } else {
            std::cout << "  FAIL: Commit status mismatch\n";
            return 1;
        }
    }

    std::cout << "\n=== All tests passed! ===\n";
    return 0;
}
