// Misaki G2P Benchmark - Find performance bottlenecks
#include "misaki_g2p.h"
#include <iostream>
#include <chrono>
#include <vector>

int main() {
    std::cout << "=== Misaki G2P Benchmark ===\n\n";

    // Benchmark English loading
    {
        auto start = std::chrono::high_resolution_clock::now();
        misaki::MisakiG2P g2p;
        g2p.initialize("misaki_export", "en-us");
        auto load_end = std::chrono::high_resolution_clock::now();

        double load_ms = std::chrono::duration<double, std::milli>(load_end - start).count();
        std::cout << "English load time: " << load_ms << " ms\n";
        std::cout << "Lexicon size: " << g2p.lexicon_size() << " entries\n\n";

        // Benchmark phonemization
        std::vector<std::string> sentences = {
            "Hello world",
            "The quick brown fox jumps over the lazy dog",
            "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled with the ends of worms and an oozy smell.",
            "To be or not to be, that is the question. Whether tis nobler in the mind to suffer the slings and arrows of outrageous fortune."
        };

        // Warmup
        for (const auto& s : sentences) {
            g2p.phonemize(s);
        }

        // Benchmark
        int iterations = 1000;
        auto phon_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            for (const auto& s : sentences) {
                g2p.phonemize(s);
            }
        }
        auto phon_end = std::chrono::high_resolution_clock::now();

        double phon_ms = std::chrono::duration<double, std::milli>(phon_end - phon_start).count();
        std::cout << "Phonemization (" << iterations << "x " << sentences.size() << " sentences):\n";
        std::cout << "  Total: " << phon_ms << " ms\n";
        std::cout << "  Per sentence: " << phon_ms / (iterations * sentences.size()) << " ms\n";
        std::cout << "  Per call: " << phon_ms / (iterations * sentences.size()) * 1000 << " µs\n\n";
    }

    // Benchmark Japanese (lighter weight)
    {
        auto start = std::chrono::high_resolution_clock::now();
        misaki::MisakiG2P g2p;
        g2p.initialize("misaki_export", "ja");
        auto load_end = std::chrono::high_resolution_clock::now();

        double load_ms = std::chrono::duration<double, std::milli>(load_end - start).count();
        std::cout << "Japanese load time: " << load_ms << " ms\n";

        // Benchmark
        int iterations = 1000;
        std::string text = "こんにちは世界";
        auto phon_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            g2p.phonemize(text);
        }
        auto phon_end = std::chrono::high_resolution_clock::now();

        double phon_ms = std::chrono::duration<double, std::milli>(phon_end - phon_start).count();
        std::cout << "Japanese phonemization (" << iterations << "x):\n";
        std::cout << "  Total: " << phon_ms << " ms\n";
        std::cout << "  Per call: " << phon_ms / iterations * 1000 << " µs\n\n";
    }

    // Benchmark espeak (Spanish)
    {
        auto start = std::chrono::high_resolution_clock::now();
        misaki::MisakiG2P g2p;
        g2p.initialize("misaki_export", "es");
        auto load_end = std::chrono::high_resolution_clock::now();

        double load_ms = std::chrono::duration<double, std::milli>(load_end - start).count();
        std::cout << "Spanish (espeak) load time: " << load_ms << " ms\n";

        // Benchmark
        int iterations = 100;  // espeak is slower
        std::string text = "Hola mundo";
        auto phon_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            g2p.phonemize(text);
        }
        auto phon_end = std::chrono::high_resolution_clock::now();

        double phon_ms = std::chrono::duration<double, std::milli>(phon_end - phon_start).count();
        std::cout << "Spanish phonemization (" << iterations << "x):\n";
        std::cout << "  Total: " << phon_ms << " ms\n";
        std::cout << "  Per call: " << phon_ms / iterations * 1000 << " µs\n\n";
    }

    std::cout << "=== Summary ===\n";
    std::cout << "Bottleneck: JSON parsing during lexicon load\n";
    std::cout << "Solution: Binary format or memory-mapped files\n";

    return 0;
}
