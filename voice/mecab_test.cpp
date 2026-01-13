/**
 * MeCab C++ Direct Usage Example
 *
 * This program demonstrates how to use MeCab directly from C++
 * without Python dependencies. Provides maximum performance.
 *
 * Compilation:
 *   g++ -std=c++17 -O3 mecab_test.cpp -o mecab_test $(mecab-config --cflags --libs)
 *
 * Usage:
 *   ./mecab_test "こんにちは世界"
 */

#include <mecab.h>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>

class MeCabWrapper {
private:
    std::unique_ptr<MeCab::Tagger, decltype(&MeCab::deleteTagger)> tagger;

public:
    MeCabWrapper() : tagger(nullptr, &MeCab::deleteTagger) {
        // Get default MeCab arguments
        // On Apple Silicon, this will automatically use /opt/homebrew/etc/mecabrc
        const char* args = "";

        auto* raw_tagger = MeCab::createTagger(args);
        if (!raw_tagger) {
            throw std::runtime_error(std::string("MeCab initialization failed: ") +
                                     MeCab::getTaggerError());
        }

        tagger.reset(raw_tagger);
        std::cerr << "MeCab initialized successfully\n";
    }

    /**
     * Parse Japanese text and return morphological analysis
     */
    std::string parse(const std::string& text) {
        const char* result = tagger->parse(text.c_str());
        if (!result) {
            throw std::runtime_error("MeCab parsing failed");
        }
        return std::string(result);
    }

    /**
     * Parse text and extract individual nodes for advanced processing
     */
    struct Node {
        std::string surface;  // Original text
        std::string feature;  // Part of speech and other features
        std::string reading;  // Pronunciation (if available)
    };

    std::vector<Node> parseNodes(const std::string& text) {
        std::vector<Node> nodes;

        // Parse the text
        const MeCab::Node* node = tagger->parseToNode(text.c_str());
        if (!node) {
            throw std::runtime_error("MeCab parsing failed");
        }

        // Iterate through nodes (skip BOS/EOS)
        for (; node; node = node->next) {
            if (node->stat == MECAB_BOS_NODE || node->stat == MECAB_EOS_NODE) {
                continue;
            }

            Node n;
            n.surface = std::string(node->surface, node->length);
            n.feature = std::string(node->feature);

            // Extract reading from feature string (format varies by dictionary)
            // For ipadic: 品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用型,活用形,原形,読み,発音
            size_t pos = 0;
            int comma_count = 0;
            for (size_t i = 0; i < n.feature.length() && comma_count < 7; ++i) {
                if (n.feature[i] == ',') {
                    comma_count++;
                    if (comma_count == 7) {
                        pos = i + 1;
                    }
                }
            }
            if (comma_count >= 7) {
                size_t end = n.feature.find(',', pos);
                if (end != std::string::npos) {
                    n.reading = n.feature.substr(pos, end - pos);
                } else {
                    n.reading = n.feature.substr(pos);
                }
            }

            nodes.push_back(n);
        }

        return nodes;
    }

    /**
     * Benchmark parsing performance
     */
    double benchmark(const std::string& text, int iterations = 1000) {
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i) {
            parse(text);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;

        return duration.count() / iterations;
    }
};

void printUsage(const char* program) {
    std::cerr << "Usage:\n"
              << "  " << program << " <text>              Parse Japanese text\n"
              << "  " << program << " --nodes <text>      Show detailed morphological analysis\n"
              << "  " << program << " --bench <text>      Benchmark parsing performance\n"
              << "\nExamples:\n"
              << "  " << program << " \"こんにちは世界\"\n"
              << "  " << program << " --nodes \"今日は良い天気です\"\n"
              << "  " << program << " --bench \"こんにちは\"\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    try {
        MeCabWrapper mecab;

        std::string mode = argv[1];

        if (mode == "--nodes" && argc >= 3) {
            // Detailed node analysis
            std::string text = argv[2];
            std::cout << "Parsing: " << text << "\n\n";

            auto nodes = mecab.parseNodes(text);
            std::cout << "Surface | Feature | Reading\n";
            std::cout << "--------|---------|--------\n";
            for (const auto& node : nodes) {
                std::cout << node.surface << " | "
                         << node.feature << " | "
                         << node.reading << "\n";
            }

        } else if (mode == "--bench" && argc >= 3) {
            // Performance benchmark
            std::string text = argv[2];
            std::cout << "Benchmarking: " << text << "\n";

            double avg_time = mecab.benchmark(text);
            std::cout << "Average time per parse: " << avg_time << " ms\n";
            std::cout << "Throughput: " << (1000.0 / avg_time) << " parses/second\n";

        } else {
            // Simple parse
            std::string text = (mode.substr(0, 2) == "--") ? "" : mode;
            if (text.empty() && argc >= 2) {
                text = argv[1];
            }

            std::cout << mecab.parse(text);
        }

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
