// Copyright 2024-2025 Andrew Yates
// CLI for Zipformer ASR inference
//
// Licensed under the Apache License, Version 2.0

#include "zipformer/zipformer.hpp"
#include "zipformer/inference.hpp"
#include <iostream>
#include <string>
#include <vector>

void print_usage(const char* program) {
    std::cerr << "Usage: " << program << " <command> [options]\n"
              << "\nCommands:\n"
              << "  transcribe <model_dir> <audio_file>    Transcribe audio file\n"
              << "  benchmark <model_dir> <audio_file> [n] Benchmark model performance\n"
              << "  validate <model_dir> <reference_file>  Validate against Python reference\n"
              << "  info <model_dir>                       Show model information\n"
              << "\nExamples:\n"
              << "  " << program << " transcribe ./checkpoints/zipformer test.wav\n"
              << "  " << program << " benchmark ./checkpoints/zipformer test.wav 10\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string command = argv[1];

    try {
        if (command == "transcribe") {
            if (argc < 4) {
                std::cerr << "Usage: " << argv[0] << " transcribe <model_dir> <audio_file>\n";
                return 1;
            }

            std::string model_dir = argv[2];
            std::string audio_file = argv[3];

            std::cout << "Loading model from " << model_dir << "...\n";
            auto result = zipformer::transcribe_file(model_dir, audio_file);

            std::cout << "\nTranscription:\n" << result.text << "\n";
            std::cout << "\nTokens: " << result.tokens.size() << "\n";
            std::cout << "Confidence: " << result.confidence << "\n";

        } else if (command == "benchmark") {
            if (argc < 4) {
                std::cerr << "Usage: " << argv[0] << " benchmark <model_dir> <audio_file> [iterations]\n";
                return 1;
            }

            std::string model_dir = argv[2];
            std::string audio_file = argv[3];
            int iterations = (argc > 4) ? std::stoi(argv[4]) : 10;

            std::cout << "Benchmarking model from " << model_dir << "...\n";
            std::cout << "Audio: " << audio_file << "\n";
            std::cout << "Iterations: " << iterations << "\n\n";

            auto result = zipformer::benchmark_model(model_dir, audio_file, iterations);

            std::cout << "Results:\n";
            std::cout << "  Total audio (s):     " << result.total_audio_seconds << "\n";
            std::cout << "  Total inference (s): " << result.total_inference_seconds << "\n";
            std::cout << "  RTF:                 " << result.rtf << "x\n";
            std::cout << "  First token (ms):    " << result.first_token_latency_ms << "\n";
            std::cout << "  Tokens/sec:          " << result.tokens_per_second << "\n";

        } else if (command == "validate") {
            if (argc < 4) {
                std::cerr << "Usage: " << argv[0] << " validate <model_dir> <reference_file>\n";
                return 1;
            }

            std::string model_dir = argv[2];
            std::string reference_file = argv[3];

            std::cout << "Validating model from " << model_dir << "...\n";
            std::cout << "Reference: " << reference_file << "\n\n";

            bool passed = zipformer::validate_against_reference(model_dir, reference_file);
            return passed ? 0 : 1;

        } else if (command == "info") {
            if (argc < 3) {
                std::cerr << "Usage: " << argv[0] << " info <model_dir>\n";
                return 1;
            }

            std::string model_dir = argv[2];

            std::cout << "Loading model from " << model_dir << "...\n";
            auto model = zipformer::load_model(model_dir);

            auto& config = model->config();
            std::cout << "\nModel Configuration:\n";
            std::cout << "  Encoder:\n";
            std::cout << "    Features:   " << config.encoder.num_features << "\n";
            std::cout << "    Layers:     [";
            for (size_t i = 0; i < config.encoder.num_encoder_layers.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << config.encoder.num_encoder_layers[i];
            }
            std::cout << "]\n";
            std::cout << "    Dims:       [";
            for (size_t i = 0; i < config.encoder.encoder_dims.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << config.encoder.encoder_dims[i];
            }
            std::cout << "]\n";
            std::cout << "    Causal:     " << (config.encoder.causal ? "yes" : "no") << "\n";
            std::cout << "  Decoder:\n";
            std::cout << "    Vocab size: " << config.decoder.vocab_size << "\n";
            std::cout << "    Dim:        " << config.decoder.decoder_dim << "\n";
            std::cout << "    Context:    " << config.decoder.context_size << "\n";
            std::cout << "  Joiner:\n";
            std::cout << "    Dim:        " << config.joiner.joiner_dim << "\n";
            std::cout << "  Features:\n";
            std::cout << "    Sample rate:" << config.features.sample_rate << "\n";
            std::cout << "    Mel bins:   " << config.features.num_mel_bins << "\n";

        } else {
            std::cerr << "Unknown command: " << command << "\n";
            print_usage(argv[0]);
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
