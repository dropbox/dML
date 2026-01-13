// ThreadSanitizer test for MPS parallel inference
// This is an Objective-C++ file to work with Metal/MPS
// Uses ATen APIs directly (not torch::) to avoid header conflicts

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <ATen/ATen.h>
#include <ATen/mps/MPSStream.h>
#include <c10/core/Device.h>

#include <thread>
#include <vector>
#include <iostream>
#include <chrono>
#include <atomic>
#include <mutex>
#include <sstream>
#include <cstdlib>

std::atomic<int> completed_count{0};
std::atomic<int> error_count{0};
std::mutex cout_mutex;

void worker_thread(int thread_id, int iterations) {
    @autoreleasepool {
        try {
            for (int i = 0; i < iterations; i++) {
                // Create random tensor on MPS using ATen API
                auto options = at::TensorOptions().device(at::kMPS).dtype(at::kFloat);
                auto x = at::randn({64, 64}, options);
                auto y = at::matmul(x, x);

                // Force synchronization using current stream
                at::mps::getCurrentMPSStream()->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);
            }
            completed_count++;
            {
                std::lock_guard<std::mutex> lock(cout_mutex);
                std::cout << "Thread " << thread_id << " completed " << iterations << " iterations" << std::endl;
            }
        } catch (const std::exception& e) {
            error_count++;
            {
                std::lock_guard<std::mutex> lock(cout_mutex);
                std::cerr << "Thread " << thread_id << " error: " << e.what() << std::endl;
            }
        }
    }
}

int main(int argc, char* argv[]) {
    @autoreleasepool {
        std::cout << "=== MPS ThreadSanitizer Test ===" << std::endl;

        // Check MPS availability via ATen
        if (!at::mps::is_available()) {
            std::cerr << "ERROR: MPS not available" << std::endl;
            NSProcessInfo* processInfo = [NSProcessInfo processInfo];
            std::cerr << "macOS: " << [[processInfo operatingSystemVersionString] UTF8String] << std::endl;

            id<MTLDevice> defaultDev = MTLCreateSystemDefaultDevice();
            if (defaultDev) {
                std::cerr << "MTLCreateSystemDefaultDevice: " << [[defaultDev name] UTF8String] << std::endl;
            } else {
                std::cerr << "MTLCreateSystemDefaultDevice: nil" << std::endl;
            }

            NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
            std::cerr << "MTLCopyAllDevices count: " << [devices count] << std::endl;
            for (id<MTLDevice> dev in devices) {
                std::cerr << "MTLDevice: " << [[dev name] UTF8String] << std::endl;
            }

            std::cerr << "NOTE: MTLCreateSystemDefaultDevice returning nil typically indicates Metal is unavailable in this process (often due to sandbox/headless restrictions)." << std::endl;
            std::cerr << "      Quick standalone check: ./tests/metal_diagnostics.sh" << std::endl;
            std::cerr << "      Run this test from a normal Terminal session with Metal device access. For the autonomous loop, see run_worker.sh (Codex uses --dangerously-bypass-approvals-and-sandbox)." << std::endl;
            return 1;
        }
        std::cout << "MPS available: YES" << std::endl;

        // Warm up MPS
        {
            auto options = at::TensorOptions().device(at::kMPS).dtype(at::kFloat);
            auto x = at::randn({32, 32}, options);
            auto y = at::matmul(x, x);
            at::mps::getDefaultMPSStream()->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);
        }
        std::cout << "MPS warmup complete" << std::endl;

        // Parse command-line arguments
        // Supports: ./tsan_mps_test [num_threads] [iterations]
        //       or: ./tsan_mps_test -t <threads> -i <iterations>
        //       or: ./tsan_mps_test --threads=<N> --iterations=<N>
        int num_threads = 8;
        int iterations = 50;
        bool show_help = false;

        for (int i = 1; i < argc; i++) {
            std::string arg(argv[i]);
            if (arg == "-h" || arg == "--help") {
                show_help = true;
            } else if (arg == "-t" && i + 1 < argc) {
                num_threads = std::atoi(argv[++i]);
            } else if (arg == "-i" && i + 1 < argc) {
                iterations = std::atoi(argv[++i]);
            } else if (arg.rfind("--threads=", 0) == 0) {
                num_threads = std::atoi(arg.substr(10).c_str());
            } else if (arg.rfind("--iterations=", 0) == 0) {
                iterations = std::atoi(arg.substr(13).c_str());
            } else if (arg[0] != '-' && i == 1) {
                // Positional: first arg is num_threads
                num_threads = std::atoi(arg.c_str());
            } else if (arg[0] != '-' && i == 2) {
                // Positional: second arg is iterations
                iterations = std::atoi(arg.c_str());
            } else {
                std::cerr << "ERROR: Unknown argument: " << arg << std::endl;
                show_help = true;
            }
        }

        if (show_help) {
            std::cout << "Usage: " << argv[0] << " [OPTIONS] [num_threads] [iterations]\n"
                      << "\nOptions:\n"
                      << "  -t, --threads=N     Number of threads (1-31, default: 8)\n"
                      << "  -i, --iterations=N  Iterations per thread (default: 50)\n"
                      << "  -h, --help          Show this help\n"
                      << "\nExamples:\n"
                      << "  " << argv[0] << " 31 100\n"
                      << "  " << argv[0] << " --threads=31 --iterations=100\n"
                      << "  " << argv[0] << " -t 8 -i 50\n";
            return 0;
        }

        if (num_threads < 1 || num_threads > 31) {
            std::cerr << "ERROR: num_threads must be 1-31 (got " << num_threads << ")" << std::endl;
            return 1;
        }
        if (iterations < 1) {
            std::cerr << "ERROR: iterations must be >= 1 (got " << iterations << ")" << std::endl;
            return 1;
        }

        std::cout << "Starting " << num_threads << " threads, " << iterations << " iterations each..." << std::endl;

        auto start = std::chrono::high_resolution_clock::now();

        std::vector<std::thread> threads;
        for (int i = 0; i < num_threads; i++) {
            threads.emplace_back(worker_thread, i, iterations);
        }

        for (auto& t : threads) {
            t.join();
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "All threads completed in " << duration.count() << "ms" << std::endl;
        std::cout << "Completed: " << completed_count.load() << "/" << num_threads << std::endl;
        std::cout << "Errors: " << error_count.load() << std::endl;

        if (error_count.load() == 0 && completed_count.load() == num_threads) {
            std::cout << "=== TEST PASSED ===" << std::endl;
            return 0;
        } else {
            std::cout << "=== TEST FAILED ===" << std::endl;
            return 1;
        }
    }
}
