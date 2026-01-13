// Test file for MPSStreamPool
// Requires building against the modified PyTorch/libtorch
//
// Build with:
// clang++ -std=c++17 -o test_mps_stream_pool test_mps_stream_pool.cpp \
//     -I${LIBTORCH_PATH}/include \
//     -I${LIBTORCH_PATH}/include/torch/csrc/api/include \
//     -L${LIBTORCH_PATH}/lib -ltorch -ltorch_cpu -lc10 \
//     -framework Metal -framework MetalPerformanceShaders -framework Foundation \
//     -Wl,-rpath,${LIBTORCH_PATH}/lib

#include <torch/torch.h>
#include <ATen/mps/MPSStream.h>
#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <set>

// Test 1: Basic stream pool access
void test_basic_pool_access() {
    std::cout << "Test 1: Basic pool access... ";

    // Get default stream
    at::mps::MPSStream* defaultStream = at::mps::getDefaultMPSStream();
    if (!defaultStream) {
        std::cout << "FAIL (getDefaultMPSStream returned nullptr)" << std::endl;
        return;
    }

    // Get current stream (should be default initially)
    at::mps::MPSStream* currentStream = at::mps::getCurrentMPSStream();
    if (currentStream != defaultStream) {
        std::cout << "FAIL (initial current != default)" << std::endl;
        return;
    }

    // Get a stream from pool
    at::mps::MPSStream* poolStream = at::mps::getStreamFromPool();
    if (!poolStream) {
        std::cout << "FAIL (getStreamFromPool returned nullptr)" << std::endl;
        return;
    }

    // Pool stream should be different from default
    if (poolStream == defaultStream) {
        std::cout << "FAIL (pool stream == default stream)" << std::endl;
        return;
    }

    std::cout << "PASS" << std::endl;
}

// Test 2: Thread-local stream assignment
void test_thread_local_streams() {
    std::cout << "Test 2: Thread-local streams... ";

    std::atomic<int> pass_count{0};
    std::atomic<int> fail_count{0};
    constexpr int num_threads = 8;

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            // Each thread gets a stream from the pool
            at::mps::MPSStream* myStream = at::mps::getStreamFromPool();
            at::mps::setCurrentMPSStream(myStream);

            // Verify current stream is what we set
            at::mps::MPSStream* current = at::mps::getCurrentMPSStream();
            if (current == myStream) {
                pass_count++;
            } else {
                fail_count++;
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    if (fail_count > 0) {
        std::cout << "FAIL (" << fail_count.load() << "/" << num_threads
                  << " threads got wrong stream)" << std::endl;
    } else {
        std::cout << "PASS (" << pass_count.load() << " threads OK)" << std::endl;
    }
}

// Test 3: Concurrent tensor creation (the actual crash scenario)
void test_concurrent_tensor_creation() {
    std::cout << "Test 3: Concurrent tensor creation... ";

    if (!torch::mps::is_available()) {
        std::cout << "SKIP (MPS not available)" << std::endl;
        return;
    }

    std::atomic<int> success{0};
    std::atomic<int> failure{0};
    constexpr int num_threads = 8;
    constexpr int iterations = 100;

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&]() {
            try {
                // Get dedicated stream for this thread
                at::mps::MPSStream* myStream = at::mps::getStreamFromPool();
                at::mps::setCurrentMPSStream(myStream);

                for (int j = 0; j < iterations; ++j) {
                    // Create tensor on MPS device
                    auto tensor = torch::randn({256, 256}, torch::device(torch::kMPS));

                    // Do some computation
                    auto result = tensor.matmul(tensor.t());

                    // Synchronize
                    torch::mps::synchronize();
                }
                success++;
            } catch (const std::exception& e) {
                std::cerr << "Exception: " << e.what() << std::endl;
                failure++;
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    if (failure > 0) {
        std::cout << "FAIL (" << failure.load() << "/" << num_threads
                  << " threads crashed)" << std::endl;
    } else {
        std::cout << "PASS (" << success.load() << " threads x "
                  << iterations << " iterations)" << std::endl;
    }
}

// Test 4: Round-robin allocation verification
void test_round_robin_allocation() {
    std::cout << "Test 4: Round-robin allocation... ";

    // Pool size is 32, streams 1-31 are used by round-robin
    constexpr int expected_pool_size = 31; // 32 - 1 (default)
    std::set<at::mps::MPSStream*> unique_streams;

    // Acquire many streams and track unique ones
    for (int i = 0; i < 100; ++i) {
        at::mps::MPSStream* stream = at::mps::getStreamFromPool();
        unique_streams.insert(stream);
    }

    // Should get all pool streams eventually
    if (unique_streams.size() < expected_pool_size) {
        std::cout << "WARN (only got " << unique_streams.size()
                  << " unique streams, expected " << expected_pool_size << ")" << std::endl;
    } else {
        std::cout << "PASS (" << unique_streams.size() << " unique streams)" << std::endl;
    }
}

// Test 5: Throughput benchmark - mutex vs pool
void test_throughput_benchmark() {
    std::cout << "Test 5: Throughput benchmark... " << std::endl;

    if (!torch::mps::is_available()) {
        std::cout << "  SKIP (MPS not available)" << std::endl;
        return;
    }

    constexpr int num_threads = 8;
    constexpr int iterations = 50;

    auto run_benchmark = [&](bool use_pool) {
        std::atomic<int> completed{0};

        auto start = std::chrono::high_resolution_clock::now();

        std::vector<std::thread> threads;
        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back([&, use_pool]() {
                if (use_pool) {
                    at::mps::MPSStream* myStream = at::mps::getStreamFromPool();
                    at::mps::setCurrentMPSStream(myStream);
                }
                // If not using pool, all threads share default stream

                for (int j = 0; j < iterations; ++j) {
                    auto tensor = torch::randn({128, 128}, torch::device(torch::kMPS));
                    auto result = tensor.matmul(tensor.t());
                    torch::mps::synchronize();
                }
                completed++;
            });
        }

        for (auto& t : threads) {
            t.join();
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        return duration.count();
    };

    std::cout << "  Running with shared stream (baseline)... ";
    auto shared_time = run_benchmark(false);
    std::cout << shared_time << "ms" << std::endl;

    std::cout << "  Running with stream pool... ";
    auto pool_time = run_benchmark(true);
    std::cout << pool_time << "ms" << std::endl;

    double speedup = static_cast<double>(shared_time) / pool_time;
    std::cout << "  Speedup: " << speedup << "x" << std::endl;
}

int main() {
    std::cout << "=== MPSStreamPool Tests ===" << std::endl;
    std::cout << "Pool size: " << at::mps::kMPSStreamsPerPool << " streams" << std::endl;
    std::cout << std::endl;

    test_basic_pool_access();
    test_thread_local_streams();
    test_round_robin_allocation();
    test_concurrent_tensor_creation();
    test_throughput_benchmark();

    std::cout << std::endl;
    std::cout << "=== Tests Complete ===" << std::endl;
    return 0;
}
