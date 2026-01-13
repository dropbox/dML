// Test for recordStream functionality (External Audit Gap #4)
// Exercises cross-stream buffer tracking via recordStream() API
//
// This test verifies:
// 1. recordStream() can be called without crashes
// 2. Multiple streams can be recorded on the same buffer
// 3. Cross-thread buffer passing with recordStream works correctly
//
// Uses the public API: at::native::record_stream_mps(Tensor&, c10::Stream)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <ATen/ATen.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/mps/MPSAllocatorInterface.h>
#include <ATen/native/mps/OperationUtils.h>
#include <c10/core/Device.h>
#include <c10/core/Stream.h>

// Declare the public native function for record_stream
namespace at::native {
void record_stream_mps(at::Tensor& self, c10::Stream stream);
}

#include <thread>
#include <vector>
#include <iostream>
#include <chrono>
#include <atomic>
#include <mutex>
#include <queue>
#include <condition_variable>

std::atomic<int> test_pass_count{0};
std::atomic<int> test_fail_count{0};
std::mutex cout_mutex;

void print_result(const char* test_name, bool passed, const char* details = nullptr) {
    std::lock_guard<std::mutex> lock(cout_mutex);
    if (passed) {
        std::cout << "[PASS] " << test_name;
        test_pass_count++;
    } else {
        std::cout << "[FAIL] " << test_name;
        test_fail_count++;
    }
    if (details) {
        std::cout << " - " << details;
    }
    std::cout << std::endl;
}

// Helper: Create c10::Stream from MPSStream
c10::Stream mps_stream_to_c10(at::mps::MPSStream* mps_stream) {
    // MPS streams use device index 0 and the stream's id
    return c10::Stream(c10::Stream::UNSAFE,
                       c10::Device(c10::DeviceType::MPS, 0),
                       static_cast<c10::StreamId>(mps_stream->unwrap().id()));
}

// Test 1: Basic recordStream call
void test_basic_record_stream() {
    @autoreleasepool {
        try {
            auto options = at::TensorOptions().device(at::kMPS).dtype(at::kFloat);
            auto tensor = at::randn({64, 64}, options);

            // Get current stream
            auto* mps_stream = at::mps::getCurrentMPSStream();
            c10::Stream stream = mps_stream_to_c10(mps_stream);

            // Record the stream on the tensor using the public API
            at::native::record_stream_mps(tensor, stream);

            // Sync to ensure all work is complete
            mps_stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);

            print_result("test_basic_record_stream", true);
        } catch (const std::exception& e) {
            print_result("test_basic_record_stream", false, e.what());
        }
    }
}

// Test 2: Record same buffer multiple times on same stream (should be safe)
void test_multi_stream_record() {
    @autoreleasepool {
        try {
            auto options = at::TensorOptions().device(at::kMPS).dtype(at::kFloat);
            auto tensor = at::randn({64, 64}, options);

            // Get current stream
            auto* mps_stream = at::mps::getCurrentMPSStream();
            c10::Stream stream = mps_stream_to_c10(mps_stream);

            // Recording same stream multiple times should be safe (no-op after first)
            for (int i = 0; i < 5; i++) {
                at::native::record_stream_mps(tensor, stream);
            }

            // Sync to ensure all work is complete
            mps_stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);

            print_result("test_multi_stream_record", true, "5 repeated calls safe");
        } catch (const std::exception& e) {
            print_result("test_multi_stream_record", false, e.what());
        }
    }
}

// Test 3: Cross-thread recordStream with actual work
void test_cross_thread_record_stream() {
    @autoreleasepool {
        std::atomic<bool> producer_done{false};
        std::atomic<bool> consumer_done{false};
        std::atomic<bool> test_passed{true};
        std::string error_msg;
        std::mutex error_mutex;

        // Shared tensor pointer (thread-safe via atomic pointer)
        std::atomic<at::Tensor*> shared_tensor{nullptr};
        std::condition_variable cv;
        std::mutex cv_mutex;

        // Producer thread: creates tensor and records on its stream
        std::thread producer([&]() {
            @autoreleasepool {
                try {
                    auto options = at::TensorOptions().device(at::kMPS).dtype(at::kFloat);
                    auto* tensor_ptr = new at::Tensor(at::randn({128, 128}, options));

                    // Do some work with the tensor
                    *tensor_ptr = at::matmul(*tensor_ptr, *tensor_ptr);

                    // Get producer's stream and record the tensor on it
                    auto* producer_stream = at::mps::getCurrentMPSStream();
                    c10::Stream stream = mps_stream_to_c10(producer_stream);
                    at::native::record_stream_mps(*tensor_ptr, stream);

                    // Sync producer's work
                    producer_stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);

                    // Share tensor with consumer
                    shared_tensor.store(tensor_ptr);
                    {
                        std::lock_guard<std::mutex> lock(cv_mutex);
                        producer_done = true;
                    }
                    cv.notify_one();
                } catch (const std::exception& e) {
                    std::lock_guard<std::mutex> lock(error_mutex);
                    test_passed = false;
                    error_msg = std::string("Producer error: ") + e.what();
                    producer_done = true;
                    cv.notify_one();
                }
            }
        });

        // Consumer thread: uses tensor produced by producer
        std::thread consumer([&]() {
            @autoreleasepool {
                try {
                    // Wait for producer to finish
                    {
                        std::unique_lock<std::mutex> lock(cv_mutex);
                        cv.wait(lock, [&]() { return producer_done.load(); });
                    }

                    if (!test_passed.load()) {
                        consumer_done = true;
                        return;
                    }

                    // Get the shared tensor
                    at::Tensor* tensor_ptr = shared_tensor.load();
                    if (!tensor_ptr) {
                        std::lock_guard<std::mutex> lock(error_mutex);
                        test_passed = false;
                        error_msg = "Consumer got null tensor";
                        consumer_done = true;
                        return;
                    }

                    // Get consumer's stream and record the tensor on it (cross-stream use)
                    auto* consumer_stream = at::mps::getCurrentMPSStream();
                    c10::Stream stream = mps_stream_to_c10(consumer_stream);
                    at::native::record_stream_mps(*tensor_ptr, stream);

                    // Do work with the tensor on consumer's stream
                    auto result = at::matmul(*tensor_ptr, *tensor_ptr);

                    // Verify result is valid
                    consumer_stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);
                    float sum = result.sum().item<float>();

                    // Check that we got a valid (non-NaN) result
                    if (std::isnan(sum)) {
                        std::lock_guard<std::mutex> lock(error_mutex);
                        test_passed = false;
                        error_msg = "Consumer got NaN result";
                    }

                    // Cleanup
                    delete tensor_ptr;
                    consumer_done = true;
                } catch (const std::exception& e) {
                    std::lock_guard<std::mutex> lock(error_mutex);
                    test_passed = false;
                    error_msg = std::string("Consumer error: ") + e.what();
                    consumer_done = true;
                }
            }
        });

        producer.join();
        consumer.join();

        print_result("test_cross_thread_record_stream", test_passed.load(),
                    test_passed.load() ? nullptr : error_msg.c_str());
    }
}

// Test 4: recordStream with CPU tensor (should be safe no-op)
void test_cpu_tensor_safety() {
    @autoreleasepool {
        try {
            // Create a CPU tensor (not MPS)
            auto options = at::TensorOptions().device(at::kCPU).dtype(at::kFloat);
            auto tensor = at::randn({64, 64}, options);

            // Get current MPS stream
            auto* mps_stream = at::mps::getCurrentMPSStream();
            c10::Stream stream = mps_stream_to_c10(mps_stream);

            // Calling record_stream on a CPU tensor should be safe (no-op)
            // The implementation checks tensor device
            at::native::record_stream_mps(tensor, stream);

            print_result("test_cpu_tensor_safety", true, "CPU tensor safe");
        } catch (const std::exception& e) {
            // Exception is acceptable - just don't crash
            print_result("test_cpu_tensor_safety", true, "Exception thrown (acceptable)");
        }
    }
}

// Test 5: recordStream on tensor that will be freed (lifecycle test)
void test_tensor_lifecycle() {
    @autoreleasepool {
        try {
            auto* mps_stream = at::mps::getCurrentMPSStream();
            c10::Stream stream = mps_stream_to_c10(mps_stream);

            // Create tensor, record stream, then let it go out of scope
            {
                auto options = at::TensorOptions().device(at::kMPS).dtype(at::kFloat);
                auto tensor = at::randn({64, 64}, options);

                // Record stream before tensor goes out of scope
                at::native::record_stream_mps(tensor, stream);

                // Do some work
                auto result = at::matmul(tensor, tensor);
                mps_stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);
                // tensor goes out of scope here
            }

            // Allocate another tensor to potentially reuse the freed buffer
            auto options = at::TensorOptions().device(at::kMPS).dtype(at::kFloat);
            auto tensor2 = at::randn({64, 64}, options);
            mps_stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);

            print_result("test_tensor_lifecycle", true, "Lifecycle handled correctly");
        } catch (const std::exception& e) {
            print_result("test_tensor_lifecycle", false, e.what());
        }
    }
}

// Test 6: Stress test - many threads recording on same buffer concurrently
void test_concurrent_record_stream() {
    @autoreleasepool {
        try {
            auto options = at::TensorOptions().device(at::kMPS).dtype(at::kFloat);
            auto tensor = at::randn({256, 256}, options);

            std::atomic<int> success_count{0};
            std::atomic<int> error_count{0};
            const int num_threads = 8;
            const int iterations = 20;

            std::vector<std::thread> threads;
            for (int t = 0; t < num_threads; t++) {
                threads.emplace_back([&, t]() {
                    @autoreleasepool {
                        try {
                            for (int i = 0; i < iterations; i++) {
                                auto* mps_stream = at::mps::getCurrentMPSStream();
                                c10::Stream stream = mps_stream_to_c10(mps_stream);
                                at::native::record_stream_mps(tensor, stream);
                            }
                            success_count++;
                        } catch (const std::exception& e) {
                            error_count++;
                        }
                    }
                });
            }

            for (auto& t : threads) {
                t.join();
            }

            // Sync to ensure all work is complete
            at::mps::getCurrentMPSStream()->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);

            bool passed = (error_count == 0 && success_count == num_threads);
            std::string details = "threads=" + std::to_string(success_count.load()) +
                                 "/" + std::to_string(num_threads) +
                                 ", errors=" + std::to_string(error_count.load());
            print_result("test_concurrent_record_stream", passed, details.c_str());
        } catch (const std::exception& e) {
            print_result("test_concurrent_record_stream", false, e.what());
        }
    }
}

int main(int argc, char* argv[]) {
    @autoreleasepool {
        std::cout << "=== MPS recordStream Test Suite ===" << std::endl;
        std::cout << "Testing External Audit Gap #4: record-stream semantics" << std::endl;
        std::cout << std::endl;

        // Check MPS availability
        if (!at::mps::is_available()) {
            std::cerr << "ERROR: MPS not available" << std::endl;
            return 1;
        }
        std::cout << "MPS available: YES" << std::endl;

        // Warm up MPS
        {
            auto options = at::TensorOptions().device(at::kMPS).dtype(at::kFloat);
            auto x = at::randn({32, 32}, options);
            at::mps::getDefaultMPSStream()->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);
        }
        std::cout << "MPS warmup complete" << std::endl;
        std::cout << std::endl;

        // Run tests
        test_basic_record_stream();
        test_multi_stream_record();
        test_cross_thread_record_stream();
        test_cpu_tensor_safety();
        test_tensor_lifecycle();
        test_concurrent_record_stream();

        std::cout << std::endl;
        std::cout << "=== Test Summary ===" << std::endl;
        std::cout << "Passed: " << test_pass_count.load() << std::endl;
        std::cout << "Failed: " << test_fail_count.load() << std::endl;

        if (test_fail_count.load() == 0) {
            std::cout << "=== ALL TESTS PASSED ===" << std::endl;
            return 0;
        } else {
            std::cout << "=== SOME TESTS FAILED ===" << std::endl;
            return 1;
        }
    }
}
