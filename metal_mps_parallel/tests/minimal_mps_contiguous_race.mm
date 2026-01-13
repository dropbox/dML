/**
 * Minimal C++ reproduction: MPS .contiguous() race condition
 *
 * This test demonstrates the race condition in PyTorch's MPS backend when
 * .contiguous() is called on complex reshaped tensors from multiple threads.
 *
 * This C++ version helps isolate whether the race is in:
 * - Python layer (PyTorch Python bindings)
 * - ATen layer (C++ tensor operations)
 * - MPS backend (Metal/MPS framework)
 *
 * Build:
 *   cd pytorch-mps-fork && mkdir -p build && cd build
 *   cmake -DCMAKE_PREFIX_PATH=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)') ..
 *   make minimal_mps_contiguous_race
 *
 * Run:
 *   ./tests/minimal_mps_contiguous_race
 *
 * Expected behavior:
 * - Test WITHOUT .contiguous(): should pass 100%
 * - Test WITH .contiguous(): may fail intermittently (~10-30% failure rate)
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <ATen/ATen.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/core/grad_mode.h>
#include <c10/core/Device.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

// Test configuration
constexpr int NUM_ITERATIONS = 30;
constexpr int NUM_THREADS = 8;
constexpr int EMBED_DIM = 256;
constexpr int BATCH_SIZE = 4;
constexpr int SEQ_LEN = 128;
constexpr int NUM_HEADS = 4;
constexpr int HEAD_DIM = EMBED_DIM / NUM_HEADS;
constexpr float TOLERANCE = 1e-3f;

std::mutex g_cout_mutex;

void print_sync(const std::string& msg) {
    std::lock_guard<std::mutex> lock(g_cout_mutex);
    std::cout << msg << std::endl;
}

/**
 * Projection operation that mimics PyTorch's _in_projection_packed pattern.
 *
 * 1. Linear projection (weight @ x + bias)
 * 2. Reshape to separate Q, K, V
 * 3. Optionally call .contiguous() (the race trigger)
 * 4. Return mean for result comparison
 */
at::Tensor projection_op(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    bool use_contiguous
) {
    // Step 1: Linear projection -> [batch, seq_len, 3*embed_dim]
    auto proj = at::linear(input, weight, bias);

    // Step 2: PyTorch's reshape pattern (from _in_projection_packed)
    // unflatten(-1, (3, embed_dim)) -> [batch, seq_len, 3, embed_dim]
    proj = proj.unflatten(-1, {3, EMBED_DIM});

    // unsqueeze(0) -> [1, batch, seq_len, 3, embed_dim]
    proj = proj.unsqueeze(0);

    // transpose(0, -2) -> [3, batch, seq_len, 1, embed_dim]
    proj = proj.transpose(0, -2);

    // squeeze(-2) -> [3, batch, seq_len, embed_dim]
    proj = proj.squeeze(-2);

    // Step 3: The race condition trigger
    if (use_contiguous) {
        proj = proj.contiguous();  // <-- RACE CONDITION HERE
    }

    // Step 4: Extract Q, K, V and reshape for multi-head attention
    auto q = proj.select(0, 0);  // [batch, seq_len, embed_dim]
    auto k = proj.select(0, 1);
    auto v = proj.select(0, 2);

    // Reshape for multi-head: [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
    if (use_contiguous) {
        // .view() requires contiguous tensor
        q = q.view({BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM}).transpose(1, 2);
        k = k.view({BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM}).transpose(1, 2);
        v = v.view({BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM}).transpose(1, 2);
    } else {
        // .reshape() handles non-contiguous tensors
        q = q.reshape({BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM}).transpose(1, 2);
        k = k.reshape({BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM}).transpose(1, 2);
        v = v.reshape({BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM}).transpose(1, 2);
    }

    // Step 5: Simple scaled dot-product attention (matmul + softmax + matmul)
    // This mimics the SDPA path in PyTorch
    auto scale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
    auto scores = at::matmul(q, k.transpose(-2, -1)) * scale;
    auto attn_weights = at::softmax(scores, -1);
    auto output = at::matmul(attn_weights, v);

    // Return mean for simple comparison
    return output.mean();
}

/**
 * Run the race condition test.
 *
 * @param use_contiguous If true, call .contiguous() in projection_op
 * @return Tuple of (passed_iterations, total_iterations, max_difference)
 */
std::tuple<int, int, float> run_test(bool use_contiguous) {
    @autoreleasepool {
        // Create weight matrices on MPS
        at::manual_seed(42);
        auto options = at::TensorOptions().device(at::kMPS).dtype(at::kFloat);
        auto weight = at::randn({3 * EMBED_DIM, EMBED_DIM}, options);
        auto bias = at::randn({3 * EMBED_DIM}, options);
        at::mps::getDefaultMPSStream()->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);

        int passed = 0;
        float max_diff = 0.0f;

        for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
            // Create unique inputs for each thread
            std::vector<at::Tensor> inputs;
            for (int tid = 0; tid < NUM_THREADS; tid++) {
                at::manual_seed(iter * 1000 + tid);
                inputs.push_back(at::randn({BATCH_SIZE, SEQ_LEN, EMBED_DIM}, options));
            }
            at::mps::getDefaultMPSStream()->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);

            // Compute expected results serially
            std::vector<at::Tensor> expected;
            {
                at::NoGradGuard no_grad;
                for (int tid = 0; tid < NUM_THREADS; tid++) {
                    expected.push_back(projection_op(inputs[tid], weight, bias, use_contiguous).clone());
                }
            }
            at::mps::getDefaultMPSStream()->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);

            // Run in parallel
            std::vector<at::Tensor> results(NUM_THREADS);
            std::vector<std::thread> threads;
            std::atomic<int> thread_errors{0};

            for (int tid = 0; tid < NUM_THREADS; tid++) {
                threads.emplace_back([&, tid]() {
                    @autoreleasepool {
                        try {
                            at::NoGradGuard no_grad;
                            results[tid] = projection_op(inputs[tid], weight, bias, use_contiguous);
                            at::mps::getCurrentMPSStream()->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);
                        } catch (const std::exception& e) {
                            thread_errors++;
                        }
                    }
                });
            }

            // Wait for all threads
            for (auto& t : threads) {
                t.join();
            }

            // Check results
            bool iteration_ok = (thread_errors == 0);
            for (int tid = 0; tid < NUM_THREADS && iteration_ok; tid++) {
                if (!results[tid].defined()) {
                    iteration_ok = false;
                    continue;
                }

                // Compare on CPU for reliability
                auto expected_cpu = expected[tid].cpu();
                auto result_cpu = results[tid].cpu();
                float diff = (result_cpu - expected_cpu).abs().item<float>();
                max_diff = std::max(max_diff, diff);

                if (diff > TOLERANCE) {
                    iteration_ok = false;
                }
            }

            if (iteration_ok) {
                passed++;
            }
        }

        return {passed, NUM_ITERATIONS, max_diff};
    }
}

int main(int /* argc */, char* /* argv */[]) {
    @autoreleasepool {
        std::cout << std::string(70, '=') << std::endl;
        std::cout << "MPS .contiguous() Race Condition - C++ Reproduction" << std::endl;
        std::cout << std::string(70, '=') << std::endl;

        // Check MPS availability
        if (!at::mps::is_available()) {
            std::cerr << "ERROR: MPS not available" << std::endl;
            return 1;
        }
        std::cout << "MPS available: YES" << std::endl;
        std::cout << "Configuration:" << std::endl;
        std::cout << "  Threads: " << NUM_THREADS << std::endl;
        std::cout << "  Iterations: " << NUM_ITERATIONS << std::endl;
        std::cout << "  Embed dim: " << EMBED_DIM << std::endl;
        std::cout << "  Batch size: " << BATCH_SIZE << std::endl;
        std::cout << "  Sequence length: " << SEQ_LEN << std::endl;
        std::cout << std::endl;

        // Warm up MPS
        {
            auto options = at::TensorOptions().device(at::kMPS).dtype(at::kFloat);
            auto x = at::randn({32, 32}, options);
            auto y = at::matmul(x, x);
            at::mps::getDefaultMPSStream()->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);
        }
        std::cout << "MPS warmup complete" << std::endl;
        std::cout << std::endl;

        // Test 1: WITHOUT .contiguous() - should always pass
        std::cout << "Test 1: WITHOUT .contiguous() (expected: PASS)" << std::endl;
        auto [passed1, total1, diff1] = run_test(false);
        std::string status1 = (passed1 == total1) ? "PASS" : "FAIL";
        std::cout << "  Result: " << status1 << " (" << passed1 << "/" << total1
                  << "), max_diff=" << std::scientific << std::setprecision(2) << diff1 << std::endl;
        std::cout << std::endl;

        // Test 2: WITH .contiguous() - demonstrates the race
        std::cout << "Test 2: WITH .contiguous() (demonstrates race condition)" << std::endl;
        auto [passed2, total2, diff2] = run_test(true);
        std::string status2 = (passed2 == total2) ? "PASS" : "FAIL";
        std::cout << "  Result: " << status2 << " (" << passed2 << "/" << total2
                  << "), max_diff=" << std::scientific << std::setprecision(2) << diff2 << std::endl;
        std::cout << std::endl;

        // Summary
        std::cout << std::string(70, '=') << std::endl;
        std::cout << "SUMMARY" << std::endl;
        std::cout << std::string(70, '=') << std::endl;

        if (status1 == "PASS" && status2 == "FAIL") {
            std::cout << "BUG REPRODUCED at C++ level: .contiguous() triggers race condition" << std::endl;
            std::cout << std::endl;
            std::cout << "This confirms the race is in ATen/MPS layer, not Python bindings." << std::endl;
            std::cout << "Root cause: Concurrent .contiguous() calls on complex strided tensors" << std::endl;
            std::cout << "trigger race conditions in MPS memory allocation or copy operations." << std::endl;
            return 1;
        } else if (status1 == "PASS" && status2 == "PASS") {
            std::cout << "Bug not reproduced this run (race is intermittent)" << std::endl;
            std::cout << "C++ layer appears stable - race may be specific to Python threading" << std::endl;
            std::cout << "or requires more aggressive parallelism to trigger." << std::endl;
            return 0;
        } else if (status1 == "FAIL") {
            std::cout << "Unexpected: Test WITHOUT .contiguous() failed" << std::endl;
            std::cout << "This indicates a different issue (not the .contiguous() race)" << std::endl;
            return 2;
        } else {
            std::cout << "Both tests failed - fundamental MPS issue" << std::endl;
            return 3;
        }
    }
}
