// bare_metal_throughput.mm
// Measure raw Metal command buffer submission rate to establish hardware ceiling.
// This is NOT a realistic workload - it measures the maximum possible command
// buffer submissions/second, which is the ceiling for any efficiency calculation.
//
// Created for Gap 6: "Maximum Efficiency" Claim Verification
//
// Build:
//   clang++ -std=c++17 -O2 -Wall -Wextra -fobjc-arc -x objective-c++ \
//     -framework Foundation -framework Metal \
//     -o tests/build/bare_metal_throughput tests/bare_metal_throughput.mm
//
// Or:
//   ./tests/build_cpp_tests.sh bare_metal_throughput
//
// Run:
//   ./tests/build/bare_metal_throughput

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

struct BenchmarkResult {
    int threads;
    int total_ops;
    double elapsed_ms;
    double ops_per_sec;
    double speedup_vs_1t;
};

// Minimal kernel - just writes thread index to prove GPU executed
static const char* minimalKernelSource = R"(
#include <metal_stdlib>
using namespace metal;

kernel void minimal_kernel(
    device uint* output [[buffer(0)]],
    uint gid [[thread_position_in_grid]]
) {
    output[gid] = gid;
}
)";

struct BenchmarkConfig {
    int warmup_iters = 10;
    int test_iters = 200;
    bool empty_buffers = false;  // true = empty command buffers, false = minimal compute
    bool per_thread_queue = true;
    bool async_mode = false;
    int max_inflight = 4;
};

static void print_usage(const char* argv0) {
    std::cout << "Usage: " << argv0 << " [options]\n\n"
              << "Options:\n"
              << "  --empty         Use empty command buffers (no GPU work)\n"
              << "  --minimal       Use minimal compute kernel (default)\n"
              << "  --shared-queue  Use single shared queue (vs per-thread)\n"
              << "  --async         Async submission (vs commit+wait)\n"
              << "  --iters <n>     Iterations per thread (default: 200)\n"
              << "  --warmup <n>    Warmup iterations (default: 10)\n"
              << "  --help          Show this help\n";
}

// Benchmark worker - empty command buffers
static void worker_empty_buffers(
    id<MTLDevice> device,
    id<MTLCommandQueue> shared_queue,
    const BenchmarkConfig& config,
    int thread_id,
    std::atomic<int>* completed
) {
    @autoreleasepool {
        id<MTLCommandQueue> queue = shared_queue;
        if (!queue) {
            queue = [device newCommandQueue];
        }

        // Warmup
        for (int i = 0; i < config.warmup_iters; i++) {
            @autoreleasepool {
                id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
                [cmdBuf commit];
                [cmdBuf waitUntilCompleted];
            }
        }

        // Timed run
        for (int i = 0; i < config.test_iters; i++) {
            @autoreleasepool {
                id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
                [cmdBuf commit];
                if (!config.async_mode) {
                    [cmdBuf waitUntilCompleted];
                }
                (*completed)++;
            }
        }

        // Wait for any outstanding if async
        if (config.async_mode) {
            @autoreleasepool {
                id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
                [cmdBuf commit];
                [cmdBuf waitUntilCompleted];
            }
        }
    }
}

// Benchmark worker - minimal compute
static void worker_minimal_compute(
    id<MTLDevice> device,
    id<MTLCommandQueue> shared_queue,
    id<MTLComputePipelineState> pipeline,
    const BenchmarkConfig& config,
    int thread_id,
    std::atomic<int>* completed
) {
    @autoreleasepool {
        id<MTLCommandQueue> queue = shared_queue;
        if (!queue) {
            queue = [device newCommandQueue];
        }

        // Small buffer - just 1KB
        id<MTLBuffer> buffer = [device newBufferWithLength:1024 options:MTLResourceStorageModeShared];

        // Warmup
        for (int i = 0; i < config.warmup_iters; i++) {
            @autoreleasepool {
                id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:buffer offset:0 atIndex:0];
                [encoder dispatchThreads:MTLSizeMake(256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [encoder endEncoding];
                [cmdBuf commit];
                [cmdBuf waitUntilCompleted];
            }
        }

        // Timed run
        std::vector<id<MTLCommandBuffer>> inflight;
        for (int i = 0; i < config.test_iters; i++) {
            @autoreleasepool {
                id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:buffer offset:0 atIndex:0];
                [encoder dispatchThreads:MTLSizeMake(256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [encoder endEncoding];
                [cmdBuf commit];

                if (config.async_mode) {
                    inflight.push_back(cmdBuf);
                    if ((int)inflight.size() >= config.max_inflight) {
                        [inflight.front() waitUntilCompleted];
                        inflight.erase(inflight.begin());
                    }
                } else {
                    [cmdBuf waitUntilCompleted];
                }
                (*completed)++;
            }
        }

        // Drain inflight
        for (auto& cb : inflight) {
            [cb waitUntilCompleted];
        }
    }
}

static BenchmarkResult run_benchmark(
    id<MTLDevice> device,
    id<MTLComputePipelineState> pipeline,
    const BenchmarkConfig& config,
    int num_threads
) {
    @autoreleasepool {
        std::atomic<int> completed(0);
        std::vector<std::thread> threads;

        id<MTLCommandQueue> shared_queue = nil;
        if (!config.per_thread_queue) {
            shared_queue = [device newCommandQueue];
        }

        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < num_threads; i++) {
            if (config.empty_buffers) {
                threads.emplace_back(worker_empty_buffers, device, shared_queue, config, i, &completed);
            } else {
                threads.emplace_back(worker_minimal_compute, device, shared_queue, pipeline, config, i, &completed);
            }
        }

        for (auto& t : threads) {
            t.join();
        }

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        int total_ops = num_threads * config.test_iters;
        double ops_per_sec = (total_ops * 1000.0) / elapsed_ms;

        BenchmarkResult result;
        result.threads = num_threads;
        result.total_ops = total_ops;
        result.elapsed_ms = elapsed_ms;
        result.ops_per_sec = ops_per_sec;
        result.speedup_vs_1t = 1.0;  // Set later
        return result;
    }
}

int main(int argc, char** argv) {
    BenchmarkConfig config;

    for (int i = 1; i < argc; i++) {
        const char* arg = argv[i];
        if (std::strcmp(arg, "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        if (std::strcmp(arg, "--empty") == 0) {
            config.empty_buffers = true;
            continue;
        }
        if (std::strcmp(arg, "--minimal") == 0) {
            config.empty_buffers = false;
            continue;
        }
        if (std::strcmp(arg, "--shared-queue") == 0) {
            config.per_thread_queue = false;
            continue;
        }
        if (std::strcmp(arg, "--async") == 0) {
            config.async_mode = true;
            continue;
        }
        if (std::strcmp(arg, "--iters") == 0 && i + 1 < argc) {
            config.test_iters = std::atoi(argv[++i]);
            continue;
        }
        if (std::strcmp(arg, "--warmup") == 0 && i + 1 < argc) {
            config.warmup_iters = std::atoi(argv[++i]);
            continue;
        }
        std::cerr << "Unknown argument: " << arg << "\n";
        return 2;
    }

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "Metal not available\n";
            return 1;
        }

        // Compile minimal kernel
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:minimalKernelSource]
                                                      options:nil
                                                        error:&error];
        if (error) {
            std::cerr << "Shader compile error: " << [[error description] UTF8String] << "\n";
            return 1;
        }

        id<MTLFunction> function = [library newFunctionWithName:@"minimal_kernel"];
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (error) {
            std::cerr << "Pipeline error: " << [[error description] UTF8String] << "\n";
            return 1;
        }

        std::cout << "========================================\n";
        std::cout << "BARE METAL THROUGHPUT BASELINE\n";
        std::cout << "Gap 6: Maximum Efficiency Verification\n";
        std::cout << "========================================\n";
        std::cout << "Device: " << [[device name] UTF8String] << "\n";
        std::cout << "Mode: " << (config.empty_buffers ? "Empty command buffers" : "Minimal compute (256 threads)") << "\n";
        std::cout << "Queue: " << (config.per_thread_queue ? "Per-thread queue" : "Shared queue") << "\n";
        std::cout << "Submit: " << (config.async_mode ? "Async" : "Commit+Wait") << "\n";
        std::cout << "Iterations/thread: " << config.test_iters << "\n";
        std::cout << "Warmup: " << config.warmup_iters << "\n";
        std::cout << "========================================\n\n";

        std::vector<int> thread_counts = {1, 2, 4, 8};
        std::vector<BenchmarkResult> results;

        double baseline_ops = 0;

        for (int threads : thread_counts) {
            BenchmarkResult r = run_benchmark(device, pipeline, config, threads);
            if (threads == 1) {
                baseline_ops = r.ops_per_sec;
            }
            r.speedup_vs_1t = r.ops_per_sec / baseline_ops;
            results.push_back(r);

            std::cout << "Threads: " << std::setw(2) << r.threads
                      << "  Ops: " << std::setw(5) << r.total_ops
                      << "  Time: " << std::fixed << std::setprecision(1) << std::setw(8) << r.elapsed_ms << "ms"
                      << "  Ops/s: " << std::setprecision(1) << std::setw(10) << r.ops_per_sec
                      << "  Efficiency: " << std::setprecision(1) << std::setw(6) << (r.speedup_vs_1t * 100.0 / r.threads) << "%"
                      << "  Speedup: " << std::setprecision(2) << r.speedup_vs_1t << "x\n";
        }

        std::cout << "\n========================================\n";
        std::cout << "ANALYSIS\n";
        std::cout << "========================================\n";

        double max_ops = results.back().ops_per_sec;
        double single_t_ops = results.front().ops_per_sec;
        double max_speedup = max_ops / single_t_ops;
        int max_threads = results.back().threads;
        double efficiency_at_max = (max_speedup * 100.0) / max_threads;

        std::cout << "Hardware ceiling (raw " << (config.empty_buffers ? "empty" : "minimal") << " submission):\n";
        std::cout << "  Single-thread baseline: " << std::fixed << std::setprecision(0) << single_t_ops << " ops/s\n";
        std::cout << "  " << max_threads << "-thread throughput: " << max_ops << " ops/s\n";
        std::cout << "  Speedup at " << max_threads << "t: " << std::setprecision(2) << max_speedup << "x\n";
        std::cout << "  Efficiency at " << max_threads << "t: " << std::setprecision(1) << efficiency_at_max << "%\n";
        std::cout << "\nInterpretation:\n";
        if (efficiency_at_max < 50) {
            std::cout << "  Metal command submission itself has <50% parallel efficiency.\n";
            std::cout << "  This is the HARDWARE CEILING - PyTorch cannot exceed this.\n";
        } else {
            std::cout << "  Metal command submission scales well (" << efficiency_at_max << "% efficiency).\n";
            std::cout << "  PyTorch serialization must be due to PyTorch/MPS overhead.\n";
        }
        std::cout << "========================================\n";

        // Output machine-readable summary
        std::cout << "\n# Machine-readable summary for Gap 6:\n";
        std::cout << "BARE_METAL_1T_OPS=" << std::fixed << std::setprecision(0) << single_t_ops << "\n";
        std::cout << "BARE_METAL_8T_OPS=" << max_ops << "\n";
        std::cout << "BARE_METAL_8T_EFFICIENCY=" << std::setprecision(1) << efficiency_at_max << "\n";
    }

    return 0;
}
