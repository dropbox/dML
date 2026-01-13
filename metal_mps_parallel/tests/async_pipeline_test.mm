// async_pipeline_test.mm
// Task 8.4: Test async command buffer pipelining for throughput improvement
//
// Tests whether pipelining multiple command buffers without waiting improves
// throughput compared to synchronous (commit+wait) submission.
//
// Build:
//   clang++ -std=c++17 -O2 -Wall -Wextra -fobjc-arc -x objective-c++ \
//     -framework Foundation -framework Metal \
//     -o async_pipeline_test async_pipeline_test.mm
//
// Run:
//   ./async_pipeline_test

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

// Configuration
static constexpr int DATA_SIZE = 65536;      // Light workload for pipelining test
static constexpr int KERNEL_ITERS = 10;      // Inner kernel iterations
static constexpr int TOTAL_OPS = 500;        // Total operations per test
static constexpr int PIPELINE_DEPTHS[] = {1, 2, 4, 8, 16, 32};
static constexpr int WARMUP_OPS = 10;

static std::string makeKernelSource(int innerIterations) {
    return std::string(R"(
#include <metal_stdlib>
using namespace metal;

#define INNER_ITERS )") + std::to_string(innerIterations) + R"(

kernel void compute_kernel(
    device float* data [[buffer(0)]],
    uint gid [[thread_position_in_grid]]
) {
    float val = data[gid];
    for (int i = 0; i < INNER_ITERS; i++) {
        val = sin(val) * cos(val) + val;
    }
    data[gid] = val;
}
)";
}

struct BenchmarkResult {
    int pipelineDepth;
    double elapsedMs;
    double opsPerSec;
    double speedup;  // vs sync (depth=1)
};

// Synchronous submission: commit + wait for each command buffer
static double runSync(id<MTLDevice> device, id<MTLComputePipelineState> pipeline,
                      id<MTLCommandQueue> queue, int totalOps) {
    @autoreleasepool {
        id<MTLBuffer> buffer = [device newBufferWithLength:DATA_SIZE * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
        float* data = static_cast<float*>([buffer contents]);
        for (int i = 0; i < DATA_SIZE; i++) {
            data[i] = static_cast<float>(i) * 1e-6f;
        }

        auto start = std::chrono::high_resolution_clock::now();

        for (int op = 0; op < totalOps; op++) {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:buffer offset:0 atIndex:0];
            MTLSize gridSize = MTLSizeMake(DATA_SIZE, 1, 1);
            MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
            [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
            [encoder endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }

        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
}

// Async pipelining: submit N command buffers before waiting for oldest
static double runAsync(id<MTLDevice> device, id<MTLComputePipelineState> pipeline,
                       id<MTLCommandQueue> queue, int totalOps, int pipelineDepth) {
    @autoreleasepool {
        id<MTLBuffer> buffer = [device newBufferWithLength:DATA_SIZE * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
        float* data = static_cast<float*>([buffer contents]);
        for (int i = 0; i < DATA_SIZE; i++) {
            data[i] = static_cast<float>(i) * 1e-6f;
        }

        // Use NSMutableArray to hold command buffers (avoids ARC issues)
        NSMutableArray<id<MTLCommandBuffer>>* pipeline_cbs =
            [NSMutableArray arrayWithCapacity:static_cast<NSUInteger>(pipelineDepth)];

        auto start = std::chrono::high_resolution_clock::now();

        for (int op = 0; op < totalOps; op++) {
            // Create and submit new command buffer
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:buffer offset:0 atIndex:0];
            MTLSize gridSize = MTLSizeMake(DATA_SIZE, 1, 1);
            MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
            [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
            [encoder endEncoding];
            [cb commit];
            [pipeline_cbs addObject:cb];

            // If we've filled the pipeline, wait for oldest
            if (static_cast<int>([pipeline_cbs count]) >= pipelineDepth) {
                [[pipeline_cbs objectAtIndex:0] waitUntilCompleted];
                [pipeline_cbs removeObjectAtIndex:0];
            }
        }

        // Wait for remaining in-flight command buffers
        for (id<MTLCommandBuffer> cb in pipeline_cbs) {
            [cb waitUntilCompleted];
        }

        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
}

// Multi-threaded async pipelining
static double runMultiThreadedAsync(id<MTLDevice> device, id<MTLComputePipelineState> pipeline,
                                    int totalOps, int numThreads, int pipelineDepth) {
    int opsPerThread = totalOps / numThreads;
    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(numThreads));

    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < numThreads; t++) {
        threads.emplace_back([&, t]() {
            @autoreleasepool {
                // Per-thread queue for maximum parallelism
                id<MTLCommandQueue> queue = [device newCommandQueue];

                id<MTLBuffer> buffer = [device newBufferWithLength:DATA_SIZE * sizeof(float)
                                                           options:MTLResourceStorageModeShared];
                float* data = static_cast<float*>([buffer contents]);
                for (int i = 0; i < DATA_SIZE; i++) {
                    data[i] = static_cast<float>(t * 0.001 + i) * 1e-6f;
                }

                NSMutableArray<id<MTLCommandBuffer>>* pipeline_cbs =
                    [NSMutableArray arrayWithCapacity:static_cast<NSUInteger>(pipelineDepth)];

                for (int op = 0; op < opsPerThread; op++) {
                    id<MTLCommandBuffer> cb = [queue commandBuffer];
                    id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
                    [encoder setComputePipelineState:pipeline];
                    [encoder setBuffer:buffer offset:0 atIndex:0];
                    MTLSize gridSize = MTLSizeMake(DATA_SIZE, 1, 1);
                    MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
                    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
                    [encoder endEncoding];
                    [cb commit];
                    [pipeline_cbs addObject:cb];

                    if (static_cast<int>([pipeline_cbs count]) >= pipelineDepth) {
                        [[pipeline_cbs objectAtIndex:0] waitUntilCompleted];
                        [pipeline_cbs removeObjectAtIndex:0];
                    }
                }

                for (id<MTLCommandBuffer> cb in pipeline_cbs) {
                    [cb waitUntilCompleted];
                }
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "Metal not available\n";
            return 1;
        }

        std::string source = makeKernelSource(KERNEL_ITERS);
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                                      options:nil
                                                        error:&error];
        if (error) {
            std::cerr << "Shader compile error: " << [[error description] UTF8String] << "\n";
            return 1;
        }

        id<MTLFunction> function = [library newFunctionWithName:@"compute_kernel"];
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                                                                                     error:&error];
        if (error) {
            std::cerr << "Pipeline error: " << [[error description] UTF8String] << "\n";
            return 1;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];

        std::cout << "========================================\n";
        std::cout << "ASYNC COMMAND BUFFER PIPELINING TEST\n";
        std::cout << "Task 8.4: Test if async submission improves throughput\n";
        std::cout << "Device: " << [[device name] UTF8String] << "\n";
        std::cout << "Config: data=" << DATA_SIZE << " kernel-iters=" << KERNEL_ITERS
                  << " total-ops=" << TOTAL_OPS << "\n";
        std::cout << "========================================\n\n";

        // Warmup
        std::cout << "Warming up...\n";
        runSync(device, pipeline, queue, WARMUP_OPS);

        // Test single-threaded pipelining
        std::cout << "\n=== SINGLE-THREADED PIPELINING ===\n";
        std::cout << "Testing different pipeline depths...\n\n";

        double syncElapsedMs = runSync(device, pipeline, queue, TOTAL_OPS);
        double syncOpsPerSec = (static_cast<double>(TOTAL_OPS) * 1000.0) / syncElapsedMs;
        int bestSingleDepth = 1;
        double bestSingleOpsPerSecMeasured = syncOpsPerSec;
        std::cout << "Sync (depth=1):  " << std::fixed << std::setprecision(1)
                  << std::setw(7) << syncElapsedMs << " ms  "
                  << std::setw(8) << syncOpsPerSec << " ops/s  (baseline)\n";

        for (int depth : PIPELINE_DEPTHS) {
            if (depth == 1) continue;  // Already tested as sync

            double elapsedMs = runAsync(device, pipeline, queue, TOTAL_OPS, depth);
            double opsPerSec = (static_cast<double>(TOTAL_OPS) * 1000.0) / elapsedMs;
            double speedup = opsPerSec / syncOpsPerSec;

            std::cout << "Async (depth=" << std::setw(2) << depth << "): "
                      << std::setw(7) << elapsedMs << " ms  "
                      << std::setw(8) << opsPerSec << " ops/s  "
                      << std::setprecision(2) << speedup << "x speedup\n";

            if (opsPerSec > bestSingleOpsPerSecMeasured) {
                bestSingleOpsPerSecMeasured = opsPerSec;
                bestSingleDepth = depth;
            }
        }

        // Test multi-threaded pipelining
        std::cout << "\n=== MULTI-THREADED PIPELINING ===\n";
        std::cout << "Testing with 8 threads and varying pipeline depths...\n\n";

        int numThreads = 8;
        int totalMultiOps = TOTAL_OPS;

        // Baseline: 8 threads, sync (depth=1)
        double mt_syncElapsedMs = runMultiThreadedAsync(device, pipeline, totalMultiOps, numThreads, 1);
        double mt_syncOpsPerSec = (static_cast<double>(totalMultiOps) * 1000.0) / mt_syncElapsedMs;
        int bestMultiDepth = 1;
        double bestMultiOpsPerSecMeasured = mt_syncOpsPerSec;
        std::cout << "8T Sync (depth=1):  " << std::fixed << std::setprecision(1)
                  << std::setw(7) << mt_syncElapsedMs << " ms  "
                  << std::setw(8) << mt_syncOpsPerSec << " ops/s  (baseline)\n";

        for (int depth : {2, 4, 8}) {
            double elapsedMs = runMultiThreadedAsync(device, pipeline, totalMultiOps, numThreads, depth);
            double opsPerSec = (static_cast<double>(totalMultiOps) * 1000.0) / elapsedMs;
            double speedup = opsPerSec / mt_syncOpsPerSec;

            std::cout << "8T Async (depth=" << depth << "): "
                      << std::setw(7) << elapsedMs << " ms  "
                      << std::setw(8) << opsPerSec << " ops/s  "
                      << std::setprecision(2) << speedup << "x speedup\n";

            if (opsPerSec > bestMultiOpsPerSecMeasured) {
                bestMultiOpsPerSecMeasured = opsPerSec;
                bestMultiDepth = depth;
            }
        }

        // Summary
        std::cout << "\n========================================\n";
        std::cout << "SUMMARY\n";
        std::cout << "========================================\n";
        std::cout << "Success criteria: >10% throughput improvement\n\n";

        // Re-run best depth for final measurement
        double bestSingleMs = (bestSingleDepth == 1)
                                  ? runSync(device, pipeline, queue, TOTAL_OPS)
                                  : runAsync(device, pipeline, queue, TOTAL_OPS, bestSingleDepth);
        double bestSingleOps = (static_cast<double>(TOTAL_OPS) * 1000.0) / bestSingleMs;
        double singleImprovement = ((bestSingleOps / syncOpsPerSec) - 1.0) * 100.0;

        double bestMultiMs = runMultiThreadedAsync(device, pipeline, totalMultiOps, numThreads, bestMultiDepth);
        double bestMultiOps = (static_cast<double>(totalMultiOps) * 1000.0) / bestMultiMs;
        double multiImprovement = ((bestMultiOps / mt_syncOpsPerSec) - 1.0) * 100.0;

        std::cout << "Single-threaded (depth=" << bestSingleDepth << "): "
                  << std::setprecision(1) << syncOpsPerSec << " → " << bestSingleOps << " ops/s ("
                  << (singleImprovement >= 0 ? "+" : "") << singleImprovement << "%)\n";

        std::cout << "Multi-threaded  (depth=" << bestMultiDepth << "): "
                  << mt_syncOpsPerSec << " → " << bestMultiOps << " ops/s ("
                  << (multiImprovement >= 0 ? "+" : "") << multiImprovement << "%)\n\n";

        bool singleSuccess = singleImprovement >= 10.0;
        bool multiSuccess = multiImprovement >= 10.0;

        std::cout << "Single-threaded async pipelining: "
                  << (singleSuccess ? "PASS" : "FAIL") << " (>10% improvement)\n";
        std::cout << "Multi-threaded async pipelining:  "
                  << (multiSuccess ? "PASS" : "FAIL") << " (>10% improvement)\n";

        return (singleSuccess || multiSuccess) ? 0 : 1;
    }
}
