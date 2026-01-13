// minimal_metal_threading_test.mm
// Minimal reproduction to isolate Metal threading behavior
// This test uses ONLY Metal, no Python, no PyTorch
//
// Created by Andrew Yates
//
// Purpose: Rule out Python GIL and PyTorch as causes of serialization
//
// Build: clang++ -std=c++17 -framework Metal -framework Foundation \
//        -o minimal_metal_test minimal_metal_threading_test.mm
//
// Run: ./minimal_metal_test

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <iostream>
#include <iomanip>

// Simple compute kernel that does some work
static const char* kernelSource = R"(
#include <metal_stdlib>
using namespace metal;

kernel void simple_compute(
    device float* data [[buffer(0)]],
    uint gid [[thread_position_in_grid]]
) {
    // Do some computation to keep GPU busy
    float val = data[gid];
    for (int i = 0; i < 100; i++) {
        val = sin(val) * cos(val) + val;
    }
    data[gid] = val;
}
)";

struct ThreadResult {
    int thread_id;
    int operations;
    double elapsed_ms;
    double ops_per_sec;
};

void worker_thread(
    id<MTLDevice> device,
    id<MTLComputePipelineState> pipeline,
    int thread_id,
    int iterations,
    std::atomic<int>* completed,
    ThreadResult* result
) {
    @autoreleasepool {
        // Each thread gets its own command queue (this is key!)
        id<MTLCommandQueue> queue = [device newCommandQueue];

        // Create buffer for this thread
        const int dataSize = 1024 * 1024;  // 1M floats
        id<MTLBuffer> buffer = [device newBufferWithLength:dataSize * sizeof(float)
                                                   options:MTLResourceStorageModeShared];

        // Initialize data
        float* data = (float*)[buffer contents];
        for (int i = 0; i < dataSize; i++) {
            data[i] = (float)i * 0.001f;
        }

        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; i++) {
            @autoreleasepool {
                id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:buffer offset:0 atIndex:0];

                MTLSize gridSize = MTLSizeMake(dataSize, 1, 1);
                MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);

                [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
                [encoder endEncoding];

                [commandBuffer commit];
                [commandBuffer waitUntilCompleted];

                (*completed)++;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

        result->thread_id = thread_id;
        result->operations = iterations;
        result->elapsed_ms = elapsed_ms;
        result->ops_per_sec = (iterations * 1000.0) / elapsed_ms;
    }
}

void run_benchmark(int num_threads, int iterations_per_thread) {
    @autoreleasepool {
        // Get Metal device
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "Metal not available" << std::endl;
            return;
        }

        // Compile kernel
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:
            [NSString stringWithUTF8String:kernelSource]
            options:nil error:&error];

        if (error) {
            std::cerr << "Shader compile error: " << [[error description] UTF8String] << std::endl;
            return;
        }

        id<MTLFunction> function = [library newFunctionWithName:@"simple_compute"];
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];

        if (error) {
            std::cerr << "Pipeline error: " << [[error description] UTF8String] << std::endl;
            return;
        }

        // Run threads
        std::vector<std::thread> threads;
        std::vector<ThreadResult> results(num_threads);
        std::atomic<int> completed(0);

        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < num_threads; i++) {
            threads.emplace_back(worker_thread, device, pipeline, i,
                                 iterations_per_thread, &completed, &results[i]);
        }

        for (auto& t : threads) {
            t.join();
        }

        auto end = std::chrono::high_resolution_clock::now();
        double total_elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

        // Calculate results
        int total_ops = num_threads * iterations_per_thread;
        double total_ops_per_sec = (total_ops * 1000.0) / total_elapsed_ms;

        std::cout << "  Threads: " << num_threads
                  << "  Ops: " << total_ops
                  << "  Time: " << std::fixed << std::setprecision(1) << total_elapsed_ms << "ms"
                  << "  Ops/s: " << std::setprecision(1) << total_ops_per_sec
                  << std::endl;
    }
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "MINIMAL METAL THREADING TEST" << std::endl;
    std::cout << "No Python, No PyTorch - Pure Metal" << std::endl;
    std::cout << "========================================" << std::endl;

    // First, establish single-thread baseline
    std::cout << "\nSingle-thread baseline:" << std::endl;
    run_benchmark(1, 50);

    std::cout << "\nScaling test:" << std::endl;

    // Test scaling
    for (int threads : {1, 2, 4, 8, 16}) {
        run_benchmark(threads, 50);
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "If ops/s stays flat as threads increase," << std::endl;
    std::cout << "the serialization is in Metal, not Python." << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
