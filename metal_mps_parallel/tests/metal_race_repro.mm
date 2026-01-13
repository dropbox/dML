/**
 * Minimal Metal Reproduction: AGX Driver Race Condition
 *
 * This standalone test demonstrates a race condition in Apple's AGX Metal driver
 * that causes crashes when multiple threads concurrently use compute command encoders.
 *
 * BUILD:
 *   clang++ -std=c++17 -framework Metal -framework Foundation \
 *     -o metal_race_repro metal_race_repro.mm
 *
 * RUN (expect crash within seconds):
 *   ./metal_race_repro
 *
 * CRASH SITES (all in Apple's AGXMetalG16X driver):
 *   1. -[AGXG16XFamilyComputeContext setComputePipelineState:] at offset 0x5c8
 *   2. AGX::ComputeContext::prepareForEnqueue at offset 0x98
 *   3. AGX::SpillInfoGen3::allocateUSCSpillBuffer at offset 0x184
 *
 * The crash occurs because the AGX driver has internal state (ContextCommon)
 * that is NOT thread-safe. When Thread A calls setComputePipelineState: while
 * Thread B is also accessing the same internal context, the context pointer
 * can become NULL, causing a SIGSEGV.
 *
 * Author: Andrew Yates
 * Part of MPS Parallel Inference Research Project
 * https://github.com/ayates_dbx/metal_mps_parallel
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <thread>
#import <atomic>
#import <vector>
#import <iostream>
#import <chrono>

// Configuration
constexpr int NUM_THREADS = 8;
constexpr int OPS_PER_THREAD = 100;
constexpr int NUM_ITERATIONS = 1000;
constexpr int BUFFER_SIZE = 1024 * 1024;  // 1MB buffers

// Statistics
std::atomic<uint64_t> g_completed_ops{0};
std::atomic<bool> g_crashed{false};

// Metal Shader Library (simple copy kernel)
const char* g_compute_shader = R"(
#include <metal_stdlib>
using namespace metal;

kernel void copy_kernel(
    device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    output[id] = input[id];
}

kernel void add_kernel(
    device float* input1 [[buffer(0)]],
    device float* input2 [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    output[id] = input1[id] + input2[id];
}
)";

// Create compute pipeline state from shader source
id<MTLComputePipelineState> createPipeline(id<MTLDevice> device, const char* functionName) {
    NSError* error = nil;

    // Compile shader
    NSString* source = [NSString stringWithUTF8String:g_compute_shader];
    id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
    if (!library) {
        std::cerr << "Failed to compile shader: " << [[error localizedDescription] UTF8String] << std::endl;
        return nil;
    }

    // Get function
    NSString* funcName = [NSString stringWithUTF8String:functionName];
    id<MTLFunction> function = [library newFunctionWithName:funcName];
    if (!function) {
        std::cerr << "Failed to find function: " << functionName << std::endl;
        return nil;
    }

    // Create pipeline
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) {
        std::cerr << "Failed to create pipeline: " << [[error localizedDescription] UTF8String] << std::endl;
        return nil;
    }

    return pipeline;
}

// Worker thread function
void workerThread(int threadId, id<MTLDevice> device, id<MTLComputePipelineState> pipeline1, id<MTLComputePipelineState> pipeline2) {
    @autoreleasepool {
        // Create per-thread command queue
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) {
            std::cerr << "Thread " << threadId << ": Failed to create command queue" << std::endl;
            return;
        }

        // Create buffers
        id<MTLBuffer> buffer1 = [device newBufferWithLength:BUFFER_SIZE options:MTLResourceStorageModeShared];
        id<MTLBuffer> buffer2 = [device newBufferWithLength:BUFFER_SIZE options:MTLResourceStorageModeShared];
        id<MTLBuffer> buffer3 = [device newBufferWithLength:BUFFER_SIZE options:MTLResourceStorageModeShared];

        if (!buffer1 || !buffer2 || !buffer3) {
            std::cerr << "Thread " << threadId << ": Failed to create buffers" << std::endl;
            return;
        }

        // Initialize buffer data
        float* data = (float*)[buffer1 contents];
        for (int i = 0; i < BUFFER_SIZE / sizeof(float); i++) {
            data[i] = (float)i;
        }

        MTLSize gridSize = MTLSizeMake(BUFFER_SIZE / sizeof(float), 1, 1);
        MTLSize threadgroupSize = MTLSizeMake(256, 1, 1);

        for (int op = 0; op < OPS_PER_THREAD && !g_crashed; op++) {
            @autoreleasepool {
                // Create command buffer
                id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
                if (!commandBuffer) {
                    std::cerr << "Thread " << threadId << ": Failed to create command buffer" << std::endl;
                    continue;
                }

                // Create compute encoder
                id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
                if (!encoder) {
                    std::cerr << "Thread " << threadId << ": Failed to create encoder" << std::endl;
                    continue;
                }

                // CRITICAL: This is where the race occurs
                // Multiple threads calling setComputePipelineState: concurrently
                // can corrupt the AGX driver's internal ContextCommon state

                // Alternate between pipelines to stress the driver
                id<MTLComputePipelineState> pipeline = (op % 2 == 0) ? pipeline1 : pipeline2;
                [encoder setComputePipelineState:pipeline];

                [encoder setBuffer:buffer1 offset:0 atIndex:0];
                [encoder setBuffer:buffer2 offset:0 atIndex:1];

                if (pipeline == pipeline2) {
                    [encoder setBuffer:buffer3 offset:0 atIndex:2];
                }

                // Dispatch compute work
                [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

                // End encoding
                [encoder endEncoding];

                // Commit and optionally wait
                [commandBuffer commit];

                // Every few ops, wait for completion (mimics synchronize calls)
                if (op % 10 == 0) {
                    [commandBuffer waitUntilCompleted];
                }

                g_completed_ops++;
            }
        }
    }
}

int main(int argc, char* argv[]) {
    @autoreleasepool {
        std::cout << "=== Metal AGX Driver Race Condition Reproduction ===" << std::endl;
        std::cout << "Configuration:" << std::endl;
        std::cout << "  Threads: " << NUM_THREADS << std::endl;
        std::cout << "  Ops/thread: " << OPS_PER_THREAD << std::endl;
        std::cout << "  Iterations: " << NUM_ITERATIONS << std::endl;
        std::cout << "  Buffer size: " << BUFFER_SIZE << " bytes" << std::endl;
        std::cout << std::endl;

        // Get Metal device
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "ERROR: No Metal device available" << std::endl;
            return 1;
        }
        std::cout << "Device: " << [[device name] UTF8String] << std::endl;
        std::cout << std::endl;

        // Create shared compute pipelines
        id<MTLComputePipelineState> copyPipeline = createPipeline(device, "copy_kernel");
        id<MTLComputePipelineState> addPipeline = createPipeline(device, "add_kernel");

        if (!copyPipeline || !addPipeline) {
            std::cerr << "ERROR: Failed to create pipelines" << std::endl;
            return 1;
        }

        std::cout << "Pipelines created successfully" << std::endl;
        std::cout << std::endl;

        // Run iterations
        for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
            std::cout << "Iteration " << (iter + 1) << "/" << NUM_ITERATIONS << "..." << std::flush;

            g_completed_ops = 0;

            auto start = std::chrono::high_resolution_clock::now();

            // Launch worker threads
            std::vector<std::thread> threads;
            for (int t = 0; t < NUM_THREADS; t++) {
                threads.emplace_back(workerThread, t, device, copyPipeline, addPipeline);
            }

            // Wait for all threads
            for (auto& t : threads) {
                t.join();
            }

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

            uint64_t ops = g_completed_ops.load();
            double ops_per_sec = (double)ops / (duration.count() / 1000.0);

            std::cout << " " << ops << " ops in " << duration.count() << "ms ("
                      << (int)ops_per_sec << " ops/s)" << std::endl;

            // If we got here without crashing, report success
            if (iter > 0 && (iter + 1) % 10 == 0) {
                std::cout << "  --> " << (iter + 1) << " iterations completed without crash" << std::endl;
            }
        }

        std::cout << std::endl;
        std::cout << "=== All iterations completed without crash ===" << std::endl;
        std::cout << "NOTE: If you see this message, the race condition may be timing-dependent." << std::endl;
        std::cout << "Try running multiple times or increasing thread count." << std::endl;

        return 0;
    }
}
