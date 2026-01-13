/**
 * Minimal Metal Reproduction: AGX Driver Race Condition (MPS Version)
 *
 * This version uses MetalPerformanceShaders (MPS) matrix operations,
 * which more closely matches PyTorch's MPS backend behavior where the
 * race condition was originally discovered.
 *
 * BUILD:
 *   clang++ -std=c++17 -framework Metal -framework Foundation \
 *     -framework MetalPerformanceShaders -o metal_race_repro_mps metal_race_repro_mps.mm
 *
 * RUN (expect crash within seconds):
 *   ./metal_race_repro_mps
 *
 * The crash occurs in Apple's AGXMetalG16X driver when:
 *   - Multiple threads encode MPS operations concurrently
 *   - One thread may commit/wait while another encodes
 *   - The AGX driver's internal ContextCommon state becomes corrupted
 *
 * Author: Andrew Yates
 * Part of MPS Parallel Inference Research Project
 * https://github.com/ayates_dbx/metal_mps_parallel
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <thread>
#import <atomic>
#import <vector>
#import <iostream>
#import <chrono>
#import <random>

// Configuration - tuned to trigger the race
constexpr int NUM_THREADS = 8;
constexpr int OPS_PER_THREAD = 50;
constexpr int NUM_ITERATIONS = 100;
constexpr int MATRIX_SIZE = 512;  // M x N matrix

// Statistics
std::atomic<uint64_t> g_completed_ops{0};
std::atomic<bool> g_stop{false};

// Shared resources (intentionally shared to create contention)
id<MTLDevice> g_device = nil;
id<MTLCommandQueue> g_shared_queue = nil;  // Shared queue like PyTorch uses

// Worker thread - encodes MPS matrix multiply operations
void mpsWorkerThread(int threadId) {
    @autoreleasepool {
        // Random number generator for varying matrix sizes
        std::mt19937 rng(threadId);
        std::uniform_int_distribution<int> sizeDist(64, MATRIX_SIZE);

        for (int op = 0; op < OPS_PER_THREAD && !g_stop.load(); op++) {
            @autoreleasepool {
                int M = sizeDist(rng);
                int N = sizeDist(rng);
                int K = sizeDist(rng);

                // Create MPS matrix descriptors
                MPSMatrixDescriptor* descA = [MPSMatrixDescriptor
                    matrixDescriptorWithRows:M
                    columns:K
                    rowBytes:K * sizeof(float)
                    dataType:MPSDataTypeFloat32];

                MPSMatrixDescriptor* descB = [MPSMatrixDescriptor
                    matrixDescriptorWithRows:K
                    columns:N
                    rowBytes:N * sizeof(float)
                    dataType:MPSDataTypeFloat32];

                MPSMatrixDescriptor* descC = [MPSMatrixDescriptor
                    matrixDescriptorWithRows:M
                    columns:N
                    rowBytes:N * sizeof(float)
                    dataType:MPSDataTypeFloat32];

                // Create buffers
                id<MTLBuffer> bufferA = [g_device newBufferWithLength:M * K * sizeof(float)
                                                             options:MTLResourceStorageModeShared];
                id<MTLBuffer> bufferB = [g_device newBufferWithLength:K * N * sizeof(float)
                                                             options:MTLResourceStorageModeShared];
                id<MTLBuffer> bufferC = [g_device newBufferWithLength:M * N * sizeof(float)
                                                             options:MTLResourceStorageModeShared];

                if (!bufferA || !bufferB || !bufferC) {
                    continue;
                }

                // Initialize matrices with random data
                float* dataA = (float*)[bufferA contents];
                float* dataB = (float*)[bufferB contents];
                for (int i = 0; i < M * K; i++) dataA[i] = (float)(rng() % 100) / 100.0f;
                for (int i = 0; i < K * N; i++) dataB[i] = (float)(rng() % 100) / 100.0f;

                // Create MPS matrices
                MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:descA];
                MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:descB];
                MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:descC];

                // Create matrix multiplication kernel
                MPSMatrixMultiplication* matmul = [[MPSMatrixMultiplication alloc]
                    initWithDevice:g_device
                    transposeLeft:NO
                    transposeRight:NO
                    resultRows:M
                    resultColumns:N
                    interiorColumns:K
                    alpha:1.0
                    beta:0.0];

                // CRITICAL: This is where the race manifests
                // Multiple threads calling encodeToCommandBuffer: on MPS kernels
                // while using the same or different command queues

                // Create command buffer from shared queue (like PyTorch)
                id<MTLCommandBuffer> commandBuffer = [g_shared_queue commandBuffer];
                if (!commandBuffer) {
                    continue;
                }

                // Encode the matrix multiplication
                // This internally calls setComputePipelineState: which can race
                [matmul encodeToCommandBuffer:commandBuffer
                                   leftMatrix:matA
                                  rightMatrix:matB
                                 resultMatrix:matC];

                // Commit
                [commandBuffer commit];

                // Randomly wait for completion (mimics PyTorch's synchronize behavior)
                // This creates the cross-thread timing that triggers the race
                if (op % 3 == 0) {
                    [commandBuffer waitUntilCompleted];
                }

                g_completed_ops++;
            }
        }
    }
}

// Synchronizer thread - periodically waits on all operations
void synchronizerThread() {
    @autoreleasepool {
        while (!g_stop.load()) {
            @autoreleasepool {
                // Create an empty command buffer just to synchronize
                id<MTLCommandBuffer> syncBuffer = [g_shared_queue commandBuffer];
                if (syncBuffer) {
                    [syncBuffer commit];
                    [syncBuffer waitUntilCompleted];
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }
}

int main(int argc, char* argv[]) {
    @autoreleasepool {
        std::cout << "=== Metal AGX Driver Race Condition (MPS Version) ===" << std::endl;
        std::cout << "Configuration:" << std::endl;
        std::cout << "  Threads: " << NUM_THREADS << std::endl;
        std::cout << "  Ops/thread: " << OPS_PER_THREAD << std::endl;
        std::cout << "  Iterations: " << NUM_ITERATIONS << std::endl;
        std::cout << "  Matrix size: up to " << MATRIX_SIZE << "x" << MATRIX_SIZE << std::endl;
        std::cout << std::endl;

        // Get Metal device
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            std::cerr << "ERROR: No Metal device available" << std::endl;
            return 1;
        }
        std::cout << "Device: " << [[g_device name] UTF8String] << std::endl;

        // Check MPS support
        if (![g_device supportsFamily:MTLGPUFamilyApple7]) {
            std::cout << "Warning: Device may not support all MPS features" << std::endl;
        }
        std::cout << std::endl;

        // Run iterations
        for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
            std::cout << "Iteration " << (iter + 1) << "/" << NUM_ITERATIONS << "..." << std::flush;

            // Create fresh shared command queue each iteration
            // This mimics PyTorch's behavior where a queue is shared across operations
            g_shared_queue = [g_device newCommandQueue];
            if (!g_shared_queue) {
                std::cerr << " Failed to create command queue" << std::endl;
                continue;
            }

            g_completed_ops = 0;
            g_stop = false;

            auto start = std::chrono::high_resolution_clock::now();

            // Start synchronizer thread (mimics PyTorch's synchronize calls)
            std::thread syncThread(synchronizerThread);

            // Launch worker threads
            std::vector<std::thread> threads;
            for (int t = 0; t < NUM_THREADS; t++) {
                threads.emplace_back(mpsWorkerThread, t);
            }

            // Wait for workers
            for (auto& t : threads) {
                t.join();
            }

            // Stop synchronizer
            g_stop = true;
            syncThread.join();

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

            uint64_t ops = g_completed_ops.load();
            double ops_per_sec = duration.count() > 0 ? (double)ops / (duration.count() / 1000.0) : 0;

            std::cout << " " << ops << " ops in " << duration.count() << "ms ("
                      << (int)ops_per_sec << " ops/s)" << std::endl;

            // Reset shared queue
            g_shared_queue = nil;

            if ((iter + 1) % 10 == 0) {
                std::cout << "  --> " << (iter + 1) << " iterations completed" << std::endl;
            }
        }

        std::cout << std::endl;
        std::cout << "=== Test completed ===" << std::endl;
        std::cout << "If no crash occurred, the race may be timing-dependent." << std::endl;
        std::cout << "The original crash was observed with PyTorch MPS at ~55% rate." << std::endl;

        return 0;
    }
}
