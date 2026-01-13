/**
 * Bare Metal API Race Condition Test
 *
 * Purpose: Determine if the .contiguous() race condition is in:
 * 1. Apple's Metal/MPS framework (unfixable by us)
 * 2. PyTorch's ATen wrapper layer (potentially fixable)
 *
 * This test uses ONLY Metal and MetalPerformanceShaders APIs - no ATen/PyTorch.
 * We attempt to reproduce the same memory access patterns that cause races
 * in the PyTorch .contiguous() operation:
 *
 * Pattern being tested:
 * 1. Allocate MTLBuffer
 * 2. Create non-contiguous view (stride pattern)
 * 3. Allocate new MTLBuffer for contiguous copy
 * 4. Blit (copy) data with striding to new buffer
 * 5. Synchronize
 *
 * If this test shows races -> Apple framework bug
 * If this test is stable -> PyTorch ATen layer bug
 *
 * Build:
 *   clang++ -std=c++17 -framework Metal -framework MetalPerformanceShaders \
 *           -framework Foundation -O2 -o minimal_metal_race minimal_metal_race.mm
 *
 * Run:
 *   ./minimal_metal_race
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

// Test configuration
constexpr int NUM_THREADS = 8;
constexpr int NUM_ITERATIONS = 30;
constexpr int BUFFER_ELEMENTS = 256 * 128 * 3;  // ~100K floats, similar to projection
constexpr float TOLERANCE = 1e-5f;

std::mutex g_cout_mutex;

void print_sync(const std::string& msg) {
    std::lock_guard<std::mutex> lock(g_cout_mutex);
    std::cout << msg << std::endl;
}

/**
 * Metal compute shader for strided copy (simulates .contiguous())
 *
 * This mimics what ATen does internally: copy from strided source to
 * contiguous destination.
 */
NSString* const kStridedCopyShader = @R"(
#include <metal_stdlib>
using namespace metal;

// Copy from strided source to contiguous destination
// Simulates the .contiguous() memory layout transformation
kernel void strided_copy(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& src_stride [[buffer(2)]],
    constant uint& num_elements [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx < num_elements) {
        // Source is strided (non-contiguous), dest is contiguous
        uint src_idx = (idx / 256) * src_stride + (idx % 256);
        dst[idx] = src[src_idx];
    }
}

// Reference copy for validation (contiguous to contiguous)
kernel void contiguous_copy(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& num_elements [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx < num_elements) {
        dst[idx] = src[idx];
    }
}

// Simple matmul for compute workload (simulates SDPA)
kernel void simple_matmul(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& N [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x < N && gid.y < N) {
        float sum = 0.0f;
        for (uint k = 0; k < N; k++) {
            sum += A[gid.y * N + k] * B[k * N + gid.x];
        }
        C[gid.y * N + gid.x] = sum;
    }
}
)";

/**
 * Thread-local Metal context
 * Each thread gets its own command queue (simulating per-stream dispatch)
 */
struct MetalContext {
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLComputePipelineState> stridedCopyPipeline;
    id<MTLComputePipelineState> contiguousCopyPipeline;
    id<MTLComputePipelineState> matmulPipeline;

    MetalContext(id<MTLDevice> dev) : device(dev) {
        // Each thread creates its own command queue
        // This mirrors PyTorch's per-thread MPS stream behavior
        commandQueue = [device newCommandQueue];

        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:kStridedCopyShader
                                                      options:nil
                                                        error:&error];
        if (!library) {
            std::cerr << "Failed to create Metal library: "
                      << [[error localizedDescription] UTF8String] << std::endl;
            return;
        }

        id<MTLFunction> stridedCopyFunc = [library newFunctionWithName:@"strided_copy"];
        id<MTLFunction> contiguousCopyFunc = [library newFunctionWithName:@"contiguous_copy"];
        id<MTLFunction> matmulFunc = [library newFunctionWithName:@"simple_matmul"];

        stridedCopyPipeline = [device newComputePipelineStateWithFunction:stridedCopyFunc error:&error];
        contiguousCopyPipeline = [device newComputePipelineStateWithFunction:contiguousCopyFunc error:&error];
        matmulPipeline = [device newComputePipelineStateWithFunction:matmulFunc error:&error];
    }
};

/**
 * Strided buffer copy operation (simulates .contiguous())
 *
 * This is the core operation we're testing for thread safety.
 * It allocates a new buffer and copies data with striding.
 */
id<MTLBuffer> stridedCopy(MetalContext& ctx, id<MTLBuffer> srcBuffer,
                          uint32_t srcStride, uint32_t numElements) {
    @autoreleasepool {
        // Allocate destination buffer (simulates at::empty_mps())
        id<MTLBuffer> dstBuffer = [ctx.device newBufferWithLength:numElements * sizeof(float)
                                                          options:MTLResourceStorageModeShared];

        uint32_t params[] = { srcStride, numElements };

        id<MTLCommandBuffer> cmdBuf = [ctx.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

        [encoder setComputePipelineState:ctx.stridedCopyPipeline];
        [encoder setBuffer:srcBuffer offset:0 atIndex:0];
        [encoder setBuffer:dstBuffer offset:0 atIndex:1];
        [encoder setBytes:&params[0] length:sizeof(uint32_t) atIndex:2];
        [encoder setBytes:&params[1] length:sizeof(uint32_t) atIndex:3];

        MTLSize gridSize = MTLSizeMake(numElements, 1, 1);
        MTLSize threadGroupSize = MTLSizeMake(MIN(256, ctx.stridedCopyPipeline.maxTotalThreadsPerThreadgroup), 1, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        return dstBuffer;
    }
}

/**
 * Simple matmul operation (simulates SDPA workload)
 */
id<MTLBuffer> matmul(MetalContext& ctx, id<MTLBuffer> A, id<MTLBuffer> B, uint32_t N) {
    @autoreleasepool {
        id<MTLBuffer> C = [ctx.device newBufferWithLength:N * N * sizeof(float)
                                                  options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmdBuf = [ctx.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

        [encoder setComputePipelineState:ctx.matmulPipeline];
        [encoder setBuffer:A offset:0 atIndex:0];
        [encoder setBuffer:B offset:0 atIndex:1];
        [encoder setBuffer:C offset:0 atIndex:2];
        [encoder setBytes:&N length:sizeof(uint32_t) atIndex:3];

        MTLSize gridSize = MTLSizeMake(N, N, 1);
        MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        return C;
    }
}

/**
 * Run the strided copy test (simulates .contiguous() race scenario)
 *
 * Each thread:
 * 1. Creates strided source buffer
 * 2. Performs strided copy to contiguous buffer (the potential race)
 * 3. Performs matmul on result
 * 4. Compares to serial baseline
 */
std::tuple<int, int, float> runStridedCopyTest(id<MTLDevice> device, bool parallel) {
    @autoreleasepool {
        int passed = 0;
        float maxDiff = 0.0f;

        // Pre-create shared weights for matmul
        constexpr uint32_t MATMUL_N = 64;  // 64x64 matmul
        id<MTLBuffer> weightBuffer = [device newBufferWithLength:MATMUL_N * MATMUL_N * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
        {
            float* weights = (float*)[weightBuffer contents];
            std::mt19937 rng(42);
            std::normal_distribution<float> dist(0.0f, 1.0f);
            for (size_t i = 0; i < MATMUL_N * MATMUL_N; i++) {
                weights[i] = dist(rng);
            }
        }

        for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
            // Create per-thread input buffers
            std::vector<id<MTLBuffer>> inputBuffers(NUM_THREADS);
            std::vector<float> expectedResults(NUM_THREADS);

            // Strided layout: data is stored with stride = 512 (double the logical size)
            constexpr uint32_t SRC_STRIDE = 512;
            constexpr uint32_t LOGICAL_SIZE = 256;
            constexpr uint32_t PHYSICAL_SIZE = SRC_STRIDE * (BUFFER_ELEMENTS / LOGICAL_SIZE);

            for (int tid = 0; tid < NUM_THREADS; tid++) {
                inputBuffers[tid] = [device newBufferWithLength:PHYSICAL_SIZE * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
                float* data = (float*)[inputBuffers[tid] contents];

                // Fill with thread-specific pattern
                std::mt19937 rng(iter * 1000 + tid);
                std::normal_distribution<float> dist(0.0f, 1.0f);
                for (size_t i = 0; i < PHYSICAL_SIZE; i++) {
                    data[i] = dist(rng);
                }
            }

            // Compute expected results serially
            {
                MetalContext serialCtx(device);
                for (int tid = 0; tid < NUM_THREADS; tid++) {
                    auto contiguousBuffer = stridedCopy(serialCtx, inputBuffers[tid], SRC_STRIDE, BUFFER_ELEMENTS);

                    // For simplicity, just sum the first MATMUL_N*MATMUL_N elements
                    float* data = (float*)[contiguousBuffer contents];
                    float sum = 0.0f;
                    for (size_t i = 0; i < MIN((size_t)(MATMUL_N * MATMUL_N), (size_t)BUFFER_ELEMENTS); i++) {
                        sum += data[i];
                    }
                    expectedResults[tid] = sum;
                }
            }

            // Run test (parallel or serial depending on flag)
            std::vector<float> actualResults(NUM_THREADS, 0.0f);
            std::atomic<int> threadErrors{0};

            if (parallel) {
                // Parallel execution - potential race condition
                std::vector<std::thread> threads;

                for (int tid = 0; tid < NUM_THREADS; tid++) {
                    threads.emplace_back([&, tid]() {
                        @autoreleasepool {
                            try {
                                MetalContext ctx(device);  // Each thread gets own context
                                auto contiguousBuffer = stridedCopy(ctx, inputBuffers[tid], SRC_STRIDE, BUFFER_ELEMENTS);

                                // Sum result
                                float* data = (float*)[contiguousBuffer contents];
                                float sum = 0.0f;
                                for (size_t i = 0; i < MIN((size_t)(MATMUL_N * MATMUL_N), (size_t)BUFFER_ELEMENTS); i++) {
                                    sum += data[i];
                                }
                                actualResults[tid] = sum;
                            } catch (...) {
                                threadErrors++;
                            }
                        }
                    });
                }

                for (auto& t : threads) {
                    t.join();
                }
            } else {
                // Serial execution - baseline
                MetalContext ctx(device);
                for (int tid = 0; tid < NUM_THREADS; tid++) {
                    auto contiguousBuffer = stridedCopy(ctx, inputBuffers[tid], SRC_STRIDE, BUFFER_ELEMENTS);

                    float* data = (float*)[contiguousBuffer contents];
                    float sum = 0.0f;
                    for (size_t i = 0; i < MIN((size_t)(MATMUL_N * MATMUL_N), (size_t)BUFFER_ELEMENTS); i++) {
                        sum += data[i];
                    }
                    actualResults[tid] = sum;
                }
            }

            // Check results
            bool iterOk = (threadErrors == 0);
            for (int tid = 0; tid < NUM_THREADS && iterOk; tid++) {
                float diff = std::abs(actualResults[tid] - expectedResults[tid]);
                maxDiff = std::max(maxDiff, diff);

                // Relative tolerance for floating point sums
                float relTol = std::abs(expectedResults[tid]) * TOLERANCE + TOLERANCE;
                if (diff > relTol) {
                    iterOk = false;
                }
            }

            if (iterOk) {
                passed++;
            }
        }

        return {passed, NUM_ITERATIONS, maxDiff};
    }
}

/**
 * Run test using MetalPerformanceShaders (MPSMatrix operations)
 *
 * This tests whether MPS-level operations have race conditions.
 */
std::tuple<int, int, float> runMPSMatrixTest(id<MTLDevice> device, bool parallel) {
    @autoreleasepool {
        int passed = 0;
        float maxDiff = 0.0f;

        constexpr uint32_t ROWS = 128;
        constexpr uint32_t COLS = 256;
        constexpr uint32_t K = 64;

        // Pre-create weight matrix (shared across all threads)
        MPSMatrixDescriptor* weightDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:K
                                                                                columns:COLS
                                                                               rowBytes:COLS * sizeof(float)
                                                                               dataType:MPSDataTypeFloat32];

        id<MTLBuffer> weightBuffer = [device newBufferWithLength:K * COLS * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
        {
            float* weights = (float*)[weightBuffer contents];
            std::mt19937 rng(42);
            std::normal_distribution<float> dist(0.0f, 0.1f);
            for (size_t i = 0; i < K * COLS; i++) {
                weights[i] = dist(rng);
            }
        }

        MPSMatrix* weightMatrix = [[MPSMatrix alloc] initWithBuffer:weightBuffer descriptor:weightDesc];

        for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
            // Create per-thread input matrices
            std::vector<id<MTLBuffer>> inputBuffers(NUM_THREADS);
            std::vector<float> expectedResults(NUM_THREADS);

            MPSMatrixDescriptor* inputDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:ROWS
                                                                                   columns:K
                                                                                  rowBytes:K * sizeof(float)
                                                                                  dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor* outputDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:ROWS
                                                                                    columns:COLS
                                                                                   rowBytes:COLS * sizeof(float)
                                                                                   dataType:MPSDataTypeFloat32];

            for (int tid = 0; tid < NUM_THREADS; tid++) {
                inputBuffers[tid] = [device newBufferWithLength:ROWS * K * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
                float* data = (float*)[inputBuffers[tid] contents];
                std::mt19937 rng(iter * 1000 + tid);
                std::normal_distribution<float> dist(0.0f, 0.1f);
                for (size_t i = 0; i < ROWS * K; i++) {
                    data[i] = dist(rng);
                }
            }

            // Compute expected results serially using MPS
            {
                id<MTLCommandQueue> queue = [device newCommandQueue];
                MPSMatrixMultiplication* matmul = [[MPSMatrixMultiplication alloc] initWithDevice:device
                                                                                   transposeLeft:NO
                                                                                  transposeRight:NO
                                                                                      resultRows:ROWS
                                                                                   resultColumns:COLS
                                                                                 interiorColumns:K
                                                                                           alpha:1.0
                                                                                            beta:0.0];

                for (int tid = 0; tid < NUM_THREADS; tid++) {
                    id<MTLBuffer> outputBuffer = [device newBufferWithLength:ROWS * COLS * sizeof(float)
                                                                     options:MTLResourceStorageModeShared];

                    MPSMatrix* inputMatrix = [[MPSMatrix alloc] initWithBuffer:inputBuffers[tid] descriptor:inputDesc];
                    MPSMatrix* outputMatrix = [[MPSMatrix alloc] initWithBuffer:outputBuffer descriptor:outputDesc];

                    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
                    [matmul encodeToCommandBuffer:cmdBuf leftMatrix:inputMatrix rightMatrix:weightMatrix resultMatrix:outputMatrix];
                    [cmdBuf commit];
                    [cmdBuf waitUntilCompleted];

                    // Sum result for comparison
                    float* outData = (float*)[outputBuffer contents];
                    float sum = 0.0f;
                    for (size_t i = 0; i < MIN((size_t)4096, (size_t)(ROWS * COLS)); i++) {
                        sum += outData[i];
                    }
                    expectedResults[tid] = sum;
                }
            }

            // Run test (parallel or serial)
            std::vector<float> actualResults(NUM_THREADS, 0.0f);
            std::atomic<int> threadErrors{0};

            auto runThread = [&](int tid, id<MTLCommandQueue> queue) {
                @autoreleasepool {
                    try {
                        id<MTLBuffer> outputBuffer = [device newBufferWithLength:ROWS * COLS * sizeof(float)
                                                                         options:MTLResourceStorageModeShared];

                        MPSMatrix* inputMatrix = [[MPSMatrix alloc] initWithBuffer:inputBuffers[tid] descriptor:inputDesc];
                        MPSMatrix* outputMatrix = [[MPSMatrix alloc] initWithBuffer:outputBuffer descriptor:outputDesc];

                        MPSMatrixMultiplication* matmul = [[MPSMatrixMultiplication alloc] initWithDevice:device
                                                                                               transposeLeft:NO
                                                                                              transposeRight:NO
                                                                                                  resultRows:ROWS
                                                                                               resultColumns:COLS
                                                                                             interiorColumns:K
                                                                                                       alpha:1.0
                                                                                                        beta:0.0];

                        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
                        [matmul encodeToCommandBuffer:cmdBuf leftMatrix:inputMatrix rightMatrix:weightMatrix resultMatrix:outputMatrix];
                        [cmdBuf commit];
                        [cmdBuf waitUntilCompleted];

                        // Sum result
                        float* outData = (float*)[outputBuffer contents];
                        float sum = 0.0f;
                        for (size_t i = 0; i < MIN((size_t)4096, (size_t)(ROWS * COLS)); i++) {
                            sum += outData[i];
                        }
                        actualResults[tid] = sum;
                    } catch (...) {
                        threadErrors++;
                    }
                }
            };

            if (parallel) {
                std::vector<std::thread> threads;
                for (int tid = 0; tid < NUM_THREADS; tid++) {
                    threads.emplace_back([&, tid]() {
                        id<MTLCommandQueue> queue = [device newCommandQueue];  // Per-thread queue
                        runThread(tid, queue);
                    });
                }
                for (auto& t : threads) {
                    t.join();
                }
            } else {
                id<MTLCommandQueue> queue = [device newCommandQueue];
                for (int tid = 0; tid < NUM_THREADS; tid++) {
                    runThread(tid, queue);
                }
            }

            // Check results
            bool iterOk = (threadErrors == 0);
            for (int tid = 0; tid < NUM_THREADS && iterOk; tid++) {
                float diff = std::abs(actualResults[tid] - expectedResults[tid]);
                maxDiff = std::max(maxDiff, diff);

                float relTol = std::abs(expectedResults[tid]) * TOLERANCE + TOLERANCE;
                if (diff > relTol) {
                    iterOk = false;
                }
            }

            if (iterOk) {
                passed++;
            }
        }

        return {passed, NUM_ITERATIONS, maxDiff};
    }
}

int main(int /* argc */, char* /* argv */[]) {
    @autoreleasepool {
        std::cout << std::string(70, '=') << std::endl;
        std::cout << "Bare Metal API Race Condition Test" << std::endl;
        std::cout << "Purpose: Isolate whether race is in Metal/MPS or ATen layer" << std::endl;
        std::cout << std::string(70, '=') << std::endl;

        // Get Metal device
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "ERROR: Metal not available" << std::endl;
            return 1;
        }

        std::cout << "Device: " << [[device name] UTF8String] << std::endl;
        std::cout << "Configuration:" << std::endl;
        std::cout << "  Threads: " << NUM_THREADS << std::endl;
        std::cout << "  Iterations: " << NUM_ITERATIONS << std::endl;
        std::cout << "  Buffer elements: " << BUFFER_ELEMENTS << std::endl;
        std::cout << std::endl;

        // Test 1: Metal compute shader strided copy (serial baseline)
        std::cout << "Test 1: Metal Strided Copy (Serial Baseline)" << std::endl;
        auto [passed1, total1, diff1] = runStridedCopyTest(device, false);
        std::string status1 = (passed1 == total1) ? "PASS" : "FAIL";
        std::cout << "  Result: " << status1 << " (" << passed1 << "/" << total1
                  << "), max_diff=" << std::scientific << std::setprecision(2) << diff1 << std::endl;

        // Test 2: Metal compute shader strided copy (parallel - potential race)
        std::cout << "\nTest 2: Metal Strided Copy (Parallel - 8 threads)" << std::endl;
        auto [passed2, total2, diff2] = runStridedCopyTest(device, true);
        std::string status2 = (passed2 == total2) ? "PASS" : "FAIL";
        std::cout << "  Result: " << status2 << " (" << passed2 << "/" << total2
                  << "), max_diff=" << std::scientific << std::setprecision(2) << diff2 << std::endl;

        // Test 3: MPS Matrix operations (serial baseline)
        std::cout << "\nTest 3: MPS Matrix Multiply (Serial Baseline)" << std::endl;
        auto [passed3, total3, diff3] = runMPSMatrixTest(device, false);
        std::string status3 = (passed3 == total3) ? "PASS" : "FAIL";
        std::cout << "  Result: " << status3 << " (" << passed3 << "/" << total3
                  << "), max_diff=" << std::scientific << std::setprecision(2) << diff3 << std::endl;

        // Test 4: MPS Matrix operations (parallel - potential race)
        std::cout << "\nTest 4: MPS Matrix Multiply (Parallel - 8 threads)" << std::endl;
        auto [passed4, total4, diff4] = runMPSMatrixTest(device, true);
        std::string status4 = (passed4 == total4) ? "PASS" : "FAIL";
        std::cout << "  Result: " << status4 << " (" << passed4 << "/" << total4
                  << "), max_diff=" << std::scientific << std::setprecision(2) << diff4 << std::endl;

        // Summary and analysis
        std::cout << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        std::cout << "SUMMARY AND ANALYSIS" << std::endl;
        std::cout << std::string(70, '=') << std::endl;

        std::cout << "\n| Test | Serial | Parallel | Status |" << std::endl;
        std::cout << "|------|--------|----------|--------|" << std::endl;
        std::cout << "| Metal Strided Copy | " << status1 << " | " << status2 << " | "
                  << ((status1 == "PASS" && status2 == "FAIL") ? "RACE DETECTED" : "STABLE") << " |" << std::endl;
        std::cout << "| MPS Matrix Multiply | " << status3 << " | " << status4 << " | "
                  << ((status3 == "PASS" && status4 == "FAIL") ? "RACE DETECTED" : "STABLE") << " |" << std::endl;

        std::cout << std::endl;

        bool metalRace = (status1 == "PASS" && status2 == "FAIL");
        bool mpsRace = (status3 == "PASS" && status4 == "FAIL");

        if (metalRace || mpsRace) {
            std::cout << "RACE DETECTED at Metal/MPS API level!" << std::endl;
            std::cout << std::endl;
            std::cout << "Conclusion: The race condition exists in Apple's Metal/MPS framework," << std::endl;
            std::cout << "not in PyTorch's ATen layer. This is an Apple framework bug that" << std::endl;
            std::cout << "we cannot fix - our BatchQueue workaround is the correct approach." << std::endl;
        } else if (status2 == "PASS" && status4 == "PASS") {
            std::cout << "No race detected at bare Metal/MPS API level." << std::endl;
            std::cout << std::endl;
            std::cout << "Conclusion: The race condition in PyTorch .contiguous() is likely" << std::endl;
            std::cout << "in the ATen/MPS wrapper layer, not in Apple's Metal/MPS framework." << std::endl;
            std::cout << "This suggests the bug may be fixable in PyTorch's code." << std::endl;
            std::cout << std::endl;
            std::cout << "Possible causes:" << std::endl;
            std::cout << "  1. MPSGraph caching/lookup race conditions" << std::endl;
            std::cout << "  2. ATen tensor metadata race during allocation" << std::endl;
            std::cout << "  3. MPS heap allocator internal state corruption" << std::endl;
        } else {
            std::cout << "Unexpected results - further investigation needed." << std::endl;
        }

        return (metalRace || mpsRace) ? 1 : 0;
    }
}
