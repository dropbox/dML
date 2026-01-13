// multi_queue_parallel_test.mm
// Test whether multiple MTLCommandQueue instances improve throughput vs a single shared queue.
//
// Build (standalone):
//   clang++ -std=c++17 -O2 -Wall -Wextra -fobjc-arc -x objective-c++ \
//     -framework Foundation -framework Metal \
//     -o multi_queue_parallel_test multi_queue_parallel_test.mm
//
// Or (repo):
//   ./tests/build_cpp_tests.sh multi_queue_parallel_test
//
// Run:
//   ./tests/build/multi_queue_parallel_test
//   ./tests/build/multi_queue_parallel_test --help

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

static std::string makeKernelSource(int innerIterations) {
    return std::string(R"(
#include <metal_stdlib>
using namespace metal;

#define INNER_ITERS )") + std::to_string(innerIterations) + R"(

kernel void simple_compute(
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

struct RunConfig {
    std::vector<int> threadCounts{1, 2, 4, 8, 16};
    int iterationsPerThread = 50;
    int dataSize = 1024 * 1024;
    int kernelInnerIterations = 100;
    int maxInflight = 8;
    bool asyncSubmit = false;
};

struct Scenario {
    const char* name = nullptr;
    bool perThreadQueue = false;
};

struct ScenarioResult {
    int threads = 0;
    int totalOps = 0;
    double elapsedMs = 0.0;
    double opsPerSec = 0.0;
};

static void warmUpPipeline(id<MTLDevice> device, id<MTLComputePipelineState> pipeline, int dataSize) {
    @autoreleasepool {
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLBuffer> buffer = [device newBufferWithLength:static_cast<NSUInteger>(dataSize) * sizeof(float)
                                                  options:MTLResourceStorageModeShared];
        std::memset([buffer contents], 0, static_cast<size_t>(dataSize) * sizeof(float));

        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buffer offset:0 atIndex:0];

        MTLSize gridSize = MTLSizeMake(static_cast<NSUInteger>(dataSize), 1, 1);
        MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
}

static void printUsage(const char* argv0) {
    std::cout << "Usage: " << argv0 << " [options]\n\n"
              << "Options:\n"
              << "  --threads <list>     Comma-separated list (default: 1,2,4,8,16)\n"
              << "  --iters <n>          Iterations per thread (default: 50)\n"
              << "  --data <n>           Elements per thread buffer (default: 1048576)\n"
              << "  --kernel-iters <n>   Inner loop iterations (default: 100)\n"
              << "  --async              Commit without per-CB wait (default: off)\n"
              << "  --inflight <n>       Max inflight per thread in --async (default: 8)\n"
              << "  --help               Show this help\n";
}

static bool parseInt(const char* s, int* out) {
    if (!s || !*s) {
        return false;
    }
    char* end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (!end || *end != '\0') {
        return false;
    }
    if (v < std::numeric_limits<int>::min() || v > std::numeric_limits<int>::max()) {
        return false;
    }
    *out = static_cast<int>(v);
    return true;
}

static bool parseThreadList(const char* s, std::vector<int>* out) {
    if (!s || !out) {
        return false;
    }
    std::vector<int> values;
    const char* p = s;
    while (*p) {
        const char* start = p;
        while (*p && *p != ',') {
            p++;
        }
        std::string token(start, static_cast<size_t>(p - start));
        int v = 0;
        if (!parseInt(token.c_str(), &v) || v <= 0) {
            return false;
        }
        values.push_back(v);
        if (*p == ',') {
            p++;
        }
    }
    if (values.empty()) {
        return false;
    }
    *out = std::move(values);
    return true;
}

static ScenarioResult runScenario(
    const RunConfig& config,
    const Scenario& scenario,
    id<MTLDevice> device,
    id<MTLComputePipelineState> pipeline,
    int numThreads
) {
    @autoreleasepool {
        id<MTLCommandQueue> sharedQueue = nil;
        if (!scenario.perThreadQueue) {
            sharedQueue = [device newCommandQueue];
        }

        std::vector<std::thread> threads;
        threads.reserve(static_cast<size_t>(numThreads));

        auto start = std::chrono::high_resolution_clock::now();

        for (int threadIndex = 0; threadIndex < numThreads; threadIndex++) {
            threads.emplace_back([&, threadIndex]() {
                @autoreleasepool {
                    id<MTLCommandQueue> queue = sharedQueue;
                    if (scenario.perThreadQueue) {
                        queue = [device newCommandQueue];
                    }

                    id<MTLBuffer> buffer = [device newBufferWithLength:static_cast<NSUInteger>(config.dataSize) * sizeof(float)
                                                              options:MTLResourceStorageModeShared];

                    float* data = static_cast<float*>([buffer contents]);
                    for (int i = 0; i < config.dataSize; i++) {
                        data[i] = static_cast<float>((threadIndex + 1) * 0.001) + static_cast<float>(i) * 1e-6f;
                    }

                    std::vector<id<MTLCommandBuffer>> inflightBuffers;
                    inflightBuffers.reserve(static_cast<size_t>(config.maxInflight));

                    for (int iter = 0; iter < config.iterationsPerThread; iter++) {
                        @autoreleasepool {
                            id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
                            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

                            [encoder setComputePipelineState:pipeline];
                            [encoder setBuffer:buffer offset:0 atIndex:0];

                            MTLSize gridSize = MTLSizeMake(static_cast<NSUInteger>(config.dataSize), 1, 1);
                            MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
                            [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
                            [encoder endEncoding];

                            [commandBuffer commit];

                            if (config.asyncSubmit) {
                                inflightBuffers.push_back(commandBuffer);
                                if (static_cast<int>(inflightBuffers.size()) >= config.maxInflight) {
                                    [inflightBuffers.front() waitUntilCompleted];
                                    inflightBuffers.erase(inflightBuffers.begin());
                                }
                            } else {
                                [commandBuffer waitUntilCompleted];
                            }

                        }
                    }

                    for (id<MTLCommandBuffer> cb : inflightBuffers) {
                        [cb waitUntilCompleted];
                    }
                }
            });
        }

        for (auto& t : threads) {
            t.join();
        }

        auto end = std::chrono::high_resolution_clock::now();
        double elapsedMs = std::chrono::duration<double, std::milli>(end - start).count();

        int totalOps = numThreads * config.iterationsPerThread;
        double opsPerSec = (static_cast<double>(totalOps) * 1000.0) / elapsedMs;

        ScenarioResult result;
        result.threads = numThreads;
        result.totalOps = totalOps;
        result.elapsedMs = elapsedMs;
        result.opsPerSec = opsPerSec;
        return result;
    }
}

int main(int argc, char** argv) {
    RunConfig config;

    for (int i = 1; i < argc; i++) {
        const char* arg = argv[i];
        if (std::strcmp(arg, "--help") == 0) {
            printUsage(argv[0]);
            return 0;
        }
        if (std::strcmp(arg, "--async") == 0) {
            config.asyncSubmit = true;
            continue;
        }
        auto requireValue = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << flag << "\n";
                return nullptr;
            }
            return argv[++i];
        };
        if (std::strcmp(arg, "--threads") == 0) {
            const char* v = requireValue("--threads");
            if (!v || !parseThreadList(v, &config.threadCounts)) {
                std::cerr << "Invalid --threads list\n";
                return 2;
            }
            continue;
        }
        if (std::strcmp(arg, "--iters") == 0) {
            const char* v = requireValue("--iters");
            if (!v || !parseInt(v, &config.iterationsPerThread) || config.iterationsPerThread <= 0) {
                std::cerr << "Invalid --iters\n";
                return 2;
            }
            continue;
        }
        if (std::strcmp(arg, "--data") == 0) {
            const char* v = requireValue("--data");
            if (!v || !parseInt(v, &config.dataSize) || config.dataSize <= 0) {
                std::cerr << "Invalid --data\n";
                return 2;
            }
            continue;
        }
        if (std::strcmp(arg, "--kernel-iters") == 0) {
            const char* v = requireValue("--kernel-iters");
            if (!v || !parseInt(v, &config.kernelInnerIterations) || config.kernelInnerIterations <= 0) {
                std::cerr << "Invalid --kernel-iters\n";
                return 2;
            }
            continue;
        }
        if (std::strcmp(arg, "--inflight") == 0) {
            const char* v = requireValue("--inflight");
            if (!v || !parseInt(v, &config.maxInflight) || config.maxInflight <= 0) {
                std::cerr << "Invalid --inflight\n";
                return 2;
            }
            continue;
        }

        std::cerr << "Unknown argument: " << arg << "\n";
        printUsage(argv[0]);
        return 2;
    }

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "Metal not available\n";
            return 1;
        }

        std::string source = makeKernelSource(config.kernelInnerIterations);
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                                      options:nil
                                                        error:&error];
        if (error) {
            std::cerr << "Shader compile error: " << [[error description] UTF8String] << "\n";
            return 1;
        }

        id<MTLFunction> function = [library newFunctionWithName:@"simple_compute"];
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (error) {
            std::cerr << "Pipeline error: " << [[error description] UTF8String] << "\n";
            return 1;
        }

        warmUpPipeline(device, pipeline, config.dataSize);

        std::cout << "========================================\n";
        std::cout << "MULTI-QUEUE PARALLELISM TEST\n";
        std::cout << "Device: " << [[device name] UTF8String] << "\n";
        std::cout << "Mode: " << (config.asyncSubmit ? "async submit (limited inflight)" : "commit+wait per command buffer") << "\n";
        std::cout << "iters/thread=" << config.iterationsPerThread
                  << "  data=" << config.dataSize
                  << "  kernel-iters=" << config.kernelInnerIterations
                  << "  inflight=" << config.maxInflight
                  << "\n";
        std::cout << "========================================\n\n";

        const Scenario scenarios[] = {
            {"Single shared MTLCommandQueue", false},
            {"Per-thread MTLCommandQueue", true},
        };

        for (const Scenario& scenario : scenarios) {
            std::cout << scenario.name << "\n";
            double baselineOpsPerSec = 0.0;

            for (int threads : config.threadCounts) {
                ScenarioResult r = runScenario(config, scenario, device, pipeline, threads);
                if (threads == config.threadCounts.front()) {
                    baselineOpsPerSec = r.opsPerSec;
                }

                double speedup = (baselineOpsPerSec > 0.0) ? (r.opsPerSec / baselineOpsPerSec) : 0.0;
                std::cout << "  Threads: " << std::setw(2) << r.threads
                          << "  Ops: " << std::setw(5) << r.totalOps
                          << "  Time: " << std::fixed << std::setprecision(1) << std::setw(7) << r.elapsedMs << "ms"
                          << "  Ops/s: " << std::setprecision(1) << std::setw(8) << r.opsPerSec
                          << "  Speedup: " << std::setprecision(2) << speedup << "x\n";
            }
            std::cout << "\n";
        }
    }

    return 0;
}
