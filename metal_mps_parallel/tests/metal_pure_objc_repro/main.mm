/*
 * Pure Metal parallel submission timing repro.
 *
 * Goal: identify whether key Metal API calls serialize across threads even when
 * each thread uses its own MTLCommandQueue.
 *
 * This is intentionally independent of PyTorch/ATen.
 *
 * Build (from repo root):
 *   clang++ -std=c++17 -O2 -framework Foundation -framework Metal \
 *     -o tests/build/metal_pure_objc_repro tests/metal_pure_objc_repro/main.mm
 *
 * Run:
 *   ./tests/build/metal_pure_objc_repro --threads 8 --iters 200 --elements 262144
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

namespace {

class Barrier final {
 public:
  explicit Barrier(int participants) : m_threshold(participants), m_count(participants) {}

  void wait() {
    std::unique_lock<std::mutex> lock(m_mutex);
    const auto gen = m_generation;
    if (--m_count == 0) {
      m_generation++;
      m_count = m_threshold;
      m_cv.notify_all();
      return;
    }
    m_cv.wait(lock, [&] { return m_generation != gen; });
  }

 private:
  std::mutex m_mutex;
  std::condition_variable m_cv;
  int m_threshold = 0;
  int m_count = 0;
  std::uint64_t m_generation = 0;
};

template <class Rep, class Period>
double to_us(const std::chrono::duration<Rep, Period>& d) {
  return std::chrono::duration<double, std::micro>(d).count();
}

double percentile_us(std::vector<double> values, double pct) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const double clamped = std::min(std::max(pct, 0.0), 100.0);
  const double pos = (clamped / 100.0) * double(values.size() - 1);
  const std::size_t idx = static_cast<std::size_t>(pos);
  return values[idx];
}

struct Samples {
  std::vector<double> cmd_buffer_create_us;
  std::vector<double> encoder_create_us;
  std::vector<double> commit_us;
  std::vector<double> wait_us;
  std::vector<double> gpu_us;
};

struct RunResult {
  int threads = 0;
  int iters = 0;
  std::uint32_t elements = 0;
  double wall_time_s = 0.0;
  Samples samples;
};

NSString* const kKernelSource = @R"(
#include <metal_stdlib>
using namespace metal;

kernel void axpy(device const float* x [[buffer(0)]],
                 device float* y [[buffer(1)]],
                 constant uint& n [[buffer(2)]],
                 uint idx [[thread_position_in_grid]]) {
  if (idx < n) {
    y[idx] = fma(x[idx], 1.0001f, y[idx]);
  }
}
)";

RunResult run_once(int threads, int iters, std::uint32_t elements, bool wait_for_completion) {
  @autoreleasepool {
    RunResult out;
    out.threads = threads;
    out.iters = iters;
    out.elements = elements;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      throw std::runtime_error("MTLCreateSystemDefaultDevice returned nil");
    }

    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:kKernelSource options:nil error:&error];
    if (!library) {
      const char* msg = error ? [[error localizedDescription] UTF8String] : "unknown";
      throw std::runtime_error(std::string("failed to compile Metal library: ") + msg);
    }

    id<MTLFunction> fn = [library newFunctionWithName:@"axpy"];
    if (!fn) {
      throw std::runtime_error("failed to find kernel function axpy");
    }

    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
      const char* msg = error ? [[error localizedDescription] UTF8String] : "unknown";
      throw std::runtime_error(std::string("failed to create compute pipeline: ") + msg);
    }

    std::vector<id<MTLCommandQueue>> queues;
    queues.reserve(threads);
    for (int i = 0; i < threads; i++) {
      id<MTLCommandQueue> q = [device newCommandQueue];
      if (!q) {
        throw std::runtime_error("newCommandQueue returned nil");
      }
      queues.push_back(q);
    }

    const std::size_t bytes = std::size_t(elements) * sizeof(float);
    std::vector<id<MTLBuffer>> xbufs;
    std::vector<id<MTLBuffer>> ybufs;
    xbufs.reserve(threads);
    ybufs.reserve(threads);
    for (int i = 0; i < threads; i++) {
      id<MTLBuffer> xb = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
      id<MTLBuffer> yb = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
      if (!xb || !yb) {
        throw std::runtime_error("failed to allocate MTLBuffer");
      }
      xbufs.push_back(xb);
      ybufs.push_back(yb);
    }

    Barrier barrier(threads);
    std::mutex samples_mutex;

    auto start_wall = std::chrono::steady_clock::now();

    std::vector<std::thread> workers;
    workers.reserve(threads);
    for (int tid = 0; tid < threads; tid++) {
      workers.emplace_back([&, tid] {
        Samples local;
        local.cmd_buffer_create_us.reserve(iters);
        local.encoder_create_us.reserve(iters);
        local.commit_us.reserve(iters);
        local.wait_us.reserve(iters);
        local.gpu_us.reserve(iters);

        for (int i = 0; i < iters; i++) {
          barrier.wait();
          @autoreleasepool {
            const auto t0 = std::chrono::steady_clock::now();
            id<MTLCommandBuffer> cb = [queues[tid] commandBuffer];
            const auto t1 = std::chrono::steady_clock::now();
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            const auto t2 = std::chrono::steady_clock::now();

            [enc setComputePipelineState:pipeline];
            [enc setBuffer:xbufs[tid] offset:0 atIndex:0];
            [enc setBuffer:ybufs[tid] offset:0 atIndex:1];
            [enc setBytes:&elements length:sizeof(elements) atIndex:2];

            const MTLSize grid = MTLSizeMake(elements, 1, 1);
            const NSUInteger tg = std::min<NSUInteger>(256, pipeline.maxTotalThreadsPerThreadgroup);
            const MTLSize group = MTLSizeMake(tg, 1, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:group];
            [enc endEncoding];

            const auto c0 = std::chrono::steady_clock::now();
            [cb commit];
            const auto c1 = std::chrono::steady_clock::now();

            double wait_us = 0.0;
            double gpu_us = 0.0;
            if (wait_for_completion) {
              const auto w0 = std::chrono::steady_clock::now();
              [cb waitUntilCompleted];
              const auto w1 = std::chrono::steady_clock::now();
              wait_us = to_us(w1 - w0);
              const NSTimeInterval gpu_s = cb.GPUEndTime - cb.GPUStartTime;
              if (gpu_s > 0) {
                gpu_us = gpu_s * 1e6;
              }
            }

            local.cmd_buffer_create_us.push_back(to_us(t1 - t0));
            local.encoder_create_us.push_back(to_us(t2 - t1));
            local.commit_us.push_back(to_us(c1 - c0));
            local.wait_us.push_back(wait_us);
            local.gpu_us.push_back(gpu_us);
          }
        }

        std::lock_guard<std::mutex> lock(samples_mutex);
        out.samples.cmd_buffer_create_us.insert(
            out.samples.cmd_buffer_create_us.end(),
            local.cmd_buffer_create_us.begin(),
            local.cmd_buffer_create_us.end());
        out.samples.encoder_create_us.insert(
            out.samples.encoder_create_us.end(),
            local.encoder_create_us.begin(),
            local.encoder_create_us.end());
        out.samples.commit_us.insert(
            out.samples.commit_us.end(), local.commit_us.begin(), local.commit_us.end());
        out.samples.wait_us.insert(
            out.samples.wait_us.end(), local.wait_us.begin(), local.wait_us.end());
        out.samples.gpu_us.insert(out.samples.gpu_us.end(), local.gpu_us.begin(), local.gpu_us.end());
      });
    }

    for (auto& t : workers) {
      t.join();
    }

    auto end_wall = std::chrono::steady_clock::now();
    out.wall_time_s = std::chrono::duration<double>(end_wall - start_wall).count();
    return out;
  }
}

void print_json(const RunResult& r) {
  const int total_ops = r.threads * r.iters;
  const double throughput = r.wall_time_s > 0 ? double(total_ops) / r.wall_time_s : 0.0;

  auto mean = [](const std::vector<double>& v) -> double {
    if (v.empty()) {
      return 0.0;
    }
    const double sum = std::accumulate(v.begin(), v.end(), 0.0);
    return sum / double(v.size());
  };

  const auto& s = r.samples;

  std::cout << "{\n";
  std::cout << "  \"threads\": " << r.threads << ",\n";
  std::cout << "  \"iters\": " << r.iters << ",\n";
  std::cout << "  \"elements\": " << r.elements << ",\n";
  std::cout << "  \"total_ops\": " << total_ops << ",\n";
  std::cout << "  \"wall_time_s\": " << std::fixed << std::setprecision(6) << r.wall_time_s
            << ",\n";
  std::cout << "  \"throughput_ops_s\": " << std::fixed << std::setprecision(3) << throughput
            << ",\n";

  auto emit_stats = [&](const char* name, const std::vector<double>& v) {
    std::cout << "  \"" << name << "\": {\n";
    std::cout << "    \"count\": " << v.size() << ",\n";
    std::cout << "    \"mean_us\": " << std::fixed << std::setprecision(3) << mean(v) << ",\n";
    std::cout << "    \"p50_us\": " << std::fixed << std::setprecision(3) << percentile_us(v, 50)
              << ",\n";
    std::cout << "    \"p95_us\": " << std::fixed << std::setprecision(3) << percentile_us(v, 95)
              << "\n";
    std::cout << "  }";
  };

  emit_stats("cmd_buffer_create", s.cmd_buffer_create_us);
  std::cout << ",\n";
  emit_stats("encoder_create", s.encoder_create_us);
  std::cout << ",\n";
  emit_stats("commit", s.commit_us);
  std::cout << ",\n";
  emit_stats("wait", s.wait_us);
  std::cout << ",\n";
  emit_stats("gpu", s.gpu_us);
  std::cout << "\n";
  std::cout << "}\n";
}

} // namespace

int main(int argc, const char* argv[]) {
  @autoreleasepool {
    int threads = 8;
    int iters = 200;
    std::uint32_t elements = 262144; // 1MB of floats
    bool wait_for_completion = true;

    for (int i = 1; i < argc; i++) {
      const std::string arg(argv[i]);
      auto require_value = [&](const char* flag) -> const char* {
        if (i + 1 >= argc) {
          throw std::runtime_error(std::string("missing value for ") + flag);
        }
        return argv[++i];
      };

      if (arg == "--threads") {
        threads = std::atoi(require_value("--threads"));
      } else if (arg == "--iters") {
        iters = std::atoi(require_value("--iters"));
      } else if (arg == "--elements") {
        elements = static_cast<std::uint32_t>(std::strtoul(require_value("--elements"), nullptr, 10));
      } else if (arg == "--no-wait") {
        wait_for_completion = false;
      } else if (arg == "--help" || arg == "-h") {
        std::cerr << "Usage: metal_pure_objc_repro [--threads N] [--iters K] [--elements E] [--no-wait]\n";
        return 0;
      } else {
        std::cerr << "Unknown argument: " << arg << "\n";
        return 2;
      }
    }

    if (threads <= 0 || iters <= 0 || elements == 0) {
      std::cerr << "Invalid arguments\n";
      return 2;
    }

    try {
      const RunResult r = run_once(threads, iters, elements, wait_for_completion);
      print_json(r);
      return 0;
    } catch (const std::exception& e) {
      std::cerr << "ERROR: " << e.what() << "\n";
      return 1;
    }
  }
}

