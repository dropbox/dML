/**
 * Test raw Metal compute parallelism
 *
 * Compares sequential vs parallel command buffer submission to
 * understand if serialization is in Metal driver or MPS specifically.
 *
 * Build:
 *   clang -o test_raw_metal test_raw_metal.m \
 *       -framework Foundation -framework Metal -fobjc-arc
 *
 * Run:
 *   ./test_raw_metal
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <dispatch/dispatch.h>
#import <mach/mach_time.h>
#import <pthread.h>

static uint64_t g_timebase_num = 0;
static uint64_t g_timebase_denom = 0;

static double mach_to_seconds(uint64_t mach_time) {
    return (double)(mach_time * g_timebase_num) / (double)(g_timebase_denom * 1000000000ULL);
}

// Sequential test - baseline
static double benchmark_sequential(id<MTLDevice> device, int num_iters) {
    id<MTLCommandQueue> queue = [device newCommandQueue];

    uint64_t start = mach_absolute_time();

    for (int i = 0; i < num_iters; i++) {
        id<MTLCommandBuffer> buffer = [queue commandBuffer];
        [buffer commit];
        [buffer waitUntilCompleted];
    }

    uint64_t end = mach_absolute_time();
    double elapsed = mach_to_seconds(end - start);

    return num_iters / elapsed;
}

// Parallel test with separate queues per thread
static double benchmark_parallel_separate_queues(id<MTLDevice> device, int num_threads, int iters_per_thread) {
    __block int total_ops = 0;
    dispatch_queue_t dispatch_queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0);
    dispatch_group_t group = dispatch_group_create();

    uint64_t start = mach_absolute_time();

    for (int t = 0; t < num_threads; t++) {
        dispatch_group_async(group, dispatch_queue, ^{
            // Each thread gets its own command queue
            id<MTLCommandQueue> queue = [device newCommandQueue];

            for (int i = 0; i < iters_per_thread; i++) {
                id<MTLCommandBuffer> buffer = [queue commandBuffer];
                [buffer commit];
                [buffer waitUntilCompleted];
            }

            @synchronized(@(total_ops)) {
                total_ops += iters_per_thread;
            }
        });
    }

    dispatch_group_wait(group, DISPATCH_TIME_FOREVER);

    uint64_t end = mach_absolute_time();
    double elapsed = mach_to_seconds(end - start);

    return total_ops / elapsed;
}

// Parallel test with shared command queue
static double benchmark_parallel_shared_queue(id<MTLDevice> device, int num_threads, int iters_per_thread) {
    id<MTLCommandQueue> queue = [device newCommandQueue];
    __block int total_ops = 0;
    dispatch_queue_t dispatch_queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0);
    dispatch_group_t group = dispatch_group_create();

    uint64_t start = mach_absolute_time();

    for (int t = 0; t < num_threads; t++) {
        dispatch_group_async(group, dispatch_queue, ^{
            for (int i = 0; i < iters_per_thread; i++) {
                id<MTLCommandBuffer> buffer = [queue commandBuffer];
                [buffer commit];
                [buffer waitUntilCompleted];
            }

            @synchronized(@(total_ops)) {
                total_ops += iters_per_thread;
            }
        });
    }

    dispatch_group_wait(group, DISPATCH_TIME_FOREVER);

    uint64_t end = mach_absolute_time();
    double elapsed = mach_to_seconds(end - start);

    return total_ops / elapsed;
}

// Parallel test with pthread (to match PyTorch's threading model)
typedef struct {
    id<MTLDevice> device;
    int iters;
    int use_shared_queue;
    id<MTLCommandQueue> shared_queue;
} thread_args_t;

static void* pthread_worker(void* arg) {
    thread_args_t* args = (thread_args_t*)arg;

    id<MTLCommandQueue> queue = args->use_shared_queue ?
        args->shared_queue : [args->device newCommandQueue];

    for (int i = 0; i < args->iters; i++) {
        @autoreleasepool {
            id<MTLCommandBuffer> buffer = [queue commandBuffer];
            [buffer commit];
            [buffer waitUntilCompleted];
        }
    }

    return NULL;
}

static double benchmark_pthread_parallel(id<MTLDevice> device, int num_threads, int iters_per_thread, BOOL shared_queue) {
    pthread_t threads[num_threads];
    thread_args_t args[num_threads];

    id<MTLCommandQueue> queue = shared_queue ? [device newCommandQueue] : nil;

    uint64_t start = mach_absolute_time();

    for (int t = 0; t < num_threads; t++) {
        args[t].device = device;
        args[t].iters = iters_per_thread;
        args[t].use_shared_queue = shared_queue;
        args[t].shared_queue = queue;
        pthread_create(&threads[t], NULL, pthread_worker, &args[t]);
    }

    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }

    uint64_t end = mach_absolute_time();
    double elapsed = mach_to_seconds(end - start);

    return (num_threads * iters_per_thread) / elapsed;
}

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        // Initialize timebase
        mach_timebase_info_data_t timebase;
        mach_timebase_info(&timebase);
        g_timebase_num = timebase.numer;
        g_timebase_denom = timebase.denom;

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            NSLog(@"ERROR: No Metal device available");
            return 1;
        }

        NSLog(@"Device: %@", device.name);
        NSLog(@"========================================");

        // Warmup
        benchmark_sequential(device, 100);

        // Sequential baseline
        double seq_ops = benchmark_sequential(device, 2000);
        NSLog(@"Sequential (1 thread):         %.0f ops/s", seq_ops);
        NSLog(@"");

        // Parallel with GCD + separate queues
        NSLog(@"--- GCD Parallel (separate queues) ---");
        for (int n = 2; n <= 8; n *= 2) {
            double ops = benchmark_parallel_separate_queues(device, n, 500);
            double scaling = ops / seq_ops;
            double efficiency = scaling / n * 100;
            NSLog(@"%d threads: %.0f ops/s (%.2fx, %.1f%% eff)", n, ops, scaling, efficiency);
        }
        NSLog(@"");

        // Parallel with GCD + shared queue
        NSLog(@"--- GCD Parallel (shared queue) ---");
        for (int n = 2; n <= 8; n *= 2) {
            double ops = benchmark_parallel_shared_queue(device, n, 500);
            double scaling = ops / seq_ops;
            double efficiency = scaling / n * 100;
            NSLog(@"%d threads: %.0f ops/s (%.2fx, %.1f%% eff)", n, ops, scaling, efficiency);
        }
        NSLog(@"");

        // Parallel with pthreads + separate queues
        NSLog(@"--- pthread Parallel (separate queues) ---");
        for (int n = 2; n <= 8; n *= 2) {
            double ops = benchmark_pthread_parallel(device, n, 500, NO);
            double scaling = ops / seq_ops;
            double efficiency = scaling / n * 100;
            NSLog(@"%d threads: %.0f ops/s (%.2fx, %.1f%% eff)", n, ops, scaling, efficiency);
        }
        NSLog(@"");

        // Parallel with pthreads + shared queue
        NSLog(@"--- pthread Parallel (shared queue) ---");
        for (int n = 2; n <= 8; n *= 2) {
            double ops = benchmark_pthread_parallel(device, n, 500, YES);
            double scaling = ops / seq_ops;
            double efficiency = scaling / n * 100;
            NSLog(@"%d threads: %.0f ops/s (%.2fx, %.1f%% eff)", n, ops, scaling, efficiency);
        }
    }

    return 0;
}
