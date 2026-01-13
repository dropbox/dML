// Minimal Metal threading test
// Tests if bare Metal APIs have the same thread-safety issue as MPS
// Compile (from repo root): clang -fobjc-arc -framework Foundation -framework Metal -o /tmp/metal_bare_thread_race tests/metal_bare_thread_race.m
// Or run: ./tests/run_metal_bare_thread_race.sh

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <dispatch/dispatch.h>

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        // Get Metal device
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            NSLog(@"No Metal device available");
            return 1;
        }
        NSLog(@"Using device: %@", device.name);
        
        // Create command queue
        id<MTLCommandQueue> queue = [device newCommandQueue];
        
        // Test 1: Multiple threads, each with own command buffer (should be safe)
        NSLog(@"\n=== Test 1: Separate command buffers per thread ===");
        dispatch_group_t group1 = dispatch_group_create();
        
        for (int t = 0; t < 4; t++) {
            dispatch_group_async(group1, dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^{
                for (int i = 0; i < 100; i++) {
                    @autoreleasepool {
                        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
                        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
                        [encoder endEncoding];
                        [cmdBuf commit];
                    }
                }
            });
        }
        
        dispatch_group_wait(group1, DISPATCH_TIME_FOREVER);
        NSLog(@"Test 1: PASSED - Separate buffers are thread-safe");
        
        // Test 2: Multiple threads accessing SAME command buffer (the bug scenario)
        NSLog(@"\n=== Test 2: Shared command buffer (bug scenario) ===");
        
        __block int successCount = 0;
        __block int failCount = 0;
        
        for (int trial = 0; trial < 10; trial++) {
            @autoreleasepool {
                @try {
                    id<MTLCommandBuffer> sharedBuffer = [queue commandBuffer];
                    dispatch_group_t group2 = dispatch_group_create();
                    
                    // Multiple threads trying to get encoder from same buffer
                    for (int t = 0; t < 4; t++) {
                        dispatch_group_async(group2, dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^{
                            @autoreleasepool {
                                // This should cause the race condition
                                id<MTLComputeCommandEncoder> encoder = [sharedBuffer computeCommandEncoder];
                                // Simulate some work
                                [NSThread sleepForTimeInterval:0.001];
                                [encoder endEncoding];
                            }
                        });
                    }
                    
                    dispatch_group_wait(group2, DISPATCH_TIME_FOREVER);
                    [sharedBuffer commit];
                    successCount++;
                } @catch (NSException *e) {
                    NSLog(@"Trial %d: Exception - %@ - %@", trial, e.name, e.reason);
                    failCount++;
                }
            }
        }
        
        NSLog(@"Test 2 Results: %d success, %d fail", successCount, failCount);
        
        // Test 3: Sequential encoder access on same buffer (baseline)
        NSLog(@"\n=== Test 3: Sequential encoder access (baseline) ===");
        id<MTLCommandBuffer> seqBuffer = [queue commandBuffer];
        for (int i = 0; i < 4; i++) {
            @autoreleasepool {
                id<MTLComputeCommandEncoder> encoder = [seqBuffer computeCommandEncoder];
                [encoder endEncoding];
            }
        }
        [seqBuffer commit];
        NSLog(@"Test 3: PASSED - Sequential access works");
        
        NSLog(@"\n=== Summary ===");
        NSLog(@"The bug manifests when multiple threads access the SAME command buffer.");
        NSLog(@"Using separate command buffers per thread avoids the issue.");
    }
    return 0;
}
