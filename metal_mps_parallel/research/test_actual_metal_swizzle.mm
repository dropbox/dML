// test_actual_metal_swizzle.mm - Test if ACTUAL Metal encoder methods respect swizzle
//
// This tests the REAL question: when we swizzle MTLComputeCommandEncoder methods,
// do actual Metal calls go through our swizzle?
//
// Build: clang++ -framework Foundation -framework Metal -fobjc-arc -O0 test_actual_metal_swizzle.mm -o test_actual_metal_swizzle
// Run: ./test_actual_metal_swizzle

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>
#import <atomic>
#import <iostream>

static std::atomic<int> g_swizzle_calls{0};
static IMP g_original_setBytes = nullptr;

// Our swizzled implementation
static void swizzled_setBytes(id self, SEL _cmd, const void* bytes, NSUInteger length, NSUInteger index) {
    g_swizzle_calls++;
    std::cout << "  [SWIZZLE INTERCEPTED] setBytes:length:atIndex: called\n";

    // Call original
    if (g_original_setBytes) {
        ((void(*)(id, SEL, const void*, NSUInteger, NSUInteger))g_original_setBytes)(self, _cmd, bytes, length, index);
    }
}

void perform_swizzle_on_real_encoder(Class encoderClass) {
    SEL sel = @selector(setBytes:length:atIndex:);
    Method method = class_getInstanceMethod(encoderClass, sel);

    if (!method) {
        std::cout << "ERROR: Could not find setBytes:length:atIndex: on "
                  << class_getName(encoderClass) << "\n";
        return;
    }

    g_original_setBytes = method_getImplementation(method);
    std::cout << "Original IMP: " << (void*)g_original_setBytes << "\n";

    IMP newIMP = (IMP)swizzled_setBytes;
    method_setImplementation(method, newIMP);

    IMP verifyIMP = method_getImplementation(method);
    std::cout << "After swizzle, method table IMP: " << (void*)verifyIMP << "\n";

    if (verifyIMP == newIMP) {
        std::cout << "SUCCESS: Method table updated\n";
    } else {
        std::cout << "ERROR: Method table not updated!\n";
    }
}

int main() {
    @autoreleasepool {
        std::cout << "============================================\n";
        std::cout << "ACTUAL METAL SWIZZLE TEST\n";
        std::cout << "============================================\n\n";

        // Get Metal device
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cout << "ERROR: No Metal device available\n";
            return 1;
        }
        std::cout << "Device: " << [[device name] UTF8String] << "\n\n";

        // Create command queue
        id<MTLCommandQueue> queue = [device newCommandQueue];

        // Create a command buffer and encoder to discover the actual class
        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

        Class actualEncoderClass = [encoder class];
        std::cout << "Actual encoder class: " << class_getName(actualEncoderClass) << "\n\n";

        // End this encoder, we'll create a new one after swizzle
        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        // NOW perform swizzle
        std::cout << "=== Performing swizzle on " << class_getName(actualEncoderClass) << " ===\n";
        perform_swizzle_on_real_encoder(actualEncoderClass);

        // Reset counter
        g_swizzle_calls = 0;

        // Create NEW encoder AFTER swizzle
        std::cout << "\n=== Creating new encoder AFTER swizzle ===\n";
        cmdBuf = [queue commandBuffer];
        encoder = [cmdBuf computeCommandEncoder];

        // Call the swizzled method
        std::cout << "\n=== Calling setBytes:length:atIndex: ===\n";
        uint32_t testData = 42;
        [encoder setBytes:&testData length:sizeof(testData) atIndex:0];

        // Check if swizzle was called
        std::cout << "\n=== RESULTS ===\n";
        std::cout << "Swizzle intercept count: " << g_swizzle_calls.load() << "\n";

        if (g_swizzle_calls == 1) {
            std::cout << "\nSWIZZLE WORKS ON REAL METAL ENCODER!\n";
            std::cout << "The actual Metal encoder method went through our swizzle.\n";
        } else if (g_swizzle_calls == 0) {
            std::cout << "\nSWIZZLE BYPASSED!\n";
            std::cout << "The Metal encoder call did NOT go through our swizzle.\n";
            std::cout << "This confirms IMP caching bypass is a REAL concern.\n";
        } else {
            std::cout << "\nUNEXPECTED: swizzle called " << g_swizzle_calls.load() << " times\n";
        }

        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        // Test 2: Multiple calls
        std::cout << "\n=== Test 2: Multiple calls ===\n";
        g_swizzle_calls = 0;

        cmdBuf = [queue commandBuffer];
        encoder = [cmdBuf computeCommandEncoder];

        for (int i = 0; i < 5; i++) {
            uint32_t data = i;
            [encoder setBytes:&data length:sizeof(data) atIndex:0];
        }

        std::cout << "Made 5 calls, swizzle intercepted: " << g_swizzle_calls.load() << "\n";

        if (g_swizzle_calls == 5) {
            std::cout << "ALL calls went through swizzle.\n";
        } else {
            std::cout << "SOME calls bypassed swizzle!\n";
        }

        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        std::cout << "\n============================================\n";
        std::cout << "CONCLUSION\n";
        std::cout << "============================================\n";

        if (g_swizzle_calls == 5) {
            std::cout << "For encoders created AFTER swizzle, all calls go through swizzle.\n";
            std::cout << "This suggests Metal does NOT pre-cache IMPs in a way that bypasses swizzle.\n";
            std::cout << "\nHOWEVER: This does not prove safety because:\n";
            std::cout << "1. Metal might cache IMPs lazily on first call\n";
            std::cout << "2. Internal Metal code paths might differ\n";
            std::cout << "3. AGX driver internal calls are not tested\n";
        }

        return 0;
    }
}
