// test_encoder_created_before_swizzle.mm - The REAL edge case
//
// This tests: if an encoder is created BEFORE swizzle,
// do subsequent calls on that encoder go through swizzle?
//
// This matters because Metal might cache IMPs at encoder creation time.
//
// Build: clang++ -framework Foundation -framework Metal -fobjc-arc -O0 test_encoder_created_before_swizzle.mm -o test_encoder_created_before_swizzle
// Run: ./test_encoder_created_before_swizzle

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>
#import <atomic>
#import <iostream>

static std::atomic<int> g_swizzle_calls{0};
static IMP g_original_setBytes = nullptr;

static void swizzled_setBytes(id self, SEL _cmd, const void* bytes, NSUInteger length, NSUInteger index) {
    g_swizzle_calls++;
    if (g_original_setBytes) {
        ((void(*)(id, SEL, const void*, NSUInteger, NSUInteger))g_original_setBytes)(self, _cmd, bytes, length, index);
    }
}

void perform_swizzle(Class cls) {
    SEL sel = @selector(setBytes:length:atIndex:);
    Method method = class_getInstanceMethod(cls, sel);
    if (!method) return;
    g_original_setBytes = method_setImplementation(method, (IMP)swizzled_setBytes);
}

int main() {
    @autoreleasepool {
        std::cout << "============================================\n";
        std::cout << "ENCODER CREATED BEFORE SWIZZLE TEST\n";
        std::cout << "============================================\n\n";

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cout << "ERROR: No Metal device\n";
            return 1;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];

        // Create encoder BEFORE swizzle
        std::cout << "=== Step 1: Create encoder BEFORE swizzle ===\n";
        id<MTLComputeCommandEncoder> encoderBeforeSwizzle = [cmdBuf computeCommandEncoder];
        Class encoderClass = [encoderBeforeSwizzle class];
        std::cout << "Encoder class: " << class_getName(encoderClass) << "\n";
        std::cout << "Encoder created: " << (__bridge void*)encoderBeforeSwizzle << "\n";

        // NOW perform swizzle
        std::cout << "\n=== Step 2: Perform swizzle ===\n";
        perform_swizzle(encoderClass);
        std::cout << "Swizzle complete.\n";

        // Call method on encoder that was created BEFORE swizzle
        std::cout << "\n=== Step 3: Call method on PRE-SWIZZLE encoder ===\n";
        g_swizzle_calls = 0;
        uint32_t data = 42;
        [encoderBeforeSwizzle setBytes:&data length:sizeof(data) atIndex:0];

        std::cout << "Swizzle intercept count: " << g_swizzle_calls.load() << "\n";

        if (g_swizzle_calls == 1) {
            std::cout << "\nSWIZZLE WORKS even on encoder created before swizzle!\n";
            std::cout << "This means Metal does NOT cache IMPs at encoder creation time.\n";
        } else {
            std::cout << "\nSWIZZLE BYPASSED on pre-swizzle encoder!\n";
            std::cout << "Metal caches IMPs at encoder creation - this is the vulnerability.\n";
        }

        [encoderBeforeSwizzle endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        // Compare with encoder created AFTER swizzle
        std::cout << "\n=== Step 4: Create encoder AFTER swizzle for comparison ===\n";
        cmdBuf = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoderAfterSwizzle = [cmdBuf computeCommandEncoder];

        g_swizzle_calls = 0;
        [encoderAfterSwizzle setBytes:&data length:sizeof(data) atIndex:0];

        std::cout << "Swizzle intercept count: " << g_swizzle_calls.load() << "\n";

        if (g_swizzle_calls == 1) {
            std::cout << "Post-swizzle encoder: swizzle works (expected).\n";
        }

        [encoderAfterSwizzle endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        std::cout << "\n============================================\n";
        std::cout << "FINAL VERDICT\n";
        std::cout << "============================================\n";

        return 0;
    }
}
