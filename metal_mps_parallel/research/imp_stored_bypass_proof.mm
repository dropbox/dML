// imp_stored_bypass_proof.mm - Proof that stored IMPs bypass swizzles
//
// This tests the ACTUAL concern: if code stores an IMP in a variable
// BEFORE swizzle, that stored IMP is NOT affected by swizzle.
//
// This is what Metal.framework might do during initialization.
//
// Build: clang++ -framework Foundation -fobjc-arc -O0 imp_stored_bypass_proof.mm -o imp_stored_bypass_proof
// Run: ./imp_stored_bypass_proof

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <atomic>
#import <iostream>

static std::atomic<int> g_original_calls{0};
static std::atomic<int> g_swizzled_calls{0};

@interface TestEncoder : NSObject
- (void)setBuffer:(id)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index;
@end

@implementation TestEncoder
- (void)setBuffer:(id)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index {
    g_original_calls++;
}
@end

static void swizzled_setBuffer(id self, SEL _cmd, id buffer, NSUInteger offset, NSUInteger index) {
    g_swizzled_calls++;
}

// Simulate Metal.framework storing an IMP during initialization
typedef void (*SetBufferIMP)(id, SEL, id, NSUInteger, NSUInteger);
static SetBufferIMP g_cached_imp = nullptr;

void metal_framework_init() {
    // This simulates what Metal.framework might do during +load or initialization:
    // Store the IMP in a global variable for fast dispatch
    Class cls = [TestEncoder class];
    SEL sel = @selector(setBuffer:offset:atIndex:);
    g_cached_imp = (SetBufferIMP)class_getMethodImplementation(cls, sel);
    std::cout << "Metal init: Cached IMP = " << (void*)g_cached_imp << "\n";
}

void metal_internal_call(TestEncoder* encoder) {
    // This simulates Metal.framework calling the method using its cached IMP
    // This is a common optimization in performance-critical code
    if (g_cached_imp) {
        g_cached_imp(encoder, @selector(setBuffer:offset:atIndex:), nil, 0, 0);
    }
}

void agx_fix_swizzle() {
    // This simulates our AGX fix swizzling the method
    Class cls = [TestEncoder class];
    SEL sel = @selector(setBuffer:offset:atIndex:);
    Method method = class_getInstanceMethod(cls, sel);

    IMP old_imp = method_setImplementation(method, (IMP)swizzled_setBuffer);
    std::cout << "AGX fix: Swizzled method. Old IMP = " << (void*)old_imp << "\n";

    IMP new_imp = class_getMethodImplementation(cls, sel);
    std::cout << "AGX fix: New IMP in method table = " << (void*)new_imp << "\n";
}

int main() {
    @autoreleasepool {
        std::cout << "============================================\n";
        std::cout << "STORED IMP BYPASS PROOF\n";
        std::cout << "============================================\n";
        std::cout << "This proves that IMPs stored before swizzle bypass the swizzle.\n\n";

        TestEncoder* encoder = [[TestEncoder alloc] init];

        // Step 1: Simulate Metal.framework initialization (stores IMP)
        std::cout << "=== Step 1: Simulate Metal.framework storing IMP ===\n";
        metal_framework_init();

        // Step 2: Simulate our dylib loading and swizzling
        std::cout << "\n=== Step 2: Our AGX fix swizzles the method ===\n";
        agx_fix_swizzle();

        // Step 3: Normal objc_msgSend call (should use swizzled IMP)
        std::cout << "\n=== Step 3: Normal [obj method] call ===\n";
        g_original_calls = 0;
        g_swizzled_calls = 0;
        [encoder setBuffer:nil offset:0 atIndex:0];
        std::cout << "Result: original=" << g_original_calls.load()
                  << " swizzled=" << g_swizzled_calls.load() << "\n";
        if (g_swizzled_calls == 1) {
            std::cout << "PASS: Normal call uses swizzled implementation.\n";
        }

        // Step 4: Metal's internal call using cached IMP (bypasses swizzle!)
        std::cout << "\n=== Step 4: Metal's internal call using CACHED IMP ===\n";
        g_original_calls = 0;
        g_swizzled_calls = 0;
        metal_internal_call(encoder);
        std::cout << "Result: original=" << g_original_calls.load()
                  << " swizzled=" << g_swizzled_calls.load() << "\n";

        std::cout << "\n============================================\n";
        std::cout << "CONCLUSION\n";
        std::cout << "============================================\n";

        if (g_original_calls == 1 && g_swizzled_calls == 0) {
            std::cout << "BYPASS CONFIRMED!\n";
            std::cout << "When code stores an IMP before swizzle, the stored IMP\n";
            std::cout << "is NOT affected by method_setImplementation.\n\n";
            std::cout << "IMPLICATIONS FOR AGX FIX:\n";
            std::cout << "  - If Metal.framework caches IMPs during initialization,\n";
            std::cout << "  - those cached IMPs will bypass our swizzle.\n";
            std::cout << "  - We have NO WAY to detect this from userspace.\n";
            std::cout << "  - This is UNFALSIFIABLE - we cannot prove all calls\n";
            std::cout << "    go through our swizzle.\n";
        } else if (g_swizzled_calls == 1) {
            std::cout << "UNEXPECTED: Cached IMP somehow used swizzled version.\n";
            std::cout << "This should not happen - need to investigate.\n";
        } else {
            std::cout << "UNEXPECTED RESULT: original=" << g_original_calls.load()
                      << " swizzled=" << g_swizzled_calls.load() << "\n";
        }

        // Additional proof: Check if the cached IMP equals original
        std::cout << "\n=== ADDITIONAL VERIFICATION ===\n";
        IMP current_method_imp = class_getMethodImplementation([TestEncoder class],
                                                               @selector(setBuffer:offset:atIndex:));
        std::cout << "Current method table IMP: " << (void*)current_method_imp << "\n";
        std::cout << "Metal's cached IMP:       " << (void*)g_cached_imp << "\n";
        std::cout << "Swizzled function addr:   " << (void*)swizzled_setBuffer << "\n";

        if ((void*)current_method_imp != (void*)g_cached_imp) {
            std::cout << "\nPROOF: Method table IMP != Cached IMP\n";
            std::cout << "The cached IMP points to the ORIGINAL implementation.\n";
            std::cout << "Any code using the cached IMP bypasses our swizzle.\n";
        }

        return 0;
    }
}
