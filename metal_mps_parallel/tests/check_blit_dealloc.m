// Check blit encoder cleanup methods
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            printf("ERROR: No Metal device\n");
            return 1;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];

        // Check compute encoder
        id<MTLCommandBuffer> cmdBuffer1 = [queue commandBuffer];
        id<MTLComputeCommandEncoder> computeEncoder = [cmdBuffer1 computeCommandEncoder];
        Class computeClass = [computeEncoder class];

        printf("=== COMPUTE ENCODER CLEANUP METHODS ===\n");
        Method m1 = class_getInstanceMethod(computeClass, @selector(destroyImpl));
        printf("destroyImpl: %s\n", m1 ? "EXISTS" : "MISSING");
        Method m2 = class_getInstanceMethod(computeClass, @selector(dealloc));
        printf("dealloc: %s\n", m2 ? "EXISTS" : "MISSING");
        [computeEncoder endEncoding];

        // Check blit encoder
        id<MTLCommandBuffer> cmdBuffer2 = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blitEncoder = [cmdBuffer2 blitCommandEncoder];
        Class blitClass = [blitEncoder class];

        printf("\n=== BLIT ENCODER CLEANUP METHODS ===\n");
        Method m3 = class_getInstanceMethod(blitClass, @selector(destroyImpl));
        printf("destroyImpl: %s\n", m3 ? "EXISTS" : "MISSING");
        Method m4 = class_getInstanceMethod(blitClass, @selector(dealloc));
        printf("dealloc: %s\n", m4 ? "EXISTS" : "MISSING");

        // Check parent for destroyImpl
        Class parent = class_getSuperclass(blitClass);
        while (parent) {
            Method pm = class_getInstanceMethod(parent, @selector(destroyImpl));
            if (pm) {
                printf("destroyImpl in parent %s: EXISTS\n", class_getName(parent));
                break;
            }
            parent = class_getSuperclass(parent);
        }

        [blitEncoder endEncoding];

        printf("\n=== ANALYSIS ===\n");
        if (!m3) {
            printf("Blit encoder has NO destroyImpl method\n");
            printf("This may be a GAP if abnormal cleanup is needed\n");
        }

        return 0;
    }
}
