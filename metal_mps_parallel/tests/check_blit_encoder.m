// Check blit encoder class and methods
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
        id<MTLCommandBuffer> cmdBuffer = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blitEncoder = [cmdBuffer blitCommandEncoder];

        if (!blitEncoder) {
            printf("ERROR: Failed to create blit encoder\n");
            return 1;
        }

        Class blitClass = [blitEncoder class];
        printf("=== BLIT ENCODER CLASS ===\n");
        printf("Blit encoder class: %s\n", class_getName(blitClass));

        // Check methods PyTorch uses
        printf("\n=== PyTorch-used methods ===\n");
        Method m1 = class_getInstanceMethod(blitClass, @selector(fillBuffer:range:value:));
        printf("fillBuffer:range:value: %s\n", m1 ? "EXISTS" : "MISSING");

        Method m2 = class_getInstanceMethod(blitClass, @selector(copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:));
        printf("copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size: %s\n", m2 ? "EXISTS" : "MISSING");

        Method m3 = class_getInstanceMethod(blitClass, @selector(endEncoding));
        printf("endEncoding: %s\n", m3 ? "EXISTS" : "MISSING");

        // List all methods containing common patterns
        printf("\n=== All methods on blit encoder ===\n");
        unsigned int count;
        Method* methods = class_copyMethodList(blitClass, &count);
        for (unsigned int i = 0; i < count; i++) {
            SEL sel = method_getName(methods[i]);
            printf("  %s\n", sel_getName(sel));
        }
        free(methods);

        // Check parent class methods
        Class parent = class_getSuperclass(blitClass);
        printf("\n=== Parent class: %s ===\n", class_getName(parent));
        methods = class_copyMethodList(parent, &count);
        for (unsigned int i = 0; i < count; i++) {
            SEL sel = method_getName(methods[i]);
            const char* name = sel_getName(sel);
            // Only show relevant methods
            if (strstr(name, "copy") || strstr(name, "fill") || strstr(name, "sync") ||
                strstr(name, "end") || strstr(name, "Encoding")) {
                printf("  %s\n", name);
            }
        }
        free(methods);

        [blitEncoder endEncoding];

        printf("\n=== VERIFICATION COMPLETE ===\n");
        return 0;
    }
}
