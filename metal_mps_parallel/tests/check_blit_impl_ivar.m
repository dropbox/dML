// Check if blit encoder has _impl ivar like compute encoder
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

        // Check compute encoder
        id<MTLComputeCommandEncoder> computeEncoder = [cmdBuffer computeCommandEncoder];
        Class computeClass = [computeEncoder class];
        printf("Compute encoder class: %s\n", class_getName(computeClass));

        Ivar computeImpl = class_getInstanceVariable(computeClass, "_impl");
        if (computeImpl) {
            printf("  _impl ivar: EXISTS at offset %td\n", ivar_getOffset(computeImpl));
        } else {
            // Check parent
            Class parent = class_getSuperclass(computeClass);
            while (parent) {
                computeImpl = class_getInstanceVariable(parent, "_impl");
                if (computeImpl) {
                    printf("  _impl ivar: EXISTS in parent %s at offset %td\n",
                           class_getName(parent), ivar_getOffset(computeImpl));
                    break;
                }
                parent = class_getSuperclass(parent);
            }
            if (!computeImpl) printf("  _impl ivar: NOT FOUND\n");
        }
        [computeEncoder endEncoding];

        // Check blit encoder
        id<MTLCommandBuffer> cmdBuffer2 = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blitEncoder = [cmdBuffer2 blitCommandEncoder];
        Class blitClass = [blitEncoder class];
        printf("\nBlit encoder class: %s\n", class_getName(blitClass));

        Ivar blitImpl = class_getInstanceVariable(blitClass, "_impl");
        if (blitImpl) {
            printf("  _impl ivar: EXISTS at offset %td\n", ivar_getOffset(blitImpl));
        } else {
            // Check parent
            Class parent = class_getSuperclass(blitClass);
            while (parent) {
                blitImpl = class_getInstanceVariable(parent, "_impl");
                if (blitImpl) {
                    printf("  _impl ivar: EXISTS in parent %s at offset %td\n",
                           class_getName(parent), ivar_getOffset(blitImpl));
                    break;
                }
                parent = class_getSuperclass(parent);
            }
            if (!blitImpl) printf("  _impl ivar: NOT FOUND\n");
        }
        [blitEncoder endEncoding];

        printf("\n=== ANALYSIS ===\n");
        if (computeImpl && blitImpl) {
            ptrdiff_t computeOffset = ivar_getOffset(computeImpl);
            ptrdiff_t blitOffset = ivar_getOffset(blitImpl);
            if (computeOffset == blitOffset) {
                printf("Both encoders have _impl at same offset (%td) - v2.3 check will work\n", computeOffset);
            } else {
                printf("WARNING: _impl at different offsets! compute=%td, blit=%td\n",
                       computeOffset, blitOffset);
                printf("v2.3 may need separate _impl offset tracking for blit encoders\n");
            }
        } else if (!blitImpl) {
            printf("Blit encoder has no _impl ivar - v2.3 _impl check may not apply\n");
            printf("This is OK - the check will be skipped (returns true)\n");
        }

        return 0;
    }
}
