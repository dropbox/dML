// Verify v2.3 blit encoder coverage
// Compile: clang -framework Foundation -framework Metal -o verify_v2_3_blit_coverage verify_v2_3_blit_coverage.m
// Run: DYLD_INSERT_LIBRARIES=../agx_fix/build/libagx_fix_v2_3.dylib ./verify_v2_3_blit_coverage

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>

int main() {
    @autoreleasepool {
        printf("=== V2.3 BLIT ENCODER COVERAGE TEST ===\n\n");

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            printf("ERROR: No Metal device\n");
            return 1;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> cmdBuffer = [queue commandBuffer];

        // Create compute encoder to trigger v2.3 init
        id<MTLComputeCommandEncoder> computeEncoder = [cmdBuffer computeCommandEncoder];
        Class computeClass = [computeEncoder class];
        printf("Compute encoder class: %s\n", class_getName(computeClass));
        [computeEncoder endEncoding];

        // Now create blit encoder
        id<MTLCommandBuffer> cmdBuffer2 = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blitEncoder = [cmdBuffer2 blitCommandEncoder];
        Class blitClass = [blitEncoder class];
        printf("Blit encoder class: %s\n\n", class_getName(blitClass));

        // Test blit operations with actual buffers
        printf("=== TESTING BLIT OPERATIONS ===\n");

        id<MTLBuffer> srcBuffer = [device newBufferWithLength:1024 options:MTLResourceStorageModeShared];
        id<MTLBuffer> dstBuffer = [device newBufferWithLength:1024 options:MTLResourceStorageModeShared];

        // Fill source with test data
        memset(srcBuffer.contents, 0xAB, 1024);

        // Perform blit operations
        [blitEncoder fillBuffer:dstBuffer range:NSMakeRange(0, 512) value:0xCD];
        printf("fillBuffer:range:value: executed\n");

        [blitEncoder copyFromBuffer:srcBuffer sourceOffset:0
                           toBuffer:dstBuffer destinationOffset:512 size:512];
        printf("copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size: executed\n");

        [blitEncoder synchronizeResource:dstBuffer];
        printf("synchronizeResource: executed\n");

        [blitEncoder endEncoding];
        printf("endEncoding: executed\n");

        // Commit and wait
        [cmdBuffer2 commit];
        [cmdBuffer2 waitUntilCompleted];

        // Verify results
        uint8_t* dstData = (uint8_t*)dstBuffer.contents;
        int fillOK = 1, copyOK = 1;

        for (int i = 0; i < 512; i++) {
            if (dstData[i] != 0xCD) { fillOK = 0; break; }
        }
        for (int i = 512; i < 1024; i++) {
            if (dstData[i] != 0xAB) { copyOK = 0; break; }
        }

        printf("\n=== RESULTS ===\n");
        printf("fillBuffer result: %s\n", fillOK ? "PASS" : "FAIL");
        printf("copyFromBuffer result: %s\n", copyOK ? "PASS" : "FAIL");
        printf("Overall: %s\n", (fillOK && copyOK) ? "SUCCESS" : "FAILURE");

        return (fillOK && copyOK) ? 0 : 1;
    }
}
