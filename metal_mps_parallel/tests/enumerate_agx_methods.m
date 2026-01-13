// Gap 5 Test: Enumerate ALL AGX encoder methods
// Compare against v2.9 swizzled methods to find coverage gaps
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>
#import <stdio.h>
#import <string.h>

// Known swizzled methods in v2.9 (from agx_fix_v2_9.mm)
static const char* g_compute_swizzled[] = {
    "destroyImpl",
    "endEncoding",
    "deferredEndEncoding",
    "dealloc",
    "setComputePipelineState:",
    "dispatchThreads:threadsPerThreadgroup:",
    "dispatchThreadgroups:threadsPerThreadgroup:",
    "setBuffer:offset:atIndex:",
    "setBuffers:offsets:withRange:",
    "setBytes:length:atIndex:",
    "setTexture:atIndex:",
    "setTextures:withRange:",
    "setSamplerState:atIndex:",
    "setSamplerStates:withRange:",
    "setThreadgroupMemoryLength:atIndex:",
    "setStageInRegion:",
    "setImageblockWidth:height:",
    "setBufferOffset:atIndex:",
    "dispatchWaitFlush",
    "dispatchFlushInvalidate",
    "dispatchFlushOnly",
    "dispatchInvalidateOnly",
    "dispatchFenceOnly",
    "dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:",
    "dispatchThreadsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:",
    "setVisibleFunctionTable:atBufferIndex:",
    "setVisibleFunctionTables:withBufferRange:",
    "setIntersectionFunctionTable:atBufferIndex:",
    "setIntersectionFunctionTables:withBufferRange:",
    "setAccelerationStructure:atBufferIndex:",
    "setBuffer:offset:attributeStride:atIndex:",
    "setBuffers:offsets:attributeStrides:withRange:",
    "setFunction:atIndex:",
    "setSamplerState:lodMinClamp:lodMaxClamp:atIndex:",
    "setSamplerStates:lodMinClamps:lodMaxClamps:withRange:",
    "setStageInRegionWithIndirectBuffer:indirectBufferOffset:",
    "useResource:usage:",
    "useResources:count:usage:",
    "useHeap:",
    "useHeaps:count:",
    "executeCommandsInBuffer:withRange:",
    "executeCommandsInBuffer:indirectBuffer:indirectBufferOffset:",
    "memoryBarrierWithScope:",
    "memoryBarrierWithResources:count:",
    NULL
};

static const char* g_blit_swizzled[] = {
    "endEncoding",
    "deferredEndEncoding",
    "dealloc",
    "fillBuffer:range:value:",
    "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:",
    "synchronizeResource:",
    NULL
};

static const char* g_render_swizzled[] = {
    "endEncoding",
    "deferredEndEncoding",
    "dealloc",
    "setVertexBuffer:offset:atIndex:",
    "setVertexBytes:length:atIndex:",
    "setFragmentBuffer:offset:atIndex:",
    "setFragmentBytes:length:atIndex:",
    "setVertexTexture:atIndex:",
    "setFragmentTexture:atIndex:",
    "setRenderPipelineState:",
    "drawPrimitives:vertexStart:vertexCount:",
    "drawPrimitives:vertexStart:vertexCount:instanceCount:",
    NULL
};

typedef struct {
    int total;
    int swizzled;
    int not_swizzled;
    const char** not_swizzled_list;
    int not_swizzled_capacity;
} CoverageStats;

void init_stats(CoverageStats* stats) {
    stats->total = 0;
    stats->swizzled = 0;
    stats->not_swizzled = 0;
    stats->not_swizzled_capacity = 256;
    stats->not_swizzled_list = malloc(sizeof(char*) * stats->not_swizzled_capacity);
}

void free_stats(CoverageStats* stats) {
    for (int i = 0; i < stats->not_swizzled; i++) {
        free((void*)stats->not_swizzled_list[i]);
    }
    free(stats->not_swizzled_list);
}

int is_method_swizzled(const char* method_name, const char** swizzle_list) {
    for (int i = 0; swizzle_list[i] != NULL; i++) {
        if (strcmp(method_name, swizzle_list[i]) == 0) {
            return 1;
        }
    }
    return 0;
}

int is_method_ignorable(const char* method_name) {
    // Methods we don't need to swizzle (internal, inherited from NSObject, etc.)
    static const char* ignore_prefixes[] = {
        ".",           // Hidden methods
        "_",           // Private Apple methods
        "class",       // NSObject basics
        "isKindOf",
        "respondsTo",
        "perform",
        "description",
        "hash",
        "zone",
        "retain",
        "release",
        "autorelease",
        "retainCount",
        "self",
        "init",        // Initialization (handled separately)
        "alloc",
        "copy",
        "mutableCopy",
        "dealloc",     // Listed in swizzled
        NULL
    };

    for (int i = 0; ignore_prefixes[i] != NULL; i++) {
        if (strncmp(method_name, ignore_prefixes[i], strlen(ignore_prefixes[i])) == 0) {
            return 1;
        }
    }
    return 0;
}

void analyze_class_methods(Class cls, const char* encoder_type, const char** swizzle_list, CoverageStats* stats) {
    printf("\n=== %s Methods Analysis ===\n", encoder_type);
    printf("Class: %s\n\n", class_getName(cls));

    unsigned int count;
    Method* methods = class_copyMethodList(cls, &count);

    printf("%-60s | %-10s\n", "Method", "Status");
    printf("%-60s-+-%-10s\n", "------------------------------------------------------------", "----------");

    for (unsigned int i = 0; i < count; i++) {
        SEL sel = method_getName(methods[i]);
        const char* name = sel_getName(sel);

        if (is_method_ignorable(name)) {
            continue;  // Skip known non-encoder methods
        }

        stats->total++;

        if (is_method_swizzled(name, swizzle_list)) {
            printf("%-60s | SWIZZLED\n", name);
            stats->swizzled++;
        } else {
            printf("%-60s | NOT SWIZZLED\n", name);
            stats->not_swizzled++;
            if (stats->not_swizzled < stats->not_swizzled_capacity) {
                stats->not_swizzled_list[stats->not_swizzled - 1] = strdup(name);
            }
        }
    }

    free(methods);

    printf("\n%s Coverage: %d/%d methods swizzled (%.1f%%)\n",
           encoder_type, stats->swizzled, stats->total,
           stats->total > 0 ? 100.0 * stats->swizzled / stats->total : 0.0);

    if (stats->not_swizzled > 0) {
        printf("\n*** UNPROTECTED METHODS ***\n");
        for (int i = 0; i < stats->not_swizzled && i < stats->not_swizzled_capacity; i++) {
            printf("  - %s\n", stats->not_swizzled_list[i]);
        }
    }
}

int main(int argc, char* argv[]) {
    @autoreleasepool {
        printf("=== AGX Method Coverage Enumeration (Gap 5 Test) ===\n");
        printf("Date: %s\n\n", __DATE__);

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            printf("ERROR: No Metal device available\n");
            return 1;
        }
        printf("Device: %s\n", [[device name] UTF8String]);

        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> cmdBuffer = [queue commandBuffer];

        // Get compute encoder class
        id<MTLComputeCommandEncoder> computeEncoder = [cmdBuffer computeCommandEncoder];
        Class computeEncoderClass = [computeEncoder class];
        [computeEncoder endEncoding];

        // Get blit encoder class
        id<MTLBlitCommandEncoder> blitEncoder = [cmdBuffer blitCommandEncoder];
        Class blitEncoderClass = [blitEncoder class];
        [blitEncoder endEncoding];

        // Get render encoder class (need a render pass descriptor)
        MTLRenderPassDescriptor* rpDesc = [[MTLRenderPassDescriptor alloc] init];
        MTLTextureDescriptor* texDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:64 height:64 mipmapped:NO];
        texDesc.usage = MTLTextureUsageRenderTarget;
        id<MTLTexture> renderTarget = [device newTextureWithDescriptor:texDesc];
        rpDesc.colorAttachments[0].texture = renderTarget;
        rpDesc.colorAttachments[0].loadAction = MTLLoadActionClear;
        rpDesc.colorAttachments[0].storeAction = MTLStoreActionStore;

        id<MTLCommandBuffer> cmdBuffer2 = [queue commandBuffer];
        id<MTLRenderCommandEncoder> renderEncoder = [cmdBuffer2 renderCommandEncoderWithDescriptor:rpDesc];
        Class renderEncoderClass = [renderEncoder class];
        [renderEncoder endEncoding];

        printf("\n=== CLASS HIERARCHY ===\n");
        printf("Compute Encoder: %s\n", class_getName(computeEncoderClass));
        printf("Blit Encoder: %s\n", class_getName(blitEncoderClass));
        printf("Render Encoder: %s\n", class_getName(renderEncoderClass));

        // Analyze each encoder type
        CoverageStats computeStats, blitStats, renderStats;
        init_stats(&computeStats);
        init_stats(&blitStats);
        init_stats(&renderStats);

        analyze_class_methods(computeEncoderClass, "Compute Encoder", g_compute_swizzled, &computeStats);
        analyze_class_methods(blitEncoderClass, "Blit Encoder", g_blit_swizzled, &blitStats);
        analyze_class_methods(renderEncoderClass, "Render Encoder", g_render_swizzled, &renderStats);

        // Summary
        printf("\n\n=== OVERALL COVERAGE SUMMARY ===\n");
        int total_methods = computeStats.total + blitStats.total + renderStats.total;
        int total_swizzled = computeStats.swizzled + blitStats.swizzled + renderStats.swizzled;
        int total_unprotected = computeStats.not_swizzled + blitStats.not_swizzled + renderStats.not_swizzled;

        printf("Total encoder methods examined: %d\n", total_methods);
        printf("Total methods swizzled: %d\n", total_swizzled);
        printf("Total methods NOT swizzled: %d\n", total_unprotected);
        printf("Overall coverage: %.1f%%\n", total_methods > 0 ? 100.0 * total_swizzled / total_methods : 0.0);

        if (total_unprotected > 0) {
            printf("\n*** WARNING: %d methods are NOT protected by AGX fix ***\n", total_unprotected);
            printf("Review these methods to determine if they require mutex protection.\n");
        } else {
            printf("\n*** SUCCESS: All examined encoder methods are protected ***\n");
        }

        free_stats(&computeStats);
        free_stats(&blitStats);
        free_stats(&renderStats);

        return total_unprotected > 0 ? 1 : 0;
    }
}
