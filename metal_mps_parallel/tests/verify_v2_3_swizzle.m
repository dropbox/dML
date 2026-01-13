// Comprehensive verification of v2.3 swizzle coverage
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>

void list_all_methods_containing(Class cls, const char* pattern) {
    printf("\nMethods containing '%s' in %s:\n", pattern, class_getName(cls));
    unsigned int count;
    Method* methods = class_copyMethodList(cls, &count);
    int found = 0;
    for (unsigned int i = 0; i < count; i++) {
        SEL sel = method_getName(methods[i]);
        const char* name = sel_getName(sel);
        if (strstr(name, pattern) != NULL) {
            printf("  %s\n", name);
            found++;
        }
    }
    if (found == 0) printf("  (none found)\n");
    free(methods);
}

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            printf("ERROR: No Metal device\n");
            return 1;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> cmdBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        if (!encoder || !cmdBuffer) {
            printf("ERROR: Failed to create test objects\n");
            return 1;
        }

        Class encoderClass = [encoder class];
        Class cmdBufferClass = [cmdBuffer class];

        printf("=== ENCODER CLASS HIERARCHY ===\n");
        printf("Encoder class: %s\n", class_getName(encoderClass));
        Class parent = class_getSuperclass(encoderClass);
        while (parent && strcmp(class_getName(parent), "NSObject") != 0) {
            printf("  -> %s\n", class_getName(parent));
            parent = class_getSuperclass(parent);
        }

        printf("\n=== COMMAND BUFFER CLASS HIERARCHY ===\n");
        printf("CommandBuffer class: %s\n", class_getName(cmdBufferClass));
        parent = class_getSuperclass(cmdBufferClass);
        while (parent && strcmp(class_getName(parent), "NSObject") != 0) {
            printf("  -> %s\n", class_getName(parent));
            parent = class_getSuperclass(parent);
        }

        // Check v2.3's swizzle targets on CommandBuffer
        printf("\n=== V2.3 SWIZZLE TARGETS (CommandBuffer) ===\n");

        SEL sel1 = @selector(computeCommandEncoder);
        SEL sel2 = @selector(computeCommandEncoderWithDescriptor:);
        SEL sel3 = @selector(computeCommandEncoderWithDispatchType:);

        Method m1 = class_getInstanceMethod(cmdBufferClass, sel1);
        Method m2 = class_getInstanceMethod(cmdBufferClass, sel2);
        Method m3 = class_getInstanceMethod(cmdBufferClass, sel3);

        printf("computeCommandEncoder: %s\n", m1 ? "EXISTS" : "MISSING!");
        printf("computeCommandEncoderWithDescriptor:: %s\n", m2 ? "EXISTS" : "MISSING!");
        printf("computeCommandEncoderWithDispatchType:: %s\n", m3 ? "EXISTS" : "MISSING!");

        // Find ALL encoder creation methods
        printf("\n=== ALL ENCODER CREATION METHODS ===\n");
        list_all_methods_containing(cmdBufferClass, "Encoder");

        // Check parent classes too
        parent = class_getSuperclass(cmdBufferClass);
        while (parent && strcmp(class_getName(parent), "NSObject") != 0) {
            list_all_methods_containing(parent, "Encoder");
            parent = class_getSuperclass(parent);
        }

        // Check v2's swizzle target (for comparison)
        printf("\n=== V2 SWIZZLE TARGET (Encoder init) ===\n");
        Method m_init = class_getInstanceMethod(encoderClass, @selector(initWithQueue:));
        printf("initWithQueue: on encoder class: %s\n", m_init ? "EXISTS" : "MISSING!");

        // Check all init methods on encoder
        list_all_methods_containing(encoderClass, "init");
        parent = class_getSuperclass(encoderClass);
        while (parent && strcmp(class_getName(parent), "NSObject") != 0) {
            list_all_methods_containing(parent, "init");
            parent = class_getSuperclass(parent);
        }

        // Check threading bugs - are data structures thread-safe?
        printf("\n=== THREADING CONCERNS ===\n");
        printf("v2.3 uses std::mutex for encoder tracking: NEEDS VERIFICATION\n");
        printf("v2.3 uses std::unordered_map: NOT THREAD-SAFE without mutex\n");
        printf("v2.3 uses std::unordered_set: NOT THREAD-SAFE without mutex\n");

        [encoder endEncoding];

        printf("\n=== VERIFICATION COMPLETE ===\n");
        return 0;
    }
}
