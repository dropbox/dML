// Check if initWithQueue: selector exists on encoder class
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
        id<MTLCommandBuffer> buffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [buffer computeCommandEncoder];

        if (!encoder) {
            printf("ERROR: Failed to create encoder\n");
            return 1;
        }

        Class encoderClass = [encoder class];
        printf("Encoder class: %s\n", class_getName(encoderClass));

        // Check for initWithQueue:
        Method m1 = class_getInstanceMethod(encoderClass, @selector(initWithQueue:));
        printf("initWithQueue: exists? %s\n", m1 ? "YES" : "NO");

        // Check for initWithCommandBuffer:
        Method m2 = class_getInstanceMethod(encoderClass, @selector(initWithCommandBuffer:));
        printf("initWithCommandBuffer: exists? %s\n", m2 ? "YES" : "NO");

        // Check for init
        Method m3 = class_getInstanceMethod(encoderClass, @selector(init));
        printf("init exists? %s\n", m3 ? "YES" : "NO");

        // List all methods starting with "init"
        printf("\nAll init* methods:\n");
        unsigned int count;
        Method* methods = class_copyMethodList(encoderClass, &count);
        for (unsigned int i = 0; i < count; i++) {
            SEL sel = method_getName(methods[i]);
            const char* name = sel_getName(sel);
            if (strncmp(name, "init", 4) == 0) {
                printf("  %s\n", name);
            }
        }
        free(methods);

        // Also check parent classes
        Class parent = class_getSuperclass(encoderClass);
        while (parent) {
            printf("\nParent class: %s\n", class_getName(parent));
            methods = class_copyMethodList(parent, &count);
            for (unsigned int i = 0; i < count; i++) {
                SEL sel = method_getName(methods[i]);
                const char* name = sel_getName(sel);
                if (strncmp(name, "init", 4) == 0) {
                    printf("  %s\n", name);
                }
            }
            free(methods);
            parent = class_getSuperclass(parent);
            if (strcmp(class_getName(parent), "NSObject") == 0) break;
        }

        [encoder endEncoding];
        return 0;
    }
}
