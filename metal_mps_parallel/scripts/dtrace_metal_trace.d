#!/usr/sbin/dtrace -s
/*
 * dtrace_metal_trace.d - Trace Metal command encoder lifecycle
 *
 * Worker: N=1474
 * Purpose: Phase 3.1 dynamic analysis of Metal encoding operations
 *
 * Usage:
 *   sudo dtrace -s dtrace_metal_trace.d -p <PID>
 *
 * Or launch with script:
 *   sudo dtrace -s dtrace_metal_trace.d -c 'python3 your_script.py'
 *
 * Note: SIP limits tracing to user-space only. AGXMetalG16X driver
 * internals (kernel-mode) cannot be traced without SIP disabled.
 */

#pragma D option quiet
#pragma D option dynvarsize=4m

BEGIN
{
    printf("=== Metal Command Encoder Trace ===\n");
    printf("Tracing PID %d\n\n", $target);
    start = timestamp;
}

/* Trace compute encoder creation */
objc$target:MTLCommandBuffer*:-computeCommandEncoder*:entry
{
    self->cmd_buf = arg0;
    printf("[%6d µs] %s %s (cmdBuf=%p)\n",
        (timestamp - start) / 1000,
        probemod, probefunc, arg0);
}

/* Trace compute encoder end */
objc$target:MTLComputeCommandEncoder*:-endEncoding:entry
{
    printf("[%6d µs] %s %s (encoder=%p)\n",
        (timestamp - start) / 1000,
        probemod, probefunc, arg0);
}

/* Trace command buffer commit */
objc$target:MTLCommandBuffer*:-commit:entry
{
    printf("[%6d µs] %s %s (cmdBuf=%p)\n",
        (timestamp - start) / 1000,
        probemod, probefunc, arg0);
}

/* Trace pipeline state setting - the crash site function */
objc$target:MTLComputeCommandEncoder*:-setComputePipelineState*:entry
{
    printf("[%6d µs] %s %s (encoder=%p, pipeline=%p)\n",
        (timestamp - start) / 1000,
        probemod, probefunc, arg0, arg2);
}

/* Trace resource usage - another potential crash site */
objc$target:MTLComputeCommandEncoder*:-useResource*:entry
{
    printf("[%6d µs] %s %s (encoder=%p)\n",
        (timestamp - start) / 1000,
        probemod, probefunc, arg0);
}

/* Trace buffer binding */
objc$target:MTLComputeCommandEncoder*:-setBuffer*:entry
{
    printf("[%6d µs] %s %s (encoder=%p)\n",
        (timestamp - start) / 1000,
        probemod, probefunc, arg0);
}

/* Trace dispatch calls */
objc$target:MTLComputeCommandEncoder*:-dispatchThreads*:entry
{
    printf("[%6d µs] %s %s (encoder=%p)\n",
        (timestamp - start) / 1000,
        probemod, probefunc, arg0);
}

/* Trace command queue creation */
objc$target:MTLDevice*:-newCommandQueue*:entry
{
    printf("[%6d µs] %s %s (device=%p)\n",
        (timestamp - start) / 1000,
        probemod, probefunc, arg0);
}

/* Track thread activity for race detection */
objc$target:MTLComputeCommandEncoder*::entry
{
    @encoder_calls[tid, probefunc] = count();
}

END
{
    printf("\n=== Encoder Call Summary by Thread ===\n");
    printa("Thread %d: %s = %@d\n", @encoder_calls);
}
