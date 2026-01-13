# Appendix A: Crash Reports

This appendix contains the full crash reports that document the AGX driver race condition.

---

## A.1 Crash Site 1: setComputePipelineState

**Incident ID**: `79BC4A1F-C45B-45D9-BB21-94FE1DDC55A9`
**Date**: 2025-12-20 17:36:18.2862 -0800
**Classification**: NULL pointer dereference at offset 0x5c8

### Exception Details

```
Exception Type:        EXC_BAD_ACCESS (SIGSEGV)
Exception Codes:       KERN_INVALID_ADDRESS at 0x00000000000005c8
Exception Codes:       0x0000000000000001, 0x00000000000005c8

Termination Reason:    Namespace SIGNAL, Code 11 Segmentation fault: 11
```

### Register State

```
x0: 0x0000600003afd8c0   x1: 0x00000001fb28cd9c   x2: 0x000000012050c170
x3: 0x000000016eaf0fd8   x4: 0x0000000130bfc0a0   x5: 0x000000000000002c
x6: 0x0000000000000000   x7: 0x0000000000000001   x8: 0x0000000000000078
x9: 0x00000001fb28cd9c  x10: 0x00000001d79ed864  x11: 0x000000000000001f
x12: 0x000000000000001c  x13: 0x000000012050c790  x14: 0x010000010ce6d019
x15: 0x000000010ce6d018  x16: 0x000000010ce6d018  x17: 0x000000010c9d52b0
x18: 0x0000000000000000  x19: 0x000000012050c170  x20: 0x0000000000000000  <-- NULL
x21: 0x0000600003afd8c0  x22: 0x000000016eaf1290  x23: 0x00000001036dafc8
x24: 0x000000016eaf1010  x25: 0x0000600000f741a0  x26: 0x0000000000000019
x27: 0x00000001036dafc8  x28: 0x00000001208da040   fp: 0x000000016eaf0fb0
lr: 0x0000000116b36f9c   sp: 0x000000016eaf0fa0   pc: 0x000000010c9d52d0
cpsr: 0x60000000
far: 0x00000000000005c8   <-- Fault address (NULL + 0x5c8)
esr: 0x92000006 (Data Abort) byte read Translation fault
```

### Stack Trace

```
Frame  Binary                Symbol                                                 Offset
-----  -------------------   ----------------------------------------------------   ------
0      AGXMetalG16X          -[AGXG16XFamilyComputeContext setComputePipelineState:]   +32
1      libtorch_cpu.dylib    invocation function for block in                       +444
                             at::native::mps::MetalShaderLibrary::exec_binary_kernel(...)
2      libtorch_cpu.dylib    invocation function for block in                       +40
                             at::native::mps::dispatch_sync_with_rethrow(...)
3      libdispatch.dylib     _dispatch_client_callout                               +16
4      libdispatch.dylib     _dispatch_lane_barrier_sync_invoke_and_complete        +56
5      libtorch_cpu.dylib    at::native::mps::dispatch_sync_with_rethrow(...)       +124
6      libtorch_cpu.dylib    at::native::mps::MetalShaderLibrary::exec_binary_kernel(...) +2124
7      libtorch_cpu.dylib    at::native::mul_mps_kernel(at::TensorIteratorBase&)    +88
8      libtorch_cpu.dylib    at::(anonymous namespace)::wrapper_MPS_mul_Tensor(...) +224
```

### System Information

| Field | Value |
|-------|-------|
| Process | Python 3.14.0 |
| Hardware | Mac16,5 (M4 Max) |
| OS | macOS 15.7.3 (24G419) |
| AGX Driver | AGXMetalG16X 329.2 |
| Metal Framework | 368.52 |
| Process Uptime | 0.43 seconds |

---

## A.2 Crash Site 2: allocateUSCSpillBuffer

**Incident ID**: `9B1CCC47-4DA8-4734-BCF1-3BFA26420074`
**Date**: 2025-12-20 17:42:41.7696 -0800
**Classification**: NULL pointer dereference at offset 0x184 (WRITE fault)

### Exception Details

```
Exception Type:        EXC_BAD_ACCESS (SIGSEGV)
Exception Codes:       KERN_INVALID_ADDRESS at 0x0000000000000184
Exception Codes:       0x0000000000000001, 0x0000000000000184

Termination Reason:    Namespace SIGNAL, Code 11 Segmentation fault: 11

far: 0x0000000000000184
esr: 0x92000046 (Data Abort) byte write Translation fault
```

**Note**: This is a WRITE fault (0x92000046) vs the first crash's READ fault (0x92000006).

### Register State

```
x0: 0x000000015809c178   x1: 0x0000000000000140   x2: 0x0000000000000000
x3: 0x0000000000000003   x4: 0x0000000000000000   x5: 0x0000000000000000
x6: 0x0000000000000400   x7: 0x00000000000023b8   x8: 0x0000000000000000
x9: 0x0000000000000000  x10: 0x00000000000045f0  x11: 0x0000000000000000
x12: 0x0000000000000000  x13: 0x0000000000007a60  x14: 0x00fd00bd00400020
x15: 0x000000000000000e  x16: 0x0000000000000003  x17: 0x00000000000045f0
x18: 0x0000000000000000  x19: 0x0000000158098000  x20: 0x0000000121a04c90
x21: 0x0000000150073200  x22: 0x0000000000004178  x23: 0x0000000000000000
x24: 0x00000001ee869a80  x25: 0x000000015809c000  x26: 0x0000000000000016
x27: 0x0000000000000016  x28: 0x0000000000000000   fp: 0x0000000175efcf10
lr: 0x0000000122013f3c   sp: 0x0000000175efce50   pc: 0x0000000122413018
cpsr: 0x80000000
far: 0x0000000000000184   <-- Fault address (NULL + 0x184)
esr: 0x92000046 (Data Abort) byte write Translation fault
```

### Stack Trace

```
Frame  Binary           Symbol                                            Offset
-----  ---------------  ------------------------------------------------  ------
0      AGXMetalG16X     AGX::SpillInfoGen3<...>::allocateUSCSpillBuffer    +192
1      AGXMetalG16X     AGX::ComputeContext<...>::performEnqueueKernel     +2148
2      AGXMetalG16X     AGX::ComputeContext<...>::executeKernelWithThreadsPerGridImpl +528
3      AGXMetalG16X     -[AGXG16XFamilyComputeContext dispatchThreads:threadsPerThreadgroup:] +292
4      libtorch_cpu     invocation function for block in exec_binary_kernel +1180
5      libtorch_cpu     invocation function for block in dispatch_sync_with_rethrow +40
6      libdispatch      _dispatch_client_callout                          +16
7      libdispatch      _dispatch_lane_barrier_sync_invoke_and_complete   +56
8      libtorch_cpu     at::native::mps::dispatch_sync_with_rethrow       +124
9      libtorch_cpu     at::native::mps::MetalShaderLibrary::exec_binary_kernel +2124
10     libtorch_cpu     at::native::mul_mps_kernel                        +88
```

---

## A.3 Crash Site 3: prepareForEnqueue

**Classification**: NULL pointer dereference at offset 0x98 (READ fault)

### Symbol
```
AGX::ComputeContext<...>::prepareForEnqueue(bool)
```

### Analysis

This crash occurs during kernel enqueue preparation. The 0x98 offset suggests access to an early field in the context structure related to command buffer or encoder state.

---

## A.4 Summary of All Crash Sites

| # | Function | Offset | Type | Stream |
|---|----------|--------|------|--------|
| 1 | `setComputePipelineState:` | 0x5c8 | READ | metal gpu stream 1 |
| 2 | `allocateUSCSpillBuffer` | 0x184 | WRITE | metal gpu stream 8 |
| 3 | `prepareForEnqueue` | 0x98 | READ | various |

### Common Pattern

All crashes share:
1. NULL pointer dereference (context pointer is 0x0)
2. Crash occurs inside Apple's AGXMetalG16X driver code
3. Only triggered under multi-threaded encoding
4. Consistent ~55% crash rate without mutex protection

### Binary Versions

| Binary | UUID | Version |
|--------|------|---------|
| AGXMetalG16X | 0ce453ff-9f17-321f-bc18-383573a48221 | 329.2 |
| Metal.framework | 1b375ba3-e776-36ca-888b-6b40daf42c92 | 368.52 |
| IOGPU.framework | ea6c7de2-55da-3056-8b03-439ae1bf175f | 104.6.3 |

---

## A.5 Reproduction Commands

```bash
# Without protective mutex (will crash ~55% of attempts):
MPS_DISABLE_ENCODING_MUTEX=1 python3 tests/benchmark_comprehensive_final.py

# With protective mutex (0% crash rate):
python3 tests/benchmark_comprehensive_final.py

# Stress test (100 iterations):
for i in {1..100}; do
  MPS_DISABLE_ENCODING_MUTEX=1 timeout 5 python3 tests/benchmark_comprehensive_final.py 2>&1 || true
done
```
