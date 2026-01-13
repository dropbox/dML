# Apple Feedback: AGX Metal Driver Race Condition

**Category**: Bug Report - Metal/GPU Driver
**Affected Component**: AGXMetalG16X (AGX GPU driver)
**macOS Version**: 15.7.3 (Build 24G419)
**Hardware**: Mac16,5 (M4 Max)
**AGX Driver Version**: 329.2

---

## Summary

The AGX Metal GPU driver (`AGXMetalG16X.bundle`) contains a race condition that causes SIGSEGV crashes when multiple threads concurrently create and use MTLComputeCommandEncoders. The crash occurs inside the driver's internal `ContextCommon` data structure due to missing synchronization.

## Severity

**HIGH** - Crashes application, data loss possible, no workaround except serializing all GPU operations (severe performance impact).

---

## Description

When multiple threads encode Metal compute operations concurrently, the AGX driver can crash with a NULL pointer dereference (SIGSEGV). The crash occurs at three known locations inside the driver:

### Crash Sites

| # | Function | Offset | Description |
|---|----------|--------|-------------|
| 1 | `-[AGXG16XFamilyComputeContext setComputePipelineState:]` | 0x5c8 | NULL deref reading context field |
| 2 | `AGX::ComputeContext::prepareForEnqueue` | 0x98 | NULL pointer read |
| 3 | `AGX::SpillInfoGen3::allocateUSCSpillBuffer` | 0x184 | NULL pointer write |

### Exception Details

```
Exception Type:        EXC_BAD_ACCESS (SIGSEGV)
Exception Codes:       KERN_INVALID_ADDRESS at 0x00000000000005c8
Termination Reason:    Namespace SIGNAL, Code 11 Segmentation fault: 11
```

### Register State at Crash

```
x20: 0x0000000000000000  <-- NULL pointer (should be context)
far: 0x00000000000005c8  <-- Fault address (NULL + offset 0x5c8)
esr: 0x92000006 (Data Abort) byte read Translation fault
```

---

## Root Cause Analysis

Based on reverse engineering of the crash reports, the AGX driver maintains internal per-encoder state in a `ContextCommon` class. When multiple threads concurrently:

1. Create command encoders
2. Set compute pipeline states
3. Dispatch compute operations
4. Commit and/or wait for completion

The internal context state can become corrupted due to missing synchronization. Specifically:

- Thread A calls `setComputePipelineState:` which accesses `ContextCommon` at offset 0x5c8
- Thread B simultaneously invalidates or deallocates the same context
- Thread A dereferences a now-NULL context pointer → crash

### Formal Verification

We modeled this race condition in TLA+ (Temporal Logic of Actions) and formally proved:
1. The race condition CAN occur in the hypothesized design
2. A global mutex prevents all race manifestations
3. Per-stream or per-encoder mutexes are insufficient

See: `tla_proofs/AGXContextRace.tla` (buggy model) and `AGXContextFixed.tla` (fixed model)

---

## Steps to Reproduce

### Quick Test (PyTorch MPS)

```bash
# Install PyTorch with MPS support
pip install torch

# Run multi-threaded MPS test (crashes ~55% of the time)
python3 -c "
import torch
import threading

device = torch.device('mps')
model = torch.nn.Linear(512, 512).to(device).eval()

def worker():
    x = torch.randn(1, 512, device=device)
    for _ in range(50):
        with torch.no_grad():
            model(x)
        torch.mps.synchronize()

threads = [threading.Thread(target=worker) for _ in range(8)]
for t in threads: t.start()
for t in threads: t.join()
print('Success')
"
```

### Standalone Metal Test

See enclosed files:
- `reproduction/metal_race_repro.mm` - Pure Metal compute (may not crash but shows driver warnings)
- `reproduction/metal_race_repro_mps.mm` - MetalPerformanceShaders version

```bash
clang++ -std=c++17 -framework Metal -framework Foundation \
  -o metal_race_repro metal_race_repro.mm
./metal_race_repro
```

### Crash Rate

| Threads | Crash Rate | Notes |
|---------|------------|-------|
| 1 | 0% | No concurrency |
| 2 | ~10% | Intermittent |
| 4 | ~30% | More likely |
| 8 | ~55% | Reliable reproduction |
| 16 | ~55% | Plateaus |

The crash rate depends on timing and system load. Multiple runs may be needed.

---

## Impact

### Affected Applications

1. **PyTorch MPS backend** - Multi-threaded inference crashes
2. **MLX** - Apple's ML framework (crashes at 2+ threads)
3. **Any application** using concurrent Metal compute operations

### Workaround

We implemented a global mutex around all command encoder operations, which prevents crashes but severely limits parallelism:

```objc
static std::mutex g_encoding_mutex;

// Wrapped every encoder operation:
{
    std::lock_guard<std::mutex> lock(g_encoding_mutex);
    [encoder setComputePipelineState:pipeline];
}
```

This workaround:
- ✅ Prevents all crashes (0% crash rate)
- ❌ Serializes all GPU encoding (limits scalability)
- ❌ Prevents true parallel inference

---

## Attached Evidence

### Crash Reports
- `crash_reports/CRASH_ANALYSIS_2025-12-20_173618.md` - Full crash analysis with stack traces
- `crash_reports/CRASH_ANALYSIS_2025-12-20_174241.md` - Second crash instance

### TLA+ Formal Proofs
- `tla_proofs/AGXContextRace.tla` - Model of buggy driver design (demonstrates race)
- `tla_proofs/AGXContextFixed.tla` - Model with mutex fix (proves correctness)

### Lean 4 Machine-Checked Proofs
- `lean4_proofs/Race.lean` - `race_condition_exists` theorem
- `lean4_proofs/Fixed.lean` - `mutex_prevents_race` theorem
- `lean4_proofs/PerStreamMutex.lean` - Proves per-stream mutexes are insufficient
- `lean4_proofs/PerOpMutex.lean` - Proves per-operation mutexes are insufficient
- `lean4_proofs/RWLock.lean` - Proves reader-writer locks are insufficient

### Reproduction Code
- `reproduction/metal_race_repro.mm` - Minimal standalone C++/Metal test
- `reproduction/metal_race_repro_mps.mm` - MPS (MetalPerformanceShaders) version

---

## Suggested Fix

The driver should add proper synchronization around the `ContextCommon` data structure:

1. **Per-context mutex**: Protect context fields during read/modify operations
2. **Reference counting**: Prevent context deallocation while in use
3. **Memory barriers**: Ensure visibility of context state across threads

The TLA+ specification `AGXContextFixed.tla` demonstrates a correct design with mutex protection.

---

## System Configuration

```
ProductName:      macOS
ProductVersion:   15.7.3
BuildVersion:     24G419
Hardware:         Mac16,5 (MacBook Pro M4 Max)
Chipset:          Apple M4 Max (40 GPU cores)
Metal Support:    Metal 3
AGX Driver:       AGXMetalG16X 329.2
Metal Framework:  368.52
IOGPU Framework:  104.6.3
```

---

## Contact

Andrew Yates
GitHub: https://github.com/dropbox/dML/metal_mps_parallel

---

## Related

- PyTorch GitHub Issue: (To be filed)
- MLX GitHub: Similar crashes reported with multi-threaded access
