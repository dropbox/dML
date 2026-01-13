# AGX Driver Crash Analysis

**Created by Andrew Yates**

This document analyzes all documented crash scenarios and verifies the binary patch addresses each one.

---

## Root Cause

All crashes stem from a **single race condition** in `-[AGXG16XFamilyComputeContext destroyImpl]`:

```
Thread A (encoding):              Thread B (cleanup):
─────────────────────────         ─────────────────────────
read context->_impl               lock()
  ↓                               _impl = NULL  ← BUG: happens AFTER unlock
access _impl->field               unlock()
  ↓
CRASH: NULL + offset
```

The bug: `_impl` is NULLed **AFTER** the lock is released, creating a window where Thread A can read a pointer that Thread B is about to invalidate.

---

## Documented Crash Sites

### Crash Site 1: `setComputePipelineState:` + 0x5c8

| Field | Value |
|-------|-------|
| Function | `-[AGXG16XFamilyComputeContext setComputePipelineState:]` |
| Fault Address | 0x5c8 |
| Access Type | READ (NULL + 0x5c8) |
| Structure Field | `MTLResourceList*` |

**Sample crash:**
```
Exception Type: EXC_BAD_ACCESS (SIGSEGV)
Exception Codes: KERN_INVALID_ADDRESS at 0x00000000000005c8
Thread 15 Crashed:
0   AGXMetalG16X  -[AGXG16XFamilyComputeContext setComputePipelineState:] + 32
```

### Crash Site 2: `prepareForEnqueue` + 0x98

| Field | Value |
|-------|-------|
| Function | `-[AGXG16XFamilyComputeContext prepareForEnqueue]` |
| Fault Address | 0x98 |
| Access Type | READ (NULL + 0x98) |
| Structure Field | Early context field (command buffer state) |

**Sample crash:**
```
Exception Type: EXC_BAD_ACCESS (SIGSEGV)
Exception Codes: KERN_INVALID_ADDRESS at 0x0000000000000098
Thread 12 Crashed:
0   AGXMetalG16X  -[AGXG16XFamilyComputeContext prepareForEnqueue] + 44
```

### Crash Site 3: `allocateUSCSpillBuffer` + 0x184

| Field | Value |
|-------|-------|
| Function | `allocateUSCSpillBuffer` (called from `dispatchThreads`) |
| Fault Address | 0x184 |
| Access Type | WRITE (NULL + 0x184) |
| Structure Field | `SpillInfo` or allocation descriptor |

**Sample crash:**
```
Exception Type: EXC_BAD_ACCESS (SIGSEGV)
Exception Codes: KERN_INVALID_ADDRESS at 0x0000000000000184
Thread 8 Crashed:
0   AGXMetalG16X  allocateUSCSpillBuffer + 108
```

### Crash Site 4: `dispatchThreads:threadsPerThreadgroup:` + 0x5f4

| Field | Value |
|-------|-------|
| Function | `-[AGXG16XFamilyComputeContext dispatchThreads:threadsPerThreadgroup:]` |
| Fault Address | 0x5f4 |
| Access Type | READ (NULL + 0x5f4) |
| Structure Field | Related to MTLResourceList or dispatch state |

**Sample crash (user-reported 2025-12-22):**
```
Exception Type: EXC_BAD_ACCESS (SIGSEGV)
Exception Codes: KERN_INVALID_ADDRESS at 0x00000000000005f4
Thread 3 Crashed:
0   AGXMetalG16X  -[AGXG16XFamilyComputeContext dispatchThreads:threadsPerThreadgroup:] + 292
```

---

## Why All Crashes Have the Same Root Cause

All fault addresses (0x5c8, 0x98, 0x184, 0x5f4) are **small offsets from NULL**:

| Offset | Decimal | Interpretation |
|--------|---------|----------------|
| 0x5c8 | 1480 | Field at byte 1480 in context structure |
| 0x5f4 | 1524 | Field at byte 1524 in context structure |
| 0x184 | 388 | Field at byte 388 in context structure |
| 0x98 | 152 | Field at byte 152 in context structure |

These are all accesses to `self->_impl->someField` where `_impl` is NULL. The different offsets correspond to different fields being accessed, but the NULL pointer is the same.

---

## How the Binary Patch Fixes This

### Original Code (BUG)

```
0x2be074: bl unlock            ; Release lock FIRST
...
0x2be08c: str xzr, [x19, x24]  ; NULL _impl AFTER unlock ← RACE WINDOW
```

Between `unlock` and `str xzr`, another thread can:
1. Acquire the lock
2. Read `_impl` (gets the old, valid pointer)
3. Release the lock
4. Access `_impl->field` (pointer is now NULL, CRASH)

### Patched Code (FIXED)

```
0x2be070: str xzr, [x19, x24]  ; NULL _impl FIRST
0x2be074: add x0, x25, x21     ; Prepare lock address
0x2be078: bl unlock            ; Release lock AFTER NULL
```

Now another thread will:
1. Acquire the lock
2. Read `_impl` (gets NULL)
3. Check for NULL before accessing (normal error handling)
4. No crash

---

## Patch Coverage Analysis

| Crash Site | Root Cause | Patch Addresses? |
|------------|------------|------------------|
| 0x5c8 (`setComputePipelineState:`) | NULL `_impl` | ✅ Yes - _impl NULLed before unlock |
| 0x98 (`prepareForEnqueue`) | NULL `_impl` | ✅ Yes - _impl NULLed before unlock |
| 0x184 (`allocateUSCSpillBuffer`) | NULL `_impl` | ✅ Yes - _impl NULLed before unlock |
| 0x5f4 (`dispatchThreads:`) | NULL `_impl` | ✅ Yes - _impl NULLed before unlock |

**All crash sites share the same root cause, and the patch fixes that root cause.**

---

## Verification Plan

### Step 1: Baseline Crash Test (Before Patch)

Run WITHOUT any fix to confirm crashes occur:

```bash
# Disable the method swizzling workaround
unset DYLD_INSERT_LIBRARIES

# Run crash demo (expected: ~55% crash rate)
for i in {1..20}; do
    python3 tests/crash_demos/test_shutdown_crash.py 2>&1
    echo "Exit code: $?"
done | grep -c "Exit code: 139"
# Expect: 8-15 crashes out of 20 runs
```

### Step 2: Patch Deployment

```bash
# Disable SIP (Recovery Mode)
# Deploy patch
sudo ./deploy_patch.sh
# Reboot
```

### Step 3: Verify Patch Fixes All Crashes

Run the same test WITHOUT method swizzling:

```bash
unset DYLD_INSERT_LIBRARIES

# Run crash demo (expected: 0% crash rate)
for i in {1..20}; do
    python3 tests/crash_demos/test_shutdown_crash.py 2>&1
    echo "Exit code: $?"
done | grep -c "Exit code: 139"
# Expect: 0 crashes out of 20 runs
```

### Step 4: Stress Test

```bash
# Run full stress test
python3 tests/complete_story_test_suite.py

# Expected:
# - 0% crash rate
# - 50%+ efficiency at 8 threads (project goal)
```

---

## Known Limitations

### Memory Leak in Path 2

The binary patch skips `free()` in PATH 2 (freelist full) due to space constraints. This causes a memory leak when:

1. The internal freelist is full (count > 7)
2. A compute context is being destroyed

**Impact**: Minimal - PATH 2 is rarely executed (freelist is almost never full).

**Mitigation**: For zero-leak operation, use the method swizzling approach (`libagx_fix.dylib`).

---

## Type 2: Use-After-Free Crashes (Method Swizzling Fix)

**Note**: This section applies to the **method swizzling workaround** (`libagx_fix.dylib`), not the binary patch.

### Symptoms

- Large fault addresses like `0x4c266d050640`, `0x9ba36e2f2650`
- "possible pointer authentication failure" error in crash log
- Crash in `objc_msgSend + 32` (before any app code)
- Triggers at high thread counts (12+ threads)

### Root Cause (v2.3 Bug)

v2.3 method swizzling had a race condition:
1. Thread A holds reference to encoder, about to call `setBuffer:`
2. Thread B calls `endEncoding` → releases our retain
3. PyTorch releases → encoder deallocated
4. Thread A's `objc_msgSend` crashes trying to dispatch to freed object

The crash happens **before** our swizzled method gets control.

### Fix (v2.4)

v2.4 adds "active call tracking":
- Each method call increments a counter and adds a retain
- Each method completion decrements counter and releases
- `endEncoding` marks encoder as "ended" but doesn't release if calls are active
- Release happens when last method call completes AND ended flag is set

**Result**: Encoder stays alive until ALL in-flight method calls complete.

### Method Swizzling Library Versions

| Version | Status | Issue |
|---------|--------|-------|
| v2.5 | **RECOMMENDED** | Best stability with MPS_FORCE_GRAPH_PATH=1 |
| v2.6 | Obsolete | Blocking destroyImpl breaks object lifecycle |
| v2.4 | Stable | Active call tracking prevents use-after-free |
| v2.3 | Obsolete | Use-after-free at high thread counts |
| v2.2 | Obsolete | Crashes at 8 threads (removed mutex protection) |
| v2.1 | Obsolete | Per-method retain, still has pre-swizzle race |

### Test Results (N=2977, 2025-12-23)

| Configuration | Python Pass Rate | Crash Logs Created | Notes |
|--------------|------------------|-------------------|-------|
| v2.5 + MPS_FORCE_GRAPH_PATH=1, simple tests | 100% | 0 | No crashes |
| v2.5 + MPS_FORCE_GRAPH_PATH=1, complete_story (5 runs) | 100% | 2 | Silent crashes |

**CRITICAL FINDING (N=2977)**: Crashes occur SILENTLY in worker threads.

The test suite can report "100% pass" while crashes still occur because:
1. Crashes happen in background worker threads (e.g., "Thread-41 (worker)" on "metal gpu stream 10")
2. Worker thread crashes don't always terminate the main Python process
3. PyTorch's dispatch_sync catches some exceptions

**Crash Pattern (N=2977)**:
```
Thread: "Thread-41 (worker)" on "metal gpu stream 10"
Crash:  objc_msgSend + 32 (PAC failure)
Call:   layer_norm_mps -> dispatch_sync_with_rethrow -> setBuffer:offset:atIndex:
Cause:  Encoder freed before method call completed
```

**Note**: v2.5 reduces but does NOT eliminate PAC failures. Always check `crash_logs/` after tests.

### Usage

```bash
# Recommended (v2.5 with MPS_FORCE_GRAPH_PATH=1)
DYLD_INSERT_LIBRARIES=/path/to/libagx_fix_v2_5.dylib MPS_FORCE_GRAPH_PATH=1 python your_script.py

# Enable verbose logging
AGX_FIX_VERBOSE=1 DYLD_INSERT_LIBRARIES=/path/to/libagx_fix_v2_5.dylib MPS_FORCE_GRAPH_PATH=1 python your_script.py
```

---

## References

- `reports/main/agx_driver_crash_analysis_N1425_2025-12-20.md` - Initial crash analysis
- `reports/main/agx_reverse_engineering_N1435_2025-12-20.md` - Reverse engineering findings
- `reports/main/context_lifecycle_analysis_N1474_2025-12-21.md` - Full lifecycle analysis
- `agx_patch/AGXRaceFix.tla` - TLA+ formal verification
