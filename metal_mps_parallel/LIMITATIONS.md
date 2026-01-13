# AGX Fix Limitations - Honest Assessment

**Date**: 2025-12-25
**Status**: Rigorous analysis complete. Swizzle provides best-effort protection, but one critical limitation is **unfalsifiable** with userspace swizzling (IMP caching bypass).

---

## Executive Summary

The AGX fix (`libagx_fix_v2_9.dylib`) has a **0% observed crash rate** in our test suite, but this provides **NO guarantee of safety**. One remaining limitation is **unfalsifiable**: we cannot prove that all relevant method calls go through our swizzled implementations.

| Limitation | Theoretical | Practical | Status |
|------------|-------------|-----------|--------|
| IMP Caching Bypass | UNFALSIFIABLE | **LOW** | External calls protected; internal paths unknown |
| ARM64 Memory Model | ~~HIGH~~ | ~~HIGH~~ | **CLOSED** - litmus tests pass (N=3690) |
| Missing Encoder Factories | ~~MEDIUM~~ | ~~MEDIUM~~ | **CLOSED** - already in v2.9 (N=3690) |

### What We Can (and Cannot) Prove

We can empirically verify that **external Objective-C message sends** (e.g., PyTorch calling `[encoder ...]`) are intercepted by swizzles under test conditions.

We **cannot** prove that **all** relevant calls go through swizzles because any code path that caches an `IMP` (function pointer) before our swizzle runs can bypass swizzling entirely. There is no userspace API to enumerate or invalidate such cached `IMP`s at call sites. See `VERIFICATION_GAPS_ROADMAP.md` (Gap 3) and `research/IMP_CACHING_BYPASS_PROOF.md`.

---

## Limitation 1: IMP Caching Bypass - UNFALSIFIABLE

### The Theoretical Problem

The AGX fix uses Objective-C method swizzling to intercept encoder method calls. If code stores an IMP before swizzle, that stored IMP would bypass swizzle.

### Why This Matters (Even If Your Code Uses objc_msgSend)

Most user code (including PyTorch) calls encoder methods via Objective-C message send (`objc_msgSend`), which consults the class method table and respects swizzles.

```
┌─────────────────────────────────────────────────────────────┐
│                      USER CODE (PyTorch MPS)                │
│   [encoder setComputePipelineState:pipeline]                │
│   [encoder dispatchThreads:size ...]                        │
│               │                                             │
│               ▼                                             │
│         objc_msgSend  ←────── Goes through swizzle ✓       │
└─────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│              AGXG16XFamilyComputeContext (RECEIVER)         │
│   The encoder RECEIVES calls, it doesn't MAKE them          │
└─────────────────────────────────────────────────────────────┘
```

However, this does **not** rule out internal framework/driver code calling into encoder methods via cached `IMP`s (or other direct call paths). If Metal.framework or the AGX driver caches an `IMP` before our swizzle runs and then calls via that cached pointer, those calls bypass swizzling entirely. We have no way to detect or prevent this from userspace.

### Rigorous Test Suite Results

Three tests were conducted to understand swizzle behavior precisely:

#### Test 1: Normal objc_msgSend calls (`test_actual_metal_swizzle.mm`)
```
Device: Apple M4 Max
Encoder class: AGXG16XFamilyComputeContext
Made 5 calls, swizzle intercepted: 5
ALL calls went through swizzle.
```
**Result**: External API calls ALWAYS go through swizzle.

#### Test 2: Encoder created BEFORE swizzle (`test_encoder_created_before_swizzle.mm`)
```
=== Step 1: Create encoder BEFORE swizzle ===
Encoder class: AGXG16XFamilyComputeContext

=== Step 2: Perform swizzle ===

=== Step 3: Call method on PRE-SWIZZLE encoder ===
Swizzle intercept count: 1
SWIZZLE WORKS even on encoder created before swizzle!
```
**Result**: Metal does NOT cache IMPs at encoder creation time. Normal calls always dispatch dynamically.

#### Test 3: Manually stored IMPs (`imp_stored_bypass_proof.mm`)
```
=== Step 1: Simulate Metal.framework storing IMP ===
Metal init: Cached IMP = 0x102b84940

=== Step 2: Our AGX fix swizzles the method ===
AGX fix: New IMP in method table = 0x102b84bf4

=== Step 3: Normal [obj method] call ===
Result: original=0 swizzled=1 ✓

=== Step 4: Metal's internal call using CACHED IMP ===
Result: original=1 swizzled=0 ✗ BYPASS CONFIRMED!
```
**Result**: If code explicitly stores an IMP, that stored pointer bypasses swizzle.

### Technical Details

**What IS protected by swizzle:**
- All external `[encoder method:args]` calls use `objc_msgSend`
- `objc_msgSend` looks up IMPs dynamically in the method table
- `method_setImplementation()` calls `flushCaches()` which invalidates the dispatch cache
- Therefore: All external API calls to encoder methods go through our swizzle

**What is NOT protected by swizzle:**
- IMPs explicitly stored in variables before swizzle
- Internal dispatch within Metal.framework/AGX driver (if they use cached IMPs)
- Function pointers derived from IMPs

Example of the bypass pattern:
```objc
// Performance-critical code sometimes does this:
IMP cachedIMP = [obj methodForSelector:@selector(setBuffer:...)];
// ...later...
cachedIMP(obj, sel, args...);  // Bypasses ANY swizzle applied after caching
```

### Precise Assessment

**VERIFIED (Empirically proven):**
- External calls to encoder methods: 100% go through swizzle ✓
- Encoders created before swizzle: calls still go through swizzle ✓
- Method table is correctly updated ✓

**UNVERIFIED / UNFALSIFIABLE (Cannot test from userspace):**
- Internal Metal.framework dispatch paths
- AGX driver internal function calls
- Whether any internal optimization uses stored IMPs

### Formal Proof of Limited Verifiability

Let:
- `S_external` = External API calls to encoder methods
- `S_internal` = Internal Metal/AGX calls to encoder methods

**Proven**: `S_external` → swizzle (100% verified by tests)
**Unknown**: Whether `S_internal` → swizzle or bypasses

**We cannot inspect** Metal.framework's internal implementation. Therefore we cannot verify `S_internal` dispatch behavior. **QED** ∎

### Final Assessment - Reconciled View

There are two valid perspectives on this limitation:

| Perspective | Assessment | Rationale |
|-------------|------------|-----------|
| **Theoretical** | CRITICAL/Unfalsifiable | Cannot prove Metal/AGX doesn't cache IMPs |
| **Practical** | LOW | Crash trigger (external API calls) is protected |

#### Why Both Are True

**Theoretical level**: The IMP caching bypass IS unfalsifiable. We genuinely cannot prove that Metal.framework or AGX driver doesn't cache IMPs internally. This is a real limitation of userspace swizzling.

**Practical level**: The crash we're fixing is triggered by **external API calls**:
- PyTorch MPS calls `[encoder setComputePipelineState:]`
- PyTorch MPS calls `[encoder dispatchThreads:...]`
- These go through `objc_msgSend` → 100% intercepted by swizzle

The encoder **receives** method calls from user code - it doesn't call these methods on itself. For IMP caching to cause our specific crash, some other framework would need to:
1. Cache IMPs to encoder methods (unlikely - frameworks don't cache IMPs to objects they don't own)
2. Call the encoder using those cached IMPs (no evidence of this pattern)
3. Race with destroy (our fix serializes destroy too)

| Question | Answer | Evidence |
|----------|--------|----------|
| Are external API calls protected? | **YES** | 100% intercept in tests |
| Is this sufficient for our crash? | **YES** | Crash triggered by external calls from PyTorch |
| Is there ANY residual risk? | **YES** | Theoretical IMP caching in Metal/AGX |
| Theoretical severity? | **UNFALSIFIABLE** | Cannot be ruled out |
| Practical severity? | **LOW** | No evidence, 0% crash rate in 2966+ rounds |

**Risk assessment**:
- **For PyTorch MPS workloads**: LOW risk. External API calls are provably protected.
- **For general Metal usage**: UNKNOWN. Cannot verify internal dispatch paths.

**0% observed crash rate across 2966+ verification rounds** provides strong empirical support for practical safety.

### What Would Fix This

Only these approaches can address IMP caching:

1. **Binary patch AGX driver** - Modify the driver itself (requires SIP disabled)
2. **Hook objc_msgSend** - Intercept ALL Objective-C calls (massive performance impact)
3. **Apple fixes the bug** - Report to Apple and wait for macOS update

### Reproduction

```bash
cd /Users/ayates/metal_mps_parallel/research
clang++ -framework Foundation -fobjc-arc -O0 \
    imp_stored_bypass_proof.mm -o imp_stored_bypass_proof
./imp_stored_bypass_proof
```

See `research/IMP_CACHING_BYPASS_PROOF.md` for complete analysis.

---

## ~~Limitation 2: ARM64 Memory Model Gap~~ - **CLOSED** (N=3690)

### Status: RESOLVED

This limitation was closed by verification in N=3690:

1. **ARM64 litmus tests PASS** on Apple M4 Max:
   - A.007: std::mutex acquire/release barriers - PASS (10,000 iterations)
   - A.008: release/acquire message passing - PASS (200,000 iterations)
   - A.003: Sequential consistency (Dekker's algorithm) - PASS (100,000 iterations)

2. **Code audit confirms** all shared state access is within mutex protection

3. **C++ standard guarantees** `unlock()` synchronizes-with the next `lock()` on the same mutex, establishing happens-before edges that make stores visible

### Remaining Caveat

Tests verified on M4 Max only. Other chips may behave differently (unlikely but not verified).

---

## ~~Limitation 3: Missing Encoder Factories~~ - **CLOSED** (N=3690)

### Status: RESOLVED

Code audit in N=3690 found this was **already implemented in v2.9**:

| Factory Method | Swizzled | Sub-Encoders |
|---------------|----------|--------------|
| `computeCommandEncoder` | ✓ | N/A |
| `blitCommandEncoder` | ✓ | N/A |
| `renderCommandEncoderWithDescriptor:` | ✓ | N/A |
| `resourceStateCommandEncoder` | ✓ | N/A |
| `accelerationStructureCommandEncoder` | ✓ | N/A |
| `parallelRenderCommandEncoderWithDescriptor:` | **✓** | **✓** |

### Implementation Details (agx_fix_v2_9.mm)

- Factory swizzle: lines 764-780, 1905-1906, 1954-1955
- Sub-encoder tracking: line 1574 (`swizzled_parallel_render_sub_encoder`)
- Lifecycle methods: lines 1550-1567

The documentation was incorrect - the code already covered this case.

---

## What "0% Crash Rate" Actually Means

When we report "0% crash rate", this means:

✓ Zero crashes observed in our test suite
✓ Zero crashes in 2+ million inference operations
✓ Zero crashes in 5-minute soak tests

This does **NOT** mean:

✗ Crashes are impossible
✗ All code paths are protected
✗ The fix is mathematically proven correct
✗ Your workload will be crash-free

### Test Coverage Limitations

| What We Test | What We Don't Test |
|--------------|-------------------|
| PyTorch MPS workloads | Other Metal frameworks |
| 8-thread stress tests | 64+ thread scenarios |
| TransformerEncoder models | All possible models |
| M4 Max hardware | All Apple Silicon variants |
| macOS 15.7.3 | Other macOS versions |

---

## Recommended Approach

Given these limitations, we recommend:

### For Production Use

1. **Use batching instead of threading** - 60x throughput gain, no race conditions
2. **Semaphore(2) throttling** - Limits concurrent MPS ops to 2, observed stable
3. **Monitor crash logs** - `crash_logs/` directory, run `check_crashes.py`

### For Development/Testing

1. **Use the AGX fix** - Still provides significant protection
2. **Run stress tests** - Verify your specific workload
3. **Accept residual risk** - Some workloads may still crash

### What NOT To Do

1. **Don't claim the fix is "proven safe"** - It is not
2. **Don't assume 0% test crashes = 0% production crashes** - Test coverage is limited
3. **Don't disable SIP without understanding risks** - Binary patching has system-wide implications

---

## References

- `VERIFICATION_GAPS_ROADMAP.md` - Detailed gap analysis with acceptance criteria
- `WORKER_DIRECTIVE.md` - Current worker priorities
- `mps-verify/specs/ASSUMPTIONS.md` - TLA+ model assumptions
- `agx_fix/README.md` - AGX fix usage documentation

---

## Revision History

| Date | Author | Changes |
|------|--------|---------|
| 2025-12-26 | Manager | RECONCILED: Both theoretical (UNFALSIFIABLE) and practical (LOW) assessments are valid |
| 2025-12-25 | Claude (N=3811) | Reverted to CRITICAL/UNFALSIFIABLE (conservative) |
| 2025-12-25 | Claude (N=3752) | Severity LOW after rigorous analysis of crash trigger |
| 2025-12-25 | Claude (N=3751) | Added empirical test results, revised from CRITICAL to LOW-MEDIUM |
| 2025-12-25 | Claude (N=3690) | Updated Limitations 2 and 3 to CLOSED status after code audit and verification |
| 2025-12-25 | Claude (N=3683) | Initial document created after rigorous self-analysis |
