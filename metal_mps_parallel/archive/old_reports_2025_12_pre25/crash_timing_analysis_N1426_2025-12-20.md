# Crash Timing Analysis (N=1426)

**Date**: 2025-12-20
**Worker**: N=1426
**Follow-up to**: N=1425 (AGX driver crash stack trace analysis)
**Finding**: Crashes occur DURING concurrent encoding, not during shutdown

## Summary

Followed up on N=1424's mutex investigation. Added detailed tracing to identify the exact crash location. Discovered that the ~60% crash rate (without mutex) occurs during **active concurrent encoding operations**, not during Python interpreter shutdown.

## Test Results

### Crash Timing Analysis

With `MPS_DISABLE_ENCODING_MUTEX=1`, added debug output at key points:
```
[1] Torch imported       <- Always reaches
[2] Warmup complete      <- Always reaches
[3] Creating threads     <- Always reaches
[4] Waiting at barrier 1 <- Always reaches
[5] Past barrier 1       <- Always reaches
*** CRASH OCCURS HERE ***
[6] Past barrier 2       <- Only on successful runs
```

The crash happens AFTER workers pass barrier 1 (tensor creation complete) and BEFORE they pass barrier 2 (work complete). This is during the concurrent execution of:
```python
for t in tensors:
    y = t * 2.0
    torch.mps.synchronize()
```

### Crash Rate Comparison

| Configuration | Crash Rate | Crash Timing |
|--------------|------------|--------------|
| WITH mutex | ~2% | Unknown (rare) |
| WITHOUT mutex | ~60% | During concurrent encoding |

## Analysis

### N=1424's Hypothesis (Partially Correct)
N=1424 attributed the crash to "static destruction order during Python interpreter shutdown". This is **partially correct** for the ~2% crash rate that remains even WITH the mutex.

### N=1425's Finding (AGX Stack Traces)
N=1425 collected actual crash reports showing the crashes occur in Apple's AGX driver code (`setComputePipelineState`, `prepareForEnqueue`).

### N=1426's Finding (Crash Timing)
The dominant crash mechanism (~60%) is the **Apple AGX driver race during concurrent Metal encoding**. This occurs while threads are actively doing MPS operations (between barriers 1 and 2), NOT during interpreter shutdown.

### Why Shutdown Was Suspected
The crashes appeared to have no output because:
1. The crash happens quickly after barrier 1
2. Python's stdout buffering means output may not flush before crash
3. The crash happens before test completion, so it looks like "nothing printed"

## Implications

1. **The global encoding mutex IS necessary** - It prevents the ~60% crash rate from concurrent encoding races

2. **The original AGX driver bug hypothesis was correct** - Apple's Metal/AGX driver does have internal race conditions when multiple threads create/use command encoders concurrently

3. **The ~2% residual crash rate is separate** - This may be from:
   - Rare timing windows the mutex doesn't fully cover
   - Static destruction order issues (N=1424's hypothesis)
   - Other unrelated races

4. **Performance improvements require Apple's fix** - We cannot safely remove the mutex without Apple fixing their driver

## TLS Sync Fix Attempt (Abandoned)

Implemented a `g_tls_sync_in_progress` counter to guard TLS destructors. This did NOT reduce crash rates because:
- The main crashes are during encoding, not TLS destruction
- The fix only protects the TLS destructor path
- That path accounts for at most ~2% of crashes (with mutex enabled)

**Decision**: Reverted the TLS sync fix as it adds complexity without measurable benefit.

## Recommendations

1. **Keep the global encoding mutex** - It prevents ~60% crash rate
2. **Accept the ~2% residual rate** - Low enough for practical use with retry logic
3. **Continue monitoring for Apple driver updates** - May eventually allow mutex removal
4. **Update documentation** - Clarify that the mutex prevents encoding races, not just shutdown races

## Files Changed

None (TLS sync fix was reverted)

## Next Steps for Future Workers

The global encoding mutex investigation is complete. Key facts established:
- Mutex prevents ~60% crash rate from concurrent encoding races
- Crash location is during active MPS operations, not Python shutdown
- No code fix can safely remove the mutex without Apple's driver fix

Project is in maintenance mode. No further mutex investigation needed.
