# AGX Driver Race Condition Fix

**Created by Andrew Yates**

Userspace workarounds for race conditions in Apple's AGX Metal driver that cause crashes during parallel MPS inference.

---

## CRITICAL: Unfalsifiable Limitation

**Read [../LIMITATIONS.md](../LIMITATIONS.md) before relying on this fix.**

This fix has **0% observed crash rate** in testing, but this provides **NO guarantee of safety**:

**IMP Caching Bypass (UNFALSIFIABLE)**: Objective-C runtime caches method implementations at call sites. If Metal.framework cached IMPs before our dylib loaded, those calls bypass our swizzle entirely. **We cannot detect or prevent this.**

Other previously identified limitations have been closed:
- ~~ARM64 Memory Model~~: **CLOSED** (N=3690) - litmus tests pass, code audit complete
- ~~Missing parallelRenderEncoder~~: **CLOSED** (N=3690) - already implemented in v2.9

**All observed stability may be coincidental if critical code paths bypass our swizzle.** The only true fix is binary patching the AGX driver (requires SIP disabled).

---

## Quick Start

```bash
# Build all versions
cd agx_fix && make

# Recommended for this repo's verification runs (FAILS if new crash logs appear):
cd .. && ./scripts/run_test_with_crash_check.sh python3 tests/test_semaphore_recommended.py

# Or inject directly (recommended version: v2.9; run from repo root)
DYLD_INSERT_LIBRARIES=agx_fix/build/libagx_fix_v2_9.dylib MPS_FORCE_GRAPH_PATH=1 python3 your_script.py
```

## Recommended: Use Shared Models Instead

Prefer a **single shared model** instance across threads rather than creating separate model instances per thread:

```python
# Higher crash rate: separate model per thread
models = [Model().to('mps') for _ in range(num_threads)]
def worker(tid):
    model = models[tid]  # Each thread has its own model
    ...

# Better: single shared model (reduces crash probability; still not 0% under long stress)
model = Model().to('mps').eval()
def worker(tid):
    y = model(x)  # All threads share the same model
    ...
```

See `scripts/test_transformer_shared.py` for a working example.

## Recommended: Semaphore(2) Throttling to Minimize Crashes

For heavy multi-threaded workloads, limit MPS concurrency to **2** simultaneous operations to reduce the probability of AGX driver race conditions (triggered at 3+ in-flight command buffers):

**Note**: This achieves 0% observed crashes under test conditions but is not a formal guarantee. See `VERIFICATION_GAPS_ROADMAP.md` for limitations.

```python
import threading
_mps_throttle = threading.Semaphore(2)

def mps_operation():
    with _mps_throttle:
        y = model(x.to("mps"))
        torch.mps.synchronize()
        return y
```

See `README.md` and `WORKER_DIRECTIVE.md` for the latest verified results and recommended usage.

## Versions

| Version | File | Status | Description |
|---------|------|--------|-------------|
| **v2.9** | `libagx_fix_v2_9.dylib` | **Recommended** | Closes formal verification gaps: commit race window, timeout escape hatch, parallel encoder coverage |
| v2.8 | `libagx_fix_v2_8.dylib` | Superseded | Adds event-safety for Bug #048 (`encodeSignalEvent:value:`) + commit-safety |
| v2.7 | `libagx_fix_v2_7.dylib` | Superseded | Adds commit-safety for `-[IOGPUMetalCommandBuffer validate]` SIGABRT |
| v2.5 | `libagx_fix_v2_5.dylib` | Superseded | Encoder tracking + mutex protection (no commit-safety) |
| v2.4 NR | `libagx_fix_v2_4_nr.dylib` | Verification-only | Never releases encoders (leaks memory); maximizes encoder lifetime |
| v2.6 | `libagx_fix_v2_6.dylib` | Disproven | Blocks destroyImpl - breaks object lifecycle |
| v2.4 | `libagx_fix_v2_4.dylib` | Superseded | Crashes on untracked encoders |
| v2.3 | `libagx_fix_v2_3.dylib` | Superseded | Complete encoder coverage, mutex protection |
| v2.2 | `libagx_fix_v2_2.dylib` | Buggy | Removed mutex protection - crashes at 8 threads |
| v2 | `libagx_fix_v2.dylib` | Superseded | Retain-from-creation approach |
| v1 | `libagx_fix.dylib` | Superseded | Original method swizzling |

## How It Works

The dylibs use Objective-C runtime method swizzling to intercept Metal encoder creation and method calls. They add:

1. **Mutex protection** - Serializes encoder method calls to prevent races
2. **Lifecycle tracking** - Retains encoders during active use
3. **Safe cleanup** - Ensures encoders are released properly

## Limitations

The userspace fix provides **partial protection only** at full concurrency:

- **Works for most use cases** - Single-threaded and low-concurrency
- **Can still crash** at heavy workloads with 3+ concurrent MPS operations (driver-level race)
- **Use Semaphore(2)** throttling for stable operation under heavy multi-threading (see above)
- **Cannot fix driver-level bugs** - BIF0 page faults occur inside the AGX driver
- **Always check crash logs** - macOS can write `.ips` reports even when Python looks “fine”; see `crash_logs/` and `python3 scripts/check_crashes.py --latest`

## Root Cause

Crashes occur at multiple levels:

1. **PAC failures** - Encoder/object lifecycle races (what swizzling tries to fix)
2. **BIF0 page faults** - GPU/driver level races (cannot be fixed by userspace)

The complete fix requires patching the AGX driver binary (see `agx_patch/`).

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `AGX_FIX_DISABLE` | 0 | Set to 1 to disable the fix |
| `AGX_FIX_VERBOSE` | 0 | Set to 1 for verbose logging |
| `AGX_FIX_DEADLOCK_DETECT` | 0 | Enable mutex wait/deadlock diagnostics (logs when mutex acquisition is slow) |
| `AGX_FIX_LOCK_WARN_MS` | 1000* | (With deadlock detect) First warning threshold in ms for mutex acquisition |
| `AGX_FIX_LOCK_LOG_INTERVAL_MS` | 5000* | (With deadlock detect) Repeat warning interval in ms while still waiting |
| `AGX_FIX_LOCK_TIMEOUT_MS` | 0* | (With deadlock detect) Timeout threshold in ms; logs an error when exceeded |
| `AGX_FIX_LOCK_ABORT_ON_TIMEOUT` | 0* | (With deadlock detect) If set, aborts the process once timeout is exceeded |

\* Defaults apply when `AGX_FIX_DEADLOCK_DETECT=1` is set.

## Building

```bash
cd agx_fix
make clean && make

# Build specific version
make v2_9
make v2_8
make v2_7
make v2_5
make v2_6
```

## Testing

```bash
# From repo root:

# Test shared model approach (recommended)
./scripts/run_mps_test.sh scripts/test_transformer_shared.py

# Test with dylib injection + crash log capture
MPS_USE_AGX_FIX=1 ./scripts/run_mps_test.sh scripts/test_transformer_shared.py

# Force a specific dylib (e.g., v2.9) with the wrapper
AGX_FIX_DYLIB=agx_fix/build/libagx_fix_v2_9.dylib MPS_USE_AGX_FIX=1 \
    ./scripts/run_mps_test.sh scripts/test_transformer_shared.py
```
