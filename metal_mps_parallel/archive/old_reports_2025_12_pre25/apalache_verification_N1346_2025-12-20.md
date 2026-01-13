# Apalache Symbolic Verification Report (Allocator + Event)

**Worker:** N=1346  
**Date:** 2025-12-20 02:15 PST  
**Tool:** Apalache 0.52.1 (`tools/apalache/bin/apalache-mc`)  

## Summary

Extended Apalache (Snowcat) compatibility beyond `MPSStreamPool` by adding explicit
type annotations to `MPSAllocator.tla` and `MPSEvent.tla`, along with small
Apalache config files. Verified both specs with Apalache bounded checking.

Also fixed `tools/run_all_verification.sh` to:
- ensure Apalache uses Homebrew OpenJDK (Apalache invokes `java` from `PATH`)
- stop masking verification failures (enable `pipefail`)
- avoid CBMC unwind false-failures (`--unwind 11` vs `--unwind 10`)
- keep the “parallel progress existence” TLC config as an **expected-fail** witness
  (skip it in the passing suite, then run it separately and assert non-zero exit)

## Changes

### 1) Apalache Type Annotations

Added Snowcat type annotations (`\* @type: ...;`) to:
- `specs/MPSAllocator.tla`
- `specs/MPSEvent.tla`

### 2) Apalache Configs

Added small, solver-friendly Apalache configs:
- `specs/MPSAllocator_Apalache.cfg`
- `specs/MPSEvent_Apalache.cfg`

## Verification Results

### Apalache: MPSAllocator

Command:
```bash
cd specs
PATH=/opt/homebrew/opt/openjdk/bin:$PATH ../tools/apalache/bin/apalache-mc check \
  --config=MPSAllocator_Apalache.cfg MPSAllocator.tla
```

Result (excerpt):
- `The outcome is: NoError`
- `Checker reports no error up to computation length 10`
- `Total time: 132.215 sec`

### Apalache: MPSEvent

Command:
```bash
cd specs
PATH=/opt/homebrew/opt/openjdk/bin:$PATH ../tools/apalache/bin/apalache-mc check \
  --config=MPSEvent_Apalache.cfg MPSEvent.tla
```

Result (excerpt):
- `The outcome is: NoError`
- `Checker reports no error up to computation length 10`
- `Total time: 85.701 sec`

### TLC Regression Check (bounded enumeration)

Commands:
```bash
cd specs
/opt/homebrew/opt/openjdk/bin/java -jar ../tools/tla2tools.jar -deadlock \
  -config MPSAllocator.cfg MPSAllocator.tla
/opt/homebrew/opt/openjdk/bin/java -jar ../tools/tla2tools.jar -deadlock \
  -config MPSEvent.cfg MPSEvent.tla
```

Results (tail):
- `MPSAllocator.tla`: 2,821,612 states generated, 396,567 distinct, finished in ~5s
- `MPSEvent.tla`: 11,914,912 states generated, 1,389,555 distinct, finished in ~22-27s

### CBMC Harnesses (unwind bound fix)

CBMC 6.8.0 reports an unwinding assertion failure in `batch_queue_harness.c` with
`--unwind 10` because the harness contains a fixed 10-iteration loop
(`MAX_REQUESTS = 10`). Using `--unwind 11` avoids this false-failure.

Re-checked all harnesses:
```bash
cd mps-verify/verification/cbmc/harnesses
for h in *_harness.c; do cbmc --unwind 11 "$h" >/dev/null; done
```

Result: all 10 harnesses exit 0.

## Notes

- Apalache’s `apalache-mc` wrapper executes `java` from `PATH` (not `$JAVA_HOME`),
  so systems with the Apple `/usr/bin/java` stub need `PATH` updated to prefer
  Homebrew OpenJDK.
- Apalache runs here are bounded (computation length 10) and check state
  invariants only (no WF/SF fairness).
