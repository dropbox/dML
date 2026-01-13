# Apple Feedback Package: AGX Driver Race Condition

This package contains all evidence for the AGX Metal driver race condition bug report.

## Contents

```
apple_feedback/
├── FEEDBACK_SUBMISSION.md     # Main bug report text
├── README.md                  # This file
├── crash_reports/             # Crash analysis reports
│   ├── CRASH_ANALYSIS_2025-12-20_173618.md
│   └── CRASH_ANALYSIS_2025-12-20_174241.md
├── tla_proofs/                # TLA+ formal verification
│   ├── AGXContextRace.tla     # Buggy driver model (proves race exists)
│   └── AGXContextFixed.tla    # Fixed model (proves mutex works)
├── lean4_proofs/              # Lean 4 machine-checked proofs
│   ├── Types.lean             # Common type definitions
│   ├── Race.lean              # race_condition_exists theorem
│   ├── Fixed.lean             # mutex_prevents_race theorem
│   ├── PerStreamMutex.lean    # per_stream_mutex_insufficient theorem
│   ├── PerOpMutex.lean        # per_op_mutex_insufficient theorem
│   └── RWLock.lean            # rw_lock_insufficient theorem
└── reproduction/              # Minimal test cases
    ├── metal_race_repro.mm    # Pure Metal compute version
    └── metal_race_repro_mps.mm # MetalPerformanceShaders version
```

## Quick Start

### Build Reproduction Tests

```bash
# Pure Metal version
clang++ -std=c++17 -framework Metal -framework Foundation \
  -O2 -o metal_race_repro reproduction/metal_race_repro.mm

# MPS version
clang++ -std=c++17 -framework Metal -framework Foundation \
  -framework MetalPerformanceShaders \
  -O2 -o metal_race_repro_mps reproduction/metal_race_repro_mps.mm
```

### Run Tests

```bash
# Run tests - may crash or show "Context leak detected" warnings
./metal_race_repro
./metal_race_repro_mps
```

### PyTorch Test (Most Reliable)

```bash
pip install torch

# This crashes ~55% of the time on affected systems
python3 -c "
import torch, threading, torch.nn as nn
device = torch.device('mps')
model = nn.Linear(512, 512).to(device).eval()
def worker():
    x = torch.randn(1, 512, device=device)
    for _ in range(50):
        with torch.no_grad(): model(x)
        torch.mps.synchronize()
[t.start() for t in (threads := [threading.Thread(target=worker) for _ in range(8)])]
[t.join() for t in threads]
print('Success')
"
```

## TLA+ Verification

To verify the TLA+ models:

```bash
# Install TLA+ Toolbox or tla2tools.jar
# Run TLC model checker on AGXContextRace.tla

java -jar tla2tools.jar -config AGXContextRace.cfg AGXContextRace.tla
```

The model checker will find a counterexample demonstrating the race condition.

## Lean 4 Machine-Checked Proofs

The `lean4_proofs/` directory contains machine-checked proofs in Lean 4 (a theorem prover):

| File | Theorem | Proves |
|------|---------|--------|
| Race.lean | `race_condition_exists` | The race condition can occur without synchronization |
| Fixed.lean | `mutex_prevents_race` | A global mutex prevents all race manifestations |
| PerStreamMutex.lean | `per_stream_mutex_insufficient` | Per-stream mutexes are insufficient |
| PerOpMutex.lean | `per_op_mutex_insufficient` | Per-operation mutexes are insufficient |
| RWLock.lean | `rw_lock_insufficient` | Reader-writer locks are insufficient |

To verify (requires Lean 4 and Lake):

```bash
# From the mps-verify directory
lake build
# Build completed successfully (50 jobs).
```

These proofs provide machine-verified evidence that the global mutex is the minimal correct solution.

## Bug Summary

- **Component**: AGXMetalG16X driver (Apple Silicon GPU)
- **Symptom**: SIGSEGV at addresses 0x5c8, 0x98, 0x184 (NULL + offset)
- **Trigger**: Multi-threaded MTLComputeCommandEncoder use
- **Crash Rate**: ~55% with 8 threads
- **Workaround**: Global mutex around all encoding operations
