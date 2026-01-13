# Research Paper Figures Index

## Overview

This directory contains ASCII art diagrams and charts for the AGX race condition research paper. All figures are in Markdown format and can be viewed directly in any text editor or markdown renderer.

## Figure List

### MPS Architecture (`mps_architecture.md`)

| Figure | Title | Description |
|--------|-------|-------------|
| 1 | Original PyTorch MPS Architecture | Thread-unsafe singleton design |
| 2 | Thread-Safe MPS Architecture | Stream pool design with mutex |
| 3 | Round-Robin Stream Allocation | CUDA-style allocation pattern |

### Race Condition Timeline (`race_condition_timeline.md`)

| Figure | Title | Description |
|--------|-------|-------------|
| 4 | Race Condition Sequence Diagram | Step-by-step crash scenario |
| 5 | Detailed State Machine | TLA+/Lean4 model states |
| 6 | Mutex Protection Timeline | How fix prevents race |

### Memory Layout (`memory_layout.md`)

| Figure | Title | Description |
|--------|-------|-------------|
| 7 | ContextCommon Structure Layout | Inferred from crash offsets |
| 8 | Three Crash Sites | Detailed crash site analysis |
| 9 | How NULL Pointer Reaches Driver | Normal vs buggy flow |

### Performance Charts (`performance_charts.md`)

| Figure | Title | Description |
|--------|-------|-------------|
| 10 | Threading Throughput vs Thread Count | Shows plateau at ~3,900 ops/s |
| 11 | Threading Efficiency Decay | Per-thread efficiency drops |
| 12 | Batching Throughput | Logarithmic scaling with batch size |
| 13 | Threading vs Batching Comparison | 20-365x advantage for batching |
| 14 | Mutex Overhead Analysis | 0.34% overhead (indistinguishable) |

### Evidence Chain (`evidence_chain.md`)

| Figure | Title | Description |
|--------|-------|-------------|
| 15 | Complete Evidence Chain | All evidence levels connected |
| 16 | Verification Pipeline | TLA+ and Lean4 workflow |
| 17 | Evidence Cross-Reference Matrix | How evidence types intersect |

## Usage

These figures are designed to be:

1. **Readable in terminal**: Works with any monospace font
2. **Embeddable in markdown**: Copy directly into documents
3. **Convertible to images**: Can use tools like `carbon-now-cli` or `monodraw` for image export

## Creating Image Versions

To create PNG versions (optional):

```bash
# Using carbon-now-cli
carbon-now papers/figures/mps_architecture.md -t nord

# Using screenshot tool
cat papers/figures/mps_architecture.md | pbcopy
# Then paste into a terminal and screenshot
```

## Paper References

These figures are referenced in:
- `papers/agx_race_condition_research.md` (main paper)
- `apple_feedback/FEEDBACK_SUBMISSION.md` (Apple bug report)
- `reports/main/*.md` (individual analysis reports)
