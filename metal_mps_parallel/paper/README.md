# Research Paper: MPS Thread-Safety Analysis

## Paper

**Title**: Formal Verification of GPU Framework Thread-Safety: A Case Study of Apple MetalPerformanceShaders

**Abstract**: We present a comprehensive analysis of thread-safety issues in Apple's MetalPerformanceShaders (MPS) framework using state-of-the-art formal verification techniques. Through binary reverse engineering combined with TLA+ model checking and Lean 4 theorem proving, we identify the root cause of race conditions that affect major machine learning frameworks including PyTorch and TensorFlow on Apple Silicon.

## Target Venues

| Venue | Focus | Deadline |
|-------|-------|----------|
| OSDI | Operating systems | Spring/Fall |
| SOSP | Systems | Biennial (odd years) |
| USENIX Security | Security research | Multiple per year |
| IEEE S&P | Security and privacy | Multiple per year |
| PLDI | Programming languages | November |
| OOPSLA | Object-oriented programming | April |

## Building

```bash
# Requires: pdflatex, bibtex (from TeX Live or MacTeX)
make

# View PDF
make view

# Word count
make wc

# Clean build files
make clean
```

## Files

| File | Description |
|------|-------------|
| `mps_thread_safety.tex` | Main paper source |
| `references.bib` | BibTeX references |
| `Makefile` | Build automation |
| `figures/` | Figures (to be added) |

## Sections to Complete

After Phase 9 (MPS Research):

1. **§3 Binary Analysis**: Fill in actual offsets and disassembly from Ghidra analysis
2. **§4 TLA+ Model**: Include actual TLA+ specs from `mps-verify/specs/`
3. **§5 Lean Proofs**: Include actual theorems from `mps-verify/MPSVerify/`
4. **§6 Patches**: Document actual PR numbers and acceptance status
5. **§7 Evaluation**: Include actual benchmark numbers

## Contributing Authors

To be determined after research phase completion.

## License

Paper content is not yet licensed for distribution. Will be determined upon submission.
