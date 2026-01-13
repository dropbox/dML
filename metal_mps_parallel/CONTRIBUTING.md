# Contributing to MPS Parallel Inference

**Created by Andrew Yates**

This project modifies PyTorch's MPS backend to enable thread-safe parallel inference on Apple Silicon. The primary goal is upstream contribution to PyTorch.

## Quick Links

- **Patch**: `patches/cumulative-v2.9.1-to-mps-stream-pool.patch` (7,226 lines, 50 files)
- **Tests**: `tests/run_all_tests.sh` (25 tests)
- **Requirements**: [UPSTREAM_SUBMISSION_AUDIT.md](UPSTREAM_SUBMISSION_AUDIT.md)
- **Proof**: [SUBMISSION_PROOF.md](SUBMISSION_PROOF.md)

---

## How to Contribute

### 1. Apply the Patch

```bash
# Clone PyTorch
git clone https://github.com/pytorch/pytorch.git pytorch-mps-fork
cd pytorch-mps-fork
git checkout v2.9.1
git submodule update --init --recursive

# Apply the patch
git apply ../patches/cumulative-v2.9.1-to-mps-stream-pool.patch
```

### 2. Build PyTorch

```bash
# Install dependencies
pip install -r requirements.txt

# Build (editable install)
python -m pip install -e . -v --no-build-isolation

# Verify
python -c "import torch; print(torch.__version__); print(torch.backends.mps.is_available())"
```

### 3. Run Tests

```bash
# Run all parallel inference tests
cd ..  # Back to metal_mps_parallel/
./tests/run_all_tests.sh

# If the runner refuses due to a stale torch build (git hash mismatch), rebuild:
# (cd pytorch-mps-fork && USE_MPS=1 USE_CUDA=0 BUILD_TEST=0 python -m pip install -e . -v --no-build-isolation)
# Override (unsafe): MPS_TESTS_ALLOW_TORCH_MISMATCH=1 ./tests/run_all_tests.sh

# Expected output:
# Passed: 25
# Failed: 0
# ALL TESTS PASSED
```

### 4. Verify Code Style

```bash
# C++ formatting (must pass)
clang-format --dry-run -Werror pytorch-mps-fork/aten/src/ATen/mps/MPSStream.h

# Python formatting (must pass)
flake8 pytorch-mps-fork/torch/mps/__init__.py
```

---

## Code Style Requirements

### C++/Objective-C++

- Follow existing PyTorch ATen patterns
- Use `clang-format` before submitting
- Add THREAD-SAFETY comments for mutex-protected sections
- Use `std::lock_guard` for RAII locking
- Use `dispatch_sync_with_rethrow` for GCD blocks

### Python

- Follow PEP 8
- Use Google-style docstrings
- Run `flake8` before submitting

---

## Testing Requirements

1. All existing MPS tests must pass
2. New parallel inference tests must pass
3. Thread Sanitizer (TSan) must show 0 data races

---

## Upstream Submission

This patch is designed for upstream contribution to pytorch/pytorch. See:

- [SUBMISSION_PROOF.md](SUBMISSION_PROOF.md) - Proof of all PyTorch requirements
- [UPSTREAM_SUBMISSION_AUDIT.md](UPSTREAM_SUBMISSION_AUDIT.md) - Full requirements checklist

### Human Actions Required

Before submitting the PR, a human must:

1. **Sign the CLA**: PyTorch's CLA bot will prompt at PR submission
2. **Create GitHub Issue**: File an issue on pytorch/pytorch describing the feature
3. **Submit PR**: Use the template in SUBMISSION_PROOF.md
4. **Respond to Reviews**: Address reviewer feedback

---

## Development Artifacts

### Historical Reports

The `reports/main/` directory contains 115 verification reports from the development process (572KB total). These provide a complete audit trail of:
- Bug discovery and verification
- Test results at each phase
- Performance measurements
- Code review findings

**Archival Strategy**: Reports are kept in the repo for transparency and audit purposes. They document the rigorous verification process but are not required for using or understanding the patch.

### Patch Archives

Incremental patches (001-034) are archived in `patches/archive/` for development history reference. Only use `cumulative-v2.9.1-to-mps-stream-pool.patch` for applying changes.

---

## License

This project is licensed under BSD-3-Clause, matching PyTorch's license. See [LICENSE](LICENSE).
