# Phase 1 Checklist: Core Infrastructure

**Status**: COMPLETE
**Target**: 15-20 commits
**Current**: 2 commits (Phase 1 infrastructure complete, all 22 tests pass)

---

## Completed Components

- [x] `analyzer/torchscript_analyzer.py` - TorchScript model parser
- [x] `analyzer/op_mapper.py` - PyTorch to MLX operation mapping
- [x] `generator/mlx_code_generator.py` - MLX code generation
- [x] `generator/weight_converter.py` - Weight format conversion (safetensors)
- [x] `requirements.txt` - Python dependencies
- [x] Package structure with `__init__.py` files

---

## Remaining Work

### 1. Numerical Validator (HIGH PRIORITY) - COMPLETE

Implemented in commit # 0 by Worker #0.

~~### 1. Numerical Validator (HIGH PRIORITY)~~

**File:** `tools/pytorch_to_mlx/validator/numerical_validator.py`

**Required Features:**
```python
class NumericalValidator:
    def __init__(self, torch_model, mlx_model, tolerance=1e-5):
        """Compare PyTorch and MLX model outputs."""

    def validate(self, test_inputs: List) -> ValidationReport:
        """Run both models, compare outputs."""
        # - Load PyTorch model
        # - Load MLX model
        # - Run identical inputs through both
        # - Compare outputs with tolerance
        # - Report: max_error, mean_error, pass/fail

    def validate_layer_by_layer(self, test_input) -> Dict[str, float]:
        """Compare intermediate activations for debugging."""
```

**Acceptance Criteria:**
- Supports TorchScript and MLX model loading
- Reports max absolute error, mean error
- Layer-by-layer comparison for debugging mismatches
- Clear pass/fail based on tolerance

---

### 2. Benchmark Tool (HIGH PRIORITY) - COMPLETE

**File:** `tools/pytorch_to_mlx/validator/benchmark.py`

**Required Features:**
```python
class Benchmark:
    def __init__(self, warmup_runs=10, benchmark_runs=100):
        """Performance comparison tool."""

    def benchmark_model(self, model, inputs, device='cpu') -> BenchmarkResult:
        """Measure latency and throughput."""
        # - Warmup runs
        # - Timed runs
        # - Report: mean, std, p50, p95, p99 latency
        # - Report: throughput (items/sec)

    def compare(self, torch_model, mlx_model, inputs) -> ComparisonReport:
        """Compare PyTorch MPS vs MLX performance."""
```

**Acceptance Criteria:**
- Accurate timing with warmup
- Reports latency statistics (mean, std, percentiles)
- Reports throughput
- Memory usage tracking
- Comparison table output

---

### 3. CLI Tool (MEDIUM PRIORITY) - COMPLETE

**File:** `tools/pytorch_to_mlx/cli.py`

**Required Interface:**
```bash
# Analyze a model
./pytorch_to_mlx analyze --input model.pt --output analysis.json

# Convert a model
./pytorch_to_mlx convert --input model.pt --output model_mlx/ --validate --benchmark

# Validate conversion
./pytorch_to_mlx validate --pytorch model.pt --mlx model_mlx/ --tolerance 1e-5

# Benchmark comparison
./pytorch_to_mlx benchmark --pytorch model.pt --mlx model_mlx/ --iterations 100
```

**Implementation:**
- Use `click` for CLI framework (already in requirements.txt)
- Integrate analyzer, converter, validator, benchmark
- Progress bars for long operations
- JSON and human-readable output

---

### 4. Unit Tests (MEDIUM PRIORITY) - COMPLETE

**Directory:** `tests/`

**Required Tests:**
```
tests/
├── test_analyzer.py      # TorchScriptAnalyzer tests
├── test_op_mapper.py     # OpMapper coverage tests
├── test_generator.py     # Code generation tests
├── test_weight_converter.py  # Weight conversion tests
├── test_validator.py     # Numerical validation tests
├── test_benchmark.py     # Benchmark accuracy tests
└── fixtures/
    └── simple_linear.pt  # Simple test model
```

**Minimum Coverage:**
- Op mapper: verify all documented mappings work
- Weight converter: round-trip conversion test
- Validator: known-good/known-bad model pairs

---

### 5. Integration Test with Simple Model

**Goal:** End-to-end test with a simple model

**Steps:**
1. Create simple PyTorch model (Linear + ReLU + Linear)
2. Export to TorchScript
3. Run full conversion pipeline
4. Validate numerical equivalence
5. Benchmark both implementations

**This validates the entire Phase 1 toolchain works.**

---

## Worker Instructions

### Starting Work

1. Check this checklist for next unclaimed item
2. Read existing code in `tools/pytorch_to_mlx/`
3. Follow existing patterns (dataclasses, type hints, docstrings)
4. Write tests alongside implementation

### Code Standards

- Type hints on all functions
- Docstrings with Args/Returns
- Use dataclasses for structured data
- Handle errors gracefully with clear messages
- No external dependencies beyond requirements.txt

### Commit Format

```
# N: [Brief title]
**Current Plan**: MLX_MIGRATION_PLAN.md Phase 1
**Checklist**: PHASE1_CHECKLIST.md item X

## Changes
[Description]

## Next AI: [Directive]
```

---

## Phase 1 Completion Criteria

Phase 1 is complete when:

1. All checklist items above are implemented
2. `./pytorch_to_mlx convert --input test.pt --output test_mlx/ --validate` works
3. Unit tests pass: `pytest tests/`
4. Integration test with simple model shows <1e-5 error

**Estimated remaining commits:** 10-15
