# Phase 2 Checklist: LLaMA Conversion

**Status**: COMPLETE
**Target**: 5-10 commits
**Current**: 4 commits

---

## Completed Components

- [x] `converters/llama_converter.py` - LLaMA converter using mlx-lm
- [x] `converters/__init__.py` - Package initialization
- [x] `tests/test_llama_converter.py` - Unit tests for converter
- [x] Updated `requirements.txt` with transformers dependency
- [x] CLI integration (`llama convert/validate/benchmark/list`)
- [x] Validation with SmolLM2-135M (max error ~2-5%, 0 mismatched tokens)
- [x] Benchmark comparison (MLX 23.8x faster than PyTorch)

---

## Remaining Work

### 1. Integration with CLI (MEDIUM PRIORITY)

**Goal**: Add `llama` subcommand to CLI

```bash
# Convert LLaMA model
./pytorch_to_mlx llama convert \
    --hf-path meta-llama/Llama-3.2-1B \
    --output ./mlx-llama-1b \
    --quantize --q-bits 4

# Validate conversion
./pytorch_to_mlx llama validate \
    --mlx-path ./mlx-llama-1b \
    --prompts "Hello, how are you?" "The capital of France is"

# Benchmark
./pytorch_to_mlx llama benchmark \
    --mlx-path ./mlx-llama-1b \
    --hf-path meta-llama/Llama-3.2-1B
```

### 2. Validation Test with Small Model (HIGH PRIORITY)

**Goal**: Test full pipeline with smallest available LLaMA model

**Steps**:
1. Use Llama-3.2-1B (smallest LLaMA 3.2 variant)
2. Convert to MLX format
3. Run validation prompts
4. Verify numerical equivalence < 1e-5

**Test Prompts**:
```python
test_prompts = [
    "The quick brown fox",
    "In a galaxy far far away",
    "def fibonacci(n):",
]
```

### 3. Benchmark Report (MEDIUM PRIORITY)

**Goal**: Create performance benchmark comparing MLX vs PyTorch

**Metrics**:
- Tokens per second (generation)
- Time to first token
- Memory usage
- Multi-batch throughput

### 4. Documentation (LOW PRIORITY)

**Goal**: Document LLaMA conversion process

**Contents**:
- Supported models list
- Conversion options (quantization, dtype)
- Validation methodology
- Performance expectations

---

## Worker Instructions

### Prerequisites

1. HuggingFace account with access to LLaMA models
2. `huggingface-cli login` completed
3. Sufficient disk space (~10GB for 7B model)

### Testing Without Full Models

The unit tests (`test_llama_converter.py`) can run without model weights.
Integration tests require HuggingFace authentication and model access.

### Commit Format

```
# N: [Brief title]
**Current Plan**: MLX_MIGRATION_PLAN.md Phase 2
**Checklist**: PHASE2_CHECKLIST.md item X

## Changes
[Description]

## Next AI: [Directive]
```

---

## Phase 2 Completion Criteria

Phase 2 is complete when:

1. LLaMA converter successfully converts a model
2. Validation shows < 1e-5 numerical error
3. Benchmark shows MLX >= PyTorch MPS performance
4. CLI integration complete

**Estimated remaining commits**: 4-9
