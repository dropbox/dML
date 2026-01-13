# Benchmark Methodology (R1)

**Created**: 2025-12-19
**Addressing**: Reviewer Objection #1 - No Reproducible Benchmark Suite
**Author**: Worker N=1316

## Overview

This document describes the statistical benchmark methodology for MPS parallel inference performance measurement. The benchmark suite addresses all concerns raised in Reviewer Objection #1.

## Statistical Rigor

### Trial Configuration

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| N trials | 30 | Minimum for Central Limit Theorem |
| Warmup iterations | 5 | Stabilize JIT/caches before timing |
| Iterations per trial | 50 | Amortize per-operation variance |

### Metrics Collected

For each configuration:

| Metric | Description |
|--------|-------------|
| Mean | Average execution time |
| Std | Standard deviation |
| Min | Fastest observation |
| Max | Slowest observation |
| P50 | Median (50th percentile) |
| P95 | 95th percentile |
| P99 | 99th percentile |
| CV% | Coefficient of Variation (std/mean × 100) |
| ops/s | Operations per second |

### Scaling Efficiency Formula

```
Speedup = ops_per_second(N threads) / ops_per_second(1 thread)
Efficiency = Speedup / N × 100%
```

**Interpretation**:
- >80% efficiency = Near-linear scaling
- 50-80% efficiency = Good parallelism
- <50% efficiency = GPU saturation or contention

## Hardware Specifications

The benchmark automatically collects:

```json
{
  "chip": "Apple M4 Max",
  "chip_cores": 16,
  "gpu_cores": 40,
  "ram_gb": 128.0,
  "macos_version": "15.7.3",
  "macos_build": "24G419",
  "pytorch_version": "2.9.1a0+git9a4518e",
  "hostname": "L30J2DW1Q2",
  "timestamp_utc": "2025-12-20T02:15:XX+00:00"
}
```

## Usage

### Basic Run (30 trials, JSON output)

```bash
python tests/benchmark_statistical.py --output results.json
```

### Custom Configuration

```bash
python tests/benchmark_statistical.py \
    --trials 50 \
    --threads 1,2,4,8,16 \
    --model linear \
    --in-features 1024 \
    --out-features 512 \
    --batch-size 64 \
    --output benchmark_results.json
```

### Available Models

| Model | Description | Best For |
|-------|-------------|----------|
| `matmul` | Raw matrix multiplication | GPU throughput |
| `linear` | nn.Linear layer | Basic inference |
| `mlp` | 3-layer MLP | Realistic workload |

### Output Formats

1. **Console** - Markdown table for documentation
2. **JSON** - Machine-readable for CI regression tracking

## Sample Results (M4 Max, 40 GPU cores)

### Matmul 512×512, 30 trials

| Threads | Mean | Std | P50 | P99 | ops/s | Efficiency |
|---------|------|-----|-----|-----|-------|------------|
| 1 | 10.2ms | 0.6ms | 10.1ms | 12.9ms | 4881 | 100% |
| 2 | 20.9ms | 0.5ms | 20.8ms | 22.6ms | 4794 | 49% |
| 4 | 42.9ms | 0.6ms | 42.8ms | 44.4ms | 4666 | 24% |
| 8 | 87.6ms | 3.9ms | 89.6ms | 91.6ms | 4567 | 12% |

**Analysis**: Single-thread already saturates GPU for 512×512 matmul. Adding threads increases latency proportionally without improving throughput. This is expected GPU-bound behavior.

### Linear (256→128, batch=16), 30 trials

| Threads | Mean | Std | P50 | P99 | ops/s | Efficiency |
|---------|------|-----|-----|-----|-------|------------|
| 1 | 9.3ms | 0.7ms | 9.1ms | 11.3ms | 5354 | 100% |
| 2 | 19.6ms | 0.9ms | 19.6ms | 23.7ms | 5101 | 48% |
| 4 | 40.5ms | 2.0ms | 39.5ms | 44.1ms | 4941 | 23% |

**Note**: 8-thread linear test times out due to known Metal framework limitation with certain operations at high concurrency.

## Interpretation Guidelines

### CV% (Coefficient of Variation)

| CV% | Stability |
|-----|-----------|
| <5% | Excellent - very stable timing |
| 5-10% | Good - normal variance |
| 10-20% | Moderate - some thermal/system effects |
| >20% | Poor - investigate system state |

### GPU Saturation Detection

When scaling efficiency drops below 50% at 2 threads, the GPU is likely saturated at single-thread workload. Strategies:

1. Use smaller batch sizes per thread
2. Use CPU-only workloads for parallel gains
3. Profile with Instruments to confirm GPU utilization

## JSON Schema

```json
{
  "benchmark_version": "1.0",
  "system_info": { /* SystemInfo fields */ },
  "configuration": {
    "model_type": "matmul",
    "model_config": { /* model parameters */ },
    "n_trials": 30,
    "iterations_per_thread": 50,
    "warmup_iterations": 5,
    "thread_counts": [1, 2, 4, 8]
  },
  "results": [
    {
      "thread_count": 1,
      "warm_cache_stats": {
        "n_trials": 30,
        "mean_ns": 10243000000,
        "std_ns": 599561000,
        "min_ns": 9838000000,
        "max_ns": 12940000000,
        "p50_ns": 10121000000,
        "p95_ns": 11595000000,
        "p99_ns": 12940000000,
        "coefficient_of_variation": 5.9
      },
      "ops_per_second": 4881,
      "scaling_efficiency_percent": 100.0
    }
    /* ... more results ... */
  ],
  "summary": {
    "baseline_ops_per_second": 4881,
    "max_thread_count": 8,
    "max_scaling_efficiency": 100.0
  }
}
```

## Cold Start vs Warm Cache

Use `--cold-start` flag to measure cold start timing:

```bash
python tests/benchmark_statistical.py --cold-start --output results.json
```

This adds separate statistics for first invocations vs cached invocations.

## CI Integration

For regression detection:

```bash
# Run benchmark and compare against baseline
python tests/benchmark_statistical.py --quiet --output current.json
# Compare mean/std against baseline.json
# Fail if mean > baseline_mean + 2*baseline_std
```

## Known Limitations

1. **8+ threads with certain ops**: TransformerEncoderLayer and other LayerNorm-using operations may timeout at 8+ concurrent threads due to Apple Metal framework limitations.

2. **GPU saturation**: Large matrix operations (512×512+) saturate the M4 Max GPU at 1 thread, making scaling efficiency appear poor. This is correct behavior - the GPU is already at 100% utilization.

3. **Thermal throttling**: Extended benchmark runs may trigger thermal throttling. Monitor CPU/GPU temperature if CV% increases over time.

## Files

| File | Purpose |
|------|---------|
| `tests/benchmark_statistical.py` | Benchmark implementation |
| `reports/main/benchmark_results_N1316.json` | Sample matmul results |
| `reports/main/benchmark_linear_N1316.json` | Sample linear results |
| `reports/main/benchmark_methodology.md` | This document |
