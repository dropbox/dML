# Performance Comparison Charts

## Figure 10: Threading Throughput vs Thread Count

```
    Total Operations per Second (Threading)
    ═════════════════════════════════════════

    ops/s
    4000 ┤                ╭────────────────────────
         │              ╭─╯
    3800 ┤            ╭─╯
         │          ╭─╯
    3600 ┤        ╭─╯
         │      ╭─╯
    3400 ┤    ╭─╯
         │  ╭─╯
    3200 ┤╭─╯
         │
    3000 ┼────────────────────────────────────────
         1     2     4     8    16    32
                    Threads

    Data Points:
    ┌──────────┬───────────┬────────────┬────────────┐
    │ Threads  │ Total ops/s│ Per-thread │ Efficiency │
    ├──────────┼───────────┼────────────┼────────────┤
    │    1     │   3,301   │   3,301    │   100%     │
    │    2     │   3,528   │   1,764    │    53%     │
    │    4     │   3,805   │     951    │    29%     │
    │    8     │   3,752   │     469    │    14%     │
    │   16     │   3,819   │     239    │     7%     │
    └──────────┴───────────┴────────────┴────────────┘

    KEY INSIGHT: Total throughput plateaus at ~3,900 ops/s
                 regardless of thread count (GPU command queue bottleneck)
```

## Figure 11: Threading Efficiency Decay

```
    Per-Thread Efficiency (% of single-thread throughput)
    ════════════════════════════════════════════════════════

    %
    100 ┤●
        │╲
     90 ┤ ╲
        │  ╲
     80 ┤   ╲
        │    ╲
     70 ┤     ╲
        │      ╲
     60 ┤       ╲
        │        ●
     50 ┤         ╲
        │          ╲
     40 ┤           ╲
        │            ╲
     30 ┤             ●
        │              ╲
     20 ┤               ╲
        │                ●
     10 ┤                 ╲
        │                  ●
      0 ┼───────────────────────────────────
         1     2     4     8    16
                   Threads

    CAUSE: GPU command queue serializes all work
           More threads = same total work distributed across more threads
           Threading provides ISOLATION, not THROUGHPUT improvement
```

## Figure 12: Batching Throughput (Logarithmic Scale)

```
    Samples per Second vs Batch Size (Log Scale)
    ═══════════════════════════════════════════════

    samples/s
    2,000,000 ┤                              ╭●
              │                            ╭─╯
    1,000,000 ┤                          ╭─╯
              │                        ╭─╯
      500,000 ┤                      ╭─╯
              │                    ╭─╯
      200,000 ┤                  ╭─╯
              │                ╭─╯
      100,000 ┤              ╭●╯
              │            ╭─╯
       50,000 ┤          ╭●╯
              │        ╭─╯
       20,000 ┤      ╭●╯
              │    ╭─╯
       10,000 ┤  ╭●╯
              │●─╯
        5,000 ┼─────────────────────────────────
              1    8   16   32   64  128  256
                       Batch Size

    ┌────────────┬─────────────┬────────────────┐
    │ Batch Size │ Samples/sec │  vs Batch=1    │
    ├────────────┼─────────────┼────────────────┤
    │     1      │     9,983   │      1.0x      │
    │     8      │    78,446   │      7.9x      │
    │    64      │   604,698   │     60.6x      │
    │   256      │ 1,424,151   │    142.7x      │
    └────────────┴─────────────┴────────────────┘

    INSIGHT: Batching scales nearly linearly with batch size
             GPU utilization improves dramatically with larger batches
```

## Figure 13: Threading vs Batching Comparison (Head-to-Head)

```
    Throughput Comparison: Threading vs Batching
    ══════════════════════════════════════════════

                         Threading              Batching
                         (8 threads)            (batch=8)
                              │                     │
                              ▼                     ▼
    ┌────────────────────────────────────────────────────────┐
    │                                                         │
    │  ████████  3,900 ops/s                                 │
    │  ████████                                               │
    │                                                         │
    │                        ████████████████████████████████ │
    │                        ████████████████████████████████ │  78,446 samples/s
    │                        ████████████████████████████████ │  (20x better)
    │                        ████████████████████████████████ │
    │                                                         │
    └────────────────────────────────────────────────────────┘

    At parallelism N=64:
    ┌────────────────────────────────────────────────────────┐
    │                                                         │
    │  █  3,900 ops/s (threading)                            │
    │                                                         │
    │     ████████████████████████████████████████████████████│
    │     ████████████████████████████████████████████████████│ 604,698 samples/s
    │     ████████████████████████████████████████████████████│ (155x better)
    │     ████████████████████████████████████████████████████│
    │                                                         │
    └────────────────────────────────────────────────────────┘

    At parallelism N=256:
    ┌────────────────────────────────────────────────────────┐
    │                                                         │
    │    3,900 ops/s (threading - barely visible)            │
    │                                                         │
    │  ██████████████████████████████████████████████████████│
    │  ██████████████████████████████████████████████████████│ 1,424,151 samples/s
    │  ██████████████████████████████████████████████████████│ (365x better!)
    │  ██████████████████████████████████████████████████████│
    │  ██████████████████████████████████████████████████████│
    └────────────────────────────────────────────────────────┘


    CONCLUSION:
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║  Batching achieves 20-365x higher throughput than threading           ║
    ║                                                                        ║
    ║  USE THREADING FOR: Multi-tenant isolation, safety guarantees         ║
    ║  USE BATCHING FOR:  Maximum throughput, production inference          ║
    ╚═══════════════════════════════════════════════════════════════════════╝
```

## Figure 14: Mutex Overhead Analysis

```
    Mutex Overhead: Global vs Per-Encoder
    ═════════════════════════════════════════

                     Global Mutex           Per-Encoder Mutex
    Overhead:        0.34% ± 2.5%           ~0.1%
                          │                       │
                          ▼                       ▼
    ┌───────────────────────────────────────────────────────┐
    │                                                        │
    │         Without Mutex    With Mutex                    │
    │         ─────────────    ──────────                    │
    │                                                        │
    │  8 thr:    N/A (crash)    8,952 ops/s                 │
    │ 16 thr:    N/A (crash)    9,235 ops/s                 │
    │                                                        │
    │         Contention:                                    │
    │         Global: 2.88%    Per-Encoder: 0.00%           │
    │                                                        │
    └───────────────────────────────────────────────────────┘

    95% Confidence Interval for Overhead:
    ┌─────────────────────────────────────────┐
    │                                          │
    │    -2.2%  ◄────── 0.34% ──────►  +2.9%  │
    │                    │                     │
    │         Statistically indistinguishable │
    │              from ZERO                   │
    │                                          │
    └─────────────────────────────────────────┘

    INSIGHT: The GPU command queue is the bottleneck,
             NOT the CPU mutex synchronization
```
