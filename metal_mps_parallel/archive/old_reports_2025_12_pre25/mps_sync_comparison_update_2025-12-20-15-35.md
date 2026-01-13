# MPS Sync Comparison Update

**Date:** 2025-12-20 15:35 PST
**Hardware:** Apple M4 Max (40 GPU cores), macOS 15.7.3
**Torch:** 2.9.1a0+git4201c80

## Summary

Re-ran `tests/mps_sync_comparison.py` to sanity-check the 1-thread baseline for event sync and to capture updated multi-thread scaling numbers. The earlier `reports/main/event_sync_verification_N1399_2025-12-20.md` report includes an anomalously low 1-thread event sync throughput (0.45x vs device), which materially inflated the derived 8-thread "event efficiency" number.

## Results (512x512 FP16 matmul)

| Threads | Device Sync (ops/s) | Event Sync (ops/s) | Event/Device Ratio |
|---------|---------------------|--------------------|-------------------|
| 1       | 5,906               | 6,187              | 1.05x             |
| 2       | 6,981               | 13,870             | 1.99x             |
| 4       | 6,473               | 14,683             | 2.27x             |
| 8       | 6,418               | 17,502             | 2.73x             |

Derived metrics:
- Device sync: 8-thread scaling 1.09x, efficiency 13.6%
- Event sync: 8-thread scaling 2.83x, efficiency 35.4%

## Command

```bash
python3 tests/mps_sync_comparison.py
```
