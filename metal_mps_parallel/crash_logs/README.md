# Crash Logs

This directory stores crash logs automatically captured from macOS diagnostic reports.

## Files

| File | Purpose |
|------|---------|
| `crash_summary.json` | JSON summary of all captured crashes |
| `latest_crash.txt` | Path to the most recent crash log |
| `*.ips` / `*.crash` | Actual crash report files (copied from macOS) |
| `monitor.log` | Output from background crash monitor |

## Usage

### Check for recent crashes

```bash
python3 scripts/check_crashes.py
```

### View latest crash details

```bash
python3 scripts/check_crashes.py --latest
```

### Watch for new crashes

```bash
python3 scripts/check_crashes.py --watch
```

### Start background monitor

```bash
./scripts/crash_monitor.sh start
```

## How It Works

1. When a Python process crashes (SIGSEGV, etc.), macOS writes a crash report to `~/Library/Logs/DiagnosticReports/`
2. The `run_worker.sh` script checks for new crash reports after each worker iteration
3. Relevant crashes (Python, AGX, Metal, MPS, torch) are copied here
4. Workers can read `crash_summary.json` or use `check_crashes.py` to see crash details

## For AI Workers

If you see a crash, run:

```bash
python3 scripts/check_crashes.py --latest
```

This will show:
- Exception type (SIGSEGV, SIGABRT, etc.)
- Fault address (e.g., 0x5c8 for AGX race condition)
- Crashed function and stack trace
