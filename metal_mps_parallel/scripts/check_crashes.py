#!/usr/bin/env python3
"""
Crash Log Checker for AI Workers

This module provides utilities for AI workers to check for recent crashes
and get detailed crash information.

Usage in worker code:
    from scripts.check_crashes import get_recent_crashes, get_latest_crash

Usage from command line:
    python3 scripts/check_crashes.py              # Show recent crashes
    python3 scripts/check_crashes.py --latest     # Show latest crash details
    python3 scripts/check_crashes.py --watch      # Watch for new crashes
"""

import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

REPO_ROOT = Path(__file__).parent.parent
CRASH_DIR = REPO_ROOT / "crash_logs"
CRASH_SUMMARY = CRASH_DIR / "crash_summary.json"
LATEST_CRASH = CRASH_DIR / "latest_crash.txt"


def get_crash_summary() -> Dict[str, Any]:
    """Load the crash summary JSON."""
    if not CRASH_SUMMARY.exists():
        return {"crashes": [], "last_check": None}
    try:
        with open(CRASH_SUMMARY) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {"crashes": [], "last_check": None}


def get_recent_crashes(hours: float = 24) -> List[Dict[str, Any]]:
    """Get crashes from the last N hours."""
    summary = get_crash_summary()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    recent = []
    for crash in summary.get("crashes", []):
        try:
            ts = datetime.fromisoformat(crash["timestamp"].replace("Z", "+00:00"))
            if ts > cutoff:
                recent.append(crash)
        except (KeyError, ValueError):
            continue

    return recent


def get_latest_crash() -> Optional[Dict[str, Any]]:
    """Get the most recent crash."""
    summary = get_crash_summary()
    crashes = summary.get("crashes", [])
    return crashes[0] if crashes else None


def get_crash_details(crash_file: str) -> str:
    """Read the full crash log file."""
    try:
        with open(crash_file) as f:
            return f.read()
    except IOError:
        return f"Could not read crash file: {crash_file}"


def _parse_json_sequence(text: str) -> list[Any]:
    """Parse one or more JSON objects concatenated in a single file (common for .ips)."""
    decoder = json.JSONDecoder()
    idx = 0
    objs: list[Any] = []
    length = len(text)
    while idx < length:
        while idx < length and text[idx].isspace():
            idx += 1
        if idx >= length:
            break
        try:
            obj, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            break
        objs.append(obj)
        idx = end
    return objs


def _extract_info_from_ips(crash_path: Path) -> Dict[str, str]:
    """
    Best-effort extraction of key fields from macOS .ips JSON crash reports.
    Returns keys: process, exception, fault_address, crashed_in
    """
    try:
        text = crash_path.read_text(errors="replace")
    except OSError:
        return {}

    objs = _parse_json_sequence(text)
    report: Dict[str, Any] | None = None
    for obj in reversed(objs):
        if isinstance(obj, dict):
            report = obj
            break
    if not report:
        return {}

    proc = report.get("procName") or report.get("app_name") or report.get("name") or ""

    exc = report.get("exception") if isinstance(report.get("exception"), dict) else {}
    exc_type = exc.get("type") or ""
    exc_sub = exc.get("subtype") or ""
    if exc_type and exc_sub:
        exc_str = f"{exc_type} - {exc_sub}"
    else:
        exc_str = exc_type or exc_sub or ""

    fault_addr = ""
    for candidate in (
        exc_sub,
        exc.get("codes"),
        (report.get("termination") or {}).get("indicator") if isinstance(report.get("termination"), dict) else None,
        report.get("vmRegionInfo"),
    ):
        if isinstance(candidate, str):
            m = re.search(r"0x[0-9a-fA-F]+", candidate)
            if m:
                fault_addr = m.group(0)
                break

    crashed_in = ""
    faulting_thread = report.get("faultingThread")
    threads = report.get("threads")
    if isinstance(faulting_thread, int) and isinstance(threads, list) and 0 <= faulting_thread < len(threads):
        t = threads[faulting_thread]
        if isinstance(t, dict):
            frames = t.get("frames")
            if isinstance(frames, list) and frames and isinstance(frames[0], dict):
                frame0 = frames[0]
                symbol = frame0.get("symbol") or ""
                symbol_loc = frame0.get("symbolLocation")
                image_index = frame0.get("imageIndex")
                parts: list[str] = []
                if image_index is not None:
                    parts.append(f"[{image_index}]")
                if symbol:
                    parts.append(str(symbol))
                if symbol_loc is not None:
                    parts.append(f"+ {symbol_loc}")
                crashed_in = " ".join(parts)

    info: Dict[str, str] = {}
    if proc:
        info["process"] = str(proc)
    if exc_str:
        info["exception"] = str(exc_str)
    if fault_addr:
        info["fault_address"] = fault_addr
    if crashed_in:
        info["crashed_in"] = crashed_in
    return info


def _extract_info_from_crash_text(crash_path: Path) -> Dict[str, str]:
    """
    Best-effort extraction of key fields from macOS .crash text reports.
    Returns keys: process, exception, fault_address, crashed_in
    """
    try:
        text = crash_path.read_text(errors="replace")
    except OSError:
        return {}

    lines = text.splitlines()
    info: Dict[str, str] = {}

    for line in lines[:200]:
        if "process" not in info:
            m = re.match(r"^Process:\s+(.+?)\s*\[", line)
            if m:
                info["process"] = m.group(1).strip()
                continue

        if "exception" not in info:
            m = re.match(r"^Exception Type:\s+(.*)$", line)
            if m:
                info["exception"] = m.group(1).strip()
                continue

        if "fault_address" not in info:
            m = re.match(r"^Exception Codes:\s+(.*)$", line)
            if m:
                addr = re.search(r"0x[0-9a-fA-F]+", m.group(1))
                if addr:
                    info["fault_address"] = addr.group(0)
                    continue

    crashed_thread_idx: Optional[int] = None
    for idx, line in enumerate(lines):
        if re.match(r"^Thread\s+\d+\s+Crashed", line):
            crashed_thread_idx = idx
            break

    if crashed_thread_idx is not None:
        for line in lines[crashed_thread_idx + 1 : crashed_thread_idx + 80]:
            m = re.match(r"^\s*\d+\s+(\S+)\s+0x[0-9a-fA-F]+\s+(.*)$", line)
            if not m:
                continue
            image = m.group(1).strip()
            symbol = m.group(2).strip()
            if image and symbol:
                info["crashed_in"] = f"{image} {symbol}"
                break

    return info


def _is_blank_field(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        stripped = value.strip()
        return stripped == "" or stripped.lower() == "unknown"
    return False


def _enrich_info_from_file(crash: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, Any]:
    crash_file = crash.get("file")
    if not crash_file:
        return info

    path = Path(crash_file)
    if path.suffix.lower() == ".ips":
        parsed = _extract_info_from_ips(path)
    elif path.suffix.lower() == ".crash":
        parsed = _extract_info_from_crash_text(path)
    else:
        parsed = {}

    if not parsed:
        return info

    enriched = dict(info)
    for key in ("process", "exception", "fault_address", "crashed_in"):
        if _is_blank_field(enriched.get(key)) and not _is_blank_field(parsed.get(key)):
            enriched[key] = parsed[key]
    return enriched


def format_crash_for_ai(crash: Dict[str, Any], include_full_log: bool = False) -> str:
    """Format a crash entry for AI consumption."""
    lines = [
        "=" * 60,
        "CRASH REPORT",
        "=" * 60,
        f"Timestamp: {crash.get('timestamp', 'unknown')}",
        f"Command: {crash.get('command', 'unknown')}",
        f"Exit Code: {crash.get('exit_code', 'unknown')}",
        f"Signal: {crash.get('signal', 'unknown')}",
        "",
    ]

    info = crash.get("info", {}) or {}
    if isinstance(info, dict):
        info = _enrich_info_from_file(crash, info)
    lines.extend([
        "Crash Info:",
        f"  Process: {info.get('process', 'unknown')}",
        f"  Exception: {info.get('exception', 'unknown')}",
        f"  Fault Address: {info.get('fault_address', 'unknown')}",
        f"  Crashed In: {info.get('crashed_in', 'unknown')}",
        "",
        f"Log File: {crash.get('file', 'unknown')}",
    ])

    if include_full_log and crash.get("file"):
        lines.extend([
            "",
            "=" * 60,
            "FULL CRASH LOG",
            "=" * 60,
            get_crash_details(crash["file"])[:5000],  # Limit to 5000 chars
        ])

    return "\n".join(lines)


def check_for_new_crashes_since(timestamp: str) -> List[Dict[str, Any]]:
    """Check for crashes newer than the given ISO timestamp."""
    try:
        since = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return []

    summary = get_crash_summary()
    new_crashes = []

    for crash in summary.get("crashes", []):
        try:
            ts = datetime.fromisoformat(crash["timestamp"].replace("Z", "+00:00"))
            if ts > since:
                new_crashes.append(crash)
        except (KeyError, ValueError):
            continue

    return new_crashes


def watch_for_crashes(interval: int = 5):
    """Watch for new crashes and print them as they occur."""
    print(f"Watching for crashes (checking every {interval}s)...")
    print("Press Ctrl+C to stop")
    print()

    last_check = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    try:
        while True:
            new_crashes = check_for_new_crashes_since(last_check)
            if new_crashes:
                for crash in reversed(new_crashes):  # Oldest first
                    print(format_crash_for_ai(crash))
                    print()
            last_check = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped watching.")


def count_crash_files() -> int:
    """Count crash log files in crash_logs directory."""
    if not CRASH_DIR.exists():
        return 0
    return sum(1 for _ in CRASH_DIR.glob("*.ips")) + sum(1 for _ in CRASH_DIR.glob("*.crash"))


def sync_crash_summary() -> int:
    """Rebuild crash_summary.json from all .ips/.crash files in crash_logs."""
    if not CRASH_DIR.exists():
        return 0

    crashes = []
    crash_files = list(CRASH_DIR.glob("*.ips")) + list(CRASH_DIR.glob("*.crash"))
    for crash_file in sorted(crash_files, key=lambda p: p.stat().st_mtime, reverse=True):
        # Extract timestamp from filename or file mtime
        mtime = crash_file.stat().st_mtime
        timestamp = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z")

        # Parse crash info
        if crash_file.suffix.lower() == ".ips":
            info = _extract_info_from_ips(crash_file)
        elif crash_file.suffix.lower() == ".crash":
            info = _extract_info_from_crash_text(crash_file)
        else:
            info = {}

        crashes.append({
            "timestamp": timestamp,
            "file": str(crash_file),
            "info": info
        })

    # Write updated summary
    summary = {
        "crashes": crashes,
        "last_check": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "total_files": len(crashes)
    }

    CRASH_DIR.mkdir(exist_ok=True)
    with open(CRASH_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)

    return len(crashes)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Check for crash logs")
    parser.add_argument("--latest", action="store_true", help="Show latest crash with full details")
    parser.add_argument("--hours", type=float, default=24, help="Show crashes from last N hours")
    parser.add_argument("--watch", action="store_true", help="Watch for new crashes")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--count", action="store_true", help="Just count crash files (fast)")
    parser.add_argument("--sync", action="store_true", help="Rebuild crash_summary.json from all .ips/.crash files")
    args = parser.parse_args()

    if args.count:
        count = count_crash_files()
        print(count)
        return

    if args.sync:
        count = sync_crash_summary()
        print(f"Synced {count} crash files to crash_summary.json")
        return

    if args.watch:
        watch_for_crashes()
        return

    if args.latest:
        crash = get_latest_crash()
        if crash:
            if args.json:
                print(json.dumps(crash, indent=2))
            else:
                print(format_crash_for_ai(crash, include_full_log=True))
        else:
            print("No crashes logged.")
        return

    # Show recent crashes
    crashes = get_recent_crashes(args.hours)
    if not crashes:
        print(f"No crashes in the last {args.hours} hours.")
        return

    if args.json:
        print(json.dumps(crashes, indent=2))
    else:
        print(f"Found {len(crashes)} crash(es) in the last {args.hours} hours:")
        print()
        for crash in crashes:
            print(format_crash_for_ai(crash))
            print()


if __name__ == "__main__":
    main()
