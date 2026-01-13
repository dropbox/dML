#!/usr/bin/env python3
"""
Summarize Instruments "Metal System Trace" output into JSON.

This is a lightweight, non-GUI way to extract a few high-signal metrics from a
.trace bundle, primarily:
- Metal command buffer submission counts
- Encoder-time distributions
- Which thread performed submissions (evidence of serialization)

It uses xctrace's XML export and converts key fields into a compact JSON file.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import subprocess


DEFAULT_XCTRACE = Path("/Applications/Xcode.app/Contents/Developer/usr/bin/xctrace")
REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _xctrace_export_toc(xctrace: Path, trace: Path, out_xml: Path) -> None:
    _run([str(xctrace), "export", "--input", str(trace), "--toc", "--output", str(out_xml)])


def _xctrace_export_submissions(xctrace: Path, trace: Path, out_xml: Path) -> None:
    xpath = '/trace-toc/run[@number="1"]/data/table[@schema="metal-application-command-buffer-submissions"]'
    _run([str(xctrace), "export", "--input", str(trace), "--xpath", xpath, "--output", str(out_xml)])


def _load_id_map(root: ET.Element) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for el in root.iter():
        el_id = el.attrib.get("id")
        if not el_id:
            continue
        out[el_id] = {
            "tag": el.tag,
            "fmt": el.attrib.get("fmt"),
            "text": (el.text.strip() if el.text else None),
        }
    return out


def _resolve(el: ET.Element, id_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ref = el.attrib.get("ref")
    if ref:
        return id_map.get(ref, {"tag": el.tag, "fmt": None, "text": None})
    el_id = el.attrib.get("id")
    if el_id and el_id in id_map:
        return id_map[el_id]
    return {"tag": el.tag, "fmt": el.attrib.get("fmt"), "text": (el.text.strip() if el.text else None)}


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    pct = min(max(pct, 0.0), 100.0)
    idx = int((pct / 100.0) * (len(values) - 1))
    return values[idx]


def _parse_trace_duration_s(toc_root: ET.Element) -> float:
    # <trace-toc><run><info><summary><duration>3.349240</duration></summary></info></run></trace-toc>
    node = toc_root.find(".//summary/duration")
    if node is None or not node.text:
        return 0.0
    try:
        return float(node.text.strip())
    except ValueError:
        return 0.0


def _parse_trace_command(toc_root: ET.Element) -> dict[str, Any]:
    proc = toc_root.find(".//info/target/process")
    if proc is None:
        return {}
    return {
        "name": proc.attrib.get("name"),
        "pid": proc.attrib.get("pid"),
        "arguments": proc.attrib.get("arguments"),
        "path": proc.attrib.get("path"),
        "exit_status": proc.attrib.get("return-exit-status"),
    }


def _iter_rows(root: ET.Element) -> Iterable[ET.Element]:
    for row in root.iter("row"):
        yield row


def _summarize_submissions(xml_root: ET.Element) -> dict[str, Any]:
    id_map = _load_id_map(xml_root)

    durations_ns: list[int] = []
    encoder_times_ns: list[int] = []
    thread_fmts: list[str] = []

    for row in _iter_rows(xml_root):
        direct = list(row)
        direct_durations = [c for c in direct if c.tag == "duration"]
        if len(direct_durations) < 2:
            continue

        duration_text = _resolve(direct_durations[0], id_map).get("text")
        encoder_text = _resolve(direct_durations[1], id_map).get("text")
        if duration_text is None or encoder_text is None:
            continue

        try:
            durations_ns.append(int(duration_text))
            encoder_times_ns.append(int(encoder_text))
        except ValueError:
            continue

        thread_el = next((c for c in direct if c.tag == "thread"), None)
        if thread_el is not None:
            thread_fmt = _resolve(thread_el, id_map).get("fmt") or "UNKNOWN"
            thread_fmts.append(thread_fmt)

    thread_counts = Counter(thread_fmts)
    top_thread, top_count = ("", 0)
    if thread_counts:
        top_thread, top_count = thread_counts.most_common(1)[0]

    to_ms = lambda ns: float(ns) / 1_000_000.0
    enc_ms = [to_ms(ns) for ns in encoder_times_ns]

    return {
        "count": len(encoder_times_ns),
        "duration_ms_sum": to_ms(sum(durations_ns)),
        "encoder_time_ms_sum": to_ms(sum(encoder_times_ns)),
        "encoder_time_ms_p50": _percentile(enc_ms, 50),
        "encoder_time_ms_p95": _percentile(enc_ms, 95),
        "encoder_time_ms_mean": (sum(enc_ms) / len(enc_ms)) if enc_ms else 0.0,
        "unique_threads": len(thread_counts),
        "top_thread": top_thread,
        "top_thread_fraction": (float(top_count) / float(len(thread_fmts))) if thread_fmts else 0.0,
        "thread_counts": dict(thread_counts),
    }


def summarize_trace(xctrace: Path, trace: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        toc_xml = td_path / "toc.xml"
        subs_xml = td_path / "submissions.xml"

        _xctrace_export_toc(xctrace, trace, toc_xml)
        _xctrace_export_submissions(xctrace, trace, subs_xml)

        toc_root = ET.parse(toc_xml).getroot()
        subs_root = ET.parse(subs_xml).getroot()

        duration_s = _parse_trace_duration_s(toc_root)
        command = _parse_trace_command(toc_root)
        submissions = _summarize_submissions(subs_root)

        cb_per_s = (submissions["count"] / duration_s) if duration_s > 0 else 0.0
        submissions["submissions_per_second"] = cb_per_s
        return {
            "path": os.path.relpath(trace, REPO_ROOT),
            "duration_s": duration_s,
            "command": command,
            "command_buffer_submissions": submissions,
        }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Summarize Metal System Trace .trace bundles")
    parser.add_argument(
        "--xctrace",
        type=Path,
        default=DEFAULT_XCTRACE,
        help="path to xctrace (default: Xcode.app bundled tool)",
    )
    parser.add_argument("--trace", type=Path, action="append", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "reports" / "main" / f"metal_system_trace_summary_{_dt.date.today().isoformat()}.json",
    )
    args = parser.parse_args(argv)

    traces = [t.resolve() for t in args.trace]
    for t in traces:
        if not t.exists():
            raise SystemExit(f"trace not found: {t}")
    if not args.xctrace.exists():
        raise SystemExit(f"xctrace not found: {args.xctrace}")

    payload = {
        "generated_at": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "xctrace": str(args.xctrace),
        "summaries": [summarize_trace(args.xctrace, t) for t in traces],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote: {os.path.relpath(args.output, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
