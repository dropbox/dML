#!/usr/bin/env python3
"""
Training Gate - QA Enforcement Before Training

This module enforces data quality checks before training can start.
Import and call `enforce_qa_gate()` at the start of training scripts.

Usage:
    from tools.data_quality.training_gate import enforce_qa_gate

    # At start of training:
    enforce_qa_gate(datasets=["ravdess", "crema_d"], strict=True)

    # Or check status only:
    from tools.data_quality.training_gate import check_qa_status
    status = check_qa_status()
    print(f"QA Status: {status}")
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


def load_dataset_config() -> dict:
    """Load dataset configuration."""
    config_path = Path(__file__).parent.parent.parent / "data" / "qa" / "datasets.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {"defaults": {}, "datasets": {}, "gates": {"required": [], "optional": []}}


def get_qa_status_path() -> Path:
    """Get path to QA status file."""
    return Path(__file__).parent.parent.parent / "data" / "qa" / "status.json"


def load_qa_status() -> dict:
    """Load QA status from file."""
    status_path = get_qa_status_path()
    if status_path.exists():
        with open(status_path) as f:
            return json.load(f)
    return {}


def save_qa_status(status: dict):
    """Save QA status to file."""
    status_path = get_qa_status_path()
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with open(status_path, 'w') as f:
        json.dump(status, f, indent=2)


def run_qa_tests(markers: list[str] = None, verbose: bool = False) -> tuple[bool, str]:
    """
    Run pytest QA tests.

    Args:
        markers: List of pytest markers to run (default: ["data_qa"])
        verbose: Show verbose output

    Returns:
        (passed, output) tuple
    """
    if markers is None:
        markers = ["data_qa"]

    marker_str = " or ".join(markers)
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/data_quality/",
        "-m", marker_str,
        "--tb=short",
        "-q" if not verbose else "-v",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )
        passed = result.returncode == 0
        output = result.stdout + result.stderr
        return passed, output
    except Exception as e:
        return False, str(e)


def check_qa_status(datasets: list[str] = None) -> dict[str, any]:
    """
    Check QA status for specified datasets.

    Args:
        datasets: List of dataset names to check (None = all)

    Returns:
        Status dict with 'passed', 'last_run', 'details'
    """
    status = load_qa_status()
    config = load_dataset_config()

    if datasets is None:
        datasets = list(config.get("datasets", {}).keys())

    result = {
        "passed": True,
        "last_run": status.get("last_run"),
        "datasets": {},
    }

    for ds_name in datasets:
        ds_status = status.get("datasets", {}).get(ds_name, {})
        result["datasets"][ds_name] = {
            "passed": ds_status.get("passed", False),
            "last_checked": ds_status.get("last_checked"),
        }
        if not ds_status.get("passed", False):
            result["passed"] = False

    return result


def enforce_qa_gate(
    datasets: list[str] = None,
    strict: bool = True,
    run_tests: bool = True,
    max_age_hours: float = 24.0,
) -> bool:
    """
    Enforce QA gate before training.

    Args:
        datasets: List of dataset names to check (None = all required)
        strict: If True, raise exception on failure
        run_tests: If True, run tests if status is stale
        max_age_hours: Maximum age of QA status before re-running tests

    Returns:
        True if all gates passed

    Raises:
        RuntimeError: If strict=True and gates fail
    """
    config = load_dataset_config()
    status = load_qa_status()

    # Check if we need to run tests
    last_run = status.get("last_run")
    needs_rerun = True

    if last_run:
        try:
            last_run_dt = datetime.fromisoformat(last_run)
            age_hours = (datetime.now() - last_run_dt).total_seconds() / 3600
            needs_rerun = age_hours > max_age_hours
        except ValueError:
            needs_rerun = True

    if run_tests and needs_rerun:
        print("Running data quality tests...")
        required_gates = config.get("gates", {}).get("required", ["data_qa"])
        passed, output = run_qa_tests(markers=required_gates, verbose=False)

        # Update status
        status["last_run"] = datetime.now().isoformat()
        status["last_result"] = "passed" if passed else "failed"
        status["last_output"] = output[:5000]  # Truncate output
        save_qa_status(status)

        if not passed:
            msg = f"Data QA tests failed:\n{output}"
            if strict:
                raise RuntimeError(msg)
            print(f"WARNING: {msg}")
            return False

        print("Data QA tests passed.")
        return True

    # Check cached status
    qa_status = check_qa_status(datasets)
    if not qa_status["passed"]:
        failed_datasets = [
            ds for ds, info in qa_status["datasets"].items()
            if not info["passed"]
        ]
        msg = f"Data QA gate failed for datasets: {failed_datasets}"
        if strict:
            raise RuntimeError(msg)
        print(f"WARNING: {msg}")
        return False

    return True


def get_dataset_paths() -> dict[str, Path]:
    """Get resolved paths for all configured datasets."""
    config = load_dataset_config()
    project_root = Path(__file__).parent.parent.parent

    paths = {}
    for name, ds_config in config.get("datasets", {}).items():
        rel_path = ds_config.get("path", "")
        if rel_path:
            paths[name] = project_root / rel_path

    return paths


def validate_dataset_paths() -> tuple[bool, list[str]]:
    """
    Validate that all configured dataset paths exist.

    Returns:
        (all_valid, list of missing paths)
    """
    paths = get_dataset_paths()
    missing = []

    for name, path in paths.items():
        if not path.exists():
            missing.append(f"{name}: {path}")

    return len(missing) == 0, missing


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Data QA Training Gate")
    parser.add_argument("--check", action="store_true", help="Check QA status only")
    parser.add_argument("--run", action="store_true", help="Run QA tests")
    parser.add_argument("--paths", action="store_true", help="Validate dataset paths")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if args.paths:
        valid, missing = validate_dataset_paths()
        if valid:
            print("All dataset paths valid.")
        else:
            print("Missing dataset paths:")
            for m in missing:
                print(f"  - {m}")
        sys.exit(0 if valid else 1)

    if args.run:
        passed, output = run_qa_tests(verbose=args.verbose)
        if args.verbose:
            print(output)
        print(f"\nQA Tests: {'PASSED' if passed else 'FAILED'}")
        sys.exit(0 if passed else 1)

    if args.check:
        status = check_qa_status()
        print(f"QA Status: {'PASSED' if status['passed'] else 'FAILED'}")
        print(f"Last run: {status['last_run']}")
        for ds, info in status['datasets'].items():
            print(f"  {ds}: {'OK' if info['passed'] else 'FAILED'}")
        sys.exit(0 if status['passed'] else 1)

    # Default: enforce gate
    try:
        enforce_qa_gate(strict=True)
        print("QA Gate: PASSED - Training can proceed.")
    except RuntimeError as e:
        print(f"QA Gate: FAILED - {e}")
        sys.exit(1)
