#!/usr/bin/env python3
"""Disk cleanup report and training curve analysis."""

import re
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CHECKPOINTS_DIR = Path("/Users/ayates/model_mlx_migration/checkpoints")
OUTPUT_DIR = Path("/Users/ayates/model_mlx_migration/reports")
OUTPUT_DIR.mkdir(exist_ok=True)


def parse_training_log(log_path: Path) -> list[dict]:
    """Parse training log file and extract step/loss data."""
    metrics = []
    with open(log_path) as f:
        for line in f:
            # Match pattern: Step N: loss=X.XXXX
            match = re.search(r'Step\s+(\d+):\s+loss=([\d.]+)', line)
            if match:
                step = int(match.group(1))
                loss = float(match.group(2))
                metrics.append({'step': step, 'loss': loss})
    return metrics


def find_checkpoints(checkpoint_dir: Path) -> list[tuple[int, Path]]:
    """Find all step checkpoints and return sorted by step number."""
    checkpoints = []
    for f in checkpoint_dir.iterdir():
        if f.name.startswith('step_') and f.name.endswith('.npz'):
            match = re.match(r'step_(\d+)\.npz', f.name)
            if match:
                step = int(match.group(1))
                checkpoints.append((step, f))
    return sorted(checkpoints, key=lambda x: x[0])


def get_checkpoint_size(path: Path) -> int:
    """Get size of checkpoint file in bytes."""
    if path.is_file():
        return path.stat().st_size
    return 0


def analyze_checkpoints():
    """Analyze all checkpoint directories."""
    results = {}

    for subdir in sorted(CHECKPOINTS_DIR.iterdir()):
        if not subdir.is_dir():
            continue

        checkpoints = find_checkpoints(subdir)
        if not checkpoints:
            continue

        total_size = sum(get_checkpoint_size(p) for _, p in checkpoints)

        # Keep last 5
        to_delete = checkpoints[:-5] if len(checkpoints) > 5 else []
        delete_size = sum(get_checkpoint_size(p) for _, p in to_delete)

        # Find training log
        log_path = subdir / "training.log"
        metrics = []
        if log_path.exists():
            metrics = parse_training_log(log_path)

        results[subdir.name] = {
            'total_checkpoints': len(checkpoints),
            'total_size_gb': total_size / (1024**3),
            'to_delete': len(to_delete),
            'delete_size_gb': delete_size / (1024**3),
            'keep': [s for s, _ in checkpoints[-5:]] if checkpoints else [],
            'delete_steps': [s for s, _ in to_delete],
            'metrics': metrics,
            'checkpoint_paths_to_delete': [str(p) for _, p in to_delete],
        }

    return results


def plot_training_curves(results: dict, output_path: Path):
    """Plot training curves for all models with metrics."""
    models_with_metrics = {k: v for k, v in results.items() if v['metrics']}

    if not models_with_metrics:
        print("No training metrics found to plot")
        return

    # Create subplots
    n_models = len(models_with_metrics)
    cols = 3
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    axes = np.array(axes).flatten()

    for idx, (name, data) in enumerate(sorted(models_with_metrics.items())):
        ax = axes[idx]
        metrics = data['metrics']
        steps = [m['step'] for m in metrics]
        losses = [m['loss'] for m in metrics]

        ax.plot(steps, losses, 'b-', linewidth=0.8)
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title(name, fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add final loss annotation
        if losses:
            ax.annotate(f'Final: {losses[-1]:.3f}',
                       xy=(steps[-1], losses[-1]),
                       fontsize=8)

    # Hide unused subplots
    for idx in range(len(models_with_metrics), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved training curves to {output_path}")
    plt.close()


def generate_report(results: dict, output_path: Path):
    """Generate markdown report."""
    total_delete_gb = sum(r['delete_size_gb'] for r in results.values())

    lines = [
        "# Disk Cleanup Report",
        "",
        f"**Total space recoverable from old checkpoints: {total_delete_gb:.1f} GB**",
        "",
        "## Checkpoint Analysis (Keep Last 5)",
        "",
        "| Model | Total | Delete | Keep | Space Freed |",
        "|-------|-------|--------|------|-------------|",
    ]

    for name, data in sorted(results.items(), key=lambda x: -x[1]['delete_size_gb']):
        if data['to_delete'] > 0:
            lines.append(
                f"| {name} | {data['total_checkpoints']} | {data['to_delete']} | "
                f"{data['keep']} | {data['delete_size_gb']:.1f} GB |"
            )

    lines.extend([
        "",
        "## Benchmark Data Analysis",
        "",
        "| Directory | Size | Notes |",
        "|-----------|------|-------|",
        "| benchmarks/librispeech/downloads | 57 GB | **Downloaded archives - SAFE TO DELETE** |",
        "| benchmarks/librispeech/LibriSpeech | 60 GB | Extracted data for benchmarking |",
        "",
        "## librimix_generation Analysis",
        "",
        "- **Size**: 108 GB",
        "- **Contents**: Generated Libri2Mix speech separation data",
        "- **Dependencies**: Symlinks to LibriSpeech_full and wham_noise",
        "- **Recommendation**: Keep if using speech separation training, regenerable if deleted",
        "",
        "## Cleanup Commands",
        "",
        "```bash",
        "# Delete old checkpoints (keeps last 5 per model)",
    ])

    # Add delete commands
    for name, data in sorted(results.items()):
        for path in data['checkpoint_paths_to_delete'][:3]:  # Show first 3
            lines.append(f"rm {path}")
        if len(data['checkpoint_paths_to_delete']) > 3:
            lines.append(f"# ... and {len(data['checkpoint_paths_to_delete'])-3} more in {name}")

    lines.extend([
        "",
        "# Delete benchmark downloads (archives)",
        "rm -rf /Users/ayates/model_mlx_migration/data/benchmarks/librispeech/downloads",
        "```",
        "",
        "## Training Curves",
        "",
        "See training_curves.png for loss curves of all models.",
    ])

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Saved report to {output_path}")


def main():
    print("Analyzing checkpoints...")
    results = analyze_checkpoints()

    print(f"Found {len(results)} checkpoint directories")

    # Generate report
    report_path = OUTPUT_DIR / "disk_cleanup_report.md"
    generate_report(results, report_path)

    # Plot training curves
    curves_path = OUTPUT_DIR / "training_curves.png"
    plot_training_curves(results, curves_path)

    # Save raw data
    json_path = OUTPUT_DIR / "checkpoint_analysis.json"
    # Remove non-serializable paths
    json_results = {k: {kk: vv for kk, vv in v.items() if kk != 'checkpoint_paths_to_delete'}
                    for k, v in results.items()}
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)

    # Summary
    total_delete = sum(r['delete_size_gb'] for r in results.values())
    print("\nSummary:")
    print(f"  Total checkpoint space recoverable: {total_delete:.1f} GB")
    print("  Benchmark downloads deletable: 57 GB")
    print(f"  Estimated total recoverable: {total_delete + 57:.1f} GB")


if __name__ == "__main__":
    main()
