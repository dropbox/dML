#!/usr/bin/env python3
"""Delete old checkpoints, keeping only the last 5 per model."""

import re
from pathlib import Path

CHECKPOINTS_DIR = Path("/Users/ayates/model_mlx_migration/checkpoints")
DRY_RUN = False  # Set to True to preview without deleting

def find_step_checkpoints(checkpoint_dir: Path) -> list[tuple[int, Path]]:
    """Find all step_*.npz checkpoints and return sorted by step number."""
    checkpoints = []
    for f in checkpoint_dir.iterdir():
        if f.name.startswith('step_') and f.name.endswith('.npz'):
            match = re.match(r'step_(\d+)\.npz', f.name)
            if match:
                step = int(match.group(1))
                checkpoints.append((step, f))
    return sorted(checkpoints, key=lambda x: x[0])


def main():
    total_deleted = 0
    total_bytes = 0

    print(f"Scanning {CHECKPOINTS_DIR}")
    print(f"Mode: {'DRY RUN' if DRY_RUN else 'DELETING'}")
    print("-" * 60)

    for subdir in sorted(CHECKPOINTS_DIR.iterdir()):
        if not subdir.is_dir():
            continue

        checkpoints = find_step_checkpoints(subdir)
        if len(checkpoints) <= 5:
            continue

        # Keep last 5
        to_delete = checkpoints[:-5]
        to_keep = checkpoints[-5:]

        print(f"\n{subdir.name}:")
        print(f"  Total: {len(checkpoints)}, Deleting: {len(to_delete)}, Keeping: {[s for s,_ in to_keep]}")

        for step, path in to_delete:
            size = path.stat().st_size
            total_bytes += size
            total_deleted += 1

            if DRY_RUN:
                print(f"  [DRY RUN] Would delete: {path.name} ({size/1e6:.1f} MB)")
            else:
                path.unlink()

        if not DRY_RUN and to_delete:
            print(f"  Deleted {len(to_delete)} checkpoints")

    print("\n" + "=" * 60)
    print(f"Total deleted: {total_deleted} files")
    print(f"Space freed: {total_bytes / (1024**3):.1f} GB")


if __name__ == "__main__":
    main()
