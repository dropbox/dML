#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./tests/run_metal_bare_thread_race.sh [--compile-only]

Options:
  --compile-only  Only compile the reproduction, do not run it
EOF
}

RUN=1
while [ $# -gt 0 ]; do
  case "$1" in
    --compile-only) RUN=0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

src="${REPO_ROOT}/tests/metal_bare_thread_race.m"
if [ ! -f "${src}" ]; then
  echo "ERROR: Source not found: ${src}" >&2
  exit 1
fi

tmp_dir="$(mktemp -d "${REPO_ROOT}/.metal_bare_thread_race_tmp.XXXXXX")"
trap 'rm -rf "${tmp_dir}"' EXIT

bin="${tmp_dir}/metal_bare_thread_race"

echo "Compiling: ${src}"
clang -fobjc-arc -framework Foundation -framework Metal "${src}" -o "${bin}"

if [ "${RUN}" -eq 0 ]; then
  echo "Compile OK."
  exit 0
fi

echo "Running: ${bin}"
echo "NOTE: This may crash if the Metal/AGX driver bug reproduces."
exec "${bin}"
