#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"

BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build/host-tools}"
OUT_BIN="${OUT_BIN:-$BUILD_DIR/latin_gen_opt}"
ORDER="${ORDER:-9}"
RUNS="${RUNS:-25}"
PERF_DATA="${PERF_DATA:-$BUILD_DIR/perf.data}"

if [ ! -x "$OUT_BIN" ]; then
  "$SCRIPT_DIR/host_build.sh" --mode=perf
fi

if ! command -v perf >/dev/null 2>&1; then
  echo "perf not found in PATH." >&2
  exit 1
fi

perf record -F 99 -g --call-graph dwarf -o "$PERF_DATA" -- bash -c \
  "for i in \$(seq 1 $RUNS); do \"$OUT_BIN\" \"$ORDER\" >/dev/null; done"

echo "perf data captured at $PERF_DATA"
