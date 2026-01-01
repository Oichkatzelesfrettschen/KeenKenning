#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"

BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build/host-tools}"
PERF_DATA="${PERF_DATA:-$BUILD_DIR/perf.data}"
OUT_SVG="${OUT_SVG:-$BUILD_DIR/flamegraph.svg}"

STACKCOLLAPSE="${STACKCOLLAPSE:-}"
FLAMEGRAPH="${FLAMEGRAPH:-}"

if [ -z "$STACKCOLLAPSE" ] || [ -z "$FLAMEGRAPH" ]; then
  if [ -n "${FLAMEGRAPH_DIR:-}" ]; then
    STACKCOLLAPSE="$FLAMEGRAPH_DIR/stackcollapse-perf.pl"
    FLAMEGRAPH="$FLAMEGRAPH_DIR/flamegraph.pl"
  else
    STACKCOLLAPSE="$(command -v stackcollapse-perf.pl || true)"
    FLAMEGRAPH="$(command -v flamegraph.pl || true)"
  fi
fi

if ! command -v perf >/dev/null 2>&1; then
  echo "perf not found in PATH." >&2
  exit 1
fi

if [ ! -f "$PERF_DATA" ]; then
  "$SCRIPT_DIR/host_perf_record.sh"
fi

if [ -z "$STACKCOLLAPSE" ] || [ -z "$FLAMEGRAPH" ] || [ ! -x "$STACKCOLLAPSE" ] || [ ! -x "$FLAMEGRAPH" ]; then
  echo "FlameGraph scripts not found. Set FLAMEGRAPH_DIR or add them to PATH." >&2
  exit 1
fi

perf script -i "$PERF_DATA" | "$STACKCOLLAPSE" | "$FLAMEGRAPH" > "$OUT_SVG"
echo "Flamegraph written to $OUT_SVG"
