#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"

BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build/bolt}"
OUT_BIN="${OUT_BIN:-$BUILD_DIR/latin_gen_opt}"
OUT_BOLT="${OUT_BOLT:-$BUILD_DIR/latin_gen_opt.bolt}"
PERF_DATA="${PERF_DATA:-$BUILD_DIR/perf.data}"
PERF_FDATA="${PERF_FDATA:-$BUILD_DIR/perf.fdata}"
ORDER="${ORDER:-9}"
RUNS="${RUNS:-25}"

for bin in llvm-bolt perf2bolt perf; do
  if ! command -v "$bin" >/dev/null 2>&1; then
    echo "$bin not found in PATH." >&2
    exit 1
  fi
done

mkdir -p "$BUILD_DIR"
"$SCRIPT_DIR/host_build.sh" \
  --mode=release \
  --cc=clang \
  --build-dir="$BUILD_DIR" \
  --out="$OUT_BIN" \
  --emit-relocs \
  --linker=lld

if [ ! -f "$PERF_DATA" ]; then
  perf record -F 99 -g --call-graph dwarf -o "$PERF_DATA" -- bash -c \
    "for i in \$(seq 1 $RUNS); do \"$OUT_BIN\" \"$ORDER\" >/dev/null; done"
fi

perf2bolt -p "$PERF_DATA" -o "$PERF_FDATA" "$OUT_BIN"
llvm-bolt "$OUT_BIN" -o "$OUT_BOLT" -data "$PERF_FDATA" \
  -reorder-blocks=ext-tsp -reorder-functions=hfsort -split-functions -split-all-cold -dyno-stats

echo "BOLT-optimized binary: $OUT_BOLT"
