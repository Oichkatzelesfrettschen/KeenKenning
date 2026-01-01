#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"

BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build/pgo}"
OUT_BIN="${OUT_BIN:-$BUILD_DIR/latin_gen_opt}"
PROFILE_DIR="${PROFILE_DIR:-$BUILD_DIR/pgo}"
PROFILE_DATA="${PROFILE_DATA:-$PROFILE_DIR/keen.profdata}"
ORDER="${ORDER:-9}"
RUNS="${RUNS:-25}"

if ! command -v llvm-profdata >/dev/null 2>&1; then
  echo "llvm-profdata not found in PATH." >&2
  exit 1
fi

mkdir -p "$PROFILE_DIR"
"$SCRIPT_DIR/host_build.sh" \
  --mode=perf \
  --cc=clang \
  --build-dir="$BUILD_DIR" \
  --out="$OUT_BIN" \
  --pgo=generate

LLVM_PROFILE_FILE="$PROFILE_DIR/keen_%p.profraw" bash -c \
  "for i in \$(seq 1 $RUNS); do \"$OUT_BIN\" \"$ORDER\" >/dev/null; done"

llvm-profdata merge -o "$PROFILE_DATA" "$PROFILE_DIR"/*.profraw
"$SCRIPT_DIR/host_build.sh" \
  --mode=release \
  --cc=clang \
  --build-dir="$BUILD_DIR" \
  --out="$OUT_BIN.pgo" \
  --pgo=use \
  --pgo-profile="$PROFILE_DATA"

echo "PGO-optimized binary: $OUT_BIN.pgo"
