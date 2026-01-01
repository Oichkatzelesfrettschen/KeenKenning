#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"

BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build/valgrind}"
OUT_BIN="${OUT_BIN:-$BUILD_DIR/latin_gen_opt}"
ORDER="${ORDER:-5}"

if ! command -v valgrind >/dev/null 2>&1; then
  echo "valgrind not found in PATH." >&2
  exit 1
fi

"$SCRIPT_DIR/host_build.sh" --mode=debug --build-dir="$BUILD_DIR" --out="$OUT_BIN"

valgrind --leak-check=full --show-leak-kinds=all --error-exitcode=1 \
  "$OUT_BIN" "$ORDER" >/dev/null
