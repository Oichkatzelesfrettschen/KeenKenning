#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"

BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build/coverage/host}"
OUT_BIN="${OUT_BIN:-$BUILD_DIR/latin_gen_opt}"
ORDER="${ORDER:-5}"
GCOV_EXECUTABLE="${GCOV_EXECUTABLE:-}"

if ! command -v gcovr >/dev/null 2>&1; then
  echo "gcovr not found in PATH." >&2
  exit 1
fi

"$SCRIPT_DIR/host_build.sh" --mode=debug --coverage --build-dir="$BUILD_DIR" --out="$OUT_BIN"

"$OUT_BIN" "$ORDER" >/dev/null

mkdir -p "$BUILD_DIR"
if [ -z "$GCOV_EXECUTABLE" ] && command -v llvm-cov >/dev/null 2>&1; then
  GCOV_EXECUTABLE="llvm-cov gcov"
fi

gcov_flags=()
if [ -n "$GCOV_EXECUTABLE" ]; then
  gcov_flags+=(--gcov-executable "$GCOV_EXECUTABLE")
fi

if ! gcovr "${gcov_flags[@]}" -r "$ROOT_DIR" \
  --filter "$ROOT_DIR/app/src/main/jni" \
  --filter "$ROOT_DIR/scripts/ai" \
  --html-details -o "$BUILD_DIR/index.html" \
  --xml "$BUILD_DIR/coverage.xml"; then
  echo "gcovr html-details failed; retrying summary HTML without syntax highlighting." >&2
  gcovr "${gcov_flags[@]}" -r "$ROOT_DIR" \
    --filter "$ROOT_DIR/app/src/main/jni" \
    --filter "$ROOT_DIR/scripts/ai" \
    --html -o "$BUILD_DIR/index.html" \
    --xml "$BUILD_DIR/coverage.xml"
fi

echo "gcovr report written to $BUILD_DIR/index.html"
