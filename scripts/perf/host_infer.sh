#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"

RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/build/infer}"
MAKE_TARGET="${MAKE_TARGET:-tools}"
# infer currently bundles clang-11, which does not accept -std=c23/c++23.
C_STD="${C_STD:-c2x}"
CXX_STD="${CXX_STD:-c++20}"
CC="${CC:-clang}"
CFLAGS_INFER="${CFLAGS_INFER:--O3 -std=$C_STD -march=native -fno-lto -ffast-math -funroll-loops -fvectorize -fslp-vectorize -fno-math-errno -I$ROOT_DIR/app/src/main/jni -DSTANDALONE_LATIN_TEST -Wall -Wextra -Werror}"

if ! command -v infer >/dev/null 2>&1; then
  echo "infer not found in PATH. See ~/Documents/Code-Analysis-Tooling/README.md" >&2
  exit 1
fi

mkdir -p "$RESULTS_DIR"
infer run --results-dir "$RESULTS_DIR" -- \
  make "$MAKE_TARGET" CC="$CC" C_STD="$C_STD" CXX_STD="$CXX_STD" CFLAGS="$CFLAGS_INFER" LDFLAGS="" PGO_MODE=off BOLT_RELOCS=0

echo "Infer results in $RESULTS_DIR"
