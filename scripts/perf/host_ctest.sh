#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build/host-ctest}"

cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" -DCMAKE_C_COMPILER=clang
cmake --build "$BUILD_DIR"
ctest --test-dir "$BUILD_DIR" --output-on-failure
