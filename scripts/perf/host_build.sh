#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"

MODE="release"
SANITIZERS=""
COVERAGE=0
CC="${CC:-clang}"
PGO_MODE="generate"
PGO_PROFILE=""
EMIT_RELOCS=1
LINKER=""

BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build/host-tools}"
OUT_BIN="${OUT_BIN:-$BUILD_DIR/latin_gen_opt}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --mode=*) MODE="${1#--mode=}" ;;
    --asan) SANITIZERS="address" ;;
    --ubsan) SANITIZERS="undefined" ;;
    --tsan) SANITIZERS="thread" ;;
    --sanitizers=*) SANITIZERS="${1#--sanitizers=}" ;;
    --coverage) COVERAGE=1 ;;
    --cc=*) CC="${1#--cc=}" ;;
    --pgo=*) PGO_MODE="${1#--pgo=}" ;;
    --pgo-profile=*) PGO_PROFILE="${1#--pgo-profile=}" ;;
    --emit-relocs) EMIT_RELOCS=1 ;;
    --linker=*) LINKER="${1#--linker=}" ;;
    --build-dir=*) BUILD_DIR="${1#--build-dir=}" ;;
    --out=*) OUT_BIN="${1#--out=}" ;;
    --help|-h)
      cat <<EOF
Usage: $0 [--mode=perf|debug|release] [--asan|--ubsan|--tsan|--sanitizers=list] [--coverage]
          [--cc=gcc|clang] [--pgo=generate|use] [--pgo-profile=path] [--emit-relocs]
          [--linker=lld|mold|bfd] [--build-dir=path] [--out=path]
EOF
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
  shift
done

if [[ "$CC" != *clang* ]]; then
  echo "clang is required for the mandatory perf toolchain." >&2
  exit 1
fi

mkdir -p "$BUILD_DIR"

CFLAGS_BASE="-std=c23 -I$ROOT_DIR/app/src/main/jni -DSTANDALONE_LATIN_TEST -Wall -Wextra -Werror -fvectorize -fslp-vectorize -fno-math-errno"
LDFLAGS=""

case "$MODE" in
  debug)
    OPTFLAGS="-O3 -g3 -fno-omit-frame-pointer"
    ;;
  release)
    OPTFLAGS="-O3 -march=native -flto -ffast-math -funroll-loops"
    ;;
  perf)
    OPTFLAGS="-O3 -g -flto -fno-omit-frame-pointer"
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    exit 1
    ;;
esac

SAN_FLAGS=""
if [ -n "$SANITIZERS" ]; then
  SAN_FLAGS="-fsanitize=$SANITIZERS -fno-omit-frame-pointer"
  LDFLAGS+=" $SAN_FLAGS"
fi

PGO_FLAGS=""
if [ -n "$PGO_MODE" ]; then
  if [[ "$CC" != *clang* ]]; then
    echo "PGO requires clang. Use --cc=clang for PGO builds." >&2
    exit 1
  fi
  case "$PGO_MODE" in
    generate)
      PGO_FLAGS="-fprofile-instr-generate"
      LDFLAGS+=" -fprofile-instr-generate"
      ;;
    use)
      if [ -z "$PGO_PROFILE" ]; then
        echo "PGO profile required. Pass --pgo-profile=path/to/profile.profdata" >&2
        exit 1
      fi
      PGO_FLAGS="-fprofile-instr-use=$PGO_PROFILE"
      LDFLAGS+=" -fprofile-instr-use=$PGO_PROFILE"
      ;;
    *)
      echo "Unknown PGO mode: $PGO_MODE (use generate|use)" >&2
      exit 1
      ;;
  esac
fi

if [ "$EMIT_RELOCS" -eq 1 ]; then
  LDFLAGS+=" -Wl,--emit-relocs"
fi

if [ -n "$LINKER" ]; then
  LDFLAGS+=" -fuse-ld=$LINKER"
fi

if [ "$COVERAGE" -eq 1 ]; then
  OPTFLAGS="-O3 -g --coverage"
  LDFLAGS+=" --coverage"
fi

CFLAGS="$OPTFLAGS $CFLAGS_BASE $SAN_FLAGS $PGO_FLAGS"

SOURCES=(
  "$ROOT_DIR/app/src/main/jni/latin.c"
  "$ROOT_DIR/app/src/main/jni/random.c"
  "$ROOT_DIR/app/src/main/jni/malloc.c"
  "$ROOT_DIR/app/src/main/jni/maxflow_optimized.c"
  "$ROOT_DIR/app/src/main/jni/tree234.c"
  "$ROOT_DIR/scripts/ai/host_compat.c"
)

"$CC" $CFLAGS "${SOURCES[@]}" -o "$OUT_BIN" $LDFLAGS
echo "Built host tool: $OUT_BIN"
