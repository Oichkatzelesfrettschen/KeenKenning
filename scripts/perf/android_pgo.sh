#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"

PACKAGE=""
OUT_DIR="$ROOT_DIR/build/pgo/android"
PROFILE_PREFIX="keen"
RUN_CMD=""
SERIAL=""
WAIT_SECS=0

usage() {
  cat <<EOF
Usage: $0 --package=org.yegie.keenkenning.kenning [options]
  --package=ID          Android package name to wrap for PGO
  --out-dir=PATH        Output directory for profraw/profdata
  --prefix=NAME         Profile file prefix (default: keen)
  --run="CMD"           adb subcommand to run while profiling
  --serial=SERIAL       adb device serial
  --wait=SECONDS        Sleep before collecting profiles
EOF
}
while [ "$#" -gt 0 ]; do
  case "$1" in
    --package=*) PACKAGE="${1#--package=}" ;;
    --out-dir=*) OUT_DIR="${1#--out-dir=}" ;;
    --prefix=*) PROFILE_PREFIX="${1#--prefix=}" ;;
    --run=*) RUN_CMD="${1#--run=}" ;;
    --serial=*) SERIAL="${1#--serial=}" ;;
    --wait=*) WAIT_SECS="${1#--wait=}" ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
  shift
done

if [ -z "$PACKAGE" ]; then
  echo "--package is required" >&2
  usage
  exit 1
fi

if ! command -v adb >/dev/null 2>&1; then
  echo "adb not found in PATH." >&2
  exit 1
fi

if ! command -v llvm-profdata >/dev/null 2>&1; then
  echo "llvm-profdata not found in PATH." >&2
  exit 1
fi

ADB=(adb)
if [ -n "$SERIAL" ]; then
  ADB=(adb -s "$SERIAL")
fi

WRAP_VALUE="LLVM_PROFILE_FILE=/data/local/tmp/${PROFILE_PREFIX}_%p.profraw %command%"
cleanup() {
  "${ADB[@]}" shell setprop "wrap.$PACKAGE" "" >/dev/null 2>&1 || true
}
trap cleanup EXIT

"${ADB[@]}" shell setprop "wrap.$PACKAGE" "$WRAP_VALUE"
if [ -n "$RUN_CMD" ]; then
  echo "Running: adb $RUN_CMD"
  # shellcheck disable=SC2086
  "${ADB[@]}" $RUN_CMD
else
  echo "Wrap enabled for $PACKAGE. Run your workload now."
  read -r -p "Press Enter to collect PGO profiles..." _
fi

if [ "$WAIT_SECS" -gt 0 ]; then
  sleep "$WAIT_SECS"
fi

RAW_DIR="$OUT_DIR/raw"
mkdir -p "$RAW_DIR"

PROFILE_LIST=$("${ADB[@]}" shell "ls /data/local/tmp/${PROFILE_PREFIX}_*.profraw 2>/dev/null" || true)
if [ -z "$PROFILE_LIST" ]; then
  echo "No profraw files found in /data/local/tmp. Did the app run?" >&2
  exit 1
fi

"${ADB[@]}" pull /data/local/tmp/${PROFILE_PREFIX}_*.profraw "$RAW_DIR" >/dev/null

llvm-profdata merge -o "$OUT_DIR/${PROFILE_PREFIX}.profdata" "$RAW_DIR"/*.profraw

echo "Merged profile: $OUT_DIR/${PROFILE_PREFIX}.profdata"
