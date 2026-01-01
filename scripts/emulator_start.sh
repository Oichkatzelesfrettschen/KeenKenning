#!/usr/bin/env bash
set -euo pipefail

if [ -n "${LD_PRELOAD:-}" ]; then
  unset LD_PRELOAD
fi

AVD="${1:-KeenKeenTester}"
shift || true

HEADLESS=0
WIPE=0
GPU="swiftshader_indirect"
ACCEL="on"
SNAPSHOT_LOAD=0

for arg in "$@"; do
  case "$arg" in
    --headless) HEADLESS=1 ;;
    --wipe-data) WIPE=1 ;;
    --gpu=*) GPU="${arg#--gpu=}" ;;
    --accel=*) ACCEL="${arg#--accel=}" ;;
    --snapshot-load) SNAPSHOT_LOAD=1 ;;
    *) echo "Unknown option: $arg" >&2; exit 1 ;;
  esac
done

EMULATOR_BIN=""
if [ -x /opt/android-sdk/emulator/emulator ]; then
  EMULATOR_BIN="/opt/android-sdk/emulator/emulator"
elif command -v emulator >/dev/null 2>&1; then
  EMULATOR_BIN="$(command -v emulator)"
else
  echo "emulator binary not found" >&2
  exit 1
fi

args=(
  -avd "$AVD"
  -no-snapshot-save
  -no-audio
  -gpu "$GPU"
  -no-boot-anim
  -accel "$ACCEL"
)

if [ "$SNAPSHOT_LOAD" -eq 0 ]; then
  args+=( -no-snapshot-load )
fi

if [ "$HEADLESS" -eq 1 ]; then
  args+=( -no-window )
fi

if [ "$WIPE" -eq 1 ]; then
  args+=( -wipe-data )
fi

echo "Starting emulator: $EMULATOR_BIN ${args[*]}" >&2
exec "$EMULATOR_BIN" "${args[@]}"
