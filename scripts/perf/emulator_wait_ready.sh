#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"

SERIAL="${1:-}"
ADB=(adb)
if [ -n "$SERIAL" ]; then
  ADB=(adb -s "$SERIAL")
fi

"${ADB[@]}" wait-for-device

unlock_device() {
  "${ADB[@]}" shell wm dismiss-keyguard >/dev/null 2>&1 || true
  "${ADB[@]}" shell input keyevent 82 >/dev/null 2>&1 || true
}

for _ in {1..120}; do
  boot=$(${ADB[@]} shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')
  if [ "$boot" = "1" ]; then
    unlock_device
    if ${ADB[@]} shell cmd activity get-current-user >/dev/null 2>&1 \
      && ${ADB[@]} shell cmd package list packages >/dev/null 2>&1 \
      && ${ADB[@]} shell ls /sdcard/Android >/dev/null 2>&1; then
      echo "emulator_ready"
      exit 0
    fi
  fi
  sleep 2
  done

echo "emulator_not_ready" >&2
exit 1
