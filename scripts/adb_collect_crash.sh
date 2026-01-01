#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-/tmp/keen_crash_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_DIR"

adb logcat -d > "$OUT_DIR/logcat.txt" || true
adb shell dumpsys activity crashes > "$OUT_DIR/crashes.txt" || true

if adb shell ls /data/tombstones >/dev/null 2>&1; then
  adb pull /data/tombstones "$OUT_DIR/tombstones" || true
fi

echo "Crash artifacts saved to $OUT_DIR"
