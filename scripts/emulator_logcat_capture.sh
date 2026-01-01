#!/usr/bin/env bash
set -euo pipefail

if [ -n "${LD_PRELOAD:-}" ]; then
  unset LD_PRELOAD
fi

OUT_FILE="${1:-}"
SERIAL="${2:-${ANDROID_SERIAL:-}}"
LINES="${LOGCAT_LINES:-600}"
FILTER_REGEX="${LOGCAT_FILTER_REGEX:-org.yegie.keenkenning|kenning\\.kenning|TestRunner|Instrumentation|AndroidRuntime|FATAL|ddmlib}"
SINCE_FILE="${LOGCAT_SINCE_FILE:-}"
SINCE="${LOGCAT_SINCE:-}"
TIMEOUT="${LOGCAT_TIMEOUT:-10}"

if [ -z "$OUT_FILE" ]; then
  echo "Usage: $0 <out-file> [serial]" >&2
  exit 2
fi

mkdir -p "$(dirname "$OUT_FILE")"

adb_cmd=(adb)
if [ -n "$SERIAL" ]; then
  adb_cmd+=( -s "$SERIAL" )
fi

timeout_cmd=()
if command -v timeout >/dev/null 2>&1; then
  timeout_cmd=(timeout "$TIMEOUT")
fi

tmp="$(mktemp)"
if [ -z "$SINCE" ] && [ -n "$SINCE_FILE" ] && [ -f "$SINCE_FILE" ]; then
  SINCE="$(cat "$SINCE_FILE" 2>/dev/null || true)"
fi

if [ -n "$SINCE" ]; then
  if ! "${timeout_cmd[@]}" "${adb_cmd[@]}" logcat -d -b all -v time -T "$SINCE" > "$tmp" 2>/dev/null; then
    echo "logcat capture failed (device offline?)" > "$OUT_FILE"
    rm -f "$tmp"
    exit 0
  fi
else
  if ! "${timeout_cmd[@]}" "${adb_cmd[@]}" logcat -d -b all -v time > "$tmp" 2>/dev/null; then
    echo "logcat capture failed (device offline?)" > "$OUT_FILE"
    rm -f "$tmp"
    exit 0
  fi
fi

filtered="$(mktemp)"
if command -v rg >/dev/null 2>&1; then
  rg -n -e "$FILTER_REGEX" "$tmp" | tail -n "$LINES" > "$filtered" || true
else
  grep -E -n "$FILTER_REGEX" "$tmp" | tail -n "$LINES" > "$filtered" || true
fi

if [ -s "$filtered" ]; then
  mv "$filtered" "$OUT_FILE"
else
  tail -n "$LINES" "$tmp" > "$OUT_FILE"
  rm -f "$filtered"
fi

rm -f "$tmp"
echo "wrote $OUT_FILE"
