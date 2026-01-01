#!/usr/bin/env bash
set -euo pipefail

if [ -n "${LD_PRELOAD:-}" ]; then
  unset LD_PRELOAD
fi

MODE="${1:-start}"
SERIAL="${2:-${ANDROID_SERIAL:-}}"
INTERVAL="${3:-${EMULATOR_KEEPALIVE_INTERVAL:-5}}"
PID_FILE="${EMULATOR_KEEPALIVE_PID:-/tmp/keen_emulator_keepalive.pid}"

adb_cmd=(adb)
if [ -n "$SERIAL" ]; then
  adb_cmd+=( -s "$SERIAL" )
fi

case "$MODE" in
  start)
    if [ -f "$PID_FILE" ]; then
      existing_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
      if [ -n "$existing_pid" ] && kill -0 "$existing_pid" 2>/dev/null; then
        echo "keepalive already running (pid $existing_pid)"
        exit 0
      fi
    fi
    if command -v setsid >/dev/null 2>&1; then
      setsid "$0" run "$SERIAL" "$INTERVAL" >/dev/null 2>&1 &
    else
      "$0" run "$SERIAL" "$INTERVAL" >/dev/null 2>&1 &
    fi
    echo $! > "$PID_FILE"
    echo "keepalive started (pid $!)"
    ;;
  run)
    while true; do
      "${adb_cmd[@]}" get-state >/dev/null 2>&1 || true
      "${adb_cmd[@]}" shell getprop sys.boot_completed >/dev/null 2>&1 || true
      sleep "$INTERVAL"
    done
    ;;
  stop)
    if [ -f "$PID_FILE" ]; then
      pid="$(cat "$PID_FILE" 2>/dev/null || true)"
      if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        kill "$pid" >/dev/null 2>&1 || true
      fi
      rm -f "$PID_FILE"
      echo "keepalive stopped"
    else
      echo "keepalive not running"
    fi
    ;;
  status)
    if [ -f "$PID_FILE" ]; then
      pid="$(cat "$PID_FILE" 2>/dev/null || true)"
      if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        echo "keepalive running (pid $pid)"
        exit 0
      fi
    fi
    echo "keepalive stopped"
    exit 1
    ;;
  *)
    echo "Usage: $0 start|stop|status [serial]" >&2
    exit 2
    ;;
esac
