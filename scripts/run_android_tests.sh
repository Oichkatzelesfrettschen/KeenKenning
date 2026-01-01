#!/usr/bin/env bash
set -euo pipefail

if [ -n "${LD_PRELOAD:-}" ]; then
  unset LD_PRELOAD
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MODE="ui"
GRADLE_TASK="connectedKenningDebugAndroidTest"
SERIAL="${ANDROID_SERIAL:-}"
KEEPALIVE=1
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/android-tests}"
LOGCAT_SINCE_TS=""
EXTRA_PROPS=()

usage() {
  cat <<EOF
Usage: $0 [--ui|--bench] [--task=gradleTask] [--serial=serial] [--log-dir=path] [--no-keepalive]
  --ui            Run UI/instrumented tests (default).
  --bench         Run benchmarks (uses AndroidBenchmarkRunner + keenBenchmark).
  --task=TASK     Gradle task to run (default: connectedKenningDebugAndroidTest).
  --serial=SERIAL Target device/emulator serial.
  --log-dir=DIR   Directory for logcat capture (default: logs/android-tests).
  --no-keepalive  Disable emulator keepalive.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --ui)
      MODE="ui"
      EXTRA_PROPS+=("-Pandroid.testInstrumentationRunner=androidx.test.runner.AndroidJUnitRunner")
      ;;
    --bench)
      MODE="bench"
      EXTRA_PROPS+=("-PkeenBenchmark" "-Pandroid.testInstrumentationRunner=androidx.benchmark.junit4.AndroidBenchmarkRunner")
      ;;
    --task=*)
      GRADLE_TASK="${1#--task=}"
      ;;
    --serial=*)
      SERIAL="${1#--serial=}"
      ;;
    --log-dir=*)
      LOG_DIR="${1#--log-dir=}"
      ;;
    --no-keepalive)
      KEEPALIVE=0
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 2
      ;;
  esac
  shift
done

adb_cmd=(adb)
if [ -n "$SERIAL" ]; then
  adb_cmd+=(-s "$SERIAL")
fi

wait_for_boot() {
  "${adb_cmd[@]}" wait-for-device >/dev/null 2>&1 || true
  for _ in $(seq 1 90); do
    if "${adb_cmd[@]}" shell getprop sys.boot_completed 2>/dev/null | tr -d '\r' | grep -q "^1$"; then
      return 0
    fi
    sleep 2
  done
  echo "Device did not finish boot" >&2
  return 1
}

capture_logcat() {
  mkdir -p "$LOG_DIR"
  local stamp
  stamp="$(date +%Y%m%d_%H%M%S)"
  local out_file="$LOG_DIR/logcat-${MODE}-${stamp}.log"

  if [ -x "$SCRIPT_DIR/emulator_logcat_capture.sh" ]; then
    LOGCAT_SINCE="$LOGCAT_SINCE_TS" "$SCRIPT_DIR/emulator_logcat_capture.sh" "$out_file" "$SERIAL" || true
  fi

  local report_dir="$ROOT_DIR/app/build/reports/androidTests/connected"
  if [ -d "$report_dir" ]; then
    mkdir -p "$report_dir/logs"
    cp "$out_file" "$report_dir/logs/" || true
  fi
}

cleanup() {
  local status=$?
  if [ "$KEEPALIVE" -eq 1 ]; then
    "$SCRIPT_DIR/emulator_keepalive.sh" stop >/dev/null 2>&1 || true
  fi
  capture_logcat
  exit "$status"
}

trap cleanup EXIT

if [ "$KEEPALIVE" -eq 1 ]; then
  "$SCRIPT_DIR/emulator_keepalive.sh" start "$SERIAL" >/dev/null 2>&1 || true
fi

wait_for_boot
LOGCAT_SINCE_TS="$("${adb_cmd[@]}" shell date "+%m-%d %H:%M:%S.000" 2>/dev/null | tr -d '\r' || true)"
if [ -z "$LOGCAT_SINCE_TS" ]; then
  LOGCAT_SINCE_TS="$(date "+%m-%d %H:%M:%S.000")"
fi

"${adb_cmd[@]}" shell input keyevent 82 >/dev/null 2>&1 || true

cd "$ROOT_DIR"
./gradlew "$GRADLE_TASK" "${EXTRA_PROPS[@]}"
