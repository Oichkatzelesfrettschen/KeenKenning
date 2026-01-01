#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"

SANITIZERS=""
COVERAGE=0
TRACE=0
LTO=0
BOLT_RELOCS=0
PGO_MODE=""
PGO_PROFILE=""
TASKS=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --sanitizers=*) SANITIZERS="${1#--sanitizers=}" ;;
    --coverage) COVERAGE=1 ;;
    --trace) TRACE=1 ;;
    --lto) LTO=1 ;;
    --bolt-relocs) BOLT_RELOCS=1 ;;
    --pgo=*) PGO_MODE="${1#--pgo=}" ;;
    --pgo-profile=*) PGO_PROFILE="${1#--pgo-profile=}" ;;
    --help|-h)
      echo "Usage: $0 [--sanitizers=address,undefined|thread] [--coverage] [--trace] [--lto] [--bolt-relocs]" >&2
      echo "          [--pgo=generate|use] [--pgo-profile=/abs/path/profile.profdata] -- <gradle tasks>" >&2
      exit 0
      ;;
    --)
      shift
      TASKS+=("$@")
      break
      ;;
    *) TASKS+=("$1") ;;
  esac
  shift
done

if [ "${#TASKS[@]}" -eq 0 ]; then
  TASKS=(assembleKenningDebug)
fi

ARGS=()
if [ -n "$SANITIZERS" ]; then
  ARGS+=("-PkeenSanitizers=$SANITIZERS")
fi
if [ "$COVERAGE" -eq 1 ]; then
  ARGS+=("-PkeenCoverage")
fi
if [ "$TRACE" -eq 1 ]; then
  ARGS+=("-PkeenFuncTrace")
fi
if [ "$LTO" -eq 1 ]; then
  ARGS+=("-PkeenLto")
fi
if [ "$BOLT_RELOCS" -eq 1 ]; then
  ARGS+=("-PkeenBoltRelocs")
fi
if [ -n "$PGO_MODE" ]; then
  ARGS+=("-PkeenPgo=$PGO_MODE")
fi
if [ -n "$PGO_PROFILE" ]; then
  ARGS+=("-PkeenPgoProfile=$PGO_PROFILE")
fi

./gradlew "${ARGS[@]}" "${TASKS[@]}"
