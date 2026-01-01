#!/usr/bin/env bash
set -euo pipefail

if [ -n "${LD_PRELOAD:-}" ]; then
  unset LD_PRELOAD
fi

if [ -z "${JAVA_HOME:-}" ] && [ -d /usr/lib/jvm/java-21-openjdk ]; then
  export JAVA_HOME=/usr/lib/jvm/java-21-openjdk
  export PATH="$JAVA_HOME/bin:$PATH"
fi
