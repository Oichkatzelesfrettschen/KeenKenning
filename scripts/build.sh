#!/bin/bash
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk
export PATH=$JAVA_HOME/bin:$PATH

echo "Using Java: $(java -version 2>&1 | head -n 1)"
./gradlew assembleDebug --warning-mode fail "$@"
