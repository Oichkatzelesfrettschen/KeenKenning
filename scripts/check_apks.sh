#!/bin/bash
./gradlew assembleDebug
echo "Listing generated APKs:"
ls app/build/outputs/apk/debug/*.apk
