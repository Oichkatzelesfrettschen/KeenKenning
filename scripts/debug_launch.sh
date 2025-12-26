#!/bin/bash
export PATH=$PATH:/opt/android-sdk/platform-tools
echo "Building APK..."
./gradlew assembleDebug
if [ $? -eq 0 ]; then
    echo "Installing APK..."
    adb install -r app/build/outputs/apk/debug/app-debug.apk
    echo "Launching KeenActivity with AI enabled..."
    adb shell am start -n org.yegie.keenkeenforandroid/.KeenActivity --ei gameSize 5 --ei gameDiff 1 --ei gameMultOnly 0 --el gameSeed 12345 --ez useAI true
else
    echo "Build failed."
    exit 1
fi
