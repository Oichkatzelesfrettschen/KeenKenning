# Keen Classik v1.0 - ARM64 Release

A traditional KenKen-style logic puzzle game for Android. Keen Classik offers the pure puzzle-solving experience without machine learning features.

## What's New in v1.0

This is the first stable release of Keen Classik, featuring:

- **Classic puzzle gameplay**: Solve KenKen-style arithmetic puzzles from 3×3 to 9×9 grids
- **Five difficulty levels**: Easy, Normal, Hard, Extreme, and Ludicrous
- **Notes support**: Add pencil marks to cells to track your deductions
- **Save and restore**: Your progress is automatically saved
- **Multi-architecture builds**: Optimized for modern ARM64 devices, with fallbacks for older hardware

## Getting Started

1. Download the APK that matches your device
2. Enable "Install from Unknown Sources" in your Android settings
3. Install the APK
4. Launch Keen Classik and start solving!

## Available Downloads

| APK File | Best For | Size |
|----------|----------|------|
| app-classik-arm64-v8a-debug.apk | Modern Android phones (2017+) | 30 MB |
| app-classik-universal-debug.apk | Any Android device | 33 MB |
| app-classik-x86_64-debug.apk | Android emulators or Intel-based devices | 30 MB |
| app-classik-armeabi-v7a-debug.apk | Older 32-bit ARM devices | 29 MB |

**Recommended**: Use the ARM64 build for best performance on modern devices.

## System Requirements

- Android 6.0 (API 23) or later
- ~30-35 MB storage space

## Installation

Using ADB (for developers):
```bash
adb install app-classik-arm64-v8a-debug.apk
```

Or download and open the APK file on your Android device.

## Known Issues

- This is a debug build, so some performance optimizations are not enabled
- First puzzle generation may take 1-2 seconds on older devices

## What's Next

Looking for more challenge? Try **Keen Kenning** (coming soon), which includes:
- Larger grids (up to 16×16)
- Machine learning-powered hints
- Additional game modes

---

**Questions or Issues?** Visit the [GitHub repository](https://github.com/Oichkatzelesfrettschen/KeenKenning) or open an issue.

Build: 456e291b  
Release Date: 2026-01-05
