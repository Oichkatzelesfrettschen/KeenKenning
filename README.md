# KeenKeen for Android

A modern, feature-rich implementation of the KenKen logic puzzle for Android. Built with Jetpack Compose and powered by native C algorithms and optional neural AI assistance.

## Download

**[Latest Debug Release (v1.3.1)](https://github.com/Oichkatzelesfrettschen/KeenKeenForAndroid/releases/tag/v1.3.1-debug)**

| APK | Architecture | Size | Notes |
|-----|--------------|------|-------|
| `app-universal-debug.apk` | All | ~46MB | Works everywhere |
| `app-arm64-v8a-debug.apk` | ARM64 | ~24MB | Modern phones |
| `app-armeabi-v7a-debug.apk` | ARM32 | ~23MB | Older phones |
| `app-x86_64-debug.apk` | x86_64 | ~25MB | Emulators, Chromebooks |
| `app-x86-debug.apk` | x86 | ~25MB | 32-bit x86 |
| `app-riscv64-debug.apk` | RISC-V | ~17MB | Emerging architecture |

## Features

### 11 Game Modes

| Mode | Description | Status |
|------|-------------|--------|
| **Standard** | All operations (+, -, ×, ÷) | ✅ Complete |
| **Multiply** | Multiplication only | ✅ Complete |
| **Mystery** | Operations hidden - deduce them! | ✅ Complete |
| **Zero Mode** | Numbers 0 to N-1 (no division) | ✅ Complete |
| **Powers** | Includes exponent (^) operation | ✅ Complete |
| **Negative** | Range -N to +N (excluding 0) | ✅ Complete |
| **Modular** | Wrap-around arithmetic (mod N) | ✅ Complete |
| **Killer** | No repeated digits in cages | ✅ Complete |
| **Tutorial** | Explainable hints with reasoning | ✅ Complete |
| **Adaptive** | Difficulty adjusts to your skill | ✅ Complete |
| **Story** | Themed puzzles with narrative | ✅ Complete |

### Grid Sizes
- **Standard**: 3×3, 4×4, 5×5, 6×6, 7×7, 8×8, 9×9
- **Extended**: 10×10, 12×12, 16×16 (uses hex digits A-G)

### Difficulty Levels
Easy • Normal • Hard • Insane • Ludicrous

### Additional Features
- **Neural AI Hints**: ONNX-based solver provides intelligent suggestions
- **Quantum Visualization**: Probability hints shown as faint overlays
- **Victory Animation**: Bouncing tiles celebration on puzzle completion
- **Save/Load**: Multiple save slots with auto-save
- **Accessibility**: Full TalkBack/screen reader support
- **Classic Mode**: Legacy canvas renderer option
- **Dark Theme**: Modern dark UI throughout

## Requirements

- Android 5.1+ (API 22)
- ~50MB storage for universal APK

## Building from Source

### Prerequisites
- JDK 21 (required by Gradle 8.6)
- Android SDK with API 35
- Android NDK 27.x

### Build Commands
```bash
# Debug build
JAVA_HOME=/usr/lib/jvm/java-21-openjdk ./gradlew assembleDebug

# Release build
JAVA_HOME=/usr/lib/jvm/java-21-openjdk ./gradlew assembleRelease

# Clean build
JAVA_HOME=/usr/lib/jvm/java-21-openjdk ./gradlew clean assembleDebug
```

Output: `app/build/outputs/apk/debug/`

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Compose UI Layer                         │
│  GameScreen.kt • MenuScreen.kt • VictoryAnimation.kt        │
├─────────────────────────────────────────────────────────────┤
│                   ViewModel Layer                           │
│  GameViewModel.kt • GameUiState.kt • SaveManager.kt         │
├─────────────────────────────────────────────────────────────┤
│                  Integration Layer                          │
│  KeenModelBuilder.java • NeuralKeenGenerator.java           │
├─────────────────────────────────────────────────────────────┤
│                    Native Layer (C)                         │
│  keen.c • keen-android-jni.c (via CMake/NDK)                │
├─────────────────────────────────────────────────────────────┤
│                     AI Layer                                │
│  keen_solver_9x9.onnx (ONNX Runtime)                        │
└─────────────────────────────────────────────────────────────┘
```

## Known Issues (Debug Build)

1. **First launch**: Initial puzzle generation may take 1-2 seconds on older devices
2. **Large grids**: 16×16 puzzles require significant computation time
3. **Neural hints**: AI suggestions work best for grids 4×4 to 9×9

## Testing Checklist for Debug Testers

- [ ] App installs and launches without crash
- [ ] Menu displays all 11 game modes
- [ ] Can select different grid sizes (3×3 to 16×16)
- [ ] Puzzle generates correctly for each mode
- [ ] Cell input works (tap cell, tap number)
- [ ] Notes mode toggles correctly
- [ ] Undo/Clear functions work
- [ ] Victory animation triggers on correct solution
- [ ] Save/Load functionality works
- [ ] Settings dialog opens
- [ ] Back button returns to menu
- [ ] Screen rotation doesn't crash app
- [ ] TalkBack announces cells and actions

## Credits

- **Core Algorithms**: [Simon Tatham's Portable Puzzle Collection](https://www.chiark.greenend.org.uk/~sgtatham/puzzles/)
- **Original Android Port**: Sergey Maltsev
- **KeenKeen Enhancements**: KeenKeen Contributors

## License

GPL-3.0-or-later

---

*Debug build - Report issues at [GitHub Issues](https://github.com/Oichkatzelesfrettschen/KeenKeenForAndroid/issues)*
